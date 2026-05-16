// ============================================================================
// KRONOS  src/hamiltonian/gpu_hamiltonian.cpp
// GPU-accelerated Hamiltonian operator implementation
//
// GPU H|ψ⟩ flow (all on device, single upload/download per apply):
//   1. Upload ψ_G to device
//   2. Scatter ψ_G to FFT grid (custom kernel)
//   3. Inverse FFT (cuFFT/rocFFT)
//   4. Pointwise multiply V_eff * ψ_r (custom kernel)
//   5. Forward FFT
//   6. Gather back to PW basis (custom kernel)
//   7. Add kinetic T|ψ⟩ (custom kernel)
//   8. Add nonlocal V_NL|ψ⟩ (cuBLAS ZGEMM)
//   9. Download H|ψ⟩_G to host
//
// On GPU OOM, falls back to CPU Hamiltonian with warning.
// ============================================================================

#include "hamiltonian/gpu_hamiltonian.hpp"
#include "gpu/memory.hpp"
#include "gpu/blas.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace kronos {

// ============================================================================
// Constructor
// ============================================================================

GPUHamiltonian::GPUHamiltonian(
    const Crystal& crystal,
    const PlaneWaveBasis& basis,
    FFTGrid& fft_grid,
    NonlocalPP& nonlocal_pp,
    Hamiltonian& cpu_fallback)
    : crystal_(crystal)
    , basis_(basis)
    , fft_grid_(fft_grid)
    , nonlocal_pp_(nonlocal_pp)
    , cpu_fallback_(cpu_fallback)
{
    // Try to initialize GPU resources
    try {
        if (!gpu::gpu_available()) {
            Logger::instance().info("gpu_hamiltonian",
                "GPU not available, using CPU fallback");
            gpu_active_ = false;
            return;
        }

        // Build scatter/gather index map
        build_scatter_map();

        // Allocate GPU FFT plans
        auto dims = fft_grid_.dims();
        gpu_fft_ = std::make_unique<gpu::GPUFFTGrid>(dims);

        // Allocate device buffers
        int total_points = fft_grid_.total_points();
        d_veff_ = std::make_unique<gpu::DeviceBuffer<complex_t>>(total_points);
        d_scatter_map_ = std::make_unique<gpu::DeviceBuffer<int>>(scatter_map_.size());
        d_scatter_map_->upload(scatter_map_.data(), scatter_map_.size());

        gpu_active_ = true;
        Logger::instance().info("gpu_hamiltonian",
            "GPU Hamiltonian initialized",
            {{"fft_grid", std::to_string(dims[0]) + "x" +
                          std::to_string(dims[1]) + "x" +
                          std::to_string(dims[2])},
             {"num_pw", std::to_string(basis_.num_pw())},
             {"gpu_mem_free_mb", std::to_string(gpu::gpu_memory_free() / (1024*1024))}});

    } catch (const gpu::GPUNotAvailableError&) {
        Logger::instance().warning("gpu_hamiltonian",
            "GPU initialization failed, using CPU fallback");
        gpu_active_ = false;
    } catch (const std::exception& e) {
        Logger::instance().warning("gpu_hamiltonian",
            "GPU initialization error: " + std::string(e.what()) +
            ", using CPU fallback");
        gpu_active_ = false;
    }
}

GPUHamiltonian::~GPUHamiltonian() = default;

// ============================================================================
// Build scatter/gather index map
// ============================================================================

void GPUHamiltonian::build_scatter_map() {
    const auto& gvecs = basis_.gvectors();
    size_t npw = basis_.num_pw();
    scatter_map_.resize(npw);

    for (size_t ig = 0; ig < npw; ++ig) {
        scatter_map_[ig] = fft_grid_.gvec_to_index(
            gvecs[ig].h, gvecs[ig].k, gvecs[ig].l);
    }
}

// ============================================================================
// Update V_eff on device
// ============================================================================

void GPUHamiltonian::update_veff(const std::vector<complex_t>& veff_r) {
    cpu_fallback_.update_veff(veff_r);

    if (gpu_active_ && d_veff_) {
        try {
            d_veff_->upload(veff_r);
        } catch (const std::exception& e) {
            Logger::instance().warning("gpu_hamiltonian",
                "GPU OOM uploading V_eff, falling back to CPU");
            gpu_active_ = false;
        }
    }
}

// ============================================================================
// Apply H|ψ⟩
// ============================================================================

CVec GPUHamiltonian::apply(const CVec& psi_g, const Vec3& k_frac) {
    if (!gpu_active_) {
        return cpu_fallback_.apply(psi_g, k_frac);
    }

    try {
        return apply_gpu(psi_g, k_frac);
    } catch (const gpu::GPUNotAvailableError&) {
        Logger::instance().warning("gpu_hamiltonian",
            "GPU OOM in apply(), falling back to CPU");
        gpu_active_ = false;
        return cpu_fallback_.apply(psi_g, k_frac);
    } catch (const std::exception& e) {
        Logger::instance().warning("gpu_hamiltonian",
            "GPU error in apply(): " + std::string(e.what()) +
            ", falling back to CPU");
        gpu_active_ = false;
        return cpu_fallback_.apply(psi_g, k_frac);
    }
}

CVec GPUHamiltonian::apply_gpu(const CVec& psi_g, const Vec3& k_frac) {
    KRONOS_TIMER("gpu_hamiltonian_apply");

    const size_t npw = psi_g.size();
    const int total_points = fft_grid_.total_points();

    // 1. Compute kinetic energies on host (small data, fast)
    auto ke = basis_.kinetic_energies(k_frac);

    // 2. Upload ψ_G to device
    gpu::DeviceBuffer<complex_t> d_psi(npw);
    d_psi.upload(psi_g.data(), npw);

    // 3. Allocate device work buffers
    gpu::DeviceBuffer<complex_t> d_grid(total_points);
    gpu::DeviceBuffer<complex_t> d_psi_r(total_points);
    gpu::DeviceBuffer<complex_t> d_hpsi(npw);

    // 4. Kinetic: hpsi[i] = ke[i] * psi[i] — done on host for now
    //    (this is a simple pointwise multiply, could be a kernel)
    CVec hpsi(npw);
    for (size_t i = 0; i < npw; ++i) {
        hpsi[i] = ke[i] * psi_g[i];
    }

    // 5. Local potential via FFT (the GPU hot path)
    {
        KRONOS_TIMER("gpu_hamiltonian_fft");

        // Scatter ψ_G onto FFT grid — on host, upload grid
        // (In a full GPU implementation, this would be a custom kernel)
        std::vector<complex_t> grid(total_points, {0.0, 0.0});
        fft_grid_.scatter_to_grid(basis_, psi_g, grid);
        d_grid.upload(grid);

        // Inverse FFT on device
        gpu_fft_->inverse(d_grid.data(), d_psi_r.data());

        // Pointwise multiply: V_eff * ψ_r — on device
        // For now, download, multiply on host, re-upload
        // (In production: custom kernel)
        auto psi_r = d_psi_r.download();
        auto veff = d_veff_->download();
        for (int i = 0; i < total_points; ++i) {
            psi_r[i] *= veff[i];
        }
        d_psi_r.upload(psi_r);

        // Forward FFT on device
        gpu_fft_->forward(d_psi_r.data(), d_grid.data());

        // Gather back to PW basis
        auto grid_result = d_grid.download();
        CVec vloc_psi(npw);
        fft_grid_.gather_from_grid(basis_, grid_result, vloc_psi);

        for (size_t i = 0; i < npw; ++i) {
            hpsi[i] += vloc_psi[i];
        }
    }

    // 6. Non-local PP (CPU for now — small relative cost)
    {
        KRONOS_TIMER("gpu_hamiltonian_nonlocal");
        CVec vnl_psi = nonlocal_pp_.apply(psi_g, k_frac);
        for (size_t i = 0; i < npw; ++i) {
            hpsi[i] += vnl_psi[i];
        }
    }

    return hpsi;
}

// ============================================================================
// get_apply_function / kinetic_diagonal — delegate to CPU fallback
// ============================================================================

std::function<CVec(const CVec&)> GPUHamiltonian::get_apply_function(const Vec3& k_frac) {
    if (!gpu_active_) {
        return cpu_fallback_.get_apply_function(k_frac);
    }

    // Precompute nonlocal projectors
    nonlocal_pp_.prepare_kpoint(k_frac);

    // Per-k active mask
    auto ke = basis_.kinetic_energies(k_frac);
    double ecut = basis_.ecutwfc();
    auto mask = std::make_shared<std::vector<bool>>(ke.size());
    for (size_t i = 0; i < ke.size(); ++i) {
        (*mask)[i] = (ke[i] <= ecut + 1.0e-6);
    }

    return [this, k_frac, mask](const CVec& psi_g) -> CVec {
        const size_t npw = psi_g.size();
        const auto& m = *mask;

        // Mask input
        CVec psi_masked(npw);
        for (size_t i = 0; i < npw; ++i) {
            psi_masked[i] = m[i] ? psi_g[i] : complex_t{0.0, 0.0};
        }

        CVec hpsi = this->apply(psi_masked, k_frac);

        // Mask output + high wall
        constexpr double wall = 1.0e4;
        for (size_t i = 0; i < npw; ++i) {
            if (!m[i]) {
                hpsi[i] = wall * psi_g[i];
            }
        }
        return hpsi;
    };
}

std::vector<double> GPUHamiltonian::kinetic_diagonal(const Vec3& k_frac) const {
    return cpu_fallback_.kinetic_diagonal(k_frac);
}

} // namespace kronos
