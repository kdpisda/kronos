#pragma once
// ============================================================================
// KRONOS  src/hamiltonian/gpu_hamiltonian.hpp
// GPU-accelerated Hamiltonian operator: H|ψ⟩ on device
//
// All operations (FFT, pointwise multiply, nonlocal GEMM) happen on the GPU.
// A single upload/download per apply() call.
//
// Fallback: if GPU OOM, falls back to CPU Hamiltonian.
// ============================================================================

#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "potential/nonlocal_pp.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/fft.hpp"

#include <vector>
#include <functional>
#include <memory>

namespace kronos {

class GPUHamiltonian {
public:
    GPUHamiltonian(const Crystal& crystal,
                   const PlaneWaveBasis& basis,
                   FFTGrid& fft_grid,
                   NonlocalPP& nonlocal_pp,
                   Hamiltonian& cpu_fallback);

    ~GPUHamiltonian();

    /// Update the effective potential on the GPU.
    /// Transfers veff_r to device memory.
    void update_veff(const std::vector<complex_t>& veff_r);

    /// Apply H|ψ⟩ on the GPU.
    /// psi_g: wavefunction in G-space (host memory)
    /// k_frac: k-point in fractional coordinates
    /// Returns H|ψ⟩ in G-space (host memory)
    ///
    /// On GPU OOM, falls back to CPU Hamiltonian.
    CVec apply(const CVec& psi_g, const Vec3& k_frac);

    /// Get a std::function wrapper for use with the Davidson solver.
    std::function<CVec(const CVec&)> get_apply_function(const Vec3& k_frac);

    /// Get kinetic diagonal (delegates to CPU — small data).
    std::vector<double> kinetic_diagonal(const Vec3& k_frac) const;

    /// Check if GPU is active (not fallen back to CPU).
    bool gpu_active() const { return gpu_active_; }

private:
    const Crystal& crystal_;
    const PlaneWaveBasis& basis_;
    FFTGrid& fft_grid_;
    NonlocalPP& nonlocal_pp_;
    Hamiltonian& cpu_fallback_;

    bool gpu_active_{false};

    // Device buffers
    std::unique_ptr<gpu::DeviceBuffer<complex_t>> d_veff_;     // V_eff on real-space grid
    std::unique_ptr<gpu::DeviceBuffer<double>> d_kinetic_;      // Kinetic energies per k
    std::unique_ptr<gpu::DeviceBuffer<int>> d_scatter_map_;     // PW → FFT grid index map
    std::unique_ptr<gpu::GPUFFTGrid> gpu_fft_;                  // GPU FFT plans

    // Precomputed scatter/gather index map
    std::vector<int> scatter_map_;
    void build_scatter_map();

    // GPU kernel dispatchers
    CVec apply_gpu(const CVec& psi_g, const Vec3& k_frac);
};

} // namespace kronos
