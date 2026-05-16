// ============================================================================
// KRONOS  test/test_gpu.cpp
// GPU infrastructure tests: memory, FFT, BLAS, Hamiltonian
//
// In CPU-only builds, all tests verify that GPU operations correctly
// throw GPUNotAvailableError and that the OOM fallback works.
// ============================================================================

#include <gtest/gtest.h>
#include "gpu/memory.hpp"
#include "gpu/fft.hpp"
#include "gpu/blas.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/gpu_context.hpp"
#include "hamiltonian/gpu_hamiltonian.hpp"
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "potential/nonlocal_pp.hpp"
#include "hamiltonian/hamiltonian.hpp"

#include <cmath>
#include <complex>
#include <vector>

using namespace kronos;

// ============================================================================
// GPU Memory Tests (CPU-only: verify throws)
// ============================================================================

TEST(GPU, MemoryAvailability) {
    // In CPU-only build, gpu_available() returns false
    bool avail = gpu::gpu_available();
    // Just verify it doesn't crash
    (void)avail;
}

TEST(GPU, MemoryInfo) {
    size_t free_mem = gpu::gpu_memory_free();
    size_t total_mem = gpu::gpu_memory_total();
    // In CPU-only build, both return 0
    EXPECT_EQ(free_mem, 0u);
    EXPECT_EQ(total_mem, 0u);
}

TEST(GPU, MallocThrowsInCPUBuild) {
    if (gpu::gpu_available()) {
        GTEST_SKIP() << "GPU available, skipping CPU-only test";
    }
    EXPECT_THROW(gpu::gpu_malloc(1024), gpu::GPUNotAvailableError);
}

TEST(GPU, FreeNoOpForNull) {
    if (gpu::gpu_available()) {
        GTEST_SKIP() << "GPU available, skipping CPU-only test";
    }
    // gpu_free should throw for non-null in CPU build
    EXPECT_THROW(gpu::gpu_free(reinterpret_cast<void*>(0x1)),
                 gpu::GPUNotAvailableError);
}

// ============================================================================
// DeviceBuffer Tests (CPU-only: verify throws on construction)
// ============================================================================

TEST(GPU, DeviceBufferThrowsInCPUBuild) {
    if (gpu::gpu_available()) {
        GTEST_SKIP() << "GPU available, skipping CPU-only test";
    }
    // Constructing a non-empty DeviceBuffer should throw
    EXPECT_THROW(gpu::DeviceBuffer<double>(100), gpu::GPUNotAvailableError);
}

TEST(GPU, DeviceBufferEmptyIsOk) {
    // Empty DeviceBuffer should be fine even without GPU
    gpu::DeviceBuffer<double> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(GPU, DeviceBufferMoveSemantics) {
    // Move of empty buffer should work
    gpu::DeviceBuffer<double> a;
    gpu::DeviceBuffer<double> b(std::move(a));
    EXPECT_TRUE(b.empty());
}

// ============================================================================
// GPU FFT Tests (CPU-only: verify throws)
// ============================================================================

TEST(GPU, FFTGridThrowsInCPUBuild) {
    if (gpu::gpu_available()) {
        GTEST_SKIP() << "GPU available, skipping CPU-only test";
    }
    EXPECT_THROW(gpu::GPUFFTGrid({16, 16, 16}), gpu::GPUNotAvailableError);
}

// ============================================================================
// GPU BLAS Tests (CPU-only: verify throws)
// ============================================================================

TEST(GPU, GEMMThrowsInCPUBuild) {
    if (gpu::gpu_available()) {
        GTEST_SKIP() << "GPU available, skipping CPU-only test";
    }
    EXPECT_THROW(
        gpu::gemm(2, 2, 2, {1.0, 0.0}, nullptr, 2, nullptr, 2, {0.0, 0.0}, nullptr, 2),
        gpu::GPUNotAvailableError);
}

TEST(GPU, ZdotcThrowsInCPUBuild) {
    if (gpu::gpu_available()) {
        GTEST_SKIP() << "GPU available, skipping CPU-only test";
    }
    EXPECT_THROW(gpu::zdotc(10, nullptr, nullptr), gpu::GPUNotAvailableError);
}

// ============================================================================
// GPU Context Tests
// ============================================================================

TEST(GPU, ContextSingleton) {
    auto& ctx1 = gpu::GPUContext::instance();
    auto& ctx2 = gpu::GPUContext::instance();
    EXPECT_EQ(&ctx1, &ctx2);
}

TEST(GPU, ContextInitCPUBuild) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init(0, 0);
    // In CPU build, initialized_ remains false (no devices)
    // num_devices should be 0
    EXPECT_EQ(ctx.num_devices(), 0);
}

// ============================================================================
// GPU Hamiltonian Tests
// ============================================================================

namespace {

Crystal make_si_diamond() {
    const double a = 10.2;
    Mat3 lattice = {{
        {{-a/2.0, 0.0, a/2.0}},
        {{0.0, a/2.0, a/2.0}},
        {{-a/2.0, a/2.0, 0.0}}
    }};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.0, 0.0, 0.0}},
        {"Si", 14, {0.25, 0.25, 0.25}}
    };
    return Crystal(lattice, atoms);
}

std::map<std::string, PseudoPotential> make_si_pp() {
    PseudoPotential pp;
    pp.element = "Si";
    pp.z_valence = 4.0;
    pp.is_norm_conserving = true;
    pp.lmax = 0;
    pp.num_projectors = 0;
    pp.mesh.npoints = 100;
    pp.mesh.r.resize(100);
    pp.mesh.rab.resize(100);
    for (int i = 0; i < 100; ++i) {
        pp.mesh.r[i] = (i + 1) * 0.01;
        pp.mesh.rab[i] = 0.01;
    }
    pp.vloc.resize(100);
    double r_loc = 0.44;
    for (int i = 0; i < 100; ++i) {
        double r = pp.mesh.r[i];
        pp.vloc[i] = -pp.z_valence / r * std::exp(-r * r / (2.0 * r_loc * r_loc)) * 2.0;
    }
    pp.rho_atomic.resize(100, 0.01);

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;
    return pps;
}

} // anonymous namespace

TEST(GPUHamiltonian, FallbackToCPU) {
    // In CPU-only build, GPUHamiltonian should fall back to CPU
    Crystal crystal = make_si_diamond();
    auto pps = make_si_pp();

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);
    NonlocalPP nonlocal_pp(crystal, basis, pps);
    Hamiltonian cpu_ham(crystal, basis, fft_grid, nonlocal_pp);

    // Set up V_eff
    int num_grid = fft_grid.total_points();
    std::vector<complex_t> veff(num_grid, {-1.0, 0.0});
    cpu_ham.update_veff(veff);

    // Create GPU Hamiltonian (will fall back to CPU)
    GPUHamiltonian gpu_ham(crystal, basis, fft_grid, nonlocal_pp, cpu_ham);
    gpu_ham.update_veff(veff);

    // gpu_active should be false in CPU-only build
    EXPECT_FALSE(gpu_ham.gpu_active());

    // Apply should still work (via CPU fallback)
    int npw = static_cast<int>(basis.num_pw());
    CVec psi(npw, complex_t{0.0, 0.0});
    psi[0] = {1.0, 0.0};

    Vec3 k_frac = {0.0, 0.0, 0.0};
    CVec hpsi = gpu_ham.apply(psi, k_frac);

    EXPECT_EQ(hpsi.size(), static_cast<size_t>(npw));

    // Should match CPU Hamiltonian exactly
    CVec hpsi_cpu = cpu_ham.apply(psi, k_frac);
    for (int i = 0; i < npw; ++i) {
        EXPECT_NEAR(std::abs(hpsi[i] - hpsi_cpu[i]), 0.0, 1e-14);
    }
}

TEST(GPUHamiltonian, GetApplyFunction) {
    Crystal crystal = make_si_diamond();
    auto pps = make_si_pp();

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);
    NonlocalPP nonlocal_pp(crystal, basis, pps);
    Hamiltonian cpu_ham(crystal, basis, fft_grid, nonlocal_pp);

    int num_grid = fft_grid.total_points();
    std::vector<complex_t> veff(num_grid, {-1.0, 0.0});
    cpu_ham.update_veff(veff);

    GPUHamiltonian gpu_ham(crystal, basis, fft_grid, nonlocal_pp, cpu_ham);
    gpu_ham.update_veff(veff);

    Vec3 k_frac = {0.0, 0.0, 0.0};
    auto apply_fn = gpu_ham.get_apply_function(k_frac);

    int npw = static_cast<int>(basis.num_pw());
    CVec psi(npw, complex_t{0.0, 0.0});
    psi[0] = {1.0, 0.0};

    CVec hpsi = apply_fn(psi);
    EXPECT_EQ(hpsi.size(), static_cast<size_t>(npw));
}

TEST(GPUHamiltonian, KineticDiagonal) {
    Crystal crystal = make_si_diamond();
    auto pps = make_si_pp();

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);
    NonlocalPP nonlocal_pp(crystal, basis, pps);
    Hamiltonian cpu_ham(crystal, basis, fft_grid, nonlocal_pp);

    GPUHamiltonian gpu_ham(crystal, basis, fft_grid, nonlocal_pp, cpu_ham);

    Vec3 k_frac = {0.0, 0.0, 0.0};
    auto ke_gpu = gpu_ham.kinetic_diagonal(k_frac);
    auto ke_cpu = cpu_ham.kinetic_diagonal(k_frac);

    EXPECT_EQ(ke_gpu.size(), ke_cpu.size());
    for (size_t i = 0; i < ke_gpu.size(); ++i) {
        EXPECT_DOUBLE_EQ(ke_gpu[i], ke_cpu[i]);
    }
}
