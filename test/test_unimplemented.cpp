// ============================================================================
// KRONOS  test/test_unimplemented.cpp
// GTEST_SKIP() stubs for features not yet implemented in v0.1.
// ============================================================================

#include <gtest/gtest.h>

// ============================================================================
// GPU offloading (v0.5)
// ============================================================================

TEST(GPU, CUDABackendOffloading) {
    GTEST_SKIP() << "GPU offloading not implemented in v0.1";
}

TEST(GPU, HIPBackendOffloading) {
    GTEST_SKIP() << "HIP/AMD GPU backend not implemented in v0.1";
}

TEST(GPU, GPUFallbackOnOOM) {
    GTEST_SKIP() << "GPU OOM auto-fallback not implemented in v0.1";
}

// ============================================================================
// MPI parallelization (v0.2)
// ============================================================================

TEST(MPI, KPointParallelization) {
    GTEST_SKIP() << "MPI k-point parallelization not implemented in v0.1";
}

TEST(MPI, BandParallelization) {
    GTEST_SKIP() << "MPI band parallelization not implemented in v0.1";
}

// ============================================================================
// PAW support (v0.8)
// ============================================================================

TEST(PAW, OneCenterEnergy) {
    GTEST_SKIP() << "PAW one-center energy not implemented in v0.1";
}

TEST(PAW, AugmentationCharges) {
    GTEST_SKIP() << "PAW augmentation charges not implemented in v0.1";
}

// ============================================================================
// Python bindings (v1.0)
// ============================================================================

TEST(Python, Pybind11Interface) {
    GTEST_SKIP() << "Python bindings not implemented in v0.1";
}

// ============================================================================
// Spin polarization (v0.8+)
// ============================================================================

TEST(Spin, CollinearLSDA) {
    GTEST_SKIP() << "Spin-polarized LSDA not implemented in v0.1";
}

TEST(Spin, NonCollinear) {
    GTEST_SKIP() << "Non-collinear magnetism not implemented (v2.0)";
}

// ============================================================================
// Checkpoint/restart (v0.5+)
// ============================================================================

TEST(Checkpoint, WriteAndRestart) {
    GTEST_SKIP() << "Checkpoint/restart not implemented in v0.1";
}

// ============================================================================
// Stress tensor (v0.5+)
// ============================================================================

TEST(Stress, StressTensorComputation) {
    GTEST_SKIP() << "Stress tensor not implemented in v0.1";
}

// ============================================================================
// LOBPCG eigensolver (v0.2+)
// ============================================================================

TEST(LOBPCG, BasicConvergence) {
    GTEST_SKIP() << "LOBPCG eigensolver not implemented in v0.1";
}

TEST(LOBPCG, DavidsonFallback) {
    GTEST_SKIP() << "Davidson→LOBPCG auto-fallback not implemented in v0.1";
}

// ============================================================================
// Hybrid functionals (v2.0)
// ============================================================================

TEST(Hybrid, HSE06) {
    GTEST_SKIP() << "HSE06 hybrid functional not implemented (v2.0)";
}

TEST(Hybrid, PBE0) {
    GTEST_SKIP() << "PBE0 hybrid functional not implemented (v2.0)";
}
