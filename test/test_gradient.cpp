// ============================================================================
// KRONOS  test/test_gradient.cpp
// Tests for GGA gradient operations: sigma computation, GGA potential.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "potential/gradient.hpp"

#include <cmath>

using namespace kronos;

// ============================================================================
// Sigma computation for uniform density
// ============================================================================

TEST(Gradient, UniformDensitySigmaIsZero) {
    // Uniform density has zero gradient
    Crystal crystal = test::make_cubic_crystal(5.0);
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int num_grid = fft.total_points();

    // Uniform density: only G=0 component is nonzero
    CVec density_g(npw, complex_t{0.0, 0.0});
    double n0 = 0.1;
    density_g[0] = complex_t{n0 * num_grid, 0.0};

    RVec sigma = compute_sigma(density_g, basis, fft);

    ASSERT_EQ(static_cast<int>(sigma.size()), num_grid);
    for (int i = 0; i < num_grid; ++i) {
        EXPECT_NEAR(sigma[i], 0.0, 1e-10)
            << "Gradient of uniform density should be zero at point " << i;
    }
}

// ============================================================================
// Sigma for plane-wave density
// ============================================================================

TEST(Gradient, PlaneWaveDensitySigma) {
    // Density n(r) = n0 + A*cos(G1.r)
    // ∇n = -A*sin(G1.r)*G1
    // |∇n|² = A²*sin²(G1.r)*|G1|²
    Crystal crystal = test::make_cubic_crystal(5.0);
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int num_grid = fft.total_points();
    auto dims = fft.dims();

    double n0 = 0.1;
    double A = 0.01;

    // Find first nonzero G-vector
    const auto& gvecs = basis.gvectors();
    size_t ig1 = 0;
    for (size_t ig = 0; ig < gvecs.size(); ++ig) {
        if (gvecs[ig].norm2 > 1e-10) { ig1 = ig; break; }
    }
    const Vec3& G1 = gvecs[ig1].cart;
    double G1_sq = gvecs[ig1].norm2;

    // Build density on real-space grid, then FFT
    Mat3 lat_bohr = crystal.lattice_bohr();
    RVec density_r(num_grid);
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                double fx = static_cast<double>(i) / dims[0];
                double fy = static_cast<double>(j) / dims[1];
                double fz = static_cast<double>(k) / dims[2];
                double rx = fx * lat_bohr[0][0] + fy * lat_bohr[1][0] + fz * lat_bohr[2][0];
                double ry = fx * lat_bohr[0][1] + fy * lat_bohr[1][1] + fz * lat_bohr[2][1];
                double rz = fx * lat_bohr[0][2] + fy * lat_bohr[1][2] + fz * lat_bohr[2][2];
                int idx = (i * dims[1] + j) * dims[2] + k;
                density_r[idx] = n0 + A * std::cos(G1[0]*rx + G1[1]*ry + G1[2]*rz);
            }
        }
    }

    std::vector<complex_t> density_c(num_grid);
    for (int i = 0; i < num_grid; ++i) {
        density_c[i] = complex_t{density_r[i], 0.0};
    }
    std::vector<complex_t> density_g_full(num_grid);
    fft.forward(density_c, density_g_full);
    CVec density_g(npw);
    fft.gather_from_grid(basis, density_g_full, density_g);

    RVec sigma = compute_sigma(density_g, basis, fft);
    ASSERT_EQ(static_cast<int>(sigma.size()), num_grid);

    // Average of |∇n|² = A²|G1|²/2 (average of sin²)
    double sigma_avg = 0.0;
    for (int i = 0; i < num_grid; ++i) {
        sigma_avg += sigma[i];
    }
    sigma_avg /= num_grid;

    double expected_avg = A * A * G1_sq / 2.0;
    EXPECT_NEAR(sigma_avg, expected_avg, expected_avg * 0.1)
        << "Average sigma should match A²|G|²/2";
}

// ============================================================================
// GGA potential for uniform density
// ============================================================================

TEST(Gradient, GGAPotentialUniformDensityIsZero) {
    Crystal crystal = test::make_cubic_crystal(5.0);
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int num_grid = fft.total_points();

    CVec density_g(npw, complex_t{0.0, 0.0});
    density_g[0] = complex_t{0.1 * num_grid, 0.0};

    // vsigma = constant (since density is uniform)
    RVec vsigma(num_grid, 0.5);

    RVec vgga = compute_gga_potential(density_g, vsigma, basis, fft);
    ASSERT_EQ(static_cast<int>(vgga.size()), num_grid);

    // For uniform density, gradient is zero, so GGA potential should be ~zero
    double max_vgga = 0.0;
    for (int i = 0; i < num_grid; ++i) {
        max_vgga = std::max(max_vgga, std::abs(vgga[i]));
    }
    EXPECT_LT(max_vgga, 1e-8)
        << "GGA potential of uniform density should be zero";
}

// ============================================================================
// GGA potential output size
// ============================================================================

TEST(Gradient, GGAPotentialOutputSize) {
    Crystal crystal = test::make_cubic_crystal(5.0);
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int num_grid = fft.total_points();

    CVec density_g(npw, complex_t{0.01, 0.0});
    RVec vsigma(num_grid, 0.1);

    RVec vgga = compute_gga_potential(density_g, vsigma, basis, fft);
    EXPECT_EQ(static_cast<int>(vgga.size()), num_grid);
}

// ============================================================================
// Sigma is non-negative
// ============================================================================

TEST(Gradient, SigmaIsNonNegative) {
    Crystal crystal = test::make_cubic_crystal(5.0);
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int num_grid = fft.total_points();

    // Random density in G-space (but ensure it's real in r-space)
    CVec density_g(npw, complex_t{0.0, 0.0});
    density_g[0] = complex_t{0.05 * num_grid, 0.0};
    for (int i = 1; i < std::min(npw, 10); ++i) {
        density_g[i] = complex_t{0.001 * (10 - i), 0.0};
    }

    RVec sigma = compute_sigma(density_g, basis, fft);
    for (int i = 0; i < num_grid; ++i) {
        EXPECT_GE(sigma[i], -1e-10)
            << "|∇n|² should be non-negative at point " << i;
    }
}

// ============================================================================
// Sigma size matches grid
// ============================================================================

TEST(Gradient, SigmaSizeMatchesGrid) {
    Crystal crystal = test::make_cubic_crystal(5.0);
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int num_grid = fft.total_points();

    CVec density_g(npw, complex_t{0.01, 0.0});
    RVec sigma = compute_sigma(density_g, basis, fft);
    EXPECT_EQ(static_cast<int>(sigma.size()), num_grid);
}

// ============================================================================
// GGA potential is real-valued
// ============================================================================

TEST(Gradient, GGAPotentialIsRealValued) {
    Crystal crystal = test::make_cubic_crystal(5.0);
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int num_grid = fft.total_points();

    CVec density_g(npw, complex_t{0.0, 0.0});
    density_g[0] = complex_t{0.05 * num_grid, 0.0};
    density_g[1] = complex_t{0.001, 0.0};

    RVec vsigma(num_grid, 0.1);
    RVec vgga = compute_gga_potential(density_g, vsigma, basis, fft);

    // All values should be finite
    for (int i = 0; i < num_grid; ++i) {
        EXPECT_TRUE(std::isfinite(vgga[i]))
            << "GGA potential should be finite at point " << i;
    }
}
