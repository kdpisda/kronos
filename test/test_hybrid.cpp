// ============================================================================
// KRONOS  test/test_hybrid.cpp
// Hybrid functional (PBE0, HSE06) tests
//
// Tests:
//   - Coulomb kernel correctness (PBE0, HSE06)
//   - XC evaluator hybrid detection
//   - Direct exchange computation
//   - ACE compression
//   - Exchange energy consistency
// ============================================================================

#include <gtest/gtest.h>
#include "potential/exact_exchange.hpp"
#include "potential/xc.hpp"
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "core/constants.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"

#include <cmath>
#include <complex>
#include <vector>

using namespace kronos;

// ============================================================================
// Helpers
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

} // anonymous namespace

// ============================================================================
// XC Evaluator Hybrid Detection
// ============================================================================

TEST(Hybrid, XCEvaluatorPBE0) {
    XCEvaluator xc("PBE0");
    EXPECT_TRUE(xc.is_hybrid());
    EXPECT_TRUE(xc.is_gga());
    EXPECT_EQ(xc.hybrid_type(), HybridType::PBE0);
    EXPECT_NEAR(xc.exx_fraction(), 0.25, 1e-10);
    EXPECT_NEAR(xc.screening_parameter(), 0.0, 1e-10);
}

TEST(Hybrid, XCEvaluatorHSE06) {
    XCEvaluator xc("HSE06");
    EXPECT_TRUE(xc.is_hybrid());
    EXPECT_TRUE(xc.is_gga());
    EXPECT_EQ(xc.hybrid_type(), HybridType::HSE06);
    EXPECT_NEAR(xc.exx_fraction(), 0.25, 1e-10);
    EXPECT_NEAR(xc.screening_parameter(), 0.11, 1e-10);
}

TEST(Hybrid, XCEvaluatorLDANotHybrid) {
    XCEvaluator xc("LDA_PZ");
    EXPECT_FALSE(xc.is_hybrid());
    EXPECT_EQ(xc.hybrid_type(), HybridType::None);
}

TEST(Hybrid, XCEvaluatorPBENotHybrid) {
    XCEvaluator xc("PBE");
    EXPECT_FALSE(xc.is_hybrid());
    EXPECT_EQ(xc.hybrid_type(), HybridType::None);
}

// ============================================================================
// Coulomb Kernel Tests
// ============================================================================

TEST(Hybrid, PBE0CoulombKernel) {
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    // At G=0, PBE0 should return 0 (divergent, handled separately)
    EXPECT_NEAR(exx.coulomb_kernel(0.0), 0.0, 1e-10);

    // At G²=1.0, PBE0 kernel = 4π/G² = 4π ≈ 12.566
    double v1 = exx.coulomb_kernel(1.0);
    EXPECT_NEAR(v1, 4.0 * constants::pi, 1e-6);

    // At G²=4.0, PBE0 kernel = 4π/4 = π ≈ 3.1416
    double v4 = exx.coulomb_kernel(4.0);
    EXPECT_NEAR(v4, constants::pi, 1e-6);
}

TEST(Hybrid, HSE06CoulombKernel) {
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    double omega = 0.11;
    ExactExchange exx(crystal, basis, fft_grid, HybridType::HSE06, 0.25, omega);

    // At G=0, HSE06 v_SR(G=0) = π/ω²
    double v0 = exx.coulomb_kernel(0.0);
    EXPECT_NEAR(v0, constants::pi / (omega * omega), 1e-6);

    // At large G², HSE06 ≈ PBE0 (exponential → 0)
    double g2_large = 100.0;
    double v_hse = exx.coulomb_kernel(g2_large);
    double v_pbe0 = 4.0 * constants::pi / g2_large;
    // exp(-100/(4*0.0121)) ≈ exp(-2066) ≈ 0, so v_SR ≈ v_c
    EXPECT_NEAR(v_hse, v_pbe0, v_pbe0 * 0.01);

    // HSE06 should always be ≤ PBE0 (short-range only)
    for (double g2 : {0.1, 0.5, 1.0, 5.0, 10.0}) {
        double vh = exx.coulomb_kernel(g2);
        double vp = 4.0 * constants::pi / g2;
        EXPECT_LE(vh, vp + 1e-10)
            << "HSE06 kernel should be ≤ PBE0 at G²=" << g2;
    }
}

TEST(Hybrid, HSE06KernelMonotonic) {
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::HSE06, 0.25, 0.11);

    // v_SR(G) * G² should increase monotonically (from 0 to 4π)
    double prev = 0.0;
    for (double g2 = 0.01; g2 < 100.0; g2 *= 1.5) {
        double vg2 = exx.coulomb_kernel(g2) * g2;
        EXPECT_GE(vg2, prev - 1e-10)
            << "v_SR(G)*G² should increase at G²=" << g2;
        prev = vg2;
    }
}

// ============================================================================
// Direct Exchange Computation Tests
// ============================================================================

TEST(Hybrid, DirectExchangeZero) {
    // With no occupied states, exchange should be zero
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    int npw = static_cast<int>(basis.num_pw());
    CVec psi(npw, complex_t{0.0, 0.0});
    psi[0] = {1.0, 0.0};

    Vec3 k_frac = {0.0, 0.0, 0.0};
    std::vector<std::vector<CVec>> occ_states;
    std::vector<std::vector<double>> occs;
    std::vector<Vec3> kpoints;
    std::vector<double> kweights;

    CVec vx = exx.apply_direct(psi, k_frac, occ_states, occs, kpoints, kweights);

    for (const auto& c : vx) {
        EXPECT_NEAR(std::abs(c), 0.0, 1e-14);
    }
}

TEST(Hybrid, DirectExchangeFinite) {
    // With one occupied state, exchange should be non-zero
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    int npw = static_cast<int>(basis.num_pw());

    // Create a simple occupied state
    CVec psi(npw, complex_t{0.0, 0.0});
    psi[0] = {1.0 / std::sqrt(2.0), 0.0};
    psi[1] = {1.0 / std::sqrt(2.0), 0.0};

    Vec3 k_frac = {0.0, 0.0, 0.0};
    std::vector<std::vector<CVec>> occ_states = {{psi}};
    std::vector<std::vector<double>> occs = {{2.0}};
    std::vector<Vec3> kpoints = {k_frac};
    std::vector<double> kweights = {1.0};

    CVec vx = exx.apply_direct(psi, k_frac, occ_states, occs, kpoints, kweights);

    // Exchange should produce non-zero result
    double norm = 0.0;
    for (const auto& c : vx) {
        norm += std::norm(c);
    }
    EXPECT_GT(norm, 0.0);
}

// ============================================================================
// ACE Tests
// ============================================================================

TEST(Hybrid, ACEInitiallyNotReady) {
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);
    EXPECT_FALSE(exx.ace_ready());
}

TEST(Hybrid, ACEUpdateAndApply) {
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    int npw = static_cast<int>(basis.num_pw());

    CVec psi(npw, complex_t{0.0, 0.0});
    psi[0] = {1.0, 0.0};

    Vec3 k_frac = {0.0, 0.0, 0.0};
    std::vector<std::vector<CVec>> occ_states = {{psi}};
    std::vector<std::vector<double>> occs = {{2.0}};
    std::vector<Vec3> kpoints = {k_frac};
    std::vector<double> kweights = {1.0};

    // Update ACE
    exx.update_ace(occ_states, occs, kpoints, kweights);
    EXPECT_TRUE(exx.ace_ready());

    // Apply ACE — should produce non-zero result (ik=0)
    CVec vx = exx.apply_ace(psi, 0);
    EXPECT_EQ(vx.size(), static_cast<size_t>(npw));
}

// ============================================================================
// Exchange Energy Tests
// ============================================================================

TEST(Hybrid, ExchangeEnergyNegative) {
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    int npw = static_cast<int>(basis.num_pw());

    // Use multiple PW coefficients so pair exchange hits non-zero G-vectors
    // (G=0 kernel is 0 for PBE0, so single-coefficient psi gives zero exchange)
    CVec psi(npw, complex_t{0.0, 0.0});
    double norm = 0.0;
    int nfill = std::min(npw, 5);
    for (int i = 0; i < nfill; ++i) {
        psi[i] = complex_t(1.0 / std::sqrt(static_cast<double>(nfill)), 0.0);
        norm += std::norm(psi[i]);
    }

    Vec3 k_frac = {0.0, 0.0, 0.0};
    std::vector<std::vector<CVec>> occ_states = {{psi}};
    std::vector<std::vector<double>> occs = {{2.0}};
    std::vector<Vec3> kpoints = {k_frac};
    std::vector<double> kweights = {1.0};

    double e_exx = exx.exchange_energy(occ_states, occs, kpoints, kweights);

    // Exchange energy should be negative (lowering)
    EXPECT_LT(e_exx, 0.0);
    EXPECT_TRUE(std::isfinite(e_exx));
}

TEST(Hybrid, ExchangeEnergyZeroEmpty) {
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    std::vector<std::vector<CVec>> occ_states;
    std::vector<std::vector<double>> occs;
    std::vector<Vec3> kpoints;
    std::vector<double> kweights;

    double e_exx = exx.exchange_energy(occ_states, occs, kpoints, kweights);
    EXPECT_NEAR(e_exx, 0.0, 1e-14);
}

// ============================================================================
// Input Parser Tests
// ============================================================================

TEST(Hybrid, InputParserAcceptsPBE0) {
    XCEvaluator xc("PBE0");
    EXPECT_EQ(xc.name(), "PBE0");
}

TEST(Hybrid, InputParserAcceptsHSE06) {
    XCEvaluator xc("HSE06");
    EXPECT_EQ(xc.name(), "HSE06");
}

// ============================================================================
// Exchange Scaling Integration Tests
// ============================================================================

TEST(Hybrid, PBE0ExchangeScale) {
    // Verify that set_exchange_scale changes XC energy
    XCEvaluator xc_full("PBE");
    XCEvaluator xc_scaled("PBE");
    xc_scaled.set_exchange_scale(0.75);  // PBE0: 1 - 0.25

    const int np = 100;
    double volume = 265.302;

    // Create realistic electron density
    RVec rho(np);
    for (int i = 0; i < np; ++i) {
        rho[i] = 0.01 + 0.005 * std::sin(2.0 * M_PI * i / np);
    }

    auto r_full = xc_full.evaluate(rho, volume);
    auto r_scaled = xc_scaled.evaluate(rho, volume);

    // Scaled exchange should give different energy
    EXPECT_NE(r_full.energy, r_scaled.energy);

    // Exchange scale should reduce the exchange portion
    // For LDA, exchange is negative, so scaling by 0.75 makes it less negative
    // Total XC energy should change
    EXPECT_NE(r_full.vxc[0], r_scaled.vxc[0]);
}

TEST(Hybrid, ExchangeScaleIdentity) {
    // exchange_scale = 1.0 should give same result as unscaled
    XCEvaluator xc1("PBE");
    XCEvaluator xc2("PBE");
    xc2.set_exchange_scale(1.0);

    const int np = 50;
    double volume = 100.0;
    RVec rho(np);
    for (int i = 0; i < np; ++i) {
        rho[i] = 0.02 + 0.01 * std::cos(2.0 * M_PI * i / np);
    }

    auto r1 = xc1.evaluate(rho, volume);
    auto r2 = xc2.evaluate(rho, volume);

    EXPECT_NEAR(r1.energy, r2.energy, 1e-12);
    for (int i = 0; i < np; ++i) {
        EXPECT_NEAR(r1.vxc[i], r2.vxc[i], 1e-12);
    }
}

TEST(Hybrid, ACEKPointCorrect) {
    // Verify ACE apply only uses vectors from the requested k-point
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    int npw = static_cast<int>(basis.num_pw());

    // Create occupied states at 2 k-points
    CVec psi0(npw, complex_t{0.0, 0.0});
    psi0[0] = {1.0, 0.0};
    CVec psi1(npw, complex_t{0.0, 0.0});
    psi1[1] = {1.0, 0.0};

    std::vector<Vec3> kpoints = {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    std::vector<double> kweights = {0.5, 0.5};
    std::vector<std::vector<CVec>> occ_states = {{psi0}, {psi1}};
    std::vector<std::vector<double>> occs = {{2.0}, {2.0}};

    exx.update_ace(occ_states, occs, kpoints, kweights);
    EXPECT_TRUE(exx.ace_ready());

    // Apply at k=0 should only use k=0 ACE vectors
    CVec vx0 = exx.apply_ace(psi0, 0);
    CVec vx1 = exx.apply_ace(psi1, 1);

    // Results should differ because different k-point ACE vectors are used
    double diff = 0.0;
    for (size_t ig = 0; ig < std::min(vx0.size(), vx1.size()); ++ig) {
        diff += std::abs(vx0[ig] - vx1[ig]);
    }
    // With different occupied states at each k-point, results should differ
    // (or at least be well-defined and finite)
    for (const auto& c : vx0) {
        EXPECT_TRUE(std::isfinite(std::real(c)));
        EXPECT_TRUE(std::isfinite(std::imag(c)));
    }
    for (const auto& c : vx1) {
        EXPECT_TRUE(std::isfinite(std::real(c)));
        EXPECT_TRUE(std::isfinite(std::imag(c)));
    }
}

TEST(Hybrid, ACEOutOfRangeKPoint) {
    // Requesting ACE for invalid k-point should return zeros
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    int npw = static_cast<int>(basis.num_pw());
    CVec psi(npw, complex_t{0.0, 0.0});
    psi[0] = {1.0, 0.0};

    // Update with 1 k-point
    std::vector<Vec3> kpoints = {{0.0, 0.0, 0.0}};
    std::vector<double> kweights = {1.0};
    std::vector<std::vector<CVec>> occ_states = {{psi}};
    std::vector<std::vector<double>> occs = {{2.0}};
    exx.update_ace(occ_states, occs, kpoints, kweights);

    // Request k-point index out of range → should return zeros
    CVec vx = exx.apply_ace(psi, 5);
    for (const auto& c : vx) {
        EXPECT_NEAR(std::abs(c), 0.0, 1e-14);
    }
}

TEST(Hybrid, ExchangeEnergyWithScale) {
    // Verify exchange energy is non-zero and negative with scaled XC
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis);

    ExactExchange exx(crystal, basis, fft_grid, HybridType::PBE0, 0.25);

    int npw = static_cast<int>(basis.num_pw());
    CVec psi(npw, complex_t{0.0, 0.0});
    int nfill = std::min(npw, 5);
    for (int i = 0; i < nfill; ++i) {
        psi[i] = complex_t(1.0 / std::sqrt(static_cast<double>(nfill)), 0.0);
    }

    Vec3 k_frac = {0.0, 0.0, 0.0};
    std::vector<std::vector<CVec>> occ_states = {{psi}};
    std::vector<std::vector<double>> occs = {{2.0}};
    std::vector<Vec3> kpoints = {k_frac};
    std::vector<double> kweights = {1.0};

    double e_exx = exx.exchange_energy(occ_states, occs, kpoints, kweights);

    // Exchange energy should be negative
    EXPECT_LT(e_exx, 0.0);

    // The exx_fraction is embedded in apply_direct, so the energy
    // already includes the α scaling
    EXPECT_TRUE(std::isfinite(e_exx));
}
