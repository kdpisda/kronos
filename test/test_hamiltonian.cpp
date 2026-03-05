// ============================================================================
// KRONOS  test/test_hamiltonian.cpp
// Tests for the Hamiltonian operator: kinetic, local, nonlocal, properties.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "potential/nonlocal_pp.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

using namespace kronos;

namespace {

// ============================================================================
// Helpers
// ============================================================================

// Build a Hamiltonian with zero potential (free electron)
struct FreeElectronFixture {
    Crystal crystal;
    PlaneWaveBasis basis;
    FFTGrid fft;
    std::map<std::string, PseudoPotential> pp_map;
    NonlocalPP nonlocal;
    Hamiltonian ham;
    int npw;

    FreeElectronFixture(double ecutwfc = 8.0)
        : crystal(test::make_cubic_crystal(5.0, "X", 1)),
          basis(crystal, ecutwfc),
          fft(basis),
          pp_map({{"X", test::make_empty_pp("X", 0.0)}}),
          nonlocal(crystal, basis, pp_map),
          ham(crystal, basis, fft, nonlocal),
          npw(static_cast<int>(basis.num_pw()))
    {
        std::vector<complex_t> veff_r(fft.total_points(), complex_t{0.0, 0.0});
        ham.update_veff(veff_r);
    }
};

// Generate a random complex vector
CVec random_cvec(int n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    CVec v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = complex_t{dist(gen), dist(gen)};
    }
    return v;
}

// Complex inner product: <a|b> = sum conj(a_i) * b_i
complex_t inner_product(const CVec& a, const CVec& b) {
    complex_t result{0.0, 0.0};
    for (size_t i = 0; i < a.size(); ++i) {
        result += std::conj(a[i]) * b[i];
    }
    return result;
}

} // anonymous namespace

// ============================================================================
// Kinetic-only: H|ψ⟩ = |G|²/2 · ψ_G for free electron gas
// ============================================================================

TEST(Hamiltonian, KineticOnlySinglePlaneWave) {
    FreeElectronFixture fixture;
    Vec3 k_gamma = {0.0, 0.0, 0.0};
    auto h_apply = fixture.ham.get_apply_function(k_gamma);
    auto kinetic = fixture.basis.kinetic_energies(k_gamma);

    // Apply H to each basis vector and check
    for (int ig = 0; ig < std::min(fixture.npw, 5); ++ig) {
        CVec e_g(fixture.npw, complex_t{0.0, 0.0});
        e_g[ig] = complex_t{1.0, 0.0};

        CVec h_eg = h_apply(e_g);
        double rayleigh = std::real(std::conj(e_g[ig]) * h_eg[ig]);
        EXPECT_NEAR(rayleigh, kinetic[ig], 1e-6)
            << "Kinetic energy mismatch for G-vector " << ig;
    }
}

TEST(Hamiltonian, KineticOnlyNonzeroK) {
    FreeElectronFixture fixture;
    Vec3 k = {0.25, 0.0, 0.0};
    auto h_apply = fixture.ham.get_apply_function(k);
    auto kinetic = fixture.basis.kinetic_energies(k);

    CVec e0(fixture.npw, complex_t{0.0, 0.0});
    e0[0] = complex_t{1.0, 0.0};
    CVec h_e0 = h_apply(e0);

    double rayleigh = std::real(std::conj(e0[0]) * h_e0[0]);
    // At k=(0.25,0,0), the kinetic energy of G=0 is |k|^2/2 > 0
    EXPECT_NEAR(rayleigh, kinetic[0], 1e-6);
    EXPECT_GT(kinetic[0], 0.0);
}

// ============================================================================
// Local potential: verify V_eff applied in real space via FFT
// ============================================================================

TEST(Hamiltonian, LocalPotentialContribution) {
    Crystal crystal = test::make_cubic_crystal(5.0, "Si");
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());

    auto pp_map = test::make_si_pp_map();
    NonlocalPP nonlocal(crystal, basis, pp_map);
    Hamiltonian ham(crystal, basis, fft, nonlocal);

    int num_grid = fft.total_points();

    // Constant potential: V_eff(r) = V0 for all r
    double V0 = 0.5;
    std::vector<complex_t> veff_r(num_grid, complex_t{V0, 0.0});
    ham.update_veff(veff_r);

    Vec3 k_gamma = {0.0, 0.0, 0.0};
    auto h_apply = ham.get_apply_function(k_gamma);
    auto kinetic = basis.kinetic_energies(k_gamma);

    // For a constant potential, H|e_G> = (KE_G + V0)|e_G>
    // (the constant potential just shifts all eigenvalues)
    CVec e0(npw, complex_t{0.0, 0.0});
    e0[0] = complex_t{1.0, 0.0};
    CVec h_e0 = h_apply(e0);

    double rayleigh = std::real(std::conj(e0[0]) * h_e0[0]);
    EXPECT_NEAR(rayleigh, kinetic[0] + V0, 0.01)
        << "Constant potential should shift eigenvalue by V0";
}

// ============================================================================
// Hermiticity: ⟨φ|H|ψ⟩ = ⟨ψ|H|φ⟩*
// ============================================================================

TEST(Hamiltonian, Hermiticity) {
    FreeElectronFixture fixture;
    Vec3 k = {0.0, 0.0, 0.0};
    auto h_apply = fixture.ham.get_apply_function(k);

    CVec psi = random_cvec(fixture.npw, 42);
    CVec phi = random_cvec(fixture.npw, 123);

    CVec h_psi = h_apply(psi);
    CVec h_phi = h_apply(phi);

    complex_t phi_H_psi = inner_product(phi, h_psi);
    complex_t psi_H_phi = inner_product(psi, h_phi);

    // <phi|H|psi> should equal <psi|H|phi>*
    EXPECT_NEAR(phi_H_psi.real(), psi_H_phi.real(), 1e-8)
        << "Hermiticity violated: real parts differ";
    EXPECT_NEAR(phi_H_psi.imag(), -psi_H_phi.imag(), 1e-8)
        << "Hermiticity violated: imaginary parts not conjugate";
}

TEST(Hamiltonian, HermiticityWithPotential) {
    Crystal crystal = test::make_cubic_crystal(5.0, "Si");
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());

    auto pp_map = test::make_si_pp_map();
    NonlocalPP nonlocal(crystal, basis, pp_map);
    Hamiltonian ham(crystal, basis, fft, nonlocal);

    int num_grid = fft.total_points();

    // Real-valued potential (required for Hermiticity)
    std::mt19937 gen(99);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<complex_t> veff_r(num_grid);
    for (int i = 0; i < num_grid; ++i) {
        veff_r[i] = complex_t{dist(gen), 0.0};
    }
    ham.update_veff(veff_r);

    Vec3 k = {0.1, 0.2, 0.0};
    auto h_apply = ham.get_apply_function(k);

    CVec psi = random_cvec(npw, 42);
    CVec phi = random_cvec(npw, 123);

    CVec h_psi = h_apply(psi);
    CVec h_phi = h_apply(phi);

    complex_t phi_H_psi = inner_product(phi, h_psi);
    complex_t psi_H_phi = inner_product(psi, h_phi);

    EXPECT_NEAR(phi_H_psi.real(), psi_H_phi.real(), 1e-6);
    EXPECT_NEAR(phi_H_psi.imag(), -psi_H_phi.imag(), 1e-6);
}

// ============================================================================
// Linearity: H(αψ+βφ) = αH(ψ) + βH(φ)
// ============================================================================

TEST(Hamiltonian, Linearity) {
    FreeElectronFixture fixture;
    Vec3 k = {0.0, 0.0, 0.0};
    auto h_apply = fixture.ham.get_apply_function(k);

    CVec psi = random_cvec(fixture.npw, 42);
    CVec phi = random_cvec(fixture.npw, 123);
    complex_t alpha{0.7, 0.3};
    complex_t beta{-0.2, 0.5};

    // H(alpha*psi + beta*phi)
    CVec combination(fixture.npw);
    for (int i = 0; i < fixture.npw; ++i) {
        combination[i] = alpha * psi[i] + beta * phi[i];
    }
    CVec h_combo = h_apply(combination);

    // alpha*H(psi) + beta*H(phi)
    CVec h_psi = h_apply(psi);
    CVec h_phi = h_apply(phi);
    CVec linear_combo(fixture.npw);
    for (int i = 0; i < fixture.npw; ++i) {
        linear_combo[i] = alpha * h_psi[i] + beta * h_phi[i];
    }

    for (int i = 0; i < fixture.npw; ++i) {
        EXPECT_NEAR(h_combo[i].real(), linear_combo[i].real(), 1e-8)
            << "Linearity violated at component " << i << " (real)";
        EXPECT_NEAR(h_combo[i].imag(), linear_combo[i].imag(), 1e-8)
            << "Linearity violated at component " << i << " (imag)";
    }
}

// ============================================================================
// Kinetic diagonal (preconditioner)
// ============================================================================

TEST(Hamiltonian, KineticDiagonalMatchesBasisKinetic) {
    FreeElectronFixture fixture;
    Vec3 k = {0.1, 0.2, 0.3};
    auto diag = fixture.ham.kinetic_diagonal(k);
    auto ke = fixture.basis.kinetic_energies(k);

    ASSERT_EQ(diag.size(), ke.size());
    for (size_t i = 0; i < diag.size(); ++i) {
        EXPECT_NEAR(diag[i], ke[i], 1e-12);
    }
}

// ============================================================================
// H application preserves vector size
// ============================================================================

TEST(Hamiltonian, OutputSizeMatchesInput) {
    FreeElectronFixture fixture;
    Vec3 k = {0.0, 0.0, 0.0};
    auto h_apply = fixture.ham.get_apply_function(k);

    CVec psi = random_cvec(fixture.npw, 42);
    CVec h_psi = h_apply(psi);

    EXPECT_EQ(h_psi.size(), psi.size());
}

// ============================================================================
// Rayleigh quotient bounds
// ============================================================================

TEST(Hamiltonian, RayleighQuotientNonNegative) {
    FreeElectronFixture fixture;
    Vec3 k = {0.0, 0.0, 0.0};
    auto h_apply = fixture.ham.get_apply_function(k);

    // For free electron (V=0), <psi|H|psi> >= 0 for all psi
    for (unsigned seed = 0; seed < 5; ++seed) {
        CVec psi = random_cvec(fixture.npw, seed);
        CVec h_psi = h_apply(psi);
        complex_t expectation = inner_product(psi, h_psi);
        double norm_sq = std::real(inner_product(psi, psi));

        // <psi|T|psi> / <psi|psi> >= 0
        EXPECT_GE(expectation.real() / norm_sq, -1e-10)
            << "Kinetic energy Rayleigh quotient should be non-negative";
        EXPECT_NEAR(expectation.imag(), 0.0, 1e-8)
            << "Hermitian operator: Rayleigh quotient should be real";
    }
}

// ============================================================================
// Nonlocal PP effect
// ============================================================================

TEST(Hamiltonian, NonlocalPPChangesEigenvalues) {
    Crystal crystal = test::make_cubic_crystal(5.0, "Si");
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());

    // Create PP with a nonlocal projector
    PseudoPotential pp = test::make_si_pseudopotential();
    // Add a beta projector
    BetaProjector beta;
    beta.index = 1;
    beta.angular_momentum = 0;
    beta.cutoff_index = pp.mesh.npoints - 1;
    beta.values.resize(pp.mesh.npoints);
    for (int i = 0; i < pp.mesh.npoints; ++i) {
        double r = pp.mesh.r[i];
        beta.values[i] = r * std::exp(-r * r);
    }
    pp.betas.push_back(beta);
    pp.num_projectors = 1;
    pp.lmax = 0;
    pp.dij = {{1.0}};

    std::map<std::string, PseudoPotential> pp_map = {{"Si", pp}};
    NonlocalPP nonlocal(crystal, basis, pp_map);
    Hamiltonian ham(crystal, basis, fft, nonlocal);

    // Zero local potential
    std::vector<complex_t> veff_r(fft.total_points(), complex_t{0.0, 0.0});
    ham.update_veff(veff_r);

    Vec3 k = {0.0, 0.0, 0.0};
    auto h_apply = ham.get_apply_function(k);
    auto kinetic = basis.kinetic_energies(k);

    // Check that the nonlocal PP changes the diagonal
    CVec e0(npw, complex_t{0.0, 0.0});
    e0[0] = complex_t{1.0, 0.0};
    CVec h_e0 = h_apply(e0);

    double rayleigh = std::real(std::conj(e0[0]) * h_e0[0]);
    // With nonlocal PP, the eigenvalue should differ from pure kinetic
    // (unless the projector happens to be zero for this G-vector, which is unlikely)
    bool differs = std::abs(rayleigh - kinetic[0]) > 1e-10;

    // It's possible the nonlocal contribution is very small for G=0, so also check
    // off-diagonal elements (nonlocal PP couples different G-vectors)
    double off_diag = 0.0;
    for (int j = 1; j < std::min(npw, 10); ++j) {
        off_diag += std::norm(h_e0[j]);
    }
    bool has_off_diagonal = off_diag > 1e-12;

    EXPECT_TRUE(differs || has_off_diagonal)
        << "Nonlocal PP should modify Hamiltonian matrix elements";
}

// ============================================================================
// Zero wavefunction gives zero result
// ============================================================================

TEST(Hamiltonian, ZeroInputGivesZeroOutput) {
    FreeElectronFixture fixture;
    Vec3 k = {0.0, 0.0, 0.0};
    auto h_apply = fixture.ham.get_apply_function(k);

    CVec zero(fixture.npw, complex_t{0.0, 0.0});
    CVec result = h_apply(zero);

    for (int i = 0; i < fixture.npw; ++i) {
        EXPECT_NEAR(std::abs(result[i]), 0.0, 1e-15);
    }
}

// ============================================================================
// Scaling: H(c*psi) = c*H(psi)
// ============================================================================

TEST(Hamiltonian, ScalingProperty) {
    FreeElectronFixture fixture;
    Vec3 k = {0.0, 0.0, 0.0};
    auto h_apply = fixture.ham.get_apply_function(k);

    CVec psi = random_cvec(fixture.npw, 42);
    complex_t c{2.5, -1.3};

    CVec scaled(fixture.npw);
    for (int i = 0; i < fixture.npw; ++i) {
        scaled[i] = c * psi[i];
    }

    CVec h_psi = h_apply(psi);
    CVec h_scaled = h_apply(scaled);

    for (int i = 0; i < fixture.npw; ++i) {
        complex_t expected = c * h_psi[i];
        EXPECT_NEAR(h_scaled[i].real(), expected.real(), 1e-8);
        EXPECT_NEAR(h_scaled[i].imag(), expected.imag(), 1e-8);
    }
}
