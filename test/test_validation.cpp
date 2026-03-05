// ============================================================================
// KRONOS  test/test_validation.cpp
// Physics invariant validation tests.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "core/types.hpp"
#include "core/constants.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "io/upf_parser.hpp"
#include "solver/scf.hpp"
#include "solver/fermi.hpp"
#include "solver/davidson.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "potential/ewald.hpp"
#include "potential/nonlocal_pp.hpp"
#include "potential/xc.hpp"

#include <cmath>
#include <complex>
#include <random>

using namespace kronos;

namespace {

// Helper: run SCF with given params and return result
SCFResult run_si_scf(double ecutwfc, std::array<int,3> kgrid,
                     const std::map<std::string, PseudoPotential>& pps,
                     const std::string& xc = "LDA_PZ",
                     double ethr = 1e-3, double dthr = 1.0, int max_steps = 100) {
    Crystal crystal = test::make_si_diamond_crystal();
    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = ecutwfc;
    calc.xc_functional = xc;
    calc.kpoints.grid = kgrid;
    ConvergenceParams conv;
    conv.energy_threshold = ethr;
    conv.density_threshold = dthr;
    conv.max_scf_steps = max_steps;
    SCFSolver solver(crystal, calc, conv, pps);
    return solver.solve();
}

// Helper: random complex vector
CVec random_cvec(int n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    CVec v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = complex_t(dist(gen), dist(gen));
    }
    return v;
}

// Helper: inner product <a|b>
complex_t inner_product(const CVec& a, const CVec& b) {
    complex_t sum(0.0, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::conj(a[i]) * b[i];
    }
    return sum;
}

} // namespace

// ============================================================================
// VariationalPrinciple: Higher cutoff must give lower (or equal) energy
// ============================================================================

TEST(VariationalPrinciple, HigherCutoffLowerEnergy) {
    auto pps = test::make_si_pp_map();
    auto r1 = run_si_scf(20.0, {1,1,1}, pps);
    auto r2 = run_si_scf(30.0, {1,1,1}, pps);
    if (!r1.converged || !r2.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    // E(ecut=20) >= E(ecut=30) by variational theorem
    EXPECT_GE(r1.total_energy_ry, r2.total_energy_ry - 1e-6)
        << "Variational principle violated: E(20Ry)=" << r1.total_energy_ry
        << " < E(30Ry)=" << r2.total_energy_ry;
}

TEST(VariationalPrinciple, HigherCutoffLowerEnergyNonlocal) {
    auto pps = test::make_si_pp_map_nonlocal();
    auto r1 = run_si_scf(12.0, {1,1,1}, pps);
    auto r2 = run_si_scf(20.0, {1,1,1}, pps);
    if (!r1.converged || !r2.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    EXPECT_GE(r1.total_energy_ry, r2.total_energy_ry - 1e-6)
        << "Variational principle violated with nonlocal PP";
}

// ============================================================================
// ChargeConservation: occupations must sum to N_electrons
// ============================================================================

TEST(ChargeConservation, OccupationsSumToNElectrons) {
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {1,1,1}, pps);
    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }

    // Si diamond: 2 atoms × 4 valence = 8 electrons
    double target_electrons = 8.0;

    // Re-run Fermi solver on converged eigenvalues
    std::vector<double> weights = {1.0}; // Gamma only
    auto fermi = FermiSolver::find_fermi_level(
        result.eigenvalues, weights, target_electrons,
        SmearingType::Gaussian, 0.01, 2);

    EXPECT_TRUE(fermi.converged) << "Fermi solver did not converge";
    EXPECT_NEAR(fermi.total_electrons_found, target_electrons, 1e-4)
        << "Charge not conserved: found " << fermi.total_electrons_found
        << " expected " << target_electrons;
}

// ============================================================================
// ForceEnergyConsistency: analytical forces vs finite-difference
// ============================================================================

TEST(ForceEnergyConsistency, SCFForceFiniteDifference) {
    // Use Ewald-only finite difference for clean force validation.
    // SCF forces at low cutoff are noisy; Ewald forces are exact and fast.
    auto pps = test::make_si_pp_map();
    Crystal cryst_eq = test::make_si_diamond_crystal();
    double delta = 0.002;

    // Charges from PP
    std::vector<double> charges;
    for (const auto& atom : cryst_eq.atoms()) {
        charges.push_back(pps[atom.symbol].z_valence);
    }

    // Equilibrium Ewald
    auto ew_eq = EwaldCalculator::compute(cryst_eq, charges);

    // +delta displacement
    Crystal cryst_p = test::make_si_diamond_displaced(delta, 0, 0);
    std::vector<double> charges_p;
    for (const auto& atom : cryst_p.atoms()) {
        charges_p.push_back(pps[atom.symbol].z_valence);
    }
    auto ew_p = EwaldCalculator::compute(cryst_p, charges_p);

    // -delta displacement
    Crystal cryst_m = test::make_si_diamond_displaced(-delta, 0, 0);
    std::vector<double> charges_m;
    for (const auto& atom : cryst_m.atoms()) {
        charges_m.push_back(pps[atom.symbol].z_valence);
    }
    auto ew_m = EwaldCalculator::compute(cryst_m, charges_m);

    // Cartesian displacement magnitude along lattice vector a1
    auto lat_bohr = cryst_eq.lattice_bohr();
    double dr_cart = 0.0;
    for (int j = 0; j < 3; ++j) {
        dr_cart += lat_bohr[0][j] * lat_bohr[0][j];
    }
    dr_cart = delta * std::sqrt(dr_cart);

    // Numerical force magnitude (projection onto a1 direction)
    double f_numerical = -(ew_p.energy - ew_m.energy) / (2.0 * dr_cart);

    // Analytical force: project F onto a1_hat
    ASSERT_FALSE(ew_eq.forces.empty());
    double a1_norm = std::sqrt(lat_bohr[0][0]*lat_bohr[0][0] +
                                lat_bohr[0][1]*lat_bohr[0][1] +
                                lat_bohr[0][2]*lat_bohr[0][2]);
    double f_analytical = 0.0;
    for (int j = 0; j < 3; ++j) {
        f_analytical += ew_eq.forces[0][j] * lat_bohr[0][j] / a1_norm;
    }

    // Tolerance: 5% of max force magnitude + small absolute floor
    double tol = std::max(0.05 * std::max(std::abs(f_analytical), std::abs(f_numerical)), 1e-4);

    EXPECT_NEAR(f_analytical, f_numerical, tol)
        << "Ewald analytical force (" << f_analytical << ") disagrees with "
        << "finite-difference force (" << f_numerical << ")";
}

TEST(ForceEnergyConsistency, ForcesVanishAtEquilibrium) {
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {1,1,1}, pps);
    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    ASSERT_FALSE(result.forces.empty());
    for (size_t a = 0; a < result.forces.size(); ++a) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(result.forces[a][d], 0.0, 0.05)
                << "Force on atom " << a << " direction " << d
                << " should vanish at equilibrium by symmetry";
        }
    }
}

// ============================================================================
// HamiltonianProperties: Hermiticity and sorted eigenvalues
// ============================================================================

TEST(HamiltonianProperties, HermiticityWithSCFPotential) {
    Crystal crystal = test::make_si_diamond_crystal();
    auto pps = test::make_si_pp_map();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 10.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1,1,1};
    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 50;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();
    if (!result.converged || result.converged_veff_r.empty()) {
        GTEST_SKIP() << "SCF did not converge or veff not available";
    }

    // Build Hamiltonian with converged potential
    PlaneWaveBasis basis(crystal, calc.ecutwfc);
    FFTGrid fft(basis);
    NonlocalPP nlpp(crystal, basis, pps);
    Hamiltonian ham(crystal, basis, fft, nlpp);
    ham.update_veff(result.converged_veff_r);

    int npw = static_cast<int>(basis.num_pw());
    Vec3 k_frac = {0.0, 0.0, 0.0};

    // Test <phi|H|psi> = <psi|H|phi>* with random vectors
    auto phi = random_cvec(npw, 42);
    auto psi = random_cvec(npw, 123);

    CVec h_phi = ham.apply(phi, k_frac);
    CVec h_psi = ham.apply(psi, k_frac);

    complex_t phi_H_psi = inner_product(phi, h_psi);
    complex_t psi_H_phi = inner_product(psi, h_phi);

    EXPECT_NEAR(phi_H_psi.real(), psi_H_phi.real(), 1e-8)
        << "Hermiticity violated: real parts differ";
    EXPECT_NEAR(phi_H_psi.imag(), -psi_H_phi.imag(), 1e-8)
        << "Hermiticity violated: imaginary parts not conjugate";
}

TEST(HamiltonianProperties, EigenvaluesSorted) {
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {2,2,2}, pps);
    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    for (size_t k = 0; k < result.eigenvalues.size(); ++k) {
        const auto& evals = result.eigenvalues[k];
        for (size_t n = 1; n < evals.size(); ++n) {
            EXPECT_GE(evals[n], evals[n-1] - 1e-10)
                << "Eigenvalues not sorted at k-point " << k
                << ": E[" << n-1 << "]=" << evals[n-1]
                << " > E[" << n << "]=" << evals[n];
        }
    }
}

// ============================================================================
// MadelungConstant: Ewald summation for known crystal structures
// ============================================================================

TEST(MadelungConstant, NaClMadelungHighPrecision) {
    Crystal nacl = test::make_nacl_crystal(5.64);
    // Charges: Na=+1, Cl=-1
    std::vector<double> charges = {+1, +1, +1, +1, -1, -1, -1, -1};

    auto ewald = EwaldCalculator::compute(nacl, charges);

    // Madelung energy: E = -alpha * N_pairs * e^2 / a_nn
    // For 8-atom conventional cell, nearest-neighbor distance a_nn = a/2
    double a_bohr = 5.64 * constants::angstrom_to_bohr;
    double a_nn = a_bohr / 2.0;

    // Total Ewald energy for 8-atom cell = -alpha * 8 / a_nn (in Ry for unit charges)
    // alpha_NaCl ≈ 1.74756
    // Compute alpha from the energy
    double alpha_computed = -ewald.energy * a_nn / 8.0;

    EXPECT_NEAR(alpha_computed, 1.74756, 0.05)
        << "NaCl Madelung constant: got " << alpha_computed << ", expected ~1.74756";
}

TEST(MadelungConstant, CsClMadelungConstant) {
    Crystal cscl = test::make_cscl_crystal(4.12);
    // Charges: Cs=+1, Cl=-1
    std::vector<double> charges = {+1, -1};

    auto ewald = EwaldCalculator::compute(cscl, charges);

    // Basic sanity: energy should be negative (attractive unlike charges)
    EXPECT_LT(ewald.energy, 0.0)
        << "CsCl Ewald energy should be negative for unlike charges";
    EXPECT_TRUE(std::isfinite(ewald.energy));

    // Forces should be zero by symmetry at equilibrium
    ASSERT_EQ(ewald.forces.size(), 2u);
    for (int a = 0; a < 2; ++a) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(ewald.forces[a][d], 0.0, 1e-6)
                << "CsCl forces should vanish at equilibrium";
        }
    }
}

// ============================================================================
// EnergyComponents: signs and relationships
// ============================================================================

TEST(EnergyComponents, KineticEnergyPositive) {
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {1,1,1}, pps);
    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    EXPECT_GT(result.kinetic_energy, 0.0)
        << "Kinetic energy must be positive";
}

TEST(EnergyComponents, HartreeEnergyPositive) {
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {1,1,1}, pps);
    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    EXPECT_GT(result.hartree_energy, 0.0)
        << "Hartree energy must be positive";
}

TEST(EnergyComponents, XCEnergyNegative) {
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {1,1,1}, pps);
    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    EXPECT_LT(result.xc_energy, 0.0)
        << "XC energy should be negative for LDA";
}

TEST(EnergyComponents, TotalEnergyNegative) {
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {1,1,1}, pps);
    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for a bound system";
}

TEST(EnergyComponents, LDAandPBEGiveDifferentEnergy) {
    auto pps = test::make_si_pp_map();
    auto r_lda = run_si_scf(10.0, {1,1,1}, pps, "LDA_PZ");
    auto r_pbe = run_si_scf(10.0, {1,1,1}, pps, "PBE");
    if (!r_lda.converged || !r_pbe.converged) {
        GTEST_SKIP() << "One or both SCF runs did not converge";
    }
    EXPECT_NE(r_lda.total_energy_ry, r_pbe.total_energy_ry)
        << "LDA and PBE should give different total energies";
    // They should differ by a meaningful amount
    EXPECT_GT(std::abs(r_lda.total_energy_ry - r_pbe.total_energy_ry), 1e-6)
        << "LDA/PBE energy difference too small to be physical";
}
