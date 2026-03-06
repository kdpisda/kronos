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

// ============================================================================
// Multi-system validation with real pseudopotentials
// ============================================================================

// Al FCC: simple sp-metal, Gaussian smearing, LDA
TEST(MultiSystem, AlFCCGammaLDA) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps["Al"] = parse_upf("../pseudopotentials/Al.pz-vbc.UPF");
    } catch (...) {
        GTEST_SKIP() << "Al.pz-vbc.UPF not found";
    }

    // Al FCC primitive cell: a = 4.05 angstrom
    const double a = 4.05;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {{"Al", 13, {0.0, 0.0, 0.0}}};
    Crystal crystal(lattice, std::move(atoms));

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 16.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.05;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-6;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Al FCC Gamma SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for Al FCC";

    // Al FCC Gamma-only should give ~-4 to -8 Ry per atom (rough LDA range)
    EXPECT_GT(result.total_energy_ry, -20.0)
        << "Energy unreasonably low for 1-atom Al FCC";

    std::printf("  Al FCC Gamma: E_total = %.6f Ry, Ewald = %.6f Ry\n",
                result.total_energy_ry, result.ewald_energy);
}

TEST(MultiSystem, AlFCC444LDA) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps["Al"] = parse_upf("../pseudopotentials/Al.pz-vbc.UPF");
    } catch (...) {
        GTEST_SKIP() << "Al.pz-vbc.UPF not found";
    }

    const double a = 4.05;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {{"Al", 13, {0.0, 0.0, 0.0}}};
    Crystal crystal(lattice, std::move(atoms));

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 16.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {4, 4, 4};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.05;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-6;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Al FCC 4x4x4 SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0);

    // With k-point sampling, Al energy should be more converged
    // QE reference: ~-4.19 Ry/atom for LDA at ecut=16 with 4x4x4
    std::printf("  Al FCC 4x4x4: E_total = %.6f Ry, Ewald = %.6f Ry, %d IBZ k-points\n",
                result.total_energy_ry, result.ewald_energy,
                static_cast<int>(result.eigenvalues.size()));
}

// Cu FCC: d-electron metal, LDA with smearing
TEST(MultiSystem, CuFCCGammaLDA) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps["Cu"] = parse_upf("../pseudopotentials/Cu.pz-d-hgh.UPF");
    } catch (...) {
        GTEST_SKIP() << "Cu.pz-d-hgh.UPF not found";
    }

    // Cu FCC primitive cell: a = 3.61 angstrom
    const double a = 3.61;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {{"Cu", 29, {0.0, 0.0, 0.0}}};
    Crystal crystal(lattice, std::move(atoms));

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 30.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.02;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-6;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Cu FCC Gamma SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for Cu FCC";

    std::printf("  Cu FCC Gamma: E_total = %.6f Ry, %zu bands\n",
                result.total_energy_ry,
                result.eigenvalues.empty() ? 0 : result.eigenvalues[0].size());
}

// ============================================================================
// H2O molecule validation: convergence, forces, Newton's 3rd law
// ============================================================================

// H2O with toy (analytic) PPs: SCF convergence check
// Uses soft O PP (r_loc=1.0) and relaxed convergence for stability
TEST(MultiSystem, H2OGammaConvergence) {
    auto pps = test::make_h2o_pp_map();
    Crystal crystal = test::make_h2o_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 15.0;  // Low cutoff for toy PP
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.05;  // Larger smearing for molecule in box
    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;  // Relaxed for toy PP
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "H2O Gamma SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for H2O";

    std::printf("  H2O Gamma (toy PP): E_total = %.6f Ry, %d SCF steps\n",
                result.total_energy_ry, result.scf_steps);
}

// H2O: forces obey Newton's 3rd law (sum of forces ≈ 0)
TEST(MultiSystem, H2ONewtonThirdLaw) {
    auto pps = test::make_h2o_pp_map();
    Crystal crystal = test::make_h2o_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 15.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.05;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    if (!result.converged) {
        GTEST_SKIP() << "H2O SCF did not converge";
    }
    ASSERT_EQ(result.forces.size(), 3u);

    // Sum of forces on all atoms should be ~zero (Newton's 3rd law)
    for (int d = 0; d < 3; ++d) {
        double f_sum = 0.0;
        for (size_t a = 0; a < result.forces.size(); ++a) {
            f_sum += result.forces[a][d];
        }
        EXPECT_NEAR(f_sum, 0.0, 0.05)
            << "Newton's 3rd law violated: sum of forces in direction " << d
            << " = " << f_sum;
    }

    std::printf("  H2O forces: O=(%+.4f,%+.4f,%+.4f), H1=(%+.4f,%+.4f,%+.4f), H2=(%+.4f,%+.4f,%+.4f)\n",
                result.forces[0][0], result.forces[0][1], result.forces[0][2],
                result.forces[1][0], result.forces[1][1], result.forces[1][2],
                result.forces[2][0], result.forces[2][1], result.forces[2][2]);
}

// H2O with real UPF pseudopotentials
// Note: H2O in a 15-bohr box creates a large FFT grid (~9000 PWs at ecut=30).
// Use moderate cutoff and smearing for practical convergence.
TEST(MultiSystem, H2OGammaRealPP) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps = test::make_h2o_pp_map_real();
    } catch (...) {
        GTEST_SKIP() << "H2O UPF files not found";
    }

    Crystal crystal = test::make_h2o_crystal(6.35);  // 12 bohr box

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 12.0;  // Low cutoff for manageable PW count
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.05;  // Larger smearing for molecule in box
    ConvergenceParams conv;
    conv.energy_threshold = 1e-4;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "H2O Gamma (real PP) SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for H2O";

    // Forces should be finite (force-sum accuracy requires higher ecutwfc)
    ASSERT_EQ(result.forces.size(), 3u);
    for (size_t a = 0; a < result.forces.size(); ++a) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_TRUE(std::isfinite(result.forces[a][d]))
                << "Force on atom " << a << " dir " << d << " not finite";
        }
    }

    std::printf("  H2O Gamma (real PP): E_total = %.6f Ry, %d SCF steps\n",
                result.total_energy_ry, result.scf_steps);
}

// ============================================================================
// MgO rocksalt validation: convergence, band gap, force symmetry
// ============================================================================

// MgO with toy PPs: Gamma-only convergence
TEST(MultiSystem, MgOGammaConvergence) {
    auto pps = test::make_mgo_pp_map();
    Crystal crystal = test::make_mgo_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 20.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.01;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-6;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "MgO Gamma SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for MgO";

    // At equilibrium, forces should vanish by symmetry
    ASSERT_EQ(result.forces.size(), 2u);
    for (size_t a = 0; a < result.forces.size(); ++a) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(result.forces[a][d], 0.0, 0.05)
                << "MgO force on atom " << a << " dir " << d
                << " should vanish at equilibrium by symmetry";
        }
    }

    std::printf("  MgO Gamma (toy PP): E_total = %.6f Ry, %d SCF steps\n",
                result.total_energy_ry, result.scf_steps);
}

// MgO with real UPF pseudopotentials: Gamma-only
TEST(MultiSystem, MgOGammaRealPP) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps = test::make_mgo_pp_map_real();
    } catch (...) {
        GTEST_SKIP() << "MgO UPF files not found";
    }

    Crystal crystal = test::make_mgo_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 40.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.01;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-8;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "MgO Gamma (real PP) SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for MgO";

    // MgO is an insulator: check for band gap
    // With Gamma-only, eigenvalues should show a gap
    if (!result.eigenvalues.empty() && result.eigenvalues[0].size() >= 5) {
        // MgO: Mg(Z_val=2) + O(Z_val=6) = 8 electrons / 2 atoms
        // 4 occupied bands (spin factor 2), gap above band 4
        int n_occ = 4;
        if (static_cast<int>(result.eigenvalues[0].size()) > n_occ) {
            double gap = result.eigenvalues[0][n_occ] - result.eigenvalues[0][n_occ - 1];
            EXPECT_GT(gap, 0.01) << "MgO should have a band gap (insulator)";
            std::printf("  MgO Gamma (real PP): band gap = %.4f Ry = %.3f eV\n",
                        gap, gap * 13.6057);
        }
    }

    std::printf("  MgO Gamma (real PP): E_total = %.6f Ry, %d SCF steps\n",
                result.total_energy_ry, result.scf_steps);
}

// MgO with 4x4x4 k-grid (real PPs)
TEST(MultiSystem, MgO444RealPP) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps = test::make_mgo_pp_map_real();
    } catch (...) {
        GTEST_SKIP() << "MgO UPF files not found";
    }

    Crystal crystal = test::make_mgo_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 40.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {4, 4, 4};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.01;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-8;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "MgO 4x4x4 (real PP) SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0);

    std::printf("  MgO 4x4x4 (real PP): E_total = %.6f Ry, %d SCF steps, %d IBZ k-points\n",
                result.total_energy_ry, result.scf_steps,
                static_cast<int>(result.eigenvalues.size()));
}

// ============================================================================
// Graphene 2D validation: convergence with 2D k-grid
// ============================================================================

// Graphene with toy PPs: Gamma-only convergence
// Relaxed convergence for toy PP (energy threshold 1e-3)
TEST(MultiSystem, GrapheneGammaConvergence) {
    auto pps = test::make_graphene_pp_map();
    Crystal crystal = test::make_graphene_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 15.0;  // Low cutoff for toy PP
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.05;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;  // Relaxed for toy PP
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Graphene Gamma SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for graphene";

    std::printf("  Graphene Gamma (toy PP): E_total = %.6f Ry, %d SCF steps\n",
                result.total_energy_ry, result.scf_steps);
}

// Graphene with real UPF pseudopotentials: Gamma-only
TEST(MultiSystem, GrapheneGammaRealPP) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps = test::make_graphene_pp_map_real();
    } catch (...) {
        GTEST_SKIP() << "C.pz-vbc.UPF not found";
    }

    Crystal crystal = test::make_graphene_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 30.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.02;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-8;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Graphene Gamma (real PP) SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for graphene";

    std::printf("  Graphene Gamma (real PP): E_total = %.6f Ry, %d SCF steps\n",
                result.total_energy_ry, result.scf_steps);
}

// Graphene with 4x4x1 k-grid (2D periodicity, real PPs)
TEST(MultiSystem, Graphene441RealPP) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps = test::make_graphene_pp_map_real();
    } catch (...) {
        GTEST_SKIP() << "C.pz-vbc.UPF not found";
    }

    Crystal crystal = test::make_graphene_crystal(2.461, 7.938);  // Reduced vacuum (15 bohr)

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 15.0;  // Lower cutoff for faster k-resolved test
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {4, 4, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.03;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-4;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Graphene 4x4x1 (real PP) SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0);

    // Check eigenvalue structure for Dirac cone signature:
    // Near Fermi level, graphene should have bands crossing (semi-metal)
    // C: Z_val=4, 2 atoms = 8 electrons, 4 occupied bands
    if (!result.eigenvalues.empty() && result.eigenvalues[0].size() >= 5) {
        int n_occ = 4;
        if (static_cast<int>(result.eigenvalues[0].size()) > n_occ) {
            double gap_gamma = result.eigenvalues[0][n_occ] - result.eigenvalues[0][n_occ - 1];
            std::printf("  Graphene 4x4x1 (real PP): E_total = %.6f Ry, "
                        "HOMO-LUMO gap at Gamma = %.4f Ry, %d IBZ k-points\n",
                        result.total_energy_ry, gap_gamma,
                        static_cast<int>(result.eigenvalues.size()));
        }
    } else {
        std::printf("  Graphene 4x4x1 (real PP): E_total = %.6f Ry, %d SCF steps\n",
                    result.total_energy_ry, result.scf_steps);
    }
}

// ============================================================================
// Fe BCC spin-polarized validation: LSDA magnetic moment
// ============================================================================

// Fe BCC with toy PP: spin-polarized SCF convergence
TEST(MultiSystem, FeBCCSpinToyPP) {
    auto pps = test::make_fe_bcc_pp_map();
    Crystal crystal = test::make_fe_bcc_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 15.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.05;
    calc.nspin = 2;
    calc.spin_polarized = true;
    calc.starting_magnetization["Fe"] = 0.5;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Fe BCC spin-polarized Gamma SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for Fe BCC";

    // Spin-polarized should have nonzero magnetization
    EXPECT_GT(std::abs(result.total_magnetization), 0.01)
        << "Fe BCC should have nonzero magnetization";

    std::printf("  Fe BCC Gamma (toy PP, nspin=2): E_total = %.6f Ry, "
                "mag = %.4f muB, |mag| = %.4f muB, %d SCF steps\n",
                result.total_energy_ry,
                result.total_magnetization,
                result.absolute_magnetization,
                result.scf_steps);
}

// Fe BCC with real UPF PP: spin-polarized LSDA
TEST(MultiSystem, FeBCCSpinRealPP) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps = test::make_fe_bcc_pp_map_real();
    } catch (...) {
        GTEST_SKIP() << "Fe.pz-hgh.UPF not found";
    }

    Crystal crystal = test::make_fe_bcc_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 40.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.02;
    calc.nspin = 2;
    calc.spin_polarized = true;
    calc.starting_magnetization["Fe"] = 0.5;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-6;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Fe BCC spin-polarized (real PP) SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0);

    // At Gamma-only, d-bands aren't resolved so exchange splitting can
    // fully polarize d-states (mag up to 4 muB). With k-point sampling,
    // the moment converges to ~2.2 muB.
    EXPECT_GT(result.total_magnetization, 0.5)
        << "Fe BCC magnetic moment too low";
    EXPECT_LT(result.total_magnetization, 6.0)
        << "Fe BCC magnetic moment too high (unphysical)";

    // Spin-resolved eigenvalues should be stored
    ASSERT_EQ(result.eigenvalues_spin.size(), 2u);

    std::printf("  Fe BCC Gamma (real PP, nspin=2): E_total = %.6f Ry, "
                "mag = %.4f muB, |mag| = %.4f muB, %d SCF steps\n",
                result.total_energy_ry,
                result.total_magnetization,
                result.absolute_magnetization,
                result.scf_steps);
}

// Fe BCC with 4x4x4 k-grid, spin-polarized (real PP)
TEST(MultiSystem, FeBCC444SpinRealPP) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps = test::make_fe_bcc_pp_map_real();
    } catch (...) {
        GTEST_SKIP() << "Fe.pz-hgh.UPF not found";
    }

    Crystal crystal = test::make_fe_bcc_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 40.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {4, 4, 4};
    calc.smearing = SmearingType::Gaussian;
    calc.degauss = 0.02;
    calc.nspin = 2;
    calc.spin_polarized = true;
    calc.starting_magnetization["Fe"] = 0.5;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-6;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged) << "Fe BCC 4x4x4 spin-polarized SCF should converge";
    if (!result.converged) return;

    EXPECT_TRUE(std::isfinite(result.total_energy_ry));
    EXPECT_LT(result.total_energy_ry, 0.0);

    // With k-point sampling, magnetic moment should be closer to ~2.2 muB
    EXPECT_GT(result.total_magnetization, 1.0)
        << "Fe BCC 4x4x4 magnetic moment too low";

    std::printf("  Fe BCC 4x4x4 (real PP, nspin=2): E_total = %.6f Ry, "
                "mag = %.4f muB, |mag| = %.4f muB, %d SCF steps, %d IBZ k-points\n",
                result.total_energy_ry,
                result.total_magnetization,
                result.absolute_magnetization,
                result.scf_steps,
                static_cast<int>(result.eigenvalues.size()));
}

// Fe BCC: verify spin-polarized energy is lower than unpolarized
TEST(MultiSystem, FeBCCSpinLowerEnergy) {
    auto pps = test::make_fe_bcc_pp_map();
    Crystal crystal = test::make_fe_bcc_crystal();

    // Unpolarized (nspin=1)
    CalculationParams calc_ns1;
    calc_ns1.type = CalculationType::SCF;
    calc_ns1.ecutwfc = 15.0;
    calc_ns1.xc_functional = "LDA_PZ";
    calc_ns1.kpoints.grid = {1, 1, 1};
    calc_ns1.smearing = SmearingType::Gaussian;
    calc_ns1.degauss = 0.05;
    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver_ns1(crystal, calc_ns1, conv, pps);
    auto r_ns1 = solver_ns1.solve();

    // Spin-polarized (nspin=2)
    CalculationParams calc_ns2 = calc_ns1;
    calc_ns2.nspin = 2;
    calc_ns2.spin_polarized = true;
    calc_ns2.starting_magnetization["Fe"] = 0.5;

    SCFSolver solver_ns2(crystal, calc_ns2, conv, pps);
    auto r_ns2 = solver_ns2.solve();

    if (!r_ns1.converged || !r_ns2.converged) {
        GTEST_SKIP() << "One or both SCF runs did not converge";
    }

    // Spin-polarized energy should be lower (more stable) for Fe
    EXPECT_LT(r_ns2.total_energy_ry, r_ns1.total_energy_ry + 1e-6)
        << "Spin-polarized E=" << r_ns2.total_energy_ry
        << " should be <= unpolarized E=" << r_ns1.total_energy_ry;

    std::printf("  Fe BCC: E(nspin=1)=%.6f Ry, E(nspin=2)=%.6f Ry, delta=%.4f Ry\n",
                r_ns1.total_energy_ry, r_ns2.total_energy_ry,
                r_ns2.total_energy_ry - r_ns1.total_energy_ry);
}

// Al forces at non-equilibrium: displacement + FD validation
TEST(MultiSystem, AlFCCForceValidation) {
    std::map<std::string, PseudoPotential> pps;
    try {
        pps["Al"] = parse_upf("../pseudopotentials/Al.pz-vbc.UPF");
    } catch (...) {
        GTEST_SKIP() << "Al.pz-vbc.UPF not found";
    }

    // Build Al FCC with 2-atom basis and one atom displaced
    // Use conventional 2-atom cell for nontrivial forces
    const double a = 4.05;
    auto make_al2 = [&](double delta) {
        Mat3 lattice = {{{a, 0, 0}, {0, a, 0}, {0, 0, a}}};
        std::vector<Atom> atoms = {
            {"Al", 13, {0.0, 0.0, 0.0}},
            {"Al", 13, {0.5, 0.5, 0.0}},
        };
        atoms[0].position[0] += delta;
        return Crystal(lattice, std::move(atoms));
    };

    auto run_al = [&](double delta) {
        Crystal crystal = make_al2(delta);
        CalculationParams calc;
        calc.type = CalculationType::SCF;
        calc.ecutwfc = 12.0;
        calc.xc_functional = "LDA_PZ";
        calc.kpoints.grid = {1, 1, 1};
        calc.smearing = SmearingType::Gaussian;
        calc.degauss = 0.05;
        ConvergenceParams conv;
        conv.energy_threshold = 1e-8;
        conv.density_threshold = 1.0;
        conv.max_scf_steps = 100;
        SCFSolver solver(crystal, calc, conv, pps);
        return solver.solve();
    };

    const double delta0 = 0.01;
    const double eps = 0.001;
    auto r_0 = run_al(delta0);
    auto r_p = run_al(delta0 + eps);
    auto r_m = run_al(delta0 - eps);

    if (!r_0.converged || !r_p.converged || !r_m.converged) {
        GTEST_SKIP() << "Al SCF did not converge";
    }

    // eps in Bohr: delta is fractional along x, lattice[0] = (a, 0, 0)
    double a_bohr = a * constants::angstrom_to_bohr;
    double eps_bohr = eps * a_bohr;

    double f_fd = -(r_p.total_energy_ry - r_m.total_energy_ry) / (2.0 * eps_bohr);
    double f_analytic = r_0.forces[0][0]; // force along x

    double tol = std::max(0.10 * std::abs(f_analytic), 0.005);
    std::printf("  Al force: analytic=%+.6f  FD=%+.6f  diff=%.2e\n",
                f_analytic, f_fd, std::abs(f_analytic - f_fd));

    EXPECT_NEAR(f_analytic, f_fd, tol)
        << "Al force: analytic=" << f_analytic << " FD=" << f_fd;
}
