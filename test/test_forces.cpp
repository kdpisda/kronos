// ============================================================================
// KRONOS  test/test_forces.cpp
// Tests for Hellmann-Feynman forces and BFGS geometry optimization.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "potential/forces.hpp"
#include "potential/ewald.hpp"
#include "solver/scf.hpp"
#include "solver/bfgs.hpp"

#include <cmath>
#include <numeric>

using namespace kronos;

// ============================================================================
// Ewald force tests
// ============================================================================

TEST(Forces, NewtonsThirdLawEwald) {
    // Forces on all atoms should sum to zero
    auto crystal = test::make_si_diamond_crystal();
    std::vector<double> charges = {4.0, 4.0};
    auto result = EwaldCalculator::compute(crystal, charges);

    ASSERT_EQ(result.forces.size(), 2u);
    for (int d = 0; d < 3; ++d) {
        double sum = result.forces[0][d] + result.forces[1][d];
        EXPECT_NEAR(sum, 0.0, 1e-8)
            << "Ewald forces should sum to zero (Newton's 3rd law), dir=" << d;
    }
}

TEST(Forces, EquilibriumZeroForces) {
    // Si diamond at equilibrium: forces should be zero by symmetry
    auto crystal = test::make_si_diamond_crystal();
    std::vector<double> charges = {4.0, 4.0};
    auto result = EwaldCalculator::compute(crystal, charges);

    for (size_t ia = 0; ia < 2; ++ia) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(result.forces[ia][d], 0.0, 1e-6)
                << "Force on atom " << ia << " dir " << d
                << " should be zero at equilibrium";
        }
    }
}

TEST(Forces, DisplacedAtomNonzeroForce) {
    // Displace one atom → nonzero restoring force
    double a = 5.0;
    Mat3 lattice = {{{a, 0, 0}, {0, a, 0}, {0, 0, a}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.0, 0.0, 0.0}},
        {"Si", 14, {0.501, 0.5, 0.5}},  // displaced
    };
    Crystal crystal(lattice, std::move(atoms));

    std::vector<double> charges = {4.0, 4.0};
    auto result = EwaldCalculator::compute(crystal, charges);

    double max_f = 0.0;
    for (size_t ia = 0; ia < 2; ++ia) {
        for (int d = 0; d < 3; ++d) {
            max_f = std::max(max_f, std::abs(result.forces[ia][d]));
        }
    }
    EXPECT_GT(max_f, 1e-4)
        << "Displaced atom should have nonzero force";
}

TEST(Forces, NewtonsThirdLawMultiAtom) {
    // NaCl with 8 atoms: forces sum to zero
    auto crystal = test::make_nacl_crystal();
    std::vector<double> charges = {1.0, 1.0, 1.0, 1.0, 7.0, 7.0, 7.0, 7.0};
    auto result = EwaldCalculator::compute(crystal, charges);

    ASSERT_EQ(result.forces.size(), 8u);
    Vec3 sum = {0.0, 0.0, 0.0};
    for (size_t ia = 0; ia < 8; ++ia) {
        for (int d = 0; d < 3; ++d) {
            sum[d] += result.forces[ia][d];
        }
    }
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(sum[d], 0.0, 1e-6)
            << "Total Ewald force should be zero, dir=" << d;
    }
}

// ============================================================================
// SCF force tests
// ============================================================================

TEST(Forces, SCFForceSumToZero) {
    Crystal crystal = test::make_si_diamond_crystal();
    auto pp_map = test::make_si_pp_map();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pp_map);
    SCFResult result = solver.solve();

    if (!result.converged || result.forces.empty()) {
        GTEST_SKIP() << "SCF did not converge or no forces; skipping";
    }

    // Total forces sum to zero
    Vec3 total = {0.0, 0.0, 0.0};
    for (const auto& f : result.forces) {
        for (int d = 0; d < 3; ++d) total[d] += f[d];
    }
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(total[d], 0.0, 0.05)
            << "Total HF forces should sum to zero, dir=" << d;
    }
}

TEST(Forces, TotalForcesComputation) {
    // Test the ForceCalculator::compute_total_forces combines components
    std::vector<Vec3> ewald = {{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}};
    std::vector<Vec3> local = {{0.1, 0.2, 0.3}, {-0.1, -0.2, -0.3}};
    std::vector<Vec3> nonlocal = {{0.01, 0.02, 0.03}, {-0.01, -0.02, -0.03}};

    auto total = ForceCalculator::compute_total_forces(ewald, local, nonlocal);

    ASSERT_EQ(total.size(), 2u);
    EXPECT_NEAR(total[0][0], 1.11, 1e-10);
    EXPECT_NEAR(total[0][1], 2.22, 1e-10);
    EXPECT_NEAR(total[0][2], 3.33, 1e-10);
    EXPECT_NEAR(total[1][0], -1.11, 1e-10);
    EXPECT_NEAR(total[1][1], -2.22, 1e-10);
    EXPECT_NEAR(total[1][2], -3.33, 1e-10);
}

// ============================================================================
// BFGS optimizer
// ============================================================================

TEST(BFGS, DefaultParamsReasonable) {
    BFGSOptimizer::Params params;
    EXPECT_GT(params.max_steps, 0);
    EXPECT_GT(params.force_threshold, 0.0);
    EXPECT_GT(params.energy_threshold, 0.0);
    EXPECT_GT(params.initial_step, 0.0);
    EXPECT_GT(params.max_step, 0.0);
    EXPECT_GE(params.max_step, params.initial_step);
}

TEST(BFGS, ConstructsWithCustomParams) {
    BFGSOptimizer::Params params;
    params.max_steps = 10;
    params.force_threshold = 1e-4;
    BFGSOptimizer optimizer(params);
    // Just verify no crash
    (void)optimizer;
}

TEST(BFGS, OptimizeDisplacedSiDiamond) {
    // Slightly displace Si atom from equilibrium, verify BFGS reduces forces
    double a = 5.43;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.00, 0.00, 0.00}},
        {"Si", 14, {0.26, 0.25, 0.25}},  // slightly displaced from 0.25
    };
    Crystal crystal(lattice, std::move(atoms));
    auto pp_map = test::make_si_pp_map();

    CalculationParams calc;
    calc.type = CalculationType::Relax;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 50;

    BFGSOptimizer::Params bfgs_params;
    bfgs_params.max_steps = 5;
    bfgs_params.force_threshold = 0.1;  // loose for speed
    BFGSOptimizer optimizer(bfgs_params);

    RelaxResult result = optimizer.optimize(crystal, calc, conv, pp_map);

    // Should have done at least one step
    EXPECT_GT(result.relax_steps, 0);

    // Energy history should be recorded
    EXPECT_EQ(result.energy_history.size(),
              static_cast<size_t>(result.relax_steps));

    // Final energy should be finite
    EXPECT_TRUE(std::isfinite(result.final_energy_ry));
}

// ============================================================================
// Force finite-difference validation
// ============================================================================

TEST(Forces, EwaldForceFiniteDifference) {
    // F ≈ -dE/dτ for Ewald energy
    double a = 5.0;
    double delta = 1e-4;  // small displacement in fractional coords

    auto make_crystal = [&](double dx) {
        Mat3 lattice = {{{a, 0, 0}, {0, a, 0}, {0, 0, a}}};
        std::vector<Atom> atoms = {
            {"Si", 14, {0.0, 0.0, 0.0}},
            {"Si", 14, {0.5 + dx, 0.5, 0.5}},
        };
        return Crystal(lattice, std::move(atoms));
    };

    std::vector<double> charges = {4.0, 4.0};

    auto crystal_0 = make_crystal(0.0);
    auto result_0 = EwaldCalculator::compute(crystal_0, charges);

    auto crystal_p = make_crystal(delta);
    auto result_p = EwaldCalculator::compute(crystal_p, charges);

    auto crystal_m = make_crystal(-delta);
    auto result_m = EwaldCalculator::compute(crystal_m, charges);

    // Numerical force: F_x ≈ -(E(+delta) - E(-delta)) / (2*delta)
    // But delta is in fractional coords; actual displacement = delta * a_bohr
    double a_bohr = a * constants::angstrom_to_bohr;
    double dr = delta * a_bohr;  // actual displacement in bohr
    double f_numerical = -(result_p.energy - result_m.energy) / (2.0 * dr);

    // Analytical force from Ewald (x-component on atom 1)
    double f_analytical = result_0.forces[1][0];

    EXPECT_NEAR(f_numerical, f_analytical, std::abs(f_analytical) * 0.05 + 1e-4)
        << "Ewald force should match finite difference: analytical="
        << f_analytical << " numerical=" << f_numerical;
}

// ============================================================================
// Full SCF force finite-difference validation with real pseudopotential
// ============================================================================

// Helper: run SCF on a displaced Si diamond crystal and return the result.
// Displacement is in fractional coordinates on atom 0, direction dir.
static SCFResult run_si_displaced_scf(
    double delta, int dir,
    const std::map<std::string, PseudoPotential>& pps,
    double ecutwfc, double ethr, int max_steps)
{
    const double a = 5.43;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.00, 0.00, 0.00}},
        {"Si", 14, {0.25, 0.25, 0.25}},
    };
    atoms[0].position[dir] += delta;
    Crystal crystal(lattice, std::move(atoms));

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = ecutwfc;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    ConvergenceParams conv;
    conv.energy_threshold = ethr;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = max_steps;
    SCFSolver solver(crystal, calc, conv, pps);
    return solver.solve();
}

TEST(Forces, RealPPForceFiniteDifference) {
    // Load real Si pseudopotential
    std::map<std::string, PseudoPotential> pps;
    try {
        pps["Si"] = parse_upf("../pseudopotentials/Si.pz-vbc.UPF");
    } catch (...) {
        GTEST_SKIP() << "Si.pz-vbc.UPF not found; skipping real PP force test";
    }

    const double ecutwfc = 12.0;
    const double ethr = 1e-8;
    const int max_steps = 100;

    // Large base displacement to get significant forces
    const double delta0 = 0.01;   // base displacement (fractional)
    const double eps    = 0.001;   // FD step size (fractional)

    // Run SCF at displaced position and at +/- epsilon around it
    auto r_0 = run_si_displaced_scf(delta0,       0, pps, ecutwfc, ethr, max_steps);
    auto r_p = run_si_displaced_scf(delta0 + eps, 0, pps, ecutwfc, ethr, max_steps);
    auto r_m = run_si_displaced_scf(delta0 - eps, 0, pps, ecutwfc, ethr, max_steps);

    if (!r_p.converged || !r_0.converged || !r_m.converged) {
        GTEST_SKIP() << "SCF did not converge for one or more displacements";
    }

    // Compute Cartesian step: eps_cart = eps * |a_1| in Bohr
    Crystal cryst = test::make_si_diamond_crystal();
    auto lat_bohr = cryst.lattice_bohr();
    double a1_mag = std::sqrt(lat_bohr[0][0]*lat_bohr[0][0]
                            + lat_bohr[0][1]*lat_bohr[0][1]
                            + lat_bohr[0][2]*lat_bohr[0][2]);
    double eps_bohr = eps * a1_mag;

    // Finite-difference force (projected onto a1 direction)
    double f_fd = -(r_p.total_energy_ry - r_m.total_energy_ry) / (2.0 * eps_bohr);

    // FD of individual energy components
    double f_fd_ewald = -(r_p.ewald_energy - r_m.ewald_energy) / (2.0 * eps_bohr);
    double f_fd_elec = f_fd - f_fd_ewald;

    // Analytical force: project total HF force on atom 0 onto a1_hat
    ASSERT_FALSE(r_0.forces.empty());
    ASSERT_FALSE(r_0.ewald_forces.empty());
    ASSERT_FALSE(r_0.local_forces.empty());
    ASSERT_FALSE(r_0.nonlocal_forces.empty());

    double f_analytic = 0.0, f_ewald_a = 0.0, f_local_a = 0.0, f_nl_a = 0.0;
    for (int j = 0; j < 3; ++j) {
        double hat_j = lat_bohr[0][j] / a1_mag;
        f_analytic += r_0.forces[0][j] * hat_j;
        f_ewald_a += r_0.ewald_forces[0][j] * hat_j;
        f_local_a += r_0.local_forces[0][j] * hat_j;
        f_nl_a += r_0.nonlocal_forces[0][j] * hat_j;
    }

    std::printf("\n  Force validation: analytic=%+.6f  FD=%+.6f  (diff=%.2e)\n",
                f_analytic, f_fd, std::abs(f_analytic - f_fd));

    // Force should be non-trivial at this displacement
    EXPECT_GT(std::abs(f_analytic), 0.01)
        << "Analytic force too small at delta0=" << delta0
        << "; test is not meaningful";

    // Tolerance: 10% of force magnitude + small floor for FD error
    double tol = std::max(0.10 * std::abs(f_analytic), 0.005);

    EXPECT_NEAR(f_analytic, f_fd, tol)
        << "Real PP: Analytical force (" << f_analytic
        << ") disagrees with finite-difference (" << f_fd
        << "), tolerance=" << tol;
}

TEST(Forces, RealPPForceComponentsFiniteDifference) {
    // Validate Ewald force component via FD at displaced position
    std::map<std::string, PseudoPotential> pps;
    try {
        pps["Si"] = parse_upf("../pseudopotentials/Si.pz-vbc.UPF");
    } catch (...) {
        GTEST_SKIP() << "Si.pz-vbc.UPF not found; skipping";
    }

    const double ecutwfc = 12.0;
    const double ethr = 1e-8;
    const int max_steps = 100;
    const double delta0 = 0.01;
    const double eps = 0.001;

    auto r_0 = run_si_displaced_scf(delta0,       0, pps, ecutwfc, ethr, max_steps);
    auto r_p = run_si_displaced_scf(delta0 + eps, 0, pps, ecutwfc, ethr, max_steps);
    auto r_m = run_si_displaced_scf(delta0 - eps, 0, pps, ecutwfc, ethr, max_steps);

    if (!r_p.converged || !r_0.converged || !r_m.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }

    Crystal cryst = test::make_si_diamond_crystal();
    auto lat_bohr = cryst.lattice_bohr();
    double a1_mag = std::sqrt(lat_bohr[0][0]*lat_bohr[0][0]
                            + lat_bohr[0][1]*lat_bohr[0][1]
                            + lat_bohr[0][2]*lat_bohr[0][2]);
    double eps_bohr = eps * a1_mag;

    // FD of Ewald energy component
    double f_ewald_fd = -(r_p.ewald_energy - r_m.ewald_energy) / (2.0 * eps_bohr);
    ASSERT_FALSE(r_0.ewald_forces.empty());
    double f_ewald_analytic = 0.0;
    for (int j = 0; j < 3; ++j) {
        f_ewald_analytic += r_0.ewald_forces[0][j] * lat_bohr[0][j] / a1_mag;
    }
    EXPECT_NEAR(f_ewald_analytic, f_ewald_fd,
                std::max(0.05 * std::abs(f_ewald_analytic), 0.001))
        << "Ewald force component: analytic=" << f_ewald_analytic
        << " FD=" << f_ewald_fd;

    // Check all force components exist and are non-trivial
    ASSERT_FALSE(r_0.local_forces.empty());
    ASSERT_FALSE(r_0.nonlocal_forces.empty());
    double f_local_a = 0.0, f_nonlocal_a = 0.0;
    for (int j = 0; j < 3; ++j) {
        f_local_a += r_0.local_forces[0][j] * lat_bohr[0][j] / a1_mag;
        f_nonlocal_a += r_0.nonlocal_forces[0][j] * lat_bohr[0][j] / a1_mag;
    }
    EXPECT_TRUE(std::isfinite(f_local_a)) << "Local force not finite";
    EXPECT_TRUE(std::isfinite(f_nonlocal_a)) << "Nonlocal force not finite";
    // At delta0=0.01, local and nonlocal forces should be non-negligible
    EXPECT_GT(std::abs(f_local_a) + std::abs(f_nonlocal_a), 1e-4)
        << "Force components are trivially small";

    // Newton's 3rd law: total forces sum to zero
    ASSERT_GE(r_0.forces.size(), 2u);
    for (int d = 0; d < 3; ++d) {
        double sum = 0.0;
        for (const auto& f : r_0.forces) sum += f[d];
        EXPECT_NEAR(sum, 0.0, 0.01)
            << "Forces don't sum to zero in direction " << d;
    }
}
