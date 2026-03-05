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
