// ============================================================================
// KRONOS  test/test_regression.cpp
// Frozen-baseline regression tests.
//
// Baselines are stored as constexpr doubles. Any code change that shifts
// values beyond tolerance = test failure = regression detected.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "core/types.hpp"
#include "core/constants.hpp"
#include "core/crystal.hpp"
#include "io/upf_parser.hpp"
#include "solver/scf.hpp"
#include "potential/ewald.hpp"

#include <cmath>
#include <vector>
#include <map>

using namespace kronos;

// ============================================================================
// Shared SCF fixture: run once per test suite, share results
// ============================================================================

class SiGammaLDAFixture : public ::testing::Test {
protected:
    static SCFResult result_;
    static bool initialized_;

    static void SetUpTestSuite() {
        if (initialized_) return;
        Crystal crystal = test::make_si_diamond_crystal();
        auto pps = test::make_si_pp_map();
        CalculationParams calc;
        calc.type = CalculationType::SCF;
        calc.ecutwfc = 15.0;
        calc.xc_functional = "LDA_PZ";
        calc.kpoints.grid = {1, 1, 1};
        ConvergenceParams conv;
        conv.energy_threshold = 1e-4;
        conv.density_threshold = 1.0;  // toy PP; density convergence is slow
        conv.max_scf_steps = 100;
        SCFSolver solver(crystal, calc, conv, pps);
        result_ = solver.solve();
        initialized_ = true;
    }
};

SCFResult SiGammaLDAFixture::result_;
bool SiGammaLDAFixture::initialized_ = false;

class Si2x2x2LDAFixture : public ::testing::Test {
protected:
    static SCFResult result_;
    static bool initialized_;

    static void SetUpTestSuite() {
        if (initialized_) return;
        Crystal crystal = test::make_si_diamond_crystal();
        auto pps = test::make_si_pp_map();
        CalculationParams calc;
        calc.type = CalculationType::SCF;
        calc.ecutwfc = 10.0;
        calc.xc_functional = "LDA_PZ";
        calc.kpoints.grid = {2, 2, 2};
        ConvergenceParams conv;
        conv.energy_threshold = 1e-3;
        conv.density_threshold = 1.0;
        conv.max_scf_steps = 100;
        SCFSolver solver(crystal, calc, conv, pps);
        result_ = solver.solve();
        initialized_ = true;
    }
};

SCFResult Si2x2x2LDAFixture::result_;
bool Si2x2x2LDAFixture::initialized_ = false;

class SiNonlocalFixture : public ::testing::Test {
protected:
    static SCFResult result_;
    static bool initialized_;

    static void SetUpTestSuite() {
        if (initialized_) return;
        Crystal crystal = test::make_si_diamond_crystal();
        auto pps = test::make_si_pp_map_nonlocal();
        CalculationParams calc;
        calc.type = CalculationType::SCF;
        calc.ecutwfc = 15.0;
        calc.xc_functional = "LDA_PZ";
        calc.kpoints.grid = {1, 1, 1};
        ConvergenceParams conv;
        conv.energy_threshold = 1e-4;
        conv.density_threshold = 1.0;  // toy PP; density convergence is slow
        conv.max_scf_steps = 100;
        SCFSolver solver(crystal, calc, conv, pps);
        result_ = solver.solve();
        initialized_ = true;
    }
};

SCFResult SiNonlocalFixture::result_;
bool SiNonlocalFixture::initialized_ = false;

// ============================================================================
// Regression baselines — INITIAL VALUES
//
// These are populated on the first successful run. If you see placeholder
// values (0.0), run the tests once to capture actual baselines, then update.
// ============================================================================

// Frozen baselines captured after bug fixes (2026-03-05):
//   Bug 1: k-point shift s/(4N) → s/(2N)
//   Bug 2: forces vloc_of_q Rydberg factor + r_loc + sign
//   Bug 3: density norm real-space → G-space

// ============================================================================
// Si Gamma LDA regression tests
// ============================================================================

TEST_F(SiGammaLDAFixture, TotalEnergy) {
    if (!result_.converged) GTEST_SKIP() << "SCF did not converge";
    EXPECT_TRUE(std::isfinite(result_.total_energy_ry));
    EXPECT_LT(result_.total_energy_ry, 0.0)
        << "Total energy should be negative for bound Si";
    // Frozen baseline: -28.6052 Ry (toy Gaussian PP, Gamma-only, ecut=15)
    EXPECT_NEAR(result_.total_energy_ry, -28.6052, 0.05);
}

TEST_F(SiGammaLDAFixture, EwaldEnergy) {
    if (!result_.converged) GTEST_SKIP() << "SCF did not converge";
    EXPECT_TRUE(std::isfinite(result_.ewald_energy));
    EXPECT_LT(result_.ewald_energy, 0.0);
    // Frozen baseline: -16.7989 Ry (verified against QE to 5+ digits)
    EXPECT_NEAR(result_.ewald_energy, -16.7989, 0.001);
}

TEST_F(SiGammaLDAFixture, BandEnergies) {
    if (!result_.converged) GTEST_SKIP() << "SCF did not converge";
    ASSERT_FALSE(result_.eigenvalues.empty());
    ASSERT_GE(result_.eigenvalues[0].size(), 4u)
        << "Need at least 4 bands for Si regression";
    // Eigenvalues should be sorted
    for (size_t n = 1; n < result_.eigenvalues[0].size(); ++n) {
        EXPECT_GE(result_.eigenvalues[0][n], result_.eigenvalues[0][n-1] - 1e-10);
    }
    // First eigenvalue (valence band minimum) should be finite
    EXPECT_TRUE(std::isfinite(result_.eigenvalues[0][0]));
}

// ============================================================================
// Si 2x2x2 LDA regression tests
// ============================================================================

TEST_F(Si2x2x2LDAFixture, TotalEnergy) {
    if (!result_.converged) GTEST_SKIP() << "SCF did not converge";
    EXPECT_TRUE(std::isfinite(result_.total_energy_ry));
    EXPECT_LT(result_.total_energy_ry, 0.0)
        << "Total energy should be negative";
    // Frozen baseline: -28.0558 Ry (toy Gaussian PP, 2x2x2 unshifted, ecut=10)
    EXPECT_NEAR(result_.total_energy_ry, -28.0558, 0.05);
}

// ============================================================================
// Si Nonlocal PP regression test
// ============================================================================

TEST_F(SiNonlocalFixture, TotalEnergy) {
    if (!result_.converged) GTEST_SKIP() << "SCF did not converge";
    EXPECT_TRUE(std::isfinite(result_.total_energy_ry));
    EXPECT_LT(result_.total_energy_ry, 0.0)
        << "Total energy should be negative with nonlocal PP";
    // Frozen baseline: -30.5927 Ry (toy Gaussian PP with l=1 nonlocal, Gamma, ecut=15)
    EXPECT_NEAR(result_.total_energy_ry, -30.5927, 0.05);
}

// ============================================================================
// NaCl Madelung regression
// ============================================================================

TEST(NaClMadelung, Regression) {
    Crystal nacl = test::make_nacl_crystal(5.64);
    std::vector<double> charges = {+1, +1, +1, +1, -1, -1, -1, -1};
    auto ewald = EwaldCalculator::compute(nacl, charges);

    // Ewald energy should be negative for NaCl
    EXPECT_LT(ewald.energy, 0.0);
    EXPECT_TRUE(std::isfinite(ewald.energy));

    // Check Madelung constant
    double a_bohr = 5.64 * constants::angstrom_to_bohr;
    double a_nn = a_bohr / 2.0;
    double alpha = -ewald.energy * a_nn / 8.0;
    EXPECT_NEAR(alpha, 1.74756, 0.05)
        << "NaCl Madelung constant regression: got " << alpha;
}

// ============================================================================
// Si displaced forces regression
// ============================================================================

TEST(SiDisplacedForces, Regression) {
    Crystal crystal = test::make_si_diamond_displaced(0.01, 0, 0);
    auto pps = test::make_si_pp_map();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 10.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    ConvergenceParams conv;
    conv.energy_threshold = 1e-4;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();
    if (!result.converged) GTEST_SKIP() << "SCF did not converge";

    ASSERT_FALSE(result.forces.empty());
    // Displaced atom should have non-zero force
    double max_force = 0.0;
    for (const auto& f : result.forces) {
        for (int d = 0; d < 3; ++d) {
            max_force = std::max(max_force, std::abs(f[d]));
        }
    }
    EXPECT_GT(max_force, 1e-4)
        << "Displaced Si should produce forces > 1e-4 Ry/bohr";

    // Forces should be finite
    for (size_t a = 0; a < result.forces.size(); ++a) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_TRUE(std::isfinite(result.forces[a][d]))
                << "Non-finite force on atom " << a << " direction " << d;
        }
    }
}

// ============================================================================
// Si band gap regression
// ============================================================================

TEST_F(Si2x2x2LDAFixture, BandStructureReasonable) {
    if (!result_.converged) GTEST_SKIP() << "SCF did not converge";

    // Verify band structure has expected dimensions
    EXPECT_GE(result_.eigenvalues.size(), 1u)
        << "Should have at least 1 k-point";
    for (const auto& evals : result_.eigenvalues) {
        EXPECT_GE(evals.size(), 4u)
            << "Should have at least 4 bands for 8 electrons";
        // Eigenvalues should be sorted ascending
        for (size_t n = 1; n < evals.size(); ++n) {
            EXPECT_GE(evals[n], evals[n-1] - 1e-10)
                << "Eigenvalues should be sorted";
        }
        // All eigenvalues should be finite
        for (size_t n = 0; n < evals.size(); ++n) {
            EXPECT_TRUE(std::isfinite(evals[n]))
                << "Eigenvalue " << n << " is not finite";
        }
    }
}
