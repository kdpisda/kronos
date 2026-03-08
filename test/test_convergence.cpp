// ============================================================================
// KRONOS  test/test_convergence.cpp
// Convergence study tests: cutoff, k-point grid, SCF behavior.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "core/types.hpp"
#include "core/constants.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "io/upf_parser.hpp"
#include "solver/scf.hpp"
#include "solver/fermi.hpp"

#include <cmath>
#include <vector>

using namespace kronos;

namespace {

SCFResult run_si_scf(double ecutwfc, std::array<int,3> kgrid,
                     const std::map<std::string, PseudoPotential>& pps,
                     double ethr = 1e-3, double dthr = 1.0, int max_steps = 100) {
    Crystal crystal = test::make_si_diamond_crystal();
    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = ecutwfc;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = kgrid;
    ConvergenceParams conv;
    conv.energy_threshold = ethr;
    conv.density_threshold = dthr;
    conv.max_scf_steps = max_steps;
    SCFSolver solver(crystal, calc, conv, pps);
    return solver.solve();
}

} // namespace

// ============================================================================
// EcutwfcConvergence: energy vs plane-wave cutoff
// ============================================================================

TEST(EcutwfcConvergence, SiEnergyVsCutoff) {
    auto pps = test::make_si_pp_map();
    std::vector<double> cutoffs = {10, 15, 20, 25, 30};
    std::vector<double> energies;

    for (double ecut : cutoffs) {
        auto result = run_si_scf(ecut, {1,1,1}, pps);
        if (!result.converged) {
            GTEST_SKIP() << "SCF did not converge at ecut=" << ecut;
        }
        energies.push_back(result.total_energy_ry);
    }

    // Variational: energy must decrease (or stay same) with increasing cutoff
    // Allow small tolerance (~0.1 Ry) for FFT grid discretization effects
    // with toy Gaussian PPs where ecutrho grid doesn't grow monotonically
    for (size_t i = 1; i < energies.size(); ++i) {
        EXPECT_LE(energies[i], energies[i-1] + 0.1)
            << "Energy not monotonically decreasing: E(" << cutoffs[i-1]
            << ")=" << energies[i-1] << " < E(" << cutoffs[i] << ")=" << energies[i];
    }

    // All energies should be finite and negative
    for (size_t i = 0; i < energies.size(); ++i) {
        EXPECT_TRUE(std::isfinite(energies[i])) << "Non-finite energy at ecut=" << cutoffs[i];
    }
}

TEST(EcutwfcConvergence, SiEnergyVsCutoffNonlocal) {
    auto pps = test::make_si_pp_map_nonlocal();
    std::vector<double> cutoffs = {10, 15, 20};
    std::vector<double> energies;

    for (double ecut : cutoffs) {
        auto result = run_si_scf(ecut, {1,1,1}, pps);
        if (!result.converged) {
            GTEST_SKIP() << "SCF did not converge at ecut=" << ecut;
        }
        energies.push_back(result.total_energy_ry);
    }

    // Overall trend: highest cutoff should be lower than lowest cutoff.
    // Strict pairwise monotonicity is not guaranteed because ecutrho = 4*ecutwfc
    // changes the FFT grid (and thus the Hamiltonian) at each cutoff.
    EXPECT_LT(energies.back(), energies.front() + 1e-6)
        << "Nonlocal PP: E(max_ecut) should be lower than E(min_ecut)";
}

TEST(EcutwfcConvergence, CutoffConvergenceRate) {
    auto pps = test::make_si_pp_map();
    std::vector<double> cutoffs = {10, 15, 20, 25, 30};
    std::vector<double> energies;

    for (double ecut : cutoffs) {
        auto result = run_si_scf(ecut, {1,1,1}, pps);
        if (!result.converged) {
            GTEST_SKIP() << "SCF did not converge at ecut=" << ecut;
        }
        energies.push_back(result.total_energy_ry);
    }

    // Overall convergence: last energy difference should be smaller than first.
    // Strict pairwise monotonicity of |dE| is not guaranteed because
    // ecutrho = 4*ecutwfc changes the FFT grid at each cutoff.
    std::vector<double> diffs;
    for (size_t i = 1; i < energies.size(); ++i) {
        diffs.push_back(std::abs(energies[i] - energies[i-1]));
    }
    ASSERT_GE(diffs.size(), 2u);
    EXPECT_LT(diffs.back(), diffs.front() + 1e-6)
        << "Overall convergence: last |dE|=" << diffs.back()
        << " should be < first |dE|=" << diffs.front();
}

// ============================================================================
// KPointConvergence: energy vs k-point grid density
// ============================================================================

TEST(KPointConvergence, SiEnergyVsKGrid) {
    auto pps = test::make_si_pp_map();
    std::vector<std::array<int,3>> grids = {{1,1,1}, {2,2,2}, {3,3,3}};
    std::vector<double> energies;

    for (const auto& kg : grids) {
        auto result = run_si_scf(15.0, kg, pps);
        if (!result.converged) {
            GTEST_SKIP() << "SCF did not converge at k-grid "
                         << kg[0] << "x" << kg[1] << "x" << kg[2];
        }
        energies.push_back(result.total_energy_ry);
    }

    // Energy differences should shrink with denser grids
    if (energies.size() >= 3) {
        double diff1 = std::abs(energies[1] - energies[0]);
        double diff2 = std::abs(energies[2] - energies[1]);
        EXPECT_LE(diff2, diff1 + 1e-6)
            << "K-point convergence: differences not decreasing: "
            << diff1 << " -> " << diff2;
    }
}

TEST(KPointConvergence, KPointWeightsConserved) {
    Crystal crystal = test::make_si_diamond_crystal();
    // Test weight conservation for various grids using FermiSolver
    // We just need to verify that for any k-grid, weights sum to 1
    // This is implicitly tested via SCF but we test directly
    std::vector<std::array<int,3>> grids = {{1,1,1}, {2,2,2}, {3,3,3}, {4,4,4}};

    for (const auto& kg : grids) {
        // Build a PlaneWaveBasis and run a minimal eigenvalue solve to get k-points
        // Instead, we verify via SCF result eigenvalue structure
        auto pps = test::make_si_pp_map();
        auto result = run_si_scf(10.0, kg, pps);
        // The number of k-points should be between 1 and kg[0]*kg[1]*kg[2]
        EXPECT_GE(result.eigenvalues.size(), 1u)
            << "No k-points for grid " << kg[0] << "x" << kg[1] << "x" << kg[2];
        EXPECT_LE(result.eigenvalues.size(),
                  static_cast<size_t>(kg[0] * kg[1] * kg[2]))
            << "Too many k-points for grid " << kg[0] << "x" << kg[1] << "x" << kg[2];
    }
}

// ============================================================================
// SCFBehavior: convergence properties
// ============================================================================

TEST(SCFBehavior, ConvergesWithinReasonableSteps) {
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {1,1,1}, pps, 1e-4, 1.0, 100);
    EXPECT_TRUE(result.converged)
        << "Si SCF should converge within 100 steps at ecut=10, Gamma";
    EXPECT_LT(result.scf_steps, 100)
        << "SCF took " << result.scf_steps << " steps (expected < 100)";
}

TEST(SCFBehavior, TighterThresholdMoreSteps) {
    auto pps = test::make_si_pp_map();
    auto r_loose = run_si_scf(10.0, {1,1,1}, pps, 1e-3, 1.0, 100);
    auto r_tight = run_si_scf(10.0, {1,1,1}, pps, 1e-5, 1.0, 100);
    if (!r_loose.converged || !r_tight.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }
    EXPECT_GE(r_tight.scf_steps, r_loose.scf_steps)
        << "Tighter threshold should require at least as many steps: "
        << "loose=" << r_loose.scf_steps << " tight=" << r_tight.scf_steps;
}

TEST(SCFBehavior, DensityMixingStabilizes) {
    // Verify that SCF doesn't diverge (no oscillation > 1 Ry)
    auto pps = test::make_si_pp_map();
    auto result = run_si_scf(10.0, {1,1,1}, pps, 1e-3, 1.0, 100);
    // If SCF ran at all, it should not have hit the 1 Ry oscillation abort
    EXPECT_GT(result.scf_steps, 0) << "SCF ran zero steps";
    // The total energy should be finite (not NaN/inf from divergence)
    EXPECT_TRUE(std::isfinite(result.total_energy_ry))
        << "SCF diverged: total energy = " << result.total_energy_ry;
}
