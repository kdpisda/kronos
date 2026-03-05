#include <gtest/gtest.h>

#include "solver/mixing.hpp"
#include "solver/davidson.hpp"
#include "solver/fermi.hpp"
#include "core/types.hpp"
#include "core/constants.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

// ============================================================================
// LinearMixer tests
// ============================================================================

TEST(LinearMixer, BasicMixing) {
    kronos::LinearMixer mixer(0.5);

    kronos::RVec n_in  = {1.0, 2.0, 3.0, 4.0};
    kronos::RVec n_out = {3.0, 4.0, 5.0, 6.0};

    auto result = mixer.mix(n_in, n_out);

    // With alpha=0.5: result = 0.5*n_in + 0.5*n_out
    ASSERT_EQ(result.size(), 4u);
    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[2], 4.0);
    EXPECT_DOUBLE_EQ(result[3], 5.0);
}

// ============================================================================
// PulayMixer tests
// ============================================================================

TEST(PulayMixer, ConvergesLinearSystem) {
    // Apply Pulay mixing to a simple residual sequence where the
    // "output" density approaches a target.
    // Simulate a system where the true solution is n* = {1, 1, 1, 1}
    // and n_out = n_in + alpha*(n* - n_in) (linearized response).

    kronos::PulayMixer mixer(8, 0.3);

    const kronos::RVec target = {1.0, 1.0, 1.0, 1.0};
    kronos::RVec n_in = {0.0, 0.0, 0.0, 0.0};  // initial guess

    // Simple model: n_out = n_in + 0.5*(target - n_in)
    // This simulates a linear dielectric response
    for (int step = 0; step < 20; ++step) {
        kronos::RVec n_out(4);
        for (size_t i = 0; i < 4; ++i) {
            n_out[i] = n_in[i] + 0.5 * (target[i] - n_in[i]);
        }
        n_in = mixer.mix(n_in, n_out);
    }

    // After 20 Pulay steps, should be close to target
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(n_in[i], target[i], 0.05)
            << "Component " << i << " did not converge to target";
    }
}

TEST(PulayMixer, ResetClearsHistory) {
    kronos::PulayMixer mixer(8, 0.3);

    kronos::RVec a = {1.0, 2.0};
    kronos::RVec b = {2.0, 3.0};

    mixer.mix(a, b);
    mixer.mix(a, b);
    EXPECT_GT(mixer.history_size(), 0);

    mixer.reset();
    EXPECT_EQ(mixer.history_size(), 0);
}

// ============================================================================
// FermiSolver tests
// ============================================================================

TEST(FermiSolver, InsulatorFermiLevel) {
    // Si-like insulator: 4 bands per k-point
    // Eigenvalues in eV, convert to Ry: [-5, -2, 2, 5] eV
    // Gap between -2 eV and 2 eV; 4 electrons fill 2 bands (spin_factor=2)
    const double ev_to_ry = kronos::constants::ev_to_rydberg;

    std::vector<std::vector<double>> eigenvalues = {
        { -5.0 * ev_to_ry, -2.0 * ev_to_ry, 2.0 * ev_to_ry, 5.0 * ev_to_ry }
    };
    std::vector<double> weights = { 1.0 };

    double target_electrons = 4.0;
    double degauss = 0.01;  // Ry, small smearing

    auto result = kronos::FermiSolver::find_fermi_level(
        eigenvalues, weights, target_electrons,
        kronos::SmearingType::Gaussian, degauss, 2);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.total_electrons_found, target_electrons, 1e-6);

    // Fermi level should be in the gap: between -2 eV and 2 eV (in Ry)
    EXPECT_GT(result.fermi_energy, -2.0 * ev_to_ry);
    EXPECT_LT(result.fermi_energy,  2.0 * ev_to_ry);
}

TEST(FermiSolver, MetalFermiLevel) {
    // Metallic system: partially filled band
    // 1 k-point, 4 bands at [-3, -1, 0.5, 3] eV, 5 electrons
    // With spin_factor=2, 2 bands fully filled = 4 electrons,
    // 1 more electron partially fills the 3rd band
    const double ev_to_ry = kronos::constants::ev_to_rydberg;

    std::vector<std::vector<double>> eigenvalues = {
        { -3.0 * ev_to_ry, -1.0 * ev_to_ry, 0.5 * ev_to_ry, 3.0 * ev_to_ry }
    };
    std::vector<double> weights = { 1.0 };

    double target_electrons = 5.0;
    double degauss = 0.02;  // Ry

    auto result = kronos::FermiSolver::find_fermi_level(
        eigenvalues, weights, target_electrons,
        kronos::SmearingType::FermiDirac, degauss, 2);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.total_electrons_found, target_electrons, 1e-6);

    // Fermi level should be between 2nd and 3rd band
    EXPECT_GT(result.fermi_energy, -1.0 * ev_to_ry);
    EXPECT_LT(result.fermi_energy,  3.0 * ev_to_ry);
}

TEST(FermiSolver, StepFunctionSmearing) {
    // No smearing: occupations should be exactly 0 or 1 (times spin_factor)
    const double ev_to_ry = kronos::constants::ev_to_rydberg;

    std::vector<std::vector<double>> eigenvalues = {
        { -3.0 * ev_to_ry, -1.0 * ev_to_ry, 1.0 * ev_to_ry, 3.0 * ev_to_ry }
    };
    std::vector<double> weights = { 1.0 };

    double target_electrons = 4.0;
    // degauss must be nonzero to avoid division by zero in x=(e-ef)/degauss
    // but SmearingType::None uses a step function so the value doesn't matter
    double degauss = 0.01;

    auto result = kronos::FermiSolver::find_fermi_level(
        eigenvalues, weights, target_electrons,
        kronos::SmearingType::None, degauss, 2);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.total_electrons_found, target_electrons, 1e-6);

    // Check occupations are 0 or 2 (spin_factor=2)
    for (size_t ib = 0; ib < eigenvalues[0].size(); ++ib) {
        double occ = result.occupations[0][ib];
        bool is_fully_occupied = std::abs(occ - 2.0) < 1e-10;
        bool is_empty = std::abs(occ) < 1e-10;
        EXPECT_TRUE(is_fully_occupied || is_empty)
            << "Band " << ib << " has occupation " << occ
            << " (expected 0 or 2)";
    }
}

// ============================================================================
// DavidsonSolver tests
// ============================================================================

TEST(DavidsonSolver, SolvesDiagonalMatrix) {
    // Use a diagonal "Hamiltonian": H|psi> returns kinetic[i] * psi[i]
    // The eigenvalues should match sorted kinetic energies.
    const int num_pw = 50;
    const int num_bands = 5;

    // Create kinetic energies (unsorted, to verify the solver finds them)
    std::vector<double> kinetic(num_pw);
    for (int i = 0; i < num_pw; ++i) {
        kinetic[i] = 0.5 * static_cast<double>(i + 1);  // 0.5, 1.0, 1.5, ...
    }

    // H|psi> = kinetic .* psi (element-wise multiplication)
    auto h_apply = [&kinetic, num_pw](const kronos::CVec& psi) -> kronos::CVec {
        kronos::CVec result(num_pw);
        for (int i = 0; i < num_pw; ++i) {
            result[i] = kinetic[i] * psi[i];
        }
        return result;
    };

    // Preconditioner: shifted kinetic energies (inexact, as in real DFT
    // where H also includes potentials beyond the kinetic term)
    std::vector<double> precond(num_pw);
    for (int i = 0; i < num_pw; ++i) {
        precond[i] = kinetic[i] + 1.0;
    }

    kronos::DavidsonSolver::Params params;
    params.max_iterations = 200;
    params.tolerance = 1e-8;

    kronos::DavidsonSolver solver(params);
    auto result = solver.solve(h_apply, precond, num_bands, num_pw);

    EXPECT_TRUE(result.converged) << "Davidson solver did not converge, "
                                   << "max_residual = " << result.max_residual;

    // Sort kinetic energies to get expected eigenvalues
    std::vector<double> expected(kinetic.begin(), kinetic.end());
    std::sort(expected.begin(), expected.end());

    // Check the lowest num_bands eigenvalues
    ASSERT_EQ(static_cast<int>(result.eigenvalues.size()), num_bands);
    for (int i = 0; i < num_bands; ++i) {
        EXPECT_NEAR(result.eigenvalues[i], expected[i], 1e-6)
            << "Eigenvalue " << i << " mismatch: got " << result.eigenvalues[i]
            << ", expected " << expected[i];
    }

    // Verify eigenvectors have the correct size
    ASSERT_EQ(static_cast<int>(result.eigenvectors.size()), num_bands);
    for (int i = 0; i < num_bands; ++i) {
        EXPECT_EQ(static_cast<int>(result.eigenvectors[i].size()), num_pw);
    }
}

// ============================================================================
// Additional LinearMixer tests
// ============================================================================

TEST(LinearMixer, SmallAlpha) {
    // Linear mixing with very small alpha: result ≈ n_in
    kronos::LinearMixer mixer(0.01);
    kronos::RVec n_in  = {1.0, 2.0, 3.0};
    kronos::RVec n_out = {5.0, 6.0, 7.0};
    auto result = mixer.mix(n_in, n_out);

    // With alpha=0.01: result is almost entirely n_in
    for (size_t i = 0; i < 3; ++i) {
        // result should be close to n_in but shifted slightly toward n_out
        EXPECT_NEAR(result[i], n_in[i], 0.1);
    }
}

TEST(LinearMixer, AlphaOneReturnInput) {
    // Linear mixing: result = alpha*n_out + (1-alpha)*n_in
    // alpha=1 means fully trust n_out
    kronos::LinearMixer mixer(1.0);
    kronos::RVec n_in  = {1.0, 2.0, 3.0};
    kronos::RVec n_out = {5.0, 6.0, 7.0};
    auto result = mixer.mix(n_in, n_out);

    // With alpha=1: result = n_out
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(result[i], n_out[i]);
    }
}

TEST(LinearMixer, PreservesSize) {
    kronos::LinearMixer mixer(0.3);
    kronos::RVec n_in(100, 1.0);
    kronos::RVec n_out(100, 2.0);
    auto result = mixer.mix(n_in, n_out);
    EXPECT_EQ(result.size(), 100u);
}

// ============================================================================
// Additional PulayMixer tests
// ============================================================================

TEST(PulayMixer, HistoryGrowsUpToMax) {
    kronos::PulayMixer mixer(4, 0.3);

    kronos::RVec a = {1.0, 2.0};
    kronos::RVec b = {2.0, 3.0};

    for (int i = 0; i < 10; ++i) {
        mixer.mix(a, b);
    }
    // History should be capped at max (4)
    EXPECT_LE(mixer.history_size(), 4);
}

TEST(PulayMixer, FirstStepIsLinear) {
    // With no history, Pulay should behave like linear mixing
    kronos::PulayMixer pulay(8, 0.5);
    kronos::LinearMixer linear(0.5);

    kronos::RVec n_in  = {1.0, 2.0, 3.0, 4.0};
    kronos::RVec n_out = {3.0, 4.0, 5.0, 6.0};

    auto result_pulay = pulay.mix(n_in, n_out);
    auto result_linear = linear.mix(n_in, n_out);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(result_pulay[i], result_linear[i], 1e-10)
            << "First Pulay step should equal linear mixing";
    }
}

// ============================================================================
// Additional Davidson tests
// ============================================================================

TEST(DavidsonSolver, ConvergesForDegenerateEigenvalues) {
    const int num_pw = 30;
    const int num_bands = 4;

    // Create a spectrum with degeneracy: 1.0, 1.0, 2.0, 2.0, 3.0, ...
    std::vector<double> kinetic(num_pw);
    for (int i = 0; i < num_pw; ++i) {
        kinetic[i] = 1.0 + static_cast<double>(i / 2);
    }

    auto h_apply = [&kinetic, num_pw](const kronos::CVec& psi) -> kronos::CVec {
        kronos::CVec result(num_pw);
        for (int i = 0; i < num_pw; ++i) {
            result[i] = kinetic[i] * psi[i];
        }
        return result;
    };

    std::vector<double> precond(num_pw);
    for (int i = 0; i < num_pw; ++i) precond[i] = kinetic[i] + 0.5;

    kronos::DavidsonSolver::Params params;
    params.max_iterations = 200;
    params.tolerance = 1e-8;
    kronos::DavidsonSolver solver(params);
    auto result = solver.solve(h_apply, precond, num_bands, num_pw);

    EXPECT_TRUE(result.converged);
    std::vector<double> expected(kinetic.begin(), kinetic.begin() + num_bands);
    std::sort(expected.begin(), expected.end());
    for (int i = 0; i < num_bands; ++i) {
        EXPECT_NEAR(result.eigenvalues[i], expected[i], 1e-6);
    }
}

TEST(DavidsonSolver, EigenvaluesSorted) {
    const int num_pw = 40;
    const int num_bands = 5;

    std::vector<double> kinetic(num_pw);
    for (int i = 0; i < num_pw; ++i) {
        kinetic[i] = 0.3 * (i + 1);
    }

    auto h_apply = [&kinetic, num_pw](const kronos::CVec& psi) -> kronos::CVec {
        kronos::CVec result(num_pw);
        for (int i = 0; i < num_pw; ++i) result[i] = kinetic[i] * psi[i];
        return result;
    };

    std::vector<double> precond(num_pw);
    for (int i = 0; i < num_pw; ++i) precond[i] = kinetic[i] + 1.0;

    kronos::DavidsonSolver solver;
    auto result = solver.solve(h_apply, precond, num_bands, num_pw);

    // Eigenvalues should be sorted ascending
    for (int i = 1; i < num_bands; ++i) {
        EXPECT_GE(result.eigenvalues[i], result.eigenvalues[i-1] - 1e-10);
    }
}

TEST(DavidsonSolver, EigenvectorsOrthogonal) {
    const int num_pw = 30;
    const int num_bands = 3;

    std::vector<double> kinetic(num_pw);
    for (int i = 0; i < num_pw; ++i) kinetic[i] = 0.5 * (i + 1);

    auto h_apply = [&kinetic, num_pw](const kronos::CVec& psi) -> kronos::CVec {
        kronos::CVec result(num_pw);
        for (int i = 0; i < num_pw; ++i) result[i] = kinetic[i] * psi[i];
        return result;
    };

    std::vector<double> precond(num_pw);
    for (int i = 0; i < num_pw; ++i) precond[i] = kinetic[i] + 1.0;

    kronos::DavidsonSolver solver;
    auto result = solver.solve(h_apply, precond, num_bands, num_pw);

    ASSERT_TRUE(result.converged);

    // Check orthogonality: <psi_i|psi_j> ≈ delta_ij
    for (int i = 0; i < num_bands; ++i) {
        for (int j = 0; j < num_bands; ++j) {
            kronos::complex_t overlap{0.0, 0.0};
            for (int k = 0; k < num_pw; ++k) {
                overlap += std::conj(result.eigenvectors[i][k])
                         * result.eigenvectors[j][k];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(std::abs(overlap), expected, 1e-6)
                << "Eigenvector orthogonality violated for (" << i << "," << j << ")";
        }
    }
}

TEST(DavidsonSolver, DefaultParamsWork) {
    // Default constructor should have reasonable defaults
    kronos::DavidsonSolver solver;
    const int num_pw = 20;
    const int num_bands = 2;

    std::vector<double> kinetic(num_pw);
    for (int i = 0; i < num_pw; ++i) kinetic[i] = 1.0 * (i + 1);

    auto h_apply = [&kinetic, num_pw](const kronos::CVec& psi) -> kronos::CVec {
        kronos::CVec result(num_pw);
        for (int i = 0; i < num_pw; ++i) result[i] = kinetic[i] * psi[i];
        return result;
    };
    std::vector<double> precond(num_pw);
    for (int i = 0; i < num_pw; ++i) precond[i] = kinetic[i] + 1.0;

    auto result = solver.solve(h_apply, precond, num_bands, num_pw);
    EXPECT_TRUE(result.converged);
}

// ============================================================================
// Additional Fermi level tests
// ============================================================================

TEST(FermiSolver, ElectronConservation) {
    // Total electrons found should match target
    const double ev_to_ry = kronos::constants::ev_to_rydberg;

    std::vector<std::vector<double>> eigenvalues = {
        {-5.0*ev_to_ry, -3.0*ev_to_ry, -1.0*ev_to_ry, 1.0*ev_to_ry, 3.0*ev_to_ry}
    };
    std::vector<double> weights = {1.0};

    for (double n_el : {2.0, 4.0, 6.0, 8.0}) {
        auto result = kronos::FermiSolver::find_fermi_level(
            eigenvalues, weights, n_el,
            kronos::SmearingType::Gaussian, 0.02, 2);

        EXPECT_TRUE(result.converged);
        EXPECT_NEAR(result.total_electrons_found, n_el, 1e-6)
            << "Electron conservation violated for n_el=" << n_el;
    }
}

TEST(FermiSolver, MultipleKPoints) {
    const double ev_to_ry = kronos::constants::ev_to_rydberg;

    std::vector<std::vector<double>> eigenvalues = {
        {-5.0*ev_to_ry, -2.0*ev_to_ry, 2.0*ev_to_ry, 5.0*ev_to_ry},
        {-4.0*ev_to_ry, -1.0*ev_to_ry, 3.0*ev_to_ry, 6.0*ev_to_ry},
        {-3.0*ev_to_ry,  0.0,          4.0*ev_to_ry, 7.0*ev_to_ry},
    };
    std::vector<double> weights = {0.25, 0.5, 0.25};

    auto result = kronos::FermiSolver::find_fermi_level(
        eigenvalues, weights, 4.0,
        kronos::SmearingType::Gaussian, 0.02, 2);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.total_electrons_found, 4.0, 1e-6);

    // Occupations should exist for each k-point
    ASSERT_EQ(result.occupations.size(), 3u);
    for (const auto& occ : result.occupations) {
        ASSERT_EQ(occ.size(), 4u);
        for (double o : occ) {
            EXPECT_GE(o, -0.1);  // MV can go slightly negative
            EXPECT_LE(o, 2.1);   // max is spin_factor
        }
    }
}

TEST(FermiSolver, OccupationsMonotonic) {
    // For a single k-point with sorted eigenvalues,
    // occupations should be monotonically non-increasing
    const double ev_to_ry = kronos::constants::ev_to_rydberg;

    std::vector<std::vector<double>> eigenvalues = {
        {-5.0*ev_to_ry, -2.0*ev_to_ry, 1.0*ev_to_ry, 5.0*ev_to_ry}
    };
    std::vector<double> weights = {1.0};

    auto result = kronos::FermiSolver::find_fermi_level(
        eigenvalues, weights, 4.0,
        kronos::SmearingType::FermiDirac, 0.02, 2);

    ASSERT_TRUE(result.converged);
    for (size_t i = 1; i < result.occupations[0].size(); ++i) {
        EXPECT_LE(result.occupations[0][i],
                  result.occupations[0][i-1] + 1e-6)
            << "Occupations should decrease with increasing eigenvalue";
    }
}

} // anonymous namespace
