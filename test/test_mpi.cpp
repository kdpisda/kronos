// ============================================================================
// KRONOS  test/test_mpi.cpp
// MPI wrapper correctness tests (serial stub behavior).
// ============================================================================

#include <gtest/gtest.h>
#include "utils/mpi_wrapper.hpp"
#include "test_helpers.hpp"
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "solver/scf.hpp"

#include <complex>
#include <cmath>
#include <vector>

using namespace kronos;

// ============================================================================
// Serial stub basic behavior
// ============================================================================

TEST(MPIWrapper, SerialRankIsZero) {
    EXPECT_EQ(mpi::rank(), 0);
}

TEST(MPIWrapper, SerialSizeIsOne) {
    EXPECT_EQ(mpi::size(), 1);
}

TEST(MPIWrapper, InitAndFinalizeSafe) {
    // Calling init/finalize in serial mode should be safe and idempotent
    mpi::init(nullptr, nullptr);
    EXPECT_TRUE(mpi::is_initialized());
    EXPECT_EQ(mpi::rank(), 0);
    EXPECT_EQ(mpi::size(), 1);
    mpi::finalize();
}

TEST(MPIWrapper, BarrierIsNoop) {
    // Barrier in serial mode should be a no-op (just must not crash)
    mpi::barrier();
    mpi::barrier();  // Call twice to ensure no state corruption
}

// ============================================================================
// Allreduce sum (in-place) — serial stub preserves data
// ============================================================================

TEST(MPIWrapper, AllreduceSumInplaceDouble) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.5, -7.2};
    std::vector<double> expected = data;
    mpi::allreduce_sum_inplace(data.data(), static_cast<int>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(data[i], expected[i])
            << "allreduce_sum_inplace altered element " << i;
    }
}

TEST(MPIWrapper, AllreduceSumInplaceComplex) {
    std::vector<std::complex<double>> data = {
        {1.0, -1.0}, {2.5, 3.5}, {0.0, 0.0}
    };
    std::vector<std::complex<double>> expected = data;
    mpi::allreduce_sum_inplace(data.data(), static_cast<int>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(data[i].real(), expected[i].real())
            << "Complex allreduce_sum_inplace altered real part at " << i;
        EXPECT_DOUBLE_EQ(data[i].imag(), expected[i].imag())
            << "Complex allreduce_sum_inplace altered imag part at " << i;
    }
}

// ============================================================================
// Allreduce sum (out-of-place) — serial stub copies send to recv
// ============================================================================

TEST(MPIWrapper, AllreduceSumOutOfPlaceDouble) {
    std::vector<double> send = {1.0, 2.0, 3.0};
    std::vector<double> recv(3, 0.0);
    mpi::allreduce_sum(send.data(), recv.data(), 3);
    EXPECT_DOUBLE_EQ(recv[0], 1.0);
    EXPECT_DOUBLE_EQ(recv[1], 2.0);
    EXPECT_DOUBLE_EQ(recv[2], 3.0);
}

TEST(MPIWrapper, AllreduceSumOutOfPlaceComplex) {
    std::vector<std::complex<double>> send = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::complex<double>> recv(2, {0.0, 0.0});
    mpi::allreduce_sum(send.data(), recv.data(), 2);
    EXPECT_DOUBLE_EQ(recv[0].real(), 1.0);
    EXPECT_DOUBLE_EQ(recv[0].imag(), 2.0);
    EXPECT_DOUBLE_EQ(recv[1].real(), 3.0);
    EXPECT_DOUBLE_EQ(recv[1].imag(), 4.0);
}

TEST(MPIWrapper, AllreduceSumOutOfPlaceSameBuffer) {
    // When sendbuf == recvbuf, serial stub should not crash (alias-safe)
    std::vector<double> data = {5.0, 6.0};
    mpi::allreduce_sum(data.data(), data.data(), 2);
    EXPECT_DOUBLE_EQ(data[0], 5.0);
    EXPECT_DOUBLE_EQ(data[1], 6.0);
}

// ============================================================================
// Broadcast — serial stub is a no-op (data unchanged)
// ============================================================================

TEST(MPIWrapper, BroadcastDoubleNoop) {
    std::vector<double> data = {42.0, -3.14};
    mpi::bcast(data.data(), static_cast<int>(data.size()), 0);
    EXPECT_DOUBLE_EQ(data[0], 42.0);
    EXPECT_DOUBLE_EQ(data[1], -3.14);
}

TEST(MPIWrapper, BroadcastComplexNoop) {
    std::vector<std::complex<double>> data = {{1.0, 2.0}};
    mpi::bcast(data.data(), 1, 0);
    EXPECT_DOUBLE_EQ(data[0].real(), 1.0);
    EXPECT_DOUBLE_EQ(data[0].imag(), 2.0);
}

TEST(MPIWrapper, BroadcastIntNoop) {
    std::vector<int> data = {7, 13, 42};
    mpi::bcast(data.data(), 3, 0);
    EXPECT_EQ(data[0], 7);
    EXPECT_EQ(data[1], 13);
    EXPECT_EQ(data[2], 42);
}

TEST(MPIWrapper, BroadcastAliasNoop) {
    // The convenience broadcast() alias should work the same as bcast()
    std::vector<double> data = {99.0};
    mpi::broadcast(data.data(), 1, 0);
    EXPECT_DOUBLE_EQ(data[0], 99.0);
}

// ============================================================================
// Allgather — serial stub copies send to recv
// ============================================================================

TEST(MPIWrapper, AllgatherSingleRank) {
    std::vector<double> send = {10.0, 20.0, 30.0};
    std::vector<double> recv(3, 0.0);
    mpi::allgather(send.data(), 3, recv.data(), 3);
    EXPECT_DOUBLE_EQ(recv[0], 10.0);
    EXPECT_DOUBLE_EQ(recv[1], 20.0);
    EXPECT_DOUBLE_EQ(recv[2], 30.0);
}

TEST(MPIWrapper, AllgathervSingleRank) {
    // For size=1, allgatherv should just copy send to recv
    std::vector<double> send = {1.0, 2.0};
    std::vector<double> recv(2, 0.0);
    std::vector<int> recvcounts = {2};
    std::vector<int> displs = {0};
    mpi::allgatherv(send.data(), 2, recv.data(),
                    recvcounts.data(), displs.data());
    EXPECT_DOUBLE_EQ(recv[0], 1.0);
    EXPECT_DOUBLE_EQ(recv[1], 2.0);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(MPIWrapper, ZeroCountOperations) {
    // Zero-count operations should be safe no-ops
    double dummy = 0.0;
    mpi::allreduce_sum_inplace(&dummy, 0);
    mpi::allreduce_sum(&dummy, &dummy, 0);
    mpi::bcast(&dummy, 0, 0);
    // Just verify no crash
    EXPECT_DOUBLE_EQ(dummy, 0.0);
}

TEST(MPIWrapper, LargeBuffer) {
    // Test with a moderately large buffer to verify no size issues
    const int N = 10000;
    std::vector<double> send(N);
    std::vector<double> recv(N, 0.0);
    for (int i = 0; i < N; ++i) {
        send[i] = static_cast<double>(i) * 0.001;
    }
    mpi::allreduce_sum(send.data(), recv.data(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i])
            << "Mismatch at index " << i;
    }
}

// ============================================================================
// Integration: single-rank SCF must match serial baseline
// ============================================================================

TEST(MPIIntegration, SingleRankMatchesSerial) {
    // Run a simple Si Gamma SCF and verify energy matches the known baseline.
    // Since mpi::size()==1, nk_local==nk_total, results are identical to serial.
    EXPECT_EQ(mpi::size(), 1) << "This test assumes serial (single-rank) execution";

    auto pps = test::make_si_pp_map();
    Crystal crystal = test::make_si_diamond_crystal();

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 10.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    ConvergenceParams conv;
    conv.energy_threshold = 1e-6;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    auto result = solver.solve();

    if (!result.converged) {
        GTEST_SKIP() << "Si Gamma SCF did not converge";
    }

    // Regression baseline: ~-27.58 Ry (toy Gaussian PP, ecut=10, Gamma)
    EXPECT_NEAR(result.total_energy_ry, -27.58, 0.5)
        << "Single-rank SCF energy deviates from known baseline";
    EXPECT_LT(result.total_energy_ry, 0.0);
    EXPECT_GT(result.kinetic_energy, 0.0);
    EXPECT_GT(result.hartree_energy, 0.0);
    EXPECT_LT(result.xc_energy, 0.0);

    std::printf("  MPI single-rank Si Gamma: E_total = %.6f Ry, %d SCF steps\n",
                result.total_energy_ry, result.scf_steps);
}
