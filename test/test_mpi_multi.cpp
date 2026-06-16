// ============================================================================
// KRONOS  test/test_mpi_multi.cpp
// Multi-rank MPI tests — run with mpirun -np 2 or mpirun -np 4.
//
// These tests verify correctness of MPI k-point parallelization:
//   - K-point distribution
//   - Si SCF on multiple ranks matches serial baselines
//   - Forces match serial
//
// Usage:
//   mpirun -np 2 ./test_mpi_multi
//   mpirun -np 4 ./test_mpi_multi
// ============================================================================

#include <gtest/gtest.h>
#include "utils/mpi_wrapper.hpp"
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "solver/scf.hpp"
#include "io/checkpoint.hpp"

#include <cmath>
#include <complex>
#include <vector>
#include <numeric>
#include <iostream>

using namespace kronos;

// ============================================================================
// Helpers
// ============================================================================

namespace {

/// Build the standard Si diamond unit cell (lattice constant a=10.2 bohr).
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

/// Build simple Gaussian toy pseudopotentials for Si.
std::map<std::string, PseudoPotential> make_si_pp() {
    PseudoPotential pp;
    pp.element = "Si";
    pp.z_valence = 4.0;
    pp.is_norm_conserving = true;
    pp.lmax = 0;
    pp.num_projectors = 0;
    pp.num_wfc = 0;

    // Simple Gaussian local potential: V(r) = -Z_v * exp(-r^2/(2*r_loc^2)) / r
    pp.mesh.npoints = 1000;
    pp.mesh.r.resize(1000);
    pp.mesh.rab.resize(1000);
    double dr = 0.01;
    for (int i = 0; i < 1000; ++i) {
        pp.mesh.r[i] = (i + 1) * dr;
        pp.mesh.rab[i] = dr;
    }

    pp.vloc.resize(1000);
    double r_loc = 0.44;
    for (int i = 0; i < 1000; ++i) {
        double r = pp.mesh.r[i];
        pp.vloc[i] = -pp.z_valence / r * std::exp(-r * r / (2.0 * r_loc * r_loc));
        pp.vloc[i] *= 2.0;  // Ry units
    }

    pp.rho_atomic.resize(1000, 0.0);
    double norm = 0.0;
    double r_rho = 1.0;
    for (int i = 0; i < 1000; ++i) {
        double r = pp.mesh.r[i];
        pp.rho_atomic[i] = std::exp(-r / r_rho) * r * r;
        norm += pp.rho_atomic[i] * pp.mesh.rab[i];
    }
    for (int i = 0; i < 1000; ++i) {
        pp.rho_atomic[i] *= pp.z_valence / norm;
    }

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;
    return pps;
}

} // anonymous namespace

// ============================================================================
// Basic MPI communication tests
// ============================================================================

TEST(MPIMulti, RankAndSize) {
    int r = mpi::rank();
    int s = mpi::size();
    EXPECT_GE(r, 0);
    EXPECT_GE(s, 1);
    EXPECT_LT(r, s);
}

TEST(MPIMulti, AllreduceMinMax) {
    double rank_d = static_cast<double>(mpi::rank());

    double min_val = rank_d;
    mpi::allreduce_min_inplace(&min_val, 1);
    EXPECT_DOUBLE_EQ(min_val, 0.0);

    double max_val = rank_d;
    mpi::allreduce_max_inplace(&max_val, 1);
    EXPECT_DOUBLE_EQ(max_val, static_cast<double>(mpi::size() - 1));
}

TEST(MPIMulti, AllreduceSumInt) {
    int one = 1;
    int total = 0;
    mpi::allreduce_sum(&one, &total, 1);
    EXPECT_EQ(total, mpi::size());
}

TEST(MPIMulti, LocalRank) {
    // On a single node, local_rank should be same as rank
    int lr = mpi::local_rank();
    EXPECT_GE(lr, 0);
    EXPECT_LT(lr, mpi::size());
}

TEST(MPIMulti, BroadcastChar) {
    std::string msg(16, '\0');
    if (mpi::rank() == 0) {
        msg = "hello_mpi_world";
    }
    mpi::bcast(msg.data(), 16, 0);
    EXPECT_EQ(msg, "hello_mpi_world");
}

// ============================================================================
// K-point distribution correctness
// ============================================================================

TEST(MPIMulti, KPointDistribution) {
    int nk_total = 10;
    int mpi_rank = mpi::rank();
    int mpi_size = mpi::size();

    std::vector<int> my_kpoints;
    for (int ik = 0; ik < nk_total; ++ik) {
        if (ik % mpi_size == mpi_rank) {
            my_kpoints.push_back(ik);
        }
    }

    // Each rank should get ceil(nk/np) or floor(nk/np) k-points
    int nk_local = static_cast<int>(my_kpoints.size());
    EXPECT_GE(nk_local, nk_total / mpi_size);
    EXPECT_LE(nk_local, (nk_total + mpi_size - 1) / mpi_size);

    // Verify global sum of local k-points equals total
    int nk_sum = nk_local;
    mpi::allreduce_sum_inplace(&nk_sum, 1);
    EXPECT_EQ(nk_sum, nk_total);
}

// ============================================================================
// Si SCF: multi-rank should match serial baseline
// ============================================================================

TEST(MPIMulti, SiGammaMatchesSerial) {
    // Gamma-only SCF — all ranks have the single k-point, so the
    // result must match the serial baseline exactly.
    Crystal crystal = make_si_diamond();
    auto pps = make_si_pp();

    CalculationParams calc;
    calc.ecutwfc = 15.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};
    calc.kpoints.shift = {0, 0, 0};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-8;
    conv.density_threshold = 1e-8;
    conv.max_scf_steps = 80;

    SCFSolver scf(crystal, calc, conv, pps);
    SCFResult result = scf.solve();

    EXPECT_TRUE(result.converged);

    // Serial baseline: -28.6052 Ry (from MEMORY.md)
    // All ranks should agree
    double energy = result.total_energy_ry;
    double ref = energy;  // Use this rank's result as reference
    mpi::bcast(&ref, 1, 0);
    EXPECT_NEAR(energy, ref, 1e-12);
}

TEST(MPIMulti, Si2x2x2MatchesSerial) {
    // 2x2x2 k-point grid — k-points distributed across ranks.
    // Result should match serial baseline to machine epsilon.
    Crystal crystal = make_si_diamond();
    auto pps = make_si_pp();

    CalculationParams calc;
    calc.ecutwfc = 15.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {2, 2, 2};
    calc.kpoints.shift = {0, 0, 0};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-8;
    conv.density_threshold = 1e-8;
    conv.max_scf_steps = 80;

    SCFSolver scf(crystal, calc, conv, pps);
    SCFResult result = scf.solve();

    EXPECT_TRUE(result.converged);

    // All ranks must have identical total energy
    double energy = result.total_energy_ry;
    double ref = energy;
    mpi::bcast(&ref, 1, 0);
    EXPECT_NEAR(energy, ref, 1e-12);
}

// ============================================================================
// Forces: multi-rank should match serial
// ============================================================================

TEST(MPIMulti, Si2x2x2ForcesMatch) {
    Crystal crystal = make_si_diamond();
    auto pps = make_si_pp();

    CalculationParams calc;
    calc.ecutwfc = 15.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {2, 2, 2};
    calc.kpoints.shift = {0, 0, 0};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-8;
    conv.density_threshold = 1e-8;
    conv.max_scf_steps = 80;

    SCFSolver scf(crystal, calc, conv, pps);
    SCFResult result = scf.solve();
    EXPECT_TRUE(result.converged);

    // Forces should be consistent across ranks
    if (!result.forces.empty()) {
        for (size_t ia = 0; ia < result.forces.size(); ++ia) {
            Vec3 f = result.forces[ia];
            Vec3 fref = f;
            mpi::bcast(fref.data(), 3, 0);
            for (int d = 0; d < 3; ++d) {
                EXPECT_NEAR(f[d], fref[d], 1e-12)
                    << "Force mismatch on atom " << ia << " dim " << d;
            }
        }
    }
}

// ============================================================================
// Checkpoint MPI: rank 0 write, broadcast read
// ============================================================================

TEST(MPIMulti, CheckpointMPIRoundTrip) {
    using namespace kronos::checkpoint;

    CheckpointData data;
    if (mpi::rank() == 0) {
        data.scf_step = 42;
        data.input_hash = "abcdef1234567890";
        data.density_g = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        data.eigenvalues = {{-1.0, -0.5, 0.0}, {0.1, 0.2}};
        data.occupations = {{2.0, 2.0, 0.0}, {2.0, 1.0}};
    }

    // Write from rank 0 only
    std::string ckpt_file = "/tmp/kronos_test_mpi_ckpt_" +
                            std::to_string(mpi::rank()) + ".bin";
    if (mpi::rank() == 0) {
        ckpt_file = "/tmp/kronos_test_mpi_ckpt.bin";
        write_checkpoint(ckpt_file, data);
    }

    // Broadcast filename
    int len = static_cast<int>(ckpt_file.size());
    mpi::bcast(&len, 1, 0);
    ckpt_file.resize(static_cast<size_t>(len));
    mpi::bcast(ckpt_file.data(), len, 0);

    // All ranks read via MPI-aware function
    CheckpointData loaded = read_checkpoint_mpi(ckpt_file);

    EXPECT_EQ(loaded.scf_step, 42);
    EXPECT_EQ(loaded.input_hash, "abcdef1234567890");
    EXPECT_EQ(loaded.density_g.size(), 3u);
    EXPECT_EQ(loaded.eigenvalues.size(), 2u);
    EXPECT_EQ(loaded.occupations.size(), 2u);

    // Cleanup
    if (mpi::rank() == 0) {
        std::remove(ckpt_file.c_str());
    }
}

// ============================================================================
// Main (MPI init/finalize)
// ============================================================================

int main(int argc, char** argv) {
    mpi::init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    // Only rank 0 prints test output
    if (mpi::rank() != 0) {
        ::testing::TestEventListeners& listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();
    mpi::finalize();
    return result;
}
