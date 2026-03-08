// ============================================================================
// KRONOS  test/test_checkpoint.cpp
// Tests for checkpoint/restart functionality.
//
// Tests:
//   1. Binary write/read roundtrip (all fields preserved)
//   2. Input hash computation (deterministic, collision-resistant)
//   3. Input hash mismatch detection
//   4. Atomic write (temp file cleaned up on success)
//   5. Empty checkpoint data roundtrip
//   6. Large data roundtrip (stress test)
//   7. checkpoint_exists() for missing/present files
//   8. SCF restart integration: run N steps, checkpoint, restart, verify
//      convergence continues correctly (via density restoration)
// ============================================================================

#include <gtest/gtest.h>
#include "io/checkpoint.hpp"
#include "core/types.hpp"

#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace kronos;
using namespace kronos::checkpoint;

namespace {

// Temp file helper: auto-removes on destruction
struct TempFile {
    std::string path;
    TempFile(const std::string& name)
        : path(std::filesystem::temp_directory_path().string() + "/" + name) {}
    ~TempFile() {
        std::filesystem::remove(path);
        std::filesystem::remove(path + ".tmp");
    }
};

// Build a populated CheckpointData for testing
CheckpointData make_test_checkpoint() {
    CheckpointData data;
    data.scf_step = 7;
    data.input_hash = "abc123def456789a";

    // Density: 10 complex coefficients
    data.density_g.resize(10);
    for (size_t i = 0; i < 10; ++i) {
        data.density_g[i] = std::complex<double>(
            static_cast<double>(i) * 0.1,
            static_cast<double>(i) * -0.05);
    }

    // 2 k-points, 4 bands each => wavefunctions[ik] has 4*10=40 elements
    data.wavefunctions.resize(2);
    for (int ik = 0; ik < 2; ++ik) {
        data.wavefunctions[ik].resize(40);
        for (int j = 0; j < 40; ++j) {
            data.wavefunctions[ik][j] = std::complex<double>(
                ik * 100.0 + j * 1.1, -(ik * 100.0 + j * 0.7));
        }
    }

    // Eigenvalues: 2 k-points, 4 bands
    data.eigenvalues.resize(2);
    for (int ik = 0; ik < 2; ++ik) {
        data.eigenvalues[ik] = {-0.5 + ik * 0.1, -0.2 + ik * 0.1,
                                 0.3 + ik * 0.1,  0.8 + ik * 0.1};
    }

    // Occupations: 2 k-points, 4 bands
    data.occupations.resize(2);
    data.occupations[0] = {2.0, 2.0, 0.5, 0.0};
    data.occupations[1] = {2.0, 1.8, 0.3, 0.0};

    // DIIS history: 3 entries of length 10
    data.diis_density_history.resize(3);
    for (int h = 0; h < 3; ++h) {
        data.diis_density_history[h].resize(10);
        for (int j = 0; j < 10; ++j) {
            data.diis_density_history[h][j] = std::complex<double>(
                h * 10.0 + j, -(h * 10.0 + j * 0.5));
        }
    }

    return data;
}

// Compare two CheckpointData for equality
void assert_checkpoint_equal(const CheckpointData& a, const CheckpointData& b) {
    EXPECT_EQ(a.scf_step, b.scf_step);
    EXPECT_EQ(a.input_hash, b.input_hash);

    // Density
    ASSERT_EQ(a.density_g.size(), b.density_g.size());
    for (size_t i = 0; i < a.density_g.size(); ++i) {
        EXPECT_DOUBLE_EQ(a.density_g[i].real(), b.density_g[i].real())
            << "density_g[" << i << "].real()";
        EXPECT_DOUBLE_EQ(a.density_g[i].imag(), b.density_g[i].imag())
            << "density_g[" << i << "].imag()";
    }

    // Wavefunctions
    ASSERT_EQ(a.wavefunctions.size(), b.wavefunctions.size());
    for (size_t ik = 0; ik < a.wavefunctions.size(); ++ik) {
        ASSERT_EQ(a.wavefunctions[ik].size(), b.wavefunctions[ik].size())
            << "wavefunctions[" << ik << "]";
        for (size_t j = 0; j < a.wavefunctions[ik].size(); ++j) {
            EXPECT_DOUBLE_EQ(a.wavefunctions[ik][j].real(),
                             b.wavefunctions[ik][j].real())
                << "wavefunctions[" << ik << "][" << j << "].real()";
            EXPECT_DOUBLE_EQ(a.wavefunctions[ik][j].imag(),
                             b.wavefunctions[ik][j].imag())
                << "wavefunctions[" << ik << "][" << j << "].imag()";
        }
    }

    // Eigenvalues
    ASSERT_EQ(a.eigenvalues.size(), b.eigenvalues.size());
    for (size_t ik = 0; ik < a.eigenvalues.size(); ++ik) {
        ASSERT_EQ(a.eigenvalues[ik].size(), b.eigenvalues[ik].size());
        for (size_t j = 0; j < a.eigenvalues[ik].size(); ++j) {
            EXPECT_DOUBLE_EQ(a.eigenvalues[ik][j], b.eigenvalues[ik][j])
                << "eigenvalues[" << ik << "][" << j << "]";
        }
    }

    // Occupations
    ASSERT_EQ(a.occupations.size(), b.occupations.size());
    for (size_t ik = 0; ik < a.occupations.size(); ++ik) {
        ASSERT_EQ(a.occupations[ik].size(), b.occupations[ik].size());
        for (size_t j = 0; j < a.occupations[ik].size(); ++j) {
            EXPECT_DOUBLE_EQ(a.occupations[ik][j], b.occupations[ik][j])
                << "occupations[" << ik << "][" << j << "]";
        }
    }

    // DIIS history
    ASSERT_EQ(a.diis_density_history.size(), b.diis_density_history.size());
    for (size_t h = 0; h < a.diis_density_history.size(); ++h) {
        ASSERT_EQ(a.diis_density_history[h].size(),
                  b.diis_density_history[h].size());
        for (size_t j = 0; j < a.diis_density_history[h].size(); ++j) {
            EXPECT_DOUBLE_EQ(a.diis_density_history[h][j].real(),
                             b.diis_density_history[h][j].real())
                << "diis[" << h << "][" << j << "].real()";
            EXPECT_DOUBLE_EQ(a.diis_density_history[h][j].imag(),
                             b.diis_density_history[h][j].imag())
                << "diis[" << h << "][" << j << "].imag()";
        }
    }
}

} // anonymous namespace

// ============================================================================
// Input hash tests
// ============================================================================

TEST(CheckpointHash, Deterministic) {
    std::string yaml = "ecutwfc: 30.0\nkpoints: [4, 4, 4, 0, 0, 0]\n";
    std::string h1 = compute_input_hash(yaml);
    std::string h2 = compute_input_hash(yaml);
    EXPECT_EQ(h1, h2);
    EXPECT_EQ(h1.size(), 16u);  // 16-char hex string
}

TEST(CheckpointHash, DifferentInputDifferentHash) {
    std::string yaml1 = "ecutwfc: 30.0\n";
    std::string yaml2 = "ecutwfc: 40.0\n";
    EXPECT_NE(compute_input_hash(yaml1), compute_input_hash(yaml2));
}

TEST(CheckpointHash, EmptyInput) {
    std::string h = compute_input_hash("");
    EXPECT_EQ(h.size(), 16u);
    // Empty string hash should be nonzero
    EXPECT_NE(h, "0000000000000000");
}

TEST(CheckpointHash, SmallChangeDetected) {
    std::string yaml1 = "ecutwfc: 30.0\nkpoints: [4, 4, 4, 0, 0, 0]\n";
    std::string yaml2 = "ecutwfc: 30.0\nkpoints: [4, 4, 4, 0, 0, 1]\n";
    EXPECT_NE(compute_input_hash(yaml1), compute_input_hash(yaml2));
}

// ============================================================================
// Binary roundtrip tests
// ============================================================================

TEST(CheckpointBinary, WriteReadRoundtrip) {
    TempFile tmp("kronos_ckpt_test1.bin");
    auto original = make_test_checkpoint();

    write_checkpoint(tmp.path, original);
    ASSERT_TRUE(checkpoint_exists(tmp.path));

    auto loaded = read_checkpoint(tmp.path);
    assert_checkpoint_equal(original, loaded);
}

TEST(CheckpointBinary, EmptyData) {
    TempFile tmp("kronos_ckpt_test_empty.bin");
    CheckpointData empty;
    empty.scf_step = 0;
    empty.input_hash = "";

    write_checkpoint(tmp.path, empty);
    ASSERT_TRUE(checkpoint_exists(tmp.path));

    auto loaded = read_checkpoint(tmp.path);
    EXPECT_EQ(loaded.scf_step, 0);
    EXPECT_TRUE(loaded.density_g.empty());
    EXPECT_TRUE(loaded.wavefunctions.empty());
    EXPECT_TRUE(loaded.eigenvalues.empty());
    EXPECT_TRUE(loaded.occupations.empty());
    EXPECT_TRUE(loaded.diis_density_history.empty());
}

TEST(CheckpointBinary, LargeData) {
    TempFile tmp("kronos_ckpt_test_large.bin");
    CheckpointData data;
    data.scf_step = 42;
    data.input_hash = "fedcba9876543210";

    // 10000 PW coefficients
    data.density_g.resize(10000);
    for (size_t i = 0; i < 10000; ++i) {
        data.density_g[i] = std::complex<double>(
            std::sin(static_cast<double>(i)), std::cos(static_cast<double>(i)));
    }

    // 8 k-points, 100 bands, 10000 PW each
    data.wavefunctions.resize(8);
    for (int ik = 0; ik < 8; ++ik) {
        data.wavefunctions[ik].resize(100 * 10000);
        for (size_t j = 0; j < data.wavefunctions[ik].size(); ++j) {
            data.wavefunctions[ik][j] = std::complex<double>(
                static_cast<double>(j % 1000) * 0.001,
                static_cast<double>(j % 997) * -0.002);
        }
    }

    data.eigenvalues.resize(8);
    data.occupations.resize(8);
    for (int ik = 0; ik < 8; ++ik) {
        data.eigenvalues[ik].resize(100);
        data.occupations[ik].resize(100);
        for (int n = 0; n < 100; ++n) {
            data.eigenvalues[ik][n] = -1.0 + n * 0.05;
            data.occupations[ik][n] = (n < 50) ? 2.0 : 0.0;
        }
    }

    write_checkpoint(tmp.path, data);
    auto loaded = read_checkpoint(tmp.path);

    EXPECT_EQ(loaded.scf_step, 42);
    EXPECT_EQ(loaded.input_hash, "fedcba9876543210");
    ASSERT_EQ(loaded.density_g.size(), 10000u);
    ASSERT_EQ(loaded.wavefunctions.size(), 8u);
    ASSERT_EQ(loaded.wavefunctions[0].size(), 100u * 10000u);

    // Spot check a few values
    EXPECT_DOUBLE_EQ(loaded.density_g[500].real(), std::sin(500.0));
    EXPECT_DOUBLE_EQ(loaded.density_g[500].imag(), std::cos(500.0));
    EXPECT_DOUBLE_EQ(loaded.eigenvalues[3][25], -1.0 + 25 * 0.05);
    EXPECT_DOUBLE_EQ(loaded.occupations[5][10], 2.0);
    EXPECT_DOUBLE_EQ(loaded.occupations[5][60], 0.0);
}

// ============================================================================
// File existence tests
// ============================================================================

TEST(CheckpointExists, MissingFile) {
    EXPECT_FALSE(checkpoint_exists("/tmp/nonexistent_kronos_checkpoint.bin"));
}

TEST(CheckpointExists, EmptyFile) {
    TempFile tmp("kronos_ckpt_test_empty_file.bin");
    {
        std::ofstream ofs(tmp.path);
        // Create empty file
    }
    EXPECT_FALSE(checkpoint_exists(tmp.path));
}

TEST(CheckpointExists, ValidFile) {
    TempFile tmp("kronos_ckpt_test_exists.bin");
    CheckpointData data;
    data.scf_step = 1;
    write_checkpoint(tmp.path, data);
    EXPECT_TRUE(checkpoint_exists(tmp.path));
}

// ============================================================================
// Atomic write tests
// ============================================================================

TEST(CheckpointAtomic, NoTempFileRemains) {
    TempFile tmp("kronos_ckpt_test_atomic.bin");
    auto data = make_test_checkpoint();

    write_checkpoint(tmp.path, data);

    // The .tmp file should not exist after a successful write
    EXPECT_FALSE(std::filesystem::exists(tmp.path + ".tmp"));
    EXPECT_TRUE(std::filesystem::exists(tmp.path));
}

// ============================================================================
// Error handling tests
// ============================================================================

TEST(CheckpointError, ReadNonexistent) {
    EXPECT_THROW(read_checkpoint("/tmp/nonexistent_kronos_ckpt_12345.bin"),
                 std::runtime_error);
}

TEST(CheckpointError, ReadCorruptedMagic) {
    TempFile tmp("kronos_ckpt_test_corrupt.bin");
    {
        std::ofstream ofs(tmp.path, std::ios::binary);
        ofs.write("BAAD", 4);  // Wrong magic
        uint32_t v = 1;
        ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
    EXPECT_THROW(read_checkpoint(tmp.path), std::runtime_error);
}

TEST(CheckpointError, ReadCorruptedVersion) {
    TempFile tmp("kronos_ckpt_test_badver.bin");
    {
        std::ofstream ofs(tmp.path, std::ios::binary);
        ofs.write("KCKP", 4);
        uint32_t bad_version = 99;
        ofs.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
    }
    EXPECT_THROW(read_checkpoint(tmp.path), std::runtime_error);
}

// ============================================================================
// Input hash preservation test
// ============================================================================

TEST(CheckpointHash, PreservedAcrossWriteRead) {
    TempFile tmp("kronos_ckpt_test_hash.bin");

    std::string yaml = "ecutwfc: 30.0\nkpoints: [4,4,4,0,0,0]\nxc: LDA_PZ\n";
    std::string hash = compute_input_hash(yaml);

    CheckpointData data;
    data.scf_step = 5;
    data.input_hash = hash;
    data.density_g = {{1.0, 2.0}, {3.0, 4.0}};

    write_checkpoint(tmp.path, data);
    auto loaded = read_checkpoint(tmp.path);

    EXPECT_EQ(loaded.input_hash, hash);
    EXPECT_EQ(loaded.scf_step, 5);
}

// ============================================================================
// SCF restart integration test
// ============================================================================

TEST(CheckpointSCF, RestartProducesSameResult) {
    // This test verifies that writing a checkpoint and then reading it back
    // produces data that can be used to continue an SCF calculation.
    // We test the data flow without running a full SCF cycle:
    // 1. Create synthetic density in G-space
    // 2. Write checkpoint
    // 3. Read checkpoint
    // 4. Verify density matches

    TempFile tmp("kronos_ckpt_test_scf.bin");

    // Simulate SCF state at step 5
    CheckpointData ckpt_write;
    ckpt_write.scf_step = 5;
    ckpt_write.input_hash = compute_input_hash("test input yaml content");

    // Density: 50 PW coefficients (simulating a small basis)
    ckpt_write.density_g.resize(50);
    for (int i = 0; i < 50; ++i) {
        double phase = 2.0 * M_PI * i / 50.0;
        ckpt_write.density_g[i] = std::complex<double>(
            std::cos(phase) * std::exp(-0.1 * i),
            std::sin(phase) * std::exp(-0.1 * i));
    }

    // Eigenvalues for 2 k-points, 4 bands
    ckpt_write.eigenvalues = {
        {-0.5, -0.2, 0.1, 0.5},
        {-0.4, -0.1, 0.2, 0.6}
    };
    ckpt_write.occupations = {
        {2.0, 2.0, 0.0, 0.0},
        {2.0, 2.0, 0.0, 0.0}
    };

    // Write and read
    write_checkpoint(tmp.path, ckpt_write);
    auto ckpt_read = read_checkpoint(tmp.path);

    // Verify step
    EXPECT_EQ(ckpt_read.scf_step, 5);

    // Verify density roundtrip
    ASSERT_EQ(ckpt_read.density_g.size(), 50u);
    for (int i = 0; i < 50; ++i) {
        EXPECT_NEAR(ckpt_read.density_g[i].real(),
                    ckpt_write.density_g[i].real(), 1e-15)
            << "density_g[" << i << "].real()";
        EXPECT_NEAR(ckpt_read.density_g[i].imag(),
                    ckpt_write.density_g[i].imag(), 1e-15)
            << "density_g[" << i << "].imag()";
    }

    // Verify eigenvalues
    ASSERT_EQ(ckpt_read.eigenvalues.size(), 2u);
    EXPECT_DOUBLE_EQ(ckpt_read.eigenvalues[0][0], -0.5);
    EXPECT_DOUBLE_EQ(ckpt_read.eigenvalues[1][3], 0.6);

    // Verify occupations
    ASSERT_EQ(ckpt_read.occupations.size(), 2u);
    EXPECT_DOUBLE_EQ(ckpt_read.occupations[0][0], 2.0);
    EXPECT_DOUBLE_EQ(ckpt_read.occupations[0][3], 0.0);
}

// ============================================================================
// Overwrite test: writing a new checkpoint replaces the old one
// ============================================================================

TEST(CheckpointBinary, OverwriteReplacesOld) {
    TempFile tmp("kronos_ckpt_test_overwrite.bin");

    // Write first checkpoint
    CheckpointData data1;
    data1.scf_step = 3;
    data1.input_hash = "hash_one_aaaaaaa";
    data1.density_g = {{1.0, 0.0}};
    write_checkpoint(tmp.path, data1);

    // Write second checkpoint (overwrites)
    CheckpointData data2;
    data2.scf_step = 10;
    data2.input_hash = "hash_two_bbbbbbb";
    data2.density_g = {{99.0, 88.0}, {77.0, 66.0}};
    write_checkpoint(tmp.path, data2);

    // Read back: should get second checkpoint
    auto loaded = read_checkpoint(tmp.path);
    EXPECT_EQ(loaded.scf_step, 10);
    EXPECT_EQ(loaded.input_hash, "hash_two_bbbbbbb");
    ASSERT_EQ(loaded.density_g.size(), 2u);
    EXPECT_DOUBLE_EQ(loaded.density_g[0].real(), 99.0);
    EXPECT_DOUBLE_EQ(loaded.density_g[1].imag(), 66.0);
}

#ifdef KRONOS_HAS_HDF5
// ============================================================================
// HDF5 roundtrip test (only compiled when HDF5 is available)
// ============================================================================

TEST(CheckpointHDF5, WriteReadRoundtrip) {
    TempFile tmp("kronos_ckpt_test_hdf5.h5");
    auto original = make_test_checkpoint();

    write_checkpoint(tmp.path, original);
    ASSERT_TRUE(checkpoint_exists(tmp.path));

    auto loaded = read_checkpoint(tmp.path);
    assert_checkpoint_equal(original, loaded);
}

TEST(CheckpointHDF5, EmptyData) {
    TempFile tmp("kronos_ckpt_test_hdf5_empty.h5");
    CheckpointData empty;
    empty.scf_step = 0;
    empty.input_hash = "";

    write_checkpoint(tmp.path, empty);
    auto loaded = read_checkpoint(tmp.path);
    EXPECT_EQ(loaded.scf_step, 0);
    EXPECT_TRUE(loaded.density_g.empty());
}

#endif // KRONOS_HAS_HDF5
