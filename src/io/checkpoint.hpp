#pragma once
// ============================================================================
// KRONOS  src/io/checkpoint.hpp
// Checkpoint/restart support for SCF calculations.
//
// Saves and restores the SCF state (density, wavefunctions, eigenvalues,
// occupations, DIIS history, step counter) to enable restarts from
// intermediate SCF iterations.
//
// Format: HDF5 if KRONOS_HAS_HDF5 is defined; otherwise a portable binary
// format with magic number "KCKP".
// ============================================================================

#include "core/types.hpp"
#include <complex>
#include <string>
#include <vector>

namespace kronos::checkpoint {

// ============================================================================
// Data container for checkpoint state
// ============================================================================

struct CheckpointData {
    // Electron density in G-space (complex coefficients on PW basis)
    std::vector<std::complex<double>> density_g;

    // Per-k-point wavefunctions: wavefunctions[ik] = flat array of
    // num_bands * num_pw complex coefficients
    std::vector<std::vector<std::complex<double>>> wavefunctions;

    // Per-k-point eigenvalues (Ry)
    std::vector<std::vector<double>> eigenvalues;

    // Per-k-point occupations
    std::vector<std::vector<double>> occupations;

    // DIIS/Pulay density mixing history vectors (each is a real-space density)
    std::vector<std::vector<std::complex<double>>> diis_density_history;

    // SCF iteration counter at time of checkpoint
    int scf_step{0};

    // Hash of the YAML input (to detect mismatches on restart)
    std::string input_hash;
};

// ============================================================================
// Public API
// ============================================================================

/// Write a checkpoint to disk.
/// Uses atomic writes: writes to a temporary file first, then renames.
/// If KRONOS_HAS_HDF5 is defined, writes HDF5; otherwise writes binary.
void write_checkpoint(const std::string& filename, const CheckpointData& data);

/// Read a checkpoint from disk.
/// Automatically detects format (HDF5 vs binary) by inspecting the file header.
CheckpointData read_checkpoint(const std::string& filename);

/// Check whether a checkpoint file exists and is readable.
bool checkpoint_exists(const std::string& filename);

/// Compute a hash of the YAML input content.
/// Uses a simple but effective hash (DJB2 variant) that produces a
/// 16-character hex string. This is NOT cryptographic; it is only used
/// to detect accidental mismatches between checkpoint and input.
std::string compute_input_hash(const std::string& yaml_content);

/// MPI-aware checkpoint write: only rank 0 writes.
/// All ranks call this; non-root ranks are no-ops.
void write_checkpoint_mpi(const std::string& filename, const CheckpointData& data);

/// MPI-aware checkpoint read: rank 0 reads, broadcasts to all ranks.
/// All ranks must call this together.
CheckpointData read_checkpoint_mpi(const std::string& filename);

} // namespace kronos::checkpoint
