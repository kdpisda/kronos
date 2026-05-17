#pragma once
// ============================================================================
// MPI wrapper for KRONOS
//
// Provides a thin abstraction over MPI collectives used by the SCF solver.
// When compiled without MPI (no KRONOS_HAS_MPI), every function has a trivial
// serial stub so that the rest of the code is completely MPI-agnostic.
// ============================================================================

#include <complex>
#include <cstddef>
#include <vector>

namespace kronos::mpi {

/// Initialize MPI (calls MPI_Init).  Safe to call even without MPI.
void init(int* argc, char*** argv);

/// Finalize MPI (calls MPI_Finalize).  Safe to call even without MPI.
void finalize();

/// Return the rank of this process (0 in serial builds).
int rank();

/// Return the total number of processes (1 in serial builds).
int size();

/// Barrier synchronization (no-op in serial builds).
void barrier();

/// Return true if MPI has been initialized (always true in serial stubs
/// after init() is called, or trivially true if init() was never called
/// in a serial build).
bool is_initialized();

// -------------------------------------------------------------------------
// Allreduce (sum)
// -------------------------------------------------------------------------

/// Out-of-place allreduce-sum for doubles.
void allreduce_sum(const double* sendbuf, double* recvbuf, int count);

/// Out-of-place allreduce-sum for complex doubles.
void allreduce_sum(const std::complex<double>* sendbuf,
                   std::complex<double>* recvbuf, int count);

/// In-place allreduce-sum for doubles.
void allreduce_sum_inplace(double* buf, int count);

/// In-place allreduce-sum for complex doubles.
void allreduce_sum_inplace(std::complex<double>* buf, int count);

// -------------------------------------------------------------------------
// Broadcast
// -------------------------------------------------------------------------

/// Broadcast doubles from root to all ranks.
void bcast(double* buf, int count, int root = 0);

/// Broadcast complex doubles from root to all ranks.
void bcast(std::complex<double>* buf, int count, int root = 0);

/// Broadcast ints from root to all ranks.
void bcast(int* buf, int count, int root = 0);

// -------------------------------------------------------------------------
// Convenience aliases
// -------------------------------------------------------------------------

/// Alias for bcast (matches common "broadcast" naming convention).
inline void broadcast(double* buf, int count, int root = 0) {
    bcast(buf, count, root);
}

inline void broadcast(std::complex<double>* buf, int count, int root = 0) {
    bcast(buf, count, root);
}

inline void broadcast(int* buf, int count, int root = 0) {
    bcast(buf, count, root);
}

// -------------------------------------------------------------------------
// Allreduce (min/max) — for diagnostics and error checking
// -------------------------------------------------------------------------

/// In-place allreduce-min for doubles.
void allreduce_min_inplace(double* buf, int count);

/// In-place allreduce-max for doubles.
void allreduce_max_inplace(double* buf, int count);

/// Out-of-place allreduce-sum for ints (used for global error checking).
void allreduce_sum(const int* sendbuf, int* recvbuf, int count);

/// In-place allreduce-sum for ints.
void allreduce_sum_inplace(int* buf, int count);

// -------------------------------------------------------------------------
// Allgather
// -------------------------------------------------------------------------

/// Allgather doubles: each rank contributes `sendcount` elements.
/// `recvbuf` must have room for sendcount * size() elements.
void allgather(const double* sendbuf, int sendcount,
               double* recvbuf, int recvcount);

/// Allgather variable-length data (allgatherv).
/// `recvcounts[i]` is the number of doubles from rank i.
/// `displs[i]` is the offset in recvbuf for rank i.
void allgatherv(const double* sendbuf, int sendcount,
                double* recvbuf, const int* recvcounts, const int* displs);

// -------------------------------------------------------------------------
// Node-local rank (for GPU assignment)
// -------------------------------------------------------------------------

/// Return node-local rank via MPI_Comm_split_type(SHARED).
/// In serial builds, returns 0.
int local_rank();

// -------------------------------------------------------------------------
// Broadcast (additional overloads)
// -------------------------------------------------------------------------

/// Broadcast chars from root to all ranks (for checkpoint data).
void bcast(char* buf, int count, int root = 0);

} // namespace kronos::mpi
