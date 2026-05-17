#include "utils/mpi_wrapper.hpp"

#include <cstring>   // std::memcpy

#ifdef KRONOS_HAS_MPI
#include <mpi.h>
#endif

namespace kronos::mpi {

// =========================================================================
// Internal state
// =========================================================================

namespace {
#ifdef KRONOS_HAS_MPI
    bool g_mpi_initialized_by_us = false;
#endif
    bool g_initialized = false;
} // anonymous namespace

// =========================================================================
// Lifecycle
// =========================================================================

void init(int* argc, char*** argv) {
#ifdef KRONOS_HAS_MPI
    int already = 0;
    MPI_Initialized(&already);
    if (!already) {
        MPI_Init(argc, argv);
        g_mpi_initialized_by_us = true;
    }
#else
    (void)argc;
    (void)argv;
#endif
    g_initialized = true;
}

void finalize() {
#ifdef KRONOS_HAS_MPI
    if (g_mpi_initialized_by_us) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
#endif
    g_initialized = false;
}

int rank() {
#ifdef KRONOS_HAS_MPI
    int r = 0;
    int init = 0;
    MPI_Initialized(&init);
    if (init) {
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
    }
    return r;
#else
    return 0;
#endif
}

int size() {
#ifdef KRONOS_HAS_MPI
    int s = 1;
    int init = 0;
    MPI_Initialized(&init);
    if (init) {
        MPI_Comm_size(MPI_COMM_WORLD, &s);
    }
    return s;
#else
    return 1;
#endif
}

void barrier() {
#ifdef KRONOS_HAS_MPI
    int init = 0;
    MPI_Initialized(&init);
    if (init) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}

bool is_initialized() {
#ifdef KRONOS_HAS_MPI
    int init = 0;
    MPI_Initialized(&init);
    return init != 0;
#else
    return g_initialized;
#endif
}

// =========================================================================
// Allreduce (sum)
// =========================================================================

void allreduce_sum(const double* sendbuf, double* recvbuf, int count) {
#ifdef KRONOS_HAS_MPI
    MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    if (sendbuf != recvbuf) {
        std::memcpy(recvbuf, sendbuf, static_cast<size_t>(count) * sizeof(double));
    }
#endif
}

void allreduce_sum(const std::complex<double>* sendbuf,
                   std::complex<double>* recvbuf, int count) {
#ifdef KRONOS_HAS_MPI
    MPI_Allreduce(sendbuf, recvbuf, count, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    if (sendbuf != recvbuf) {
        std::memcpy(recvbuf, sendbuf,
                    static_cast<size_t>(count) * sizeof(std::complex<double>));
    }
#endif
}

void allreduce_sum_inplace(double* buf, int count) {
#ifdef KRONOS_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
#endif
}

void allreduce_sum_inplace(std::complex<double>* buf, int count) {
#ifdef KRONOS_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
#endif
}

// =========================================================================
// Broadcast
// =========================================================================

void bcast(double* buf, int count, int root) {
#ifdef KRONOS_HAS_MPI
    MPI_Bcast(buf, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
    (void)root;
#endif
}

void bcast(std::complex<double>* buf, int count, int root) {
#ifdef KRONOS_HAS_MPI
    MPI_Bcast(buf, count, MPI_C_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
    (void)root;
#endif
}

void bcast(int* buf, int count, int root) {
#ifdef KRONOS_HAS_MPI
    MPI_Bcast(buf, count, MPI_INT, root, MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
    (void)root;
#endif
}

// =========================================================================
// Allreduce (min/max/int)
// =========================================================================

void allreduce_min_inplace(double* buf, int count) {
#ifdef KRONOS_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
#endif
}

void allreduce_max_inplace(double* buf, int count) {
#ifdef KRONOS_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
#endif
}

void allreduce_sum(const int* sendbuf, int* recvbuf, int count) {
#ifdef KRONOS_HAS_MPI
    MPI_Allreduce(sendbuf, recvbuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    if (sendbuf != recvbuf) {
        std::memcpy(recvbuf, sendbuf, static_cast<size_t>(count) * sizeof(int));
    }
#endif
}

void allreduce_sum_inplace(int* buf, int count) {
#ifdef KRONOS_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
#endif
}

// =========================================================================
// Allgather
// =========================================================================

void allgather(const double* sendbuf, int sendcount,
               double* recvbuf, int recvcount) {
#ifdef KRONOS_HAS_MPI
    MPI_Allgather(sendbuf, sendcount, MPI_DOUBLE,
                  recvbuf, recvcount, MPI_DOUBLE,
                  MPI_COMM_WORLD);
#else
    (void)recvcount;
    std::memcpy(recvbuf, sendbuf, static_cast<size_t>(sendcount) * sizeof(double));
#endif
}

void allgatherv(const double* sendbuf, int sendcount,
                double* recvbuf, const int* recvcounts, const int* displs) {
#ifdef KRONOS_HAS_MPI
    MPI_Allgatherv(sendbuf, sendcount, MPI_DOUBLE,
                   recvbuf, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
#else
    (void)recvcounts;
    (void)displs;
    std::memcpy(recvbuf, sendbuf, static_cast<size_t>(sendcount) * sizeof(double));
#endif
}

// =========================================================================
// Node-local rank
// =========================================================================

int local_rank() {
#ifdef KRONOS_HAS_MPI
    int init = 0;
    MPI_Initialized(&init);
    if (!init) return 0;

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    int lr = 0;
    MPI_Comm_rank(local_comm, &lr);
    MPI_Comm_free(&local_comm);
    return lr;
#else
    return 0;
#endif
}

// =========================================================================
// Broadcast (char overload)
// =========================================================================

void bcast(char* buf, int count, int root) {
#ifdef KRONOS_HAS_MPI
    MPI_Bcast(buf, count, MPI_CHAR, root, MPI_COMM_WORLD);
#else
    (void)buf;
    (void)count;
    (void)root;
#endif
}

} // namespace kronos::mpi
