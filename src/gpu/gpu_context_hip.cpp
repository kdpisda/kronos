// ============================================================================
// KRONOS  src/gpu/gpu_context_hip.cpp
// HIP/ROCm GPU context implementation
// ============================================================================

#ifdef KRONOS_GPU_HIP

#include "gpu/gpu_context.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <string>

namespace kronos::gpu {

GPUContext& GPUContext::instance() {
    static GPUContext ctx;
    return ctx;
}

void GPUContext::init(int mpi_rank, int local_rank) {
    if (initialized_) return;

    hipError_t err = hipGetDeviceCount(&num_devices_);
    if (err != hipSuccess || num_devices_ == 0) {
        std::cerr << "KRONOS GPU: No HIP devices found (rank " << mpi_rank << ")\n";
        return;
    }

    device_id_ = local_rank % num_devices_;
    err = hipSetDevice(device_id_);
    if (err != hipSuccess) {
        std::cerr << "KRONOS GPU: hipSetDevice(" << device_id_
                  << ") failed: " << hipGetErrorString(err) << "\n";
        return;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id_);
    device_name_ = std::string(prop.name);

    rocblas_handle handle;
    rocblas_status blas_err = rocblas_create_handle(&handle);
    if (blas_err != rocblas_status_success) {
        std::cerr << "KRONOS GPU: rocblas_create_handle failed\n";
        return;
    }
    blas_handle_ = static_cast<void*>(handle);

    initialized_ = true;

    std::cerr << "KRONOS GPU: rank " << mpi_rank
              << " → device " << device_id_
              << " (" << device_name_ << ")"
              << ", " << prop.totalGlobalMem / (1024*1024) << " MB\n";
}

void GPUContext::finalize() {
    if (!initialized_) return;

    if (blas_handle_) {
        rocblas_destroy_handle(static_cast<rocblas_handle>(blas_handle_));
        blas_handle_ = nullptr;
    }

    initialized_ = false;
}

GPUContext::~GPUContext() {
    finalize();
}

void GPUContext::set_deterministic(bool /*enable*/) {
    // ROCm does not have an equivalent workspace config
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_HIP
