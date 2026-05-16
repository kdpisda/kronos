// ============================================================================
// KRONOS  src/gpu/gpu_context_cuda.cu
// CUDA GPU context implementation
// ============================================================================

#ifdef KRONOS_GPU_CUDA

#include "gpu/gpu_context.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <iostream>
#include <string>

namespace kronos::gpu {

GPUContext& GPUContext::instance() {
    static GPUContext ctx;
    return ctx;
}

void GPUContext::init(int mpi_rank, int local_rank) {
    if (initialized_) return;

    // Query number of devices
    cudaError_t err = cudaGetDeviceCount(&num_devices_);
    if (err != cudaSuccess || num_devices_ == 0) {
        std::cerr << "KRONOS GPU: No CUDA devices found (rank " << mpi_rank << ")\n";
        return;
    }

    // Select device based on node-local rank (round-robin)
    device_id_ = local_rank % num_devices_;
    err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        std::cerr << "KRONOS GPU: cudaSetDevice(" << device_id_
                  << ") failed: " << cudaGetErrorString(err) << "\n";
        return;
    }

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    device_name_ = std::string(prop.name);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t blas_err = cublasCreate(&handle);
    if (blas_err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "KRONOS GPU: cublasCreate failed\n";
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
        cublasDestroy(static_cast<cublasHandle_t>(blas_handle_));
        blas_handle_ = nullptr;
    }

    initialized_ = false;
}

GPUContext::~GPUContext() {
    finalize();
}

void GPUContext::set_deterministic(bool enable) {
    if (enable) {
        // Set workspace config for deterministic cuBLAS results
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
    }
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_CUDA
