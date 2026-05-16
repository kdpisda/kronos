// ============================================================================
// KRONOS  src/gpu/memory_cuda.cu
// CUDA memory management implementation
// ============================================================================

#ifdef KRONOS_GPU_CUDA

#include "gpu/memory.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace kronos::gpu {

void* gpu_malloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "cudaMalloc failed (" + std::to_string(bytes) + " bytes): " +
            cudaGetErrorString(err));
    }
    return ptr;
}

void gpu_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "cudaMemcpy H2D failed: " + std::string(cudaGetErrorString(err)));
    }
}

void gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "cudaMemcpy D2H failed: " + std::string(cudaGetErrorString(err)));
    }
}

void gpu_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "cudaMemcpy D2D failed: " + std::string(cudaGetErrorString(err)));
    }
}

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

size_t gpu_memory_free() {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t gpu_memory_total() {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_CUDA
