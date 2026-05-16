// ============================================================================
// KRONOS  src/gpu/memory_hip.cpp
// HIP memory management implementation
// ============================================================================

#ifdef KRONOS_GPU_HIP

#include "gpu/memory.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace kronos::gpu {

void* gpu_malloc(size_t bytes) {
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, bytes);
    if (err != hipSuccess) {
        throw std::runtime_error(
            "hipMalloc failed (" + std::to_string(bytes) + " bytes): " +
            hipGetErrorString(err));
    }
    return ptr;
}

void gpu_free(void* ptr) {
    if (ptr) {
        hipFree(ptr);
    }
}

void gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        throw std::runtime_error(
            "hipMemcpy H2D failed: " + std::string(hipGetErrorString(err)));
    }
}

void gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        throw std::runtime_error(
            "hipMemcpy D2H failed: " + std::string(hipGetErrorString(err)));
    }
}

void gpu_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
        throw std::runtime_error(
            "hipMemcpy D2D failed: " + std::string(hipGetErrorString(err)));
    }
}

bool gpu_available() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    return (err == hipSuccess && count > 0);
}

size_t gpu_memory_free() {
    size_t free_mem = 0, total_mem = 0;
    hipMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t gpu_memory_total() {
    size_t free_mem = 0, total_mem = 0;
    hipMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_HIP
