#pragma once
#include <cstddef>
#include <stdexcept>

namespace kronos::gpu {

// GPU memory management - dispatches to cudaMalloc/hipMalloc
// v0.1: stub

class GPUNotAvailableError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

void* gpu_malloc(size_t bytes);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dst, const void* src, size_t bytes);
void gpu_memcpy_d2h(void* dst, const void* src, size_t bytes);
void gpu_memcpy_d2d(void* dst, const void* src, size_t bytes);

// Check if GPU is available
bool gpu_available();

// Get GPU memory info
size_t gpu_memory_free();
size_t gpu_memory_total();

} // namespace kronos::gpu
