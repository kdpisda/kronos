// ============================================================================
// KRONOS  src/gpu/memory_metal.cpp
// Metal GPU memory wrapper (Apple Silicon unified memory)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include "gpu/memory.hpp"

namespace kronos::gpu {

void* gpu_malloc(size_t /*bytes*/) {
    throw GPUNotAvailableError("metal memory not yet implemented (Task 8)");
}

void gpu_free(void* /*ptr*/) {
    // Implemented in Task 8
}

void gpu_memcpy_h2d(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError("metal memory not yet implemented (Task 8)");
}

void gpu_memcpy_d2h(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError("metal memory not yet implemented (Task 8)");
}

void gpu_memcpy_d2d(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError("metal memory not yet implemented (Task 8)");
}

bool gpu_available() {
    return false;  // Updated in Task 7
}

size_t gpu_memory_free() {
    return 0;
}

size_t gpu_memory_total() {
    return 0;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
