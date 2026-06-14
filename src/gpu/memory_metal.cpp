// ============================================================================
// KRONOS  src/gpu/memory_metal.cpp
// Metal GPU memory wrapper (Apple Silicon unified memory)
// ============================================================================

#ifdef KRONOS_GPU_METAL

// NOTE: Do NOT define NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION /
// CA_PRIVATE_IMPLEMENTATION here — those are emitted once in gpu_context_metal.cpp.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <sys/sysctl.h>

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
    // Cheap probe: ask Metal whether a device exists. Safe before init().
    MTL::Device* probe = MTL::CreateSystemDefaultDevice();
    if (!probe) return false;
    probe->release();
    return true;
}

size_t gpu_memory_free() {
    MTL::Device* d = MTL::CreateSystemDefaultDevice();
    if (!d) return 0;
    // recommendedMaxWorkingSetSize is the closest Metal analog to "free GPU mem".
    size_t v = d->recommendedMaxWorkingSetSize();
    d->release();
    return v;
}

size_t gpu_memory_total() {
    // On unified-memory Apple Silicon, "total GPU mem" == system RAM.
    int64_t physmem = 0;
    size_t len = sizeof(physmem);
    sysctlbyname("hw.memsize", &physmem, &len, nullptr, 0);
    return static_cast<size_t>(physmem);
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
