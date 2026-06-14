// ============================================================================
// KRONOS  src/gpu/memory_metal.cpp
// Apple Silicon unified-memory wrapper for the gpu::memory interface.
// ============================================================================

#ifdef KRONOS_GPU_METAL

// NOTE: Do NOT define NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION /
// CA_PRIVATE_IMPLEMENTATION here — those are emitted once in gpu_context_metal.cpp.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <sys/sysctl.h>

#include "gpu/memory.hpp"
#include "gpu/gpu_context.hpp"

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace kronos::gpu {

namespace {

struct BufferRegistry {
    std::mutex                              mtx;
    std::unordered_map<void*, MTL::Buffer*> map;

    static BufferRegistry& instance() {
        static BufferRegistry r;
        return r;
    }

    void register_buffer(void* contents, MTL::Buffer* buf) {
        std::lock_guard<std::mutex> g(mtx);
        map[contents] = buf;
    }

    MTL::Buffer* lookup(void* contents) {
        std::lock_guard<std::mutex> g(mtx);
        auto it = map.find(contents);
        return (it == map.end()) ? nullptr : it->second;
    }

    MTL::Buffer* pop(void* contents) {
        std::lock_guard<std::mutex> g(mtx);
        auto it = map.find(contents);
        if (it == map.end()) return nullptr;
        MTL::Buffer* b = it->second;
        map.erase(it);
        return b;
    }
};

} // anonymous namespace

void* gpu_malloc(size_t bytes) {
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw GPUNotAvailableError("Metal context not initialized");
    }
    MTL::Device* device = static_cast<MTL::Device*>(ctx.blas_handle());
    MTL::Buffer* buf = device->newBuffer(
        bytes, MTL::ResourceStorageModeShared);
    if (!buf) {
        throw GPUNotAvailableError(
            "MTLBuffer allocation failed (" + std::to_string(bytes) + " bytes)");
    }
    void* contents = buf->contents();
    BufferRegistry::instance().register_buffer(contents, buf);
    return contents;
}

void gpu_free(void* ptr) {
    if (!ptr) return;
    MTL::Buffer* buf = BufferRegistry::instance().pop(ptr);
    if (buf) {
        buf->release();
    }
}

void gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    // On unified memory, dst is already host-visible. memcpy is the transfer.
    std::memcpy(dst, src, bytes);
    // Make CPU writes visible to GPU: didModifyRange on the underlying buffer.
    MTL::Buffer* buf = BufferRegistry::instance().lookup(dst);
    if (buf) {
        buf->didModifyRange(NS::Range::Make(0, bytes));
    }
}

void gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    // Caller is responsible for completing any in-flight GPU work first
    // (done via waitUntilCompleted in the FFT/GEMM wrappers).
    std::memcpy(dst, src, bytes);
}

void gpu_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
    MTL::Buffer* buf = BufferRegistry::instance().lookup(dst);
    if (buf) {
        buf->didModifyRange(NS::Range::Make(0, bytes));
    }
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
