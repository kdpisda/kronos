// ============================================================================
// KRONOS  src/gpu/gpu_context_metal.cpp
// Metal GPU context (Apple Silicon)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include "gpu/gpu_context.hpp"
#include "gpu/memory.hpp"

namespace kronos::gpu {

GPUContext& GPUContext::instance() {
    static GPUContext ctx;
    return ctx;
}

void GPUContext::init(int /*mpi_rank*/, int /*local_rank*/) {
    // Implemented in Task 7
    initialized_ = false;
}

void GPUContext::finalize() {
    initialized_ = false;
}

GPUContext::~GPUContext() {
    finalize();
}

void GPUContext::set_deterministic(bool /*enable*/) {
    // No-op for skeleton
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
