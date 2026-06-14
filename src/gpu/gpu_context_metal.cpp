// ============================================================================
// KRONOS  src/gpu/gpu_context_metal.cpp
// Metal GPU context (Apple Silicon)
// ============================================================================

#ifdef KRONOS_GPU_METAL

// metal-cpp implementation-emission macros — MUST appear in exactly one .cpp
// in the entire program. This is that file. Do NOT define these in any other
// Metal source file.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "gpu/gpu_context.hpp"
#include "gpu/memory.hpp"
#include <iostream>
#include <string>

namespace kronos::gpu {

GPUContext& GPUContext::instance() {
    static GPUContext ctx;
    return ctx;
}

void GPUContext::init(int mpi_rank, int /*local_rank*/) {
    if (initialized_) return;

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "KRONOS GPU: no Metal device found (rank "
                  << mpi_rank << ")\n";
        return;
    }
    // metal-cpp uses retain/release; CreateSystemDefaultDevice returns +1.
    // We hold it for the lifetime of the singleton.
    blas_handle_ = static_cast<void*>(device);

    MTL::CommandQueue* queue = device->newCommandQueue();
    if (!queue) {
        device->release();
        blas_handle_ = nullptr;
        std::cerr << "KRONOS GPU: failed to create Metal command queue\n";
        return;
    }
    metal_queue_ = static_cast<void*>(queue);

    num_devices_ = 1;  // metal-cpp default-device API; multi-GPU later
    device_id_   = 0;

    NS::String* name = device->name();
    device_name_ = std::string(name->utf8String());

    initialized_ = true;

    std::cerr << "KRONOS GPU: rank " << mpi_rank
              << " \xe2\x86\x92 Metal device '" << device_name_ << "'\n";
}

void GPUContext::finalize() {
    if (!initialized_) return;
    if (metal_queue_) {
        static_cast<MTL::CommandQueue*>(metal_queue_)->release();
        metal_queue_ = nullptr;
    }
    if (blas_handle_) {
        static_cast<MTL::Device*>(blas_handle_)->release();
        blas_handle_ = nullptr;
    }
    initialized_ = false;
}

GPUContext::~GPUContext() {
    finalize();
}

void GPUContext::set_deterministic(bool /*enable*/) {
    // Metal does not expose a per-queue determinism flag.
    // Determinism is enforced at the kernel level (Task 10).
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
