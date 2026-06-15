// ============================================================================
// KRONOS  src/gpu/blas_metal.cpp
// Metal complex GEMM dispatcher (Apple Silicon, fp32 ONLY).
//
// Per the 2026-06-15 design pivot: Apple MSL has no fp64 support, so this
// dispatcher narrows complex<double> inputs to complex<float> when
// apple_fast_mode is enabled. When the flag is off, throws GPUNotAvailableError
// so the GPUHamiltonian fallback routes the call to CPU BLAS.
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "gpu/blas.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/memory.hpp"

#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace kronos::gpu {

// Defined in memory_metal.cpp; extern-declared here (same namespace).
extern MTL::Buffer* metal_buffer_for(const void* p);

namespace {

struct PipelineCache {
    std::mutex                 mtx;
    MTL::Library*              library  = nullptr;
    MTL::ComputePipelineState* psoFP32  = nullptr;

    static PipelineCache& instance() {
        static PipelineCache c;
        return c;
    }
};

MTL::Library* load_library(MTL::Device* device) {
    namespace fs = std::filesystem;
    // Search order: CWD and several ancestor/sibling paths.
    // The metallib is built to ${CMAKE_BINARY_DIR}/kronos.metallib.
    // Tests run with WORKING_DIRECTORY = test source dir.
    // Direct invocation may use build_metal/test/ as CWD.
    // We search broadly so tests pass regardless of how they're invoked.
    auto cwd = fs::current_path();
    std::vector<fs::path> candidates = {
        cwd / "kronos.metallib",
        cwd.parent_path() / "kronos.metallib",
        cwd.parent_path().parent_path() / "kronos.metallib",
        cwd / ".." / "kronos.metallib",
        cwd / ".." / ".." / "kronos.metallib",
        // Common build directory names relative to CWD
        cwd / "build_metal" / "kronos.metallib",
        cwd.parent_path() / "build_metal" / "kronos.metallib",
    };
    for (auto& p : candidates) {
        std::error_code ec;
        if (!fs::exists(p, ec) || ec) continue;
        NS::String* url_s = NS::String::string(
            p.string().c_str(), NS::UTF8StringEncoding);
        NS::URL* url = NS::URL::fileURLWithPath(url_s);
        NS::Error* err = nullptr;
        MTL::Library* lib = device->newLibrary(url, &err);
        if (lib) return lib;
    }
    throw std::runtime_error(
        "kronos.metallib not found; searched " + cwd.string() + " and ancestors/siblings");
}

MTL::ComputePipelineState* get_fp32_pipeline() {
    auto& c = PipelineCache::instance();
    std::lock_guard<std::mutex> g(c.mtx);

    auto& ctx = GPUContext::instance();
    MTL::Device* device = static_cast<MTL::Device*>(ctx.blas_handle());

    if (!c.library) {
        c.library = load_library(device);
    }
    if (c.psoFP32) return c.psoFP32;

    NS::String* fn_s = NS::String::string("zgemm_fp32", NS::UTF8StringEncoding);
    MTL::Function* func = c.library->newFunction(fn_s);
    if (!func) {
        throw std::runtime_error("Metal: function 'zgemm_fp32' missing from kronos.metallib");
    }
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* pso = device->newComputePipelineState(func, &err);
    func->release();
    if (!pso) {
        throw std::runtime_error("Metal: newComputePipelineState failed for zgemm_fp32");
    }
    c.psoFP32 = pso;
    return pso;
}

// Narrow a complex<double> buffer to a freshly-allocated float2 MTLBuffer.
// The returned pointer is registered in the BufferRegistry (via gpu_malloc),
// so it must be released with gpu_free.
void* narrow_to_float2(const complex_t* src, int count) {
    void* dst_ptr = gpu_malloc(static_cast<size_t>(count) * 2 * sizeof(float));
    float* dst = static_cast<float*>(dst_ptr);
    for (int i = 0; i < count; ++i) {
        dst[2*i    ] = static_cast<float>(src[i].real());
        dst[2*i + 1] = static_cast<float>(src[i].imag());
    }
    // Make CPU writes visible to the GPU.
    MTL::Buffer* b = metal_buffer_for(dst_ptr);
    if (b) {
        b->didModifyRange(NS::Range::Make(0, static_cast<size_t>(count) * 2 * sizeof(float)));
    }
    return dst_ptr;
}

// Widen float2 GPU result back to complex<double> host buffer.
void widen_from_float2(const void* src_ptr, complex_t* dst, int count) {
    const float* src = static_cast<const float*>(src_ptr);
    for (int i = 0; i < count; ++i) {
        dst[i] = complex_t{static_cast<double>(src[2*i]),
                           static_cast<double>(src[2*i + 1])};
    }
}

} // anonymous namespace

void gemm(int m, int n, int k,
          complex_t alpha,
          const complex_t* A, int lda,
          const complex_t* B, int ldb,
          complex_t beta,
          complex_t* C, int ldc)
{
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw GPUNotAvailableError("Metal context not initialized");
    }
    if (!ctx.apple_fast_mode()) {
        throw GPUNotAvailableError(
            "Metal GEMM requires apple_fast_mode=true (Apple MSL has no fp64). "
            "Caller should fall back to CPU BLAS.");
    }

    MTL::Device* device = static_cast<MTL::Device*>(ctx.blas_handle());
    MTL::CommandQueue* queue = static_cast<MTL::CommandQueue*>(ctx.metal_queue());

    MTL::ComputePipelineState* pso = get_fp32_pipeline();

    // Narrow A, B, C from complex<double> → float2 MTLBuffers.
    // Column-major: A is m×k, B is k×n, C is m×n.
    void* d_A_f32 = narrow_to_float2(A, lda * k);
    void* d_B_f32 = narrow_to_float2(B, ldb * n);
    void* d_C_f32 = narrow_to_float2(C, ldc * n);

    MTL::Buffer* bA = metal_buffer_for(d_A_f32);
    MTL::Buffer* bB = metal_buffer_for(d_B_f32);
    MTL::Buffer* bC = metal_buffer_for(d_C_f32);

    MTL::CommandBuffer* cmd = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);

    // Kernel buffer bindings match the MSL signature in complex_gemm.metal.
    enc->setBytes(&m, sizeof(int), 0);
    enc->setBytes(&n, sizeof(int), 1);
    enc->setBytes(&k, sizeof(int), 2);

    float alpha_f[2] = { float(alpha.real()), float(alpha.imag()) };
    float beta_f [2] = { float(beta.real()),  float(beta.imag())  };
    enc->setBytes(alpha_f, sizeof(alpha_f), 3);
    enc->setBytes(beta_f,  sizeof(beta_f),  4);

    enc->setBuffer(bA, 0, 5);
    enc->setBuffer(bB, 0, 6);
    enc->setBuffer(bC, 0, 7);

    enc->setBytes(&lda, sizeof(int), 8);
    enc->setBytes(&ldb, sizeof(int), 9);
    enc->setBytes(&ldc, sizeof(int), 10);

    // Dispatch a grid large enough to cover all (m × n) output elements.
    // Each threadgroup is 16×16; grid rounds up to the nearest tile boundary.
    constexpr int TS = 16;
    MTL::Size threads = MTL::Size::Make(
        static_cast<NS::UInteger>((n + TS - 1) / TS * TS),
        static_cast<NS::UInteger>((m + TS - 1) / TS * TS),
        1);
    MTL::Size group = MTL::Size::Make(TS, TS, 1);
    enc->dispatchThreads(threads, group);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    // Widen C result back to complex<double> and write into the caller's buffer.
    widen_from_float2(d_C_f32, C, ldc * n);

    gpu_free(d_A_f32);
    gpu_free(d_B_f32);
    gpu_free(d_C_f32);
}

complex_t zdotc(int /*n*/, const complex_t* /*x*/, const complex_t* /*y*/) {
    // Not used on the GPU hot path in v0.5.1. Implement when a caller needs it.
    throw GPUNotAvailableError(
        "Metal zdotc not implemented in v0.5.1 (no caller yet)");
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
