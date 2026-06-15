// ============================================================================
// KRONOS  src/gpu/fft_metal.cpp
// 3D complex FFT on Apple GPU via VkFFT (Metal backend), fp32 ONLY.
//
// Per the 2026-06-15 pivot: Apple MSL has no fp64. This wrapper narrows
// complex<double> inputs to complex<float> at the device boundary when
// apple_fast_mode is enabled. When the flag is off, the constructor throws
// GPUNotAvailableError so GPUHamiltonian falls back to CPU FFTW.
//
// Design:
//   - Constructor initializes a VkFFTApplication (compiles MSL kernels).
//   - forward()/inverse() allocate a temporary fp32 scratch MTLBuffer,
//     narrow the complex<double> input into it, run VkFFT in-place, then
//     widen the result back to complex<double>.
//   - The VkFFT buffer is supplied per-call via VkFFTLaunchParams::buffer
//     (which overrides cfg.buffer at run time — see VkFFTCheckUpdateBufferSet).
// ============================================================================

#ifdef KRONOS_GPU_METAL

// vkFFT.h (when VKFFT_BACKEND==5) defines NS/CA/MTL_PRIVATE_IMPLEMENTATION and
// includes Foundation.hpp, QuartzCore.hpp, Metal.hpp — emitting the metal-cpp
// private selector/class implementations in this translation unit. Do NOT
// include these metal-cpp headers before vkFFT.h, and do NOT define the impl
// macros in any other translation unit.
#include "vkFFT.h"

#include "gpu/fft.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/memory.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace kronos::gpu {

// Defined in memory_metal.cpp; extern-declared here (same namespace).
extern MTL::Buffer* metal_buffer_for(const void* p);

namespace {

// VkFFT state owned by GPUFFTGrid.
struct VkFFTState {
    VkFFTApplication   app{};
    VkFFTConfiguration cfg{};
    std::array<int, 3> dims{};
    bool               initialized = false;
};

// Allocate a fp32 device buffer (via gpu_malloc) and narrow N complex<double>
// values into it. Returns the raw pointer (registered in BufferRegistry).
void* narrow_to_float2(const complex_t* src, int count) {
    const size_t bytes = static_cast<size_t>(count) * 2 * sizeof(float);
    void* dst_ptr = gpu_malloc(bytes);
    float* dst = static_cast<float*>(dst_ptr);
    for (int i = 0; i < count; ++i) {
        dst[2*i    ] = static_cast<float>(src[i].real());
        dst[2*i + 1] = static_cast<float>(src[i].imag());
    }
    // Mark the entire buffer dirty so GPU sees the CPU writes.
    MTL::Buffer* b = metal_buffer_for(dst_ptr);
    if (b) b->didModifyRange(NS::Range::Make(0, bytes));
    return dst_ptr;
}

// Widen N float2 values from a device buffer back to complex<double>.
void widen_from_float2(const void* src_ptr, complex_t* dst, int count) {
    const float* src = static_cast<const float*>(src_ptr);
    for (int i = 0; i < count; ++i) {
        dst[i] = complex_t{static_cast<double>(src[2*i]),
                           static_cast<double>(src[2*i + 1])};
    }
}

// Run VkFFT in-place on d_f32 (a fp32 MTLBuffer).
// direction: -1 = forward, +1 = inverse (VkFFT convention).
void run_vkfft_inplace(VkFFTState* st, void* d_f32, int direction) {
    MTL::Buffer* buf = metal_buffer_for(d_f32);
    if (!buf) throw std::runtime_error("run_vkfft_inplace: pointer not from gpu_malloc");

    auto& ctx = GPUContext::instance();
    MTL::CommandQueue* queue = static_cast<MTL::CommandQueue*>(ctx.metal_queue());

    MTL::CommandBuffer* cmd  = queue->commandBuffer();
    if (!cmd) throw std::runtime_error("VkFFT Metal: commandBuffer() returned null");

    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    if (!enc) throw std::runtime_error("VkFFT Metal: computeCommandEncoder() returned null");

    VkFFTLaunchParams lp{};
    lp.commandBuffer  = cmd;
    lp.commandEncoder = enc;
    lp.buffer         = &buf;   // MTL::Buffer** — pointer to the buffer pointer

    VkFFTResult r = VkFFTAppend(&st->app, direction, &lp);
    // VkFFT has encoded all compute commands into enc; seal it.
    enc->endEncoding();
    if (r != VKFFT_SUCCESS) {
        cmd->commit();   // commit so the command buffer doesn't leak
        throw std::runtime_error(
            std::string("VkFFTAppend failed (dir=") + std::to_string(direction) +
            "): error " + std::to_string(static_cast<int>(r)));
    }
    cmd->commit();
    cmd->waitUntilCompleted();

    enc->release();
    cmd->release();
}

} // anonymous namespace

// ----------------------------------------------------------------------------
// GPUFFTGrid
// ----------------------------------------------------------------------------

GPUFFTGrid::GPUFFTGrid(std::array<int, 3> dims)
    : dims_(dims)
{
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw GPUNotAvailableError("Metal context not initialized for FFT");
    }
    if (!ctx.apple_fast_mode()) {
        throw GPUNotAvailableError(
            "Metal FFT requires apple_fast_mode=true (Apple MSL has no fp64). "
            "Caller should fall back to CPU FFTW.");
    }

    VkFFTState* st = new VkFFTState();
    st->dims = dims;

    MTL::Device*       device = static_cast<MTL::Device*>(ctx.blas_handle());
    MTL::CommandQueue* queue  = static_cast<MTL::CommandQueue*>(ctx.metal_queue());

    st->cfg.FFTdim = 3;
    st->cfg.size[0] = static_cast<uint64_t>(dims[0]);
    st->cfg.size[1] = static_cast<uint64_t>(dims[1]);
    st->cfg.size[2] = static_cast<uint64_t>(dims[2]);
    st->cfg.device  = device;
    st->cfg.queue   = queue;
    st->cfg.doublePrecision = 0u;   // fp32 only — Apple MSL has no fp64
    st->cfg.performR2C      = 0u;   // complex-to-complex

    VkFFTResult r = initializeVkFFT(&st->app, st->cfg);
    if (r != VKFFT_SUCCESS) {
        delete st;
        throw std::runtime_error(
            "initializeVkFFT failed: error " + std::to_string(static_cast<int>(r)));
    }
    st->initialized = true;
    plan_forward_ = static_cast<void*>(st);
    plan_inverse_ = nullptr;  // VkFFT uses one app for both directions
}

GPUFFTGrid::~GPUFFTGrid() {
    if (plan_forward_) {
        VkFFTState* st = static_cast<VkFFTState*>(plan_forward_);
        if (st->initialized) deleteVkFFT(&st->app);
        delete st;
        plan_forward_ = nullptr;
    }
}

void GPUFFTGrid::forward(const complex_t* d_input, complex_t* d_output) {
    VkFFTState* st = static_cast<VkFFTState*>(plan_forward_);
    const int N = st->dims[0] * st->dims[1] * st->dims[2];

    // Narrow complex<double> → float2 scratch buffer.
    void* d_f32 = narrow_to_float2(d_input, N);

    // In-place forward FFT on the scratch buffer (direction = -1).
    run_vkfft_inplace(st, d_f32, /*inverse=*/-1);

    // Widen float2 result → complex<double> output.
    widen_from_float2(d_f32, d_output, N);

    gpu_free(d_f32);
}

void GPUFFTGrid::inverse(const complex_t* d_input, complex_t* d_output) {
    VkFFTState* st = static_cast<VkFFTState*>(plan_forward_);
    const int N = st->dims[0] * st->dims[1] * st->dims[2];

    // Narrow complex<double> → float2 scratch buffer.
    void* d_f32 = narrow_to_float2(d_input, N);

    // In-place inverse FFT on the scratch buffer (direction = +1).
    run_vkfft_inplace(st, d_f32, /*inverse=*/+1);

    // Widen float2 result → complex<double> output.
    widen_from_float2(d_f32, d_output, N);

    gpu_free(d_f32);
}

std::array<int, 3> GPUFFTGrid::dims() const {
    return dims_;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
