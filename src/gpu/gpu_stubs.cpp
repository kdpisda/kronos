#include "gpu/fft.hpp"
#include "gpu/blas.hpp"
#include "gpu/memory.hpp"
#include "gpu/gpu_context.hpp"

namespace kronos::gpu {

static const char* GPU_NOT_AVAILABLE_MSG =
    "GPU support not compiled. Build with -DKRONOS_GPU_BACKEND=cuda or hip";

// ---------------------------------------------------------------------------
// GPUContext stubs
// ---------------------------------------------------------------------------

GPUContext& GPUContext::instance() {
    static GPUContext ctx;
    return ctx;
}

void GPUContext::init(int /*mpi_rank*/, int /*local_rank*/) {
    // No GPU available in CPU-only build
    initialized_ = false;
    num_devices_ = 0;
    device_name_ = "none (CPU-only build)";
}

void GPUContext::finalize() {
    initialized_ = false;
}

GPUContext::~GPUContext() {
    finalize();
}

void GPUContext::set_deterministic(bool /*enable*/) {
    // No-op in CPU-only build
}

// ---------------------------------------------------------------------------
// GPUFFTGrid stubs
// ---------------------------------------------------------------------------

GPUFFTGrid::GPUFFTGrid(std::array<int, 3> dims)
    : dims_(dims) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

GPUFFTGrid::~GPUFFTGrid() = default;

void GPUFFTGrid::forward(const complex_t* /*d_input*/, complex_t* /*d_output*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

void GPUFFTGrid::inverse(const complex_t* /*d_input*/, complex_t* /*d_output*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

std::array<int, 3> GPUFFTGrid::dims() const {
    return dims_;
}

// ---------------------------------------------------------------------------
// GPU BLAS stubs
// ---------------------------------------------------------------------------

void gemm(int /*m*/, int /*n*/, int /*k*/,
          complex_t /*alpha*/,
          const complex_t* /*A*/, int /*lda*/,
          const complex_t* /*B*/, int /*ldb*/,
          complex_t /*beta*/,
          complex_t* /*C*/, int /*ldc*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

complex_t zdotc(int /*n*/, const complex_t* /*x*/, const complex_t* /*y*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

// ---------------------------------------------------------------------------
// GPU memory stubs
// ---------------------------------------------------------------------------

void* gpu_malloc(size_t /*bytes*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

void gpu_free(void* /*ptr*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

void gpu_memcpy_h2d(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

void gpu_memcpy_d2h(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

void gpu_memcpy_d2d(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError(GPU_NOT_AVAILABLE_MSG);
}

bool gpu_available() {
    return false;
}

size_t gpu_memory_free() {
    return 0;
}

size_t gpu_memory_total() {
    return 0;
}

} // namespace kronos::gpu
