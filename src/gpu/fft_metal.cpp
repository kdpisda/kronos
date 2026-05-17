// ============================================================================
// KRONOS  src/gpu/fft_metal.cpp
// Metal 3D complex FFT (via VkFFT)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include "gpu/fft.hpp"
#include "gpu/memory.hpp"

namespace kronos::gpu {

GPUFFTGrid::GPUFFTGrid(std::array<int, 3> dims)
    : dims_(dims) {
    throw GPUNotAvailableError("metal FFT not yet implemented (Task 13)");
}

GPUFFTGrid::~GPUFFTGrid() = default;

void GPUFFTGrid::forward(const complex_t* /*d_input*/, complex_t* /*d_output*/) {
    throw GPUNotAvailableError("metal FFT not yet implemented (Task 13)");
}

void GPUFFTGrid::inverse(const complex_t* /*d_input*/, complex_t* /*d_output*/) {
    throw GPUNotAvailableError("metal FFT not yet implemented (Task 13)");
}

std::array<int, 3> GPUFFTGrid::dims() const {
    return dims_;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
