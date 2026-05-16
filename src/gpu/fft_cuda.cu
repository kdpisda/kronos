// ============================================================================
// KRONOS  src/gpu/fft_cuda.cu
// CUDA FFT implementation using cuFFT
// ============================================================================

#ifdef KRONOS_GPU_CUDA

#include "gpu/fft.hpp"
#include <cufft.h>
#include <stdexcept>
#include <string>

namespace kronos::gpu {

GPUFFTGrid::GPUFFTGrid(std::array<int, 3> dims)
    : dims_(dims)
{
    cufftHandle plan_fwd, plan_inv;

    cufftResult err = cufftPlan3d(&plan_fwd, dims[0], dims[1], dims[2], CUFFT_Z2Z);
    if (err != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftPlan3d (forward) failed: " +
                                 std::to_string(static_cast<int>(err)));
    }
    plan_forward_ = reinterpret_cast<void*>(static_cast<uintptr_t>(plan_fwd));

    err = cufftPlan3d(&plan_inv, dims[0], dims[1], dims[2], CUFFT_Z2Z);
    if (err != CUFFT_SUCCESS) {
        cufftDestroy(plan_fwd);
        throw std::runtime_error("cufftPlan3d (inverse) failed: " +
                                 std::to_string(static_cast<int>(err)));
    }
    plan_inverse_ = reinterpret_cast<void*>(static_cast<uintptr_t>(plan_inv));
}

GPUFFTGrid::~GPUFFTGrid() {
    if (plan_forward_) {
        cufftDestroy(static_cast<cufftHandle>(
            reinterpret_cast<uintptr_t>(plan_forward_)));
    }
    if (plan_inverse_) {
        cufftDestroy(static_cast<cufftHandle>(
            reinterpret_cast<uintptr_t>(plan_inverse_)));
    }
}

void GPUFFTGrid::forward(const complex_t* d_input, complex_t* d_output) {
    cufftHandle plan = static_cast<cufftHandle>(
        reinterpret_cast<uintptr_t>(plan_forward_));

    cufftResult err = cufftExecZ2Z(plan,
        reinterpret_cast<cufftDoubleComplex*>(const_cast<complex_t*>(d_input)),
        reinterpret_cast<cufftDoubleComplex*>(d_output),
        CUFFT_FORWARD);

    if (err != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftExecZ2Z (forward) failed: " +
                                 std::to_string(static_cast<int>(err)));
    }
}

void GPUFFTGrid::inverse(const complex_t* d_input, complex_t* d_output) {
    cufftHandle plan = static_cast<cufftHandle>(
        reinterpret_cast<uintptr_t>(plan_inverse_));

    cufftResult err = cufftExecZ2Z(plan,
        reinterpret_cast<cufftDoubleComplex*>(const_cast<complex_t*>(d_input)),
        reinterpret_cast<cufftDoubleComplex*>(d_output),
        CUFFT_INVERSE);

    if (err != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftExecZ2Z (inverse) failed: " +
                                 std::to_string(static_cast<int>(err)));
    }
}

std::array<int, 3> GPUFFTGrid::dims() const {
    return dims_;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_CUDA
