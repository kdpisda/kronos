// ============================================================================
// KRONOS  src/gpu/fft_hip.cpp
// HIP FFT implementation using rocFFT
// ============================================================================

#ifdef KRONOS_GPU_HIP

#include "gpu/fft.hpp"
#include <rocfft/rocfft.h>
#include <stdexcept>
#include <string>

namespace kronos::gpu {

GPUFFTGrid::GPUFFTGrid(std::array<int, 3> dims)
    : dims_(dims)
{
    const size_t lengths[3] = {
        static_cast<size_t>(dims[0]),
        static_cast<size_t>(dims[1]),
        static_cast<size_t>(dims[2])
    };

    // Forward plan
    rocfft_plan plan_fwd = nullptr;
    rocfft_status err = rocfft_plan_create(
        &plan_fwd,
        rocfft_placement_notinplace,
        rocfft_transform_type_complex_forward,
        rocfft_precision_double,
        3, lengths, 1, nullptr);

    if (err != rocfft_status_success) {
        throw std::runtime_error("rocfft_plan_create (forward) failed");
    }
    plan_forward_ = static_cast<void*>(plan_fwd);

    // Inverse plan
    rocfft_plan plan_inv = nullptr;
    err = rocfft_plan_create(
        &plan_inv,
        rocfft_placement_notinplace,
        rocfft_transform_type_complex_inverse,
        rocfft_precision_double,
        3, lengths, 1, nullptr);

    if (err != rocfft_status_success) {
        rocfft_plan_destroy(plan_fwd);
        throw std::runtime_error("rocfft_plan_create (inverse) failed");
    }
    plan_inverse_ = static_cast<void*>(plan_inv);
}

GPUFFTGrid::~GPUFFTGrid() {
    if (plan_forward_) {
        rocfft_plan_destroy(static_cast<rocfft_plan>(plan_forward_));
    }
    if (plan_inverse_) {
        rocfft_plan_destroy(static_cast<rocfft_plan>(plan_inverse_));
    }
}

void GPUFFTGrid::forward(const complex_t* d_input, complex_t* d_output) {
    void* in_buffer[1] = {const_cast<complex_t*>(d_input)};
    void* out_buffer[1] = {d_output};

    rocfft_execution_info info = nullptr;
    rocfft_execution_info_create(&info);

    rocfft_status err = rocfft_execute(
        static_cast<rocfft_plan>(plan_forward_),
        in_buffer, out_buffer, info);

    rocfft_execution_info_destroy(info);

    if (err != rocfft_status_success) {
        throw std::runtime_error("rocfft_execute (forward) failed");
    }
}

void GPUFFTGrid::inverse(const complex_t* d_input, complex_t* d_output) {
    void* in_buffer[1] = {const_cast<complex_t*>(d_input)};
    void* out_buffer[1] = {d_output};

    rocfft_execution_info info = nullptr;
    rocfft_execution_info_create(&info);

    rocfft_status err = rocfft_execute(
        static_cast<rocfft_plan>(plan_inverse_),
        in_buffer, out_buffer, info);

    rocfft_execution_info_destroy(info);

    if (err != rocfft_status_success) {
        throw std::runtime_error("rocfft_execute (inverse) failed");
    }
}

std::array<int, 3> GPUFFTGrid::dims() const {
    return dims_;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_HIP
