#pragma once
#include "core/types.hpp"
#include <vector>
#include <array>

namespace kronos::gpu {

// GPU FFT wrapper - dispatches to cuFFT (CUDA) or rocFFT (HIP)
// v0.1: stub implementation that throws if called
class GPUFFTGrid {
public:
    GPUFFTGrid(std::array<int, 3> dims);
    ~GPUFFTGrid();

    void forward(const complex_t* d_input, complex_t* d_output);
    void inverse(const complex_t* d_input, complex_t* d_output);

    std::array<int, 3> dims() const;

private:
    std::array<int, 3> dims_;
    void* plan_forward_{nullptr};
    void* plan_inverse_{nullptr};
};

} // namespace kronos::gpu
