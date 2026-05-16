// ============================================================================
// KRONOS  src/gpu/blas_hip.cpp
// HIP BLAS implementation using rocBLAS
// ============================================================================

#ifdef KRONOS_GPU_HIP

#include "gpu/blas.hpp"
#include "gpu/gpu_context.hpp"
#include <rocblas/rocblas.h>
#include <stdexcept>

namespace kronos::gpu {

void gemm(int m, int n, int k,
          complex_t alpha,
          const complex_t* A, int lda,
          const complex_t* B, int ldb,
          complex_t beta,
          complex_t* C, int ldc)
{
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw std::runtime_error("GPU context not initialized for rocBLAS ZGEMM");
    }

    rocblas_handle handle = static_cast<rocblas_handle>(ctx.blas_handle());

    rocblas_double_complex rb_alpha = {alpha.real(), alpha.imag()};
    rocblas_double_complex rb_beta  = {beta.real(),  beta.imag()};

    rocblas_status err = rocblas_zgemm(handle,
        rocblas_operation_none, rocblas_operation_none,
        m, n, k,
        &rb_alpha,
        reinterpret_cast<const rocblas_double_complex*>(A), lda,
        reinterpret_cast<const rocblas_double_complex*>(B), ldb,
        &rb_beta,
        reinterpret_cast<rocblas_double_complex*>(C), ldc);

    if (err != rocblas_status_success) {
        throw std::runtime_error("rocblas_zgemm failed: " +
                                 std::to_string(static_cast<int>(err)));
    }
}

complex_t zdotc(int n, const complex_t* x, const complex_t* y) {
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw std::runtime_error("GPU context not initialized for rocBLAS Zdotc");
    }

    rocblas_handle handle = static_cast<rocblas_handle>(ctx.blas_handle());

    rocblas_double_complex result;
    rocblas_status err = rocblas_zdotc(handle, n,
        reinterpret_cast<const rocblas_double_complex*>(x), 1,
        reinterpret_cast<const rocblas_double_complex*>(y), 1,
        &result);

    if (err != rocblas_status_success) {
        throw std::runtime_error("rocblas_zdotc failed: " +
                                 std::to_string(static_cast<int>(err)));
    }

    return complex_t(result.real(), result.imag());
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_HIP
