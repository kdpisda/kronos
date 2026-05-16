// ============================================================================
// KRONOS  src/gpu/blas_cuda.cu
// CUDA BLAS implementation using cuBLAS
// ============================================================================

#ifdef KRONOS_GPU_CUDA

#include "gpu/blas.hpp"
#include "gpu/gpu_context.hpp"
#include <cublas_v2.h>
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
        throw std::runtime_error("GPU context not initialized for cuBLAS ZGEMM");
    }

    cublasHandle_t handle = static_cast<cublasHandle_t>(ctx.blas_handle());

    cuDoubleComplex cu_alpha = {alpha.real(), alpha.imag()};
    cuDoubleComplex cu_beta  = {beta.real(),  beta.imag()};

    cublasStatus_t err = cublasZgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &cu_alpha,
        reinterpret_cast<const cuDoubleComplex*>(A), lda,
        reinterpret_cast<const cuDoubleComplex*>(B), ldb,
        &cu_beta,
        reinterpret_cast<cuDoubleComplex*>(C), ldc);

    if (err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasZgemm failed: " +
                                 std::to_string(static_cast<int>(err)));
    }
}

complex_t zdotc(int n, const complex_t* x, const complex_t* y) {
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw std::runtime_error("GPU context not initialized for cuBLAS Zdotc");
    }

    cublasHandle_t handle = static_cast<cublasHandle_t>(ctx.blas_handle());

    cuDoubleComplex result;
    cublasStatus_t err = cublasZdotc(handle, n,
        reinterpret_cast<const cuDoubleComplex*>(x), 1,
        reinterpret_cast<const cuDoubleComplex*>(y), 1,
        &result);

    if (err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasZdotc failed: " +
                                 std::to_string(static_cast<int>(err)));
    }

    return complex_t(result.x, result.y);
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_CUDA
