#pragma once
#include "core/types.hpp"
#include <cstddef>

namespace kronos::gpu {

// GPU BLAS wrapper - dispatches to cuBLAS or rocBLAS
// v0.1: stub

// ZGEMM: C = alpha*A*B + beta*C
void gemm(int m, int n, int k,
          complex_t alpha,
          const complex_t* A, int lda,
          const complex_t* B, int ldb,
          complex_t beta,
          complex_t* C, int ldc);

// Dot product: result = x^H * y
complex_t zdotc(int n, const complex_t* x, const complex_t* y);

} // namespace kronos::gpu
