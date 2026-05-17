// ============================================================================
// KRONOS  src/gpu/blas_metal.cpp
// Metal complex GEMM (Apple Silicon)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include "gpu/blas.hpp"
#include "gpu/memory.hpp"

namespace kronos::gpu {

void gemm(int /*m*/, int /*n*/, int /*k*/,
          complex_t /*alpha*/,
          const complex_t* /*A*/, int /*lda*/,
          const complex_t* /*B*/, int /*ldb*/,
          complex_t /*beta*/,
          complex_t* /*C*/, int /*ldc*/) {
    throw GPUNotAvailableError("metal gemm not yet implemented (Task 12)");
}

complex_t zdotc(int /*n*/, const complex_t* /*x*/, const complex_t* /*y*/) {
    throw GPUNotAvailableError("metal zdotc not yet implemented (Task 12)");
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
