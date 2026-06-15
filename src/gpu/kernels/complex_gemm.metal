// ============================================================================
// KRONOS  src/gpu/kernels/complex_gemm.metal
// Templated complex GEMM: C = alpha*A*B + beta*C
// One implementation, specialized at compile time for fp32 and fp64.
//
// NOTE ON fp64 / double:
//   Apple's Metal Shading Language (all versions, all hardware as of 2026)
//   does NOT support 'double' in kernel/device code. Apple GPUs have no
//   hardware fp64 ALUs and the type is deliberately reserved in MSL.
//   Therefore:
//     - zgemm_fp32: real fp32-complex tile GEMM — used by Metal dispatcher.
//     - zgemm_fp64: fp64 inputs are WIDENED to float2 pairs on the host and
//       this kernel executes in fp32 precision. The Metal dispatcher in
//       blas_metal.cpp MUST NOT call zgemm_fp64 for production results; it
//       should fall back to CPU BLAS (MKL / OpenBLAS) for any call that
//       requires true fp64 precision. This entry exists so that the metallib
//       symbol is present and Task 12's dispatch table can detect it without
//       separate compile-time branching.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// 16x16 tiled complex GEMM over float2 pairs.
// A[row + col*lda] stores (re, im) packed as float2 — column-major layout.
// ---------------------------------------------------------------------------
inline void zgemm_tiled_f32(
    constant int&    M,
    constant int&    N,
    constant int&    K,
    constant float2& alpha,
    constant float2& beta,
    device const float2* A,
    device const float2* B,
    device       float2* C,
    constant int&    lda,
    constant int&    ldb,
    constant int&    ldc,
    uint2 gid,
    uint2 lid,
    threadgroup float2* tileA,
    threadgroup float2* tileB)
{
    constexpr int TS = 16;

    int row = int(gid.y);
    int col = int(gid.x);

    float2 acc = float2(0.0f, 0.0f);

    for (int kk = 0; kk < K; kk += TS) {
        // Cooperatively load TS×TS tile of A (rows=M, cols=K, col-major).
        int a_row = row;
        int a_col = kk + int(lid.x);
        tileA[lid.y * TS + lid.x] =
            (a_row < M && a_col < K) ? A[a_row + a_col * lda] : float2(0.0f, 0.0f);

        // Cooperatively load TS×TS tile of B (rows=K, cols=N, col-major).
        int b_row = kk + int(lid.y);
        int b_col = col;
        tileB[lid.y * TS + lid.x] =
            (b_row < K && b_col < N) ? B[b_row + b_col * ldb] : float2(0.0f, 0.0f);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < TS; ++t) {
            float2 a = tileA[lid.y * TS + t];
            float2 b = tileB[t    * TS + lid.x];
            // (a.re + i*a.im) * (b.re + i*b.im)
            acc.x += a.x * b.x - a.y * b.y;
            acc.y += a.x * b.y + a.y * b.x;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        // C[row,col] = alpha * acc + beta * C[row,col]
        float2 c  = C[row + col * ldc];
        float2 ac = float2(
            alpha.x * acc.x - alpha.y * acc.y,
            alpha.x * acc.y + alpha.y * acc.x);
        float2 bc = float2(
            beta.x * c.x - beta.y * c.y,
            beta.x * c.y + beta.y * c.x);
        C[row + col * ldc] = ac + bc;
    }
}

// ---------- fp32 entry ------------------------------------------------------
// Primary production entry. Called by blas_metal.cpp for all Metal GEMM.

kernel void zgemm_fp32(
    constant int&    M       [[buffer(0)]],
    constant int&    N       [[buffer(1)]],
    constant int&    K       [[buffer(2)]],
    constant float2& alpha   [[buffer(3)]],
    constant float2& beta    [[buffer(4)]],
    device const float2* A   [[buffer(5)]],
    device const float2* B   [[buffer(6)]],
    device       float2* C   [[buffer(7)]],
    constant int&    lda     [[buffer(8)]],
    constant int&    ldb     [[buffer(9)]],
    constant int&    ldc     [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float2 tileA[16 * 16];
    threadgroup float2 tileB[16 * 16];
    zgemm_tiled_f32(M, N, K, alpha, beta,
                    A, B, C, lda, ldb, ldc,
                    gid, lid, tileA, tileB);
}

// ---------- fp64 stub entry -------------------------------------------------
// MSL has no hardware double; this entry accepts float2 buffers (the host
// MUST narrow/cast inputs from double to float before dispatch).
// blas_metal.cpp dispatcher MUST prefer CPU BLAS for true fp64 precision.
// This symbol exists only so the dispatch table can resolve it by name.

kernel void zgemm_fp64(
    constant int&    M       [[buffer(0)]],
    constant int&    N       [[buffer(1)]],
    constant int&    K       [[buffer(2)]],
    constant float2& alpha   [[buffer(3)]],
    constant float2& beta    [[buffer(4)]],
    device const float2* A   [[buffer(5)]],
    device const float2* B   [[buffer(6)]],
    device       float2* C   [[buffer(7)]],
    constant int&    lda     [[buffer(8)]],
    constant int&    ldb     [[buffer(9)]],
    constant int&    ldc     [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    // Executes in fp32 precision. Host must have narrowed inputs to float2.
    // blas_metal.cpp should fall back to CPU BLAS for fp64-precision work.
    threadgroup float2 tileA[16 * 16];
    threadgroup float2 tileB[16 * 16];
    zgemm_tiled_f32(M, N, K, alpha, beta,
                    A, B, C, lda, ldb, ldc,
                    gid, lid, tileA, tileB);
}
