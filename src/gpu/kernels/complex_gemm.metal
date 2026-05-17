// ============================================================================
// KRONOS  src/gpu/kernels/complex_gemm.metal
// Placeholder. Real complex-GEMM kernel arrives in Task 10.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

kernel void zgemm_placeholder(device const float* a [[buffer(0)]],
                              device       float* c [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
    c[gid] = a[gid];
}
