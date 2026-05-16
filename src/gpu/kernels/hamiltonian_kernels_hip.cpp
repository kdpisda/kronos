// ============================================================================
// KRONOS  src/gpu/kernels/hamiltonian_kernels_hip.cpp
// HIP port of custom GPU kernels for Hamiltonian operator
// ============================================================================

#ifdef KRONOS_GPU_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

namespace kronos::gpu::kernels {

// ---------------------------------------------------------------------------
// Scatter PW coefficients onto full FFT grid
// ---------------------------------------------------------------------------
__global__ void scatter_to_grid_kernel(
    const hipDoubleComplex* pw_coeffs,
    hipDoubleComplex* fft_grid,
    const int* scatter_map,
    int npw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npw) {
        int grid_idx = scatter_map[i];
        fft_grid[grid_idx] = pw_coeffs[i];
    }
}

void scatter_to_grid(const hipDoubleComplex* d_pw, hipDoubleComplex* d_grid,
                     const int* d_map, int npw, int grid_size) {
    hipMemset(d_grid, 0, grid_size * sizeof(hipDoubleComplex));
    int blockSize = 256;
    int numBlocks = (npw + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(scatter_to_grid_kernel, dim3(numBlocks), dim3(blockSize),
                       0, 0, d_pw, d_grid, d_map, npw);
}

// ---------------------------------------------------------------------------
// Gather PW coefficients from full FFT grid
// ---------------------------------------------------------------------------
__global__ void gather_from_grid_kernel(
    const hipDoubleComplex* fft_grid,
    hipDoubleComplex* pw_coeffs,
    const int* scatter_map,
    int npw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npw) {
        int grid_idx = scatter_map[i];
        pw_coeffs[i] = fft_grid[grid_idx];
    }
}

void gather_from_grid(const hipDoubleComplex* d_grid, hipDoubleComplex* d_pw,
                      const int* d_map, int npw) {
    int blockSize = 256;
    int numBlocks = (npw + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(gather_from_grid_kernel, dim3(numBlocks), dim3(blockSize),
                       0, 0, d_grid, d_pw, d_map, npw);
}

// ---------------------------------------------------------------------------
// Pointwise multiply
// ---------------------------------------------------------------------------
__global__ void pointwise_multiply_kernel(
    hipDoubleComplex* result,
    const hipDoubleComplex* a,
    const hipDoubleComplex* b,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = hipCmul(a[i], b[i]);
    }
}

void pointwise_multiply(hipDoubleComplex* d_result,
                        const hipDoubleComplex* d_a,
                        const hipDoubleComplex* d_b,
                        int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(pointwise_multiply_kernel, dim3(numBlocks), dim3(blockSize),
                       0, 0, d_result, d_a, d_b, n);
}

// ---------------------------------------------------------------------------
// Apply kinetic energy
// ---------------------------------------------------------------------------
__global__ void apply_kinetic_kernel(
    hipDoubleComplex* hpsi,
    const hipDoubleComplex* psi,
    const double* kinetic_energies,
    int npw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npw) {
        double ke = kinetic_energies[i];
        hpsi[i] = make_hipDoubleComplex(
            ke * hipCreal(psi[i]),
            ke * hipCimag(psi[i]));
    }
}

void apply_kinetic(hipDoubleComplex* d_hpsi,
                   const hipDoubleComplex* d_psi,
                   const double* d_ke,
                   int npw) {
    int blockSize = 256;
    int numBlocks = (npw + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(apply_kinetic_kernel, dim3(numBlocks), dim3(blockSize),
                       0, 0, d_hpsi, d_psi, d_ke, npw);
}

// ---------------------------------------------------------------------------
// FFT normalize
// ---------------------------------------------------------------------------
__global__ void fft_normalize_kernel(
    hipDoubleComplex* data,
    double inv_n,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = make_hipDoubleComplex(
            hipCreal(data[i]) * inv_n,
            hipCimag(data[i]) * inv_n);
    }
}

void fft_normalize(hipDoubleComplex* d_data, int n) {
    double inv_n = 1.0 / static_cast<double>(n);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(fft_normalize_kernel, dim3(numBlocks), dim3(blockSize),
                       0, 0, d_data, inv_n, n);
}

// ---------------------------------------------------------------------------
// Vector add
// ---------------------------------------------------------------------------
__global__ void vector_add_kernel(
    hipDoubleComplex* hpsi,
    const hipDoubleComplex* addition,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        hpsi[i] = hipCadd(hpsi[i], addition[i]);
    }
}

void vector_add(hipDoubleComplex* d_hpsi,
                const hipDoubleComplex* d_addition,
                int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vector_add_kernel, dim3(numBlocks), dim3(blockSize),
                       0, 0, d_hpsi, d_addition, n);
}

} // namespace kronos::gpu::kernels

#endif // KRONOS_GPU_HIP
