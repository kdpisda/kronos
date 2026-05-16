// ============================================================================
// KRONOS  src/gpu/kernels/hamiltonian_kernels.cu
// Custom CUDA kernels for the GPU Hamiltonian operator
//
// Kernels:
//   - scatter_to_grid:   PW coefficients → FFT grid
//   - gather_from_grid:  FFT grid → PW coefficients
//   - pointwise_multiply: V_eff(r) * ψ(r) in real space
//   - apply_kinetic:     T|ψ⟩ = |k+G|² × ψ_G
//   - fft_normalize:     Divide by N after inverse FFT
// ============================================================================

#ifdef KRONOS_GPU_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace kronos::gpu::kernels {

// ---------------------------------------------------------------------------
// Scatter PW coefficients onto full FFT grid
// ---------------------------------------------------------------------------
__global__ void scatter_to_grid_kernel(
    const cuDoubleComplex* pw_coeffs,    // Input: PW basis coefficients
    cuDoubleComplex* fft_grid,           // Output: Full FFT grid (zeroed)
    const int* scatter_map,              // Index map: PW → FFT grid index
    int npw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npw) {
        int grid_idx = scatter_map[i];
        fft_grid[grid_idx] = pw_coeffs[i];
    }
}

void scatter_to_grid(const cuDoubleComplex* d_pw, cuDoubleComplex* d_grid,
                     const int* d_map, int npw, int grid_size) {
    // Zero the grid first
    cudaMemset(d_grid, 0, grid_size * sizeof(cuDoubleComplex));

    int blockSize = 256;
    int numBlocks = (npw + blockSize - 1) / blockSize;
    scatter_to_grid_kernel<<<numBlocks, blockSize>>>(d_pw, d_grid, d_map, npw);
}

// ---------------------------------------------------------------------------
// Gather PW coefficients from full FFT grid
// ---------------------------------------------------------------------------
__global__ void gather_from_grid_kernel(
    const cuDoubleComplex* fft_grid,     // Input: Full FFT grid
    cuDoubleComplex* pw_coeffs,          // Output: PW basis coefficients
    const int* scatter_map,              // Index map: PW → FFT grid index
    int npw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npw) {
        int grid_idx = scatter_map[i];
        pw_coeffs[i] = fft_grid[grid_idx];
    }
}

void gather_from_grid(const cuDoubleComplex* d_grid, cuDoubleComplex* d_pw,
                      const int* d_map, int npw) {
    int blockSize = 256;
    int numBlocks = (npw + blockSize - 1) / blockSize;
    gather_from_grid_kernel<<<numBlocks, blockSize>>>(d_grid, d_pw, d_map, npw);
}

// ---------------------------------------------------------------------------
// Pointwise multiply: result[i] = a[i] * b[i]
// Used for V_eff(r) * ψ(r) in real space
// ---------------------------------------------------------------------------
__global__ void pointwise_multiply_kernel(
    cuDoubleComplex* result,
    const cuDoubleComplex* a,
    const cuDoubleComplex* b,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = cuCmul(a[i], b[i]);
    }
}

void pointwise_multiply(cuDoubleComplex* d_result,
                        const cuDoubleComplex* d_a,
                        const cuDoubleComplex* d_b,
                        int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    pointwise_multiply_kernel<<<numBlocks, blockSize>>>(d_result, d_a, d_b, n);
}

// ---------------------------------------------------------------------------
// Apply kinetic energy: hpsi[i] = ke[i] * psi[i]
// ---------------------------------------------------------------------------
__global__ void apply_kinetic_kernel(
    cuDoubleComplex* hpsi,
    const cuDoubleComplex* psi,
    const double* kinetic_energies,
    int npw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npw) {
        double ke = kinetic_energies[i];
        hpsi[i] = make_cuDoubleComplex(
            ke * cuCreal(psi[i]),
            ke * cuCimag(psi[i]));
    }
}

void apply_kinetic(cuDoubleComplex* d_hpsi,
                   const cuDoubleComplex* d_psi,
                   const double* d_ke,
                   int npw) {
    int blockSize = 256;
    int numBlocks = (npw + blockSize - 1) / blockSize;
    apply_kinetic_kernel<<<numBlocks, blockSize>>>(d_hpsi, d_psi, d_ke, npw);
}

// ---------------------------------------------------------------------------
// FFT normalize: data[i] /= N (after inverse FFT, cuFFT is unnormalized)
// ---------------------------------------------------------------------------
__global__ void fft_normalize_kernel(
    cuDoubleComplex* data,
    double inv_n,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = make_cuDoubleComplex(
            cuCreal(data[i]) * inv_n,
            cuCimag(data[i]) * inv_n);
    }
}

void fft_normalize(cuDoubleComplex* d_data, int n) {
    double inv_n = 1.0 / static_cast<double>(n);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    fft_normalize_kernel<<<numBlocks, blockSize>>>(d_data, inv_n, n);
}

// ---------------------------------------------------------------------------
// Add vectors: hpsi[i] += vloc_psi[i]
// ---------------------------------------------------------------------------
__global__ void vector_add_kernel(
    cuDoubleComplex* hpsi,
    const cuDoubleComplex* addition,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        hpsi[i] = cuCadd(hpsi[i], addition[i]);
    }
}

void vector_add(cuDoubleComplex* d_hpsi,
                const cuDoubleComplex* d_addition,
                int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add_kernel<<<numBlocks, blockSize>>>(d_hpsi, d_addition, n);
}

} // namespace kronos::gpu::kernels

#endif // KRONOS_GPU_CUDA
