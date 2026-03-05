#pragma once
#include "core/types.hpp"
#include <vector>
#include <array>
#include <memory>

// Forward declare FFTW types to avoid including fftw3.h in header
// (implementation includes fftw3.h)

namespace kronos {

class PlaneWaveBasis;

// FFT grid dimensions and operations
// Handles forward (r->G) and inverse (G->r) 3D FFTs
class FFTGrid {
public:
    // Construct FFT grid sized for the density cutoff ecutrho (in Ry).
    // The grid accommodates G-vectors up to G_max = sqrt(2*ecutrho),
    // which is needed for alias-free density computation (density = |psi|^2).
    FFTGrid(const PlaneWaveBasis& basis, double ecutrho);

    // Backward-compatible constructor: uses ecutrho = 4*ecutwfc
    explicit FFTGrid(const PlaneWaveBasis& basis);

    ~FFTGrid();

    // Non-copyable (FFTW plans)
    FFTGrid(const FFTGrid&) = delete;
    FFTGrid& operator=(const FFTGrid&) = delete;
    FFTGrid(FFTGrid&&) noexcept;
    FFTGrid& operator=(FFTGrid&&) noexcept;

    // Grid dimensions
    std::array<int, 3> dims() const;
    int total_points() const;  // n1 * n2 * n3

    // Forward FFT: real-space -> G-space
    // Input: real-space values on grid (size = total_points)
    // Output: G-space coefficients (size = total_points, complex)
    void forward(const std::vector<complex_t>& r_space,
                 std::vector<complex_t>& g_space);

    // Inverse FFT: G-space -> real-space
    // Input: G-space coefficients (size = total_points, complex)
    // Output: real-space values (size = total_points)
    void inverse(const std::vector<complex_t>& g_space,
                 std::vector<complex_t>& r_space);

    // Map a G-vector (h,k,l) to its linear index in the FFT grid
    // Uses standard FFT convention: negative indices wrap around
    int gvec_to_index(int h, int k, int l) const;

    // Scatter plane-wave coefficients onto full FFT grid
    void scatter_to_grid(const PlaneWaveBasis& basis,
                         const CVec& pw_coeffs,
                         std::vector<complex_t>& grid) const;

    // Gather plane-wave coefficients from full FFT grid
    void gather_from_grid(const PlaneWaveBasis& basis,
                          const std::vector<complex_t>& grid,
                          CVec& pw_coeffs) const;

private:
    std::array<int, 3> dims_{};

    // FFTW plan handles (stored as void* to avoid fftw3.h in header)
    void* plan_forward_{nullptr};
    void* plan_inverse_{nullptr};

    // Work arrays for FFTW (FFTW operates in-place or needs aligned arrays)
    std::vector<complex_t> work_;

    static int next_fft_friendly(int n);  // round up to product of 2,3,5
    void create_plans();
    void destroy_plans();
};

} // namespace kronos
