#include "basis/fft_grid.hpp"
#include "basis/plane_wave.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace kronos {

// -------------------------------------------------------------------------
// Helper: find next FFT-friendly size (only factors of 2, 3, 5)
// -------------------------------------------------------------------------

int FFTGrid::next_fft_friendly(int n) {
    if (n <= 1) return 1;
    while (true) {
        int m = n;
        while (m % 2 == 0) m /= 2;
        while (m % 3 == 0) m /= 3;
        while (m % 5 == 0) m /= 5;
        if (m == 1) return n;
        ++n;
    }
}

// -------------------------------------------------------------------------
// FFTW plan management
// -------------------------------------------------------------------------

void FFTGrid::create_plans() {
    int n0 = dims_[0];
    int n1 = dims_[1];
    int n2 = dims_[2];
    int total = n0 * n1 * n2;

    work_.resize(static_cast<size_t>(total));

    // FFTW expects fftw_complex*, which is layout-compatible with
    // std::complex<double> (guaranteed by C++ standard).
    auto* data = reinterpret_cast<fftw_complex*>(work_.data());

    plan_forward_ = fftw_plan_dft_3d(
        n0, n1, n2, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inverse_ = fftw_plan_dft_3d(
        n0, n1, n2, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (!plan_forward_ || !plan_inverse_) {
        throw std::runtime_error("FFTGrid: fftw_plan_dft_3d failed");
    }
}

void FFTGrid::destroy_plans() {
    if (plan_forward_) {
        fftw_destroy_plan(static_cast<fftw_plan>(plan_forward_));
        plan_forward_ = nullptr;
    }
    if (plan_inverse_) {
        fftw_destroy_plan(static_cast<fftw_plan>(plan_inverse_));
        plan_inverse_ = nullptr;
    }
}

// -------------------------------------------------------------------------
// Construction / destruction
// -------------------------------------------------------------------------

FFTGrid::FFTGrid(const PlaneWaveBasis& basis, double ecutrho) {
    // Scale the wavefunction max-miller indices by sqrt(ecutrho/ecutwfc).
    // PlaneWaveBasis already computes max_miller using the QE formula
    // floor(G_max_wfc / |b_i|) + 1, so scaling by 2 (for ecutrho = 4*ecutwfc)
    // gives a grid that comfortably fits all density G-vectors and reduces
    // aliasing in the V_eff × ψ product.
    auto mm_wfc = basis.max_miller();
    double scale = std::sqrt(ecutrho / basis.ecutwfc());
    for (int i = 0; i < 3; ++i) {
        int max_miller_rho = static_cast<int>(std::ceil(mm_wfc[i] * scale));
        dims_[i] = next_fft_friendly(2 * max_miller_rho + 1);
    }
    create_plans();
}

FFTGrid::FFTGrid(const PlaneWaveBasis& basis)
    : FFTGrid(basis, 4.0 * basis.ecutwfc()) {
}

FFTGrid::~FFTGrid() {
    destroy_plans();
}

FFTGrid::FFTGrid(FFTGrid&& other) noexcept
    : dims_(other.dims_),
      plan_forward_(other.plan_forward_),
      plan_inverse_(other.plan_inverse_),
      work_(std::move(other.work_))
{
    other.plan_forward_ = nullptr;
    other.plan_inverse_ = nullptr;
    other.dims_ = {0, 0, 0};
}

FFTGrid& FFTGrid::operator=(FFTGrid&& other) noexcept {
    if (this != &other) {
        destroy_plans();
        dims_ = other.dims_;
        plan_forward_ = other.plan_forward_;
        plan_inverse_ = other.plan_inverse_;
        work_ = std::move(other.work_);
        other.plan_forward_ = nullptr;
        other.plan_inverse_ = nullptr;
        other.dims_ = {0, 0, 0};
    }
    return *this;
}

// -------------------------------------------------------------------------
// Public accessors
// -------------------------------------------------------------------------

std::array<int, 3> FFTGrid::dims() const { return dims_; }

int FFTGrid::total_points() const {
    return dims_[0] * dims_[1] * dims_[2];
}

// -------------------------------------------------------------------------
// FFT operations
// -------------------------------------------------------------------------

void FFTGrid::forward(const std::vector<complex_t>& r_space,
                      std::vector<complex_t>& g_space) {
    int total = total_points();
    if (static_cast<int>(r_space.size()) != total) {
        throw std::invalid_argument(
            "FFTGrid::forward: input size does not match grid");
    }

    // Copy input to work array
    std::copy(r_space.begin(), r_space.end(), work_.begin());

    // Execute forward FFT (in-place on work_)
    fftw_execute(static_cast<fftw_plan>(plan_forward_));

    // Copy result to output
    g_space.resize(static_cast<size_t>(total));
    std::copy(work_.begin(), work_.end(), g_space.begin());
}

void FFTGrid::inverse(const std::vector<complex_t>& g_space,
                      std::vector<complex_t>& r_space) {
    int total = total_points();
    if (static_cast<int>(g_space.size()) != total) {
        throw std::invalid_argument(
            "FFTGrid::inverse: input size does not match grid");
    }

    // Copy input to work array
    std::copy(g_space.begin(), g_space.end(), work_.begin());

    // Execute inverse FFT (in-place on work_)
    fftw_execute(static_cast<fftw_plan>(plan_inverse_));

    // Normalize by 1/N and copy to output
    double inv_n = 1.0 / static_cast<double>(total);
    r_space.resize(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) {
        r_space[static_cast<size_t>(i)] =
            work_[static_cast<size_t>(i)] * inv_n;
    }
}

// -------------------------------------------------------------------------
// Index mapping
// -------------------------------------------------------------------------

int FFTGrid::gvec_to_index(int h, int k, int l) const {
    int n0 = dims_[0];
    int n1 = dims_[1];
    int n2 = dims_[2];

    // Wrap negative indices using standard FFT convention
    int h_idx = h >= 0 ? h : h + n0;
    int k_idx = k >= 0 ? k : k + n1;
    int l_idx = l >= 0 ? l : l + n2;

    return h_idx * n1 * n2 + k_idx * n2 + l_idx;
}

// -------------------------------------------------------------------------
// Scatter / gather
// -------------------------------------------------------------------------

void FFTGrid::scatter_to_grid(const PlaneWaveBasis& basis,
                              const CVec& pw_coeffs,
                              std::vector<complex_t>& grid) const {
    int total = total_points();
    grid.assign(static_cast<size_t>(total), complex_t{0.0, 0.0});

    const auto& gvecs = basis.gvectors();
    if (pw_coeffs.size() != gvecs.size()) {
        throw std::invalid_argument(
            "FFTGrid::scatter_to_grid: pw_coeffs size mismatch");
    }

    for (size_t i = 0; i < gvecs.size(); ++i) {
        int idx = gvec_to_index(gvecs[i].h, gvecs[i].k, gvecs[i].l);
        grid[static_cast<size_t>(idx)] = pw_coeffs[i];
    }
}

void FFTGrid::gather_from_grid(const PlaneWaveBasis& basis,
                               const std::vector<complex_t>& grid,
                               CVec& pw_coeffs) const {
    const auto& gvecs = basis.gvectors();
    pw_coeffs.resize(gvecs.size());

    for (size_t i = 0; i < gvecs.size(); ++i) {
        int idx = gvec_to_index(gvecs[i].h, gvecs[i].k, gvecs[i].l);
        pw_coeffs[i] = grid[static_cast<size_t>(idx)];
    }
}

} // namespace kronos
