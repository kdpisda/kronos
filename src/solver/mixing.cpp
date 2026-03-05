#include "solver/mixing.hpp"
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace kronos {

// ============================================================================
// LinearMixer
// ============================================================================

LinearMixer::LinearMixer(double alpha)
    : alpha_(alpha) {
    assert(alpha > 0.0 && alpha <= 1.0);
}

RVec LinearMixer::mix(const RVec& n_in, const RVec& n_out) {
    assert(n_in.size() == n_out.size());
    const size_t n = n_in.size();
    RVec result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = (1.0 - alpha_) * n_in[i] + alpha_ * n_out[i];
    }
    return result;
}

// ============================================================================
// PulayMixer
// ============================================================================

PulayMixer::PulayMixer(int max_history, double alpha)
    : max_history_(max_history), alpha_(alpha) {
    assert(max_history >= 1);
    assert(alpha > 0.0 && alpha <= 1.0);
}

RVec PulayMixer::mix(const RVec& n_in, const RVec& n_out) {
    assert(n_in.size() == n_out.size());
    const size_t n = n_in.size();

    // 1. Compute residual R = n_out - n_in
    RVec residual(n);
    for (size_t i = 0; i < n; ++i) {
        residual[i] = n_out[i] - n_in[i];
    }

    // 2. Store in history, trim to max_history
    density_history_.push_back(n_in);
    residual_history_.push_back(residual);

    while (static_cast<int>(density_history_.size()) > max_history_) {
        density_history_.pop_front();
        residual_history_.pop_front();
    }

    // 3. If only one entry, do simple linear mixing
    if (density_history_.size() == 1) {
        RVec result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = n_in[i] + alpha_ * residual[i];
        }
        return result;
    }

    // 4. Solve DIIS system for optimal coefficients
    std::vector<double> coeffs = solve_diis_coefficients();

    // 5. Construct optimal density: n_mixed = sum_i c_i * (n_in_i + alpha * R_i)
    RVec result(n, 0.0);
    for (size_t j = 0; j < density_history_.size(); ++j) {
        for (size_t i = 0; i < n; ++i) {
            result[i] += coeffs[j] * (density_history_[j][i]
                                       + alpha_ * residual_history_[j][i]);
        }
    }

    return result;
}

void PulayMixer::reset() {
    density_history_.clear();
    residual_history_.clear();
}

int PulayMixer::history_size() const {
    return static_cast<int>(density_history_.size());
}

std::vector<double> PulayMixer::solve_diis_coefficients() const {
    const int m = static_cast<int>(residual_history_.size());
    assert(m >= 2);

    // Build the augmented DIIS matrix:
    //   [B  1] [c]   [0]
    //   [1  0] [L] = [1]
    //
    // where B_ij = <R_i|R_j> = sum_k R_i[k]*R_j[k]
    // System dimension is (m+1) x (m+1)

    const int dim = m + 1;
    // Store matrix in row-major flat vector
    std::vector<double> A(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);

    // Fill B_ij
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double dot = 0.0;
            const size_t n = residual_history_[i].size();
            for (size_t k = 0; k < n; ++k) {
                dot += residual_history_[i][k] * residual_history_[j][k];
            }
            A[i * dim + j] = dot;
            A[j * dim + i] = dot;
        }
    }

    // Fill constraint row/column: last row and last column are 1 (except corner)
    for (int i = 0; i < m; ++i) {
        A[i * dim + m] = 1.0;
        A[m * dim + i] = 1.0;
    }
    A[m * dim + m] = 0.0;

    // RHS: [0, 0, ..., 0, 1]
    rhs[m] = 1.0;

    // Solve via Gaussian elimination with partial pivoting
    // Augment A with rhs column for in-place solve
    std::vector<double> aug(dim * (dim + 1));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            aug[i * (dim + 1) + j] = A[i * dim + j];
        }
        aug[i * (dim + 1) + dim] = rhs[i];
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < dim; ++col) {
        // Find pivot
        int pivot_row = col;
        double max_val = std::abs(aug[col * (dim + 1) + col]);
        for (int row = col + 1; row < dim; ++row) {
            double val = std::abs(aug[row * (dim + 1) + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = row;
            }
        }

        // Swap rows
        if (pivot_row != col) {
            for (int j = 0; j <= dim; ++j) {
                std::swap(aug[col * (dim + 1) + j],
                          aug[pivot_row * (dim + 1) + j]);
            }
        }

        double pivot = aug[col * (dim + 1) + col];
        if (std::abs(pivot) < 1e-15) {
            // Singular matrix fallback: use only the most recent entry
            // (equivalent to simple linear mixing on current step).
            // Equal coefficients would destroy converged results.
            std::vector<double> coeffs(m, 0.0);
            coeffs[m - 1] = 1.0;
            return coeffs;
        }

        // Eliminate below
        for (int row = col + 1; row < dim; ++row) {
            double factor = aug[row * (dim + 1) + col] / pivot;
            for (int j = col; j <= dim; ++j) {
                aug[row * (dim + 1) + j] -= factor * aug[col * (dim + 1) + j];
            }
        }
    }

    // Back substitution
    std::vector<double> x(dim, 0.0);
    for (int i = dim - 1; i >= 0; --i) {
        double sum = aug[i * (dim + 1) + dim];
        for (int j = i + 1; j < dim; ++j) {
            sum -= aug[i * (dim + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (dim + 1) + i];
    }

    // Return first m elements (the coefficients; x[m] is the Lagrange multiplier)
    return std::vector<double>(x.begin(), x.begin() + m);
}

// ============================================================================
// KerkerPreconditioner
// ============================================================================

KerkerPreconditioner::KerkerPreconditioner(double q0)
    : q0_sq_(q0 * q0) {}

CVec KerkerPreconditioner::apply(const CVec& residual_g,
                                  const std::vector<double>& g_norm2) const {
    assert(residual_g.size() == g_norm2.size());
    const size_t n = residual_g.size();
    CVec result(n);
    for (size_t i = 0; i < n; ++i) {
        // Kerker filter: |G|^2 / (|G|^2 + q0^2)
        // At G=0 this gives 0 (suppresses uniform charge oscillation)
        double factor = g_norm2[i] / (g_norm2[i] + q0_sq_);
        result[i] = residual_g[i] * factor;
    }
    return result;
}

} // namespace kronos
