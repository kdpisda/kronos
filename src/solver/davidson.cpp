#include "solver/davidson.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <stdexcept>

// LAPACK declaration for complex Hermitian eigenvalue problem
extern "C" {
    void zheev_(const char* jobz, const char* uplo, const int* n,
                std::complex<double>* a, const int* lda, double* w,
                std::complex<double>* work, const int* lwork,
                double* rwork, int* info);
}

namespace kronos {

// ============================================================================
// Helper: inner product <a|b> = sum_i conj(a_i) * b_i
// ============================================================================
static complex_t dot_product(const CVec& a, const CVec& b) {
    assert(a.size() == b.size());
    complex_t result{0.0, 0.0};
    for (size_t i = 0; i < a.size(); ++i) {
        result += std::conj(a[i]) * b[i];
    }
    return result;
}

// ============================================================================
// Helper: norm of a complex vector
// ============================================================================
static double vec_norm(const CVec& v) {
    double s = 0.0;
    for (const auto& x : v) {
        s += std::norm(x);  // |x|^2
    }
    return std::sqrt(s);
}

// ============================================================================
// Helper: axpy  y += alpha * x
// ============================================================================
static void axpy(CVec& y, complex_t alpha, const CVec& x) {
    assert(y.size() == x.size());
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

// ============================================================================
// DavidsonSolver
// ============================================================================

DavidsonSolver::DavidsonSolver()
    : params_() {}

DavidsonSolver::DavidsonSolver(Params params)
    : params_(params) {}

void DavidsonSolver::orthogonalize(std::vector<CVec>& vectors) {
    // Modified Gram-Schmidt with reorthogonalization
    for (size_t i = 0; i < vectors.size(); ++i) {
        // Two passes for numerical stability (reorthogonalization)
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t j = 0; j < i; ++j) {
                complex_t overlap = dot_product(vectors[j], vectors[i]);
                axpy(vectors[i], -overlap, vectors[j]);
            }
        }
        // Normalize
        double nrm = vec_norm(vectors[i]);
        if (nrm > 1e-14) {
            complex_t inv_nrm{1.0 / nrm, 0.0};
            for (auto& x : vectors[i]) {
                x *= inv_nrm;
            }
        }
    }
}

EigenResult DavidsonSolver::solve_subspace(
    const std::vector<std::vector<complex_t>>& h_sub,
    int num_bands)
{
    const int m = static_cast<int>(h_sub.size());
    assert(m > 0);
    assert(num_bands <= m);

    // Pack h_sub into column-major format for LAPACK
    // LAPACK zheev expects column-major storage
    std::vector<complex_t> a_mat(m * m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            // Column-major: element (i,j) at index i + j*m
            a_mat[i + j * m] = h_sub[i][j];
        }
    }

    std::vector<double> eigenvalues(m);

    // Workspace query
    int lwork = -1;
    int info = 0;
    std::vector<double> rwork(std::max(1, 3 * m - 2));
    complex_t work_query;
    zheev_("V", "U", &m, a_mat.data(), &m, eigenvalues.data(),
           &work_query, &lwork, rwork.data(), &info);

    lwork = static_cast<int>(work_query.real());
    std::vector<complex_t> work(lwork);

    // Actual diagonalization
    zheev_("V", "U", &m, a_mat.data(), &m, eigenvalues.data(),
           work.data(), &lwork, rwork.data(), &info);

    if (info != 0) {
        throw std::runtime_error("zheev failed in Davidson subspace diagonalization, info = "
                                 + std::to_string(info));
    }

    EigenResult result;
    result.eigenvalues.assign(eigenvalues.begin(),
                              eigenvalues.begin() + num_bands);

    // Extract eigenvectors (first num_bands columns of a_mat)
    result.eigenvectors.resize(num_bands);
    for (int n = 0; n < num_bands; ++n) {
        result.eigenvectors[n].resize(m);
        for (int i = 0; i < m; ++i) {
            result.eigenvectors[n][i] = a_mat[i + n * m];
        }
    }

    return result;
}

CVec DavidsonSolver::apply_preconditioner(
    const CVec& residual,
    double eigenvalue,
    const std::vector<double>& diagonal)
{
    assert(residual.size() == diagonal.size());
    const size_t n = residual.size();
    CVec result(n);
    for (size_t i = 0; i < n; ++i) {
        double denom = diagonal[i] - eigenvalue;
        // Prevent division by zero near the eigenvalue
        // Use a small epsilon to avoid blowing up, but not so large
        // that we lose the preconditioner's effectiveness
        if (std::abs(denom) < 1e-4) {
            denom = (denom >= 0.0) ? 1e-4 : -1e-4;
        }
        result[i] = residual[i] / denom;
    }
    return result;
}

EigenResult DavidsonSolver::solve(
    const std::function<CVec(const CVec&)>& h_apply,
    const std::vector<double>& preconditioner,
    int num_bands,
    int num_pw,
    const std::vector<CVec>& initial_guess)
{
    assert(num_bands > 0);
    assert(num_pw > 0);
    assert(static_cast<int>(preconditioner.size()) == num_pw);

    int max_subspace = params_.max_subspace;
    if (max_subspace <= 0) {
        max_subspace = params_.subspace_factor * num_bands;
    }
    // Ensure max_subspace doesn't exceed problem dimension
    max_subspace = std::min(max_subspace, num_pw);

    // 1. Initialize subspace V
    std::vector<CVec> V;

    if (!initial_guess.empty()) {
        for (const auto& v : initial_guess) {
            if (static_cast<int>(V.size()) >= num_bands) break;
            V.push_back(v);
        }
    }

    // Fill remaining with random vectors if needed
    if (static_cast<int>(V.size()) < num_bands) {
        std::mt19937 rng(42);  // deterministic seed for reproducibility
        std::normal_distribution<double> dist(0.0, 1.0);

        while (static_cast<int>(V.size()) < num_bands) {
            CVec v(num_pw);
            for (int i = 0; i < num_pw; ++i) {
                v[i] = complex_t(dist(rng), dist(rng));
            }
            V.push_back(std::move(v));
        }
    }

    // Orthogonalize initial basis
    orthogonalize(V);

    // 2. Apply H to all basis vectors, store H*V
    std::vector<CVec> HV;
    HV.reserve(max_subspace);
    for (const auto& v : V) {
        HV.push_back(h_apply(v));
    }

    EigenResult final_result;
    final_result.converged = false;

    for (int iter = 0; iter < params_.max_iterations; ++iter) {
        const int m = static_cast<int>(V.size());

        // 2b. Build projected Hamiltonian H_sub[i][j] = <V_i|H|V_j>
        std::vector<std::vector<complex_t>> h_sub(m, std::vector<complex_t>(m));
        for (int i = 0; i < m; ++i) {
            for (int j = i; j < m; ++j) {
                complex_t val = dot_product(V[i], HV[j]);
                h_sub[i][j] = val;
                h_sub[j][i] = std::conj(val);
            }
        }

        // 2c. Diagonalize H_sub
        int bands_to_solve = std::min(num_bands, m);
        auto sub_result = solve_subspace(h_sub, bands_to_solve);

        // 2d. Compute Ritz vectors: psi_n = sum_i c_ni * V_i
        // and H*psi_n = sum_i c_ni * HV_i
        std::vector<CVec> ritz_vecs(bands_to_solve, CVec(num_pw, {0.0, 0.0}));
        std::vector<CVec> h_ritz_vecs(bands_to_solve, CVec(num_pw, {0.0, 0.0}));

        for (int n = 0; n < bands_to_solve; ++n) {
            for (int i = 0; i < m; ++i) {
                complex_t c = sub_result.eigenvectors[n][i];
                axpy(ritz_vecs[n], c, V[i]);
                axpy(h_ritz_vecs[n], c, HV[i]);
            }
        }

        // 2e. Compute residuals: r_n = H|psi_n> - epsilon_n * |psi_n>
        std::vector<CVec> residuals(bands_to_solve);
        double max_residual = 0.0;

        for (int n = 0; n < bands_to_solve; ++n) {
            residuals[n].resize(num_pw);
            for (int i = 0; i < num_pw; ++i) {
                residuals[n][i] = h_ritz_vecs[n][i]
                                  - sub_result.eigenvalues[n] * ritz_vecs[n][i];
            }
            double rnorm = vec_norm(residuals[n]);
            max_residual = std::max(max_residual, rnorm);
        }

        // 2f. Check convergence
        if (max_residual < params_.tolerance) {
            final_result.eigenvalues = sub_result.eigenvalues;
            final_result.eigenvectors = std::move(ritz_vecs);
            final_result.iterations = iter + 1;
            final_result.converged = true;
            final_result.max_residual = max_residual;
            return final_result;
        }

        // 2g-i. Apply preconditioner and expand subspace
        // Check if subspace is too large -> restart
        if (m + bands_to_solve > max_subspace) {
            // Restart: replace V with current Ritz vectors
            V = ritz_vecs;
            HV.clear();
            for (const auto& v : V) {
                HV.push_back(h_apply(v));
            }
            continue;
        }

        // Apply preconditioner to residuals and add to subspace
        int vectors_added = 0;
        for (int n = 0; n < bands_to_solve; ++n) {
            CVec t = apply_preconditioner(residuals[n],
                                          sub_result.eigenvalues[n],
                                          preconditioner);

            // Orthogonalize t against all current V vectors (two passes)
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& v : V) {
                    complex_t overlap = dot_product(v, t);
                    axpy(t, -overlap, v);
                }
            }

            double tnorm = vec_norm(t);
            if (tnorm > 1e-10) {
                // Normalize and add
                complex_t inv_norm{1.0 / tnorm, 0.0};
                for (auto& x : t) {
                    x *= inv_norm;
                }
                V.push_back(t);
                HV.push_back(h_apply(t));
                ++vectors_added;
            }
        }

        // If no new vectors could be added (preconditioner is too exact),
        // the subspace cannot expand further — stop iterating
        if (vectors_added == 0) {
            final_result.eigenvalues = sub_result.eigenvalues;
            final_result.eigenvectors = std::move(ritz_vecs);
            final_result.iterations = iter + 1;
            final_result.max_residual = max_residual;
            break;
        }

        // Store last result for the case we exhaust iterations
        final_result.eigenvalues = sub_result.eigenvalues;
        final_result.eigenvectors = std::move(ritz_vecs);
        final_result.iterations = iter + 1;
        final_result.max_residual = max_residual;
    }

    // Did not converge within max_iterations
    final_result.converged = false;
    return final_result;
}

} // namespace kronos
