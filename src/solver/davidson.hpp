#pragma once
#include "core/types.hpp"
#include <vector>
#include <functional>

namespace kronos {

// Result of eigenvalue problem
struct EigenResult {
    std::vector<double> eigenvalues;      // sorted ascending (Ry)
    std::vector<CVec> eigenvectors;       // corresponding wavefunctions in G-space
    int iterations{0};
    bool converged{false};
    double max_residual{0.0};
};

// Davidson iterative eigensolver
// Finds the lowest N eigenvalues of H|psi> = E|psi>
class DavidsonSolver {
public:
    struct Params {
        int max_iterations = 100;
        double tolerance = 1e-6;      // residual norm convergence
        int subspace_factor = 3;      // subspace size = factor * num_bands
        int max_subspace = 0;         // 0 = auto (3 * num_bands)
    };

    DavidsonSolver();
    explicit DavidsonSolver(Params params);

    // Solve for lowest num_bands eigenvalues
    // h_apply: function that applies H to a wavefunction (H|psi> in G-space)
    // preconditioner: diagonal preconditioner (typically kinetic energy based)
    // num_bands: number of eigenvalues/vectors to find
    // num_pw: number of plane waves (dimension of the problem)
    // initial_guess: optional starting vectors (if empty, random initialization)
    // s_apply: optional overlap operator S for generalized eigenvalue H|ψ⟩=ε S|ψ⟩
    EigenResult solve(
        const std::function<CVec(const CVec&)>& h_apply,
        const std::vector<double>& preconditioner,
        int num_bands,
        int num_pw,
        const std::vector<CVec>& initial_guess = {},
        const std::function<CVec(const CVec&)>& s_apply = nullptr);

private:
    Params params_;

    // Gram-Schmidt orthogonalization of vectors
    static void orthogonalize(std::vector<CVec>& vectors);

    // Solve the projected eigenvalue problem (subspace diagonalization)
    // H_sub = V^H * H * V, S_sub = V^H * S * V (identity if no S)
    // Returns eigenvalues and rotation matrix
    static EigenResult solve_subspace(
        const std::vector<std::vector<complex_t>>& h_sub,
        int num_bands);

    // Solve generalized subspace eigenvalue problem H_sub c = λ S_sub c
    static EigenResult solve_subspace_generalized(
        const std::vector<std::vector<complex_t>>& h_sub,
        const std::vector<std::vector<complex_t>>& s_sub,
        int num_bands);

    // Apply diagonal preconditioner to residual
    // P_ii = 1 / (eigenvalue - H_ii) where H_ii ~ kinetic energy
    static CVec apply_preconditioner(
        const CVec& residual,
        double eigenvalue,
        const std::vector<double>& diagonal);
};

} // namespace kronos
