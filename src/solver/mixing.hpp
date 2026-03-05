#pragma once
#include "core/types.hpp"
#include <vector>
#include <deque>

namespace kronos {

// Simple linear mixing: n_new = alpha * n_out + (1-alpha) * n_in
class LinearMixer {
public:
    explicit LinearMixer(double alpha = 0.3);
    RVec mix(const RVec& n_in, const RVec& n_out);
private:
    double alpha_;
};

// Pulay/DIIS mixing with configurable history
// Minimizes ||sum_i c_i R_i||^2 where R_i = n_out_i - n_in_i
class PulayMixer {
public:
    explicit PulayMixer(int max_history = 8, double alpha = 0.3);

    // Mix densities and return new input density
    RVec mix(const RVec& n_in, const RVec& n_out);

    // Reset mixing history (e.g., on restart)
    void reset();

    // Current history depth
    int history_size() const;

private:
    int max_history_;
    double alpha_;

    // History of input densities and residuals
    std::deque<RVec> density_history_;
    std::deque<RVec> residual_history_;

    // Solve DIIS linear system to get optimal coefficients
    std::vector<double> solve_diis_coefficients() const;
};

// Kerker preconditioner for metals (suppresses charge sloshing)
// Modifies the residual in G-space: R_precond(G) = R(G) * |G|^2 / (|G|^2 + q0^2)
class KerkerPreconditioner {
public:
    explicit KerkerPreconditioner(double q0 = 1.5);  // q0 in bohr^{-1}

    // Apply Kerker preconditioning to residual in G-space
    CVec apply(const CVec& residual_g, const std::vector<double>& g_norm2) const;

private:
    double q0_sq_;
};

} // namespace kronos
