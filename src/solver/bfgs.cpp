#include "solver/bfgs.hpp"
#include "core/constants.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace kronos {

BFGSOptimizer::BFGSOptimizer()
    : params_() {}

BFGSOptimizer::BFGSOptimizer(Params params)
    : params_(params) {}

std::vector<double> BFGSOptimizer::positions_to_vector(const Crystal& crystal) {
    const auto& atoms = crystal.atoms();
    std::vector<double> pos(3 * atoms.size());
    for (size_t i = 0; i < atoms.size(); ++i) {
        Vec3 cart = crystal.frac_to_cart(atoms[i].position);
        pos[3 * i + 0] = cart[0];
        pos[3 * i + 1] = cart[1];
        pos[3 * i + 2] = cart[2];
    }
    return pos;
}

Crystal BFGSOptimizer::update_positions(const Crystal& crystal,
                                         const std::vector<double>& pos_cart) {
    auto atoms = crystal.atoms();
    assert(pos_cart.size() == 3 * atoms.size());

    for (size_t i = 0; i < atoms.size(); ++i) {
        Vec3 cart = {pos_cart[3 * i], pos_cart[3 * i + 1], pos_cart[3 * i + 2]};
        atoms[i].position = crystal.cart_to_frac(cart);
    }

    return Crystal(crystal.lattice(), std::move(atoms));
}

std::vector<double> BFGSOptimizer::forces_to_vector(const std::vector<Vec3>& forces) {
    std::vector<double> f(3 * forces.size());
    for (size_t i = 0; i < forces.size(); ++i) {
        f[3 * i + 0] = forces[i][0];
        f[3 * i + 1] = forces[i][1];
        f[3 * i + 2] = forces[i][2];
    }
    return f;
}

double BFGSOptimizer::max_force(const std::vector<Vec3>& forces) {
    double max_f = 0.0;
    for (const auto& f : forces) {
        for (int d = 0; d < 3; ++d) {
            max_f = std::max(max_f, std::abs(f[d]));
        }
    }
    return max_f;
}

void BFGSOptimizer::update_inverse_hessian(
    std::vector<double>& H,
    const std::vector<double>& s,
    const std::vector<double>& y,
    int N)
{
    // BFGS update: H_{k+1} = (I - rho*s*y^T) * H_k * (I - rho*y*s^T) + rho*s*s^T
    // where rho = 1 / (y^T * s)

    double ys = 0.0;
    for (int i = 0; i < N; ++i) {
        ys += y[i] * s[i];
    }

    // Skip update if curvature condition not satisfied
    if (ys < 1e-10) return;

    double rho = 1.0 / ys;

    // Compute H * y
    std::vector<double> Hy(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Hy[i] += H[i * N + j] * y[j];
        }
    }

    // Compute y^T * H * y
    double yHy = 0.0;
    for (int i = 0; i < N; ++i) {
        yHy += y[i] * Hy[i];
    }

    // Update H: H += (ys + y^T*H*y) * rho^2 * s*s^T - rho * (H*y*s^T + s*y^T*H)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            H[i * N + j] += (ys + yHy) * rho * rho * s[i] * s[j]
                            - rho * (Hy[i] * s[j] + s[i] * Hy[j]);
        }
    }
}

void BFGSOptimizer::limit_step(std::vector<double>& step, double max_step) {
    // Find the maximum displacement per atom
    const size_t natoms = step.size() / 3;
    double max_disp = 0.0;
    for (size_t i = 0; i < natoms; ++i) {
        double disp = 0.0;
        for (int d = 0; d < 3; ++d) {
            disp += step[3 * i + d] * step[3 * i + d];
        }
        max_disp = std::max(max_disp, std::sqrt(disp));
    }

    // Scale if any atom moves more than max_step
    if (max_disp > max_step) {
        double scale = max_step / max_disp;
        for (auto& s : step) {
            s *= scale;
        }
    }
}

RelaxResult BFGSOptimizer::optimize(
    Crystal crystal,
    const CalculationParams& calc_params,
    const ConvergenceParams& conv_params,
    const std::map<std::string, PseudoPotential>& pseudopotentials)
{
    RelaxResult result;
    const int natoms = static_cast<int>(crystal.num_atoms());
    const int N = 3 * natoms;  // degrees of freedom

    // Initialize inverse Hessian to identity scaled by initial_step
    std::vector<double> H(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        H[i * N + i] = params_.initial_step;
    }

    std::vector<double> prev_pos;
    std::vector<double> prev_grad;
    double prev_energy = 0.0;

    std::printf("\n");
    std::printf("  BFGS Geometry Optimization\n");
    std::printf("  ══════════════════════════════════════════════════════════════\n");
    std::printf("  Step  Energy (Ry)        ΔE (Ry)        Max Force (Ry/bohr)\n");
    std::printf("  ──────────────────────────────────────────────────────────────\n");

    for (int step = 0; step < params_.max_steps; ++step) {
        // Run SCF for current geometry
        SCFSolver scf(crystal, calc_params, conv_params, pseudopotentials);
        auto scf_result = scf.solve();

        if (!scf_result.converged) {
            Logger::instance().warning("BFGS", "SCF did not converge at ionic step " +
                         std::to_string(step + 1));
        }

        double energy = scf_result.total_energy_ry;
        double max_f = max_force(scf_result.forces);
        double de = (step == 0) ? energy : energy - prev_energy;

        result.energy_history.push_back(energy);
        result.force_history.push_back(max_f);

        std::printf("  %4d  %16.8f  %14.6e  %14.6e\n",
                    step + 1, energy, de, max_f);

        // Check convergence
        if (max_f < params_.force_threshold &&
            (step == 0 || std::abs(de) < params_.energy_threshold)) {
            std::printf("  ──────────────────────────────────────────────────────────────\n");
            std::printf("  Converged: max force < %.1e Ry/bohr\n",
                        params_.force_threshold);

            result.converged = true;
            result.relax_steps = step + 1;
            result.final_energy_ry = energy;
            result.final_energy_ev = energy * constants::rydberg_to_ev;
            result.max_force_ry_bohr = max_f;
            result.final_crystal = crystal;
            result.final_scf = std::move(scf_result);
            return result;
        }

        // Current positions and gradient (negative of forces)
        std::vector<double> pos = positions_to_vector(crystal);
        std::vector<double> forces_vec = forces_to_vector(scf_result.forces);
        std::vector<double> grad(N);
        for (int i = 0; i < N; ++i) {
            grad[i] = -forces_vec[i];
        }

        // BFGS update of inverse Hessian (skip first step)
        if (step > 0) {
            std::vector<double> s(N), y(N);
            for (int i = 0; i < N; ++i) {
                s[i] = pos[i] - prev_pos[i];
                y[i] = grad[i] - prev_grad[i];
            }
            update_inverse_hessian(H, s, y, N);
        }

        // Compute search direction: p = -H * grad
        std::vector<double> step_vec(N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                step_vec[i] -= H[i * N + j] * grad[j];
            }
        }

        // Limit step size
        limit_step(step_vec, params_.max_step);

        // Save for next BFGS update
        prev_pos = pos;
        prev_grad = grad;
        prev_energy = energy;

        // Update positions
        std::vector<double> new_pos(N);
        for (int i = 0; i < N; ++i) {
            new_pos[i] = pos[i] + step_vec[i];
        }

        crystal = update_positions(crystal, new_pos);
    }

    // Did not converge
    std::printf("  ──────────────────────────────────────────────────────────────\n");
    std::printf("  NOT CONVERGED after %d steps\n", params_.max_steps);

    result.converged = false;
    result.relax_steps = params_.max_steps;
    result.final_energy_ry = result.energy_history.back();
    result.final_energy_ev = result.final_energy_ry * constants::rydberg_to_ev;
    result.max_force_ry_bohr = result.force_history.back();
    result.final_crystal = crystal;
    return result;
}

} // namespace kronos
