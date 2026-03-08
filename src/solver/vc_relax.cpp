#include "solver/vc_relax.hpp"
#include "core/constants.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace kronos {

// Conversion factor: 1 Ry/bohr^3 = 14710.507 GPa
static constexpr double RY_BOHR3_TO_GPA = 14710.507;
// 1 GPa = 10 kbar
static constexpr double GPA_TO_KBAR = 10.0;

VCRelaxOptimizer::VCRelaxOptimizer()
    : params_() {}

VCRelaxOptimizer::VCRelaxOptimizer(Params params)
    : params_(params) {}

std::vector<double> VCRelaxOptimizer::crystal_to_vector(const Crystal& crystal) {
    const auto& atoms = crystal.atoms();
    const int natoms = static_cast<int>(atoms.size());
    const int N = 3 * natoms + 9;
    std::vector<double> x(N);

    // Atomic positions in Cartesian (bohr)
    for (int i = 0; i < natoms; ++i) {
        Vec3 cart = crystal.frac_to_cart(atoms[i].position);
        x[3 * i + 0] = cart[0];
        x[3 * i + 1] = cart[1];
        x[3 * i + 2] = cart[2];
    }

    // Lattice vectors in bohr (row-major: a1x,a1y,a1z, a2x,a2y,a2z, a3x,a3y,a3z)
    Mat3 lb = crystal.lattice_bohr();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            x[3 * natoms + 3 * i + j] = lb[i][j];
        }
    }

    return x;
}

std::vector<double> VCRelaxOptimizer::build_gradient(
    const Crystal& crystal,
    const std::vector<Vec3>& forces,
    const Mat3& stress,
    double press_target_gpa)
{
    const int natoms = static_cast<int>(forces.size());
    const int N = 3 * natoms + 9;
    std::vector<double> grad(N, 0.0);

    // Atomic gradient = -forces (gradient of energy w.r.t. position)
    for (int i = 0; i < natoms; ++i) {
        grad[3 * i + 0] = -forces[i][0];
        grad[3 * i + 1] = -forces[i][1];
        grad[3 * i + 2] = -forces[i][2];
    }

    // Cell gradient:
    // The generalized force on the cell is:
    //   f_cell_ab = -Omega * (sigma_ab - P_target * delta_ab)
    // where sigma is the stress tensor in Ry/bohr^3 and Omega is the volume.
    // The gradient (negative of generalized force) is:
    //   g_cell_ab = Omega * (sigma_ab - P_target * delta_ab)
    //
    // For the BFGS, we need the gradient w.r.t. the cell vector components.
    // The strain-based formulation gives:
    //   g_{ia} = sum_b Omega * (sigma_ab - P_target * delta_ab) * (h^{-T})_{bj}
    // but for simplicity, we use the direct cell derivative which maps to:
    //   g_cell_{ia} = Omega * sum_b (sigma_{ab} - P_target * delta_{ab})
    // applied along each lattice vector component.

    double volume = crystal.volume();  // bohr^3
    double press_target_ry_bohr3 = press_target_gpa / RY_BOHR3_TO_GPA;

    // Build the stress deviation tensor: sigma - P_target * I
    Mat3 stress_dev{};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            stress_dev[a][b] = stress[a][b];
            if (a == b) {
                stress_dev[a][b] -= press_target_ry_bohr3;
            }
        }
    }

    // Cell gradient: for each lattice vector a_i, the gradient w.r.t. a_{i,b} is:
    //   g_{i,b} = Omega * sum_a stress_dev_{ba}
    // More precisely, dE/d(h_{ia}) = Omega * sum_b sigma_{ab} * (h^{-T})_{bi}
    // In the simple approach consistent with QE's vc-relax:
    //   g_{ia} = volume * stress_dev_{ab}  (contracted with the lattice)
    //
    // We use the Parrinello-Rahman formulation:
    //   g_cell[i][a] = volume * sum_b stress_dev[a][b]  (for lattice vector i, component a)
    // But the cell vectors are independent, so:
    //   g_cell_flat[3*i + a] = volume * stress_dev[i_row][a]
    // Wait -- this needs to be the derivative of energy w.r.t. h_{ia}.
    //
    // The correct expression is:
    //   dE/dh_{ia} = V * sum_b sigma_{ab} * (h^{-T})_{bi}
    // Let's compute h^{-T}:

    Mat3 lb = crystal.lattice_bohr();

    // Compute the inverse transpose of the lattice matrix
    // First compute the inverse: h^{-1}_{ij} such that sum_k h_{ik} h^{-1}_{kj} = delta_{ij}
    // For 3x3: h^{-1} = adj(h) / det(h)
    double det = lb[0][0] * (lb[1][1] * lb[2][2] - lb[1][2] * lb[2][1])
               - lb[0][1] * (lb[1][0] * lb[2][2] - lb[1][2] * lb[2][0])
               + lb[0][2] * (lb[1][0] * lb[2][1] - lb[1][1] * lb[2][0]);

    if (std::abs(det) < 1e-30) {
        // Degenerate lattice, set zero cell gradient
        return grad;
    }

    // Adjugate (cofactor transpose)
    Mat3 adj{};
    adj[0][0] =  (lb[1][1] * lb[2][2] - lb[1][2] * lb[2][1]);
    adj[0][1] = -(lb[1][0] * lb[2][2] - lb[1][2] * lb[2][0]);
    adj[0][2] =  (lb[1][0] * lb[2][1] - lb[1][1] * lb[2][0]);
    adj[1][0] = -(lb[0][1] * lb[2][2] - lb[0][2] * lb[2][1]);
    adj[1][1] =  (lb[0][0] * lb[2][2] - lb[0][2] * lb[2][0]);
    adj[1][2] = -(lb[0][0] * lb[2][1] - lb[0][1] * lb[2][0]);
    adj[2][0] =  (lb[0][1] * lb[1][2] - lb[0][2] * lb[1][1]);
    adj[2][1] = -(lb[0][0] * lb[1][2] - lb[0][2] * lb[1][0]);
    adj[2][2] =  (lb[0][0] * lb[1][1] - lb[0][1] * lb[1][0]);

    // h^{-1}_{ij} = adj_{ij} / det
    // h^{-T}_{ij} = h^{-1}_{ji} = adj_{ji} / det

    // dE/dh_{ia} = V * sum_b sigma_dev_{ab} * h^{-T}_{bi}
    //            = V * sum_b sigma_dev_{ab} * adj_{ib} / det
    // Since V = |det|, and det can be positive (right-handed):
    // dE/dh_{ia} = |det| * sum_b sigma_dev_{ab} * adj_{ib} / det
    //            = sign(det) * sum_b sigma_dev_{ab} * adj_{ib}
    // For right-handed lattice, sign(det) = 1, so:
    // dE/dh_{ia} = sum_b sigma_dev_{ab} * adj_{ib}

    double sign_det = (det > 0) ? 1.0 : -1.0;

    for (int i = 0; i < 3; ++i) {
        for (int a = 0; a < 3; ++a) {
            double g = 0.0;
            for (int b = 0; b < 3; ++b) {
                g += stress_dev[a][b] * adj[i][b];
            }
            grad[3 * natoms + 3 * i + a] = sign_det * g;
        }
    }

    return grad;
}

Crystal VCRelaxOptimizer::vector_to_crystal(const std::vector<double>& x,
                                             int natoms,
                                             const Crystal& old_crystal) {
    // Extract new lattice vectors (in bohr)
    Mat3 new_lb{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            new_lb[i][j] = x[3 * natoms + 3 * i + j];
        }
    }

    // Convert lattice from bohr to angstrom
    Mat3 new_lat_ang{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            new_lat_ang[i][j] = new_lb[i][j] * constants::bohr_to_angstrom;
        }
    }

    // Construct a temporary crystal with new lattice to get cart_to_frac
    // We need the old atoms to have correct symbols/atomic numbers
    auto old_atoms = old_crystal.atoms();

    // Extract new Cartesian positions (bohr) and convert to fractional
    // with respect to the new lattice
    std::vector<Atom> new_atoms = old_atoms;
    Crystal temp_crystal(new_lat_ang, old_atoms);

    for (int i = 0; i < natoms; ++i) {
        Vec3 cart = {x[3 * i], x[3 * i + 1], x[3 * i + 2]};
        new_atoms[i].position = temp_crystal.cart_to_frac(cart);
    }

    return Crystal(new_lat_ang, std::move(new_atoms));
}

void VCRelaxOptimizer::update_inverse_hessian(
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

void VCRelaxOptimizer::limit_step(std::vector<double>& step, int natoms) const {
    // Limit atomic displacements
    double max_disp = 0.0;
    for (int i = 0; i < natoms; ++i) {
        double disp = 0.0;
        for (int d = 0; d < 3; ++d) {
            disp += step[3 * i + d] * step[3 * i + d];
        }
        max_disp = std::max(max_disp, std::sqrt(disp));
    }

    // Scale atomic part if any atom moves more than max_step
    if (max_disp > params_.max_step) {
        double scale = params_.max_step / max_disp;
        for (int i = 0; i < 3 * natoms; ++i) {
            step[i] *= scale;
        }
    }

    // Limit cell strain: find max cell step component
    double max_cell_step = 0.0;
    for (int i = 3 * natoms; i < static_cast<int>(step.size()); ++i) {
        max_cell_step = std::max(max_cell_step, std::abs(step[i]));
    }

    // Scale cell part if strain is too large
    // (rough estimate: strain ~ cell_step / typical_cell_dimension)
    // Use a fixed max_strain threshold instead
    if (max_cell_step > params_.max_strain * 10.0) {
        // Scale to keep cell changes reasonable
        double scale = params_.max_strain * 10.0 / max_cell_step;
        for (int i = 3 * natoms; i < static_cast<int>(step.size()); ++i) {
            step[i] *= scale;
        }
    }
}

double VCRelaxOptimizer::max_force(const std::vector<Vec3>& forces) {
    double max_f = 0.0;
    for (const auto& f : forces) {
        for (int d = 0; d < 3; ++d) {
            max_f = std::max(max_f, std::abs(f[d]));
        }
    }
    return max_f;
}

double VCRelaxOptimizer::max_stress_deviation(const Mat3& stress,
                                               double press_target_gpa) {
    double press_target_ry_bohr3 = press_target_gpa / RY_BOHR3_TO_GPA;
    double max_dev = 0.0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            double dev = stress[a][b];
            if (a == b) {
                dev -= press_target_ry_bohr3;
            }
            // Convert to kbar: Ry/bohr^3 -> GPa -> kbar
            double dev_kbar = std::abs(dev) * RY_BOHR3_TO_GPA * GPA_TO_KBAR;
            max_dev = std::max(max_dev, dev_kbar);
        }
    }
    return max_dev;
}

double VCRelaxOptimizer::compute_pressure_gpa(const Mat3& stress) {
    // Pressure = -trace(stress)/3, in Ry/bohr^3, then convert to GPa
    // Note: convention is that stress = -dE/dV, so P = -Tr(sigma)/3
    // But in typical DFT codes, stress is defined such that positive diagonal
    // means compression, so P = (sigma_xx + sigma_yy + sigma_zz) / 3
    double trace = stress[0][0] + stress[1][1] + stress[2][2];
    return trace / 3.0 * RY_BOHR3_TO_GPA;
}

VCRelaxResult VCRelaxOptimizer::optimize(
    Crystal crystal,
    const CalculationParams& calc_params,
    const ConvergenceParams& conv_params,
    const std::map<std::string, PseudoPotential>& pseudopotentials)
{
    VCRelaxResult result;
    const int natoms = static_cast<int>(crystal.num_atoms());
    const int N = generalized_dim(natoms);  // 3*natoms + 9

    // Initialize inverse Hessian to identity
    // Scale atomic and cell parts differently
    std::vector<double> H(N * N, 0.0);
    for (int i = 0; i < 3 * natoms; ++i) {
        H[i * N + i] = params_.initial_step;
    }
    // Cell part: scale by cell_factor
    for (int i = 3 * natoms; i < N; ++i) {
        H[i * N + i] = params_.initial_step * params_.cell_factor;
    }

    std::vector<double> prev_x;
    std::vector<double> prev_grad;
    double prev_energy = 0.0;

    std::printf("\n");
    std::printf("  Variable-Cell Relaxation (vc-relax)\n");
    std::printf("  Target pressure: %.4f GPa\n", params_.press_target);
    std::printf("  ════════════════════════════════════════════════════════════════════════\n");
    std::printf("  Step  Energy (Ry)        dE (Ry)        Max|F| (Ry/bohr)  P (GPa)  Max|sigma-P| (kbar)\n");
    std::printf("  ────────────────────────────────────────────────────────────────────────\n");

    for (int step = 0; step < params_.max_steps; ++step) {
        // Run SCF for current geometry
        // Make sure the calc type is treated as VCRelax to get forces computed
        CalculationParams scf_params = calc_params;
        scf_params.type = CalculationType::VCRelax;

        SCFSolver scf(crystal, scf_params, conv_params, pseudopotentials);
        auto scf_result = scf.solve();

        if (!scf_result.converged) {
            Logger::instance().warning("vc-relax", "SCF did not converge at vc-relax step " +
                         std::to_string(step + 1));
        }

        double energy = scf_result.total_energy_ry;
        double max_f = max_force(scf_result.forces);
        double pressure = compute_pressure_gpa(scf_result.stress);
        double max_stress_dev = max_stress_deviation(scf_result.stress,
                                                      params_.press_target);
        double de = (step == 0) ? energy : energy - prev_energy;

        result.energy_history.push_back(energy);
        result.pressure_history.push_back(pressure);

        std::printf("  %4d  %16.8f  %14.6e  %14.6e  %10.4f  %10.4f\n",
                    step + 1, energy, de, max_f, pressure, max_stress_dev);

        // Check convergence: both forces and stress must be below threshold
        bool forces_converged = (max_f < params_.force_threshold);
        bool stress_converged = (max_stress_dev < params_.stress_threshold);
        bool energy_converged = (step == 0 || std::abs(de) < params_.energy_threshold);

        if (forces_converged && stress_converged && energy_converged) {
            std::printf("  ────────────────────────────────────────────────────────────────────────\n");
            std::printf("  Converged: max|force| < %.1e Ry/bohr, max|stress-P| < %.1f kbar\n",
                        params_.force_threshold, params_.stress_threshold);

            result.converged = true;
            result.vc_steps = step + 1;
            result.final_energy_ry = energy;
            result.final_pressure_gpa = pressure;
            result.final_crystal = crystal;
            result.final_scf = std::move(scf_result);
            return result;
        }

        // Build generalized coordinate and gradient
        std::vector<double> x = crystal_to_vector(crystal);
        std::vector<double> grad = build_gradient(crystal, scf_result.forces,
                                                   scf_result.stress,
                                                   params_.press_target);

        // BFGS update of inverse Hessian (skip first step)
        if (step > 0 && !prev_x.empty()) {
            std::vector<double> s(N), y(N);
            for (int i = 0; i < N; ++i) {
                s[i] = x[i] - prev_x[i];
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
        limit_step(step_vec, natoms);

        // Save for next BFGS update
        prev_x = x;
        prev_grad = grad;
        prev_energy = energy;

        // Update generalized coordinates
        std::vector<double> new_x(N);
        for (int i = 0; i < N; ++i) {
            new_x[i] = x[i] + step_vec[i];
        }

        // Reconstruct crystal from updated vector
        crystal = vector_to_crystal(new_x, natoms, crystal);

        // Print updated lattice info
        std::printf("         New volume: %.4f bohr^3, lattice (bohr):\n", crystal.volume());
        Mat3 lb = crystal.lattice_bohr();
        for (int i = 0; i < 3; ++i) {
            std::printf("           a%d = [%10.6f, %10.6f, %10.6f]\n",
                        i + 1, lb[i][0], lb[i][1], lb[i][2]);
        }
    }

    // Did not converge
    std::printf("  ────────────────────────────────────────────────────────────────────────\n");
    std::printf("  NOT CONVERGED after %d vc-relax steps\n", params_.max_steps);

    result.converged = false;
    result.vc_steps = params_.max_steps;
    result.final_energy_ry = result.energy_history.back();
    result.final_pressure_gpa = result.pressure_history.back();
    result.final_crystal = crystal;
    return result;
}

} // namespace kronos
