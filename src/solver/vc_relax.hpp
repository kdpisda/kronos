#pragma once
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "solver/scf.hpp"
#include "io/upf_parser.hpp"
#include <map>
#include <string>
#include <vector>

namespace kronos {

// Result of a variable-cell relaxation
struct VCRelaxResult {
    bool converged{false};
    int vc_steps{0};
    double final_energy_ry{0.0};
    double final_pressure_gpa{0.0};
    Crystal final_crystal;
    SCFResult final_scf;
    std::vector<double> energy_history;
    std::vector<double> pressure_history;
};

// Variable-cell relaxation optimizer.
// Simultaneously optimizes atomic positions and lattice vectors using a
// generalized BFGS algorithm. The generalized coordinate vector is:
//   [3N atomic positions (Cartesian bohr)] + [9 cell matrix elements]
// The generalized gradient vector is:
//   [-forces on atoms] + [Omega * (sigma - P_target * I)]
class VCRelaxOptimizer {
public:
    struct Params {
        int max_steps = 50;
        double force_threshold = 1e-3;    // Ry/bohr
        double stress_threshold = 0.5;    // kbar
        double energy_threshold = 1e-6;   // Ry
        double initial_step = 0.5;        // Initial trust radius
        double max_step = 1.0;            // Max displacement per atom (bohr)
        double max_strain = 0.05;         // Max strain component per step
        double press_target = 0.0;        // Target pressure in GPa
        double cell_factor = 2.0;         // Factor for cell optimization step
    };

    VCRelaxOptimizer();
    explicit VCRelaxOptimizer(Params params);

    // Run variable-cell relaxation
    VCRelaxResult optimize(
        Crystal crystal,
        const CalculationParams& calc_params,
        const ConvergenceParams& conv_params,
        const std::map<std::string, PseudoPotential>& pseudopotentials);

private:
    Params params_;

    // Generalized coordinate dimension: 3*natoms + 9 (cell)
    static int generalized_dim(int natoms) { return 3 * natoms + 9; }

    // Build generalized coordinate vector from crystal
    static std::vector<double> crystal_to_vector(const Crystal& crystal);

    // Build generalized gradient vector from forces and stress
    // gradient = [-forces, Omega * (sigma - P_target * I) applied to cell vectors]
    static std::vector<double> build_gradient(
        const Crystal& crystal,
        const std::vector<Vec3>& forces,
        const Mat3& stress,
        double press_target_gpa);

    // Update crystal from generalized coordinate vector
    static Crystal vector_to_crystal(const std::vector<double>& x, int natoms,
                                     const Crystal& old_crystal);

    // BFGS inverse Hessian update (same algorithm as ionic BFGS)
    static void update_inverse_hessian(
        std::vector<double>& H,
        const std::vector<double>& s,
        const std::vector<double>& y,
        int N);

    // Apply step size limits: both atomic displacement and strain
    void limit_step(std::vector<double>& step, int natoms) const;

    // Compute max force component magnitude
    static double max_force(const std::vector<Vec3>& forces);

    // Compute max |stress_ij - P_target * delta_ij| in kbar
    static double max_stress_deviation(const Mat3& stress, double press_target_gpa);

    // Compute pressure from stress tensor in GPa
    static double compute_pressure_gpa(const Mat3& stress);
};

} // namespace kronos
