#pragma once
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "solver/scf.hpp"
#include "io/upf_parser.hpp"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace kronos {

// Result of a geometry optimization
struct RelaxResult {
    bool converged{false};
    int relax_steps{0};
    double final_energy_ry{0.0};
    double final_energy_ev{0.0};
    double max_force_ry_bohr{0.0};

    // Final crystal structure
    Crystal final_crystal;

    // Final SCF result (from last ionic step)
    SCFResult final_scf;

    // History of energies and max forces per step
    std::vector<double> energy_history;
    std::vector<double> force_history;
};

// BFGS quasi-Newton geometry optimizer
// Minimizes total energy with respect to atomic positions
class BFGSOptimizer {
public:
    struct Params {
        int max_steps = 50;
        double force_threshold = 1e-3;    // Ry/bohr
        double energy_threshold = 1e-6;   // Ry
        double initial_step = 0.5;        // bohr (initial trust radius)
        double max_step = 1.0;            // bohr (max displacement per atom)
    };

    BFGSOptimizer();
    explicit BFGSOptimizer(Params params);

    // Run geometry optimization
    // calc_params, conv_params: SCF parameters for each ionic step
    // pseudopotentials: loaded pseudopotential data
    RelaxResult optimize(
        Crystal crystal,
        const CalculationParams& calc_params,
        const ConvergenceParams& conv_params,
        const std::map<std::string, PseudoPotential>& pseudopotentials);

private:
    Params params_;

    // Flatten atomic positions (Cartesian, bohr) into a 1D vector
    static std::vector<double> positions_to_vector(const Crystal& crystal);

    // Update crystal with new positions from 1D vector
    static Crystal update_positions(const Crystal& crystal,
                                    const std::vector<double>& pos_cart);

    // Flatten forces into 1D vector (negated gradient)
    static std::vector<double> forces_to_vector(const std::vector<Vec3>& forces);

    // Compute max force component magnitude
    static double max_force(const std::vector<Vec3>& forces);

    // BFGS inverse Hessian update
    static void update_inverse_hessian(
        std::vector<double>& H,   // N x N inverse Hessian (row-major)
        const std::vector<double>& s,  // position change
        const std::vector<double>& y,  // gradient change
        int N);

    // Apply step size limit
    static void limit_step(std::vector<double>& step, double max_step);
};

} // namespace kronos
