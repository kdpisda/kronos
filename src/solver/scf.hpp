#pragma once
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "io/upf_parser.hpp"
#include <map>
#include <string>
#include <vector>

namespace kronos {

// Forward declarations
struct CalculationParams;
struct ConvergenceParams;

// SCF calculation result
struct SCFResult {
    bool converged{false};
    int scf_steps{0};
    double total_energy_ry{0.0};
    double total_energy_ev{0.0};
    double fermi_energy_ev{0.0};

    // Energy components (Ry)
    double kinetic_energy{0.0};
    double hartree_energy{0.0};
    double xc_energy{0.0};
    double local_pp_energy{0.0};
    double nonlocal_pp_energy{0.0};
    double ewald_energy{0.0};      // ion-ion Ewald energy

    // Per-atom forces (Ry/bohr) — total Hellmann-Feynman forces
    std::vector<Vec3> forces;

    // Per-atom force components (Ry/bohr)
    std::vector<Vec3> ewald_forces;
    std::vector<Vec3> local_forces;
    std::vector<Vec3> nonlocal_forces;

    // Per-k-point eigenvalues (Ry)
    std::vector<std::vector<double>> eigenvalues;

    // Converged effective potential V_eff(r) on real-space grid
    // (used by band structure calculations)
    std::vector<complex_t> converged_veff_r;

    // Timing info
    std::map<std::string, double> timing;
};

class SCFSolver {
public:
    SCFSolver(const Crystal& crystal,
              const CalculationParams& calc_params,
              const ConvergenceParams& conv_params,
              const std::map<std::string, PseudoPotential>& pseudopotentials);

    // Run the SCF calculation
    SCFResult solve();

private:
    const Crystal& crystal_;
    const CalculationParams& calc_params_;
    const ConvergenceParams& conv_params_;
    const std::map<std::string, PseudoPotential>& pseudopotentials_;

    // Determine number of bands to compute
    int compute_num_bands() const;

    // Initialize density from superposition of atomic densities
    RVec initial_density(const PlaneWaveBasis& basis, FFTGrid& fft_grid) const;

    // Compute total energy from components
    double compute_total_energy(double kinetic, double hartree, double xc,
                                double local_pp, double nonlocal_pp,
                                double ewald) const;

    // Compute band structure energy: E_band = sum_nk f_nk * epsilon_nk
    double compute_band_energy(const std::vector<std::vector<double>>& eigenvalues,
                               const std::vector<std::vector<double>>& occupations,
                               const std::vector<double>& kweights) const;

    // Print SCF step info to stdout (format from spec)
    void print_scf_step(int step, double energy, double de, double dn,
                        double wall_time) const;
};

} // namespace kronos
