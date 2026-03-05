#pragma once
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "io/upf_parser.hpp"
#include <map>
#include <vector>
#include <string>

namespace kronos {

// Ewald summation for ion-ion electrostatic energy in periodic systems
// E_ion = E_real + E_recip + E_self + E_charged
class EwaldCalculator {
public:
    struct Result {
        double energy;          // total ion-ion energy (Ry)
        std::vector<Vec3> forces; // force on each atom (Ry/bohr)
    };

    // Compute Ewald energy and forces
    // crystal: crystal structure
    // charges: Z_valence for each atom (from pseudopotentials)
    static Result compute(const Crystal& crystal,
                          const std::vector<double>& charges);

    // Convenience: extract charges from pseudopotentials and compute
    static Result compute(const Crystal& crystal,
                          const std::map<std::string, PseudoPotential>& pseudopotentials);

private:
    // Optimal Ewald parameter: eta = sqrt(pi) * (N_atoms / V^2)^(1/6)
    static double optimal_eta(double volume, int num_atoms);

    // Real-space sum
    static double real_space_energy(const Crystal& crystal,
                                    const std::vector<double>& charges,
                                    double eta);

    // Reciprocal-space sum
    static double recip_space_energy(const Crystal& crystal,
                                     const std::vector<double>& charges,
                                     double eta);

    // Self-interaction correction
    static double self_energy(const std::vector<double>& charges, double eta);

    // Charged system correction (if sum of charges != 0)
    static double charged_correction(const std::vector<double>& charges,
                                     double volume, double eta);

    // Forces: real-space contribution
    static std::vector<Vec3> real_space_forces(const Crystal& crystal,
                                               const std::vector<double>& charges,
                                               double eta);

    // Forces: reciprocal-space contribution
    static std::vector<Vec3> recip_space_forces(const Crystal& crystal,
                                                 const std::vector<double>& charges,
                                                 double eta);
};

} // namespace kronos
