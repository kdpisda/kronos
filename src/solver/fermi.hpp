#pragma once
#include "core/types.hpp"
#include <vector>

namespace kronos {

struct FermiResult {
    double fermi_energy{0.0};      // Ry
    std::vector<std::vector<double>> occupations;  // [kpoint][band]
    double total_electrons_found{0.0};
    bool converged{false};
};

class FermiSolver {
public:
    // Find Fermi level such that total electron count matches target
    // eigenvalues[k][n]: eigenvalue for k-point k, band n (Ry)
    // weights[k]: k-point weight (sum to 1 for non-spin-polarized)
    // target_electrons: total number of electrons
    // smearing: smearing type
    // degauss: smearing width (Ry)
    // spin_factor: 2 for non-spin-polarized, 1 for spin-polarized
    static FermiResult find_fermi_level(
        const std::vector<std::vector<double>>& eigenvalues,
        const std::vector<double>& weights,
        double target_electrons,
        SmearingType smearing,
        double degauss,
        int spin_factor = 2);

private:
    // Occupation function for given smearing type
    static double occupation(double x, SmearingType smearing);

    // Smearing functions
    static double fermi_dirac(double x);
    static double gaussian_smearing(double x);
    static double marzari_vanderbilt(double x);

    // Count electrons at a given Fermi level
    static double count_electrons(
        double ef,
        const std::vector<std::vector<double>>& eigenvalues,
        const std::vector<double>& weights,
        SmearingType smearing,
        double degauss,
        int spin_factor);
};

} // namespace kronos
