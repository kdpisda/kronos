#pragma once

#include "core/types.hpp"
#include <string>
#include <vector>

namespace kronos {

// Density of states data.
struct DOSData {
    std::vector<double> energies;        // energy grid in eV
    std::vector<double> dos_values;      // DOS in states/eV
    std::vector<double> integrated_dos;  // integrated DOS (cumulative)
};

// Density of states calculator.
// Computes a smeared DOS from a set of eigenvalues and k-point weights.
class DOSCalculator {
public:
    // Compute the density of states via smearing broadening.
    //
    // eigenvalues[k][n]:  eigenvalue for k-point k, band n (Rydberg)
    // weights[k]:         k-point weight (should sum to 1 for non-spin-polarized)
    // smearing:           type of broadening kernel (Gaussian, FermiDirac, etc.)
    // degauss:            broadening width in eV (default ~0.05 eV)
    // energy_min:         lower bound of energy window in eV
    // energy_max:         upper bound of energy window in eV
    // num_points:         number of energy grid points
    // spin_factor:        2 for non-spin-polarized, 1 for spin-polarized
    //
    // The broadening kernel delta(E - e) is approximated by:
    //   Gaussian:          (1 / (sigma*sqrt(2*pi))) * exp(-(E-e)^2 / (2*sigma^2))
    //   FermiDirac:        -(d/dE) f_FD(E-e, sigma)  =  beta*exp(x) / (1+exp(x))^2
    //   MarzariVanderbilt: corresponding derivative of the MV smearing function
    //   None:              Gaussian with degauss (fallback)
    static DOSData compute_dos(
        const std::vector<std::vector<double>>& eigenvalues,
        const std::vector<double>& weights,
        SmearingType smearing,
        double degauss = 0.05,
        double energy_min = -20.0,
        double energy_max =  20.0,
        int num_points = 2001,
        int spin_factor = 2);

    // Write DOS data to a text file.
    // Columns: energy(eV)  dos(states/eV)  integrated_dos
    static void write_dos(const std::string& filename, const DOSData& dos_data);
};

} // namespace kronos
