#include "postprocessing/dos.hpp"
#include "core/constants.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace kronos {

// ============================================================================
// Broadening kernels
//
// Each kernel approximates delta(E - e) for DOS broadening.
// Arguments:
//   x = (E - e) / sigma      (dimensionless)
//   sigma                     (broadening width in eV)
// Return value:               kernel value in 1/eV  (normalized)
// ============================================================================

namespace {

/// Gaussian kernel: (1 / (sigma * sqrt(2*pi))) * exp(-x^2 / 2)
/// where x = (E - e) / sigma.
double gaussian_kernel(double x, double sigma) {
    static const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * constants::pi);
    return (inv_sqrt_2pi / sigma) * std::exp(-0.5 * x * x);
}

/// Lorentzian (Cauchy) approximation to delta, used as the Fermi-Dirac
/// broadening kernel: -(df/dE) = beta * exp(x) / (1 + exp(x))^2
/// where x = (E - e) / sigma, beta = 1/sigma.
double fermi_dirac_kernel(double x, double sigma) {
    // Numerically stable form:
    //   exp(x) / (1+exp(x))^2 = 1 / (2 + 2*cosh(x))
    if (std::abs(x) > 40.0) return 0.0;
    return (1.0 / sigma) / (2.0 + 2.0 * std::cosh(x));
}

/// Marzari-Vanderbilt "cold smearing" kernel:
///   delta_MV(x) = (1/sqrt(pi)) * exp(-x^2) * (2 - sqrt(2)*x)
/// where x = (E - e) / sigma, and the whole thing is divided by sigma for
/// units of 1/eV.
double marzari_vanderbilt_kernel(double x, double sigma) {
    static const double inv_sqrt_pi = 1.0 / std::sqrt(constants::pi);
    double ex2 = std::exp(-x * x);
    return (inv_sqrt_pi / sigma) * ex2 * (2.0 - constants::sqrt2 * x);
}

/// Choose the appropriate kernel based on smearing type.
double broadening_kernel(double x, double sigma, SmearingType smearing) {
    switch (smearing) {
        case SmearingType::FermiDirac:
            return fermi_dirac_kernel(x, sigma);
        case SmearingType::MarzariVanderbilt:
            return marzari_vanderbilt_kernel(x, sigma);
        case SmearingType::Gaussian:
        case SmearingType::None:
        default:
            return gaussian_kernel(x, sigma);
    }
}

} // anonymous namespace

// ============================================================================
// compute_dos
// ============================================================================
DOSData DOSCalculator::compute_dos(
    const std::vector<std::vector<double>>& eigenvalues,
    const std::vector<double>& weights,
    SmearingType smearing,
    double degauss,
    double energy_min,
    double energy_max,
    int num_points,
    int spin_factor)
{
    if (eigenvalues.empty()) {
        throw std::invalid_argument("DOSCalculator::compute_dos: empty eigenvalues");
    }
    if (eigenvalues.size() != weights.size()) {
        throw std::invalid_argument(
            "DOSCalculator::compute_dos: eigenvalues and weights size mismatch");
    }
    if (num_points < 2) {
        throw std::invalid_argument("DOSCalculator::compute_dos: num_points must be >= 2");
    }
    if (degauss <= 0.0) {
        throw std::invalid_argument("DOSCalculator::compute_dos: degauss must be > 0");
    }

    DOSData data;
    data.energies.resize(num_points);
    data.dos_values.resize(num_points, 0.0);
    data.integrated_dos.resize(num_points, 0.0);

    const double de = (energy_max - energy_min) / static_cast<double>(num_points - 1);

    // Build the energy grid.
    for (int i = 0; i < num_points; ++i) {
        data.energies[i] = energy_min + i * de;
    }

    // Accumulate DOS contributions from every eigenvalue at every k-point.
    // eigenvalues are in Rydberg; convert to eV for comparison with the
    // energy grid.
    const size_t nk = eigenvalues.size();

    for (size_t ik = 0; ik < nk; ++ik) {
        double wk = weights[ik];
        for (size_t ib = 0; ib < eigenvalues[ik].size(); ++ib) {
            double e_nk_ev = eigenvalues[ik][ib] * constants::rydberg_to_ev;

            // Only bother with energy points within a reasonable window
            // around this eigenvalue (say +/- 10 * degauss).
            double window = 10.0 * degauss;
            int i_lo = static_cast<int>(std::floor((e_nk_ev - window - energy_min) / de));
            int i_hi = static_cast<int>(std::ceil ((e_nk_ev + window - energy_min) / de));
            i_lo = std::max(i_lo, 0);
            i_hi = std::min(i_hi, num_points - 1);

            for (int i = i_lo; i <= i_hi; ++i) {
                double E = data.energies[i];
                double x = (E - e_nk_ev) / degauss;
                double kernel_val = broadening_kernel(x, degauss, smearing);
                data.dos_values[i] += spin_factor * wk * kernel_val;
            }
        }
    }

    // Compute the integrated DOS via trapezoidal rule.
    data.integrated_dos[0] = 0.0;
    for (int i = 1; i < num_points; ++i) {
        data.integrated_dos[i] = data.integrated_dos[i - 1]
                                + 0.5 * de * (data.dos_values[i - 1] + data.dos_values[i]);
    }

    Logger::instance().info("dos", "DOS computed",
        {{"num_points", std::to_string(num_points)},
         {"degauss_eV", std::to_string(degauss)},
         {"num_kpoints", std::to_string(nk)}});

    return data;
}

// ============================================================================
// write_dos
// ============================================================================
void DOSCalculator::write_dos(const std::string& filename, const DOSData& dos_data) {
    if (dos_data.energies.empty()) {
        throw std::invalid_argument("DOSCalculator::write_dos: empty DOS data");
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error(
            "DOSCalculator::write_dos: cannot open file: " + filename);
    }

    ofs << std::setprecision(10);
    ofs << "# KRONOS density of states\n";
    ofs << "# Columns: energy(eV)  dos(states/eV)  integrated_dos(states)\n";

    for (size_t i = 0; i < dos_data.energies.size(); ++i) {
        ofs << dos_data.energies[i]
            << "  " << dos_data.dos_values[i]
            << "  " << dos_data.integrated_dos[i]
            << "\n";
    }

    ofs.flush();
    if (!ofs.good()) {
        throw std::runtime_error(
            "DOSCalculator::write_dos: write error on: " + filename);
    }

    Logger::instance().info("dos", "DOS data written",
        {{"filename", filename},
         {"num_points", std::to_string(dos_data.energies.size())}});
}

} // namespace kronos
