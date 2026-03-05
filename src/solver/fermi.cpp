#include "solver/fermi.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace kronos {

// ============================================================================
// Smearing functions
// ============================================================================

double FermiSolver::fermi_dirac(double x) {
    // f(x) = 1 / (1 + exp(x))
    // Guard against overflow: for large positive x, exp(x) overflows
    if (x > 40.0) return 0.0;
    if (x < -40.0) return 1.0;
    return 1.0 / (1.0 + std::exp(x));
}

double FermiSolver::gaussian_smearing(double x) {
    // f(x) = 0.5 * erfc(x)
    return 0.5 * std::erfc(x);
}

double FermiSolver::marzari_vanderbilt(double x) {
    // Marzari-Vanderbilt-DeVita-Payne "cold smearing" (PRL 82, 3296, 1999)
    // f(x) = 0.5 * erfc(x + 1/sqrt(2)) + (1/sqrt(2*pi)) * exp(-(x + 1/sqrt(2))^2)
    static const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    static const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);
    double xp = x + inv_sqrt2;
    return 0.5 * std::erfc(xp) + inv_sqrt_2pi * std::exp(-xp * xp);
}

double FermiSolver::occupation(double x, SmearingType smearing) {
    switch (smearing) {
        case SmearingType::None:
            // Step function: occupied if at or below Fermi level
            return (x <= 0.0) ? 1.0 : 0.0;
        case SmearingType::FermiDirac:
            return fermi_dirac(x);
        case SmearingType::Gaussian:
            return gaussian_smearing(x);
        case SmearingType::MarzariVanderbilt:
            return marzari_vanderbilt(x);
        default:
            return (x < 0.0) ? 1.0 : 0.0;
    }
}

// ============================================================================
// Electron counting
// ============================================================================

double FermiSolver::count_electrons(
    double ef,
    const std::vector<std::vector<double>>& eigenvalues,
    const std::vector<double>& weights,
    SmearingType smearing,
    double degauss,
    int spin_factor)
{
    double total = 0.0;
    const size_t nk = eigenvalues.size();

    for (size_t ik = 0; ik < nk; ++ik) {
        for (size_t ib = 0; ib < eigenvalues[ik].size(); ++ib) {
            // x = (epsilon - ef) / degauss
            double x = (eigenvalues[ik][ib] - ef) / degauss;
            double f = occupation(x, smearing);
            total += spin_factor * weights[ik] * f;
        }
    }

    return total;
}

// ============================================================================
// Fermi level search by bisection
// ============================================================================

FermiResult FermiSolver::find_fermi_level(
    const std::vector<std::vector<double>>& eigenvalues,
    const std::vector<double>& weights,
    double target_electrons,
    SmearingType smearing,
    double degauss,
    int spin_factor)
{
    assert(!eigenvalues.empty());
    assert(eigenvalues.size() == weights.size());

    // 1. Find energy bounds from all eigenvalues
    double e_min = std::numeric_limits<double>::max();
    double e_max = std::numeric_limits<double>::lowest();

    for (const auto& ek : eigenvalues) {
        for (double e : ek) {
            e_min = std::min(e_min, e);
            e_max = std::max(e_max, e);
        }
    }

    // Expand bounds by 10 * degauss
    double margin = 10.0 * degauss;
    e_min -= margin;
    e_max += margin;

    // 2. Bisection loop
    constexpr int max_bisection_steps = 200;
    constexpr double bisection_tol = 1e-10;  // Ry

    for (int iter = 0; iter < max_bisection_steps; ++iter) {
        if (e_max - e_min < bisection_tol) break;

        double e_mid = 0.5 * (e_min + e_max);
        double ne = count_electrons(e_mid, eigenvalues, weights,
                                    smearing, degauss, spin_factor);

        // Early exit when electron count matches target (important for
        // step-function smearing where count is discontinuous)
        if (std::abs(ne - target_electrons) < 1e-8) {
            e_min = e_mid;
            e_max = e_mid;
            break;
        }

        if (ne < target_electrons) {
            e_min = e_mid;
        } else {
            e_max = e_mid;
        }
    }

    double ef = 0.5 * (e_min + e_max);

    // 3. Compute final occupations
    FermiResult result;
    result.fermi_energy = ef;
    result.occupations.resize(eigenvalues.size());

    double total_found = 0.0;
    for (size_t ik = 0; ik < eigenvalues.size(); ++ik) {
        result.occupations[ik].resize(eigenvalues[ik].size());
        for (size_t ib = 0; ib < eigenvalues[ik].size(); ++ib) {
            double x = (eigenvalues[ik][ib] - ef) / degauss;
            double f = occupation(x, smearing);
            result.occupations[ik][ib] = spin_factor * f;
            total_found += spin_factor * weights[ik] * f;
        }
    }

    result.total_electrons_found = total_found;
    result.converged = (std::abs(total_found - target_electrons) < 1e-6);

    return result;
}

} // namespace kronos
