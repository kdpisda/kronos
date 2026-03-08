#pragma once

#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "io/upf_parser.hpp"

#include <map>
#include <string>
#include <vector>

namespace kronos {

/// Stress tensor calculator.
///
/// Computes the 3x3 stress tensor sigma_ab as the strain derivative of the
/// total energy per unit volume:
///
///   sigma_ab = (1/Omega) * dE_total / d(epsilon_ab)
///
/// The total stress is a sum of six contributions:
///   sigma = sigma_kinetic + sigma_hartree + sigma_xc
///         + sigma_local + sigma_nonlocal + sigma_ewald
///
/// All quantities are in Rydberg atomic units: stress in Ry/bohr^3.
/// Pressure in GPa is obtained via:  P = trace(sigma)/3 * Ry_per_bohr3_to_GPa
///
/// Convention: positive diagonal => tensile stress (system wants to expand).
/// Pressure is P = -trace(sigma)/3 (positive pressure = compressive).
class StressCalculator {
public:
    /// Conversion factor: 1 Ry/bohr^3 = 14710.5 GPa
    static constexpr double RY_BOHR3_TO_GPA = 14710.507;

    /// Kinetic stress tensor.
    ///
    /// sigma_ab^kin = -(1/Omega) * sum_{n,k} f_{nk} w_k
    ///               * sum_G (k+G)_a (k+G)_b |c_{nk}(G)|^2
    ///
    /// (In Rydberg units, T = |k+G|^2, so the energy derivative gives
    /// the product of two (k+G) components.)
    ///
    /// Sign: the kinetic contribution to dE/d(epsilon) is negative because
    /// expanding the cell reduces kinetic energy (fewer G-vectors inside cutoff).
    static Mat3 compute_kinetic_stress(
        const Crystal& crystal,
        const PlaneWaveBasis& basis,
        const std::vector<std::vector<CVec>>& wavefunctions,
        const std::vector<std::vector<double>>& occupations,
        const std::vector<Vec3>& k_points,
        const std::vector<double>& k_weights);

    /// Hartree stress tensor.
    ///
    /// sigma_ab^H = (1/Omega) * sum_{G!=0} 8*pi*|n(G)|^2
    ///             * (delta_ab / (2*G^2) - G_a*G_b / G^4)
    ///
    /// Uses the full FFT density grid (un-normalized coefficients).
    static Mat3 compute_hartree_stress(
        const std::vector<complex_t>& density_g_full,
        const std::vector<Vec3>& grid_gcart,
        const std::vector<double>& grid_g2,
        double ecutrho,
        double volume,
        int num_grid);

    /// Exchange-correlation stress tensor (LDA only, isotropic).
    ///
    /// sigma_ab^xc = -delta_ab * (1/Omega) * (E_xc - integral v_xc(r)*n(r) dr)
    ///
    /// For LDA, the XC stress is diagonal (isotropic).
    /// GGA terms can be added later as a TODO.
    ///
    /// @param exc_energy   Total E_xc from the XC evaluator (Ry).
    /// @param vxc_r        V_xc(r) on the real-space grid.
    /// @param density_r    n(r) on the real-space grid.
    /// @param volume       Cell volume in bohr^3.
    /// @param num_grid     Total number of FFT grid points.
    static Mat3 compute_xc_stress(
        double exc_energy,
        const RVec& vxc_r,
        const RVec& density_r,
        double volume,
        int num_grid);

    /// Local pseudopotential stress tensor.
    ///
    /// sigma_ab^loc = (1/Omega) * sum_{G!=0} n*(G) * V_loc(G)
    ///               * (delta_ab + G_a*G_b * d ln V_loc / d(G^2) * 2)
    ///
    /// Uses the full FFT density grid for consistency with the local energy.
    static Mat3 compute_local_stress(
        const Crystal& crystal,
        const std::map<std::string, PseudoPotential>& pseudopotentials,
        const std::vector<complex_t>& density_g_full,
        const std::vector<complex_t>& vloc_full_g,
        const std::vector<Vec3>& grid_gcart,
        const std::vector<double>& grid_g2,
        double ecutrho,
        double volume,
        int num_grid);

    /// Nonlocal pseudopotential stress tensor.
    ///
    /// sigma_ab^NL = -(1/Omega) * sum_{n,k} f_{nk} w_k
    ///              * sum_{a,i,j} D_{ij} * 2*Re[ conj(P_j) * dP_i^{ab} ]
    ///
    /// where dP_i^{ab} = sum_G (k+G)_a * d(beta_i(|k+G|))/d(|k+G|) * (k+G)_b/|k+G|
    ///                  * Y_lm * exp(-i(k+G).tau) * psi(G)
    ///
    /// Follows the nonlocal_forces pattern closely.
    static Mat3 compute_nonlocal_stress(
        const Crystal& crystal,
        const PlaneWaveBasis& basis,
        const std::map<std::string, PseudoPotential>& pseudopotentials,
        const std::vector<std::vector<CVec>>& wavefunctions,
        const std::vector<std::vector<double>>& occupations,
        const std::vector<Vec3>& k_points,
        const std::vector<double>& k_weights);

    /// Ewald (ion-ion) stress tensor.
    ///
    /// Strain derivative of the Ewald energy in both real and reciprocal
    /// space, plus corrections.
    static Mat3 compute_ewald_stress(
        const Crystal& crystal,
        const std::map<std::string, PseudoPotential>& pseudopotentials);

    /// Combine all stress contributions into the total stress tensor.
    static Mat3 compute_total_stress(
        const Mat3& kinetic,
        const Mat3& hartree,
        const Mat3& xc,
        const Mat3& local_pp,
        const Mat3& nonlocal_pp,
        const Mat3& ewald);

    /// Compute hydrostatic pressure in GPa from the stress tensor.
    /// P = -trace(sigma)/3 * conversion  (negative sign: compression = positive P)
    static double pressure_gpa(const Mat3& stress);

private:
    /// Helper: radial Fourier transform of V_loc at wavevector magnitude q.
    /// (Same as ForceCalculator::vloc_of_q.)
    static double vloc_of_q(const PseudoPotential& pp, double q, double volume);

    /// Helper: d(vloc_of_q)/d(q^2) — derivative of the form factor with
    /// respect to G^2, needed for the local stress tensor.
    static double dvloc_dq2(const PseudoPotential& pp, double q, double volume);
};

} // namespace kronos
