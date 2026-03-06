#pragma once

#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "io/upf_parser.hpp"
#include "potential/nonlocal_pp.hpp"

#include <map>
#include <string>
#include <vector>

namespace kronos {

/// Hellmann-Feynman force calculator.
///
/// Computes the total force on each atom as the sum of three contributions:
///
///   F_I = F_ewald_I + F_local_I + F_nonlocal_I
///
/// where:
///   - F_ewald:    ion-ion Ewald forces (computed externally by EwaldCalculator)
///   - F_local:    derivative of the local pseudopotential energy w.r.t. atomic
///                 positions:
///                   F_local_I = -dE_loc/dR_I
///                             = -sum_G  V_loc^species(|G|) * n(G) * iG * exp(-iG.R_I) * Omega
///                 (summed over the species of atom I)
///   - F_nonlocal: derivative of the KB nonlocal energy w.r.t. atomic positions:
///                   F_nonlocal_I = -dE_NL/dR_I
///
/// All quantities are in Rydberg atomic units.
class ForceCalculator {
public:
    /// Force contributions from the local pseudopotential.
    ///
    /// The local energy is E_loc = (Ω/N) Σ_G conj(V_loc(G)) · n_FFT(G),
    /// summed over the full density grid (G² ≤ ecutrho).  The force must
    /// be the exact derivative of this energy, so it must sum over the
    /// same G-vector set.
    ///
    /// @param crystal          Crystal structure.
    /// @param pseudopotentials Map of pseudopotentials by element symbol.
    /// @param density_g_full   Electron density on full FFT grid (G-space, un-normalized).
    /// @param grid_gcart       Cartesian G-vectors for every FFT grid point.
    /// @param grid_g2          |G|² for every FFT grid point.
    /// @param ecutrho          Density cutoff (Ry); only G² ≤ ecutrho contribute.
    /// @param volume           Unit cell volume (bohr³).
    /// @param num_grid         Total number of FFT grid points.
    /// @return                 Force vector (Ry/bohr) per atom.
    static std::vector<Vec3> compute_local_forces(
        const Crystal& crystal,
        const std::map<std::string, PseudoPotential>& pseudopotentials,
        const std::vector<complex_t>& density_g_full,
        const std::vector<Vec3>& grid_gcart,
        const std::vector<double>& grid_g2,
        double ecutrho,
        double volume,
        int num_grid);

    /// Force contributions from the nonlocal pseudopotential (KB projectors).
    ///
    /// For each atom I the nonlocal force is:
    ///   F_NL_I = -dE_NL/dR_I
    ///          = -sum_{n,k} f_{n,k} w_k * spin_factor
    ///            * d/dR_I <psi_{n,k}|V_NL|psi_{n,k}>
    ///
    /// The position dependence enters through the phase factor exp(-i(k+G).tau_I)
    /// in the projectors beta_i^I(G). The derivative gives -iG * beta(G) (the
    /// d(beta)/d(tau) term).
    ///
    /// @param crystal          Crystal structure.
    /// @param basis            Plane-wave basis set.
    /// @param pseudopotentials Map of pseudopotentials by element symbol.
    /// @param wavefunctions    Converged wavefunctions per k-point, per band.
    /// @param occupations      Occupation numbers per k-point, per band.
    /// @param k_points         k-point coordinates in fractional reciprocal space.
    /// @param k_weights        k-point integration weights.
    /// @param spin_factor      2 for spin-unpolarized, 1 for spin-polarized.
    /// @return                 Force vector (Ry/bohr) per atom.
    static std::vector<Vec3> compute_nonlocal_forces(
        const Crystal& crystal,
        const PlaneWaveBasis& basis,
        const std::map<std::string, PseudoPotential>& pseudopotentials,
        const std::vector<std::vector<CVec>>& wavefunctions,
        const std::vector<std::vector<double>>& occupations,
        const std::vector<Vec3>& k_points,
        const std::vector<double>& k_weights,
        int spin_factor);

    /// Combine all force contributions into the total Hellmann-Feynman force.
    ///
    /// F_total_I = F_ewald_I + F_local_I + F_nonlocal_I
    ///
    /// @param ewald_forces     Ion-ion Ewald forces per atom (Ry/bohr).
    /// @param local_forces     Local PP forces per atom (Ry/bohr).
    /// @param nonlocal_forces  Nonlocal PP forces per atom (Ry/bohr).
    /// @return                 Total force per atom (Ry/bohr).
    static std::vector<Vec3> compute_total_forces(
        const std::vector<Vec3>& ewald_forces,
        const std::vector<Vec3>& local_forces,
        const std::vector<Vec3>& nonlocal_forces);

private:
    /// Radial Fourier transform of V_loc at wavevector magnitude q.
    /// This is the *form factor* (without structure factor), i.e. the
    /// species-dependent part V_loc^s(|G|), normalized per unit cell volume.
    ///
    /// Same as LocalPPEvaluator::vloc_of_q but exposed here for the force
    /// calculation which needs the per-species form factor separately.
    static double vloc_of_q(const PseudoPotential& pp, double q, double volume);
};

} // namespace kronos
