#pragma once
// ============================================================================
// KRONOS  src/potential/paw.hpp
// PAW (Projector Augmented Wave) Calculator
//
// Handles:
//   - Augmentation charge density n_aug(G)
//   - One-center energy corrections (AE - PS)
//   - D_ij^PAW contributions to nonlocal PP
//   - Overlap operator S for generalized eigenvalue problem
//   - PAW force and stress corrections
// ============================================================================

#include "core/types.hpp"
#include "core/crystal.hpp"
#include "io/upf_parser.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"

#include <map>
#include <string>
#include <vector>

namespace kronos {

class PAWCalculator {
public:
    PAWCalculator(const Crystal& crystal,
                  const PlaneWaveBasis& basis,
                  FFTGrid& fft_grid,
                  const std::map<std::string, PseudoPotential>& pseudopotentials);

    /// Check if any PAW PPs are present
    bool has_paw() const { return has_paw_; }

    // -----------------------------------------------------------------------
    // Occupation matrix ρ_ij
    // -----------------------------------------------------------------------

    /// Compute PAW occupation matrix ρ_ij from wavefunctions and projectors.
    /// rho_ij[atom_idx][i*np+j] stores the matrix for each atom.
    /// projections[ik][ib][ip] = <β_ip|ψ_{ik,ib}> (precomputed).
    void compute_rho_ij(
        const std::vector<std::vector<std::vector<complex_t>>>& projections,
        const std::vector<std::vector<double>>& occupations,
        const std::vector<double>& kweights,
        int spin_factor);

    // -----------------------------------------------------------------------
    // Augmentation density
    // -----------------------------------------------------------------------

    /// Add augmentation charge density to n(G) on FFT grid.
    /// n_aug(G) = Σ_{I,ij} ρ_ij · Q_ij(|G|) · exp(-iG·τ_I)
    void add_augmentation_density(std::vector<complex_t>& density_g,
                                  const std::vector<Vec3>& grid_gcart,
                                  const std::vector<double>& grid_g2,
                                  double ecutrho) const;

    // -----------------------------------------------------------------------
    // One-center energy correction
    // -----------------------------------------------------------------------

    /// Compute PAW one-center energy: E_1 = E_AE - E_PS
    /// This is the difference between all-electron and pseudo one-center
    /// energies evaluated using the occupation matrix.
    double one_center_energy() const;

    // -----------------------------------------------------------------------
    // D_ij PAW correction
    // -----------------------------------------------------------------------

    /// Compute PAW contribution to D_ij:
    ///   D_ij^PAW = ∫ Q_ij(r) V_eff(r) dr + D_ij^one_center
    /// Returns D_ij corrections per atom: dij_paw[atom_idx][i*np+j]
    std::vector<std::vector<double>> compute_dij_paw(
        const std::vector<complex_t>& veff_g,
        const std::vector<Vec3>& grid_gcart,
        const std::vector<double>& grid_g2,
        double ecutrho) const;

    // -----------------------------------------------------------------------
    // Overlap operator S
    // -----------------------------------------------------------------------

    /// Apply overlap operator: S|ψ⟩ = |ψ⟩ + Σ_{ij} q_ij |β_i⟩⟨β_j|ψ⟩
    /// projections[ip] = <β_ip|ψ> for the current atom
    /// beta_g[ip] is the G-space projector
    CVec apply_s(const CVec& psi_g,
                 const std::vector<CVec>& beta_g,
                 const std::vector<std::vector<complex_t>>& projections_per_atom) const;

    /// Get q_ij overlap integrals for a given atom species.
    /// q_ij[i*np+j] = ∫ Q_ij(r) dr
    const std::vector<double>& get_qij(const std::string& species) const;

    // -----------------------------------------------------------------------
    // PAW force and stress corrections
    // -----------------------------------------------------------------------

    /// Compute augmentation contribution to forces:
    /// F_aug_I = -Σ_{ij} ρ_ij Σ_G Q_ij(|G|) V_eff(G) iG exp(-iG·τ_I)
    std::vector<Vec3> compute_paw_forces(
        const std::vector<complex_t>& veff_g,
        const std::vector<Vec3>& grid_gcart,
        const std::vector<double>& grid_g2,
        double ecutrho) const;

    /// Compute augmentation contribution to stress (isotropic approximation for v0.8)
    Mat3 compute_paw_stress(
        const std::vector<complex_t>& veff_g,
        const std::vector<Vec3>& grid_gcart,
        const std::vector<double>& grid_g2,
        double ecutrho) const;

    // -----------------------------------------------------------------------
    // Access to internal state
    // -----------------------------------------------------------------------

    /// Get the current occupation matrix
    const std::vector<std::vector<double>>& rho_ij() const { return rho_ij_; }

private:
    const Crystal& crystal_;
    const PlaneWaveBasis& basis_;
    FFTGrid& fft_grid_;
    bool has_paw_{false};

    // Per-atom data
    struct AtomPAWData {
        size_t atom_index;
        std::string species;
        Vec3 position_cart;
        int num_projectors;
        const PAWData* paw;           // Pointer to PAW data from PP
        const PseudoPotential* pp;    // Full PP for mesh, betas, etc.
    };
    std::vector<AtomPAWData> atoms_;

    // Per-species overlap integrals q_ij
    std::map<std::string, std::vector<double>> qij_cache_;

    // Current occupation matrix: rho_ij_[atom_idx][i*np+j]
    std::vector<std::vector<double>> rho_ij_;

    // Internal helpers
    void compute_qij_cache();
    double compute_one_center_ae(size_t atom_idx) const;
    double compute_one_center_ps(size_t atom_idx) const;
};

} // namespace kronos
