#pragma once

#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "io/upf_parser.hpp"

#include <map>
#include <string>
#include <vector>

namespace kronos {

/// Non-local pseudopotential in Kleinman-Bylander form.
///
/// For each atom a at position tau_a with projectors {beta_i}, the
/// non-local contribution to the Hamiltonian is
///
///   V_NL = sum_{a,i,j}  D_{ij}^a  |beta_i^a> <beta_j^a|
///
/// In reciprocal space:
///   <G|beta_i^a> = (4*pi / sqrt(Omega)) * i^l
///                  * integral r^2 beta_i(r) j_l(|k+G|*r) dr
///                  * Y_{lm}(k+G_hat) * exp(-i(k+G).tau_a)
///
/// Each UPF projector with angular momentum l is expanded into (2l+1)
/// channels, one for each m = -l, ..., +l, using the real spherical
/// harmonics Y_lm.  The D_ij matrix is correspondingly expanded into a
/// block-diagonal form where each m-channel carries the same D_ij value.
///
/// The projectors are computed on-the-fly for each k-point so that the
/// correct |k+G| and exp(-i(k+G).tau) are used.  The same G-vector set
/// (determined by the energy cutoff on |G|) is used for all k-points.
class NonlocalPP {
public:
    NonlocalPP(const Crystal& crystal,
               const PlaneWaveBasis& basis,
               const std::map<std::string, PseudoPotential>& pseudopotentials);

    /// Precompute and cache beta projectors for a given k-point.
    /// Call this once before apply() for each new k-point to avoid
    /// recomputing the expensive radial Bessel transforms every time.
    void prepare_kpoint(const Vec3& k_frac);

    /// Apply V_NL to a single wavefunction in G-space:
    ///   (V_NL |psi>)_G = sum_{a,i,j} D_{ij} <beta_j|psi> beta_i(G)
    ///
    /// Uses cached projectors from prepare_kpoint(). If prepare_kpoint()
    /// was not called for this k-point, falls back to on-the-fly computation.
    ///
    /// @param psi_g   Wavefunction coefficients in the plane-wave basis.
    /// @param k_frac  k-point in fractional reciprocal coordinates.
    /// @return        V_NL|psi> in the same basis.
    [[nodiscard]] CVec apply(const CVec& psi_g, const Vec3& k_frac) const;

    /// Compute the total non-local energy:
    ///   E_NL = sum_n f_n <psi_n|V_NL|psi_n>
    ///
    /// @param wavefunctions  Wavefunction coefficients per band.
    /// @param occupations    Occupation numbers f_n.
    /// @param k_frac         k-point in fractional reciprocal coordinates.
    [[nodiscard]] double energy(const std::vector<CVec>& wavefunctions,
                                const std::vector<double>& occupations,
                                const Vec3& k_frac) const;

    /// Total number of expanded projectors across all atoms.
    /// Each UPF projector with angular momentum l contributes (2l+1) entries.
    [[nodiscard]] int num_projectors() const;

    /// Compute <β_j|ψ> projections for all atoms at cached k-point.
    /// Returns projections[atom_idx][expanded_proj_idx].
    [[nodiscard]] std::vector<std::vector<complex_t>> compute_projections(const CVec& psi_g) const;

    /// Save base D_ij (call once after construction, before PAW corrections).
    void save_base_dij();

    /// Restore D_ij to saved base values.
    void reset_dij();

    /// Add PAW correction to D_ij for a given atom (in UPF projector indices).
    /// correction[i_upf * np + j_upf] is mapped to the expanded (l,m) basis.
    void add_dij_correction(size_t atom_idx, const std::vector<double>& correction);

    /// Number of atoms with nonlocal projectors.
    [[nodiscard]] size_t num_atoms() const { return atom_data_.size(); }

    /// Crystal atom index for the given NonlocalPP atom index.
    [[nodiscard]] int crystal_atom_index(size_t idx) const { return atom_data_[idx].atom_index; }

    /// Number of UPF (unexpanded) projectors for a given atom.
    [[nodiscard]] int num_upf_projectors(size_t idx) const {
        return static_cast<int>(atom_data_[idx].projectors.size());
    }

    /// Access cached beta projectors (per atom, per expanded proj).
    [[nodiscard]] const std::vector<std::vector<CVec>>& cached_beta() const { return cached_beta_kg_; }

    /// Access expanded_map for a given atom (for UPF↔expanded index mapping).
    [[nodiscard]] const auto& expanded_map(size_t atom_idx) const {
        return atom_data_[atom_idx].expanded_map;
    }

private:
    /// Per-atom data: species info, position, D_ij, and the projector
    /// metadata needed to recompute beta(k+G) on the fly.
    struct AtomData {
        int atom_index;
        std::string species;
        Vec3 position_cart;                     // bohr

        /// Per-UPF-projector metadata (angular momentum, radial beta, etc.)
        struct ProjInfo {
            int angular_momentum;
            int cutoff_index;
            std::vector<double> values;         // beta(r) on radial grid
        };
        std::vector<ProjInfo> projectors;

        /// D_ij matrix in the expanded (l,m) basis (Ry).
        /// Block-diagonal: nonzero only when the two expanded indices
        /// share the same m value and their UPF projectors share the same l.
        std::vector<std::vector<double>> dij;   // [num_proj_expanded][num_proj_expanded]

        /// Mapping from expanded index to (UPF beta index, l, m).
        struct ExpandedIndex {
            int upf_beta_index;
            int l;
            int m;
        };
        std::vector<ExpandedIndex> expanded_map;

        /// Total number of expanded projectors for this atom.
        int num_expanded{0};
    };

    /// Per-atom data for all atoms with nonlocal projectors.
    std::vector<AtomData> atom_data_;

    /// Reference to the basis (needed for G-vectors in apply/energy).
    const PlaneWaveBasis& basis_;

    /// Cell volume (bohr^3), cached for the prefactor.
    double volume_;

    /// Radial grid data per species (shared across atoms of the same species).
    std::map<std::string, RadialGrid> species_meshes_;

    /// Cached beta projectors per atom for the current k-point.
    /// cached_beta_kg_[atom_index][expanded_proj_index] is a CVec of length npw.
    mutable std::vector<std::vector<CVec>> cached_beta_kg_;
    mutable Vec3 cached_kpoint_{1e30, 1e30, 1e30};  // sentinel

    /// Saved base D_ij for PAW reset/restore cycle.
    std::vector<std::vector<std::vector<double>>> dij_base_;

    /// Compute the expanded projectors beta_kg[proj_expanded][ig] for a given
    /// k-point.  Uses |k+G| for the Bessel transform and exp(-i(k+G).tau)
    /// for the phase factor.
    [[nodiscard]] std::vector<CVec> compute_beta_kg(
        const AtomData& ad, const Vec3& k_frac) const;

    /// Compute the spherical Bessel transform of a beta projector at |q|:
    ///   integral_0^inf r^2 * beta(r) * j_l(q*r) * dr
    /// using the trapezoidal rule with rab weights.
    static double beta_of_q(int l, int cutoff_index,
                            const std::vector<double>& beta_values,
                            const RadialGrid& mesh,
                            double q);
};

} // namespace kronos
