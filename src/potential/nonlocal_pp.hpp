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

    /// Apply V_NL to a single wavefunction in G-space:
    ///   (V_NL |psi>)_G = sum_{a,i,j} D_{ij} <beta_j|psi> beta_i(G)
    ///
    /// The projectors are computed using |k+G| (not |G|) and the phase
    /// factor uses exp(-i(k+G).tau).
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
