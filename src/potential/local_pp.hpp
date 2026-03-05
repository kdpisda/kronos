#pragma once

#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "io/upf_parser.hpp"

#include <map>
#include <string>
#include <vector>

namespace kronos {

/// Local pseudopotential evaluator.
///
/// Computes the total local pseudopotential in reciprocal space
///   V_loc(G) = sum_species  V_loc^species(|G|) * S_species(G)
/// where S_species is the structure factor and V_loc^species(q) is the
/// radial Fourier transform of the local part of the pseudopotential.
class LocalPPEvaluator {
public:
    /// Construct and precompute V_loc(G) for every G-vector in the basis.
    LocalPPEvaluator(const Crystal& crystal,
                     const PlaneWaveBasis& basis,
                     const std::map<std::string, PseudoPotential>& pseudopotentials);

    /// The precomputed local potential in G-space (Ry).
    [[nodiscard]] const CVec& vloc_g() const;

    /// Local pseudopotential energy:
    ///   E_loc = Omega * sum_G  Re[ conj(V_loc(G)) * n(G) ]
    ///
    /// @param density_g  Electron density in G-space.
    /// @param volume     Unit cell volume in bohr^3.
    [[nodiscard]] double energy(const CVec& density_g, double volume,
                                int num_grid) const;

private:
    CVec vloc_g_;

    /// Radial Fourier transform of V_loc(r) at wavevector magnitude q,
    /// using Coulomb tail subtraction for numerical stability.
    ///
    /// V_loc(r) is split into a short-range part and a long-range
    /// analytic part:
    ///   V_loc_short(r) = V_loc(r) + Z_val * erf(r / r_loc) / r
    ///   V_loc_long(r)  = -Z_val * erf(r / r_loc) / r
    ///
    /// The short-range part is integrated numerically (converges fast),
    /// and the long-range Coulomb contribution is added analytically.
    ///
    /// @param pp      Pseudopotential data.
    /// @param q       Wavevector magnitude |G| in 1/bohr.
    /// @param volume  Unit cell volume in bohr^3.
    static double vloc_of_q(const PseudoPotential& pp, double q,
                            double volume);

    /// Structure factor for a set of atoms of a single species.
    ///
    /// S(G) = sum_j  exp(-i G . tau_j)
    ///
    /// @param positions_cart  Cartesian positions (bohr) of all atoms of
    ///                        the species.
    /// @param g_cart          Cartesian G-vector (1/bohr).
    static complex_t structure_factor(const std::vector<Vec3>& positions_cart,
                                      const Vec3& g_cart);
};

} // namespace kronos
