#pragma once

#include "core/types.hpp"
#include "basis/plane_wave.hpp"

namespace kronos {

/// Hartree potential solver.
///
/// Computes the Coulomb potential arising from the electron charge density
/// by solving the Poisson equation in reciprocal space.
///
/// In Rydberg atomic units the relation is:
///   V_H(G) = 8*pi * n(G) / |G|^2   for G != 0
///   V_H(G=0) = 0                     (arbitrary constant; cancels with
///                                      the ion-electron G=0 term)
class HartreeSolver {
public:
    explicit HartreeSolver(const PlaneWaveBasis& basis);

    /// Compute the Hartree potential in G-space from the density in G-space.
    ///
    /// @param density_g  Electron density Fourier coefficients n(G),
    ///                   indexed consistently with basis.gvectors().
    /// @return           V_H(G) in Rydberg atomic units.
    [[nodiscard]] CVec compute(const CVec& density_g) const;

    /// Compute the Hartree energy.
    ///
    /// E_H = (Omega / 2) * sum_G  conj(V_H(G)) * n(G)
    ///
    /// @param density_g   n(G)
    /// @param vhartree_g  V_H(G) as returned by compute()
    /// @param volume      Unit cell volume Omega in bohr^3
    /// @return            Hartree energy in Ry.
    [[nodiscard]] double energy(const CVec& density_g,
                                const CVec& vhartree_g,
                                double volume,
                                int num_grid) const;

private:
    const PlaneWaveBasis& basis_;
};

} // namespace kronos
