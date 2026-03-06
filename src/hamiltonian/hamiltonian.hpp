#pragma once
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "potential/hartree.hpp"
#include "potential/xc.hpp"
#include "potential/local_pp.hpp"
#include "potential/nonlocal_pp.hpp"
#include <vector>
#include <functional>

namespace kronos {

// The Kohn-Sham Hamiltonian operator
// H|psi> = T|psi> + V_eff|psi> + V_NL|psi>
// where V_eff(r) = V_H(r) + V_xc(r) + V_loc(r) (effective local potential)
class Hamiltonian {
public:
    Hamiltonian(const Crystal& crystal,
                const PlaneWaveBasis& basis,
                FFTGrid& fft_grid,
                NonlocalPP& nonlocal_pp);

    // Update the effective local potential V_eff(r) on the real-space grid
    // Called each SCF step after computing new Hartree + XC + local PP potentials
    // veff_r: V_eff on real-space grid points (size = fft_grid.total_points())
    void update_veff(const std::vector<complex_t>& veff_r);

    // Apply H to a wavefunction: H|psi> in G-space
    // psi_g: wavefunction in G-space (size = num_pw)
    // k_frac: k-point in fractional reciprocal coordinates
    // Returns H|psi> in G-space
    CVec apply(const CVec& psi_g, const Vec3& k_frac) const;

    // Get a std::function wrapper for use with the Davidson solver.
    // Precomputes and caches nonlocal projectors for this k-point.
    std::function<CVec(const CVec&)> get_apply_function(const Vec3& k_frac);

    // Get kinetic energy diagonal (used as preconditioner for Davidson)
    std::vector<double> kinetic_diagonal(const Vec3& k_frac) const;

private:
    const Crystal& crystal_;
    const PlaneWaveBasis& basis_;
    FFTGrid& fft_grid_;
    NonlocalPP& nonlocal_pp_;

    std::vector<complex_t> veff_r_;  // effective potential on real-space grid
};

} // namespace kronos
