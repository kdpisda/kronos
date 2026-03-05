#include "potential/hartree.hpp"
#include "core/constants.hpp"

#include <cassert>
#include <cmath>

namespace kronos {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

HartreeSolver::HartreeSolver(const PlaneWaveBasis& basis)
    : basis_(basis)
{}

// ---------------------------------------------------------------------------
// compute  --  V_H(G) = 8*pi * n(G) / |G|^2   (Rydberg units)
// ---------------------------------------------------------------------------

CVec HartreeSolver::compute(const CVec& density_g) const
{
    const auto& gvecs = basis_.gvectors();
    const size_t npw = gvecs.size();

    assert(density_g.size() == npw);

    // Pre-factor: 8*pi  (= 2 * 4*pi, the factor of 2 converts Ha -> Ry)
    constexpr double prefactor = 2.0 * constants::four_pi;

    CVec vhartree(npw, complex_t{0.0, 0.0});

    for (size_t ig = 0; ig < npw; ++ig) {
        const double g2 = gvecs[ig].norm2;

        if (g2 < 1.0e-12) {
            // G = 0 component: set to zero (arbitrary constant that cancels
            // with the ionic G=0 contribution in the total energy).
            vhartree[ig] = complex_t{0.0, 0.0};
        } else {
            vhartree[ig] = prefactor * density_g[ig] / g2;
        }
    }

    return vhartree;
}

// ---------------------------------------------------------------------------
// energy  --  E_H = (Omega / 2) * sum_G  conj(V_H(G)) * n(G)
// ---------------------------------------------------------------------------

double HartreeSolver::energy(const CVec& density_g,
                             const CVec& vhartree_g,
                             double volume,
                             int num_grid) const
{
    const size_t npw = density_g.size();
    assert(vhartree_g.size() == npw);
    assert(num_grid > 0);

    double esum = 0.0;
    for (size_t ig = 0; ig < npw; ++ig) {
        // conj(V_H) * n  -- take the real part
        esum += std::real(std::conj(vhartree_g[ig]) * density_g[ig]);
    }

    // Both density_g and vhartree_g are in "FFT convention" (each carries a
    // factor of N_grid relative to the physics convention).  The product
    // conj(V_H)*n is therefore N_grid^2 too large; divide by N_grid^2.
    const double n2 = static_cast<double>(num_grid) * static_cast<double>(num_grid);
    return 0.5 * volume * esum / n2;
}

} // namespace kronos
