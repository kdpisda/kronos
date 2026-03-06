#include "potential/local_pp.hpp"
#include "core/constants.hpp"
#include "utils/radial_integral.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace kronos {

// ===================================================================
// Construction  --  precompute V_loc(G) for all G-vectors
// ===================================================================

LocalPPEvaluator::LocalPPEvaluator(
    const Crystal& crystal,
    const PlaneWaveBasis& basis,
    const std::map<std::string, PseudoPotential>& pseudopotentials)
{
    const auto& gvecs = basis.gvectors();
    const size_t npw = gvecs.size();
    const double volume = crystal.volume();

    vloc_g_.assign(npw, complex_t{0.0, 0.0});

    // Group atom indices by species so we can compute S(G) per species.
    // species_positions[symbol] = vector of Cartesian positions (bohr)
    std::map<std::string, std::vector<Vec3>> species_positions;
    for (size_t ia = 0; ia < crystal.num_atoms(); ++ia) {
        const auto& atom = crystal.atom(ia);
        Vec3 cart = crystal.frac_to_cart(atom.position);
        species_positions[atom.symbol].push_back(cart);
    }

    // For each G-vector, accumulate contributions from all species
    for (size_t ig = 0; ig < npw; ++ig) {
        const Vec3& g_cart = gvecs[ig].cart;
        const double g_mag = std::sqrt(gvecs[ig].norm2);

        complex_t vloc_total{0.0, 0.0};

        for (const auto& [symbol, positions] : species_positions) {
            auto pp_it = pseudopotentials.find(symbol);
            if (pp_it == pseudopotentials.end()) {
                // No pseudopotential for this species -- skip
                continue;
            }

            // Radial Fourier transform of V_loc at |G|
            const double vloc_q = vloc_of_q(pp_it->second, g_mag, volume);

            // Structure factor for this species at G
            const complex_t sf = structure_factor(positions, g_cart);

            vloc_total += vloc_q * sf;
        }

        vloc_g_[ig] = vloc_total;
    }
}

// ===================================================================
// vloc_g accessor
// ===================================================================

const CVec& LocalPPEvaluator::vloc_g() const
{
    return vloc_g_;
}

// ===================================================================
// energy  --  E_loc = Omega * sum_G Re[ conj(V_loc(G)) * n(G) ]
// ===================================================================

double LocalPPEvaluator::energy(const CVec& density_g, double volume,
                                int num_grid) const
{
    assert(density_g.size() == vloc_g_.size());
    assert(num_grid > 0);

    double esum = 0.0;
    for (size_t ig = 0; ig < vloc_g_.size(); ++ig) {
        esum += std::real(std::conj(vloc_g_[ig]) * density_g[ig]);
    }
    // Divide by N_grid to account for un-normalized FFT density coefficients.
    return volume * esum / num_grid;
}

// ===================================================================
// vloc_of_q  --  radial Fourier transform of V_loc(r)
// ===================================================================
//
// Uses Coulomb tail subtraction for numerical stability.
//
// In Rydberg units, V_loc(r) -> -2*Z_val / r for large r.
// Direct numerical FT of a 1/r tail converges very slowly.  We split:
//
//   V_loc_short(r) = V_loc(r) + 2*Z_val * erf(r / r_loc) / r  (short-range)
//   V_loc_long(r)  = -2*Z_val * erf(r / r_loc) / r             (long-range)
//
// The short-range part is numerically Fourier-transformed (converges
// fast because V_loc_short -> 0 for large r), and V_loc_long has an
// analytic FT:
//
//   FT[V_loc_long](q) = -2*Z_val * 4*pi / q^2 * exp(-q^2 * r_loc^2 / 4)
//   FT[V_loc_long](0) = -2*Z_val * pi * r_loc^2
//
// Then: V_loc(q) = [4*pi * integral_short + FT_long(q)] / Omega
//
// r_loc = 1.0 bohr ensures erf(r/r_loc) ≈ 1 in the Coulomb tail region.
// ===================================================================

double LocalPPEvaluator::vloc_of_q(const PseudoPotential& pp, double q,
                                   double volume)
{
    const auto& r   = pp.mesh.r;
    const auto& rab = pp.mesh.rab;
    const auto& vloc = pp.vloc;
    const int npts = pp.mesh.npoints;
    const double z_val = pp.z_valence;

    assert(static_cast<int>(r.size()) >= npts);
    assert(static_cast<int>(vloc.size()) >= npts);
    assert(static_cast<int>(rab.size()) >= npts);
    assert(npts > 0);

    // Choose r_loc: controls the erf splitting between short-range
    // (numerically integrated) and long-range (analytic) parts.
    // Must be small enough that erf(r/r_loc) ≈ 1 in the Coulomb tail
    // region (r > ~3 bohr for most elements), ensuring rapid convergence
    // of the short-range integral.
    const double r_loc = 1.0;  // bohr; standard choice for NC pseudopotentials

    // In Rydberg units, the Coulomb potential is -2Z/r (not -Z/r).
    // The long-range part we subtract analytically is:
    //   V_long(r) = -2*Z_val * erf(r/r_loc) / r
    //
    // The short-range remainder is:
    //   V_short(r) = V_loc(r) - V_long(r) = V_loc(r) + 2*Z_val * erf(r/r_loc) / r
    const double z2 = 2.0 * z_val;  // factor of 2 for Rydberg units

    // Build integrand array for Simpson's rule
    std::vector<double> integrand(npts, 0.0);

    if (q < 1.0e-12) {
        // q = 0: sinc(0) = 1
        for (int i = 0; i < npts; ++i) {
            const double ri = r[i];
            if (ri < 1.0e-30) continue;
            const double vloc_short_i = vloc[i] + z2 * std::erf(ri / r_loc) / ri;
            integrand[i] = ri * ri * vloc_short_i;
        }
        double integral_short = simpson_radial(integrand, rab, npts);

        // Analytic FT of V_long at q=0 (regularized).
        //
        // FT[-2Z*erf(r/σ)/r](q) = -8πZ * exp(-q²σ²/4) / q²
        //                        = -8πZ/q² + 2πZσ² + O(q²)
        //
        // The -8πZ/q² divergence cancels with the Hartree G=0 (set to 0)
        // and the Ewald charged correction.  The finite remainder that
        // must be kept is +2πZσ² = +z2 * π * σ².
        return (constants::four_pi * integral_short
                + z2 * constants::pi * r_loc * r_loc) / volume;
    } else {
        for (int i = 0; i < npts; ++i) {
            const double ri = r[i];
            if (ri < 1.0e-30) continue;
            const double vloc_short_i = vloc[i] + z2 * std::erf(ri / r_loc) / ri;
            const double qr = q * ri;
            const double sinc_qr = std::sin(qr) / qr;
            integrand[i] = ri * ri * vloc_short_i * sinc_qr;
        }
        double integral_short = simpson_radial(integrand, rab, npts);

        // Analytic FT of V_long at q != 0:
        //   FT[-2Z*erf(r/r_loc)/r](q) = -2Z * 4*pi / q^2 * exp(-q^2 * r_loc^2 / 4)
        const double q2 = q * q;
        const double vloc_long_q = -z2 * constants::four_pi / q2
                                   * std::exp(-q2 * r_loc * r_loc / 4.0);

        return (constants::four_pi * integral_short + vloc_long_q) / volume;
    }
}

// ===================================================================
// structure_factor  --  S(G) = sum_j exp(-i G . tau_j)
// ===================================================================

complex_t LocalPPEvaluator::structure_factor(
    const std::vector<Vec3>& positions_cart,
    const Vec3& g_cart)
{
    complex_t sf{0.0, 0.0};

    for (const auto& tau : positions_cart) {
        const double gdottau = g_cart[0] * tau[0]
                             + g_cart[1] * tau[1]
                             + g_cart[2] * tau[2];
        // exp(-i * G . tau)
        sf += complex_t{std::cos(gdottau), -std::sin(gdottau)};
    }

    return sf;
}

} // namespace kronos
