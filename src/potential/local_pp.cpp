#include "potential/local_pp.hpp"
#include "core/constants.hpp"

#include <cassert>
#include <cmath>

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
// V_loc(r) -> -Z_val / r  for large r.  Direct numerical FT of a
// 1/r tail converges very slowly and is inaccurate.  We split:
//
//   V_loc_short(r) = V_loc(r) - V_loc_long(r)       (short-range)
//   V_loc_long(r)  = -Z_val * erf(r / r_loc) / r    (long-range)
//
// The short-range part is numerically Fourier-transformed (converges
// fast because V_loc_short -> 0 for large r), and V_loc_long has an
// analytic FT:
//
//   FT[V_loc_long](q) = -Z_val * 4*pi / q^2  * exp(-q^2 * r_loc^2 / 4)
//   FT[V_loc_long](0) = -Z_val * pi * r_loc^2
//
// Then: V_loc(q) = [V_loc_short_num(q) + V_loc_long_analytic(q)] / Omega
//
// r_loc is chosen as r[npoints-1] / 5.0 following standard practice.
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

    // Choose r_loc: a fraction of the outermost radial grid point.
    // This ensures erf(r/r_loc) ~ 1 well within the grid, making
    // V_loc_short(r) decay rapidly.
    const double r_loc = r[npts - 1] / 5.0;

    // Numerical integration of the short-range part:
    //   V_loc_short_num(q) = 4*pi * integral r^2
    //       [V_loc(r) + Z_val * erf(r/r_loc) / r] * sinc(qr) * rab * dr
    //
    // Note: V_loc_short(r) = V_loc(r) - V_loc_long(r)
    //     = V_loc(r) - (-Z_val * erf(r/r_loc)/r)
    //     = V_loc(r) + Z_val * erf(r/r_loc)/r

    double integral_short = 0.0;

    if (q < 1.0e-12) {
        // q = 0: sinc(0) = 1
        for (int i = 0; i < npts; ++i) {
            const double ri = r[i];
            // For r -> 0: erf(r/r_loc)/r -> 2/(sqrt(pi)*r_loc), so
            // r^2 * Z_val * erf(r/r_loc)/r -> 0. Safe to skip r=0.
            if (ri < 1.0e-30) continue;
            const double vloc_short_i = vloc[i] + z_val * std::erf(ri / r_loc) / ri;
            integral_short += ri * ri * vloc_short_i * rab[i];
        }

        // Analytic FT of the long-range part at q=0:
        //   FT[-Z*erf(r/r_loc)/r](q=0) = -Z * pi * r_loc^2
        const double vloc_long_0 = -z_val * constants::pi * r_loc * r_loc;

        // Combine: V_loc(0) = (4*pi * integral_short + 4*pi * vloc_long_0) / Omega
        //        = 4*pi * (integral_short + vloc_long_0) / Omega
        // Note: vloc_long_0 already has the 4*pi factored out in the
        // standard derivation. Let's be precise:
        //
        // FT_full[f](q) = 4*pi * integral r^2 f(r) sinc(qr) dr
        // For the long-range part alone:
        //   FT_full[V_long](0) = 4*pi * integral r^2 (-Z*erf(r/r_loc)/r) dr
        //                      = -Z * 4*pi * integral r * erf(r/r_loc) dr
        //   = -Z * pi * r_loc^2  (standard result)
        //
        // So: total = [4*pi * integral_short + (-Z * pi * r_loc^2)] / Omega
        return (constants::four_pi * integral_short
                - z_val * constants::pi * r_loc * r_loc) / volume;
    } else {
        for (int i = 0; i < npts; ++i) {
            const double ri = r[i];
            if (ri < 1.0e-30) continue;
            const double vloc_short_i = vloc[i] + z_val * std::erf(ri / r_loc) / ri;
            const double qr = q * ri;
            const double sinc_qr = std::sin(qr) / qr;
            integral_short += ri * ri * vloc_short_i * sinc_qr * rab[i];
        }

        // Analytic FT of the long-range part at q != 0:
        //   FT[-Z*erf(r/r_loc)/r](q) = -Z * 4*pi / q^2 * exp(-q^2 * r_loc^2 / 4)
        const double q2 = q * q;
        const double vloc_long_q = -z_val * constants::four_pi / q2
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
