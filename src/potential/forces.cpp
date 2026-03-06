#include "potential/forces.hpp"
#include "core/constants.hpp"
#include "core/spherical_harmonics.hpp"
#include "utils/radial_integral.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

namespace kronos {

// ===================================================================
// Helper: spherical Bessel functions (duplicated from nonlocal_pp.cpp
// because they are in an anonymous namespace there)
// ===================================================================

namespace {

inline double sbessel_0(double x)
{
    if (std::abs(x) < 1.0e-12) return 1.0;
    return std::sin(x) / x;
}

inline double sbessel_1(double x)
{
    if (std::abs(x) < 1.0e-12) return 0.0;
    return std::sin(x) / (x * x) - std::cos(x) / x;
}

inline double sbessel_2(double x)
{
    if (std::abs(x) < 1.0e-12) return 0.0;
    const double x2 = x * x;
    return ((3.0 / x2 - 1.0) * std::sin(x) - 3.0 * std::cos(x) / x) / x;
}

inline double sbessel_3(double x)
{
    if (std::abs(x) < 1.0e-12) return 0.0;
    const double x2 = x * x;
    const double x3 = x2 * x;
    return ((15.0 / x3 - 6.0 / x) * std::sin(x)
            - (15.0 / x2 - 1.0) * std::cos(x)) / x;
}

double sbessel(int l, double x)
{
    switch (l) {
    case 0: return sbessel_0(x);
    case 1: return sbessel_1(x);
    case 2: return sbessel_2(x);
    case 3: return sbessel_3(x);
    default:
        break;
    }

    if (std::abs(x) < 1.0e-12) return 0.0;

    double jlm1 = sbessel_0(x);
    double jl   = sbessel_1(x);
    for (int ll = 1; ll < l; ++ll) {
        const double jlp1 = static_cast<double>(2 * ll + 1) / x * jl - jlm1;
        jlm1 = jl;
        jl = jlp1;
    }
    return jl;
}

/// Spherical Bessel transform of a beta projector at |q|.
/// UPF convention: beta.values stores r*beta(r), not beta(r).
/// The 3D radial FT integral is: integral r^2 beta(r) j_l(qr) dr
/// Since beta.values = r*beta(r): r^2 * beta(r) = r * beta.values
double beta_of_q(const BetaProjector& beta,
                 const RadialGrid& mesh,
                 double q)
{
    const int l = beta.angular_momentum;
    const int npts = std::min(beta.cutoff_index,
                              static_cast<int>(beta.values.size()));
    if (npts <= 0) return 0.0;

    std::vector<double> integrand(npts);
    for (int i = 0; i < npts; ++i) {
        const double ri = mesh.r[i];
        const double qr = q * ri;
        const double jl = sbessel(l, qr);
        // r * beta.values[i] = r * (r*beta(r)) = r^2 * beta(r)
        integrand[i] = ri * beta.values[i] * jl;
    }
    return simpson_radial(integrand, mesh.rab, npts);
}

} // anonymous namespace

// ===================================================================
// vloc_of_q  --  radial Fourier transform of V_loc(r)
// ===================================================================
//
// Identical to LocalPPEvaluator::vloc_of_q. Duplicated here because
// LocalPPEvaluator::vloc_of_q is private. We need the per-species
// form factor (without structure factor) for the force calculation.
// ===================================================================

double ForceCalculator::vloc_of_q(const PseudoPotential& pp, double q,
                                  double volume)
{
    const auto& r   = pp.mesh.r;
    const auto& rab = pp.mesh.rab;
    const auto& vloc = pp.vloc;
    const int npts = pp.mesh.npoints;
    const double z_val = pp.z_valence;
    const double z2 = 2.0 * z_val;  // factor of 2 for Rydberg units

    assert(static_cast<int>(r.size()) >= npts);
    assert(static_cast<int>(vloc.size()) >= npts);
    assert(static_cast<int>(rab.size()) >= npts);
    assert(npts > 0);

    const double r_loc = 1.0;  // bohr; must match LocalPPEvaluator::vloc_of_q

    std::vector<double> integrand(npts, 0.0);

    if (q < 1.0e-12) {
        for (int i = 0; i < npts; ++i) {
            const double ri = r[i];
            if (ri < 1.0e-30) continue;
            const double vloc_short_i = vloc[i] + z2 * std::erf(ri / r_loc) / ri;
            integrand[i] = ri * ri * vloc_short_i;
        }
        double integral_short = simpson_radial(integrand, rab, npts);

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

        const double q2 = q * q;
        const double vloc_long_q = -z2 * constants::four_pi / q2
                                   * std::exp(-q2 * r_loc * r_loc / 4.0);

        return (constants::four_pi * integral_short + vloc_long_q) / volume;
    }
}

// ===================================================================
// compute_local_forces
// ===================================================================
//
// The local pseudopotential energy is:
//   E_loc = Omega * sum_G conj(V_loc(G)) * n(G)
//
// where V_loc(G) = sum_species V_loc^s(|G|) * S_s(G)
// and S_s(G) = sum_{atoms of s} exp(-i G . tau_I).
//
// For the force on atom I (of species s), we differentiate E_loc w.r.t.
// the atomic position tau_I. The only tau_I dependence is in S_s(G):
//
//   d/dtau_I [exp(-i G . tau_I)] = -iG * exp(-i G . tau_I)
//
// So: dE_loc/dtau_I = Omega * sum_G conj(V_loc^s(|G|) * (-iG) * exp(-iG.tau_I)) * n(G)
//                   = Omega * sum_G V_loc^s(|G|) * n(G) * (iG) * exp(iG.tau_I)
//
// where we used conj(-iG * exp(-iG.tau)) = iG * exp(iG.tau).
//
// The force is F_I = -dE_loc/dtau_I, so:
//   F_local_I = -Omega * sum_G V_loc^s(|G|) * n(G) * iG * exp(iG.tau_I)
//
// The G=0 term is excluded because iG * exp(iG.tau_I) = 0 at G=0.
//
// Taking the real part (forces must be real):
//   F_local_I[d] = -Omega * sum_{G!=0} V_loc^s(|G|)
//                  * Re[ n(G) * i * G[d] * exp(iG.tau_I) ]
//
// Using n(G) = n_r + i*n_i and exp(iG.tau) = cos(G.tau) + i*sin(G.tau):
//   i * exp(iG.tau) = -sin(G.tau) + i*cos(G.tau)
//
//   Re[ (n_r + i*n_i) * (-sin + i*cos) * G[d] ]
//     = G[d] * (-n_r*sin - n_i*cos)
//     = -G[d] * (n_r*sin(G.tau) + n_i*cos(G.tau))
//
// So: F_local_I[d] = Omega * sum_{G!=0} V_loc^s(|G|) * G[d]
//                    * (n_r*sin(G.tau_I) + n_i*cos(G.tau_I))
// ===================================================================

std::vector<Vec3> ForceCalculator::compute_local_forces(
    const Crystal& crystal,
    const std::map<std::string, PseudoPotential>& pseudopotentials,
    const std::vector<complex_t>& density_g_full,
    const std::vector<Vec3>& grid_gcart,
    const std::vector<double>& grid_g2,
    double ecutrho,
    double volume,
    int num_grid)
{
    const size_t natoms = crystal.num_atoms();

    assert(static_cast<int>(density_g_full.size()) == num_grid);
    assert(static_cast<int>(grid_gcart.size()) == num_grid);
    assert(static_cast<int>(grid_g2.size()) == num_grid);
    assert(num_grid > 0);

    std::vector<Vec3> forces(natoms, {0.0, 0.0, 0.0});

    // Precompute Cartesian positions for each atom
    std::vector<Vec3> positions(natoms);
    for (size_t ia = 0; ia < natoms; ++ia) {
        positions[ia] = crystal.frac_to_cart(crystal.atom(ia).position);
    }

    // Loop over ALL G-vectors on the full FFT grid (matching the energy sum).
    // Only include G² ≤ ecutrho (same cutoff as vloc_full_g in the SCF loop).
    for (int ig = 0; ig < num_grid; ++ig) {
        if (grid_g2[ig] > ecutrho + 1.0e-6) continue;

        const double g_mag = std::sqrt(grid_g2[ig]);

        // Skip G=0: the derivative iG*exp(iG.tau_I) vanishes at G=0
        if (g_mag < 1.0e-12) continue;

        const Vec3& g_cart = grid_gcart[ig];
        const double n_r = density_g_full[ig].real();
        const double n_i = density_g_full[ig].imag();

        // For each atom, accumulate the force contribution from this G
        for (size_t ia = 0; ia < natoms; ++ia) {
            const std::string& symbol = crystal.atom(ia).symbol;
            auto pp_it = pseudopotentials.find(symbol);
            if (pp_it == pseudopotentials.end()) continue;

            // Per-species form factor V_loc^s(|G|) (normalized by volume)
            const double vloc_q = vloc_of_q(pp_it->second, g_mag, volume);

            // Phase: G . tau_I
            const double gdottau = g_cart[0] * positions[ia][0]
                                 + g_cart[1] * positions[ia][1]
                                 + g_cart[2] * positions[ia][2];
            const double sin_gt = std::sin(gdottau);
            const double cos_gt = std::cos(gdottau);

            // Force contribution (see derivation in header):
            // F[d] += Omega * vloc_q * G[d] * (n_r*sin(G.tau) + n_i*cos(G.tau))
            // Divide by num_grid for un-normalized FFT density coefficients.
            const double factor = volume * vloc_q
                                  * (n_r * sin_gt + n_i * cos_gt)
                                  / num_grid;

            for (int d = 0; d < 3; ++d) {
                forces[ia][d] += factor * g_cart[d];
            }
        }
    }

    return forces;
}

// ===================================================================
// compute_nonlocal_forces
// ===================================================================
//
// The nonlocal energy is:
//   E_NL = sum_{n,k} f_{n,k} w_k * spin
//          * sum_{a,i,j} D_{ij}^a <psi|beta_j^a> <beta_i^a|psi>
//
// The position of atom a enters through the phase factor in beta:
//   beta_{i,m}^a(G) = prefactor * i^l * radial(|G|) * Y_lm(G_hat) * exp(-iG.tau_a)
//
// The derivative w.r.t. tau_a:
//   d/dtau_a [beta_{i,m}^a(G)] = (-iG) * beta_{i,m}^a(G)
//   d/dtau_a [conj(beta_{j,m}^a(G))] = (iG) * conj(beta_{j,m}^a(G))
//
// Define:
//   P_j = <beta_j|psi> = sum_G beta_j^*(G) psi(G)
//   dP_j[d] = <d(beta_j)/dtau_d | psi> = i * sum_G G_d * beta_j^*(G) * psi(G)
//
// Then: F_a[d] = -dE_NL/dtau_a[d]
//              = -sum_{n,k} f w s * sum_{i,j} D_{ij} * 2 * Re[ conj(P_j) * dP_i[d] ]
//
// All indices here are in the expanded (l,m) basis.  Each UPF projector
// with angular momentum l generates (2l+1) expanded projectors, one per
// m = -l, ..., +l.  The D_ij is block-diagonal in m.
// ===================================================================

std::vector<Vec3> ForceCalculator::compute_nonlocal_forces(
    const Crystal& crystal,
    const PlaneWaveBasis& basis,
    const std::map<std::string, PseudoPotential>& pseudopotentials,
    const std::vector<std::vector<CVec>>& wavefunctions,
    const std::vector<std::vector<double>>& occupations,
    const std::vector<Vec3>& k_points,
    const std::vector<double>& k_weights,
    int /* spin_factor: unused, occupations already include spin */)
{
    const size_t natoms = crystal.num_atoms();
    const auto& gvecs = basis.gvectors();
    const size_t npw = gvecs.size();
    const double volume = crystal.volume();

    // Prefactor for projectors: 4*pi / sqrt(Omega)
    const double prefactor = constants::four_pi / std::sqrt(volume);

    // Powers of i
    static const complex_t il_table[4] = {
        {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}
    };

    // Reciprocal lattice for converting k_frac -> k_cart
    const Mat3 recip_lat = crystal.reciprocal_lattice();

    std::vector<Vec3> forces(natoms, {0.0, 0.0, 0.0});

    // Loop over k-points
    for (size_t ik = 0; ik < k_points.size(); ++ik) {
        const int num_bands = static_cast<int>(wavefunctions[ik].size());
        const Vec3& k_frac = k_points[ik];

        // Convert k from fractional to Cartesian (1/bohr)
        Vec3 k_cart{};
        for (int d = 0; d < 3; ++d) {
            k_cart[d] = k_frac[0] * recip_lat[0][d]
                      + k_frac[1] * recip_lat[1][d]
                      + k_frac[2] * recip_lat[2][d];
        }

        // Precompute k+G vectors and magnitudes for this k-point
        std::vector<Vec3> kpg_cart(npw);
        std::vector<double> kpg_mag(npw);
        for (size_t ig = 0; ig < npw; ++ig) {
            const Vec3& gc = gvecs[ig].cart;
            kpg_cart[ig] = {k_cart[0] + gc[0],
                            k_cart[1] + gc[1],
                            k_cart[2] + gc[2]};
            kpg_mag[ig] = std::sqrt(kpg_cart[ig][0] * kpg_cart[ig][0]
                                  + kpg_cart[ig][1] * kpg_cart[ig][1]
                                  + kpg_cart[ig][2] * kpg_cart[ig][2]);
        }

        // Loop over atoms
        for (size_t ia = 0; ia < natoms; ++ia) {
            const auto& atom = crystal.atom(ia);
            auto pp_it = pseudopotentials.find(atom.symbol);
            if (pp_it == pseudopotentials.end()) continue;

            const auto& pp = pp_it->second;
            if (pp.betas.empty()) continue;

            const int nproj_upf = static_cast<int>(pp.betas.size());
            const Vec3 tau = crystal.frac_to_cart(atom.position);

            // Count expanded projectors
            int nproj_expanded = 0;
            for (int ib = 0; ib < nproj_upf; ++ib) {
                nproj_expanded += 2 * pp.betas[ib].angular_momentum + 1;
            }

            // Build the expanded index mapping and D_ij matrix
            struct ExpandedIndex {
                int upf_beta_index;
                int l;
                int m;
            };
            std::vector<ExpandedIndex> expanded_map(nproj_expanded);

            {
                int ie = 0;
                for (int ib = 0; ib < nproj_upf; ++ib) {
                    const int l = pp.betas[ib].angular_momentum;
                    for (int m_val = -l; m_val <= l; ++m_val) {
                        expanded_map[ie].upf_beta_index = ib;
                        expanded_map[ie].l = l;
                        expanded_map[ie].m = m_val;
                        ++ie;
                    }
                }
            }

            // Build expanded D_ij (block-diagonal in m)
            std::vector<std::vector<double>> dij_expanded(
                nproj_expanded, std::vector<double>(nproj_expanded, 0.0));

            for (int ie_i = 0; ie_i < nproj_expanded; ++ie_i) {
                const auto& map_i = expanded_map[ie_i];
                for (int ie_j = 0; ie_j < nproj_expanded; ++ie_j) {
                    const auto& map_j = expanded_map[ie_j];
                    if (map_i.l == map_j.l && map_i.m == map_j.m) {
                        dij_expanded[ie_i][ie_j] =
                            pp.dij[map_i.upf_beta_index][map_j.upf_beta_index];
                    }
                }
            }

            // Precompute beta_{ie}(k+G) and dbeta_{ie}(k+G) for this atom
            // Key: must use k+G (not G) for radial, angular, phase, derivative
            std::vector<CVec> beta_kg_exp(nproj_expanded);
            std::vector<std::array<CVec, 3>> dbeta_kg_exp(nproj_expanded);

            {
                int ie = 0;
                for (int ib = 0; ib < nproj_upf; ++ib) {
                    const auto& beta = pp.betas[ib];
                    const int l = beta.angular_momentum;
                    const complex_t il = il_table[l % 4];

                    // Precompute radial transforms at |k+G| (same for all m)
                    std::vector<double> radial_cache(npw);
                    for (size_t ig = 0; ig < npw; ++ig) {
                        radial_cache[ig] = beta_of_q(beta, pp.mesh, kpg_mag[ig]);
                    }

                    for (int m_val = -l; m_val <= l; ++m_val) {
                        beta_kg_exp[ie].resize(npw);
                        for (int d = 0; d < 3; ++d) {
                            dbeta_kg_exp[ie][d].resize(npw);
                        }

                        for (size_t ig = 0; ig < npw; ++ig) {
                            const double radial = radial_cache[ig];

                            // Phase factor: exp(-i (k+G) . tau)
                            const double kpg_dot_tau =
                                kpg_cart[ig][0] * tau[0]
                              + kpg_cart[ig][1] * tau[1]
                              + kpg_cart[ig][2] * tau[2];
                            const complex_t phase{std::cos(kpg_dot_tau),
                                                  -std::sin(kpg_dot_tau)};

                            // Angular: Y_lm at (k+G) direction
                            double angular;
                            if (kpg_mag[ig] < 1.0e-12) {
                                angular = real_spherical_harmonic(
                                    l, m_val, 0.0, 0.0, 0.0);
                            } else {
                                angular = real_spherical_harmonic(
                                    l, m_val,
                                    kpg_cart[ig][0], kpg_cart[ig][1],
                                    kpg_cart[ig][2]);
                            }

                            const complex_t beta_val =
                                prefactor * il * radial * angular * phase;
                            beta_kg_exp[ie][ig] = beta_val;

                            // d(beta)/d(tau_d) = -i(k+G)_d * beta
                            const complex_t neg_i{0.0, -1.0};
                            for (int d = 0; d < 3; ++d) {
                                dbeta_kg_exp[ie][d][ig] =
                                    neg_i * kpg_cart[ig][d] * beta_val;
                            }
                        }

                        ++ie;
                    }
                }
            }

            // Loop over bands
            // Note: occupations already include spin_factor (from FermiSolver),
            // so we must NOT multiply by spin_factor again here.
            for (int n = 0; n < num_bands; ++n) {
                const double occ = k_weights[ik] * occupations[ik][n];
                if (std::abs(occ) < 1.0e-15) continue;

                const auto& psi = wavefunctions[ik][n];

                // Compute P_j = <beta_j|psi> for expanded projectors
                std::vector<complex_t> proj(nproj_expanded,
                                            complex_t{0.0, 0.0});
                for (int j = 0; j < nproj_expanded; ++j) {
                    complex_t sum{0.0, 0.0};
                    for (size_t ig = 0; ig < npw; ++ig) {
                        sum += std::conj(beta_kg_exp[j][ig]) * psi[ig];
                    }
                    proj[j] = sum;
                }

                // Compute dP_i[d] = <d(beta_i)/dtau_d | psi>
                std::vector<std::array<complex_t, 3>> dproj(nproj_expanded);
                for (int i = 0; i < nproj_expanded; ++i) {
                    for (int d = 0; d < 3; ++d) {
                        complex_t sum{0.0, 0.0};
                        for (size_t ig = 0; ig < npw; ++ig) {
                            sum += std::conj(dbeta_kg_exp[i][d][ig])
                                   * psi[ig];
                        }
                        dproj[i][d] = sum;
                    }
                }

                // Accumulate force:
                // F_a[d] -= occ * sum_{i,j} D_{ij} * 2 * Re[conj(P_j) * dP_i[d]]
                for (int d = 0; d < 3; ++d) {
                    double f_d = 0.0;
                    for (int i = 0; i < nproj_expanded; ++i) {
                        for (int j = 0; j < nproj_expanded; ++j) {
                            const complex_t term =
                                std::conj(proj[j]) * dproj[i][d];
                            f_d += dij_expanded[i][j] * term.real();
                        }
                    }
                    forces[ia][d] -= occ * 2.0 * f_d;
                }
            }
        }
    }

    return forces;
}

// ===================================================================
// compute_total_forces
// ===================================================================

std::vector<Vec3> ForceCalculator::compute_total_forces(
    const std::vector<Vec3>& ewald_forces,
    const std::vector<Vec3>& local_forces,
    const std::vector<Vec3>& nonlocal_forces)
{
    assert(ewald_forces.size() == local_forces.size());
    assert(ewald_forces.size() == nonlocal_forces.size());

    const size_t natoms = ewald_forces.size();
    std::vector<Vec3> total(natoms);

    for (size_t ia = 0; ia < natoms; ++ia) {
        for (int d = 0; d < 3; ++d) {
            total[ia][d] = ewald_forces[ia][d]
                         + local_forces[ia][d]
                         + nonlocal_forces[ia][d];
        }
    }

    return total;
}

} // namespace kronos
