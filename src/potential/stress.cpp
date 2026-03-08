#include "potential/stress.hpp"
#include "core/constants.hpp"
#include "core/spherical_harmonics.hpp"
#include "utils/radial_integral.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

namespace kronos {

// ===================================================================
// Helper: zero-initialize a Mat3
// ===================================================================
static Mat3 zero_mat3()
{
    return {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
}

// ===================================================================
// Helper: spherical Bessel functions (same as in forces.cpp)
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
        integrand[i] = ri * beta.values[i] * jl;
    }
    return simpson_radial(integrand, mesh.rab, npts);
}

/// Derivative d(beta_of_q)/d(q) — needed for nonlocal stress.
/// Uses j_l'(x) = (l/x)*j_l(x) - j_{l+1}(x).
double dbeta_of_q_dq(const BetaProjector& beta,
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
        // d/dq [ j_l(qr) ] = r * j_l'(qr) = r * [l/(qr) * j_l(qr) - j_{l+1}(qr)]
        double djl;
        if (std::abs(qr) < 1.0e-12) {
            // Taylor expansion: j_l(x) ~ x^l / (2l+1)!! so
            // d(j_l(qr))/dq = r * d(j_l)/d(qr) -> 0 for l > 0 at q=0
            // For l=0: j_0(x)=sinc(x), j_0'(0) = 0
            djl = 0.0;
        } else {
            const double jl = sbessel(l, qr);
            const double jl1 = sbessel(l + 1, qr);
            djl = ri * (static_cast<double>(l) / qr * jl - jl1);
        }
        integrand[i] = ri * beta.values[i] * djl;
    }
    return simpson_radial(integrand, mesh.rab, npts);
}

// ===================================================================
// vec3 helpers (local)
// ===================================================================

inline double vec3_dot(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

} // anonymous namespace

// ===================================================================
// vloc_of_q — identical to ForceCalculator::vloc_of_q
// ===================================================================

double StressCalculator::vloc_of_q(const PseudoPotential& pp, double q,
                                    double volume)
{
    const auto& r   = pp.mesh.r;
    const auto& rab = pp.mesh.rab;
    const auto& vloc = pp.vloc;
    const int npts = pp.mesh.npoints;
    const double z_val = pp.z_valence;
    const double z2 = 2.0 * z_val;

    const double r_loc = 1.0;

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
// dvloc_dq2 — derivative d(V_loc(q)*Omega)/d(q^2) / Omega
//
// We compute this numerically via finite differences for robustness:
//   d(vloc)/d(q^2) ≈ [vloc(q+dq) - vloc(q-dq)] / (2*dq) * dq/d(q^2)
//                   = [vloc(q+dq) - vloc(q-dq)] / (2*dq) * 1/(2*q)
// ===================================================================

double StressCalculator::dvloc_dq2(const PseudoPotential& pp, double q,
                                    double volume)
{
    // Use finite difference: d(vloc)/d(q^2) = d(vloc)/dq * 1/(2q)
    // d(vloc)/dq ≈ [vloc(q+h) - vloc(q-h)] / (2h)
    if (q < 1.0e-8) return 0.0;  // G=0 excluded from stress sums

    const double h = std::max(1.0e-4, q * 1.0e-4);
    const double vp = vloc_of_q(pp, q + h, volume);
    const double vm = vloc_of_q(pp, q - h, volume);
    const double dvdq = (vp - vm) / (2.0 * h);
    return dvdq / (2.0 * q);
}

// ===================================================================
// compute_kinetic_stress
// ===================================================================

Mat3 StressCalculator::compute_kinetic_stress(
    const Crystal& crystal,
    const PlaneWaveBasis& basis,
    const std::vector<std::vector<CVec>>& wavefunctions,
    const std::vector<std::vector<double>>& occupations,
    const std::vector<Vec3>& k_points,
    const std::vector<double>& k_weights)
{
    Mat3 stress = zero_mat3();
    const auto& gvecs = basis.gvectors();
    const size_t npw = gvecs.size();
    const double volume = crystal.volume();
    const Mat3 recip_lat = crystal.reciprocal_lattice();

    for (size_t ik = 0; ik < k_points.size(); ++ik) {
        const Vec3& k_frac = k_points[ik];

        // k in Cartesian
        Vec3 k_cart{};
        for (int d = 0; d < 3; ++d) {
            k_cart[d] = k_frac[0] * recip_lat[0][d]
                      + k_frac[1] * recip_lat[1][d]
                      + k_frac[2] * recip_lat[2][d];
        }

        // Precompute k+G
        std::vector<Vec3> kpg(npw);
        for (size_t ig = 0; ig < npw; ++ig) {
            const Vec3& gc = gvecs[ig].cart;
            kpg[ig] = {k_cart[0] + gc[0], k_cart[1] + gc[1], k_cart[2] + gc[2]};
        }

        const int num_bands = static_cast<int>(wavefunctions[ik].size());
        for (int n = 0; n < num_bands; ++n) {
            const double occ = k_weights[ik] * occupations[ik][n];
            if (std::abs(occ) < 1.0e-15) continue;

            const auto& psi = wavefunctions[ik][n];
            for (size_t ig = 0; ig < npw; ++ig) {
                const double c2 = std::norm(psi[ig]);  // |c_{nk}(G)|^2
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        stress[a][b] -= occ * kpg[ig][a] * kpg[ig][b] * c2;
                    }
                }
            }
        }
    }

    // Divide by volume
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            stress[a][b] /= volume;

    return stress;
}

// ===================================================================
// compute_hartree_stress
// ===================================================================

Mat3 StressCalculator::compute_hartree_stress(
    const std::vector<complex_t>& density_g_full,
    const std::vector<Vec3>& grid_gcart,
    const std::vector<double>& grid_g2,
    double ecutrho,
    double volume,
    int num_grid)
{
    Mat3 stress = zero_mat3();

    // Prefactor: 8*pi (Rydberg units: V_H = 8*pi*n/G^2)
    constexpr double prefactor = 2.0 * constants::four_pi;  // 8*pi
    const double ng2 = static_cast<double>(num_grid) * static_cast<double>(num_grid);

    for (int ig = 0; ig < num_grid; ++ig) {
        if (grid_g2[ig] > ecutrho + 1.0e-6) continue;
        if (grid_g2[ig] < 1.0e-12) continue;  // skip G=0

        const double g2 = grid_g2[ig];
        const double n2 = std::norm(density_g_full[ig]);  // |n(G)|^2
        const Vec3& gc = grid_gcart[ig];

        // sigma_ab = (1/Omega) * 8*pi * |n(G)|^2 * (delta_ab/(2*G^2) - G_a*G_b/G^4)
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double delta_ab = (a == b) ? 1.0 : 0.0;
                stress[a][b] += prefactor * n2
                    * (delta_ab / (2.0 * g2) - gc[a] * gc[b] / (g2 * g2));
            }
        }
    }

    // Divide by volume and N_grid^2 (un-normalized FFT)
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            stress[a][b] /= (volume * ng2);

    return stress;
}

// ===================================================================
// compute_xc_stress (LDA — isotropic)
// ===================================================================

Mat3 StressCalculator::compute_xc_stress(
    double exc_energy,
    const RVec& vxc_r,
    const RVec& density_r,
    double volume,
    int num_grid)
{
    Mat3 stress = zero_mat3();

    // Integral: int v_xc(r) * n(r) dr = (Omega/N) * sum_i vxc[i]*n[i]
    double vxc_n_sum = 0.0;
    for (int i = 0; i < num_grid; ++i) {
        vxc_n_sum += vxc_r[i] * density_r[i];
    }
    double int_vxc_n = vxc_n_sum * volume / num_grid;

    // sigma_ab = -delta_ab * (E_xc - int v_xc * n dr) / Omega
    // Note: for LDA, (E_xc - int v_xc*n dr) < 0 typically, so the
    // contribution is positive on the diagonal.
    double diag_val = -(exc_energy - int_vxc_n) / volume;

    for (int a = 0; a < 3; ++a) {
        stress[a][a] = diag_val;
    }

    return stress;
}

// ===================================================================
// compute_local_stress
// ===================================================================
//
// The local PP energy is E_loc = Omega * sum_G conj(V_loc(G)) * n(G) / N_grid
// where V_loc(G) = sum_s vloc_s(|G|) * S_s(G).
//
// The strain derivative dE_loc/d(epsilon_ab) gives:
//   sigma_ab = (1/Omega) * dE_loc/d(epsilon_ab)
//
// For the full derivation, the key identity is:
//   dG_a/d(epsilon_cd) = -delta_ac * G_d
// and the G-vector magnitudes change under strain via:
//   d(G^2)/d(epsilon_ab) = -2 * G_a * G_b
//
// The local stress has two contributions:
// 1. A "diagonal" term from the volume derivative: -E_loc_G * delta_ab
// 2. A "G-derivative" term from the |G| dependence of vloc_s(|G|)
//
// Combined:
//   sigma_ab^loc = sum_{G!=0} Re[conj(n(G)) * V_loc(G)] * delta_ab / N_grid
//                + sum_{G!=0} Re[conj(n(G)) * sum_s dV_s/d(G^2) * S_s(G)]
//                  * (-2*G_a*G_b) / N_grid
//
// Where dV_s/d(G^2) = d(vloc_s(|G|)*Omega) / d(G^2) / Omega
// ===================================================================

Mat3 StressCalculator::compute_local_stress(
    const Crystal& crystal,
    const std::map<std::string, PseudoPotential>& pseudopotentials,
    const std::vector<complex_t>& density_g_full,
    const std::vector<complex_t>& vloc_full_g,
    const std::vector<Vec3>& grid_gcart,
    const std::vector<double>& grid_g2,
    double ecutrho,
    double volume,
    int num_grid)
{
    Mat3 stress = zero_mat3();

    // Group atoms by species with their Cartesian positions
    std::map<std::string, std::vector<Vec3>> species_positions;
    for (size_t ia = 0; ia < crystal.num_atoms(); ++ia) {
        const auto& atom = crystal.atom(ia);
        Vec3 cart = crystal.frac_to_cart(atom.position);
        species_positions[atom.symbol].push_back(cart);
    }

    for (int ig = 0; ig < num_grid; ++ig) {
        if (grid_g2[ig] > ecutrho + 1.0e-6) continue;
        const double g_mag = std::sqrt(grid_g2[ig]);
        if (g_mag < 1.0e-12) continue;  // skip G=0

        const Vec3& gc = grid_gcart[ig];

        // Term 1: Re[conj(n(G)) * V_loc(G)] for the diagonal part
        const double n_vloc = std::real(
            std::conj(density_g_full[ig]) * vloc_full_g[ig]);

        // Term 2: Re[conj(n(G)) * sum_s dV_s/d(G^2) * S_s(G)] for the off-diagonal part
        // Compute sum_s dV_s/d(G^2) * S_s(G)
        complex_t dvloc_sf{0.0, 0.0};
        for (const auto& [symbol, positions] : species_positions) {
            auto pp_it = pseudopotentials.find(symbol);
            if (pp_it == pseudopotentials.end()) continue;

            const double dv = dvloc_dq2(pp_it->second, g_mag, volume);

            // Structure factor S(G) = sum_j exp(-i G.tau_j)
            complex_t sf{0.0, 0.0};
            for (const auto& tau : positions) {
                const double gdottau = gc[0] * tau[0] + gc[1] * tau[1] + gc[2] * tau[2];
                sf += complex_t{std::cos(gdottau), -std::sin(gdottau)};
            }

            dvloc_sf += dv * sf;
        }

        const double n_dvloc = std::real(
            std::conj(density_g_full[ig]) * dvloc_sf);

        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double delta_ab = (a == b) ? 1.0 : 0.0;
                // Diagonal term + G-derivative term
                stress[a][b] += n_vloc * delta_ab
                              + n_dvloc * (-2.0 * gc[a] * gc[b]);
            }
        }
    }

    // Factor: Omega / N_grid for un-normalized FFT, then /Omega for stress = (1/Omega)*dE/deps
    // Net: 1/N_grid
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            stress[a][b] /= num_grid;

    return stress;
}

// ===================================================================
// compute_nonlocal_stress
// ===================================================================
//
// Similar to nonlocal forces but instead of d(beta)/d(tau), we need
// d(<psi|V_NL|psi>)/d(epsilon_ab).
//
// The strain derivative acts on the projectors beta(k+G):
//   d/d(eps_ab) beta(k+G) depends on d|k+G|/d(eps_ab)
//   and d(k+G)_c/d(eps_ab) = -delta_ca * (k+G)_b
//
// For the nonlocal stress we compute:
//   sigma_ab = -(1/Omega) sum_{n,k} f w sum_{a,i,j} D_{ij}
//              * 2*Re[ conj(P_j) * dP_i^{ab} ]
//
// where dP_i^{ab} involves the derivative of beta w.r.t. strain.
// ===================================================================

Mat3 StressCalculator::compute_nonlocal_stress(
    const Crystal& crystal,
    const PlaneWaveBasis& basis,
    const std::map<std::string, PseudoPotential>& pseudopotentials,
    const std::vector<std::vector<CVec>>& wavefunctions,
    const std::vector<std::vector<double>>& occupations,
    const std::vector<Vec3>& k_points,
    const std::vector<double>& k_weights)
{
    Mat3 stress = zero_mat3();
    const auto& gvecs = basis.gvectors();
    const size_t npw = gvecs.size();
    const double volume = crystal.volume();
    const size_t natoms = crystal.num_atoms();

    const double prefactor = constants::four_pi / std::sqrt(volume);

    static const complex_t il_table[4] = {
        {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}
    };

    const Mat3 recip_lat = crystal.reciprocal_lattice();

    for (size_t ik = 0; ik < k_points.size(); ++ik) {
        const int num_bands = static_cast<int>(wavefunctions[ik].size());
        const Vec3& k_frac = k_points[ik];

        Vec3 k_cart{};
        for (int d = 0; d < 3; ++d) {
            k_cart[d] = k_frac[0] * recip_lat[0][d]
                      + k_frac[1] * recip_lat[1][d]
                      + k_frac[2] * recip_lat[2][d];
        }

        std::vector<Vec3> kpg_cart(npw);
        std::vector<double> kpg_mag(npw);
        for (size_t ig = 0; ig < npw; ++ig) {
            const Vec3& gc = gvecs[ig].cart;
            kpg_cart[ig] = {k_cart[0] + gc[0], k_cart[1] + gc[1], k_cart[2] + gc[2]};
            kpg_mag[ig] = std::sqrt(kpg_cart[ig][0] * kpg_cart[ig][0]
                                  + kpg_cart[ig][1] * kpg_cart[ig][1]
                                  + kpg_cart[ig][2] * kpg_cart[ig][2]);
        }

        for (size_t ia = 0; ia < natoms; ++ia) {
            const auto& atom = crystal.atom(ia);
            auto pp_it = pseudopotentials.find(atom.symbol);
            if (pp_it == pseudopotentials.end()) continue;

            const auto& pp = pp_it->second;
            if (pp.betas.empty()) continue;

            const int nproj_upf = static_cast<int>(pp.betas.size());
            const Vec3 tau = crystal.frac_to_cart(atom.position);

            int nproj_expanded = 0;
            for (int ib = 0; ib < nproj_upf; ++ib) {
                nproj_expanded += 2 * pp.betas[ib].angular_momentum + 1;
            }

            struct ExpandedIndex { int upf_beta_index; int l; int m; };
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

            // Build expanded D_ij
            std::vector<std::vector<double>> dij_expanded(
                nproj_expanded, std::vector<double>(nproj_expanded, 0.0));
            for (int ie_i = 0; ie_i < nproj_expanded; ++ie_i) {
                for (int ie_j = 0; ie_j < nproj_expanded; ++ie_j) {
                    if (expanded_map[ie_i].l == expanded_map[ie_j].l &&
                        expanded_map[ie_i].m == expanded_map[ie_j].m) {
                        dij_expanded[ie_i][ie_j] =
                            pp.dij[expanded_map[ie_i].upf_beta_index]
                                  [expanded_map[ie_j].upf_beta_index];
                    }
                }
            }

            // Precompute beta and d(beta)/d(epsilon_ab)
            // For the stress, we need the derivative of <beta_ie|psi> w.r.t. strain.
            //
            // Under strain epsilon_ab, (k+G) -> (k+G) - epsilon . (k+G)
            // so d/d(eps_ab) [beta_ie(k+G)] has contributions from:
            // 1. d/d|k+G| [radial(|k+G|)] * d|k+G|/d(eps_ab) * angular * phase
            //    where d|k+G|/d(eps_ab) = -(k+G)_a * (k+G)_b / |k+G|
            // 2. d/d(k+G)_c [Y_lm(k+G_hat)] * d(k+G)_c/d(eps_ab)
            //    — this is complicated and often small for low l. We include only
            //    the radial derivative term (the dominant contribution).
            //
            // The full d(beta)/d(eps_ab) = beta_ie * (-delta_ab / 2)  [volume factor]
            //   + d(beta_radial)/d|k+G| * (-(k+G)_a*(k+G)_b/|k+G|) * angular * phase
            //
            // Combining with the -1/2*delta_ab from the sqrt(Omega) prefactor:
            //   d(P_ie)/d(eps_ab) = sum_G conj(d(beta_ie)/d(eps_ab)) * psi(G)

            std::vector<CVec> beta_kg(nproj_expanded);
            // For each projector, store the "radial derivative" contribution
            // dbeta_ab[ie][ig] = (component that when multiplied by -(k+G)_a*(k+G)_b/|k+G|
            //                    gives the d(beta)/d(radial) contribution)
            std::vector<CVec> beta_kg_radial_deriv(nproj_expanded);

            {
                int ie = 0;
                for (int ib = 0; ib < nproj_upf; ++ib) {
                    const auto& beta = pp.betas[ib];
                    const int l = beta.angular_momentum;
                    const complex_t il = il_table[l % 4];

                    // Precompute radial and its derivative at |k+G|
                    std::vector<double> radial_cache(npw);
                    std::vector<double> dradial_cache(npw);
                    for (size_t ig = 0; ig < npw; ++ig) {
                        radial_cache[ig] = beta_of_q(beta, pp.mesh, kpg_mag[ig]);
                        dradial_cache[ig] = dbeta_of_q_dq(beta, pp.mesh, kpg_mag[ig]);
                    }

                    for (int m_val = -l; m_val <= l; ++m_val) {
                        beta_kg[ie].resize(npw);
                        beta_kg_radial_deriv[ie].resize(npw);

                        for (size_t ig = 0; ig < npw; ++ig) {
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
                                angular = real_spherical_harmonic(l, m_val, 0.0, 0.0, 0.0);
                            } else {
                                angular = real_spherical_harmonic(
                                    l, m_val,
                                    kpg_cart[ig][0], kpg_cart[ig][1], kpg_cart[ig][2]);
                            }

                            beta_kg[ie][ig] = prefactor * il * radial_cache[ig] * angular * phase;

                            // Radial derivative contribution:
                            // d(beta)/d(eps_ab) has a term from d(radial)/d|k+G|
                            // For the stress, we multiply by -(k+G)_a*(k+G)_b/|k+G| later
                            beta_kg_radial_deriv[ie][ig] =
                                prefactor * il * dradial_cache[ig] * angular * phase;
                        }
                        ++ie;
                    }
                }
            }

            // Loop over bands
            for (int n = 0; n < num_bands; ++n) {
                const double occ = k_weights[ik] * occupations[ik][n];
                if (std::abs(occ) < 1.0e-15) continue;

                const auto& psi = wavefunctions[ik][n];

                // Compute P_j = <beta_j|psi>
                std::vector<complex_t> proj(nproj_expanded, {0.0, 0.0});
                for (int j = 0; j < nproj_expanded; ++j) {
                    complex_t sum{0.0, 0.0};
                    for (size_t ig = 0; ig < npw; ++ig) {
                        sum += std::conj(beta_kg[j][ig]) * psi[ig];
                    }
                    proj[j] = sum;
                }

                // For each (a,b), compute d(P_i)/d(eps_ab) and accumulate stress
                // d(P_i)/d(eps_ab) = sum_G conj(d(beta_i)/d(eps_ab)) * psi(G)
                //
                // d(beta)/d(eps_ab) = -beta/2 * delta_ab  (from 1/sqrt(Omega))
                //                   + beta_radial_deriv * (-(k+G)_a*(k+G)_b/|k+G|)
                //
                // So: d(P_i)/d(eps_ab) = -P_i/2 * delta_ab
                //                       + sum_G conj(beta_radial_deriv) * (-(k+G)_a*(k+G)_b/|k+G|) * psi(G)

                // Precompute the radial derivative projections
                // dP_rad_i = sum_G conj(beta_radial_deriv_i(G)) * psi(G)
                // weighted by -(k+G)_a*(k+G)_b/|k+G|
                // We need this for each (a,b) pair.

                // More efficiently: compute the unweighted projection and then weight
                // Actually we need the per-G weighting, so we compute for each (a,b):
                // dP_rad_i^{ab} = sum_G conj(beta_rad_deriv_i(G)) * psi(G) * (-(k+G)_a*(k+G)_b/|k+G|)

                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {  // symmetric
                        // Compute dP_i^{ab} for each expanded projector
                        std::vector<complex_t> dproj_ab(nproj_expanded, {0.0, 0.0});
                        for (int ie = 0; ie < nproj_expanded; ++ie) {
                            complex_t sum{0.0, 0.0};
                            for (size_t ig = 0; ig < npw; ++ig) {
                                double weight = 0.0;
                                if (kpg_mag[ig] > 1.0e-12) {
                                    weight = -kpg_cart[ig][a] * kpg_cart[ig][b] / kpg_mag[ig];
                                }
                                sum += std::conj(beta_kg_radial_deriv[ie][ig])
                                       * weight * psi[ig];
                            }
                            // Add the -1/2 delta_ab term from the volume prefactor
                            double delta_ab = (a == b) ? 1.0 : 0.0;
                            dproj_ab[ie] = sum - 0.5 * delta_ab * proj[ie];
                        }

                        // Accumulate stress
                        double s_ab = 0.0;
                        for (int ie_i = 0; ie_i < nproj_expanded; ++ie_i) {
                            for (int ie_j = 0; ie_j < nproj_expanded; ++ie_j) {
                                s_ab += dij_expanded[ie_i][ie_j]
                                        * std::real(std::conj(proj[ie_j]) * dproj_ab[ie_i]);
                            }
                        }

                        stress[a][b] -= occ * 2.0 * s_ab;
                        if (a != b) {
                            stress[b][a] -= occ * 2.0 * s_ab;
                        }
                    }
                }
            }
        }
    }

    // Divide by volume
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            stress[a][b] /= volume;

    return stress;
}

// ===================================================================
// compute_ewald_stress
// ===================================================================
//
// The Ewald stress has real-space and reciprocal-space contributions,
// plus the self-energy term (which gives a diagonal contribution).
//
// Reciprocal space:
//   sigma_ab = (e2*4*pi/(2*V)) * sum_{G!=0} exp(-G^2/(4*eta^2))/G^2
//              * |S(G)|^2 * (2*G_a*G_b/G^2 * (1 + G^2/(4*eta^2)) - delta_ab)
//
// Real space:
//   sigma_ab = (e2/2) * sum'_{i,j,R} Z_i*Z_j * h(r) * r_a*r_b/r^2
//              - (e2/2) * delta_ab * sum'_{i,j,R} Z_i*Z_j * erfc(eta*r)/r
//   where h(r) = -erfc(eta*r)/r^3 - (2*eta/sqrt(pi))*exp(-eta^2*r^2)/r^2
//
// Note: these formulas differ between codes. We follow the convention in
// QE's stres_ewa.f90 / stres_har.f90.
// ===================================================================

Mat3 StressCalculator::compute_ewald_stress(
    const Crystal& crystal,
    const std::map<std::string, PseudoPotential>& pseudopotentials)
{
    static constexpr double e2 = 2.0;  // Rydberg
    static constexpr double convergence_tol = 1.0e-12;

    const size_t natoms = crystal.num_atoms();

    // Extract charges
    std::vector<double> charges(natoms);
    for (size_t i = 0; i < natoms; ++i) {
        const std::string& symbol = crystal.atom(i).symbol;
        auto it = pseudopotentials.find(symbol);
        if (it == pseudopotentials.end()) {
            charges[i] = 0.0;
        } else {
            charges[i] = it->second.z_valence;
        }
    }

    const double volume = crystal.volume();

    // Optimal eta
    const double eta = std::sqrt(constants::pi)
        * std::pow(static_cast<double>(natoms) / (volume * volume), 1.0 / 6.0);

    // Cartesian positions
    std::vector<Vec3> pos(natoms);
    for (size_t i = 0; i < natoms; ++i) {
        pos[i] = crystal.frac_to_cart(crystal.atom(i).position);
    }

    const Mat3 lat = crystal.lattice_bohr();
    const Mat3 bmat = crystal.reciprocal_lattice();

    Mat3 stress = zero_mat3();

    // --- Reciprocal-space contribution ---
    {
        const double gcut2 = -4.0 * eta * eta * std::log(convergence_tol);
        const double gcut = std::sqrt(gcut2);
        const double four_eta2 = 4.0 * eta * eta;

        auto vec_len = [](const Vec3& v) {
            return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        };

        const int m1max = static_cast<int>(std::ceil(gcut / vec_len(bmat[0]))) + 1;
        const int m2max = static_cast<int>(std::ceil(gcut / vec_len(bmat[1]))) + 1;
        const int m3max = static_cast<int>(std::ceil(gcut / vec_len(bmat[2]))) + 1;

        // Prefactor: e2 * 4*pi / (2*V) = e2 * 2*pi / V
        const double pf = 0.5 * e2 * constants::four_pi / volume;

        for (int m1 = -m1max; m1 <= m1max; ++m1) {
            for (int m2 = -m2max; m2 <= m2max; ++m2) {
                for (int m3 = -m3max; m3 <= m3max; ++m3) {
                    if (m1 == 0 && m2 == 0 && m3 == 0) continue;

                    Vec3 G{};
                    for (int d = 0; d < 3; ++d) {
                        G[d] = m1*bmat[0][d] + m2*bmat[1][d] + m3*bmat[2][d];
                    }
                    const double g2 = vec3_dot(G, G);
                    if (g2 > gcut2) continue;

                    const double gaussian = std::exp(-g2 / four_eta2);

                    // Structure factor
                    double sr = 0.0, si = 0.0;
                    for (size_t ia = 0; ia < natoms; ++ia) {
                        const double phase = vec3_dot(G, pos[ia]);
                        sr += charges[ia] * std::cos(phase);
                        si -= charges[ia] * std::sin(phase);
                    }
                    const double sg2 = sr * sr + si * si;

                    // Stress contribution:
                    // pf * gaussian/g2 * |S|^2 * (2*G_a*G_b/g2*(1 + g2/(4*eta^2)) - delta_ab)
                    const double factor = pf * gaussian / g2 * sg2;
                    const double g2_4eta2 = g2 / four_eta2;

                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            double delta_ab = (a == b) ? 1.0 : 0.0;
                            stress[a][b] += factor
                                * (2.0 * G[a] * G[b] / g2 * (1.0 + g2_4eta2)
                                   - delta_ab);
                        }
                    }
                }
            }
        }
    }

    // --- Real-space contribution ---
    {
        const double rcut = std::max(6.0 / eta,
                                     std::sqrt(-std::log(convergence_tol)) / eta);
        auto vec_len = [](const Vec3& v) {
            return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        };

        const int n1max = static_cast<int>(std::ceil(rcut / vec_len(lat[0]))) + 1;
        const int n2max = static_cast<int>(std::ceil(rcut / vec_len(lat[1]))) + 1;
        const int n3max = static_cast<int>(std::ceil(rcut / vec_len(lat[2]))) + 1;

        const double two_eta_sqrtpi = 2.0 * eta / std::sqrt(constants::pi);
        const double eta2 = eta * eta;

        for (int n1 = -n1max; n1 <= n1max; ++n1) {
            for (int n2 = -n2max; n2 <= n2max; ++n2) {
                for (int n3 = -n3max; n3 <= n3max; ++n3) {
                    Vec3 R{};
                    for (int d = 0; d < 3; ++d) {
                        R[d] = n1*lat[0][d] + n2*lat[1][d] + n3*lat[2][d];
                    }
                    const bool R_is_zero = (n1 == 0 && n2 == 0 && n3 == 0);

                    for (size_t i = 0; i < natoms; ++i) {
                        for (size_t j = 0; j < natoms; ++j) {
                            if (R_is_zero && i == j) continue;

                            Vec3 rij{pos[j][0] - pos[i][0] + R[0],
                                     pos[j][1] - pos[i][1] + R[1],
                                     pos[j][2] - pos[i][2] + R[2]};
                            const double dist = std::sqrt(vec3_dot(rij, rij));

                            if (dist < 1.0e-14) continue;
                            if (dist > rcut) continue;

                            const double erfc_val = std::erfc(eta * dist);
                            const double exp_val = std::exp(-eta2 * dist * dist);

                            // h(r) = erfc(eta*r)/r^3 + (2*eta/sqrt(pi))*exp(-eta^2*r^2)/r^2
                            const double hr = erfc_val / (dist * dist * dist)
                                            + two_eta_sqrtpi * exp_val / (dist * dist);

                            const double zz = charges[i] * charges[j];

                            for (int a = 0; a < 3; ++a) {
                                for (int b = 0; b < 3; ++b) {
                                    // sigma_ab += (e2/2) * Z_i*Z_j * h(r) * r_a*r_b
                                    stress[a][b] -= 0.5 * e2 * zz * hr
                                                    * rij[a] * rij[b];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Charged correction stress ---
    // E_charged = -0.5 * e2 * pi / (V * eta^2) * Q^2
    // where Q = sum Z_i.
    //
    // Under strain: d(1/(V*eta^2))/d(eps_ab) = -delta_ab/(V*eta^2)
    // (holding eta fixed).
    //
    // So d(E_charged)/d(eps_ab) = +0.5 * e2 * pi * Q^2 * delta_ab / (V*eta^2)
    //                            = -E_charged * delta_ab
    //
    // sigma_charged_ab = (1/V) * d(E_charged)/d(eps_ab)
    //                  = -E_charged / V * delta_ab
    {
        double zsum = 0.0;
        for (size_t i = 0; i < natoms; ++i) zsum += charges[i];
        const double e_charged = -0.5 * e2 * constants::pi / (volume * eta * eta) * zsum * zsum;

        for (int a = 0; a < 3; ++a) {
            stress[a][a] -= e_charged;  // will be divided by V below
        }
    }

    // Divide by volume (stress = sigma / Omega)
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            stress[a][b] /= volume;

    return stress;
}

// ===================================================================
// compute_total_stress
// ===================================================================

Mat3 StressCalculator::compute_total_stress(
    const Mat3& kinetic,
    const Mat3& hartree,
    const Mat3& xc,
    const Mat3& local_pp,
    const Mat3& nonlocal_pp,
    const Mat3& ewald)
{
    Mat3 total = zero_mat3();
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            total[a][b] = kinetic[a][b] + hartree[a][b] + xc[a][b]
                        + local_pp[a][b] + nonlocal_pp[a][b] + ewald[a][b];
        }
    }
    return total;
}

// ===================================================================
// pressure_gpa
// ===================================================================

double StressCalculator::pressure_gpa(const Mat3& stress)
{
    // P = -trace(sigma)/3
    const double trace = stress[0][0] + stress[1][1] + stress[2][2];
    return -trace / 3.0 * RY_BOHR3_TO_GPA;
}

} // namespace kronos
