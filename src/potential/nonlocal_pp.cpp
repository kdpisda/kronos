#include "potential/nonlocal_pp.hpp"
#include "core/constants.hpp"
#include "core/spherical_harmonics.hpp"

#include <cassert>
#include <cmath>
#include <complex>

namespace kronos {

// ===================================================================
// Spherical Bessel functions  j_l(x)
// ===================================================================

namespace {

/// j_0(x) = sin(x)/x                       (j_0(0) = 1)
inline double sbessel_0(double x)
{
    if (std::abs(x) < 1.0e-12) return 1.0;
    return std::sin(x) / x;
}

/// j_1(x) = sin(x)/x^2 - cos(x)/x         (j_1(0) = 0)
inline double sbessel_1(double x)
{
    if (std::abs(x) < 1.0e-12) return 0.0;
    return std::sin(x) / (x * x) - std::cos(x) / x;
}

/// j_2(x) = (3/x^2 - 1)*sin(x)/x - 3*cos(x)/x^2   (j_2(0) = 0)
inline double sbessel_2(double x)
{
    if (std::abs(x) < 1.0e-12) return 0.0;
    const double x2 = x * x;
    return ((3.0 / x2 - 1.0) * std::sin(x) - 3.0 * std::cos(x) / x) / x;
}

/// j_3(x) = ((15/x^3 - 6/x)*sin(x) - (15/x^2 - 1)*cos(x)) / x
inline double sbessel_3(double x)
{
    if (std::abs(x) < 1.0e-12) return 0.0;
    const double x2 = x * x;
    const double x3 = x2 * x;
    return ((15.0 / x3 - 6.0 / x) * std::sin(x)
            - (15.0 / x2 - 1.0) * std::cos(x)) / x;
}

/// General j_l(x) using the recurrence relation for l >= 4.
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

    // Upward recurrence: j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)
    double jlm1 = sbessel_0(x);
    double jl   = sbessel_1(x);
    for (int ll = 1; ll < l; ++ll) {
        const double jlp1 = static_cast<double>(2 * ll + 1) / x * jl - jlm1;
        jlm1 = jl;
        jl = jlp1;
    }
    return jl;
}

} // anonymous namespace

// ===================================================================
// beta_of_q  --  spherical Bessel transform of a beta projector
// ===================================================================

double NonlocalPP::beta_of_q(int l, int cutoff_index,
                             const std::vector<double>& beta_values,
                             const RadialGrid& mesh,
                             double q)
{
    const int npts = std::min(cutoff_index,
                              static_cast<int>(beta_values.size()));
    if (npts <= 0) return 0.0;

    double integral = 0.0;

    for (int i = 0; i < npts; ++i) {
        const double ri = mesh.r[i];
        const double qr = q * ri;
        const double jl = sbessel(l, qr);
        integral += ri * ri * beta_values[i] * jl * mesh.rab[i];
    }

    return integral;
}

// ===================================================================
// Constructor  --  store per-atom metadata and expanded D_ij
// ===================================================================
//
// The constructor does NOT precompute projectors in G-space; that is
// done on the fly by compute_beta_kg() for each k-point.  It does
// store the expanded D_ij matrix and the mapping from expanded index
// to (UPF beta, l, m).
// ===================================================================

NonlocalPP::NonlocalPP(
    const Crystal& crystal,
    const PlaneWaveBasis& basis,
    const std::map<std::string, PseudoPotential>& pseudopotentials)
    : basis_(basis),
      volume_(crystal.volume())
{
    for (size_t ia = 0; ia < crystal.num_atoms(); ++ia) {
        const auto& atom = crystal.atom(ia);
        auto pp_it = pseudopotentials.find(atom.symbol);
        if (pp_it == pseudopotentials.end()) continue;

        const auto& pp = pp_it->second;
        if (pp.betas.empty()) continue;

        AtomData ad;
        ad.atom_index = static_cast<int>(ia);
        ad.species = atom.symbol;
        ad.position_cart = crystal.frac_to_cart(atom.position);

        // Store the radial grid for this species (if not already stored)
        if (species_meshes_.find(atom.symbol) == species_meshes_.end()) {
            species_meshes_[atom.symbol] = pp.mesh;
        }

        // Copy per-UPF-projector metadata
        const int nproj_upf = static_cast<int>(pp.betas.size());
        ad.projectors.resize(nproj_upf);
        for (int ib = 0; ib < nproj_upf; ++ib) {
            ad.projectors[ib].angular_momentum = pp.betas[ib].angular_momentum;
            ad.projectors[ib].cutoff_index = pp.betas[ib].cutoff_index;
            ad.projectors[ib].values = pp.betas[ib].values;
        }

        // Count expanded projectors
        int nproj_expanded = 0;
        for (int ib = 0; ib < nproj_upf; ++ib) {
            nproj_expanded += 2 * pp.betas[ib].angular_momentum + 1;
        }
        ad.num_expanded = nproj_expanded;

        // Build expanded index mapping
        ad.expanded_map.resize(nproj_expanded);
        {
            int ie = 0;
            for (int ib = 0; ib < nproj_upf; ++ib) {
                const int l = pp.betas[ib].angular_momentum;
                for (int m = -l; m <= l; ++m) {
                    ad.expanded_map[ie].upf_beta_index = ib;
                    ad.expanded_map[ie].l = l;
                    ad.expanded_map[ie].m = m;
                    ++ie;
                }
            }
            assert(ie == nproj_expanded);
        }

        // Build expanded D_ij (block-diagonal in m)
        ad.dij.assign(nproj_expanded,
                      std::vector<double>(nproj_expanded, 0.0));

        for (int ie_i = 0; ie_i < nproj_expanded; ++ie_i) {
            const auto& map_i = ad.expanded_map[ie_i];
            for (int ie_j = 0; ie_j < nproj_expanded; ++ie_j) {
                const auto& map_j = ad.expanded_map[ie_j];
                if (map_i.l == map_j.l && map_i.m == map_j.m) {
                    ad.dij[ie_i][ie_j] =
                        pp.dij[map_i.upf_beta_index][map_j.upf_beta_index];
                }
            }
        }

        atom_data_.push_back(std::move(ad));
    }
}

// ===================================================================
// compute_beta_kg  --  build expanded projectors for a given k-point
// ===================================================================
//
// For each expanded projector (UPF beta ib, m):
//   beta_{ib,m}(G) = (4*pi/sqrt(Omega)) * i^l
//                     * integral r^2 beta_ib(r) j_l(|k+G|r) dr
//                     * Y_lm(k+G_hat) * exp(-i(k+G).tau)
//
// For the Gamma-only case (k=0), |k+G| = |G| and exp(-i(k+G).tau) =
// exp(-iG.tau).
// ===================================================================

std::vector<CVec> NonlocalPP::compute_beta_kg(
    const AtomData& ad, const Vec3& k_frac) const
{
    const auto& gvecs = basis_.gvectors();
    const size_t npw = gvecs.size();
    const double prefactor = constants::four_pi / std::sqrt(volume_);

    static const complex_t il_table[4] = {
        {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}
    };

    // Convert k from fractional reciprocal coordinates to Cartesian (1/bohr).
    // We extract the reciprocal lattice vectors from the G-vector list by
    // finding G-vectors with known Miller indices (h=1,k=0,l=0), etc.
    Vec3 b1{}, b2{}, b3{};
    for (const auto& gv : gvecs) {
        if (gv.h == 1 && gv.k == 0 && gv.l == 0) b1 = gv.cart;
        if (gv.h == 0 && gv.k == 1 && gv.l == 0) b2 = gv.cart;
        if (gv.h == 0 && gv.k == 0 && gv.l == 1) b3 = gv.cart;
    }

    Vec3 k_cart{};
    for (int j = 0; j < 3; ++j) {
        k_cart[j] = k_frac[0] * b1[j] + k_frac[1] * b2[j] + k_frac[2] * b3[j];
    }

    const auto mesh_it = species_meshes_.find(ad.species);
    assert(mesh_it != species_meshes_.end());
    const RadialGrid& mesh = mesh_it->second;

    const int nproj_upf = static_cast<int>(ad.projectors.size());
    const int nproj_expanded = ad.num_expanded;

    std::vector<CVec> beta_kg(nproj_expanded);

    // Precompute |k+G| for each G-vector and the k+G Cartesian components
    std::vector<double> kpg_mag(npw);
    std::vector<Vec3> kpg_cart(npw);
    for (size_t ig = 0; ig < npw; ++ig) {
        const Vec3& gc = gvecs[ig].cart;
        const double kpg0 = k_cart[0] + gc[0];
        const double kpg1 = k_cart[1] + gc[1];
        const double kpg2 = k_cart[2] + gc[2];
        kpg_cart[ig] = {kpg0, kpg1, kpg2};
        kpg_mag[ig] = std::sqrt(kpg0 * kpg0 + kpg1 * kpg1 + kpg2 * kpg2);
    }

    int ie = 0;
    for (int ib = 0; ib < nproj_upf; ++ib) {
        const auto& proj = ad.projectors[ib];
        const int l = proj.angular_momentum;
        const complex_t il = il_table[l % 4];

        // Precompute radial transforms at |k+G| (same for all m at this ib)
        std::vector<double> radial_cache(npw);
        for (size_t ig = 0; ig < npw; ++ig) {
            radial_cache[ig] = beta_of_q(l, proj.cutoff_index,
                                         proj.values, mesh, kpg_mag[ig]);
        }

        for (int m = -l; m <= l; ++m) {
            beta_kg[ie].resize(npw);

            for (size_t ig = 0; ig < npw; ++ig) {
                const double radial = radial_cache[ig];

                // Phase factor: exp(-i (k+G) . tau)
                const double kpg_dot_tau =
                    kpg_cart[ig][0] * ad.position_cart[0]
                  + kpg_cart[ig][1] * ad.position_cart[1]
                  + kpg_cart[ig][2] * ad.position_cart[2];
                const complex_t phase{std::cos(kpg_dot_tau),
                                      -std::sin(kpg_dot_tau)};

                // Angular: Y_lm evaluated at the (k+G) direction
                double angular;
                if (kpg_mag[ig] < 1.0e-12) {
                    angular = real_spherical_harmonic(l, m, 0.0, 0.0, 0.0);
                } else {
                    angular = real_spherical_harmonic(
                        l, m,
                        kpg_cart[ig][0], kpg_cart[ig][1], kpg_cart[ig][2]);
                }

                beta_kg[ie][ig] = prefactor * il * radial * angular * phase;
            }

            ++ie;
        }
    }
    assert(ie == nproj_expanded);

    return beta_kg;
}

// ===================================================================
// apply  --  V_NL |psi>
// ===================================================================
//
// For each atom a:
//   1. Compute beta_kg for this k-point
//   2. proj_j = <beta_j^a | psi> = sum_G conj(beta_j^a(G)) * psi(G)
//   3. c_i = sum_j D_{ij}^a * proj_j
//   4. vnl_psi(G) += sum_i c_i * beta_i^a(G)
//
// All indices are in the expanded (l,m) basis.
// ===================================================================

CVec NonlocalPP::apply(const CVec& psi_g, const Vec3& k_frac) const
{
    const size_t npw = basis_.num_pw();
    assert(psi_g.size() == npw);

    CVec vnl_psi(npw, complex_t{0.0, 0.0});

    for (const auto& ad : atom_data_) {
        const int nproj = ad.num_expanded;
        const auto beta_kg = compute_beta_kg(ad, k_frac);

        // Step 1: compute projections <beta_j | psi>
        std::vector<complex_t> proj(nproj, complex_t{0.0, 0.0});
        for (int j = 0; j < nproj; ++j) {
            complex_t sum{0.0, 0.0};
            for (size_t ig = 0; ig < npw; ++ig) {
                sum += std::conj(beta_kg[j][ig]) * psi_g[ig];
            }
            proj[j] = sum;
        }

        // Step 2: apply D matrix: c_i = sum_j D_{ij} * proj_j
        std::vector<complex_t> coeff(nproj, complex_t{0.0, 0.0});
        for (int i = 0; i < nproj; ++i) {
            complex_t ci{0.0, 0.0};
            for (int j = 0; j < nproj; ++j) {
                ci += ad.dij[i][j] * proj[j];
            }
            coeff[i] = ci;
        }

        // Step 3: accumulate into result
        for (int i = 0; i < nproj; ++i) {
            for (size_t ig = 0; ig < npw; ++ig) {
                vnl_psi[ig] += coeff[i] * beta_kg[i][ig];
            }
        }
    }

    return vnl_psi;
}

// ===================================================================
// energy  --  E_NL = sum_n f_n <psi_n | V_NL | psi_n>
// ===================================================================

double NonlocalPP::energy(const std::vector<CVec>& wavefunctions,
                          const std::vector<double>& occupations,
                          const Vec3& k_frac) const
{
    assert(wavefunctions.size() == occupations.size());

    double enl = 0.0;

    for (size_t n = 0; n < wavefunctions.size(); ++n) {
        if (std::abs(occupations[n]) < 1.0e-15) continue;

        const auto& psi = wavefunctions[n];
        const CVec vnl_psi = apply(psi, k_frac);

        // <psi | V_NL | psi> = sum_G conj(psi(G)) * (V_NL psi)(G)
        complex_t braket{0.0, 0.0};
        for (size_t ig = 0; ig < psi.size(); ++ig) {
            braket += std::conj(psi[ig]) * vnl_psi[ig];
        }

        enl += occupations[n] * braket.real();
    }

    return enl;
}

// ===================================================================
// num_projectors
// ===================================================================

int NonlocalPP::num_projectors() const
{
    int total = 0;
    for (const auto& ad : atom_data_) {
        total += ad.num_expanded;
    }
    return total;
}

} // namespace kronos
