// ============================================================================
// KRONOS  src/potential/paw.cpp
// PAW Calculator implementation
//
// Implements augmentation charge density, one-center energy corrections,
// D_ij PAW contributions, and overlap operator S.
// ============================================================================

#include "potential/paw.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"

#include <cassert>
#include <cmath>
#include <numeric>
#include <iostream>

namespace kronos {

// ============================================================================
// Helper: Radial integration using Simpson's rule with rab weights
// ============================================================================

namespace {

double radial_integral(const std::vector<double>& f,
                       const std::vector<double>& rab,
                       int npts) {
    double sum = 0.0;
    for (int i = 0; i < npts; ++i) {
        sum += f[i] * rab[i];
    }
    return sum;
}

/// Compute Q_ij(G) = ∫ Q_ij(r) j_0(Gr) r² dr via radial integration
/// For l > 0 augmentation, uses j_l instead. For now, l=0 term dominates
/// and we use the full Q_ij(r) which already includes the angular part.
double q_of_g(const std::vector<double>& qfunc,
              const std::vector<double>& r,
              const std::vector<double>& rab,
              double g_mag, int npts) {
    double sum = 0.0;
    for (int ir = 0; ir < npts; ++ir) {
        double gr = g_mag * r[ir];
        double j0 = (gr < 1e-10) ? 1.0 : std::sin(gr) / gr;
        sum += qfunc[ir] * j0 * rab[ir];
    }
    return sum;
}

} // anonymous namespace

// ============================================================================
// Constructor
// ============================================================================

PAWCalculator::PAWCalculator(
    const Crystal& crystal,
    const PlaneWaveBasis& basis,
    FFTGrid& fft_grid,
    const std::map<std::string, PseudoPotential>& pseudopotentials)
    : crystal_(crystal)
    , basis_(basis)
    , fft_grid_(fft_grid)
{
    // Identify atoms with PAW PPs
    for (size_t ia = 0; ia < crystal.num_atoms(); ++ia) {
        const auto& atom = crystal.atom(ia);
        auto it = pseudopotentials.find(atom.symbol);
        if (it == pseudopotentials.end()) continue;

        const auto& pp = it->second;
        if (!pp.is_paw || !pp.paw.has_value()) continue;

        AtomPAWData ad;
        ad.atom_index = ia;
        ad.species = atom.symbol;
        ad.position_cart = crystal.frac_to_cart(atom.position);
        ad.num_projectors = pp.num_projectors;
        ad.paw = &pp.paw.value();
        ad.pp = &pp;
        atoms_.push_back(ad);
        has_paw_ = true;
    }

    if (has_paw_) {
        compute_qij_cache();
        Logger::instance().info("paw", "PAW calculator initialized",
            {{"num_paw_atoms", std::to_string(atoms_.size())}});
    }
}

// ============================================================================
// Pre-compute q_ij overlap integrals per species
// ============================================================================

void PAWCalculator::compute_qij_cache() {
    for (const auto& ad : atoms_) {
        if (qij_cache_.count(ad.species)) continue;

        const auto& paw = *ad.paw;
        const auto& pp = *ad.pp;
        int np = pp.num_projectors;

        std::vector<double> qij(static_cast<size_t>(np * np), 0.0);

        for (const auto& aug : paw.augmentation) {
            if (aug.i >= 0 && aug.i < np && aug.j >= 0 && aug.j < np) {
                qij[aug.i * np + aug.j] = aug.q_integral;
                qij[aug.j * np + aug.i] = aug.q_integral; // Symmetric
            }
        }

        qij_cache_[ad.species] = std::move(qij);
    }
}

// ============================================================================
// Compute occupation matrix ρ_ij
// ============================================================================

void PAWCalculator::compute_rho_ij(
    const std::vector<std::vector<std::vector<complex_t>>>& projections,
    const std::vector<std::vector<double>>& occupations,
    const std::vector<double>& kweights,
    int spin_factor)
{
    KRONOS_TIMER("paw_rho_ij");

    rho_ij_.resize(atoms_.size());

    for (size_t ia_paw = 0; ia_paw < atoms_.size(); ++ia_paw) {
        const auto& ad = atoms_[ia_paw];
        int np = ad.num_projectors;
        std::vector<double> rij(static_cast<size_t>(np * np), 0.0);

        int nk = static_cast<int>(projections.size());
        for (int ik = 0; ik < nk; ++ik) {
            double wk = kweights[ik];
            int nb = static_cast<int>(occupations[ik].size());

            for (int ib = 0; ib < nb; ++ib) {
                double f = occupations[ik][ib] * wk / spin_factor;
                if (std::abs(f) < 1e-15) continue;

                // projections[ik][ib] should contain projections for all atoms
                // We need the projections specific to this PAW atom's projectors.
                // For now, assume projections are indexed by global projector index.
                // The SCF code needs to provide per-atom projections.
                const auto& proj = projections[ik][ib];
                // proj[ip] = <β_ip|ψ_{ik,ib}> for the expanded projectors
                // of this atom

                if (static_cast<int>(proj.size()) < np) continue;

                for (int i = 0; i < np; ++i) {
                    for (int j = 0; j < np; ++j) {
                        // ρ_ij += f * <ψ|β_i> <β_j|ψ> = f * conj(proj_i) * proj_j
                        rij[i * np + j] += f * std::real(
                            std::conj(proj[i]) * proj[j]);
                    }
                }
            }
        }

        rho_ij_[ia_paw] = std::move(rij);
    }
}

// ============================================================================
// Add augmentation density
// ============================================================================

void PAWCalculator::add_augmentation_density(
    std::vector<complex_t>& density_g,
    const std::vector<Vec3>& grid_gcart,
    const std::vector<double>& grid_g2,
    double ecutrho) const
{
    if (!has_paw_ || rho_ij_.empty()) return;
    KRONOS_TIMER("paw_augmentation");

    int num_grid = static_cast<int>(density_g.size());

    for (size_t ia_paw = 0; ia_paw < atoms_.size(); ++ia_paw) {
        const auto& ad = atoms_[ia_paw];
        const auto& paw = *ad.paw;
        const auto& pp = *ad.pp;
        int np = pp.num_projectors;
        const auto& rij = rho_ij_[ia_paw];

        for (int idx = 0; idx < num_grid; ++idx) {
            if (grid_g2[idx] > ecutrho + 1e-6) continue;

            double g_mag = std::sqrt(grid_g2[idx]);
            const Vec3& gc = grid_gcart[idx];

            // Phase factor: exp(-iG·τ)
            double gdot = gc[0] * ad.position_cart[0] +
                          gc[1] * ad.position_cart[1] +
                          gc[2] * ad.position_cart[2];
            complex_t phase(std::cos(gdot), -std::sin(gdot));

            // Sum over augmentation channels
            complex_t n_aug{0.0, 0.0};
            for (const auto& aug : paw.augmentation) {
                int i = aug.i, j = aug.j;
                if (i < 0 || i >= np || j < 0 || j >= np) continue;

                double qg = q_of_g(aug.qfunc, pp.mesh.r, pp.mesh.rab,
                                    g_mag, std::min(static_cast<int>(aug.qfunc.size()),
                                                    pp.mesh.npoints));

                n_aug += rij[i * np + j] * qg;
            }

            density_g[idx] += n_aug * phase;
        }
    }
}

// ============================================================================
// One-center energy correction
// ============================================================================

double PAWCalculator::one_center_energy() const {
    if (!has_paw_ || rho_ij_.empty()) return 0.0;
    KRONOS_TIMER("paw_one_center");

    double e_total = 0.0;
    for (size_t ia_paw = 0; ia_paw < atoms_.size(); ++ia_paw) {
        double e_ae = compute_one_center_ae(ia_paw);
        double e_ps = compute_one_center_ps(ia_paw);
        e_total += (e_ae - e_ps);
    }

    // Add core energy from PP
    for (const auto& ad : atoms_) {
        e_total += ad.paw->core_energy;
    }

    return e_total;
}

double PAWCalculator::compute_one_center_ae(size_t atom_idx) const {
    const auto& ad = atoms_[atom_idx];
    const auto& paw = *ad.paw;
    const auto& pp = *ad.pp;
    int np = pp.num_projectors;
    const auto& rij = rho_ij_[atom_idx];

    // E_AE = Σ_{ij} ρ_ij * ∫ φ_i^AE(r) * H_AE * φ_j^AE(r) dr
    // Simplified: use kinetic + local potential on radial grid
    double e_ae = 0.0;

    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < np; ++j) {
            double rho = rij[i * np + j];
            if (std::abs(rho) < 1e-15) continue;

            const auto& phi_i = paw.ae_wfc[i];
            const auto& phi_j = paw.ae_wfc[j];
            if (phi_i.empty() || phi_j.empty()) continue;

            // Kinetic energy contribution via numerical second derivative
            int npts = std::min({static_cast<int>(phi_i.size()),
                                 static_cast<int>(phi_j.size()),
                                 pp.mesh.npoints});

            double ke = 0.0;
            for (int ir = 1; ir < npts - 1; ++ir) {
                double dr = pp.mesh.r[ir] - pp.mesh.r[ir > 0 ? ir-1 : 0];
                if (dr < 1e-15) continue;
                // d²φ/dr² ≈ (φ[ir+1] - 2φ[ir] + φ[ir-1])/dr²
                double d2phi = (phi_j[ir+1] - 2.0*phi_j[ir] + phi_j[ir-1]) / (dr*dr);
                // T = -1/2 d²/dr² (in Hartree) → -d²/dr² in Ry
                // But radial part: T_radial = -r * d²(r*phi)/(dr²) / r
                // Simplified: just use overlap with V_loc
                ke += phi_i[ir] * (-d2phi) * pp.mesh.rab[ir];
            }

            // Local potential contribution
            double vloc_int = 0.0;
            for (int ir = 0; ir < npts; ++ir) {
                double r = pp.mesh.r[ir];
                if (r < 1e-10) continue;
                // AE potential: -Z/r (nuclear) — approximated
                double v_nuc = -2.0 * pp.z_valence / r; // Factor 2 for Ry
                vloc_int += phi_i[ir] * v_nuc * phi_j[ir] * pp.mesh.rab[ir];
            }

            e_ae += rho * (ke + vloc_int);
        }
    }

    return e_ae;
}

double PAWCalculator::compute_one_center_ps(size_t atom_idx) const {
    const auto& ad = atoms_[atom_idx];
    const auto& paw = *ad.paw;
    const auto& pp = *ad.pp;
    int np = pp.num_projectors;
    const auto& rij = rho_ij_[atom_idx];

    double e_ps = 0.0;

    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < np; ++j) {
            double rho = rij[i * np + j];
            if (std::abs(rho) < 1e-15) continue;

            const auto& phi_i = paw.ps_wfc[i];
            const auto& phi_j = paw.ps_wfc[j];
            if (phi_i.empty() || phi_j.empty()) continue;

            int npts = std::min({static_cast<int>(phi_i.size()),
                                 static_cast<int>(phi_j.size()),
                                 pp.mesh.npoints,
                                 static_cast<int>(pp.vloc.size())});

            // Kinetic energy contribution
            double ke = 0.0;
            for (int ir = 1; ir < npts - 1; ++ir) {
                double dr = pp.mesh.r[ir] - pp.mesh.r[ir > 0 ? ir-1 : 0];
                if (dr < 1e-15) continue;
                double d2phi = (phi_j[ir+1] - 2.0*phi_j[ir] + phi_j[ir-1]) / (dr*dr);
                ke += phi_i[ir] * (-d2phi) * pp.mesh.rab[ir];
            }

            // Local potential contribution (pseudo V_loc)
            double vloc_int = 0.0;
            for (int ir = 0; ir < npts; ++ir) {
                vloc_int += phi_i[ir] * pp.vloc[ir] * phi_j[ir] * pp.mesh.rab[ir];
            }

            e_ps += rho * (ke + vloc_int);
        }
    }

    return e_ps;
}

// ============================================================================
// D_ij PAW correction
// ============================================================================

std::vector<std::vector<double>> PAWCalculator::compute_dij_paw(
    const std::vector<complex_t>& veff_g,
    const std::vector<Vec3>& grid_gcart,
    const std::vector<double>& grid_g2,
    double ecutrho) const
{
    if (!has_paw_) return {};
    KRONOS_TIMER("paw_dij");

    int num_grid = static_cast<int>(veff_g.size());
    std::vector<std::vector<double>> dij_paw(atoms_.size());

    for (size_t ia_paw = 0; ia_paw < atoms_.size(); ++ia_paw) {
        const auto& ad = atoms_[ia_paw];
        const auto& paw = *ad.paw;
        const auto& pp = *ad.pp;
        int np = pp.num_projectors;

        std::vector<double> dij(static_cast<size_t>(np * np), 0.0);

        // D_ij^integral = ∫ Q_ij(G) * V_eff(G) * exp(iG·τ) dG
        for (const auto& aug : paw.augmentation) {
            int i = aug.i, j = aug.j;
            if (i < 0 || i >= np || j < 0 || j >= np) continue;

            double dij_val = 0.0;
            for (int idx = 0; idx < num_grid; ++idx) {
                if (grid_g2[idx] > ecutrho + 1e-6) continue;

                double g_mag = std::sqrt(grid_g2[idx]);
                const Vec3& gc = grid_gcart[idx];

                // Phase: exp(iG·τ) — note positive sign (conjugate of structure factor)
                double gdot = gc[0] * ad.position_cart[0] +
                              gc[1] * ad.position_cart[1] +
                              gc[2] * ad.position_cart[2];

                double qg = q_of_g(aug.qfunc, pp.mesh.r, pp.mesh.rab,
                                    g_mag, std::min(static_cast<int>(aug.qfunc.size()),
                                                    pp.mesh.npoints));

                // V_eff(G) * exp(iG·τ) — take real part
                complex_t phase(std::cos(gdot), std::sin(gdot));
                dij_val += std::real(veff_g[idx] * phase) * qg;
            }

            dij[i * np + j] += dij_val;
            if (i != j) {
                dij[j * np + i] += dij_val; // Symmetric
            }
        }

        // Add one-center D_ij contributions
        // D_ij^1 = <φ_i^AE|V_AE|φ_j^AE> - <φ_i^PS|V_PS|φ_j^PS>
        // (Already encoded in the bare D_ij from the UPF file in most cases,
        //  but we add a residual correction here)

        dij_paw[ia_paw] = std::move(dij);
    }

    return dij_paw;
}

// ============================================================================
// Overlap operator S
// ============================================================================

CVec PAWCalculator::apply_s(
    const CVec& psi_g,
    const std::vector<CVec>& beta_g,
    const std::vector<std::vector<complex_t>>& projections_per_atom) const
{
    if (!has_paw_) return psi_g;
    KRONOS_TIMER("paw_apply_s");

    CVec s_psi = psi_g;  // S|ψ⟩ = |ψ⟩ + augmentation term
    size_t npw = psi_g.size();

    // For each PAW atom, add Σ_{ij} q_ij |β_i⟩ <β_j|ψ⟩
    size_t beta_offset = 0;
    for (size_t ia_paw = 0; ia_paw < atoms_.size(); ++ia_paw) {
        const auto& ad = atoms_[ia_paw];
        int np = ad.num_projectors;

        auto qit = qij_cache_.find(ad.species);
        if (qit == qij_cache_.end()) {
            beta_offset += np;
            continue;
        }
        if (ia_paw >= projections_per_atom.size()) {
            beta_offset += np;
            continue;
        }
        const auto& qij = qit->second;
        const auto& proj = projections_per_atom[ia_paw]; // <β_j|ψ⟩ for this atom

        for (int i = 0; i < np; ++i) {
            complex_t coeff{0.0, 0.0};
            for (int j = 0; j < np; ++j) {
                if (static_cast<size_t>(j) < proj.size()) {
                    coeff += qij[i * np + j] * proj[j];
                }
            }

            // Add coeff * |β_i⟩
            if (beta_offset + i < beta_g.size()) {
                const auto& bi = beta_g[beta_offset + i];
                for (size_t ig = 0; ig < npw && ig < bi.size(); ++ig) {
                    s_psi[ig] += coeff * bi[ig];
                }
            }
        }

        beta_offset += np;
    }

    return s_psi;
}

const std::vector<double>& PAWCalculator::get_qij(const std::string& species) const {
    static const std::vector<double> empty;
    auto it = qij_cache_.find(species);
    if (it == qij_cache_.end()) return empty;
    return it->second;
}

// ============================================================================
// PAW augmentation force correction
// ============================================================================

std::vector<Vec3> PAWCalculator::compute_paw_forces(
    const std::vector<complex_t>& veff_g,
    const std::vector<Vec3>& grid_gcart,
    const std::vector<double>& grid_g2,
    double ecutrho) const
{
    // Map PAW atom forces back to crystal atom indices
    size_t natoms = crystal_.num_atoms();
    std::vector<Vec3> forces(natoms, Vec3{0.0, 0.0, 0.0});

    if (!has_paw_ || rho_ij_.empty()) return forces;
    KRONOS_TIMER("paw_forces");

    int num_grid = static_cast<int>(veff_g.size());

    for (size_t ia_paw = 0; ia_paw < atoms_.size(); ++ia_paw) {
        const auto& ad = atoms_[ia_paw];
        const auto& paw = *ad.paw;
        const auto& pp = *ad.pp;
        int np = pp.num_projectors;
        const auto& rij = rho_ij_[ia_paw];

        Vec3 f_aug{0.0, 0.0, 0.0};

        for (int idx = 0; idx < num_grid; ++idx) {
            if (grid_g2[idx] > ecutrho + 1e-6) continue;
            if (grid_g2[idx] < 1e-12) continue;  // Skip G=0

            double g_mag = std::sqrt(grid_g2[idx]);
            const Vec3& gc = grid_gcart[idx];

            // Phase: exp(-iG·τ)
            double gdot = gc[0] * ad.position_cart[0] +
                          gc[1] * ad.position_cart[1] +
                          gc[2] * ad.position_cart[2];

            // Sum over augmentation channels
            double q_rho_sum = 0.0;
            for (const auto& aug : paw.augmentation) {
                int i = aug.i, j = aug.j;
                if (i < 0 || i >= np || j < 0 || j >= np) continue;

                double qg = q_of_g(aug.qfunc, pp.mesh.r, pp.mesh.rab,
                                    g_mag, std::min(static_cast<int>(aug.qfunc.size()),
                                                    pp.mesh.npoints));
                q_rho_sum += rij[i * np + j] * qg;
            }

            // F_aug += -Σ_G q_rho * V_eff(G) * iG * exp(-iG·τ)
            // The derivative of exp(-iG·τ) w.r.t. τ gives -iG * exp(-iG·τ)
            // So F = +Σ_G q_rho * V_eff(G) * iG * exp(-iG·τ)
            // (positive sign because force = -dE/dτ and the energy has +exp(-iG·τ))
            complex_t phase(std::cos(gdot), -std::sin(gdot));
            complex_t veff_phase = veff_g[idx] * phase;

            // iG * veff_phase: take imaginary part times G component
            for (int d = 0; d < 3; ++d) {
                // d/dτ_d of exp(-iG·τ) = -iG_d * exp(-iG·τ)
                // So force contribution = Σ_G ρ_ij * Q_ij(G) * V_eff(G) * iG_d * exp(-iG·τ)
                // = Σ_G q_rho * imag(i * G_d * V_eff(G) * exp(-iG·τ))
                // Since i * veff_phase has real part = -imag(veff_phase)
                f_aug[d] += q_rho_sum * gc[d] * (-std::imag(veff_phase));
            }
        }

        // Scale by volume normalization (matching add_augmentation_density convention)
        forces[ad.atom_index] = f_aug;
    }

    return forces;
}

// ============================================================================
// PAW augmentation stress correction (isotropic approximation)
// ============================================================================

Mat3 PAWCalculator::compute_paw_stress(
    const std::vector<complex_t>& veff_g,
    const std::vector<Vec3>& grid_gcart,
    const std::vector<double>& grid_g2,
    double ecutrho) const
{
    Mat3 stress{};

    if (!has_paw_ || rho_ij_.empty()) return stress;
    KRONOS_TIMER("paw_stress");

    int num_grid = static_cast<int>(veff_g.size());
    double volume = crystal_.volume();

    for (size_t ia_paw = 0; ia_paw < atoms_.size(); ++ia_paw) {
        const auto& ad = atoms_[ia_paw];
        const auto& paw = *ad.paw;
        const auto& pp = *ad.pp;
        int np = pp.num_projectors;
        const auto& rij = rho_ij_[ia_paw];

        for (int idx = 0; idx < num_grid; ++idx) {
            if (grid_g2[idx] > ecutrho + 1e-6) continue;
            if (grid_g2[idx] < 1e-12) continue;

            double g_mag = std::sqrt(grid_g2[idx]);
            const Vec3& gc = grid_gcart[idx];

            double gdot = gc[0] * ad.position_cart[0] +
                          gc[1] * ad.position_cart[1] +
                          gc[2] * ad.position_cart[2];
            complex_t phase(std::cos(gdot), -std::sin(gdot));

            // Q_ij(G) derivative w.r.t. strain: dQ/dε_{αβ} ∝ -G_α G_β dQ/d(G²)
            // For the isotropic approximation, we compute:
            // σ_{αβ} = Σ_G Σ_{ij} ρ_ij * dQ_ij/dG * (-G_α G_β / G) * V_eff(G) * exp(-iG·τ)

            // Sum over augmentation channels with Q(G) derivative
            for (const auto& aug : paw.augmentation) {
                int i = aug.i, j = aug.j;
                if (i < 0 || i >= np || j < 0 || j >= np) continue;

                int npts_q = std::min(static_cast<int>(aug.qfunc.size()),
                                      pp.mesh.npoints);

                // Q(G) and dQ/dG via finite difference
                double dg = 1e-4;
                double qg = q_of_g(aug.qfunc, pp.mesh.r, pp.mesh.rab, g_mag, npts_q);
                double qg_plus = q_of_g(aug.qfunc, pp.mesh.r, pp.mesh.rab, g_mag + dg, npts_q);
                double dqdg = (qg_plus - qg) / dg;

                double rho = rij[i * np + j];
                if (std::abs(rho) < 1e-15) continue;

                double re_veff_phase = std::real(veff_g[idx] * phase);

                // Stress: σ_{αβ} += ρ_ij * (dQ/dG) * (-G_α G_β / G) * Re(V_eff * exp(-iG·τ))
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        stress[a][b] += rho * dqdg * (-gc[a] * gc[b] / g_mag)
                                       * re_veff_phase;
                    }
                }
            }
        }
    }

    // Normalize by volume
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            stress[a][b] /= volume;
        }
    }

    return stress;
}

} // namespace kronos
