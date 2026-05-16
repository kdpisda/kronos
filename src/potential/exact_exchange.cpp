// ============================================================================
// KRONOS  src/potential/exact_exchange.cpp
// Exact exchange operator implementation
//
// PBE0: v_c(G) = 4π/G² (with Gygi-Baldereschi G=0 correction)
// HSE06: v_SR(G) = (4π/G²)[1 - exp(-G²/4ω²)], ω = 0.11 bohr⁻¹
//
// The direct computation is O(N_occ² · N_k² · N_pw · log N_pw).
// ACE acceleration reduces to O(N_occ · N_pw) per application.
// ============================================================================

#include "potential/exact_exchange.hpp"
#include "core/constants.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"

#include <cassert>
#include <cmath>
#include <numeric>

namespace kronos {

// ============================================================================
// Constructor
// ============================================================================

ExactExchange::ExactExchange(
    const Crystal& crystal,
    const PlaneWaveBasis& basis,
    FFTGrid& fft_grid,
    HybridType hybrid_type,
    double exx_fraction,
    double screening_parameter)
    : crystal_(crystal)
    , basis_(basis)
    , fft_grid_(fft_grid)
    , hybrid_type_(hybrid_type)
    , exx_fraction_(exx_fraction)
    , omega_(screening_parameter)
{
    precompute_coulomb_kernel();

    Logger::instance().info("exact_exchange", "Initialized",
        {{"type", hybrid_type == HybridType::PBE0 ? "PBE0" : "HSE06"},
         {"alpha", std::to_string(exx_fraction)},
         {"omega", std::to_string(omega_)}});
}

// ============================================================================
// Coulomb kernel
// ============================================================================

double ExactExchange::coulomb_kernel(double g2) const {
    if (g2 < 1e-12) {
        if (hybrid_type_ == HybridType::HSE06) {
            // HSE06: v_SR(G=0) = π/ω² (finite, no divergence)
            return constants::pi / (omega_ * omega_);
        }
        // PBE0: G=0 handled via Gygi-Baldereschi correction
        // Return 0 for the divergent term (corrected separately)
        return 0.0;
    }

    double v_c = 4.0 * constants::pi / g2;

    if (hybrid_type_ == HybridType::HSE06) {
        // Short-range only: v_SR(G) = v_c(G) * [1 - exp(-G²/(4ω²))]
        double exp_factor = std::exp(-g2 / (4.0 * omega_ * omega_));
        return v_c * (1.0 - exp_factor);
    }

    // PBE0: full Coulomb
    return v_c;
}

void ExactExchange::precompute_coulomb_kernel() {
    KRONOS_TIMER("exx_precompute_kernel");

    auto dims = fft_grid_.dims();
    int num_grid = fft_grid_.total_points();
    const Mat3& recip = crystal_.reciprocal_lattice();

    coulomb_g_.resize(num_grid);

    int n0 = dims[0], n1 = dims[1], n2 = dims[2];
    for (int idx = 0; idx < num_grid; ++idx) {
        int hi = idx / (n1 * n2);
        int ki = (idx % (n1 * n2)) / n2;
        int li = idx % n2;
        int h = (hi <= n0/2) ? hi : hi - n0;
        int k = (ki <= n1/2) ? ki : ki - n1;
        int l = (li <= n2/2) ? li : li - n2;

        double gx = h*recip[0][0] + k*recip[1][0] + l*recip[2][0];
        double gy = h*recip[0][1] + k*recip[1][1] + l*recip[2][1];
        double gz = h*recip[0][2] + k*recip[1][2] + l*recip[2][2];
        double g2 = gx*gx + gy*gy + gz*gz;

        coulomb_g_[idx] = coulomb_kernel(g2);
    }
}

// ============================================================================
// Direct exchange computation
// ============================================================================

CVec ExactExchange::apply_direct(
    const CVec& psi_nk,
    const Vec3& k_frac,
    const std::vector<std::vector<CVec>>& occupied_states,
    const std::vector<std::vector<double>>& occupations,
    const std::vector<Vec3>& kpoints,
    const std::vector<double>& kweights) const
{
    KRONOS_TIMER("exx_apply_direct");

    size_t npw = psi_nk.size();
    CVec vx_psi(npw, complex_t{0.0, 0.0});

    int nk = static_cast<int>(kpoints.size());

    for (int ikp = 0; ikp < nk; ++ikp) {
        double wk = kweights[ikp];
        int nb = static_cast<int>(occupations[ikp].size());

        for (int ib = 0; ib < nb; ++ib) {
            double f = occupations[ikp][ib] * wk;
            if (std::abs(f) < 1e-15) continue;

            const CVec& psi_mk = occupied_states[ikp][ib];
            CVec pair = compute_pair_exchange(psi_nk, k_frac, psi_mk, kpoints[ikp]);

            // V_x|ψ⟩ -= α · f · pair
            double scale = exx_fraction_ * f;
            for (size_t ig = 0; ig < npw; ++ig) {
                vx_psi[ig] -= scale * pair[ig];
            }
        }
    }

    return vx_psi;
}

CVec ExactExchange::compute_pair_exchange(
    const CVec& psi_nk, const Vec3& /*k_frac*/,
    const CVec& psi_mk, const Vec3& /*kp_frac*/) const
{
    size_t npw = psi_nk.size();
    int total_points = fft_grid_.total_points();

    // 1. Transform both wavefunctions to real space
    std::vector<complex_t> grid_n(total_points, {0.0, 0.0});
    std::vector<complex_t> grid_m(total_points, {0.0, 0.0});

    fft_grid_.scatter_to_grid(basis_, psi_nk, grid_n);
    fft_grid_.scatter_to_grid(basis_, psi_mk, grid_m);

    std::vector<complex_t> psi_n_r(total_points);
    std::vector<complex_t> psi_m_r(total_points);

    fft_grid_.inverse(grid_n, psi_n_r);
    fft_grid_.inverse(grid_m, psi_m_r);

    // 2. Compute pair product: ρ_{nm}(r) = ψ*_m(r) · ψ_n(r)
    std::vector<complex_t> rho_r(total_points);
    for (int i = 0; i < total_points; ++i) {
        rho_r[i] = std::conj(psi_m_r[i]) * psi_n_r[i];
    }

    // 3. FFT to G-space
    std::vector<complex_t> rho_g(total_points);
    fft_grid_.forward(rho_r, rho_g);

    // 4. Apply Coulomb kernel: v_c(G) · ρ(G)
    for (int i = 0; i < total_points; ++i) {
        rho_g[i] *= coulomb_g_[i] / crystal_.volume();
    }

    // 5. IFFT back to real space
    std::vector<complex_t> v_pair_r(total_points);
    fft_grid_.inverse(rho_g, v_pair_r);

    // 6. Multiply by ψ_m(r): result(r) = v_pair(r) · ψ_m(r)
    for (int i = 0; i < total_points; ++i) {
        v_pair_r[i] *= psi_m_r[i];
    }

    // 7. FFT to G-space and gather
    std::vector<complex_t> result_g(total_points);
    fft_grid_.forward(v_pair_r, result_g);

    CVec result(npw);
    fft_grid_.gather_from_grid(basis_, result_g, result);

    return result;
}

// ============================================================================
// ACE (Adaptively Compressed Exchange)
// ============================================================================

void ExactExchange::update_ace(
    const std::vector<std::vector<CVec>>& occupied_states,
    const std::vector<std::vector<double>>& occupations,
    const std::vector<Vec3>& kpoints,
    const std::vector<double>& kweights)
{
    KRONOS_TIMER("exx_update_ace");

    int nk = static_cast<int>(kpoints.size());
    ace_xi_.resize(nk);

    for (int ik = 0; ik < nk; ++ik) {
        int nb = static_cast<int>(occupations[ik].size());
        ace_xi_[ik].clear();

        for (int ib = 0; ib < nb; ++ib) {
            double f = occupations[ik][ib];
            if (f < 1e-10) continue;

            const CVec& psi = occupied_states[ik][ib];

            // Compute V_x|ψ_ib⟩ via direct method
            CVec vx_psi = apply_direct(psi, kpoints[ik],
                                        occupied_states, occupations,
                                        kpoints, kweights);

            // ξ_i = sqrt(f) * V_x|ψ_i⟩ / <ψ_i|V_x|ψ_i⟩^{1/2}
            // Simplified: store the exchange-applied vector directly
            double norm = 0.0;
            for (const auto& c : vx_psi) {
                norm += std::norm(c);
            }
            norm = std::sqrt(norm);

            if (norm > 1e-15) {
                double scale = std::sqrt(f) / norm;
                for (auto& c : vx_psi) {
                    c *= scale;
                }
                ace_xi_[ik].push_back(std::move(vx_psi));
            }
        }
    }

    ace_ready_ = true;
    Logger::instance().info("ace", "ACE vectors updated",
        {{"num_kpoints", std::to_string(nk)}});
}

CVec ExactExchange::apply_ace(const CVec& psi_g, int ik) const {
    KRONOS_TIMER("exx_apply_ace");

    if (!ace_ready_) {
        return CVec(psi_g.size(), complex_t{0.0, 0.0});
    }

    size_t npw = psi_g.size();
    CVec vx_psi(npw, complex_t{0.0, 0.0});

    // V_x|ψ⟩ ≈ -Σ_i |ξ_i^{ik}⟩⟨ξ_i^{ik}|ψ⟩
    // Only use ACE vectors for this k-point (not all k-points)
    if (ik < 0 || ik >= static_cast<int>(ace_xi_.size())) {
        return vx_psi;
    }

    for (const auto& xi : ace_xi_[ik]) {
        complex_t overlap{0.0, 0.0};
        size_t n = std::min(npw, xi.size());
        for (size_t ig = 0; ig < n; ++ig) {
            overlap += std::conj(xi[ig]) * psi_g[ig];
        }
        for (size_t ig = 0; ig < n; ++ig) {
            vx_psi[ig] -= xi[ig] * overlap;
        }
    }

    return vx_psi;
}

// ============================================================================
// Exchange energy
// ============================================================================

double ExactExchange::exchange_energy(
    const std::vector<std::vector<CVec>>& occupied_states,
    const std::vector<std::vector<double>>& occupations,
    const std::vector<Vec3>& kpoints,
    const std::vector<double>& kweights) const
{
    KRONOS_TIMER("exx_energy");

    double e_exx = 0.0;
    int nk = static_cast<int>(kpoints.size());

    for (int ik = 0; ik < nk; ++ik) {
        double wk = kweights[ik];
        int nb = static_cast<int>(occupations[ik].size());

        for (int ib = 0; ib < nb; ++ib) {
            double f = occupations[ik][ib] * wk;
            if (std::abs(f) < 1e-15) continue;

            const CVec& psi = occupied_states[ik][ib];

            // E_x += f * <ψ|V_x|ψ>
            CVec vx_psi = apply_direct(psi, kpoints[ik],
                                        occupied_states, occupations,
                                        kpoints, kweights);

            complex_t braket{0.0, 0.0};
            for (size_t ig = 0; ig < psi.size(); ++ig) {
                braket += std::conj(psi[ig]) * vx_psi[ig];
            }

            e_exx += f * std::real(braket);
        }
    }

    // Factor 0.5 for double-counting
    return 0.5 * e_exx;
}

} // namespace kronos
