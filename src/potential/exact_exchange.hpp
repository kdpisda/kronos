#pragma once
// ============================================================================
// KRONOS  src/potential/exact_exchange.hpp
// Exact exchange operator for hybrid functionals (PBE0, HSE06)
//
// Implements:
//   - Direct exact exchange: V_x|ψ⟩ (reference, O(N²))
//   - ACE (Adaptively Compressed Exchange) acceleration: O(N) per apply
//   - PBE0 Coulomb kernel: v_c(G) = 4π/G² with G=0 correction
//   - HSE06 short-range Coulomb: v_SR(G) = (4π/G²)[1 - exp(-G²/4ω²)]
// ============================================================================

#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "potential/xc.hpp"

#include <map>
#include <string>
#include <vector>

namespace kronos {

class ExactExchange {
public:
    ExactExchange(const Crystal& crystal,
                  const PlaneWaveBasis& basis,
                  FFTGrid& fft_grid,
                  HybridType hybrid_type,
                  double exx_fraction = 0.25,
                  double screening_parameter = 0.11);

    /// Get the hybrid type
    HybridType type() const { return hybrid_type_; }

    /// Get the exact exchange fraction (α)
    double exx_fraction() const { return exx_fraction_; }

    /// Get the screening parameter (ω, bohr⁻¹) for HSE06
    double screening_parameter() const { return omega_; }

    // -----------------------------------------------------------------------
    // Direct computation (reference)
    // -----------------------------------------------------------------------

    /// Compute V_x|ψ_{nk}⟩ directly:
    ///   V_x|ψ_{nk}⟩ = -α Σ_{mk'} f_{mk'} IFFT[v_c(G) · FFT[ψ*_{mk'}·ψ_{nk}]] · ψ_{mk'}
    ///
    /// occupied_states[ik][ib] = wavefunction coefficients (G-space)
    /// occupations[ik][ib] = occupation number
    /// kpoints: all k-points
    /// kweights: k-point weights
    CVec apply_direct(const CVec& psi_nk,
                      const Vec3& k_frac,
                      const std::vector<std::vector<CVec>>& occupied_states,
                      const std::vector<std::vector<double>>& occupations,
                      const std::vector<Vec3>& kpoints,
                      const std::vector<double>& kweights) const;

    // -----------------------------------------------------------------------
    // ACE (Adaptively Compressed Exchange)
    // -----------------------------------------------------------------------

    /// Update ACE vectors from current wavefunctions.
    /// Call every N SCF steps (or on first step).
    /// After update, apply_ace() is O(N_occ * N_pw) per band.
    void update_ace(const std::vector<std::vector<CVec>>& occupied_states,
                    const std::vector<std::vector<double>>& occupations,
                    const std::vector<Vec3>& kpoints,
                    const std::vector<double>& kweights);

    /// Apply exchange using ACE compression at k-point index ik:
    ///   V_x|ψ⟩ ≈ -Σ_i |ξ_i^{ik}⟩⟨ξ_i^{ik}|ψ⟩
    CVec apply_ace(const CVec& psi_g, int ik) const;

    /// Check if ACE vectors have been computed
    bool ace_ready() const { return ace_ready_; }

    // -----------------------------------------------------------------------
    // Coulomb kernel
    // -----------------------------------------------------------------------

    /// Get the Coulomb kernel value v_c(G) for a given |G|²
    double coulomb_kernel(double g2) const;

    /// Get the total exchange energy from current wavefunctions
    double exchange_energy(
        const std::vector<std::vector<CVec>>& occupied_states,
        const std::vector<std::vector<double>>& occupations,
        const std::vector<Vec3>& kpoints,
        const std::vector<double>& kweights) const;

private:
    const Crystal& crystal_;
    const PlaneWaveBasis& basis_;
    FFTGrid& fft_grid_;
    HybridType hybrid_type_;
    double exx_fraction_;     // α (0.25 for PBE0 and HSE06)
    double omega_;            // ω screening parameter (HSE06: 0.11 bohr⁻¹)

    // ACE vectors: ace_xi_[ik][i_occ] = compressed exchange vector ξ_i
    std::vector<std::vector<CVec>> ace_xi_;
    bool ace_ready_{false};

    // Precomputed Coulomb kernel on FFT grid
    std::vector<double> coulomb_g_;
    void precompute_coulomb_kernel();

    // Internal helper for pair exchange
    CVec compute_pair_exchange(const CVec& psi_nk, const Vec3& k_frac,
                               const CVec& psi_mk, const Vec3& kp_frac) const;
};

} // namespace kronos
