#pragma once

#include "core/types.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"

#include <vector>

namespace kronos {

/// Compute |nabla n(r)|^2 from density in G-space.
///
/// Algorithm:
///   1. For each Cartesian direction d = x, y, z:
///      (partial n / partial d)_G = i * G_d * n(G)
///   2. IFFT each component to real space
///   3. sigma(r) = |nabla n(r)|^2 = sum_d (partial n / partial d (r))^2
///
/// @param density_g   Density in plane-wave (G-space) representation.
/// @param basis       The plane-wave basis set.
/// @param fft_grid    FFT grid for forward/inverse transforms.
/// @return sigma(r) = |nabla n(r)|^2 on the real-space grid.
RVec compute_sigma(const CVec& density_g,
                   const PlaneWaveBasis& basis,
                   FFTGrid& fft_grid);

/// Compute the GGA potential correction in real space.
///
/// V_gga(r) = -2 * div( vsigma(r) * nabla n(r) )
///
/// Computed in G-space:
///   For each direction d:  h_d(r) = vsigma(r) * (partial n / partial d)(r)
///   FFT h_d to G-space
///   div_G = sum_d  i * G_d * h_d(G)
///   IFFT div to get div(h) in real space
///   V_gga(r) = -2 * div(h)(r)
///
/// @param density_g   Density in plane-wave (G-space) representation.
/// @param vsigma      dE/d(sigma) on the real-space grid (Ry).
/// @param basis       The plane-wave basis set.
/// @param fft_grid    FFT grid for forward/inverse transforms.
/// @return V_gga(r) on the real-space grid (Ry).
RVec compute_gga_potential(const CVec& density_g,
                           const RVec& vsigma,
                           const PlaneWaveBasis& basis,
                           FFTGrid& fft_grid);

// ============================================================================
// Spin-polarized GGA gradient routines
// ============================================================================

/// Result of spin-resolved sigma computation.
struct SpinSigmaResult {
    RVec sigma_uu;  ///< |nabla n_up|^2
    RVec sigma_ud;  ///< nabla n_up . nabla n_dn
    RVec sigma_dd;  ///< |nabla n_dn|^2
    /// Per-spin real-space gradient components [spin][dir][grid_point]
    /// Used internally for potential computation.
    std::vector<std::vector<RVec>> grad_r;  // [2][3][num_grid]
};

/// Compute per-spin density gradients and the 3-component sigma array.
///
/// For spin s = up(0), dn(1):
///   (d n_s / d x_d)_G = i * G_d * n_s(G)
///   sigma_uu(r) = sum_d (d n_up / d x_d)^2
///   sigma_ud(r) = sum_d (d n_up / d x_d)(d n_dn / d x_d)
///   sigma_dd(r) = sum_d (d n_dn / d x_d)^2
///
/// @param density_up_g  Spin-up density in G-space.
/// @param density_dn_g  Spin-down density in G-space.
/// @param basis         Plane-wave basis set.
/// @param fft_grid      FFT grid.
/// @return SpinSigmaResult containing sigma_uu, sigma_ud, sigma_dd,
///         and per-spin gradient components for use in potential computation.
SpinSigmaResult compute_spin_sigma(const CVec& density_up_g,
                                    const CVec& density_dn_g,
                                    const PlaneWaveBasis& basis,
                                    FFTGrid& fft_grid);

/// Compute spin-polarized GGA potential correction for each spin channel.
///
/// V_gga_up(r) = -2 * div(vsigma_uu * nabla n_up + vsigma_ud * nabla n_dn)
/// V_gga_dn(r) = -2 * div(vsigma_dd * nabla n_dn + vsigma_ud * nabla n_up)
///
/// @param density_up_g  Spin-up density in G-space.
/// @param density_dn_g  Spin-down density in G-space.
/// @param vsigma_uu     dE/d(sigma_uu) on real-space grid.
/// @param vsigma_ud     dE/d(sigma_ud) on real-space grid.
/// @param vsigma_dd     dE/d(sigma_dd) on real-space grid.
/// @param spin_sigma    Precomputed spin sigma result (contains gradients).
/// @param basis         Plane-wave basis set.
/// @param fft_grid      FFT grid.
/// @param vgga_up       [out] GGA potential for spin-up.
/// @param vgga_dn       [out] GGA potential for spin-down.
void compute_spin_gga_potential(const CVec& density_up_g,
                                const CVec& density_dn_g,
                                const RVec& vsigma_uu,
                                const RVec& vsigma_ud,
                                const RVec& vsigma_dd,
                                const SpinSigmaResult& spin_sigma,
                                const PlaneWaveBasis& basis,
                                FFTGrid& fft_grid,
                                RVec& vgga_up,
                                RVec& vgga_dn);

} // namespace kronos
