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

} // namespace kronos
