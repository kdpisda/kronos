#pragma once

#include "core/types.hpp"
#include "core/crystal.hpp"

#include <vector>

namespace kronos {

/// Result of k-point generation: a list of k-points in fractional reciprocal
/// coordinates and their associated integration weights (summing to 1.0).
struct KPointData {
    std::vector<Vec3> kpoints;   // fractional reciprocal coords
    std::vector<double> weights; // integration weights (sum to 1.0)
};

/// Generates Monkhorst-Pack k-point grids with time-reversal symmetry folding.
///
/// The Monkhorst-Pack scheme produces a uniform grid in the Brillouin zone:
///
///   k_i = (2*n_i - N_i - 1) / (2*N_i) + shift_i / (2*N_i)
///
/// for n_i = 1, ..., N_i along each reciprocal lattice direction.
///
/// Time-reversal symmetry (k and -k are equivalent) is used to reduce the
/// number of k-points: if -k (modulo a reciprocal lattice vector) is already
/// in the list, its weight is doubled and the duplicate is removed.
class KPointGenerator {
public:
    /// Generate a Monkhorst-Pack grid with optional shift, folded using
    /// time-reversal symmetry.
    ///
    /// @param grid    Grid specification: grid dimensions and shift (0 or 1).
    /// @param crystal Crystal structure (used for reciprocal lattice info).
    /// @return        Irreducible k-points and their weights.
    static KPointData generate_monkhorst_pack(
        const KPointGrid& grid,
        const Crystal& crystal);
};

} // namespace kronos
