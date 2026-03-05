#include "basis/kpoints.hpp"

#include <algorithm>
#include <cmath>

namespace kronos {

// =========================================================================
// Helper: wrap a fractional coordinate into [-0.5, 0.5)
// =========================================================================

namespace {

double wrap_to_bz(double x) {
    // Shift into [0, 1), then map to [-0.5, 0.5)
    x = x - std::floor(x);          // now in [0, 1)
    if (x >= 0.5) x -= 1.0;         // now in [-0.5, 0.5)
    return x;
}

/// Check whether two k-points are equivalent modulo a reciprocal lattice
/// vector, using a tolerance on each fractional component.
bool kpoints_equivalent(const Vec3& k1, const Vec3& k2, double tol = 1.0e-8) {
    for (int i = 0; i < 3; ++i) {
        double diff = k1[i] - k2[i];
        diff = diff - std::round(diff);  // reduce to [-0.5, 0.5]
        if (std::abs(diff) > tol) return false;
    }
    return true;
}

} // anonymous namespace

// =========================================================================
// generate_monkhorst_pack
// =========================================================================

KPointData KPointGenerator::generate_monkhorst_pack(
    const KPointGrid& grid,
    [[maybe_unused]] const Crystal& crystal)
{
    const int N0 = grid.grid[0];
    const int N1 = grid.grid[1];
    const int N2 = grid.grid[2];

    // Special case: 1x1x1 grid -> Gamma point only
    if (N0 == 1 && N1 == 1 && N2 == 1) {
        KPointData data;
        data.kpoints.push_back(Vec3{0.0, 0.0, 0.0});
        data.weights.push_back(1.0);
        return data;
    }

    const double s0 = static_cast<double>(grid.shift[0]);
    const double s1 = static_cast<double>(grid.shift[1]);
    const double s2 = static_cast<double>(grid.shift[2]);

    const int total = N0 * N1 * N2;
    const double base_weight = 1.0 / static_cast<double>(total);

    // Generate the full (unreduced) Monkhorst-Pack grid.
    //
    // Base formula (standard Monkhorst-Pack):
    //   k_i = (2*n_i - N_i - 1) / (2*N_i)    for n_i = 1, ..., N_i
    //
    // When shift_i = 1, a half-half-grid-spacing offset is applied:
    //   k_i += shift_i / (4 * N_i)
    //
    // For even N with shift=0: the grid avoids Gamma and admits TR folding.
    // For shift=1: the grid is displaced from the standard MP positions,
    // avoiding both Gamma and the original grid points.
    struct KW {
        Vec3 k;
        double w;
    };
    std::vector<KW> full_grid;
    full_grid.reserve(total);

    for (int n0 = 1; n0 <= N0; ++n0) {
        for (int n1 = 1; n1 <= N1; ++n1) {
            for (int n2 = 1; n2 <= N2; ++n2) {
                double k0 = (2.0 * n0 - N0 - 1.0) / (2.0 * N0) + s0 / (4.0 * N0);
                double k1 = (2.0 * n1 - N1 - 1.0) / (2.0 * N1) + s1 / (4.0 * N1);
                double k2 = (2.0 * n2 - N2 - 1.0) / (2.0 * N2) + s2 / (4.0 * N2);

                // Wrap into [-0.5, 0.5)
                k0 = wrap_to_bz(k0);
                k1 = wrap_to_bz(k1);
                k2 = wrap_to_bz(k2);

                full_grid.push_back(KW{Vec3{k0, k1, k2}, base_weight});
            }
        }
    }

    // Apply time-reversal symmetry: k and -k are equivalent.
    // For each k-point, check if -k (modulo G) is already in the reduced set.
    // If so, double the weight of the existing point. Otherwise, add k.
    std::vector<KW> reduced;
    reduced.reserve(full_grid.size());

    for (const auto& kw : full_grid) {
        // Compute -k (wrapped)
        Vec3 neg_k{
            wrap_to_bz(-kw.k[0]),
            wrap_to_bz(-kw.k[1]),
            wrap_to_bz(-kw.k[2])
        };

        // Check if -k is already in the reduced set
        bool found = false;
        for (auto& rkw : reduced) {
            if (kpoints_equivalent(rkw.k, neg_k)) {
                // -k is already present; add our weight to it
                rkw.w += kw.w;
                found = true;
                break;
            }
        }

        if (!found) {
            // Also check if k itself is already present (self-inverse: k == -k mod G)
            bool self_found = false;
            for (auto& rkw : reduced) {
                if (kpoints_equivalent(rkw.k, kw.k)) {
                    // This shouldn't happen for a proper MP grid (no duplicates),
                    // but guard against it
                    rkw.w += kw.w;
                    self_found = true;
                    break;
                }
            }
            if (!self_found) {
                reduced.push_back(kw);
            }
        }
    }

    // Build output
    KPointData data;
    data.kpoints.reserve(reduced.size());
    data.weights.reserve(reduced.size());

    for (const auto& kw : reduced) {
        data.kpoints.push_back(kw.k);
        data.weights.push_back(kw.w);
    }

    // Normalize weights to sum to 1.0 (should already be exact, but guard
    // against floating-point drift)
    double wsum = 0.0;
    for (double w : data.weights) wsum += w;
    if (wsum > 0.0) {
        for (double& w : data.weights) w /= wsum;
    }

    return data;
}

} // namespace kronos
