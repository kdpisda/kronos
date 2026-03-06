#include "basis/kpoints.hpp"

#include <algorithm>
#include <cmath>
#include <map>

#ifdef KRONOS_HAS_SPGLIB
extern "C" {
#include <spglib.h>
}
#endif

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

/// Time-reversal-only reduction (fallback when spglib is unavailable).
KPointData reduce_time_reversal(const std::vector<Vec3>& full_kpoints) {
    const int total = static_cast<int>(full_kpoints.size());
    const double base_weight = 1.0 / static_cast<double>(total);

    struct KW { Vec3 k; double w; };
    std::vector<KW> reduced;
    reduced.reserve(total);

    for (const auto& kpt : full_kpoints) {
        Vec3 neg_k{wrap_to_bz(-kpt[0]), wrap_to_bz(-kpt[1]), wrap_to_bz(-kpt[2])};

        bool found = false;
        for (auto& rkw : reduced) {
            if (kpoints_equivalent(rkw.k, neg_k)) {
                rkw.w += base_weight;
                found = true;
                break;
            }
        }

        if (!found) {
            bool self_found = false;
            for (auto& rkw : reduced) {
                if (kpoints_equivalent(rkw.k, kpt)) {
                    rkw.w += base_weight;
                    self_found = true;
                    break;
                }
            }
            if (!self_found) {
                reduced.push_back(KW{kpt, base_weight});
            }
        }
    }

    KPointData data;
    data.kpoints.reserve(reduced.size());
    data.weights.reserve(reduced.size());
    for (const auto& kw : reduced) {
        data.kpoints.push_back(kw.k);
        data.weights.push_back(kw.w);
    }

    // Normalize weights
    double wsum = 0.0;
    for (double w : data.weights) wsum += w;
    if (wsum > 0.0) {
        for (double& w : data.weights) w /= wsum;
    }

    return data;
}

} // anonymous namespace

// =========================================================================
// generate_monkhorst_pack
// =========================================================================
//
// When spglib is available, uses spg_get_ir_reciprocal_mesh to reduce the
// k-point grid using the full space-group symmetry of the crystal.
// Falls back to time-reversal-only reduction otherwise.
// =========================================================================

KPointData KPointGenerator::generate_monkhorst_pack(
    const KPointGrid& grid,
    const Crystal& crystal)
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

#ifdef KRONOS_HAS_SPGLIB
    // =====================================================================
    // spglib-based IBZ reduction using full space-group symmetry
    // =====================================================================
    //
    // spg_get_ir_reciprocal_mesh takes the mesh dimensions, shift flags,
    // and crystal structure, and returns:
    //   - grid_address[N_total][3]: integer grid addresses for each point
    //   - ir_mapping_table[N_total]: maps each point to its irreducible rep
    //
    // The k-point in fractional coords is:
    //   k_i = (grid_address[i] + shift[i]/2.0) / mesh[i]
    //
    // This is the QE convention. We wrap to [-0.5, 0.5).
    // =====================================================================

    const int total = N0 * N1 * N2;
    int mesh[3] = {N0, N1, N2};
    int is_shift[3] = {grid.shift[0], grid.shift[1], grid.shift[2]};

    // Prepare crystal structure for spglib
    // spglib expects lattice as row vectors in a [3][3] array (Angstrom or
    // any consistent unit — it only uses ratios for symmetry detection).
    // We use Angstrom (Crystal stores lattice in Angstrom).
    const auto& lat = crystal.lattice();
    double lattice[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            lattice[i][j] = lat[i][j];

    const size_t natoms = crystal.num_atoms();
    std::vector<std::array<double, 3>> positions(natoms);
    std::vector<int> types(natoms);

    // Build type mapping: unique integer per element symbol
    std::map<std::string, int> symbol_to_type;
    int next_type = 0;
    for (size_t ia = 0; ia < natoms; ++ia) {
        const auto& atom = crystal.atom(ia);
        auto it = symbol_to_type.find(atom.symbol);
        if (it == symbol_to_type.end()) {
            symbol_to_type[atom.symbol] = next_type;
            types[ia] = next_type++;
        } else {
            types[ia] = it->second;
        }
        positions[ia] = {atom.position[0], atom.position[1], atom.position[2]};
    }

    // Allocate output arrays
    std::vector<std::array<int, 3>> grid_address(total);
    std::vector<int> ir_mapping_table(total);

    int num_ir = spg_get_ir_reciprocal_mesh(
        reinterpret_cast<int(*)[3]>(grid_address.data()),
        ir_mapping_table.data(),
        mesh,
        is_shift,
        /* is_time_reversal = */ 1,
        lattice,
        reinterpret_cast<double(*)[3]>(positions.data()),
        types.data(),
        static_cast<int>(natoms),
        /* symprec = */ 1e-5);

    if (num_ir <= 0) {
        // spglib failed — fall back to time-reversal-only reduction
        std::vector<Vec3> full_kpoints;
        full_kpoints.reserve(total);
        for (int n0 = 1; n0 <= N0; ++n0)
            for (int n1 = 1; n1 <= N1; ++n1)
                for (int n2 = 1; n2 <= N2; ++n2) {
                    double k0 = (2.0*n0 - N0 - 1.0)/(2.0*N0) + static_cast<double>(grid.shift[0])/(2.0*N0);
                    double k1 = (2.0*n1 - N1 - 1.0)/(2.0*N1) + static_cast<double>(grid.shift[1])/(2.0*N1);
                    double k2 = (2.0*n2 - N2 - 1.0)/(2.0*N2) + static_cast<double>(grid.shift[2])/(2.0*N2);
                    full_kpoints.push_back(Vec3{wrap_to_bz(k0), wrap_to_bz(k1), wrap_to_bz(k2)});
                }
        return reduce_time_reversal(full_kpoints);
    }

    // Count weight of each irreducible k-point
    // ir_mapping_table[i] gives the index of the representative for point i.
    std::map<int, int> ir_weight;  // ir_index -> count
    for (int i = 0; i < total; ++i) {
        ir_weight[ir_mapping_table[i]]++;
    }

    // Build the irreducible k-point list
    KPointData data;
    data.kpoints.reserve(num_ir);
    data.weights.reserve(num_ir);

    for (const auto& [ir_idx, count] : ir_weight) {
        // Convert grid address to fractional k-point
        double k0 = (static_cast<double>(grid_address[ir_idx][0])
                     + static_cast<double>(is_shift[0]) / 2.0)
                    / static_cast<double>(mesh[0]);
        double k1 = (static_cast<double>(grid_address[ir_idx][1])
                     + static_cast<double>(is_shift[1]) / 2.0)
                    / static_cast<double>(mesh[1]);
        double k2 = (static_cast<double>(grid_address[ir_idx][2])
                     + static_cast<double>(is_shift[2]) / 2.0)
                    / static_cast<double>(mesh[2]);

        data.kpoints.push_back(Vec3{wrap_to_bz(k0), wrap_to_bz(k1), wrap_to_bz(k2)});
        data.weights.push_back(static_cast<double>(count) / static_cast<double>(total));
    }

    return data;

#else
    // =====================================================================
    // Fallback: time-reversal-only reduction (no spglib)
    // =====================================================================

    const double s0 = static_cast<double>(grid.shift[0]);
    const double s1 = static_cast<double>(grid.shift[1]);
    const double s2 = static_cast<double>(grid.shift[2]);
    const int total = N0 * N1 * N2;

    std::vector<Vec3> full_kpoints;
    full_kpoints.reserve(total);

    for (int n0 = 1; n0 <= N0; ++n0) {
        for (int n1 = 1; n1 <= N1; ++n1) {
            for (int n2 = 1; n2 <= N2; ++n2) {
                double k0 = (2.0 * n0 - N0 - 1.0) / (2.0 * N0) + s0 / (2.0 * N0);
                double k1 = (2.0 * n1 - N1 - 1.0) / (2.0 * N1) + s1 / (2.0 * N1);
                double k2 = (2.0 * n2 - N2 - 1.0) / (2.0 * N2) + s2 / (2.0 * N2);

                full_kpoints.push_back(Vec3{wrap_to_bz(k0), wrap_to_bz(k1), wrap_to_bz(k2)});
            }
        }
    }

    return reduce_time_reversal(full_kpoints);
#endif
}

} // namespace kronos
