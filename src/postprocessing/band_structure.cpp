#include "postprocessing/band_structure.hpp"
#include "core/constants.hpp"
#include "utils/logger.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace kronos {

// ============================================================================
// Helper: convert a fractional k-point to Cartesian (1/bohr) using the
// reciprocal lattice.  k_cart[j] = sum_i k_frac[i] * b[i][j]
// ============================================================================
static Vec3 frac_to_cart_k(const Vec3& k_frac, const Mat3& recip) {
    Vec3 k_cart{};
    for (int j = 0; j < 3; ++j) {
        k_cart[j] = k_frac[0] * recip[0][j]
                   + k_frac[1] * recip[1][j]
                   + k_frac[2] * recip[2][j];
    }
    return k_cart;
}

// ============================================================================
// Helper: Euclidean distance between two Cartesian vectors.
// ============================================================================
static double cart_distance(const Vec3& a, const Vec3& b) {
    double d2 = 0.0;
    for (int j = 0; j < 3; ++j) {
        double diff = a[j] - b[j];
        d2 += diff * diff;
    }
    return std::sqrt(d2);
}

// ============================================================================
// generate_kpath
// ============================================================================
BandData BandStructureCalculator::generate_kpath(
    const Crystal& crystal,
    const KPathSpec& path_spec,
    int npoints_per_segment)
{
    if (path_spec.size() < 2) {
        throw std::invalid_argument(
            "BandStructureCalculator::generate_kpath: path_spec must have at "
            "least 2 high-symmetry points");
    }
    if (npoints_per_segment < 2) {
        throw std::invalid_argument(
            "BandStructureCalculator::generate_kpath: npoints_per_segment must "
            "be >= 2");
    }

    const Mat3 recip = crystal.reciprocal_lattice();

    BandData data;
    double cumulative_distance = 0.0;

    // Record the first tick.
    data.tick_positions.emplace_back(0.0, path_spec[0].label);

    for (size_t seg = 0; seg + 1 < path_spec.size(); ++seg) {
        const Vec3& k_start_frac = path_spec[seg].frac;
        const Vec3& k_end_frac   = path_spec[seg + 1].frac;

        Vec3 k_start_cart = frac_to_cart_k(k_start_frac, recip);
        Vec3 k_end_cart   = frac_to_cart_k(k_end_frac,   recip);
        double seg_length = cart_distance(k_start_cart, k_end_cart);

        // For non-final segments, sample npoints_per_segment points in [0, 1)
        // to avoid duplicating the endpoint (which becomes the next segment's
        // start).  For the final segment, sample npoints_per_segment points
        // in [0, 1] to include the last high-symmetry point.
        bool is_last_segment = (seg + 2 == path_spec.size());

        for (int i = 0; i < npoints_per_segment; ++i) {
            double t;
            if (!is_last_segment) {
                t = static_cast<double>(i) / static_cast<double>(npoints_per_segment);
            } else {
                t = static_cast<double>(i) / static_cast<double>(npoints_per_segment - 1);
            }

            Vec3 k_frac{};
            for (int j = 0; j < 3; ++j) {
                k_frac[j] = k_start_frac[j] + t * (k_end_frac[j] - k_start_frac[j]);
            }

            double d = cumulative_distance + t * seg_length;

            data.kpoints.push_back(k_frac);
            data.distances.push_back(d);
        }

        cumulative_distance += seg_length;

        // Record the tick at the end of this segment.
        data.tick_positions.emplace_back(cumulative_distance,
                                         path_spec[seg + 1].label);
    }

    Logger::instance().info("bands", "K-path generated",
        {{"num_kpoints", std::to_string(data.kpoints.size())},
         {"num_segments", std::to_string(path_spec.size() - 1)}});

    return data;
}

// ============================================================================
// compute_bands
// ============================================================================
void BandStructureCalculator::compute_bands(
    BandData& kpath,
    const std::function<std::function<CVec(const CVec&)>(const Vec3&)>& h_apply_factory,
    const std::function<std::vector<double>(const Vec3&)>& precond_factory,
    int num_bands,
    const std::function<int(const Vec3&)>& num_pw_func)
{
    const size_t nk = kpath.kpoints.size();
    kpath.eigenvalues.resize(nk);

    DavidsonSolver eigensolver;

    for (size_t ik = 0; ik < nk; ++ik) {
        const Vec3& k_frac = kpath.kpoints[ik];

        auto h_apply = h_apply_factory(k_frac);
        auto precond = precond_factory(k_frac);
        int npw = num_pw_func(k_frac);

        EigenResult eigen = eigensolver.solve(h_apply, precond, num_bands, npw);
        kpath.eigenvalues[ik] = std::move(eigen.eigenvalues);

        if ((ik + 1) % 10 == 0 || ik + 1 == nk) {
            Logger::instance().info("bands", "Diagonalizing k-points",
                {{"progress", std::to_string(ik + 1) + "/" + std::to_string(nk)}});
        }
    }
}

// ============================================================================
// write_bands_gnuplot
// ============================================================================
void BandStructureCalculator::write_bands_gnuplot(
    const std::string& filename,
    const BandData& band_data)
{
    if (band_data.kpoints.empty()) {
        throw std::invalid_argument(
            "BandStructureCalculator::write_bands_gnuplot: empty band data");
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error(
            "BandStructureCalculator::write_bands_gnuplot: cannot open file: "
            + filename);
    }

    ofs << std::setprecision(10);

    // Header
    ofs << "# KRONOS band structure data\n";
    ofs << "# Columns: k_distance (1/bohr)";

    const size_t nbands = band_data.eigenvalues.empty()
                          ? 0 : band_data.eigenvalues[0].size();
    for (size_t ib = 0; ib < nbands; ++ib) {
        ofs << "  band" << (ib + 1) << "(eV)";
    }
    ofs << "\n";

    // Tick positions for gnuplot vertical lines
    ofs << "# High-symmetry points:\n";
    for (const auto& [pos, label] : band_data.tick_positions) {
        ofs << "#   " << label << "  " << pos << "\n";
    }
    ofs << "#\n";

    // Data rows
    for (size_t ik = 0; ik < band_data.kpoints.size(); ++ik) {
        ofs << band_data.distances[ik];
        if (ik < band_data.eigenvalues.size()) {
            for (size_t ib = 0; ib < band_data.eigenvalues[ik].size(); ++ib) {
                double e_ev = band_data.eigenvalues[ik][ib] * constants::rydberg_to_ev;
                ofs << "  " << e_ev;
            }
        }
        ofs << "\n";
    }

    ofs.flush();
    if (!ofs.good()) {
        throw std::runtime_error(
            "BandStructureCalculator::write_bands_gnuplot: write error on: "
            + filename);
    }

    Logger::instance().info("bands", "Band data written",
        {{"filename", filename},
         {"num_kpoints", std::to_string(band_data.kpoints.size())},
         {"num_bands", std::to_string(nbands)}});
}

// ============================================================================
// Default high-symmetry paths
// ============================================================================

// FCC: Gamma - X - W - K - Gamma - L - U - W - L - K
KPathSpec BandStructureCalculator::default_path_fcc() {
    return {
        {"G",     {0.0,   0.0,   0.0  }},
        {"X",     {0.5,   0.0,   0.5  }},
        {"W",     {0.5,   0.25,  0.75 }},
        {"K",     {0.375, 0.375, 0.75 }},
        {"G",     {0.0,   0.0,   0.0  }},
        {"L",     {0.5,   0.5,   0.5  }},
        {"U",     {0.625, 0.25,  0.625}},
        {"W",     {0.5,   0.25,  0.75 }},
        {"L",     {0.5,   0.5,   0.5  }},
        {"K",     {0.375, 0.375, 0.75 }},
    };
}

// BCC: Gamma - H - N - Gamma - P - H
KPathSpec BandStructureCalculator::default_path_bcc() {
    return {
        {"G",     {0.0,   0.0,   0.0 }},
        {"H",     {0.5,  -0.5,   0.5 }},
        {"N",     {0.0,   0.0,   0.5 }},
        {"G",     {0.0,   0.0,   0.0 }},
        {"P",     {0.25,  0.25,  0.25}},
        {"H",     {0.5,  -0.5,   0.5 }},
    };
}

// Simple cubic: Gamma - X - M - Gamma - R - X
KPathSpec BandStructureCalculator::default_path_sc() {
    return {
        {"G",     {0.0, 0.0, 0.0}},
        {"X",     {0.0, 0.5, 0.0}},
        {"M",     {0.5, 0.5, 0.0}},
        {"G",     {0.0, 0.0, 0.0}},
        {"R",     {0.5, 0.5, 0.5}},
        {"X",     {0.0, 0.5, 0.0}},
    };
}

// HCP: Gamma - M - K - Gamma - A - L - H - A
KPathSpec BandStructureCalculator::default_path_hcp() {
    return {
        {"G",     {0.0,       0.0,       0.0}},
        {"M",     {0.5,       0.0,       0.0}},
        {"K",     {1.0 / 3.0, 1.0 / 3.0, 0.0}},
        {"G",     {0.0,       0.0,       0.0}},
        {"A",     {0.0,       0.0,       0.5}},
        {"L",     {0.5,       0.0,       0.5}},
        {"H",     {1.0 / 3.0, 1.0 / 3.0, 0.5}},
        {"A",     {0.0,       0.0,       0.5}},
    };
}

} // namespace kronos
