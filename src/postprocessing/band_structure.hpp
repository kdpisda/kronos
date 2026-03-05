#pragma once

#include "core/types.hpp"
#include "core/crystal.hpp"
#include "solver/davidson.hpp"
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace kronos {

// A labeled high-symmetry k-point in fractional reciprocal coordinates.
struct HighSymmetryPoint {
    std::string label;
    Vec3 frac;   // fractional coordinates in reciprocal space
};

// Specification of a k-path: an ordered sequence of high-symmetry points.
// Consecutive points define segments along which k-points are interpolated.
using KPathSpec = std::vector<HighSymmetryPoint>;

// Band structure data produced by the calculator.
struct BandData {
    std::vector<Vec3> kpoints;                    // fractional coords of each k-point
    std::vector<double> distances;                // cumulative distance along path (1/bohr)
    std::vector<std::vector<double>> eigenvalues; // [nk][nbands], in Rydberg
    std::vector<std::pair<double, std::string>> tick_positions; // (distance, label) for plot ticks
};

// Band structure calculator: generates a k-path and computes eigenvalues
// along that path using a converged Hamiltonian.
class BandStructureCalculator {
public:
    // Default number of interpolated k-points per path segment.
    static constexpr int default_npoints_per_segment = 50;

    // Generate a k-path from a specification of high-symmetry points.
    // The Crystal is needed to convert fractional k-coords to Cartesian
    // for computing inter-point distances.
    // npoints_per_segment: number of equally-spaced k-points in each segment
    //                      (including the start, excluding the end except for
    //                       the final segment).
    static BandData generate_kpath(const Crystal& crystal,
                                   const KPathSpec& path_spec,
                                   int npoints_per_segment = default_npoints_per_segment);

    // Compute band eigenvalues along a pre-generated k-path.
    //
    // kpath:             BandData with kpoints/distances already filled
    // h_apply_factory:   given a k-point (frac), returns a function that
    //                    applies H to a G-space vector (CVec -> CVec)
    // precond_factory:   given a k-point (frac), returns diagonal preconditioner
    // num_bands:         number of eigenvalues to compute at each k
    // num_pw_func:       returns the plane-wave basis size (may depend on k
    //                    in future; for now typically constant)
    //
    // Fills kpath.eigenvalues in-place and returns the same reference.
    static void compute_bands(
        BandData& kpath,
        const std::function<std::function<CVec(const CVec&)>(const Vec3&)>& h_apply_factory,
        const std::function<std::vector<double>(const Vec3&)>& precond_factory,
        int num_bands,
        const std::function<int(const Vec3&)>& num_pw_func);

    // Write band data to a gnuplot-friendly text file.
    // Columns: k_distance  band1_eV  band2_eV  ...
    // A header comment lists the tick positions and labels.
    static void write_bands_gnuplot(const std::string& filename,
                                    const BandData& band_data);

    // -----------------------------------------------------------------------
    // Default high-symmetry paths for common crystal structures.
    // All coordinates are in fractional reciprocal-lattice units.
    // -----------------------------------------------------------------------
    static KPathSpec default_path_fcc();
    static KPathSpec default_path_bcc();
    static KPathSpec default_path_sc();
    static KPathSpec default_path_hcp();
};

} // namespace kronos
