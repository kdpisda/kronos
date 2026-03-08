#pragma once
#include "core/types.hpp"
#include "core/crystal.hpp"
#include <vector>
#include <cstddef>

namespace kronos {

// A single G-vector in the plane-wave basis
struct GVector {
    int h{0}, k{0}, l{0};          // Miller indices
    Vec3 cart{};                     // Cartesian coordinates in 1/bohr
    double norm2{0.0};              // |G|^2
};

// Plane-wave basis set for a given crystal and energy cutoff
class PlaneWaveBasis {
public:
    /// Construct basis: enumerate all G such that |G|^2 <= ecutwfc (Ry).
    /// When k_max > 0, the basis is expanded to include all G where
    /// |k+G|^2 <= ecutwfc for any k with |k| <= k_max. This ensures
    /// the shared basis is complete for all k-points in the grid.
    PlaneWaveBasis(const Crystal& crystal, double ecutwfc, double k_max = 0.0);

    // Number of plane waves (G-vectors)
    size_t num_pw() const;

    // Access G-vectors
    const std::vector<GVector>& gvectors() const;
    const GVector& gvec(size_t i) const;

    // Kinetic energy |k+G|^2 for a given k-point (Ry, Rydberg units)
    // k is in fractional reciprocal coords
    std::vector<double> kinetic_energies(const Vec3& k_frac) const;

    // Energy cutoff used (the physics cutoff, not the expanded sphere)
    double ecutwfc() const;

    // Maximum Miller index in each direction
    std::array<int, 3> max_miller() const;

    // Reciprocal lattice vectors (for FFT grid computation)
    const Mat3& reciprocal_lattice() const;

private:
    double ecutwfc_;
    double gvec_cutoff_;   // actual G-vector sphere cutoff (>= ecutwfc when k_max > 0)
    Mat3 recip_lattice_;
    std::vector<GVector> gvecs_;
    std::array<int, 3> max_miller_{};

    void enumerate_gvectors(const Crystal& crystal);
};

} // namespace kronos
