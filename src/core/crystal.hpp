#pragma once

#include "core/types.hpp"
#include <stdexcept>
#include <vector>

namespace kronos {

/// Crystal structure: lattice vectors + atomic basis.
///
/// Lattice vectors are stored in angstrom (rows of the 3x3 matrix).
/// Derived quantities (reciprocal lattice, volume) are cached on construction
/// and expressed in atomic units (bohr, 1/bohr, bohr^3).
class Crystal {
public:
    Crystal() = default;

    /// Construct from lattice vectors (angstrom) and atom list.
    /// Throws std::invalid_argument if the lattice is degenerate or no atoms
    /// are provided.
    Crystal(const Mat3& lattice_angstrom, std::vector<Atom> atoms);

    /// Lattice vectors (rows of the matrix) in angstrom.
    [[nodiscard]] const Mat3& lattice() const;

    /// Lattice vectors in bohr.
    [[nodiscard]] Mat3 lattice_bohr() const;

    /// Reciprocal lattice vectors (2*pi/a convention) in 1/bohr.
    [[nodiscard]] Mat3 reciprocal_lattice() const;

    /// Unit cell volume in bohr^3.
    [[nodiscard]] double volume() const;

    /// Number of atoms in the unit cell.
    [[nodiscard]] size_t num_atoms() const;

    /// Read-only access to the full atom list.
    [[nodiscard]] const std::vector<Atom>& atoms() const;

    /// Access the i-th atom (bounds-checked).
    [[nodiscard]] const Atom& atom(size_t i) const;

    /// Total electron count (sum of atomic_number values).
    /// Note: in production this should use pseudopotential valence electrons.
    [[nodiscard]] int total_electrons() const;

    /// Convert fractional coordinates to Cartesian coordinates (bohr).
    [[nodiscard]] Vec3 frac_to_cart(const Vec3& frac) const;

    /// Convert Cartesian coordinates (bohr) to fractional coordinates.
    [[nodiscard]] Vec3 cart_to_frac(const Vec3& cart) const;

private:
    Mat3 lattice_{};                // angstrom
    std::vector<Atom> atoms_;
    Mat3 recip_lattice_{};          // cached reciprocal lattice (1/bohr)
    double volume_{0.0};            // cached volume in bohr^3

    /// Compute reciprocal lattice and volume from the stored lattice.
    void compute_derived();
};

} // namespace kronos
