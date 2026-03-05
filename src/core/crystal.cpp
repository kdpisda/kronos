#include "core/crystal.hpp"
#include "core/constants.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace kronos {

// -------------------------------------------------------------------------
// Construction
// -------------------------------------------------------------------------

Crystal::Crystal(const Mat3& lattice_angstrom, std::vector<Atom> atoms)
    : lattice_(lattice_angstrom), atoms_(std::move(atoms))
{
    if (atoms_.empty()) {
        throw std::invalid_argument("Crystal: at least one atom is required");
    }
    compute_derived();
    if (volume_ <= 0.0) {
        throw std::invalid_argument(
            "Crystal: lattice vectors are degenerate (volume <= 0)");
    }
}

// -------------------------------------------------------------------------
// Public accessors
// -------------------------------------------------------------------------

const Mat3& Crystal::lattice() const { return lattice_; }

Mat3 Crystal::lattice_bohr() const {
    Mat3 lb{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            lb[i][j] = lattice_[i][j] * constants::angstrom_to_bohr;
    return lb;
}

Mat3 Crystal::reciprocal_lattice() const { return recip_lattice_; }

double Crystal::volume() const { return volume_; }

size_t Crystal::num_atoms() const { return atoms_.size(); }

const std::vector<Atom>& Crystal::atoms() const { return atoms_; }

const Atom& Crystal::atom(size_t i) const {
    if (i >= atoms_.size()) {
        throw std::out_of_range("Crystal::atom: index out of range");
    }
    return atoms_[i];
}

int Crystal::total_electrons() const {
    int total = 0;
    for (const auto& a : atoms_) {
        total += a.atomic_number;
    }
    return total;
}

// -------------------------------------------------------------------------
// Coordinate transformations
// -------------------------------------------------------------------------

Vec3 Crystal::frac_to_cart(const Vec3& frac) const {
    // cart = frac[0]*a0 + frac[1]*a1 + frac[2]*a2  (in bohr)
    const Mat3 lb = lattice_bohr();
    Vec3 cart{};
    for (int j = 0; j < 3; ++j) {
        cart[j] = frac[0] * lb[0][j]
                + frac[1] * lb[1][j]
                + frac[2] * lb[2][j];
    }
    return cart;
}

Vec3 Crystal::cart_to_frac(const Vec3& cart) const {
    // frac = cart . (b_i / 2pi)  since b_i . a_j = 2pi delta_ij
    // => frac[i] = cart . b_i / (2pi)
    Vec3 frac{};
    for (int i = 0; i < 3; ++i) {
        frac[i] = (cart[0] * recip_lattice_[i][0]
                 + cart[1] * recip_lattice_[i][1]
                 + cart[2] * recip_lattice_[i][2]) / constants::two_pi;
    }
    return frac;
}

// -------------------------------------------------------------------------
// Derived quantities
// -------------------------------------------------------------------------

namespace {

/// 3D cross product.
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

/// 3D dot product.
inline double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

} // anonymous namespace

void Crystal::compute_derived() {
    // Convert lattice from angstrom to bohr.
    Mat3 lb{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            lb[i][j] = lattice_[i][j] * constants::angstrom_to_bohr;

    const Vec3& a0 = lb[0];
    const Vec3& a1 = lb[1];
    const Vec3& a2 = lb[2];

    // Volume = |a0 . (a1 x a2)|
    Vec3 a1_cross_a2 = cross(a1, a2);
    volume_ = std::abs(dot(a0, a1_cross_a2));

    // Reciprocal lattice: b_i = 2*pi * (a_j x a_k) / V
    // b0 = 2pi * (a1 x a2) / V
    // b1 = 2pi * (a2 x a0) / V
    // b2 = 2pi * (a0 x a1) / V
    Vec3 a2_cross_a0 = cross(a2, a0);
    Vec3 a0_cross_a1 = cross(a0, a1);

    for (int j = 0; j < 3; ++j) {
        recip_lattice_[0][j] = constants::two_pi * a1_cross_a2[j] / volume_;
        recip_lattice_[1][j] = constants::two_pi * a2_cross_a0[j] / volume_;
        recip_lattice_[2][j] = constants::two_pi * a0_cross_a1[j] / volume_;
    }
}

} // namespace kronos
