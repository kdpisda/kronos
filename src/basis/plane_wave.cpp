#include "basis/plane_wave.hpp"
#include "core/constants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kronos {

// -------------------------------------------------------------------------
// Construction
// -------------------------------------------------------------------------

PlaneWaveBasis::PlaneWaveBasis(const Crystal& crystal, double ecutwfc, double k_max)
    : ecutwfc_(ecutwfc), recip_lattice_(crystal.reciprocal_lattice())
{
    if (ecutwfc_ <= 0.0) {
        throw std::invalid_argument(
            "PlaneWaveBasis: ecutwfc must be positive");
    }

    // Expand the G-vector sphere to ensure completeness for all k-points.
    //
    // The physics cutoff is |k+G|^2 <= ecutwfc for each k-point.
    // With a shared basis, we need all G where |k+G|^2 <= ecutwfc for ANY k
    // with |k| <= k_max. The worst case is when k and G are anti-parallel:
    //   |k+G|^2 = (|G| - |k|)^2 <= ecutwfc
    //   |G| <= sqrt(ecutwfc) + |k|
    //   |G|^2 <= (sqrt(ecutwfc) + k_max)^2
    if (k_max > 0.0) {
        double g_max_expanded = std::sqrt(ecutwfc_) + k_max;
        gvec_cutoff_ = g_max_expanded * g_max_expanded;
    } else {
        gvec_cutoff_ = ecutwfc_;
    }

    enumerate_gvectors(crystal);
}

// -------------------------------------------------------------------------
// Public accessors
// -------------------------------------------------------------------------

size_t PlaneWaveBasis::num_pw() const { return gvecs_.size(); }

const std::vector<GVector>& PlaneWaveBasis::gvectors() const {
    return gvecs_;
}

const GVector& PlaneWaveBasis::gvec(size_t i) const {
    if (i >= gvecs_.size()) {
        throw std::out_of_range("PlaneWaveBasis::gvec: index out of range");
    }
    return gvecs_[i];
}

std::vector<double> PlaneWaveBasis::kinetic_energies(const Vec3& k_frac) const {
    // Convert k from fractional reciprocal coordinates to Cartesian
    // k_cart = k_frac[0]*b1 + k_frac[1]*b2 + k_frac[2]*b3
    Vec3 k_cart{};
    for (int j = 0; j < 3; ++j) {
        k_cart[j] = k_frac[0] * recip_lattice_[0][j]
                  + k_frac[1] * recip_lattice_[1][j]
                  + k_frac[2] * recip_lattice_[2][j];
    }

    std::vector<double> ekin(gvecs_.size());
    for (size_t i = 0; i < gvecs_.size(); ++i) {
        const auto& g = gvecs_[i];
        double kpg0 = k_cart[0] + g.cart[0];
        double kpg1 = k_cart[1] + g.cart[1];
        double kpg2 = k_cart[2] + g.cart[2];
        // Rydberg atomic units: T = |k+G|^2 (m_e = 1/2, so T = p^2/(2m) = |k+G|^2)
        ekin[i] = kpg0 * kpg0 + kpg1 * kpg1 + kpg2 * kpg2;
    }
    return ekin;
}

double PlaneWaveBasis::ecutwfc() const { return ecutwfc_; }

std::array<int, 3> PlaneWaveBasis::max_miller() const { return max_miller_; }

// -------------------------------------------------------------------------
// G-vector enumeration
// -------------------------------------------------------------------------

void PlaneWaveBasis::enumerate_gvectors(const Crystal& crystal) {
    const Mat3& b = recip_lattice_;

    // Use the (possibly expanded) cutoff for the G-vector sphere
    double g_max = std::sqrt(gvec_cutoff_);

    for (int i = 0; i < 3; ++i) {
        double b_norm = std::sqrt(b[i][0] * b[i][0]
                                + b[i][1] * b[i][1]
                                + b[i][2] * b[i][2]);
        max_miller_[i] = static_cast<int>(std::floor(g_max / b_norm)) + 1;
    }

    // Enumerate all integer triplets (h,k,l)
    gvecs_.clear();
    gvecs_.reserve(static_cast<size_t>(
        (2 * max_miller_[0] + 1) *
        (2 * max_miller_[1] + 1) *
        (2 * max_miller_[2] + 1)));

    for (int h = -max_miller_[0]; h <= max_miller_[0]; ++h) {
        for (int k = -max_miller_[1]; k <= max_miller_[1]; ++k) {
            for (int l = -max_miller_[2]; l <= max_miller_[2]; ++l) {
                // G_cart = h*b1 + k*b2 + l*b3
                Vec3 g_cart{};
                for (int j = 0; j < 3; ++j) {
                    g_cart[j] = h * b[0][j] + k * b[1][j] + l * b[2][j];
                }

                double norm2 = g_cart[0] * g_cart[0]
                             + g_cart[1] * g_cart[1]
                             + g_cart[2] * g_cart[2];

                // Include G if |G|^2 <= gvec_cutoff
                if (norm2 <= gvec_cutoff_) {
                    GVector gv;
                    gv.h = h;
                    gv.k = k;
                    gv.l = l;
                    gv.cart = g_cart;
                    gv.norm2 = norm2;
                    gvecs_.push_back(gv);
                }
            }
        }
    }

    // Sort by kinetic energy (ascending) so G=0 comes first
    std::sort(gvecs_.begin(), gvecs_.end(),
              [](const GVector& a, const GVector& b) {
                  return a.norm2 < b.norm2;
              });
}

} // namespace kronos
