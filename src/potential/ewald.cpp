#include "potential/ewald.hpp"
#include "core/constants.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace kronos {

// ---------------------------------------------------------------------------
// Coulomb constant in Rydberg atomic units: e^2 = 2
// ---------------------------------------------------------------------------
static constexpr double e2 = 2.0;

// Tolerance for convergence of lattice sums
static constexpr double convergence_tol = 1.0e-12;

// ---------------------------------------------------------------------------
// Helper: compute Cartesian positions of all atoms (in bohr)
// ---------------------------------------------------------------------------
static std::vector<Vec3> cartesian_positions(const Crystal& crystal)
{
    const size_t natoms = crystal.num_atoms();
    std::vector<Vec3> pos(natoms);
    for (size_t i = 0; i < natoms; ++i) {
        pos[i] = crystal.frac_to_cart(crystal.atom(i).position);
    }
    return pos;
}

// ---------------------------------------------------------------------------
// Helper: vector arithmetic on Vec3
// ---------------------------------------------------------------------------
static inline Vec3 vec3_add(const Vec3& a, const Vec3& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

static inline Vec3 vec3_sub(const Vec3& a, const Vec3& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

static inline double vec3_dot(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline double vec3_norm(const Vec3& v)
{
    return std::sqrt(vec3_dot(v, v));
}

// ---------------------------------------------------------------------------
// optimal_eta  --  balances real/reciprocal convergence
//
// eta = sqrt(pi) * (N / V^2)^(1/6)
// ---------------------------------------------------------------------------
double EwaldCalculator::optimal_eta(double volume, int num_atoms)
{
    assert(volume > 0.0);
    assert(num_atoms > 0);

    const double pi = constants::pi;
    return std::sqrt(pi)
           * std::pow(static_cast<double>(num_atoms) / (volume * volume),
                      1.0 / 6.0);
}

// ---------------------------------------------------------------------------
// real_space_energy
//
// E_real = e^2 * sum_{i<j, all R} Z_i Z_j erfc(eta |r_ij+R|) / |r_ij+R|
//        + e^2 * (1/2) sum_{i, R!=0} Z_i^2 erfc(eta |R|) / |R|
//
// We implement this as the standard double sum over all pairs (i,j) and
// all lattice vectors R, counting each (i,j,R) once with the appropriate
// weight:
//   - For i != j: sum over all R (no restriction), weight 1/2 (pair counted
//     twice when we also do j,i).
//   - For i == j: sum over R != 0, weight 1/2.
//   Altogether: (1/2) sum_{i,j,R}' Z_i Z_j erfc(eta|r_ij+R|)/|r_ij+R|
//   where the prime means exclude i==j, R==0.
// ---------------------------------------------------------------------------
double EwaldCalculator::real_space_energy(const Crystal& crystal,
                                          const std::vector<double>& charges,
                                          double eta)
{
    const size_t natoms = crystal.num_atoms();
    const auto pos = cartesian_positions(crystal);
    const Mat3 lat = crystal.lattice_bohr();

    // Real-space cutoff: choose R_cut so that erfc(eta*R_cut) < tol
    // erfc(x) ~ exp(-x^2), so we need eta*R_cut ~ sqrt(-ln(tol))
    // Use R_cut = sqrt(-ln(tol)) / eta, but at least 6/eta
    const double rcut = std::max(6.0 / eta,
                                 std::sqrt(-std::log(convergence_tol)) / eta);

    // Determine N_max for each lattice direction.
    // |n1 a1 + n2 a2 + n3 a3| <= rcut
    // Conservative: n_i_max = ceil(rcut / |a_i| * safety)
    // We use the minimum image approach: for each direction, the maximum
    // number of cells is ceil(rcut / (perpendicular height)).
    // Simpler and safe: n_max = ceil(rcut / min_lattice_component_norm)
    // Actually, the safe way is to compute n_max per direction from the
    // lattice vector lengths.
    auto vec_len = [](const std::array<double, 3>& v) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    };

    const int n1max = static_cast<int>(std::ceil(rcut / vec_len(lat[0]))) + 1;
    const int n2max = static_cast<int>(std::ceil(rcut / vec_len(lat[1]))) + 1;
    const int n3max = static_cast<int>(std::ceil(rcut / vec_len(lat[2]))) + 1;

    double esum = 0.0;

    for (int n1 = -n1max; n1 <= n1max; ++n1) {
        for (int n2 = -n2max; n2 <= n2max; ++n2) {
            for (int n3 = -n3max; n3 <= n3max; ++n3) {
                // Lattice translation R = n1*a1 + n2*a2 + n3*a3
                Vec3 R{};
                for (int d = 0; d < 3; ++d) {
                    R[d] = n1 * lat[0][d] + n2 * lat[1][d] + n3 * lat[2][d];
                }

                const bool R_is_zero = (n1 == 0 && n2 == 0 && n3 == 0);

                for (size_t i = 0; i < natoms; ++i) {
                    for (size_t j = 0; j < natoms; ++j) {
                        // Skip i==j when R==0
                        if (R_is_zero && i == j) continue;

                        Vec3 rij = vec3_add(vec3_sub(pos[j], pos[i]), R);
                        const double dist = vec3_norm(rij);

                        if (dist < 1.0e-14) continue; // safety
                        if (dist > rcut) continue;

                        esum += charges[i] * charges[j]
                                * std::erfc(eta * dist) / dist;
                    }
                }
            }
        }
    }

    // Factor 1/2 for double counting, times e^2 for Rydberg units
    return 0.5 * e2 * esum;
}

// ---------------------------------------------------------------------------
// recip_space_energy
//
// E_recip = (e^2 * 4*pi / V) sum_{G!=0} exp(-|G|^2/(4*eta^2)) / |G|^2
//           * |S(G)|^2
//
// where S(G) = sum_i Z_i exp(-i G . r_i)
// ---------------------------------------------------------------------------
double EwaldCalculator::recip_space_energy(const Crystal& crystal,
                                           const std::vector<double>& charges,
                                           double eta)
{
    const size_t natoms = crystal.num_atoms();
    const auto pos = cartesian_positions(crystal);
    const double volume = crystal.volume();
    const Mat3 b = crystal.reciprocal_lattice();

    // Reciprocal-space cutoff: exp(-G_cut^2/(4*eta^2)) < tol
    // => G_cut^2 > -4*eta^2 * ln(tol)
    const double gcut2 = -4.0 * eta * eta * std::log(convergence_tol);
    const double gcut = std::sqrt(gcut2);

    // Determine M_max for each reciprocal lattice direction
    auto vec_len = [](const std::array<double, 3>& v) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    };

    const int m1max = static_cast<int>(std::ceil(gcut / vec_len(b[0]))) + 1;
    const int m2max = static_cast<int>(std::ceil(gcut / vec_len(b[1]))) + 1;
    const int m3max = static_cast<int>(std::ceil(gcut / vec_len(b[2]))) + 1;

    const double four_eta2 = 4.0 * eta * eta;
    const double prefactor = e2 * constants::four_pi / volume;

    double esum = 0.0;

    for (int m1 = -m1max; m1 <= m1max; ++m1) {
        for (int m2 = -m2max; m2 <= m2max; ++m2) {
            for (int m3 = -m3max; m3 <= m3max; ++m3) {
                // Skip G = 0
                if (m1 == 0 && m2 == 0 && m3 == 0) continue;

                // G = m1*b1 + m2*b2 + m3*b3
                Vec3 G{};
                for (int d = 0; d < 3; ++d) {
                    G[d] = m1 * b[0][d] + m2 * b[1][d] + m3 * b[2][d];
                }

                const double g2 = vec3_dot(G, G);
                if (g2 > gcut2) continue;

                const double gaussian = std::exp(-g2 / four_eta2);

                // Structure factor S(G) = sum_i Z_i exp(-i G.r_i)
                double sr = 0.0; // real part
                double si = 0.0; // imaginary part
                for (size_t ia = 0; ia < natoms; ++ia) {
                    const double phase = vec3_dot(G, pos[ia]);
                    sr += charges[ia] * std::cos(phase);
                    si -= charges[ia] * std::sin(phase);
                }

                // |S(G)|^2 = sr^2 + si^2
                const double sg2 = sr * sr + si * si;

                esum += gaussian / g2 * sg2;
            }
        }
    }

    return prefactor * esum;
}

// ---------------------------------------------------------------------------
// self_energy
//
// E_self = -e^2 * (eta / sqrt(pi)) * sum_i Z_i^2
// ---------------------------------------------------------------------------
double EwaldCalculator::self_energy(const std::vector<double>& charges,
                                    double eta)
{
    double z2sum = 0.0;
    for (const double z : charges) {
        z2sum += z * z;
    }

    return -e2 * (eta / std::sqrt(constants::pi)) * z2sum;
}

// ---------------------------------------------------------------------------
// charged_correction
//
// E_charged = -e^2 * pi / (V * eta^2) * (sum_i Z_i)^2
//
// This is nonzero only for charged unit cells (rare, but handle correctly).
// ---------------------------------------------------------------------------
double EwaldCalculator::charged_correction(const std::vector<double>& charges,
                                           double volume, double eta)
{
    double zsum = 0.0;
    for (const double z : charges) {
        zsum += z;
    }

    if (std::abs(zsum) < 1.0e-14) return 0.0;

    return -e2 * constants::pi / (volume * eta * eta) * zsum * zsum;
}

// ---------------------------------------------------------------------------
// real_space_forces
//
// Derivation:
//   E_real = (e2/2) sum'_{i,j,R} Z_i Z_j erfc(eta r)/r
//   where r = |r_j - r_i + R| and prime excludes i==j, R==0.
//
//   Let f(r) = erfc(eta r)/r.  Then:
//     f'(r) = -erfc(eta r)/r^2 - (2 eta/sqrt(pi)) exp(-eta^2 r^2)/r
//     dr/dr_i = -(r_j - r_i + R)/r
//
//   So d[f(r)]/dr_i = -f'(r) * (r_j-r_i+R)/r
//                    = [erfc(eta r)/r^2 + (2eta/sqrt(pi)) exp(-eta^2 r^2)/r]
//                      * (r_j - r_i + R) / r
//
//   In the double sum (i,j) and (j,i) both appear; combining contributions
//   to dE/dr_i gives a factor of 2 that cancels the 1/2 prefactor.
//
//   F_i = -dE/dr_i = -e2 * sum'_{j,R} Z_i Z_j
//         * [erfc(eta r)/r^2 + (2eta/sqrt(pi)) exp(-eta^2 r^2)/r]
//         * (r_j - r_i + R) / r
// ---------------------------------------------------------------------------
std::vector<Vec3> EwaldCalculator::real_space_forces(const Crystal& crystal,
                                                      const std::vector<double>& charges,
                                                      double eta)
{
    const size_t natoms = crystal.num_atoms();
    const auto pos = cartesian_positions(crystal);
    const Mat3 lat = crystal.lattice_bohr();

    const double rcut = std::max(6.0 / eta,
                                 std::sqrt(-std::log(convergence_tol)) / eta);

    auto vec_len = [](const std::array<double, 3>& v) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    };

    const int n1max = static_cast<int>(std::ceil(rcut / vec_len(lat[0]))) + 1;
    const int n2max = static_cast<int>(std::ceil(rcut / vec_len(lat[1]))) + 1;
    const int n3max = static_cast<int>(std::ceil(rcut / vec_len(lat[2]))) + 1;

    const double two_eta_over_sqrtpi = 2.0 * eta / std::sqrt(constants::pi);
    const double eta2 = eta * eta;

    std::vector<Vec3> forces(natoms, {0.0, 0.0, 0.0});

    for (int n1 = -n1max; n1 <= n1max; ++n1) {
        for (int n2 = -n2max; n2 <= n2max; ++n2) {
            for (int n3 = -n3max; n3 <= n3max; ++n3) {
                Vec3 R{};
                for (int d = 0; d < 3; ++d) {
                    R[d] = n1 * lat[0][d] + n2 * lat[1][d] + n3 * lat[2][d];
                }

                const bool R_is_zero = (n1 == 0 && n2 == 0 && n3 == 0);

                for (size_t i = 0; i < natoms; ++i) {
                    for (size_t j = 0; j < natoms; ++j) {
                        if (R_is_zero && i == j) continue;

                        // r_vec = r_j - r_i + R
                        Vec3 rij = vec3_add(vec3_sub(pos[j], pos[i]), R);
                        const double dist = vec3_norm(rij);

                        if (dist < 1.0e-14) continue;
                        if (dist > rcut) continue;

                        const double erfc_val = std::erfc(eta * dist);
                        const double exp_val = std::exp(-eta2 * dist * dist);

                        // Scalar part of the force:
                        // [erfc(eta r)/r^2 + (2 eta/sqrt(pi)) exp(-eta^2 r^2)/r]
                        // divided by r for the unit vector
                        const double scalar =
                            charges[i] * charges[j]
                            * (erfc_val / (dist * dist)
                               + two_eta_over_sqrtpi * exp_val / dist)
                            / dist;

                        // F_i -= e2 * scalar * r_vec
                        // (negative because F = -dE/dr)
                        for (int d = 0; d < 3; ++d) {
                            forces[i][d] -= e2 * scalar * rij[d];
                        }
                    }
                }
            }
        }
    }

    return forces;
}

// ---------------------------------------------------------------------------
// recip_space_forces
//
// Derivation:
//   E_recip = (e2 4pi/V) sum_{G!=0} exp(-G^2/(4 eta^2))/G^2 * |S(G)|^2
//   S(G) = sum_j Z_j exp(-i G.r_j) = sr + i*si
//     where sr = sum Z_j cos(G.r_j), si = -sum Z_j sin(G.r_j)
//
//   d|S|^2/dr_i = -2 G Z_i Im[S^* exp(-iG.r_i)]
//
//   Expanding S^* exp(-iG.r_i) = (sr - i si)(cos phi_i - i sin phi_i):
//     Im[S^* exp(-iG.r_i)] = -[sr sin(phi_i) + si cos(phi_i)]
//
//   So d|S|^2/dr_i = 2 G Z_i [sr sin(phi_i) + si cos(phi_i)]
//
//   dE/dr_i = (e2 8pi/V) Z_i sum_{G!=0} [exp(-G^2/(4eta^2))/G^2]
//             * G * [sr sin(phi_i) + si cos(phi_i)]
//
//   F_i = -dE/dr_i = -(e2 8pi/V) Z_i sum_{G!=0} ...
//         * G * [sr sin(phi_i) + si cos(phi_i)]
// ---------------------------------------------------------------------------
std::vector<Vec3> EwaldCalculator::recip_space_forces(const Crystal& crystal,
                                                       const std::vector<double>& charges,
                                                       double eta)
{
    const size_t natoms = crystal.num_atoms();
    const auto pos = cartesian_positions(crystal);
    const double volume = crystal.volume();
    const Mat3 bmat = crystal.reciprocal_lattice();

    const double gcut2 = -4.0 * eta * eta * std::log(convergence_tol);
    const double gcut = std::sqrt(gcut2);
    const double four_eta2 = 4.0 * eta * eta;

    auto vec_len = [](const std::array<double, 3>& v) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    };

    const int m1max = static_cast<int>(std::ceil(gcut / vec_len(bmat[0]))) + 1;
    const int m2max = static_cast<int>(std::ceil(gcut / vec_len(bmat[1]))) + 1;
    const int m3max = static_cast<int>(std::ceil(gcut / vec_len(bmat[2]))) + 1;

    const double prefactor = e2 * 8.0 * constants::pi / volume;

    std::vector<Vec3> forces(natoms, {0.0, 0.0, 0.0});

    for (int m1 = -m1max; m1 <= m1max; ++m1) {
        for (int m2 = -m2max; m2 <= m2max; ++m2) {
            for (int m3 = -m3max; m3 <= m3max; ++m3) {
                if (m1 == 0 && m2 == 0 && m3 == 0) continue;

                Vec3 G{};
                for (int d = 0; d < 3; ++d) {
                    G[d] = m1 * bmat[0][d] + m2 * bmat[1][d] + m3 * bmat[2][d];
                }

                const double g2 = vec3_dot(G, G);
                if (g2 > gcut2) continue;

                const double gaussian_over_g2 = std::exp(-g2 / four_eta2) / g2;

                // Structure factor S(G) = sum_j Z_j exp(-i G.r_j)
                double sr = 0.0;
                double si = 0.0;
                for (size_t j = 0; j < natoms; ++j) {
                    const double phase = vec3_dot(G, pos[j]);
                    sr += charges[j] * std::cos(phase);
                    si -= charges[j] * std::sin(phase);
                }

                // For each atom, accumulate force contribution from this G
                for (size_t i = 0; i < natoms; ++i) {
                    const double phase_i = vec3_dot(G, pos[i]);
                    const double sin_i = std::sin(phase_i);
                    const double cos_i = std::cos(phase_i);

                    // Im[S^* exp(-iG.r_i)] = -[sr sin(G.r_i) + si cos(G.r_i)]
                    // The term in the force is:
                    //   [sr sin(G.r_i) + si cos(G.r_i)]
                    const double imag_term = sr * sin_i + si * cos_i;

                    const double coeff = -prefactor * charges[i]
                                         * gaussian_over_g2 * imag_term;

                    for (int d = 0; d < 3; ++d) {
                        forces[i][d] += coeff * G[d];
                    }
                }
            }
        }
    }

    return forces;
}

// ---------------------------------------------------------------------------
// compute  --  main entry point
// ---------------------------------------------------------------------------
EwaldCalculator::Result
EwaldCalculator::compute(const Crystal& crystal,
                         const std::vector<double>& charges)
{
    const size_t natoms = crystal.num_atoms();
    if (charges.size() != natoms) {
        throw std::invalid_argument(
            "EwaldCalculator::compute: charges.size() != num_atoms");
    }

    const double volume = crystal.volume();
    const double eta = optimal_eta(volume, static_cast<int>(natoms));

    // --- Energy ---
    const double e_real  = real_space_energy(crystal, charges, eta);
    const double e_recip = recip_space_energy(crystal, charges, eta);
    const double e_self  = self_energy(charges, eta);
    const double e_chg   = charged_correction(charges, volume, eta);

    const double energy = e_real + e_recip + e_self + e_chg;

    // --- Forces ---
    auto f_real  = real_space_forces(crystal, charges, eta);
    auto f_recip = recip_space_forces(crystal, charges, eta);

    std::vector<Vec3> forces(natoms);
    for (size_t i = 0; i < natoms; ++i) {
        for (int d = 0; d < 3; ++d) {
            forces[i][d] = f_real[i][d] + f_recip[i][d];
        }
    }

    return Result{energy, std::move(forces)};
}

// ---------------------------------------------------------------------------
// compute  --  convenience overload extracting charges from pseudopotentials
// ---------------------------------------------------------------------------
EwaldCalculator::Result
EwaldCalculator::compute(const Crystal& crystal,
                         const std::map<std::string, PseudoPotential>& pseudopotentials)
{
    const size_t natoms = crystal.num_atoms();
    std::vector<double> charges(natoms);

    for (size_t i = 0; i < natoms; ++i) {
        const std::string& symbol = crystal.atom(i).symbol;
        auto it = pseudopotentials.find(symbol);
        if (it == pseudopotentials.end()) {
            throw std::invalid_argument(
                "EwaldCalculator::compute: no pseudopotential for element '"
                + symbol + "'");
        }
        charges[i] = it->second.z_valence;
    }

    return compute(crystal, charges);
}

} // namespace kronos
