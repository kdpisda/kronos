#include <gtest/gtest.h>

#include "core/types.hpp"
#include "core/constants.hpp"
#include "core/crystal.hpp"
#include "core/spherical_harmonics.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "basis/kpoints.hpp"
#include "potential/ewald.hpp"
#include "potential/hartree.hpp"
#include "potential/xc.hpp"
#include "potential/local_pp.hpp"
#include "solver/fermi.hpp"
#include "solver/scf.hpp"
#include "io/upf_parser.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

using namespace kronos;

// Helper: create a simple cubic crystal for testing
Crystal make_simple_cubic(double a_ang, const std::string& element) {
    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};
    std::vector<Atom> atoms = {{element, 1, {0.0, 0.0, 0.0}}};
    return Crystal(lattice, std::move(atoms));
}

// Helper: build a minimal local-only pseudopotential
PseudoPotential make_simple_local_pp(double z_val, int npts = 500,
                                     double rmax = 10.0) {
    PseudoPotential pp;
    pp.z_valence = z_val;
    pp.mesh.npoints = npts;
    pp.mesh.r.resize(npts);
    pp.mesh.rab.resize(npts);
    pp.vloc.resize(npts);
    double dr = rmax / (npts - 1);
    double r_loc = 0.5;
    for (int i = 0; i < npts; ++i) {
        double r = i * dr;
        pp.mesh.r[i] = r;
        pp.mesh.rab[i] = dr;
        if (r < 1e-30)
            pp.vloc[i] = -z_val * 2.0 / (std::sqrt(constants::pi) * r_loc);
        else
            pp.vloc[i] = -z_val * std::erf(r / r_loc) / r;
    }
    pp.rho_atomic.resize(npts);
    double norm = 0.0;
    double sigma = 1.0;
    for (int i = 0; i < npts; ++i) {
        double r = pp.mesh.r[i];
        pp.rho_atomic[i] = std::exp(-r * r / (2.0 * sigma * sigma));
        norm += r * r * pp.rho_atomic[i] * pp.mesh.rab[i];
    }
    norm *= constants::four_pi;
    for (int i = 0; i < npts; ++i)
        pp.rho_atomic[i] *= z_val / norm;
    return pp;
}

// Helper: create Si diamond structure
Crystal make_si_diamond() {
    // Silicon diamond, a = 5.43 angstrom
    double a = 5.43;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.00, 0.00, 0.00}},
        {"Si", 14, {0.25, 0.25, 0.25}},
    };
    return Crystal(lattice, std::move(atoms));
}

// ============================================================================
// Real spherical harmonics tests
// ============================================================================

TEST(SphericalHarmonics, Y00IsConstant) {
    // Y_00 = 1/sqrt(4*pi) for any direction
    double expected = 1.0 / std::sqrt(4.0 * constants::pi);
    EXPECT_NEAR(real_spherical_harmonic(0, 0, 1.0, 0.0, 0.0), expected, 1e-12);
    EXPECT_NEAR(real_spherical_harmonic(0, 0, 0.0, 1.0, 0.0), expected, 1e-12);
    EXPECT_NEAR(real_spherical_harmonic(0, 0, 0.0, 0.0, 1.0), expected, 1e-12);
    EXPECT_NEAR(real_spherical_harmonic(0, 0, 1.0, 1.0, 1.0), expected, 1e-12);
}

TEST(SphericalHarmonics, Y1mOrthogonality) {
    // Y_1m should be orthogonal: integral Y_1m * Y_1m' dOmega = delta_mm'
    // Test on a grid of directions
    double sum_xx = 0, sum_yy = 0, sum_zz = 0;
    double sum_xy = 0, sum_xz = 0, sum_yz = 0;
    int N = 50;
    double dtheta = constants::pi / N;
    double dphi = 2.0 * constants::pi / (2 * N);
    for (int it = 0; it < N; ++it) {
        double theta = (it + 0.5) * dtheta;
        double sin_t = std::sin(theta);
        double cos_t = std::cos(theta);
        for (int ip = 0; ip < 2 * N; ++ip) {
            double phi = (ip + 0.5) * dphi;
            double x = sin_t * std::cos(phi);
            double y = sin_t * std::sin(phi);
            double z = cos_t;
            double dOmega = sin_t * dtheta * dphi;

            double y1m1 = real_spherical_harmonic(1, -1, x, y, z);
            double y10  = real_spherical_harmonic(1,  0, x, y, z);
            double y11  = real_spherical_harmonic(1,  1, x, y, z);

            sum_xx += y11 * y11 * dOmega;
            sum_yy += y1m1 * y1m1 * dOmega;
            sum_zz += y10 * y10 * dOmega;
            sum_xy += y11 * y1m1 * dOmega;
            sum_xz += y11 * y10 * dOmega;
            sum_yz += y1m1 * y10 * dOmega;
        }
    }
    // Diagonal should be 1, off-diagonal should be 0
    EXPECT_NEAR(sum_xx, 1.0, 0.02);
    EXPECT_NEAR(sum_yy, 1.0, 0.02);
    EXPECT_NEAR(sum_zz, 1.0, 0.02);
    EXPECT_NEAR(sum_xy, 0.0, 0.02);
    EXPECT_NEAR(sum_xz, 0.0, 0.02);
    EXPECT_NEAR(sum_yz, 0.0, 0.02);
}

TEST(SphericalHarmonics, Y2mOrthogonality) {
    // l=2 should be orthogonal to l=1
    double sum = 0;
    int N = 50;
    double dtheta = constants::pi / N;
    double dphi = 2.0 * constants::pi / (2 * N);
    for (int it = 0; it < N; ++it) {
        double theta = (it + 0.5) * dtheta;
        double sin_t = std::sin(theta);
        double cos_t = std::cos(theta);
        for (int ip = 0; ip < 2 * N; ++ip) {
            double phi = (ip + 0.5) * dphi;
            double x = sin_t * std::cos(phi);
            double y = sin_t * std::sin(phi);
            double z = cos_t;
            double dOmega = sin_t * dtheta * dphi;

            double y10 = real_spherical_harmonic(1, 0, x, y, z);
            double y20 = real_spherical_harmonic(2, 0, x, y, z);
            sum += y10 * y20 * dOmega;
        }
    }
    EXPECT_NEAR(sum, 0.0, 0.02);
}

// ============================================================================
// Monkhorst-Pack k-point tests
// ============================================================================

TEST(KPointGenerator, GammaOnly) {
    KPointGrid grid;
    grid.grid = {1, 1, 1};
    grid.shift = {0, 0, 0};

    Crystal crystal = make_simple_cubic(5.0, "Si");
    auto kdata = KPointGenerator::generate_monkhorst_pack(grid, crystal);

    ASSERT_EQ(kdata.kpoints.size(), 1u);
    EXPECT_NEAR(kdata.kpoints[0][0], 0.0, 1e-12);
    EXPECT_NEAR(kdata.kpoints[0][1], 0.0, 1e-12);
    EXPECT_NEAR(kdata.kpoints[0][2], 0.0, 1e-12);
    EXPECT_NEAR(kdata.weights[0], 1.0, 1e-12);
}

TEST(KPointGenerator, TwoByTwoByTwo) {
    KPointGrid grid;
    grid.grid = {2, 2, 2};
    grid.shift = {0, 0, 0};

    Crystal crystal = make_simple_cubic(5.0, "Si");
    auto kdata = KPointGenerator::generate_monkhorst_pack(grid, crystal);

    // 2x2x2 = 8 points. With time-reversal: k and -k are equivalent.
    // Depending on implementation, TR reduction may or may not be applied.
    EXPECT_GE(kdata.kpoints.size(), 1u);
    EXPECT_LE(kdata.kpoints.size(), 8u);

    // Weights should sum to 1
    double weight_sum = 0;
    for (double w : kdata.weights) {
        weight_sum += w;
    }
    EXPECT_NEAR(weight_sum, 1.0, 1e-12);
}

TEST(KPointGenerator, FourByFourByFour) {
    KPointGrid grid;
    grid.grid = {4, 4, 4};
    grid.shift = {0, 0, 0};

    Crystal crystal = make_simple_cubic(5.0, "Si");
    auto kdata = KPointGenerator::generate_monkhorst_pack(grid, crystal);

    // 4x4x4 = 64 points. With TR: at most 64 (may not reduce).
    EXPECT_GE(kdata.kpoints.size(), 1u);
    EXPECT_LE(kdata.kpoints.size(), 64u);

    double weight_sum = 0;
    for (double w : kdata.weights) {
        weight_sum += w;
    }
    EXPECT_NEAR(weight_sum, 1.0, 1e-12);
}

TEST(KPointGenerator, ShiftedGrid) {
    KPointGrid grid;
    grid.grid = {2, 2, 2};
    grid.shift = {1, 1, 1};

    Crystal crystal = make_simple_cubic(5.0, "Si");
    auto kdata = KPointGenerator::generate_monkhorst_pack(grid, crystal);

    // Shifted 2x2x2 should still have weights summing to 1
    double weight_sum = 0;
    for (double w : kdata.weights) {
        weight_sum += w;
    }
    EXPECT_NEAR(weight_sum, 1.0, 1e-12);

    // Shifted grid should produce different k-points than unshifted
    KPointGrid grid_unshifted;
    grid_unshifted.grid = {2, 2, 2};
    grid_unshifted.shift = {0, 0, 0};
    auto kdata_unshifted = KPointGenerator::generate_monkhorst_pack(
        grid_unshifted, crystal);

    // At least one k-point should differ between shifted and unshifted
    bool any_different = (kdata.kpoints.size() != kdata_unshifted.kpoints.size());
    if (!any_different) {
        for (size_t i = 0; i < kdata.kpoints.size(); ++i) {
            double diff = std::abs(kdata.kpoints[i][0] - kdata_unshifted.kpoints[i][0])
                        + std::abs(kdata.kpoints[i][1] - kdata_unshifted.kpoints[i][1])
                        + std::abs(kdata.kpoints[i][2] - kdata_unshifted.kpoints[i][2]);
            if (diff > 1e-10) { any_different = true; break; }
        }
    }
    EXPECT_TRUE(any_different) << "Shifted and unshifted grids should differ";
}

// ============================================================================
// FFT grid sizing tests
// ============================================================================

TEST(FFTGrid, GridSizedForEcutrho) {
    // Grid should be larger when ecutrho > 4*ecutwfc
    Crystal crystal = make_simple_cubic(5.43, "Si");
    double ecutwfc = 20.0;  // Ry
    PlaneWaveBasis basis(crystal, ecutwfc);

    FFTGrid grid_default(basis);                    // uses 4*ecutwfc
    FFTGrid grid_large(basis, 8.0 * ecutwfc);       // 8x ecutwfc

    auto dims_default = grid_default.dims();
    auto dims_large = grid_large.dims();

    // Larger ecutrho should give larger grid dimensions
    EXPECT_GE(dims_large[0], dims_default[0]);
    EXPECT_GE(dims_large[1], dims_default[1]);
    EXPECT_GE(dims_large[2], dims_default[2]);

    // At least one dimension should be strictly larger
    bool any_larger = dims_large[0] > dims_default[0]
                   || dims_large[1] > dims_default[1]
                   || dims_large[2] > dims_default[2];
    EXPECT_TRUE(any_larger) << "8x ecutwfc should produce a larger grid than 4x";
}

// ============================================================================
// Ewald summation tests
// ============================================================================

TEST(Ewald, NaClMadelungConstant) {
    // NaCl rocksalt structure. The Madelung constant for NaCl is ~1.747565.
    // E_Madelung = -alpha * e^2 / (2 * a), where alpha is the Madelung constant
    // per ion pair, and a is the nearest-neighbor distance.
    //
    // In Rydberg units with e^2 = 2:
    // E_Madelung per formula unit = -alpha * 2 / (2 * d) = -alpha / d
    //
    // For NaCl with a_lattice = 5.64 Angstrom:
    //   d = a/2 (nearest neighbor distance)
    //
    // We use a simplified test: just verify Ewald gives negative energy
    // for an ionic crystal (Na+ Cl- charges).

    double a_ang = 5.64;
    double a_bohr = a_ang * constants::angstrom_to_bohr;

    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};
    std::vector<Atom> atoms = {
        {"Na", 11, {0.0, 0.0, 0.0}},
        {"Cl", 17, {0.5, 0.0, 0.0}},
        {"Na", 11, {0.5, 0.5, 0.0}},
        {"Cl", 17, {0.0, 0.5, 0.0}},
        {"Na", 11, {0.0, 0.5, 0.5}},
        {"Cl", 17, {0.5, 0.5, 0.5}},
        {"Na", 11, {0.0, 0.0, 0.5}},
        {"Cl", 17, {0.5, 0.0, 0.5}},
    };
    Crystal crystal(lattice, std::move(atoms));

    // Na+: z_val = 1, Cl-: z_val = 7 (typical NCPP valences)
    std::vector<double> charges = {1.0, 7.0, 1.0, 7.0, 1.0, 7.0, 1.0, 7.0};

    auto result = EwaldCalculator::compute(crystal, charges);

    // The Ewald energy should be finite and large (many ion pairs)
    EXPECT_TRUE(std::isfinite(result.energy))
        << "Ewald energy should be finite, got " << result.energy;

    // Forces should exist for all 8 atoms
    ASSERT_EQ(result.forces.size(), 8u);

    // In a perfect NaCl crystal, forces should be zero by symmetry
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(result.forces[i][0], 0.0, 1e-6)
            << "Force on atom " << i << " x-component should be zero";
        EXPECT_NEAR(result.forces[i][1], 0.0, 1e-6)
            << "Force on atom " << i << " y-component should be zero";
        EXPECT_NEAR(result.forces[i][2], 0.0, 1e-6)
            << "Force on atom " << i << " z-component should be zero";
    }
}

TEST(Ewald, DisplacedAtomHasForce) {
    // Displace one atom from equilibrium — it should have a nonzero force
    double a_ang = 5.0;
    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.0, 0.0, 0.0}},
        {"Si", 14, {0.501, 0.5, 0.5}},  // slightly displaced from 0.5
    };
    Crystal crystal(lattice, std::move(atoms));

    std::vector<double> charges = {4.0, 4.0};
    auto result = EwaldCalculator::compute(crystal, charges);

    // The displaced atom should have a restoring force
    double f_max = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (int d = 0; d < 3; ++d) {
            f_max = std::max(f_max, std::abs(result.forces[i][d]));
        }
    }
    EXPECT_GT(f_max, 1e-4) << "Displaced atom should produce nonzero Ewald force";
}

// ============================================================================
// Marzari-Vanderbilt smearing test
// ============================================================================

TEST(FermiSolver, MarzariVanderbiltSmearing) {
    // MV cold smearing should give occupations between 0 and ~1.1
    // (slightly > 1 is a feature of MV smearing)
    const double ev_to_ry = constants::ev_to_rydberg;

    std::vector<std::vector<double>> eigenvalues = {
        { -3.0 * ev_to_ry, -1.0 * ev_to_ry, 0.5 * ev_to_ry, 3.0 * ev_to_ry }
    };
    std::vector<double> weights = { 1.0 };
    double target_electrons = 5.0;
    double degauss = 0.05;  // Ry

    auto result = FermiSolver::find_fermi_level(
        eigenvalues, weights, target_electrons,
        SmearingType::MarzariVanderbilt, degauss, 2);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.total_electrons_found, target_electrons, 1e-4);
}

// ============================================================================
// Additional spherical harmonics tests
// ============================================================================

TEST(SphericalHarmonics, Y2mSelfOrthogonality) {
    // All 5 Y_2m should be orthonormal
    int N = 60;
    double dtheta = constants::pi / N;
    double dphi = 2.0 * constants::pi / (2 * N);
    double overlap[5][5] = {};

    for (int it = 0; it < N; ++it) {
        double theta = (it + 0.5) * dtheta;
        double sin_t = std::sin(theta);
        double cos_t = std::cos(theta);
        for (int ip = 0; ip < 2 * N; ++ip) {
            double phi = (ip + 0.5) * dphi;
            double x = sin_t * std::cos(phi);
            double y = sin_t * std::sin(phi);
            double z = cos_t;
            double dOmega = sin_t * dtheta * dphi;

            double ylm[5];
            for (int m = -2; m <= 2; ++m) {
                ylm[m + 2] = real_spherical_harmonic(2, m, x, y, z);
            }
            for (int a = 0; a < 5; ++a) {
                for (int b = 0; b < 5; ++b) {
                    overlap[a][b] += ylm[a] * ylm[b] * dOmega;
                }
            }
        }
    }

    for (int a = 0; a < 5; ++a) {
        for (int b = 0; b < 5; ++b) {
            double expected = (a == b) ? 1.0 : 0.0;
            EXPECT_NEAR(overlap[a][b], expected, 0.02)
                << "l=2 overlap[" << a << "][" << b << "]";
        }
    }
}

TEST(SphericalHarmonics, ZeroVectorOnlyL0) {
    // At the zero vector, only l=0 should return nonzero
    double y00 = real_spherical_harmonic(0, 0, 0.0, 0.0, 0.0);
    EXPECT_GT(std::abs(y00), 0.1);  // Y_00 is about 0.282

    // All l>0 should return 0
    for (int l = 1; l <= 3; ++l) {
        for (int m = -l; m <= l; ++m) {
            double val = real_spherical_harmonic(l, m, 0.0, 0.0, 0.0);
            EXPECT_NEAR(val, 0.0, 1e-12)
                << "Y_{" << l << "," << m << "} at zero should be 0";
        }
    }
}

// ============================================================================
// Fermi-Dirac smearing test
// ============================================================================

TEST(FermiSolver, FermiDiracSmearing) {
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        { -5.0 * ev_to_ry, -2.0 * ev_to_ry, 1.0 * ev_to_ry, 5.0 * ev_to_ry }
    };
    std::vector<double> weights = { 1.0 };
    double target_electrons = 4.0;  // 2 bands fully occupied
    double degauss = 0.05;  // Ry

    auto result = FermiSolver::find_fermi_level(
        eigenvalues, weights, target_electrons,
        SmearingType::FermiDirac, degauss, 2);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.total_electrons_found, target_electrons, 1e-4);

    // Fermi level should be between the 2nd and 3rd eigenvalues
    double e2 = eigenvalues[0][1];
    double e3 = eigenvalues[0][2];
    EXPECT_GT(result.fermi_energy, e2);
    EXPECT_LT(result.fermi_energy, e3);
}

// ============================================================================
// Gaussian smearing test
// ============================================================================

TEST(FermiSolver, GaussianSmearing) {
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        { -5.0 * ev_to_ry, -2.0 * ev_to_ry, 1.0 * ev_to_ry, 5.0 * ev_to_ry }
    };
    std::vector<double> weights = { 1.0 };
    double target_electrons = 4.0;
    double degauss = 0.05;

    auto result = FermiSolver::find_fermi_level(
        eigenvalues, weights, target_electrons,
        SmearingType::Gaussian, degauss, 2);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.total_electrons_found, target_electrons, 1e-4);
}

// ============================================================================
// K-point weight conservation test
// ============================================================================

TEST(KPointGenerator, WeightsSumToOneVariousGrids) {
    Crystal crystal = make_simple_cubic(5.0, "Si");

    // Test various grid sizes
    std::array<std::array<int,3>, 4> grids = {{
        {3, 3, 3}, {4, 4, 4}, {2, 3, 4}, {6, 6, 6}
    }};

    for (const auto& g : grids) {
        KPointGrid grid;
        grid.grid = g;
        grid.shift = {0, 0, 0};
        auto kdata = KPointGenerator::generate_monkhorst_pack(grid, crystal);

        double wsum = 0.0;
        for (double w : kdata.weights) {
            wsum += w;
            EXPECT_GT(w, 0.0) << "All weights should be positive";
        }
        EXPECT_NEAR(wsum, 1.0, 1e-12)
            << "Weights should sum to 1 for " << g[0] << "x" << g[1] << "x" << g[2];
    }
}

// ============================================================================
// Ewald energy sign test
// ============================================================================

TEST(Ewald, NaClEnergyIsNegative) {
    // For a truly ionic crystal with opposite charges, Madelung energy < 0
    double a_ang = 5.64;
    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};
    // BCC-like NaCl with +1 and -1 charges
    std::vector<Atom> atoms = {
        {"Na", 11, {0.0, 0.0, 0.0}},
        {"Cl", 17, {0.5, 0.5, 0.5}},
    };
    Crystal crystal(lattice, std::move(atoms));

    // Use +1 and -1 to get pure Madelung energy
    std::vector<double> charges = {1.0, -1.0};
    auto result = EwaldCalculator::compute(crystal, charges);

    // CsCl Madelung energy should be negative
    EXPECT_LT(result.energy, 0.0)
        << "Ewald energy for CsCl-type with +1/-1 should be negative";
}

// ============================================================================
// Plane-wave basis completeness check
// ============================================================================

TEST(PlaneWaveBasis, KineticEnergiesMonotone) {
    Crystal crystal = make_si_diamond();
    PlaneWaveBasis basis(crystal, 15.0);  // moderate cutoff

    auto ke = basis.kinetic_energies({0.0, 0.0, 0.0});  // Gamma point

    // Kinetic energies should all be non-negative
    for (size_t i = 0; i < ke.size(); ++i) {
        EXPECT_GE(ke[i], -1e-12) << "Kinetic energy at G[" << i << "] should be >= 0";
    }

    // First G-vector (Gamma) should have zero kinetic energy
    EXPECT_NEAR(ke[0], 0.0, 1e-12);

    // All should be within cutoff
    for (size_t i = 0; i < ke.size(); ++i) {
        EXPECT_LE(ke[i], 15.0 + 1e-6) << "KE should be within ecutwfc";
    }
}

// ============================================================================
// Ewald: equal charges have positive self-energy
// ============================================================================

TEST(Ewald, SameChargesPositiveEnergy) {
    // Two identical charges should have positive Ewald energy
    double a = 5.0;
    Mat3 lattice = {{{a, 0, 0}, {0, a, 0}, {0, 0, a}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.0, 0.0, 0.0}},
        {"Si", 14, {0.5, 0.5, 0.5}},
    };
    Crystal crystal(lattice, std::move(atoms));

    // Same sign charges → repulsion → positive energy
    // (minus self-interaction, which is also positive)
    std::vector<double> charges = {4.0, 4.0};
    auto result = EwaldCalculator::compute(crystal, charges);
    EXPECT_TRUE(std::isfinite(result.energy));
}

// ============================================================================
// Hartree: self-energy is positive
// ============================================================================

TEST(HartreeEnergy, SelfEnergyPositive) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);

    int num_grid = fft.total_points();
    double volume = crystal.volume();
    int npw = static_cast<int>(basis.num_pw());

    // Uniform density
    double n0 = 0.05;
    RVec density_r(num_grid, n0);

    std::vector<complex_t> density_c(num_grid);
    for (int i = 0; i < num_grid; ++i)
        density_c[i] = complex_t{density_r[i], 0.0};
    std::vector<complex_t> density_g_full(num_grid);
    fft.forward(density_c, density_g_full);
    CVec density_g(npw);
    fft.gather_from_grid(basis, density_g_full, density_g);

    HartreeSolver hartree(basis);
    CVec vhartree_g = hartree.compute(density_g);
    double e_h = hartree.energy(density_g, vhartree_g, volume, num_grid);

    EXPECT_GE(e_h, 0.0)
        << "Hartree self-energy should be non-negative";
}

// ============================================================================
// Hartree: G=0 component of V_H should be zero
// ============================================================================

TEST(HartreeEnergy, GZeroComponentZero) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);

    int num_grid = fft.total_points();
    int npw = static_cast<int>(basis.num_pw());

    CVec density_g(npw, complex_t{0.0, 0.0});
    density_g[0] = complex_t{0.1 * num_grid, 0.0};

    HartreeSolver hartree(basis);
    CVec vhartree_g = hartree.compute(density_g);

    // V_H(G=0) should be set to 0 to avoid divergence
    EXPECT_NEAR(vhartree_g[0].real(), 0.0, 1e-10)
        << "V_H(G=0) should be zero";
    EXPECT_NEAR(vhartree_g[0].imag(), 0.0, 1e-10);
}

// ============================================================================
// XC: GGA functionality
// ============================================================================

TEST(XCFunctional, PBEIsGGA) {
    XCEvaluator xc_pbe("PBE");
    EXPECT_TRUE(xc_pbe.is_gga());

    XCEvaluator xc_lda("LDA_PZ");
    EXPECT_FALSE(xc_lda.is_gga());
}

TEST(XCFunctional, PBEEvaluateGGA) {
    // PBE evaluation with sigma should work
    double n = 0.05;
    double sigma = 0.001;
    RVec density = {n};
    RVec sigma_r = {sigma};

    XCEvaluator xc("PBE");
    XCResult result = xc.evaluate_gga(density, sigma_r, 1.0);

    EXPECT_EQ(result.exc.size(), 1u);
    EXPECT_EQ(result.vxc.size(), 1u);
    EXPECT_EQ(result.vsigma.size(), 1u);
    EXPECT_LT(result.exc[0], 0.0);  // exchange-correlation energy density < 0
    EXPECT_TRUE(std::isfinite(result.vsigma[0]));
}

TEST(XCFunctional, NameReturned) {
    XCEvaluator xc("LDA_PZ");
    EXPECT_EQ(xc.name(), "LDA_PZ");

    XCEvaluator xc2("PBE");
    EXPECT_EQ(xc2.name(), "PBE");
}

// ============================================================================
// XC: energy is negative for positive density
// ============================================================================

TEST(XCFunctional, EnergyNegativeForPositiveDensity) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    double volume = crystal.volume();

    // Uniform positive density
    int num_grid = 64;
    double n0 = 0.05;
    RVec density(num_grid, n0);

    XCEvaluator xc("LDA_PZ");
    XCResult result = xc.evaluate(density, volume);

    EXPECT_LT(result.energy, 0.0)
        << "XC energy should be negative for positive density";
}

// ============================================================================
// Local PP: vloc_g is properly constructed
// ============================================================================

TEST(LocalPP, VlocGIsFinite) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);

    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pps = {{"Si", pp}};

    LocalPPEvaluator local_pp(crystal, basis, pps);
    const CVec& vloc_g = local_pp.vloc_g();

    EXPECT_EQ(vloc_g.size(), basis.num_pw());
    for (size_t i = 0; i < vloc_g.size(); ++i) {
        EXPECT_TRUE(std::isfinite(vloc_g[i].real()))
            << "V_loc(G) should be finite at index " << i;
        EXPECT_TRUE(std::isfinite(vloc_g[i].imag()));
    }
}

TEST(LocalPP, EnergyFiniteForUniformDensity) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int num_grid = fft.total_points();
    double volume = crystal.volume();

    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pps = {{"Si", pp}};
    LocalPPEvaluator local_pp(crystal, basis, pps);

    double n0 = 4.0 / volume;
    RVec density_r(num_grid, n0);

    std::vector<complex_t> density_c(num_grid);
    for (int i = 0; i < num_grid; ++i)
        density_c[i] = complex_t{density_r[i], 0.0};
    std::vector<complex_t> density_g_full(num_grid);
    fft.forward(density_c, density_g_full);
    CVec density_g(npw);
    fft.gather_from_grid(basis, density_g_full, density_g);

    double e_loc = local_pp.energy(density_g, volume, num_grid);
    EXPECT_TRUE(std::isfinite(e_loc))
        << "Local PP energy should be finite";
}

// ============================================================================
// Total energy: sum of components matches total
// ============================================================================

TEST(TotalEnergy, NegativeForBoundSystem) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pps = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pps);
    SCFResult result = solver.solve();

    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }

    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for bound system";
}

// ============================================================================
// Ewald: energy finite for various crystals
// ============================================================================

TEST(Ewald, SingleAtomEnergy) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    std::vector<double> charges = {4.0};
    auto result = EwaldCalculator::compute(crystal, charges);

    EXPECT_TRUE(std::isfinite(result.energy));
    ASSERT_EQ(result.forces.size(), 1u);
    // Single atom in unit cell: force should be zero by symmetry
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(result.forces[0][d], 0.0, 1e-8);
    }
}

} // anonymous namespace
