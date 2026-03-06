#include <gtest/gtest.h>

#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "core/crystal.hpp"
#include "core/constants.hpp"
#include "core/types.hpp"

#include <cmath>
#include <algorithm>

namespace {

// Helper: create Si diamond crystal (a = 5.43 angstrom)
kronos::Crystal make_si_diamond() {
    const double a = 5.43;  // angstrom
    // FCC lattice vectors
    kronos::Mat3 lattice = {{
        {{ 0.0,   a/2.0, a/2.0 }},
        {{ a/2.0, 0.0,   a/2.0 }},
        {{ a/2.0, a/2.0, 0.0   }}
    }};
    // Two Si atoms in the basis
    std::vector<kronos::Atom> atoms = {
        {"Si", 14, {0.0,  0.0,  0.0 }},
        {"Si", 14, {0.25, 0.25, 0.25}}
    };
    return kronos::Crystal(lattice, atoms);
}

// Helper: check if n factors only into 2, 3, 5
bool is_fft_friendly(int n) {
    if (n <= 0) return false;
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
}

} // anonymous namespace

// ============================================================================
// PlaneWaveBasis tests
// ============================================================================

TEST(PlaneWaveBasis, SiBulkBasisSize) {
    auto crystal = make_si_diamond();
    double ecutwfc = 30.0;  // Ry
    kronos::PlaneWaveBasis basis(crystal, ecutwfc);

    // For Si diamond at 30 Ry, the number of plane waves should be
    // in the hundreds (typically ~500-2000 depending on convention).
    // We just check it is in a reasonable range.
    EXPECT_GT(basis.num_pw(), 100u);
    EXPECT_LT(basis.num_pw(), 5000u);
}

TEST(PlaneWaveBasis, GammaVectorFirst) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 30.0);

    // G=0 should be the first G-vector (sorted by |G|^2)
    ASSERT_GT(basis.num_pw(), 0u);
    const auto& g0 = basis.gvec(0);
    EXPECT_EQ(g0.h, 0);
    EXPECT_EQ(g0.k, 0);
    EXPECT_EQ(g0.l, 0);
    EXPECT_DOUBLE_EQ(g0.norm2, 0.0);
}

TEST(PlaneWaveBasis, KineticEnergiesGamma) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 30.0);

    // At Gamma point (k=0), kinetic energies = |G|^2 (Rydberg units)
    kronos::Vec3 k_gamma = {0.0, 0.0, 0.0};
    auto ekin = basis.kinetic_energies(k_gamma);

    ASSERT_EQ(ekin.size(), basis.num_pw());
    for (size_t i = 0; i < basis.num_pw(); ++i) {
        EXPECT_NEAR(ekin[i], basis.gvec(i).norm2, 1e-12)
            << "Mismatch at G-vector index " << i;
    }
}

TEST(PlaneWaveBasis, AllGVectorsBelowCutoff) {
    auto crystal = make_si_diamond();
    double ecutwfc = 20.0;
    kronos::PlaneWaveBasis basis(crystal, ecutwfc);

    for (size_t i = 0; i < basis.num_pw(); ++i) {
        double ekin = basis.gvec(i).norm2 / 2.0;
        EXPECT_LE(ekin, ecutwfc + 1e-10)
            << "G-vector " << i << " exceeds cutoff: "
            << ekin << " > " << ecutwfc;
    }
}

TEST(PlaneWaveBasis, SortedByNorm) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 25.0);

    for (size_t i = 1; i < basis.num_pw(); ++i) {
        EXPECT_GE(basis.gvec(i).norm2, basis.gvec(i - 1).norm2)
            << "G-vectors not sorted at index " << i;
    }
}

// ============================================================================
// FFTGrid tests (basic)
// ============================================================================

TEST(FFTGrid, DimensionsAreFFTFriendly) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 30.0);
    kronos::FFTGrid grid(basis);

    auto dims = grid.dims();
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(is_fft_friendly(dims[i]))
            << "Dimension " << i << " = " << dims[i]
            << " is not an FFT-friendly number";
    }

    // Grid should be at least 2*max_miller+1 in each direction
    auto mm = basis.max_miller();
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(dims[i], 2 * mm[i] + 1)
            << "Dimension " << i << " = " << dims[i]
            << " is smaller than minimum 2*" << mm[i] << "+1";
    }
}

// ============================================================================
// Additional PlaneWaveBasis tests
// ============================================================================

TEST(PlaneWaveBasis, GSphereSymmetry) {
    // If G is in the basis, then -G should also be present
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 20.0);

    for (size_t i = 0; i < basis.num_pw(); ++i) {
        const auto& g = basis.gvec(i);
        // Search for -G
        bool found_neg = false;
        for (size_t j = 0; j < basis.num_pw(); ++j) {
            const auto& gn = basis.gvec(j);
            if (gn.h == -g.h && gn.k == -g.k && gn.l == -g.l) {
                found_neg = true;
                break;
            }
        }
        EXPECT_TRUE(found_neg)
            << "G=(" << g.h << "," << g.k << "," << g.l
            << ") present but -G not found";
    }
}

TEST(PlaneWaveBasis, VeryLowCutoff) {
    auto crystal = make_si_diamond();
    // ecutwfc=10 Ry should still produce at least a few G-vectors
    kronos::PlaneWaveBasis basis(crystal, 10.0);
    EXPECT_GT(basis.num_pw(), 0u);
    // G=0 should always be there
    EXPECT_EQ(basis.gvec(0).h, 0);
    EXPECT_EQ(basis.gvec(0).k, 0);
    EXPECT_EQ(basis.gvec(0).l, 0);
}

TEST(PlaneWaveBasis, HighCutoffMorePW) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis_low(crystal, 15.0);
    kronos::PlaneWaveBasis basis_high(crystal, 30.0);

    // Higher cutoff should have more plane waves
    EXPECT_GT(basis_high.num_pw(), basis_low.num_pw());
}

TEST(PlaneWaveBasis, EcutwfcStored) {
    auto crystal = make_si_diamond();
    double ecut = 25.0;
    kronos::PlaneWaveBasis basis(crystal, ecut);
    EXPECT_DOUBLE_EQ(basis.ecutwfc(), ecut);
}

TEST(PlaneWaveBasis, MaxMillerIndices) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 20.0);
    auto mm = basis.max_miller();

    // Max miller indices should be positive
    for (int i = 0; i < 3; ++i) {
        EXPECT_GT(mm[i], 0) << "Max miller index " << i << " should be positive";
    }

    // All G-vectors should have |h| <= max_miller[0], etc.
    for (size_t i = 0; i < basis.num_pw(); ++i) {
        const auto& g = basis.gvec(i);
        EXPECT_LE(std::abs(g.h), mm[0]);
        EXPECT_LE(std::abs(g.k), mm[1]);
        EXPECT_LE(std::abs(g.l), mm[2]);
    }
}

TEST(PlaneWaveBasis, KPlusGKineticEnergy) {
    // At nonzero k, kinetic energies should be |k+G|²/2
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 20.0);

    kronos::Vec3 k_frac = {0.25, 0.0, 0.0};
    auto ekin = basis.kinetic_energies(k_frac);

    // All kinetic energies should be non-negative
    for (size_t i = 0; i < ekin.size(); ++i) {
        EXPECT_GE(ekin[i], -1e-12);
    }

    // At nonzero k, the lowest kinetic energy should be > 0
    // (since |k+G|² > 0 for any G when k ≠ 0 in general)
    // Actually, k+G could be zero if G = -k in reciprocal space
    // But for k = (0.25,0,0) and integer G, |k+G| > 0 always
    EXPECT_GT(ekin[0], 0.0);
}

TEST(PlaneWaveBasis, KineticEnergiesSizeMatchesPW) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 20.0);

    auto ekin_gamma = basis.kinetic_energies({0.0, 0.0, 0.0});
    auto ekin_k = basis.kinetic_energies({0.1, 0.2, 0.3});

    EXPECT_EQ(ekin_gamma.size(), basis.num_pw());
    EXPECT_EQ(ekin_k.size(), basis.num_pw());
}

TEST(PlaneWaveBasis, CubicCrystalBasis) {
    // Simple cubic should also work
    double a = 5.0;
    kronos::Mat3 lattice = {{{a, 0, 0}, {0, a, 0}, {0, 0, a}}};
    std::vector<kronos::Atom> atoms = {{"Si", 14, {0.0, 0.0, 0.0}}};
    kronos::Crystal crystal(lattice, atoms);

    kronos::PlaneWaveBasis basis(crystal, 20.0);
    EXPECT_GT(basis.num_pw(), 0u);
}

TEST(PlaneWaveBasis, GVectorNorm2Positive) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 20.0);

    // All G-vectors except G=0 should have positive norm2
    for (size_t i = 1; i < basis.num_pw(); ++i) {
        EXPECT_GT(basis.gvec(i).norm2, 0.0)
            << "G-vector " << i << " should have positive norm2";
    }
}

TEST(PlaneWaveBasis, GVectorCartesianConsistentWithNorm) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);

    for (size_t i = 0; i < std::min(basis.num_pw(), static_cast<size_t>(20)); ++i) {
        const auto& g = basis.gvec(i);
        double norm2_from_cart = g.cart[0]*g.cart[0] + g.cart[1]*g.cart[1] + g.cart[2]*g.cart[2];
        EXPECT_NEAR(norm2_from_cart, g.norm2, 1e-10)
            << "G-vector " << i << " norm2 inconsistent with Cartesian coords";
    }
}

// ============================================================================
// FFTGrid tests
// ============================================================================

TEST(FFTGrid, RoundTrip) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 20.0);
    kronos::FFTGrid grid(basis);

    int total = grid.total_points();

    // Create known real-space data: a smooth function
    std::vector<kronos::complex_t> r_space(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) {
        double phase = 2.0 * kronos::constants::pi * i / total;
        r_space[static_cast<size_t>(i)] = kronos::complex_t{
            std::cos(phase), std::sin(phase)};
    }

    // Forward FFT then inverse FFT
    std::vector<kronos::complex_t> g_space;
    std::vector<kronos::complex_t> r_recovered;
    grid.forward(r_space, g_space);
    grid.inverse(g_space, r_recovered);

    ASSERT_EQ(r_recovered.size(), r_space.size());
    for (int i = 0; i < total; ++i) {
        EXPECT_NEAR(r_recovered[static_cast<size_t>(i)].real(),
                    r_space[static_cast<size_t>(i)].real(), 1e-10)
            << "Real part mismatch at index " << i;
        EXPECT_NEAR(r_recovered[static_cast<size_t>(i)].imag(),
                    r_space[static_cast<size_t>(i)].imag(), 1e-10)
            << "Imaginary part mismatch at index " << i;
    }
}
