#include <gtest/gtest.h>

#include "basis/fft_grid.hpp"
#include "basis/plane_wave.hpp"
#include "core/crystal.hpp"
#include "core/constants.hpp"
#include "core/types.hpp"

#include <cmath>
#include <numeric>

namespace {

// Helper: create Si diamond crystal (a = 5.43 angstrom)
kronos::Crystal make_si_diamond() {
    const double a = 5.43;  // angstrom
    kronos::Mat3 lattice = {{
        {{ 0.0,   a/2.0, a/2.0 }},
        {{ a/2.0, 0.0,   a/2.0 }},
        {{ a/2.0, a/2.0, 0.0   }}
    }};
    std::vector<kronos::Atom> atoms = {
        {"Si", 14, {0.0,  0.0,  0.0 }},
        {"Si", 14, {0.25, 0.25, 0.25}}
    };
    return kronos::Crystal(lattice, atoms);
}

} // anonymous namespace

// ============================================================================
// FFT scatter / gather tests
// ============================================================================

TEST(FFTGrid, ScatterGather) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);
    kronos::FFTGrid grid(basis);

    size_t npw = basis.num_pw();

    // Create known plane-wave coefficients
    kronos::CVec pw_original(npw);
    for (size_t i = 0; i < npw; ++i) {
        double phase = 2.0 * kronos::constants::pi * static_cast<double>(i)
                       / static_cast<double>(npw);
        pw_original[i] = kronos::complex_t{std::cos(phase), std::sin(phase)};
    }

    // Scatter to full FFT grid
    std::vector<kronos::complex_t> full_grid;
    grid.scatter_to_grid(basis, pw_original, full_grid);

    ASSERT_EQ(static_cast<int>(full_grid.size()), grid.total_points());

    // Gather back from grid
    kronos::CVec pw_recovered;
    grid.gather_from_grid(basis, full_grid, pw_recovered);

    ASSERT_EQ(pw_recovered.size(), npw);
    for (size_t i = 0; i < npw; ++i) {
        EXPECT_NEAR(pw_recovered[i].real(), pw_original[i].real(), 1e-14)
            << "Real part mismatch at PW index " << i;
        EXPECT_NEAR(pw_recovered[i].imag(), pw_original[i].imag(), 1e-14)
            << "Imaginary part mismatch at PW index " << i;
    }
}

TEST(FFTGrid, ForwardInverseConsistency) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);
    kronos::FFTGrid grid(basis);

    int total = grid.total_points();

    // Set known values: a delta function at the origin
    std::vector<kronos::complex_t> r_space(static_cast<size_t>(total),
                                           kronos::complex_t{0.0, 0.0});
    r_space[0] = kronos::complex_t{1.0, 0.0};  // delta at origin

    // Forward FFT: delta -> all components equal to 1.0
    std::vector<kronos::complex_t> g_space;
    grid.forward(r_space, g_space);

    ASSERT_EQ(static_cast<int>(g_space.size()), total);
    // In G-space, all components should be 1.0 (FFT of delta is constant)
    for (int i = 0; i < total; ++i) {
        EXPECT_NEAR(g_space[static_cast<size_t>(i)].real(), 1.0, 1e-10)
            << "G-space real part mismatch at index " << i;
        EXPECT_NEAR(g_space[static_cast<size_t>(i)].imag(), 0.0, 1e-10)
            << "G-space imaginary part mismatch at index " << i;
    }

    // Inverse FFT: should recover the original delta
    std::vector<kronos::complex_t> r_recovered;
    grid.inverse(g_space, r_recovered);

    ASSERT_EQ(static_cast<int>(r_recovered.size()), total);
    for (int i = 0; i < total; ++i) {
        double expected_real = (i == 0) ? 1.0 : 0.0;
        EXPECT_NEAR(r_recovered[static_cast<size_t>(i)].real(),
                    expected_real, 1e-10)
            << "Recovered real part mismatch at index " << i;
        EXPECT_NEAR(r_recovered[static_cast<size_t>(i)].imag(),
                    0.0, 1e-10)
            << "Recovered imaginary part mismatch at index " << i;
    }
}

TEST(FFTGrid, ScatterGatherPreservesZeros) {
    // Verify that non-basis G-vector positions remain zero after scatter
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 10.0);
    kronos::FFTGrid grid(basis);

    size_t npw = basis.num_pw();

    // Set all PW coefficients to 1.0
    kronos::CVec pw_coeffs(npw, kronos::complex_t{1.0, 0.0});

    std::vector<kronos::complex_t> full_grid;
    grid.scatter_to_grid(basis, pw_coeffs, full_grid);

    // Count non-zero entries in the grid
    int nonzero_count = 0;
    for (const auto& val : full_grid) {
        if (std::abs(val) > 1e-15) {
            ++nonzero_count;
        }
    }

    // The number of non-zero entries should equal the number of plane waves
    EXPECT_EQ(static_cast<size_t>(nonzero_count), npw);
}

TEST(FFTGrid, GvecToIndexConsistency) {
    // Verify that gvec_to_index maps (0,0,0) to index 0
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);
    kronos::FFTGrid grid(basis);

    EXPECT_EQ(grid.gvec_to_index(0, 0, 0), 0);

    // Verify wrapping: negative indices should map to valid range
    auto dims = grid.dims();
    // (-1, 0, 0) should wrap to (dims[0]-1, 0, 0)
    int expected = (dims[0] - 1) * dims[1] * dims[2];
    EXPECT_EQ(grid.gvec_to_index(-1, 0, 0), expected);
}

// ============================================================================
// Parseval's theorem: energy conservation under FFT
// ============================================================================

TEST(FFTGrid, ParsevalTheorem) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);
    kronos::FFTGrid grid(basis);

    int total = grid.total_points();

    // Create random real-space data
    std::vector<kronos::complex_t> r_space(total);
    for (int i = 0; i < total; ++i) {
        double val = std::sin(2.0 * kronos::constants::pi * i / total + 0.5);
        r_space[i] = kronos::complex_t{val, 0.0};
    }

    // Compute sum |f(r)|² in real space
    double energy_r = 0.0;
    for (int i = 0; i < total; ++i) {
        energy_r += std::norm(r_space[i]);
    }

    // Forward FFT
    std::vector<kronos::complex_t> g_space;
    grid.forward(r_space, g_space);

    // Compute sum |F(G)|² in G-space
    double energy_g = 0.0;
    for (int i = 0; i < total; ++i) {
        energy_g += std::norm(g_space[i]);
    }

    // Parseval: sum|f|² = (1/N) * sum|F|²
    EXPECT_NEAR(energy_r, energy_g / total, energy_r * 1e-10)
        << "Parseval's theorem: real-space energy = G-space energy / N";
}

// ============================================================================
// Known transform: cosine -> two peaks
// ============================================================================

TEST(FFTGrid, CosineTransformTwoPeaks) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 10.0);
    kronos::FFTGrid grid(basis);

    auto dims = grid.dims();
    int total = grid.total_points();

    // f(r) = cos(2*pi*i/n1) — depends only on first direction
    // FFT should give peaks at G = ±(1,0,0) * 2π/a
    std::vector<kronos::complex_t> r_space(total, kronos::complex_t{0.0, 0.0});
    for (int i = 0; i < dims[0]; ++i) {
        double phase = 2.0 * kronos::constants::pi * i / dims[0];
        double val = std::cos(phase);
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                int idx = (i * dims[1] + j) * dims[2] + k;
                r_space[idx] = kronos::complex_t{val, 0.0};
            }
        }
    }

    std::vector<kronos::complex_t> g_space;
    grid.forward(r_space, g_space);

    // G=0 component: mean of cos is 0
    EXPECT_NEAR(std::abs(g_space[0]), 0.0, 1.0)
        << "G=0 component of cosine should be near zero";

    // (1,0,0) component should be nonzero
    int idx_1 = grid.gvec_to_index(1, 0, 0);
    double mag_1 = std::abs(g_space[idx_1]);
    EXPECT_GT(mag_1, 0.1 * total / dims[0])
        << "G=(1,0,0) component should be large for cos(2pi*x/a)";

    // (-1,0,0) component should also be nonzero (cos = (e^ix + e^-ix)/2)
    int idx_m1 = grid.gvec_to_index(-1, 0, 0);
    double mag_m1 = std::abs(g_space[idx_m1]);
    EXPECT_GT(mag_m1, 0.1 * total / dims[0]);
}

// ============================================================================
// Grid dimensions are even
// ============================================================================

TEST(FFTGrid, GridDimsAreEven) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);
    kronos::FFTGrid grid(basis);

    auto dims = grid.dims();
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(dims[i] % 2, 0)
            << "FFT grid dimension " << i << " = " << dims[i] << " should be even";
    }
}

// ============================================================================
// Total points matches product
// ============================================================================

TEST(FFTGrid, TotalPointsMatchesProduct) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);
    kronos::FFTGrid grid(basis);

    auto dims = grid.dims();
    EXPECT_EQ(grid.total_points(), dims[0] * dims[1] * dims[2]);
}

// ============================================================================
// FFT of real data has conjugate symmetry
// ============================================================================

TEST(FFTGrid, RealDataConjugateSymmetry) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 10.0);
    kronos::FFTGrid grid(basis);

    int total = grid.total_points();
    auto dims = grid.dims();

    // Real-valued data
    std::vector<kronos::complex_t> r_space(total);
    for (int i = 0; i < total; ++i) {
        r_space[i] = kronos::complex_t{std::sin(0.1 * i + 0.3), 0.0};
    }

    std::vector<kronos::complex_t> g_space;
    grid.forward(r_space, g_space);

    // For real input, F(-G) = conj(F(G))
    // Check a few G-vectors
    for (int h = 0; h <= 2; ++h) {
        for (int k = 0; k <= 2; ++k) {
            for (int l = 0; l <= 2; ++l) {
                if (h == 0 && k == 0 && l == 0) continue;
                int idx_pos = grid.gvec_to_index(h, k, l);
                int idx_neg = grid.gvec_to_index(-h, -k, -l);
                if (idx_pos < total && idx_neg < total) {
                    EXPECT_NEAR(g_space[idx_pos].real(), g_space[idx_neg].real(), 1e-8);
                    EXPECT_NEAR(g_space[idx_pos].imag(), -g_space[idx_neg].imag(), 1e-8);
                }
            }
        }
    }
}

// ============================================================================
// Explicit density cutoff grid
// ============================================================================

TEST(FFTGrid, ExplicitEcutrhoGrid) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);

    kronos::FFTGrid grid_4x(basis);              // default: 4*ecutwfc
    kronos::FFTGrid grid_8x(basis, 120.0);        // 8*ecutwfc

    auto dims_4x = grid_4x.dims();
    auto dims_8x = grid_8x.dims();

    // 8x ecutwfc grid should be at least as large
    EXPECT_GE(dims_8x[0], dims_4x[0]);
    EXPECT_GE(dims_8x[1], dims_4x[1]);
    EXPECT_GE(dims_8x[2], dims_4x[2]);
}

TEST(FFTGrid, MoveConstruction) {
    auto crystal = make_si_diamond();
    kronos::PlaneWaveBasis basis(crystal, 15.0);

    kronos::FFTGrid grid1(basis);
    auto dims_orig = grid1.dims();
    int total_orig = grid1.total_points();

    // Move-construct a new grid
    kronos::FFTGrid grid2(std::move(grid1));
    EXPECT_EQ(grid2.dims(), dims_orig);
    EXPECT_EQ(grid2.total_points(), total_orig);

    // The moved-from grid should have zero dimensions
    EXPECT_EQ(grid1.total_points(), 0);

    // Verify the moved-to grid still works: forward/inverse round-trip
    int total = grid2.total_points();
    std::vector<kronos::complex_t> r_in(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) {
        r_in[static_cast<size_t>(i)] = kronos::complex_t{
            static_cast<double>(i), 0.0};
    }
    std::vector<kronos::complex_t> g_space, r_out;
    grid2.forward(r_in, g_space);
    grid2.inverse(g_space, r_out);

    for (int i = 0; i < total; ++i) {
        EXPECT_NEAR(r_out[static_cast<size_t>(i)].real(),
                    r_in[static_cast<size_t>(i)].real(), 1e-8);
        EXPECT_NEAR(r_out[static_cast<size_t>(i)].imag(),
                    r_in[static_cast<size_t>(i)].imag(), 1e-8);
    }
}
