// ============================================================================
// KRONOS  test/test_stress.cpp
// Tests for the stress tensor implementation.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "potential/stress.hpp"
#include "potential/ewald.hpp"
#include "solver/scf.hpp"

#include <cmath>

using namespace kronos;

// ============================================================================
// Ewald stress tests
// ============================================================================

TEST(Stress, EwaldStressCubicSymmetry) {
    // For a cubic crystal, the Ewald stress tensor should be isotropic:
    // sigma_xx = sigma_yy = sigma_zz, and off-diagonal elements ~ 0.
    auto crystal = test::make_si_diamond_crystal();
    auto pp_map = test::make_si_pp_map();

    Mat3 ewald_stress = StressCalculator::compute_ewald_stress(crystal, pp_map);

    // Check isotropy: diagonal elements should be equal
    EXPECT_NEAR(ewald_stress[0][0], ewald_stress[1][1], 1e-8)
        << "Ewald stress xx should equal yy for cubic crystal";
    EXPECT_NEAR(ewald_stress[0][0], ewald_stress[2][2], 1e-8)
        << "Ewald stress xx should equal zz for cubic crystal";

    // Check off-diagonal elements are zero
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            if (a != b) {
                EXPECT_NEAR(ewald_stress[a][b], 0.0, 1e-8)
                    << "Off-diagonal Ewald stress [" << a << "][" << b
                    << "] should be zero for cubic crystal";
            }
        }
    }
}

TEST(Stress, EwaldStressSymmetric) {
    // Stress tensor should be symmetric: sigma_ab = sigma_ba
    auto crystal = test::make_si_diamond_crystal();
    auto pp_map = test::make_si_pp_map();

    Mat3 ewald_stress = StressCalculator::compute_ewald_stress(crystal, pp_map);

    for (int a = 0; a < 3; ++a) {
        for (int b = a + 1; b < 3; ++b) {
            EXPECT_NEAR(ewald_stress[a][b], ewald_stress[b][a], 1e-10)
                << "Ewald stress should be symmetric: [" << a << "][" << b << "]";
        }
    }
}

TEST(Stress, EwaldStressNonzero) {
    // The Ewald stress should be nonzero for a crystal
    auto crystal = test::make_si_diamond_crystal();
    auto pp_map = test::make_si_pp_map();

    Mat3 ewald_stress = StressCalculator::compute_ewald_stress(crystal, pp_map);

    // At least one component should be nonzero
    double max_val = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            max_val = std::max(max_val, std::abs(ewald_stress[a][b]));

    EXPECT_GT(max_val, 1e-6) << "Ewald stress should be nonzero";
}

// ============================================================================
// Hartree stress tests
// ============================================================================

TEST(Stress, HartreeStressSymmetric) {
    // The Hartree stress tensor should be symmetric
    auto crystal = test::make_si_diamond_crystal();
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft_grid(basis, 40.0);

    int num_grid = fft_grid.total_points();
    double volume = crystal.volume();
    auto grid_dims = fft_grid.dims();
    const Mat3 recip_lat = crystal.reciprocal_lattice();

    // Build G-vector data
    std::vector<double> grid_g2(num_grid);
    std::vector<Vec3> grid_gcart(num_grid);
    for (int idx = 0; idx < num_grid; ++idx) {
        int hi = idx / (grid_dims[1] * grid_dims[2]);
        int ki = (idx % (grid_dims[1] * grid_dims[2])) / grid_dims[2];
        int li = idx % grid_dims[2];
        int h = (hi <= grid_dims[0]/2) ? hi : hi - grid_dims[0];
        int k = (ki <= grid_dims[1]/2) ? ki : ki - grid_dims[1];
        int l = (li <= grid_dims[2]/2) ? li : li - grid_dims[2];
        double gx = h*recip_lat[0][0] + k*recip_lat[1][0] + l*recip_lat[2][0];
        double gy = h*recip_lat[0][1] + k*recip_lat[1][1] + l*recip_lat[2][1];
        double gz = h*recip_lat[0][2] + k*recip_lat[1][2] + l*recip_lat[2][2];
        grid_g2[idx] = gx*gx + gy*gy + gz*gz;
        grid_gcart[idx] = {gx, gy, gz};
    }

    // Create a mock density: Gaussian in G-space
    std::vector<complex_t> density_g(num_grid, {0.0, 0.0});
    double sigma2 = 2.0;
    for (int ig = 0; ig < num_grid; ++ig) {
        density_g[ig] = complex_t{std::exp(-grid_g2[ig] * sigma2), 0.0};
    }

    Mat3 hartree_stress = StressCalculator::compute_hartree_stress(
        density_g, grid_gcart, grid_g2, 40.0, volume, num_grid);

    // Check symmetry
    for (int a = 0; a < 3; ++a) {
        for (int b = a + 1; b < 3; ++b) {
            EXPECT_NEAR(hartree_stress[a][b], hartree_stress[b][a], 1e-12)
                << "Hartree stress should be symmetric: [" << a << "][" << b << "]";
        }
    }
}

// ============================================================================
// XC stress tests
// ============================================================================

TEST(Stress, XCStressIsotropic) {
    // For LDA, the XC stress should be purely diagonal (isotropic)
    int num_grid = 1000;
    double volume = 100.0;

    RVec density_r(num_grid, 0.01);  // uniform density
    RVec vxc_r(num_grid, -0.5);      // constant potential
    double exc_energy = -0.3;

    Mat3 xc_stress = StressCalculator::compute_xc_stress(
        exc_energy, vxc_r, density_r, volume, num_grid);

    // Check that off-diagonal elements are zero
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            if (a != b) {
                EXPECT_NEAR(xc_stress[a][b], 0.0, 1e-15)
                    << "LDA XC stress should have zero off-diagonal [" << a << "][" << b << "]";
            }
        }
    }

    // Check that diagonal elements are equal
    EXPECT_NEAR(xc_stress[0][0], xc_stress[1][1], 1e-15);
    EXPECT_NEAR(xc_stress[0][0], xc_stress[2][2], 1e-15);

    // Check the value: sigma_aa = -(E_xc - int v_xc * n dr) / Omega
    double int_vxc_n = -0.5 * 0.01 * volume;  // = -0.05
    double expected = -(exc_energy - int_vxc_n) / volume;
    EXPECT_NEAR(xc_stress[0][0], expected, 1e-12);
}

// ============================================================================
// Pressure tests
// ============================================================================

TEST(Stress, PressureFromIsotropicStress) {
    // P = -trace(sigma)/3 * conversion
    Mat3 stress = {{{-0.001, 0, 0}, {0, -0.001, 0}, {0, 0, -0.001}}};
    double p = StressCalculator::pressure_gpa(stress);

    double expected = -(-0.003) / 3.0 * StressCalculator::RY_BOHR3_TO_GPA;
    EXPECT_NEAR(p, expected, 1e-3);
    EXPECT_GT(p, 0.0) << "Negative diagonal stress (compression) should give positive pressure";
}

TEST(Stress, PressureZeroForTracelessStress) {
    // Off-diagonal or traceless stress should give zero pressure
    Mat3 stress = {{{0.001, 0.0005, 0}, {0.0005, -0.0005, 0}, {0, 0, -0.0005}}};
    double p = StressCalculator::pressure_gpa(stress);
    EXPECT_NEAR(p, 0.0, 1e-6) << "Traceless stress should give zero pressure";
}

// ============================================================================
// SCF stress integration test
// ============================================================================

TEST(Stress, SCFStressCubicSymmetry) {
    // Si diamond: cubic symmetry => diagonal stress, all equal
    Crystal crystal = test::make_si_diamond_crystal();
    auto pp_map = test::make_si_pp_map();

    CalculationParams calc;
    calc.ecutwfc = 12.0;
    calc.kpoints.grid = {1, 1, 1};
    calc.kpoints.shift = {0, 0, 0};
    calc.xc_functional = "LDA_PZ";

    ConvergenceParams conv;
    conv.energy_threshold = 1e-4;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pp_map);
    SCFResult result = solver.solve();

    ASSERT_TRUE(result.converged) << "SCF should converge for Si Gamma-only";

    // Check cubic isotropy: sigma_xx = sigma_yy = sigma_zz
    // Use loose tolerance since stress involves derivatives
    double diag_avg = (result.stress[0][0] + result.stress[1][1] + result.stress[2][2]) / 3.0;
    if (std::abs(diag_avg) > 1e-10) {
        double rel_tol = 0.05;  // 5% relative tolerance
        EXPECT_NEAR(result.stress[0][0], diag_avg, std::abs(diag_avg) * rel_tol)
            << "Stress xx should match diagonal average for cubic crystal";
        EXPECT_NEAR(result.stress[1][1], diag_avg, std::abs(diag_avg) * rel_tol)
            << "Stress yy should match diagonal average for cubic crystal";
        EXPECT_NEAR(result.stress[2][2], diag_avg, std::abs(diag_avg) * rel_tol)
            << "Stress zz should match diagonal average for cubic crystal";
    }

    // Off-diagonal should be small
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            if (a != b) {
                EXPECT_NEAR(result.stress[a][b], 0.0,
                            std::max(1e-6, std::abs(diag_avg) * 0.05))
                    << "Off-diagonal stress [" << a << "][" << b
                    << "] should be small for cubic crystal";
            }
        }
    }

    // Pressure should be nonzero (cell is not at equilibrium volume with toy PP)
    // Just check it is finite and reasonable
    EXPECT_TRUE(std::isfinite(result.pressure_gpa))
        << "Pressure should be finite";
}

TEST(Stress, EwaldStressPressureMonotonic) {
    // The Ewald pressure should change monotonically with volume.
    // For a neutral cell of positive ions, the Ewald energy becomes more
    // negative when compressed (higher density packing). The Ewald contribution
    // alone gives negative pressure (wants to expand, because Madelung energy
    // penalizes close ion-ion approach). Compression makes the pressure more
    // negative (stronger repulsion), expansion makes it less negative.
    auto pp_map = test::make_si_pp_map();

    auto ewald_pressure_at_scale = [&](double scale) {
        double a = 5.43 * scale;
        Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
        std::vector<Atom> atoms = {
            {"Si", 14, {0.00, 0.00, 0.00}},
            {"Si", 14, {0.25, 0.25, 0.25}},
        };
        Crystal crystal(lattice, std::move(atoms));
        Mat3 s = StressCalculator::compute_ewald_stress(crystal, pp_map);
        return StressCalculator::pressure_gpa(s);
    };

    double p_small = ewald_pressure_at_scale(0.95);
    double p_equil = ewald_pressure_at_scale(1.00);
    double p_large = ewald_pressure_at_scale(1.05);

    // Ewald pressure is negative and more negative when compressed
    EXPECT_LT(p_small, p_equil)
        << "Compressed cell should have more negative Ewald pressure";
    EXPECT_LT(p_equil, p_large)
        << "Expanded cell should have less negative Ewald pressure";
}

TEST(Stress, EwaldStressNumericalDerivative) {
    // Verify Ewald stress via numerical derivative of Ewald energy w.r.t. volume:
    //   P_ew = -dE_ew/dV
    auto pp_map = test::make_si_pp_map();

    auto ewald_at_scale = [&](double scale) -> std::pair<double, double> {
        double a = 5.43 * scale;
        Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
        std::vector<Atom> atoms = {
            {"Si", 14, {0.00, 0.00, 0.00}},
            {"Si", 14, {0.25, 0.25, 0.25}},
        };
        Crystal crystal(lattice, std::move(atoms));
        auto result = EwaldCalculator::compute(crystal, pp_map);
        return {result.energy, crystal.volume()};
    };

    double ds = 0.001;
    auto [e0, v0] = ewald_at_scale(1.0);
    auto [ep, vp] = ewald_at_scale(1.0 + ds);
    auto [em, vm] = ewald_at_scale(1.0 - ds);

    double dEdV = (ep - em) / (vp - vm);
    double P_fd = -dEdV * StressCalculator::RY_BOHR3_TO_GPA;

    // Analytic Ewald stress
    double a = 5.43;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    Crystal crystal(lattice, {{"Si", 14, {0,0,0}}, {"Si", 14, {0.25,0.25,0.25}}});
    Mat3 ewald_stress = StressCalculator::compute_ewald_stress(crystal, pp_map);
    double P_analytic = StressCalculator::pressure_gpa(ewald_stress);

    std::printf("\n  Ewald stress numerical derivative:\n");
    std::printf("    P (FD)       = %.4f GPa\n", P_fd);
    std::printf("    P (analytic) = %.4f GPa\n", P_analytic);

    // Should agree to within a few percent
    if (std::abs(P_fd) > 1.0) {
        EXPECT_NEAR(P_analytic, P_fd, std::abs(P_fd) * 0.05)
            << "Ewald stress should agree with numerical derivative to 5%";
    }
}

TEST(Stress, SCFStressNonzero) {
    // The total stress from a converged SCF should be nonzero and finite.
    // (The cell is not at equilibrium volume with the toy PP, so there is stress.)
    Crystal crystal = test::make_si_diamond_crystal();
    auto pp_map = test::make_si_pp_map();

    CalculationParams calc;
    calc.ecutwfc = 12.0;
    calc.kpoints.grid = {1, 1, 1};
    calc.kpoints.shift = {0, 0, 0};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-4;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pp_map);
    SCFResult result = solver.solve();
    ASSERT_TRUE(result.converged);

    // Total stress should be nonzero
    double max_stress = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            max_stress = std::max(max_stress, std::abs(result.stress[a][b]));

    EXPECT_GT(max_stress, 1e-6)
        << "Total stress should be nonzero for a toy PP at arbitrary lattice constant";

    // All components should be finite
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            EXPECT_TRUE(std::isfinite(result.stress[a][b]))
                << "Stress [" << a << "][" << b << "] should be finite";

    // Pressure should be finite
    EXPECT_TRUE(std::isfinite(result.pressure_gpa));

    // Print individual contributions
    std::printf("\n  Stress components (Ry/bohr^3):\n");
    std::printf("    Kinetic:   %.8f\n", result.stress_kinetic[0][0]);
    std::printf("    Hartree:   %.8f\n", result.stress_hartree[0][0]);
    std::printf("    XC:        %.8f\n", result.stress_xc[0][0]);
    std::printf("    Local PP:  %.8f\n", result.stress_local[0][0]);
    std::printf("    Nonlocal:  %.8f\n", result.stress_nonlocal[0][0]);
    std::printf("    Ewald:     %.8f\n", result.stress_ewald[0][0]);
    std::printf("    Total:     %.8f\n", result.stress[0][0]);
    std::printf("    Pressure:  %.4f GPa\n", result.pressure_gpa);
}

// ============================================================================
// Total stress combination test
// ============================================================================

TEST(Stress, TotalStressCombination) {
    Mat3 a = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
    Mat3 b = {{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}};
    Mat3 c = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    Mat3 d = {{{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}}};
    Mat3 e = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    Mat3 f = {{{-0.5, -2, -3}, {-4, -4.5, -6}, {-7, -8, -8.5}}};

    Mat3 total = StressCalculator::compute_total_stress(a, b, c, d, e, f);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = a[i][j] + b[i][j] + c[i][j] + d[i][j] + e[i][j] + f[i][j];
            EXPECT_NEAR(total[i][j], expected, 1e-15)
                << "Total stress [" << i << "][" << j << "] should be sum of components";
        }
    }
}
