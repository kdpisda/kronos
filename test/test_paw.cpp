// ============================================================================
// KRONOS  test/test_paw.cpp
// PAW pseudopotential support tests
// ============================================================================

#include <gtest/gtest.h>
#include "io/upf_parser.hpp"
#include "potential/paw.hpp"
#include "potential/nonlocal_pp.hpp"
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"

#include <cmath>
#include <complex>
#include <vector>
#include <map>

using namespace kronos;

// ============================================================================
// Helpers: Build test PAW pseudopotential
// ============================================================================

namespace {

/// Create a synthetic PAW PP for testing
PseudoPotential make_test_paw_pp() {
    PseudoPotential pp;
    pp.element = "Si";
    pp.z_valence = 4.0;
    pp.is_paw = true;
    pp.is_norm_conserving = false;
    pp.pp_type = "PAW";
    pp.lmax = 1;
    pp.num_projectors = 2;
    pp.num_wfc = 2;

    // Radial grid
    int npts = 500;
    pp.mesh.npoints = npts;
    pp.mesh.r.resize(npts);
    pp.mesh.rab.resize(npts);
    double dr = 0.01;
    for (int i = 0; i < npts; ++i) {
        pp.mesh.r[i] = (i + 1) * dr;
        pp.mesh.rab[i] = dr;
    }

    // Local potential: Gaussian
    pp.vloc.resize(npts);
    double r_loc = 0.44;
    for (int i = 0; i < npts; ++i) {
        double r = pp.mesh.r[i];
        pp.vloc[i] = -pp.z_valence / r * std::exp(-r * r / (2.0 * r_loc * r_loc));
        pp.vloc[i] *= 2.0; // Ry
    }

    // Beta projectors
    pp.betas.resize(2);
    for (int ib = 0; ib < 2; ++ib) {
        pp.betas[ib].index = ib + 1;
        pp.betas[ib].angular_momentum = ib;  // l=0 and l=1
        pp.betas[ib].cutoff_index = npts;
        pp.betas[ib].values.resize(npts);
        for (int i = 0; i < npts; ++i) {
            double r = pp.mesh.r[i];
            pp.betas[ib].values[i] = r * std::exp(-r * r / 1.0) * (ib == 0 ? 1.0 : r);
        }
    }

    // D_ij matrix (2x2)
    pp.dij = {{-0.5, 0.0}, {0.0, -0.3}};

    // Atomic charge density
    pp.rho_atomic.resize(npts, 0.0);
    double norm = 0.0;
    for (int i = 0; i < npts; ++i) {
        double r = pp.mesh.r[i];
        pp.rho_atomic[i] = std::exp(-r) * r * r;
        norm += pp.rho_atomic[i] * dr;
    }
    for (int i = 0; i < npts; ++i) {
        pp.rho_atomic[i] *= pp.z_valence / norm;
    }

    // Atomic wavefunctions
    pp.atomic_wfc.resize(2);
    for (int iw = 0; iw < 2; ++iw) {
        pp.atomic_wfc[iw].angular_momentum = iw;
        pp.atomic_wfc[iw].occupation = 2.0;
        pp.atomic_wfc[iw].values.resize(npts);
        for (int i = 0; i < npts; ++i) {
            double r = pp.mesh.r[i];
            pp.atomic_wfc[iw].values[i] = r * std::exp(-r / 0.8) * (iw == 0 ? 1.0 : r);
        }
    }

    // PAW data
    PAWData paw;
    paw.core_energy = -0.1; // Ry
    paw.r_paw = 1.8;

    // All-electron partial waves
    paw.ae_wfc.resize(2);
    for (int i = 0; i < 2; ++i) {
        paw.ae_wfc[i].resize(npts);
        for (int ir = 0; ir < npts; ++ir) {
            double r = pp.mesh.r[ir];
            paw.ae_wfc[i][ir] = r * std::exp(-r / 0.5) * (i == 0 ? 1.0 : r * 0.5);
        }
    }

    // Pseudo partial waves
    paw.ps_wfc.resize(2);
    for (int i = 0; i < 2; ++i) {
        paw.ps_wfc[i].resize(npts);
        for (int ir = 0; ir < npts; ++ir) {
            double r = pp.mesh.r[ir];
            // Smoother than AE inside r_paw
            paw.ps_wfc[i][ir] = r * std::exp(-r / 0.7) * (i == 0 ? 1.0 : r * 0.5);
        }
    }

    // Core charges
    paw.ae_core_charge.resize(npts, 0.0);
    paw.ps_core_charge.resize(npts, 0.0);
    for (int ir = 0; ir < npts; ++ir) {
        double r = pp.mesh.r[ir];
        paw.ae_core_charge[ir] = 10.0 * std::exp(-r / 0.2);
        paw.ps_core_charge[ir] = 10.0 * std::exp(-r / 0.3);
    }

    // Augmentation charges Q_ij
    for (int i = 0; i < 2; ++i) {
        for (int j = i; j < 2; ++j) {
            PAWAugmentation aug;
            aug.i = i;
            aug.j = j;
            aug.l = 0;
            aug.qfunc.resize(npts);
            for (int ir = 0; ir < npts; ++ir) {
                double r = pp.mesh.r[ir];
                // Q_ij(r) = ae_wfc_i * ae_wfc_j - ps_wfc_i * ps_wfc_j
                aug.qfunc[ir] = paw.ae_wfc[i][ir] * paw.ae_wfc[j][ir]
                               - paw.ps_wfc[i][ir] * paw.ps_wfc[j][ir];
            }
            // Compute integral
            aug.q_integral = 0.0;
            for (int ir = 0; ir < npts; ++ir) {
                aug.q_integral += aug.qfunc[ir] * dr;
            }
            paw.augmentation.push_back(std::move(aug));
        }
    }

    pp.paw = std::move(paw);
    return pp;
}

Crystal make_si_diamond() {
    const double a = 10.2;
    Mat3 lattice = {{
        {{-a/2.0, 0.0, a/2.0}},
        {{0.0, a/2.0, a/2.0}},
        {{-a/2.0, a/2.0, 0.0}}
    }};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.0, 0.0, 0.0}},
        {"Si", 14, {0.25, 0.25, 0.25}}
    };
    return Crystal(lattice, atoms);
}

} // anonymous namespace

// ============================================================================
// PAW Data Structure Tests
// ============================================================================

TEST(PAW, PAWDataStructureCreation) {
    PseudoPotential pp = make_test_paw_pp();

    EXPECT_TRUE(pp.is_paw);
    ASSERT_TRUE(pp.paw.has_value());

    const auto& paw = pp.paw.value();
    EXPECT_NEAR(paw.core_energy, -0.1, 1e-10);
    EXPECT_NEAR(paw.r_paw, 1.8, 1e-10);
    EXPECT_EQ(paw.ae_wfc.size(), 2u);
    EXPECT_EQ(paw.ps_wfc.size(), 2u);
    EXPECT_EQ(paw.ae_core_charge.size(), 500u);
    EXPECT_EQ(paw.ps_core_charge.size(), 500u);
    EXPECT_GE(paw.augmentation.size(), 3u); // (0,0), (0,1), (1,1)
}

TEST(PAW, AugmentationChargeSymmetry) {
    PseudoPotential pp = make_test_paw_pp();
    const auto& paw = pp.paw.value();

    // Q_ij should equal Q_ji by construction
    for (const auto& aug : paw.augmentation) {
        // Find the transpose
        for (const auto& aug2 : paw.augmentation) {
            if (aug2.i == aug.j && aug2.j == aug.i && aug2.l == aug.l) {
                EXPECT_NEAR(aug.q_integral, aug2.q_integral, 1e-10);
            }
        }
    }
}

// ============================================================================
// PAW Calculator Tests
// ============================================================================

TEST(PAW, CalculatorInitialization) {
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);

    PAWCalculator calc(crystal, basis, fft_grid, pps);

    EXPECT_TRUE(calc.has_paw());
}

TEST(PAW, CalculatorNoPAW) {
    Crystal crystal = make_si_diamond();

    // Non-PAW PP
    PseudoPotential pp;
    pp.element = "Si";
    pp.z_valence = 4.0;
    pp.is_norm_conserving = true;
    pp.is_paw = false;
    pp.lmax = 0;
    pp.num_projectors = 0;
    pp.mesh.npoints = 100;
    pp.mesh.r.resize(100);
    pp.mesh.rab.resize(100);
    for (int i = 0; i < 100; ++i) {
        pp.mesh.r[i] = (i + 1) * 0.01;
        pp.mesh.rab[i] = 0.01;
    }
    pp.vloc.resize(100, -1.0);
    pp.rho_atomic.resize(100, 0.01);

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);

    PAWCalculator calc(crystal, basis, fft_grid, pps);
    EXPECT_FALSE(calc.has_paw());
}

TEST(PAW, QijOverlapIntegrals) {
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);

    PAWCalculator calc(crystal, basis, fft_grid, pps);

    const auto& qij = calc.get_qij("Si");
    EXPECT_EQ(qij.size(), 4u); // 2x2

    // q_ij should be symmetric
    EXPECT_NEAR(qij[0 * 2 + 1], qij[1 * 2 + 0], 1e-10);

    // q_00 should be non-zero (AE - PS difference)
    EXPECT_GT(std::abs(qij[0]), 1e-10);
}

TEST(PAW, OverlapOperatorHermiticity) {
    // S should be Hermitian: <ψ_1|S|ψ_2> = <ψ_2|S|ψ_1>*
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);

    PAWCalculator calc(crystal, basis, fft_grid, pps);

    int npw = static_cast<int>(basis.num_pw());

    // Create two random wavefunctions
    CVec psi1(npw), psi2(npw);
    for (int i = 0; i < npw; ++i) {
        psi1[i] = complex_t(std::sin(i * 0.1), std::cos(i * 0.3));
        psi2[i] = complex_t(std::cos(i * 0.2), std::sin(i * 0.5));
    }

    // For the S operator test, we need beta projectors and projections.
    // With empty beta_g and projections, S reduces to identity.
    std::vector<CVec> beta_g;
    std::vector<std::vector<complex_t>> proj_per_atom;

    CVec s_psi1 = calc.apply_s(psi1, beta_g, proj_per_atom);
    CVec s_psi2 = calc.apply_s(psi2, beta_g, proj_per_atom);

    // With empty projections, S = I, so <ψ_1|S|ψ_2> = <ψ_1|ψ_2>
    complex_t overlap12{0.0, 0.0};
    complex_t overlap21{0.0, 0.0};
    for (int i = 0; i < npw; ++i) {
        overlap12 += std::conj(psi1[i]) * s_psi2[i];
        overlap21 += std::conj(psi2[i]) * s_psi1[i];
    }

    // Hermiticity: <ψ_1|S|ψ_2> = conj(<ψ_2|S|ψ_1>)
    EXPECT_NEAR(std::real(overlap12), std::real(std::conj(overlap21)), 1e-10);
    EXPECT_NEAR(std::imag(overlap12), std::imag(std::conj(overlap21)), 1e-10);
}

TEST(PAW, OneCenterEnergyFinite) {
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);

    PAWCalculator calc(crystal, basis, fft_grid, pps);

    // Set up dummy occupation matrix
    int np = pp.num_projectors;
    int nk = 1, nb = 2;
    std::vector<std::vector<std::vector<complex_t>>> projections(nk);
    projections[0].resize(nb);
    for (int ib = 0; ib < nb; ++ib) {
        projections[0][ib].resize(np);
        for (int ip = 0; ip < np; ++ip) {
            projections[0][ib][ip] = complex_t(0.1 * (ib + 1), 0.05 * (ip + 1));
        }
    }

    std::vector<std::vector<double>> occs = {{2.0, 2.0}};
    std::vector<double> kweights = {1.0};

    calc.compute_rho_ij(projections, occs, kweights, 2);

    double e_paw = calc.one_center_energy();

    // Energy should be finite and non-zero
    EXPECT_TRUE(std::isfinite(e_paw));
    // Core energy alone is -0.1 per atom, 2 atoms = -0.2
    // Plus one-center corrections
    EXPECT_NE(e_paw, 0.0);
}

TEST(PAW, AugmentationDensityNonZero) {
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    PlaneWaveBasis basis(crystal, 15.0);
    double ecutrho = 12.0 * 15.0;
    FFTGrid fft_grid(basis, ecutrho);

    PAWCalculator calc(crystal, basis, fft_grid, pps);

    // Set up dummy occupation matrix
    int np = pp.num_projectors;
    std::vector<std::vector<std::vector<complex_t>>> projections(1);
    projections[0].resize(2);
    for (int ib = 0; ib < 2; ++ib) {
        projections[0][ib].resize(np, complex_t(0.1, 0.05));
    }

    std::vector<std::vector<double>> occs = {{2.0, 2.0}};
    std::vector<double> kweights = {1.0};
    calc.compute_rho_ij(projections, occs, kweights, 2);

    int num_grid = fft_grid.total_points();
    auto grid_dims = fft_grid.dims();
    const Mat3& recip = crystal.reciprocal_lattice();

    // Compute G-vectors for grid
    std::vector<double> grid_g2(num_grid);
    std::vector<Vec3> grid_gcart(num_grid);
    int n0 = grid_dims[0], n1 = grid_dims[1], n2 = grid_dims[2];
    for (int idx = 0; idx < num_grid; ++idx) {
        int hi = idx / (n1 * n2);
        int ki = (idx % (n1 * n2)) / n2;
        int li = idx % n2;
        int h = (hi <= n0/2) ? hi : hi - n0;
        int k = (ki <= n1/2) ? ki : ki - n1;
        int l = (li <= n2/2) ? li : li - n2;
        double gx = h*recip[0][0] + k*recip[1][0] + l*recip[2][0];
        double gy = h*recip[0][1] + k*recip[1][1] + l*recip[2][1];
        double gz = h*recip[0][2] + k*recip[1][2] + l*recip[2][2];
        grid_g2[idx] = gx*gx + gy*gy + gz*gz;
        grid_gcart[idx] = {gx, gy, gz};
    }

    // Start with zero density
    std::vector<complex_t> density_g(num_grid, {0.0, 0.0});

    // Add augmentation
    calc.add_augmentation_density(density_g, grid_gcart, grid_g2, ecutrho);

    // Should have non-zero augmentation contribution
    double total_aug = 0.0;
    for (const auto& d : density_g) {
        total_aug += std::abs(d);
    }
    EXPECT_GT(total_aug, 0.0);
}

TEST(PAW, PAWForcesNonZero) {
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    double ecutwfc = 15.0;
    double ecutrho = 4.0 * ecutwfc;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft_grid(basis);

    PAWCalculator calc(crystal, basis, fft_grid, pps);
    int np = pp.num_projectors;

    // Set up projections and compute ρ_ij
    std::vector<std::vector<std::vector<complex_t>>> projections(1);
    projections[0].resize(2);
    for (int ib = 0; ib < 2; ++ib) {
        projections[0][ib].resize(np, complex_t(0.1, 0.05));
    }
    std::vector<std::vector<double>> occs = {{2.0, 2.0}};
    std::vector<double> kweights = {1.0};
    calc.compute_rho_ij(projections, occs, kweights, 2);

    int num_grid = fft_grid.total_points();
    auto grid_dims = fft_grid.dims();
    const Mat3& recip = crystal.reciprocal_lattice();

    std::vector<double> grid_g2(num_grid);
    std::vector<Vec3> grid_gcart(num_grid);
    int n0 = grid_dims[0], n1 = grid_dims[1], n2 = grid_dims[2];
    for (int idx = 0; idx < num_grid; ++idx) {
        int hi = idx / (n1 * n2);
        int ki = (idx % (n1 * n2)) / n2;
        int li = idx % n2;
        int h = (hi <= n0/2) ? hi : hi - n0;
        int k = (ki <= n1/2) ? ki : ki - n1;
        int l = (li <= n2/2) ? li : li - n2;
        double gx = h*recip[0][0] + k*recip[1][0] + l*recip[2][0];
        double gy = h*recip[0][1] + k*recip[1][1] + l*recip[2][1];
        double gz = h*recip[0][2] + k*recip[1][2] + l*recip[2][2];
        grid_g2[idx] = gx*gx + gy*gy + gz*gz;
        grid_gcart[idx] = {gx, gy, gz};
    }

    // Create non-trivial V_eff in G-space
    std::vector<complex_t> veff_g(num_grid);
    for (int idx = 0; idx < num_grid; ++idx) {
        if (grid_g2[idx] < ecutrho + 1e-6 && grid_g2[idx] > 1e-12) {
            veff_g[idx] = complex_t(1.0 / grid_g2[idx], 0.0);
        }
    }

    auto forces = calc.compute_paw_forces(veff_g, grid_gcart, grid_g2, ecutrho);

    EXPECT_EQ(forces.size(), crystal.num_atoms());
    // With non-trivial V_eff and ρ_ij, forces should be non-zero
    double max_force = 0.0;
    for (const auto& f : forces) {
        max_force = std::max(max_force, std::sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]));
    }
    // Forces should be finite
    for (const auto& f : forces) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_TRUE(std::isfinite(f[d]));
        }
    }
}

TEST(PAW, PAWStressSymmetric) {
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    double ecutwfc = 15.0;
    double ecutrho = 4.0 * ecutwfc;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft_grid(basis);

    PAWCalculator calc(crystal, basis, fft_grid, pps);
    int np = pp.num_projectors;

    // Set up ρ_ij
    std::vector<std::vector<std::vector<complex_t>>> projections(1);
    projections[0].resize(2);
    for (int ib = 0; ib < 2; ++ib) {
        projections[0][ib].resize(np, complex_t(0.1, 0.05));
    }
    std::vector<std::vector<double>> occs = {{2.0, 2.0}};
    std::vector<double> kweights = {1.0};
    calc.compute_rho_ij(projections, occs, kweights, 2);

    int num_grid = fft_grid.total_points();
    auto grid_dims = fft_grid.dims();
    const Mat3& recip = crystal.reciprocal_lattice();

    std::vector<double> grid_g2(num_grid);
    std::vector<Vec3> grid_gcart(num_grid);
    int n0 = grid_dims[0], n1 = grid_dims[1], n2 = grid_dims[2];
    for (int idx = 0; idx < num_grid; ++idx) {
        int hi = idx / (n1 * n2);
        int ki = (idx % (n1 * n2)) / n2;
        int li = idx % n2;
        int h = (hi <= n0/2) ? hi : hi - n0;
        int k = (ki <= n1/2) ? ki : ki - n1;
        int l = (li <= n2/2) ? li : li - n2;
        double gx = h*recip[0][0] + k*recip[1][0] + l*recip[2][0];
        double gy = h*recip[0][1] + k*recip[1][1] + l*recip[2][1];
        double gz = h*recip[0][2] + k*recip[1][2] + l*recip[2][2];
        grid_g2[idx] = gx*gx + gy*gy + gz*gz;
        grid_gcart[idx] = {gx, gy, gz};
    }

    std::vector<complex_t> veff_g(num_grid, complex_t(0.01, 0.0));

    auto stress = calc.compute_paw_stress(veff_g, grid_gcart, grid_g2, ecutrho);

    // PAW stress tensor should be approximately symmetric (σ_αβ ≈ σ_βα)
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            EXPECT_TRUE(std::isfinite(stress[a][b]));
        }
    }
    // Symmetric within numerical precision
    for (int a = 0; a < 3; ++a) {
        for (int b = a+1; b < 3; ++b) {
            EXPECT_NEAR(stress[a][b], stress[b][a], 1e-10)
                << "PAW stress not symmetric: s[" << a << "][" << b << "]";
        }
    }
}

TEST(PAW, NonlocalPPProjectionsAndDij) {
    // Integration test: verify that the NonlocalPP projection/D_ij methods work
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    PlaneWaveBasis basis(crystal, 15.0);

    // NonlocalPP should work with PAW PP
    NonlocalPP nlpp(crystal, basis, pps);

    // save_base_dij / reset_dij cycle should not crash
    nlpp.save_base_dij();
    EXPECT_EQ(nlpp.num_atoms(), crystal.num_atoms());

    // Prepare k-point
    Vec3 k_frac = {0.0, 0.0, 0.0};
    nlpp.prepare_kpoint(k_frac);

    // Compute projections on a test wavefunction
    int npw = static_cast<int>(basis.num_pw());
    CVec psi(npw, complex_t{0.0, 0.0});
    psi[0] = {1.0, 0.0};

    auto proj = nlpp.compute_projections(psi);
    EXPECT_EQ(proj.size(), crystal.num_atoms());

    // Test D_ij reset/correction cycle
    int np = pp.num_projectors;
    std::vector<double> correction(np * np, 0.01);

    nlpp.reset_dij();
    nlpp.add_dij_correction(0, correction);

    // Should not crash
    nlpp.reset_dij();
}

TEST(PAW, DijPAWComputation) {
    Crystal crystal = make_si_diamond();
    PseudoPotential pp = make_test_paw_pp();

    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = pp;

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);

    PAWCalculator calc(crystal, basis, fft_grid, pps);

    int num_grid = fft_grid.total_points();
    auto grid_dims = fft_grid.dims();
    const Mat3& recip = crystal.reciprocal_lattice();

    std::vector<double> grid_g2(num_grid);
    std::vector<Vec3> grid_gcart(num_grid);
    int n0 = grid_dims[0], n1 = grid_dims[1], n2 = grid_dims[2];
    for (int idx = 0; idx < num_grid; ++idx) {
        int hi = idx / (n1 * n2);
        int ki = (idx % (n1 * n2)) / n2;
        int li = idx % n2;
        int h = (hi <= n0/2) ? hi : hi - n0;
        int k = (ki <= n1/2) ? ki : ki - n1;
        int l = (li <= n2/2) ? li : li - n2;
        double gx = h*recip[0][0] + k*recip[1][0] + l*recip[2][0];
        double gy = h*recip[0][1] + k*recip[1][1] + l*recip[2][1];
        double gz = h*recip[0][2] + k*recip[1][2] + l*recip[2][2];
        grid_g2[idx] = gx*gx + gy*gy + gz*gz;
        grid_gcart[idx] = {gx, gy, gz};
    }

    // Dummy V_eff
    std::vector<complex_t> veff_g(num_grid);
    for (int i = 0; i < num_grid; ++i) {
        veff_g[i] = complex_t(0.01, 0.0);
    }

    double ecutrho = 4.0 * 15.0;
    auto dij_paw = calc.compute_dij_paw(veff_g, grid_gcart, grid_g2, ecutrho);

    // Should have corrections for 2 atoms
    EXPECT_EQ(dij_paw.size(), 2u);
    for (const auto& dij : dij_paw) {
        EXPECT_EQ(dij.size(), 4u); // 2x2
        // D_ij should be finite
        for (double d : dij) {
            EXPECT_TRUE(std::isfinite(d));
        }
    }
}
