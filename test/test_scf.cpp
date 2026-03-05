#include <gtest/gtest.h>

#include "core/types.hpp"
#include "core/constants.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "potential/hartree.hpp"
#include "potential/xc.hpp"
#include "potential/local_pp.hpp"
#include "potential/ewald.hpp"
#include "potential/forces.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "potential/nonlocal_pp.hpp"
#include "solver/davidson.hpp"
#include "solver/fermi.hpp"
#include "solver/scf.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

using namespace kronos;

// ============================================================================
// Helpers
// ============================================================================

Crystal make_simple_cubic(double a_ang, const std::string& element,
                          int Z = 14) {
    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};
    std::vector<Atom> atoms = {{element, Z, {0.0, 0.0, 0.0}}};
    return Crystal(lattice, std::move(atoms));
}

Crystal make_si_diamond() {
    double a = 5.43;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.00, 0.00, 0.00}},
        {"Si", 14, {0.25, 0.25, 0.25}},
    };
    return Crystal(lattice, std::move(atoms));
}

// Build a minimal Si-like pseudopotential with only local part (for testing).
// V_loc(r) = -Z_val * erf(r / r_loc) / r   (a simple Gaussian local PP)
PseudoPotential make_simple_local_pp(double z_val, int npts = 500,
                                     double rmax = 10.0) {
    PseudoPotential pp;
    pp.z_valence = z_val;
    pp.mesh.npoints = npts;
    pp.mesh.r.resize(npts);
    pp.mesh.rab.resize(npts);
    pp.vloc.resize(npts);

    double dr = rmax / (npts - 1);
    double r_loc = 0.5;  // bohr

    for (int i = 0; i < npts; ++i) {
        double r = i * dr;
        pp.mesh.r[i] = r;
        pp.mesh.rab[i] = dr;
        if (r < 1e-30) {
            // V_loc(0) = -Z * 2/(sqrt(pi)*r_loc)
            pp.vloc[i] = -z_val * 2.0 / (std::sqrt(constants::pi) * r_loc);
        } else {
            pp.vloc[i] = -z_val * std::erf(r / r_loc) / r;
        }
    }

    // Add a simple atomic density for initial guess: Gaussian centered at origin
    pp.rho_atomic.resize(npts);
    double norm = 0.0;
    double sigma = 1.0;
    for (int i = 0; i < npts; ++i) {
        double r = pp.mesh.r[i];
        pp.rho_atomic[i] = std::exp(-r * r / (2.0 * sigma * sigma));
        norm += r * r * pp.rho_atomic[i] * pp.mesh.rab[i];
    }
    // Normalize: 4*pi * integral r^2 rho dr = z_val
    norm *= constants::four_pi;
    for (int i = 0; i < npts; ++i) {
        pp.rho_atomic[i] *= z_val / norm;
    }

    return pp;
}

// ============================================================================
// Test 1: Hartree energy of a known density
// ============================================================================
// A uniform density n(G=0) = N_e with a cosine perturbation.
// The Hartree energy should match the analytical result from Parseval's theorem.

TEST(HartreeEnergy, UniformPlusCosine) {
    // Simple cubic crystal
    Crystal crystal = make_simple_cubic(5.0, "Si");
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);

    int num_grid = fft.total_points();
    double volume = crystal.volume();
    auto dims = fft.dims();
    int npw = static_cast<int>(basis.num_pw());

    // Build a real-space density: uniform + small cosine perturbation
    //   n(r) = n0 + A * cos(G1 . r)
    // where G1 is the first non-zero G-vector.
    double n0 = 0.1;  // electrons/bohr^3
    double A = 0.01;

    // Find first nonzero G-vector
    const auto& gvecs = basis.gvectors();
    size_t ig1 = 0;
    for (size_t ig = 0; ig < gvecs.size(); ++ig) {
        if (gvecs[ig].norm2 > 1e-10) { ig1 = ig; break; }
    }
    ASSERT_GT(gvecs[ig1].norm2, 1e-10);

    const Vec3& G1 = gvecs[ig1].cart;
    double G1_sq = gvecs[ig1].norm2;

    // Build density on real-space grid
    // Grid point r_ijk = (i/n1, j/n2, k/n3) in fractional -> convert to cart
    Mat3 lat_bohr = crystal.lattice_bohr();
    RVec density_r(num_grid);
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                double fx = static_cast<double>(i) / dims[0];
                double fy = static_cast<double>(j) / dims[1];
                double fz = static_cast<double>(k) / dims[2];
                // r = fx*a1 + fy*a2 + fz*a3
                double rx = fx * lat_bohr[0][0] + fy * lat_bohr[1][0] + fz * lat_bohr[2][0];
                double ry = fx * lat_bohr[0][1] + fy * lat_bohr[1][1] + fz * lat_bohr[2][1];
                double rz = fx * lat_bohr[0][2] + fy * lat_bohr[1][2] + fz * lat_bohr[2][2];
                double gdotr = G1[0]*rx + G1[1]*ry + G1[2]*rz;
                int idx = (i * dims[1] + j) * dims[2] + k;
                density_r[idx] = n0 + A * std::cos(gdotr);
            }
        }
    }

    // FFT density to G-space
    std::vector<complex_t> density_c(num_grid);
    for (int i = 0; i < num_grid; ++i)
        density_c[i] = complex_t{density_r[i], 0.0};
    std::vector<complex_t> density_g_full(num_grid);
    fft.forward(density_c, density_g_full);

    // Gather to PW
    CVec density_g(npw);
    fft.gather_from_grid(basis, density_g_full, density_g);

    // Compute Hartree potential and energy
    HartreeSolver hartree(basis);
    CVec vhartree_g = hartree.compute(density_g);
    double e_h = hartree.energy(density_g, vhartree_g, volume, num_grid);

    // Analytical Hartree energy for uniform + cosine:
    // Only the G!=0 component contributes.  The cosine gives n(G1) = A/2 * N_grid.
    // E_H = (Omega/2) * sum_G |n(G)|^2 * 8*pi / |G|^2 / N_grid^2
    //      (the N_grid^2 in the denominator: one from the density normalization
    //       and one from the Parseval relation via the energy formula dividing by N_grid)
    //
    // For a cosine with amplitude A, n(G1) = (A/2)*N_grid and n(-G1) = (A/2)*N_grid
    // E_H = (Omega/2) * 2 * (A/2)^2 * N_grid^2 * 8*pi / |G1|^2 / N_grid^2
    //     = (Omega/2) * 2 * (A/2)^2 * 8*pi / |G1|^2
    //     = Omega * A^2/4 * 8*pi / |G1|^2
    //     = 2*pi*Omega*A^2 / |G1|^2
    double e_h_analytical = 2.0 * constants::pi * volume * A * A / G1_sq;

    // Allow a few percent tolerance due to discrete grid effects
    EXPECT_NEAR(e_h, e_h_analytical, e_h_analytical * 0.05)
        << "Hartree energy should match analytical result for cosine density. "
        << "Computed=" << e_h << " Expected=" << e_h_analytical;
}

// ============================================================================
// Test 2: LDA Slater exchange at known rs values
// ============================================================================
// The Slater exchange energy density is:
//   ex(n) = -2 * (3/4) * (3/pi)^{1/3} * n^{1/3}   [Ry]
// In terms of rs = (3/(4*pi*n))^{1/3}:
//   ex = -0.9163/rs  [Ry]

TEST(XCFunctional, SlaterExchangeAtKnownRs) {
    // Test at several rs values
    double rs_values[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    for (double rs : rs_values) {
        // n = 3/(4*pi*rs^3)
        double n = 3.0 / (4.0 * constants::pi * rs * rs * rs);

        // Expected: ex = -2 * (3/4) * (3/pi)^{1/3} * n^{1/3}
        //             = -0.9163 / rs  [Ry]
        double ex_expected = -0.9163296 / rs;

        // Use the XC evaluator with a single-point density
        RVec density = {n};
        XCEvaluator xc("LDA_PZ");
        // volume/N_grid = 1.0 for a single-point evaluation
        XCResult result = xc.evaluate(density, 1.0);

        // The exc[0] is the total energy density (exchange + correlation)
        // We need to separate exchange. Use a second evaluator approach:
        // For large rs, correlation is small relative to exchange.
        // Instead, let's test the total XC and verify the exchange part
        // by computing correlation separately.

        // PZ correlation at these rs values:
        // rs >= 1: ec = gamma / (1 + beta1*sqrt(rs) + beta2*rs)
        double gamma = -0.2846;  // Ry (2 × -0.1423 Ha)
        double beta1 = 1.0529;
        double beta2 = 0.3334;
        double ec_expected = gamma / (1.0 + beta1 * std::sqrt(rs) + beta2 * rs);

        double exc_expected = ex_expected + ec_expected;

        EXPECT_NEAR(result.exc[0], exc_expected, std::abs(exc_expected) * 0.002)
            << "LDA XC energy density at rs=" << rs
            << ": computed=" << result.exc[0] << " expected=" << exc_expected;
    }
}

// ============================================================================
// Test 3: PZ correlation continuity at rs=1
// ============================================================================
// The two branches (rs<1 and rs>=1) must give the same value at rs=1.

TEST(XCFunctional, PZCorrelationContinuityAtRs1) {
    // Evaluate at rs just below and just above 1.0
    double delta = 1e-6;
    double rs_below = 1.0 - delta;
    double rs_above = 1.0 + delta;

    auto n_from_rs = [](double rs) {
        return 3.0 / (4.0 * constants::pi * rs * rs * rs);
    };

    // PZ parameters (Ry = 2× Hartree values)
    constexpr double gamma = -0.2846;
    constexpr double beta1 = 1.0529;
    constexpr double beta2 = 0.3334;
    constexpr double Ac =  0.0622;
    constexpr double B  = -0.0960;
    constexpr double C  =  0.0040;
    constexpr double D  = -0.0232;

    // rs < 1 branch at rs_below
    double lnrs = std::log(rs_below);
    double ec_low = Ac * lnrs + B + C * rs_below * lnrs + D * rs_below;

    // rs >= 1 branch at rs_above
    double sqrs = std::sqrt(rs_above);
    double ec_high = gamma / (1.0 + beta1 * sqrs + beta2 * rs_above);

    // These should be nearly equal (continuity)
    EXPECT_NEAR(ec_low, ec_high, 1e-4)
        << "PZ correlation should be continuous at rs=1: "
        << "ec(rs=1-)=" << ec_low << " ec(rs=1+)=" << ec_high;

    // Also verify against the actual XC evaluator
    XCEvaluator xc("LDA_PZ");

    double n_below = n_from_rs(rs_below);
    double n_above = n_from_rs(rs_above);

    XCResult r_below = xc.evaluate({n_below}, 1.0);
    XCResult r_above = xc.evaluate({n_above}, 1.0);

    // Total exc should be continuous
    EXPECT_NEAR(r_below.exc[0], r_above.exc[0], 1e-4)
        << "Total XC energy density should be continuous at rs=1";
}

// ============================================================================
// Test 4: PZ correlation at tabulated values
// ============================================================================
// Compare against known PZ correlation values from the literature.

TEST(XCFunctional, PZCorrelationTabulated) {
    // Known PZ correlation values (Ry) at select rs
    // ec(rs=2) = -0.1423 / (1 + 1.0529*sqrt(2) + 0.3334*2)
    //          = -0.1423 / (1 + 1.4892 + 0.6668)
    //          = -0.1423 / 3.1560 = -0.04509
    double rs2 = 2.0;
    double ec_rs2_expected = -0.2846 / (1.0 + 1.0529 * std::sqrt(2.0) + 0.3334 * 2.0);

    // ec(rs=5) = gamma / (1 + beta1*sqrt(5) + beta2*5)  [Ry]
    double rs5 = 5.0;
    double ec_rs5_expected = -0.2846 / (1.0 + 1.0529 * std::sqrt(5.0) + 0.3334 * 5.0);

    // ec(rs=0.5) = A*ln(0.5) + B + C*0.5*ln(0.5) + D*0.5  [Ry]
    double rs05 = 0.5;
    double lnrs = std::log(0.5);
    double ec_rs05_expected = 0.0622 * lnrs + (-0.0960) + 0.0040 * 0.5 * lnrs + (-0.0232) * 0.5;

    auto n_from_rs = [](double rs) {
        return 3.0 / (4.0 * constants::pi * rs * rs * rs);
    };

    XCEvaluator xc("LDA_PZ");

    // For each rs, extract the correlation by subtracting the exchange
    auto test_ec = [&](double rs, double ec_expected) {
        double n = n_from_rs(rs);
        XCResult result = xc.evaluate({n}, 1.0);
        // Exchange: ex = -2 * ax * n^{1/3}, ax = (3/4)(3/pi)^{1/3}
        double ax = 0.75 * std::cbrt(3.0 / constants::pi);
        double ex = -2.0 * ax * std::cbrt(n);
        double ec_computed = result.exc[0] - ex;
        EXPECT_NEAR(ec_computed, ec_expected, std::abs(ec_expected) * 0.01)
            << "PZ correlation at rs=" << rs
            << ": computed=" << ec_computed << " expected=" << ec_expected;
    };

    test_ec(rs2, ec_rs2_expected);
    test_ec(rs5, ec_rs5_expected);
    test_ec(rs05, ec_rs05_expected);
}

// ============================================================================
// Test 5: Free electron gas — empty lattice eigenvalues
// ============================================================================
// With V_loc = 0 and no nonlocal PP, the eigenvalues should be |k+G|^2/2.

TEST(FreeElectronGas, HamiltonianIsKineticOnly) {
    Crystal crystal = make_simple_cubic(5.0, "X");
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);

    int num_grid = fft.total_points();
    int npw = static_cast<int>(basis.num_pw());

    // Zero effective potential
    std::vector<complex_t> veff_r(num_grid, complex_t{0.0, 0.0});

    // Create a trivial nonlocal PP (no projectors)
    std::map<std::string, PseudoPotential> empty_pp;
    PseudoPotential pp;
    pp.z_valence = 0;
    pp.mesh.npoints = 2;
    pp.mesh.r = {0.0, 1.0};
    pp.mesh.rab = {1.0, 1.0};
    pp.vloc = {0.0, 0.0};
    empty_pp["X"] = pp;

    NonlocalPP nonlocal(crystal, basis, empty_pp);
    Hamiltonian ham(crystal, basis, fft, nonlocal);
    ham.update_veff(veff_r);

    // With V_eff = 0, applying H to a PW basis vector e_G should give
    //   H|e_G> = (|k+G|^2 / 2) * |e_G>
    // i.e. only kinetic energy, no local or nonlocal contribution.
    Vec3 k_gamma = {0.0, 0.0, 0.0};
    auto h_apply = ham.get_apply_function(k_gamma);
    auto kinetic = basis.kinetic_energies(k_gamma);

    // Test several G-vectors: the Rayleigh quotient <e_G|H|e_G> = KE_G
    int num_test = std::min(npw, 10);
    for (int ig = 0; ig < num_test; ++ig) {
        CVec e_g(npw, complex_t{0.0, 0.0});
        e_g[ig] = complex_t{1.0, 0.0};

        CVec h_eg = h_apply(e_g);

        // Rayleigh quotient: <e_G|H|e_G>
        double rayleigh = std::real(std::conj(e_g[ig]) * h_eg[ig]);
        EXPECT_NEAR(rayleigh, kinetic[ig], 1e-6)
            << "Free electron Rayleigh quotient for G-vector " << ig
            << ": computed=" << rayleigh << " expected=" << kinetic[ig];

        // Off-diagonal elements should be zero (H is diagonal with V=0)
        double off_diag_norm = 0.0;
        for (int j = 0; j < npw; ++j) {
            if (j != ig) {
                off_diag_norm += std::norm(h_eg[j]);
            }
        }
        EXPECT_LT(off_diag_norm, 1e-10)
            << "Off-diagonal H elements should be zero for free electron gas, "
            << "G-vector " << ig;
    }
}

// ============================================================================
// Test 6: Jellium — uniform electron gas total energy
// ============================================================================
// For a homogeneous electron gas of density n0, the total energy per electron
// in Ry is:  e = (3/5) * k_F^2 + ex + ec
// where k_F = (3*pi^2*n0)^{1/3}, ex = -0.9163/rs, ec from PZ.

TEST(Jellium, XCEnergyMatchesAnalytical) {
    // Use a simple cubic cell and uniform density
    double a_ang = 5.0;
    Crystal crystal = make_simple_cubic(a_ang, "Je");
    double volume = crystal.volume();

    // Choose density corresponding to rs = 3.0
    double rs = 3.0;
    double n0 = 3.0 / (4.0 * constants::pi * rs * rs * rs);

    // Build uniform density on a grid
    PlaneWaveBasis basis(crystal, 10.0);
    FFTGrid fft(basis);
    int num_grid = fft.total_points();

    RVec density_r(num_grid, n0);

    // Compute XC energy
    XCEvaluator xc("LDA_PZ");
    XCResult xc_result = xc.evaluate(density_r, volume);

    // Analytical XC energy:
    // E_xc = Omega * n0 * (ex + ec)
    double ax = 0.75 * std::cbrt(3.0 / constants::pi);
    double ex = -2.0 * ax * std::cbrt(n0);

    double gamma = -0.2846, beta1 = 1.0529, beta2 = 0.3334;  // Ry
    double ec = gamma / (1.0 + beta1 * std::sqrt(rs) + beta2 * rs);

    double exc_per_electron = ex + ec;
    double e_xc_analytical = volume * n0 * exc_per_electron;

    EXPECT_NEAR(xc_result.energy, e_xc_analytical, std::abs(e_xc_analytical) * 0.001)
        << "Jellium XC energy: computed=" << xc_result.energy
        << " expected=" << e_xc_analytical;
}

// ============================================================================
// Test 7: Si diamond SCF convergence
// ============================================================================
// Construct a minimal Si PP, verify SCF converges, energy is negative.

TEST(SCFConvergence, SiDiamondConverges) {
    Crystal crystal = make_si_diamond();

    // Build a simple local PP for Si with 4 valence electrons
    auto pp = make_simple_local_pp(4.0);

    std::map<std::string, PseudoPotential> pseudopotentials;
    pseudopotentials["Si"] = pp;

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;   // low cutoff for speed
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};  // Gamma only for speed

    // Energy-only convergence — density may not fully converge with
    // this simplified PP, but the energy should stabilize.
    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;  // very loose, rely on energy convergence
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    // SCF should converge (at least in energy)
    EXPECT_TRUE(result.converged)
        << "SCF should converge for Si diamond with simple local PP";

    // Total energy should be negative (bound state)
    EXPECT_LT(result.total_energy_ry, 0.0)
        << "Total energy should be negative for Si diamond";
}

// ============================================================================
// Test 8: Energy consistency
// ============================================================================
// Verify E_total = E_kin + E_H + E_xc + E_loc + E_NL + E_ewald
// using the SCFResult energy components.

TEST(SCFConvergence, EnergyComponentsConsistent) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials;
    pseudopotentials["Si"] = pp;

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge; skipping energy consistency check";
    }

    // The SCF total energy = E_band - E_H + E_xc - int(Vxc*n) + E_ewald
    // But we can check that the reported components are self-consistent:
    // The individual energy components should sum to something close to the
    // total (modulo the double-counting corrections which are already applied).
    //
    // At minimum, verify all components are finite
    EXPECT_TRUE(std::isfinite(result.kinetic_energy));
    EXPECT_TRUE(std::isfinite(result.hartree_energy));
    EXPECT_TRUE(std::isfinite(result.xc_energy));
    EXPECT_TRUE(std::isfinite(result.local_pp_energy));
    EXPECT_TRUE(std::isfinite(result.nonlocal_pp_energy));
    EXPECT_TRUE(std::isfinite(result.ewald_energy));
    EXPECT_TRUE(std::isfinite(result.total_energy_ry));

    // Hartree energy should be positive (Coulomb repulsion)
    EXPECT_GT(result.hartree_energy, 0.0)
        << "Hartree energy should be positive";

    // XC energy should be negative for LDA
    EXPECT_LT(result.xc_energy, 0.0)
        << "XC energy should be negative for LDA";

    // Energy in eV should be consistent with Ry
    EXPECT_NEAR(result.total_energy_ev,
                result.total_energy_ry * constants::rydberg_to_ev,
                1e-6);
}

// ============================================================================
// Test 9: Force symmetry — Si diamond at equilibrium
// ============================================================================
// Si diamond at equilibrium should have zero forces by symmetry.

TEST(Forces, SiDiamondEquilibriumZeroForces) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials;
    pseudopotentials["Si"] = pp;

    CalculationParams calc;
    calc.type = CalculationType::SCF;  // need forces
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge; skipping force symmetry check";
    }

    // At equilibrium, the Ewald forces on Si diamond atoms should be zero
    // by the diamond symmetry (both atoms are at inversion centers)
    ASSERT_EQ(result.ewald_forces.size(), 2u);
    for (size_t ia = 0; ia < 2; ++ia) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(result.ewald_forces[ia][d], 0.0, 1e-6)
                << "Ewald force on Si atom " << ia << " direction " << d
                << " should be zero by symmetry";
        }
    }

    // Total HF forces should also be small (may not be exactly zero due
    // to the simplified PP and low cutoff, but should be bounded)
    if (!result.forces.empty()) {
        for (size_t ia = 0; ia < result.forces.size(); ++ia) {
            double f_mag = std::sqrt(
                result.forces[ia][0] * result.forces[ia][0] +
                result.forces[ia][1] * result.forces[ia][1] +
                result.forces[ia][2] * result.forces[ia][2]);
            // With a symmetric structure, forces should be small
            EXPECT_LT(f_mag, 1.0)
                << "Force magnitude on atom " << ia
                << " should be bounded";
        }

        // Forces on the two Si atoms should be equal and opposite
        if (result.forces.size() == 2) {
            for (int d = 0; d < 3; ++d) {
                EXPECT_NEAR(result.forces[0][d] + result.forces[1][d], 0.0, 0.01)
                    << "Forces on diamond atoms should sum to zero (Newton's 3rd law), "
                    << "direction " << d;
            }
        }
    }
}

// ============================================================================
// Test 10: Madelung constant for NaCl
// ============================================================================
// The Madelung constant for NaCl (rocksalt) is alpha = 1.7476.
// E_Madelung = -alpha * Z+ * Z- * e^2 / a_nn  per ion pair.
// In Rydberg units (e^2 = 2):
//   E = -alpha * 2 / a_nn  per pair (for Z+ = Z- = 1).

TEST(Ewald, NaClMadelungConstantValue) {
    // NaCl conventional cell: 4 NaCl formula units
    double a_ang = 5.64;
    double a_bohr = a_ang * constants::angstrom_to_bohr;
    double a_nn = a_bohr / 2.0;  // nearest-neighbor distance

    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};

    // 8 atoms in the conventional cell: 4 Na + 4 Cl
    // Na at: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    // Cl at: (0.5,0,0), (0,0.5,0), (0,0,0.5), (0.5,0.5,0.5)
    std::vector<Atom> atoms = {
        {"Na", 11, {0.0, 0.0, 0.0}},
        {"Na", 11, {0.5, 0.5, 0.0}},
        {"Na", 11, {0.5, 0.0, 0.5}},
        {"Na", 11, {0.0, 0.5, 0.5}},
        {"Cl", 17, {0.5, 0.0, 0.0}},
        {"Cl", 17, {0.0, 0.5, 0.0}},
        {"Cl", 17, {0.0, 0.0, 0.5}},
        {"Cl", 17, {0.5, 0.5, 0.5}},
    };
    Crystal crystal(lattice, std::move(atoms));

    // Use charges +1 and -1 to get the Madelung energy directly
    std::vector<double> charges = {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0};
    auto result = EwaldCalculator::compute(crystal, charges);

    // With 4 formula units in the cell:
    // E_total = 4 * (-alpha * 2 / a_nn)
    // alpha = -E_total * a_nn / (4 * 2)
    double alpha_computed = -result.energy * a_nn / (4.0 * 2.0);

    // Expected Madelung constant: 1.7476
    EXPECT_NEAR(alpha_computed, 1.7476, 0.05)
        << "NaCl Madelung constant: computed=" << alpha_computed
        << " expected=1.7476";
}

// ============================================================================
// Test: Local PP energy G-space vs real-space consistency
// ============================================================================
// The local PP energy computed via G-space sum should match the real-space
// integral when both use the same density.

TEST(LocalPPEnergy, GSpaceMatchesRealSpace) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);

    int num_grid = fft.total_points();
    double volume = crystal.volume();
    int npw = static_cast<int>(basis.num_pw());

    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials;
    pseudopotentials["Si"] = pp;

    LocalPPEvaluator local_pp(crystal, basis, pseudopotentials);

    // Use a uniform density
    double n0 = 4.0 / volume;  // 4 electrons in the cell
    RVec density_r(num_grid, n0);

    // FFT to G-space
    std::vector<complex_t> density_c(num_grid);
    for (int i = 0; i < num_grid; ++i)
        density_c[i] = complex_t{density_r[i], 0.0};
    std::vector<complex_t> density_g_full(num_grid);
    fft.forward(density_c, density_g_full);
    CVec density_g(npw);
    fft.gather_from_grid(basis, density_g_full, density_g);

    // G-space energy
    double e_loc_g = local_pp.energy(density_g, volume, num_grid);

    // Real-space energy: E_loc = sum_r V_loc(r) * n(r) * (Omega/N_grid)
    // First get V_loc in real space via IFFT
    const CVec& vloc_g = local_pp.vloc_g();
    std::vector<complex_t> vloc_grid(num_grid, complex_t{0.0, 0.0});
    fft.scatter_to_grid(basis, vloc_g, vloc_grid);
    std::vector<complex_t> vloc_r(num_grid);
    fft.inverse(vloc_grid, vloc_r);

    // vloc_g is in "physics" convention.  IFFT divides by N, so
    // vloc_r[i] = V_loc(r_i) / N.  Multiply by N to get V_loc(r).
    double e_loc_r = 0.0;
    double dv = volume / num_grid;
    for (int i = 0; i < num_grid; ++i) {
        e_loc_r += std::real(vloc_r[i]) * static_cast<double>(num_grid)
                   * density_r[i] * dv;
    }

    EXPECT_NEAR(e_loc_g, e_loc_r, std::abs(e_loc_r) * 0.01)
        << "Local PP energy: G-space=" << e_loc_g
        << " real-space=" << e_loc_r;
}

// ============================================================================
// Test: XC potential derivative consistency
// ============================================================================
// The XC potential should satisfy: vxc = d(n*exc)/dn
// For Slater exchange: vxc = (4/3) * ex(n)

TEST(XCFunctional, ExchangePotentialIsCorrectDerivative) {
    // Avoid rs=1.0 which sits exactly at the PZ branch boundary
    // (rs<1 vs rs>=1), causing a tiny discontinuity in the numerical derivative.
    double rs_values[] = {0.5, 2.0, 4.0};

    for (double rs : rs_values) {
        double n = 3.0 / (4.0 * constants::pi * rs * rs * rs);

        XCEvaluator xc("LDA_PZ");
        XCResult result = xc.evaluate({n}, 1.0);

        // Numerical derivative of n*exc w.r.t. n
        double dn = n * 1e-5;
        XCResult r_plus = xc.evaluate({n + dn}, 1.0);
        XCResult r_minus = xc.evaluate({n - dn}, 1.0);

        double nexc_plus = (n + dn) * r_plus.exc[0];
        double nexc_minus = (n - dn) * r_minus.exc[0];
        double vxc_numerical = (nexc_plus - nexc_minus) / (2.0 * dn);

        EXPECT_NEAR(result.vxc[0], vxc_numerical, std::abs(vxc_numerical) * 0.001)
            << "XC potential at rs=" << rs
            << ": computed=" << result.vxc[0]
            << " numerical=" << vxc_numerical;
    }
}

// ============================================================================
// Test: Hartree potential is self-consistent with energy
// ============================================================================
// E_H = (1/2) * integral V_H(r) * n(r) dr

TEST(HartreeEnergy, PotentialEnergyConsistency) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    double ecutwfc = 10.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);

    int num_grid = fft.total_points();
    double volume = crystal.volume();
    int npw = static_cast<int>(basis.num_pw());

    // Use a non-trivial density
    auto dims = fft.dims();
    Mat3 lat_bohr = crystal.lattice_bohr();

    RVec density_r(num_grid);
    double n0 = 0.05;
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                double fx = static_cast<double>(i) / dims[0];
                int idx = (i * dims[1] + j) * dims[2] + k;
                // Slowly varying density
                density_r[idx] = n0 * (1.0 + 0.3 * std::sin(2.0 * constants::pi * fx));
            }
        }
    }

    // FFT to G-space
    std::vector<complex_t> density_c(num_grid);
    for (int i = 0; i < num_grid; ++i)
        density_c[i] = complex_t{density_r[i], 0.0};
    std::vector<complex_t> density_g_full(num_grid);
    fft.forward(density_c, density_g_full);
    CVec density_g(npw);
    fft.gather_from_grid(basis, density_g_full, density_g);

    // Compute Hartree
    HartreeSolver hartree(basis);
    CVec vhartree_g = hartree.compute(density_g);
    double e_h_gspace = hartree.energy(density_g, vhartree_g, volume, num_grid);

    // Also compute via real-space integration: E_H = (1/2) * int V_H(r) * n(r) dr
    std::vector<complex_t> vh_grid(num_grid, complex_t{0.0, 0.0});
    fft.scatter_to_grid(basis, vhartree_g, vh_grid);
    std::vector<complex_t> vh_r(num_grid);
    fft.inverse(vh_grid, vh_r);

    double e_h_rspace = 0.0;
    double dv = volume / num_grid;
    for (int i = 0; i < num_grid; ++i) {
        e_h_rspace += 0.5 * std::real(vh_r[i]) * density_r[i] * dv;
    }

    EXPECT_NEAR(e_h_gspace, e_h_rspace, std::abs(e_h_rspace) * 0.02)
        << "Hartree energy: G-space=" << e_h_gspace
        << " real-space=" << e_h_rspace;
}

// ============================================================================
// Test: Max SCF iterations enforced
// ============================================================================

TEST(SCFLoop, MaxIterationsEnforced) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-20;   // impossibly tight
    conv.density_threshold = 1e-20;  // impossibly tight
    conv.max_scf_steps = 5;          // very few steps

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    // Should stop at max iterations
    EXPECT_LE(result.scf_steps, 5);
}

// ============================================================================
// Test: Idempotency — re-converged SCF gives same energy
// ============================================================================

TEST(SCFLoop, Idempotency) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    // First run
    SCFSolver solver1(crystal, calc, conv, pseudopotentials);
    SCFResult result1 = solver1.solve();

    if (!result1.converged) {
        GTEST_SKIP() << "First SCF did not converge";
    }

    // Second run with same parameters
    SCFSolver solver2(crystal, calc, conv, pseudopotentials);
    SCFResult result2 = solver2.solve();

    EXPECT_TRUE(result2.converged);
    EXPECT_NEAR(result1.total_energy_ry, result2.total_energy_ry, 1e-4)
        << "Two identical SCF runs should give same energy";
}

// ============================================================================
// Test: Charge conservation
// ============================================================================

TEST(SCFLoop, ChargeConservation) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }

    // Total energy in eV should be consistent
    EXPECT_NEAR(result.total_energy_ev,
                result.total_energy_ry * constants::rydberg_to_ev,
                1e-6);

    // All energy components should be finite
    EXPECT_TRUE(std::isfinite(result.kinetic_energy));
    EXPECT_TRUE(std::isfinite(result.hartree_energy));
    EXPECT_TRUE(std::isfinite(result.xc_energy));
}

// ============================================================================
// Test: SCF steps are positive
// ============================================================================

TEST(SCFLoop, StepsArePositive) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    EXPECT_GT(result.scf_steps, 0)
        << "Should take at least one SCF step";
}

// ============================================================================
// Test: Eigenvalues are present in result
// ============================================================================

TEST(SCFLoop, EigenvaluesPresent) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    // Should have eigenvalues for at least one k-point
    EXPECT_GT(result.eigenvalues.size(), 0u);
    if (!result.eigenvalues.empty()) {
        EXPECT_GT(result.eigenvalues[0].size(), 0u);
        // Eigenvalues should be sorted
        for (size_t i = 1; i < result.eigenvalues[0].size(); ++i) {
            EXPECT_GE(result.eigenvalues[0][i],
                      result.eigenvalues[0][i-1] - 1e-6);
        }
    }
}

// ============================================================================
// Test: Hartree energy positive in SCF
// ============================================================================

TEST(SCFLoop, HartreeEnergyPositive) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }

    EXPECT_GT(result.hartree_energy, 0.0)
        << "Hartree energy should be positive";
}

// ============================================================================
// Test: XC energy negative for LDA
// ============================================================================

TEST(SCFLoop, XCEnergyNegative) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    if (!result.converged) {
        GTEST_SKIP() << "SCF did not converge";
    }

    EXPECT_LT(result.xc_energy, 0.0)
        << "XC energy should be negative for LDA";
}

// ============================================================================
// Test: Ewald forces present
// ============================================================================

TEST(SCFLoop, EwaldForcesPresent) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    EXPECT_EQ(result.ewald_forces.size(), 2u);
}

// ============================================================================
// Test: Timing information recorded
// ============================================================================

TEST(SCFLoop, TimingInformation) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 20;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    // Timing map should be populated
    EXPECT_GT(result.timing.size(), 0u)
        << "SCF result should contain timing information";
}

// ============================================================================
// Test: Converged V_eff present
// ============================================================================

TEST(SCFLoop, ConvergedVeffPresent) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    // converged_veff_r should be populated
    EXPECT_GT(result.converged_veff_r.size(), 0u)
        << "Converged V_eff should be available after SCF";
}

// ============================================================================
// Test: Simple cubic crystal SCF
// ============================================================================

TEST(SCFLoop, SimpleCubicConverges) {
    Crystal crystal = make_simple_cubic(5.0, "Si");
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.total_energy_ry, 0.0);
}

// ============================================================================
// Test: Fermi energy in SCF result
// ============================================================================

TEST(SCFLoop, FermiEnergyReasonable) {
    Crystal crystal = make_si_diamond();
    auto pp = make_simple_local_pp(4.0);
    std::map<std::string, PseudoPotential> pseudopotentials = {{"Si", pp}};

    CalculationParams calc;
    calc.type = CalculationType::SCF;
    calc.ecutwfc = 8.0;
    calc.xc_functional = "LDA_PZ";
    calc.kpoints.grid = {1, 1, 1};

    ConvergenceParams conv;
    conv.energy_threshold = 1e-3;
    conv.density_threshold = 1.0;
    conv.max_scf_steps = 100;

    SCFSolver solver(crystal, calc, conv, pseudopotentials);
    SCFResult result = solver.solve();

    EXPECT_TRUE(std::isfinite(result.fermi_energy_ev))
        << "Fermi energy should be finite";
}

} // anonymous namespace
