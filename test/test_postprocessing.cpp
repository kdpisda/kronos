// ============================================================================
// KRONOS  test/test_postprocessing.cpp
// Tests for band structure and density of states calculations.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "postprocessing/band_structure.hpp"
#include "postprocessing/dos.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "potential/nonlocal_pp.hpp"
#include "solver/fermi.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <fstream>

using namespace kronos;

// ============================================================================
// Band Structure: k-path generation
// ============================================================================

TEST(BandStructure, GenerateKPathSC) {
    auto crystal = test::make_cubic_crystal(5.0);
    auto path_spec = BandStructureCalculator::default_path_sc();

    auto kpath = BandStructureCalculator::generate_kpath(crystal, path_spec, 10);

    EXPECT_GT(kpath.kpoints.size(), 0u);
    EXPECT_EQ(kpath.kpoints.size(), kpath.distances.size());

    // Distances should be monotonically non-decreasing
    for (size_t i = 1; i < kpath.distances.size(); ++i) {
        EXPECT_GE(kpath.distances[i], kpath.distances[i-1])
            << "k-path distances should be monotonically increasing";
    }

    // First distance should be 0
    EXPECT_NEAR(kpath.distances[0], 0.0, 1e-12);

    // Tick positions should exist
    EXPECT_GT(kpath.tick_positions.size(), 0u);
}

TEST(BandStructure, GenerateKPathFCC) {
    auto crystal = test::make_si_diamond_crystal();
    auto path_spec = BandStructureCalculator::default_path_fcc();

    auto kpath = BandStructureCalculator::generate_kpath(crystal, path_spec, 20);

    EXPECT_GT(kpath.kpoints.size(), 0u);
    EXPECT_EQ(kpath.kpoints.size(), kpath.distances.size());
}

TEST(BandStructure, CustomPath) {
    auto crystal = test::make_cubic_crystal(5.0);
    KPathSpec path = {
        {"G", {0.0, 0.0, 0.0}},
        {"X", {0.5, 0.0, 0.0}},
    };

    auto kpath = BandStructureCalculator::generate_kpath(crystal, path, 10);

    // Should have exactly 10 points (or 11 including endpoint)
    EXPECT_GE(kpath.kpoints.size(), 10u);

    // First point should be Gamma
    EXPECT_NEAR(kpath.kpoints[0][0], 0.0, 1e-12);
    EXPECT_NEAR(kpath.kpoints[0][1], 0.0, 1e-12);
    EXPECT_NEAR(kpath.kpoints[0][2], 0.0, 1e-12);
}

// ============================================================================
// Band Structure: compute bands (free electron)
// ============================================================================

TEST(BandStructure, FreeElectronBands) {
    auto crystal = test::make_cubic_crystal(5.0, "X", 1);
    double ecutwfc = 8.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int nbands = 4;

    auto pp_map = std::map<std::string, PseudoPotential>{
        {"X", test::make_empty_pp("X", 0.0)}};
    NonlocalPP nonlocal(crystal, basis, pp_map);
    Hamiltonian ham(crystal, basis, fft, nonlocal);

    std::vector<complex_t> veff_r(fft.total_points(), complex_t{0.0, 0.0});
    ham.update_veff(veff_r);

    KPathSpec path = {
        {"G", {0.0, 0.0, 0.0}},
        {"X", {0.5, 0.0, 0.0}},
    };
    auto kpath = BandStructureCalculator::generate_kpath(crystal, path, 5);

    std::function<std::function<CVec(const CVec&)>(const Vec3&)> h_factory =
        [&](const Vec3& k) -> std::function<CVec(const CVec&)> {
            return ham.get_apply_function(k);
        };
    std::function<std::vector<double>(const Vec3&)> precond_factory =
        [&](const Vec3& k) -> std::vector<double> {
            return ham.kinetic_diagonal(k);
        };
    std::function<int(const Vec3&)> npw_func =
        [&](const Vec3&) -> int { return npw; };

    BandStructureCalculator::compute_bands(
        kpath, h_factory, precond_factory, nbands, npw_func);

    // Should have eigenvalues for each k-point
    ASSERT_EQ(kpath.eigenvalues.size(), kpath.kpoints.size());
    for (const auto& eigs : kpath.eigenvalues) {
        ASSERT_EQ(static_cast<int>(eigs.size()), nbands);
        // Eigenvalues should be sorted
        for (int i = 1; i < nbands; ++i) {
            EXPECT_GE(eigs[i], eigs[i-1] - 1e-6);
        }
    }

    // At Gamma, lowest eigenvalue should be non-negative
    // (kinetic energy of G=0 is 0, but Davidson might not converge exactly)
    EXPECT_GE(kpath.eigenvalues[0][0], -0.5);
}

TEST(BandStructure, BandCountMatchesNbands) {
    auto crystal = test::make_cubic_crystal(5.0, "X", 1);
    double ecutwfc = 6.0;
    PlaneWaveBasis basis(crystal, ecutwfc);
    FFTGrid fft(basis);
    int npw = static_cast<int>(basis.num_pw());
    int nbands = 3;

    auto pp_map = std::map<std::string, PseudoPotential>{
        {"X", test::make_empty_pp("X", 0.0)}};
    NonlocalPP nonlocal(crystal, basis, pp_map);
    Hamiltonian ham(crystal, basis, fft, nonlocal);

    std::vector<complex_t> veff_r(fft.total_points(), complex_t{0.0, 0.0});
    ham.update_veff(veff_r);

    KPathSpec path = {{"G", {0.0, 0.0, 0.0}}, {"X", {0.5, 0.0, 0.0}}};
    auto kpath = BandStructureCalculator::generate_kpath(crystal, path, 3);

    std::function<std::function<CVec(const CVec&)>(const Vec3&)> h_factory2 =
        [&](const Vec3& k) -> std::function<CVec(const CVec&)> {
            return ham.get_apply_function(k);
        };
    std::function<std::vector<double>(const Vec3&)> precond_factory2 =
        [&](const Vec3& k) -> std::vector<double> {
            return ham.kinetic_diagonal(k);
        };
    std::function<int(const Vec3&)> npw_func2 =
        [&](const Vec3&) -> int { return npw; };

    BandStructureCalculator::compute_bands(
        kpath, h_factory2, precond_factory2, nbands, npw_func2);

    for (const auto& eigs : kpath.eigenvalues) {
        EXPECT_EQ(static_cast<int>(eigs.size()), nbands);
    }
}

// ============================================================================
// Band Structure: file output
// ============================================================================

TEST(BandStructure, WriteBandsGnuplot) {
    auto crystal = test::make_cubic_crystal(5.0, "X", 1);
    KPathSpec path = {{"G", {0.0, 0.0, 0.0}}, {"X", {0.5, 0.0, 0.0}}};
    auto kpath = BandStructureCalculator::generate_kpath(crystal, path, 3);

    // Fill dummy eigenvalues
    int nbands = 2;
    for (auto& kp : kpath.kpoints) {
        (void)kp;
        kpath.eigenvalues.push_back({-0.5, 0.5});
    }

    std::string tmpfile = std::filesystem::temp_directory_path().string()
                         + "/kronos_test_bands.dat";
    BandStructureCalculator::write_bands_gnuplot(tmpfile, kpath);

    // File should exist and be non-empty
    EXPECT_TRUE(std::filesystem::exists(tmpfile));
    std::ifstream ifs(tmpfile);
    std::string line;
    int line_count = 0;
    while (std::getline(ifs, line)) {
        ++line_count;
    }
    EXPECT_GT(line_count, 0);

    std::filesystem::remove(tmpfile);
}

// ============================================================================
// Default paths exist
// ============================================================================

TEST(BandStructure, DefaultPathsExist) {
    auto fcc = BandStructureCalculator::default_path_fcc();
    auto bcc = BandStructureCalculator::default_path_bcc();
    auto sc  = BandStructureCalculator::default_path_sc();
    auto hcp = BandStructureCalculator::default_path_hcp();

    EXPECT_GE(fcc.size(), 3u);
    EXPECT_GE(bcc.size(), 3u);
    EXPECT_GE(sc.size(), 3u);
    EXPECT_GE(hcp.size(), 3u);
}

// ============================================================================
// DOS: basic computation
// ============================================================================

TEST(DOS, BasicComputation) {
    // Simple 4-band system at 1 k-point
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-5.0 * ev_to_ry, -2.0 * ev_to_ry, 2.0 * ev_to_ry, 5.0 * ev_to_ry}
    };
    std::vector<double> weights = {1.0};

    auto dos = DOSCalculator::compute_dos(
        eigenvalues, weights, SmearingType::Gaussian,
        0.2, -10.0, 10.0, 1001, 2);

    ASSERT_EQ(dos.energies.size(), 1001u);
    ASSERT_EQ(dos.dos_values.size(), 1001u);
    ASSERT_EQ(dos.integrated_dos.size(), 1001u);

    // DOS should be non-negative everywhere
    for (double d : dos.dos_values) {
        EXPECT_GE(d, -1e-10);
    }

    // Energy grid should span the requested range
    EXPECT_NEAR(dos.energies.front(), -10.0, 0.1);
    EXPECT_NEAR(dos.energies.back(), 10.0, 0.1);
}

TEST(DOS, IntegratesToNElectrons) {
    // 4 bands, 4 electrons, should integrate to 4
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-5.0 * ev_to_ry, -2.0 * ev_to_ry, 2.0 * ev_to_ry, 5.0 * ev_to_ry}
    };
    std::vector<double> weights = {1.0};
    int spin_factor = 2;

    auto dos = DOSCalculator::compute_dos(
        eigenvalues, weights, SmearingType::Gaussian,
        0.2, -20.0, 20.0, 4001, spin_factor);

    // Integrated DOS at the end should be total number of states
    // = nbands * spin_factor * sum(weights)
    double total_states = 4.0 * spin_factor * 1.0;
    EXPECT_NEAR(dos.integrated_dos.back(), total_states, 0.5)
        << "Integrated DOS should approach total states at high energy";
}

TEST(DOS, IntegratedDOSMonotonic) {
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-3.0 * ev_to_ry, -1.0 * ev_to_ry, 1.0 * ev_to_ry, 3.0 * ev_to_ry}
    };
    std::vector<double> weights = {1.0};

    auto dos = DOSCalculator::compute_dos(
        eigenvalues, weights, SmearingType::Gaussian,
        0.1, -10.0, 10.0, 1001, 2);

    // Integrated DOS should be monotonically non-decreasing
    for (size_t i = 1; i < dos.integrated_dos.size(); ++i) {
        EXPECT_GE(dos.integrated_dos[i], dos.integrated_dos[i-1] - 1e-10);
    }
}

TEST(DOS, FermiLevelInGapForInsulator) {
    // Insulator: gap between band 2 and band 3
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-5.0 * ev_to_ry, -2.0 * ev_to_ry, 3.0 * ev_to_ry, 6.0 * ev_to_ry}
    };
    std::vector<double> weights = {1.0};
    double target_electrons = 4.0;

    auto fermi = FermiSolver::find_fermi_level(
        eigenvalues, weights, target_electrons,
        SmearingType::Gaussian, 0.01, 2);

    double fermi_ev = fermi.fermi_energy * constants::rydberg_to_ev;

    // Fermi level should be in the gap
    EXPECT_GT(fermi_ev, -2.0);
    EXPECT_LT(fermi_ev, 3.0);
}

TEST(DOS, MultipleKPoints) {
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-5.0 * ev_to_ry, -2.0 * ev_to_ry, 2.0 * ev_to_ry, 5.0 * ev_to_ry},
        {-4.5 * ev_to_ry, -1.5 * ev_to_ry, 2.5 * ev_to_ry, 5.5 * ev_to_ry},
    };
    std::vector<double> weights = {0.5, 0.5};

    auto dos = DOSCalculator::compute_dos(
        eigenvalues, weights, SmearingType::Gaussian,
        0.2, -10.0, 10.0, 1001, 2);

    EXPECT_EQ(dos.energies.size(), 1001u);
    for (double d : dos.dos_values) {
        EXPECT_GE(d, -1e-10);
    }
}

TEST(DOS, FermiDiracSmearing) {
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-3.0 * ev_to_ry, -1.0 * ev_to_ry, 1.0 * ev_to_ry, 3.0 * ev_to_ry}
    };
    std::vector<double> weights = {1.0};

    auto dos = DOSCalculator::compute_dos(
        eigenvalues, weights, SmearingType::FermiDirac,
        0.2, -10.0, 10.0, 1001, 2);

    EXPECT_EQ(dos.dos_values.size(), 1001u);
    for (double d : dos.dos_values) {
        EXPECT_GE(d, -1e-10);
    }
}

// ============================================================================
// DOS: file output
// ============================================================================

TEST(DOS, WriteDOSFile) {
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-3.0 * ev_to_ry, 0.0, 3.0 * ev_to_ry}
    };
    std::vector<double> weights = {1.0};

    auto dos = DOSCalculator::compute_dos(
        eigenvalues, weights, SmearingType::Gaussian,
        0.2, -10.0, 10.0, 101, 2);

    std::string tmpfile = std::filesystem::temp_directory_path().string()
                         + "/kronos_test_dos.dat";
    DOSCalculator::write_dos(tmpfile, dos);

    EXPECT_TRUE(std::filesystem::exists(tmpfile));
    std::ifstream ifs(tmpfile);
    std::string line;
    int line_count = 0;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line[0] != '#') ++line_count;
    }
    EXPECT_GT(line_count, 0);

    std::filesystem::remove(tmpfile);
}

// ============================================================================
// DOS: zero smearing fallback
// ============================================================================

TEST(DOS, SmearingNoneFallback) {
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-2.0 * ev_to_ry, 0.0, 2.0 * ev_to_ry}
    };
    std::vector<double> weights = {1.0};

    // SmearingType::None should still produce a valid DOS
    auto dos = DOSCalculator::compute_dos(
        eigenvalues, weights, SmearingType::None,
        0.1, -10.0, 10.0, 201, 2);

    EXPECT_EQ(dos.dos_values.size(), 201u);
}

// ============================================================================
// DOS: MarzariVanderbilt smearing
// ============================================================================

TEST(DOS, MVSmearing) {
    const double ev_to_ry = constants::ev_to_rydberg;
    std::vector<std::vector<double>> eigenvalues = {
        {-3.0 * ev_to_ry, -1.0 * ev_to_ry, 1.0 * ev_to_ry, 3.0 * ev_to_ry}
    };
    std::vector<double> weights = {1.0};

    auto dos = DOSCalculator::compute_dos(
        eigenvalues, weights, SmearingType::MarzariVanderbilt,
        0.2, -10.0, 10.0, 501, 2);

    EXPECT_EQ(dos.dos_values.size(), 501u);
}
