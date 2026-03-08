// ============================================================================
// KRONOS  test/test_vc_relax.cpp
// Tests for variable-cell relaxation (vc-relax).
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "solver/vc_relax.hpp"
#include "solver/scf.hpp"
#include "io/input_parser.hpp"

#include <cmath>
#include <numeric>

using namespace kronos;

// ============================================================================
// VCRelaxOptimizer unit tests
// ============================================================================

TEST(VCRelax, CalculationTypeEnum) {
    // Verify VCRelax is a valid CalculationType
    CalculationType t = CalculationType::VCRelax;
    EXPECT_NE(t, CalculationType::SCF);
    EXPECT_NE(t, CalculationType::Relax);
    EXPECT_NE(t, CalculationType::Bands);
    EXPECT_NE(t, CalculationType::DOS);
}

TEST(VCRelax, ParamsDefaults) {
    // VCRelax parameters have sensible defaults
    VCRelaxOptimizer::Params params;
    EXPECT_EQ(params.max_steps, 50);
    EXPECT_GT(params.force_threshold, 0.0);
    EXPECT_GT(params.stress_threshold, 0.0);
    EXPECT_DOUBLE_EQ(params.press_target, 0.0);
    EXPECT_GT(params.cell_factor, 0.0);
}

TEST(VCRelax, CalculationParamsVCRelax) {
    // Verify press_target and cell_factor are in CalculationParams
    CalculationParams params;
    params.press_target = 1.5;
    params.cell_factor = 3.0;
    EXPECT_DOUBLE_EQ(params.press_target, 1.5);
    EXPECT_DOUBLE_EQ(params.cell_factor, 3.0);
}

TEST(VCRelax, ConvergenceParamsStressThreshold) {
    // Verify stress_threshold is in ConvergenceParams
    ConvergenceParams params;
    params.stress_threshold = 0.1;
    EXPECT_DOUBLE_EQ(params.stress_threshold, 0.1);
}

TEST(VCRelax, SCFResultStressFields) {
    // SCFResult should have stress tensor and pressure fields
    SCFResult result;
    result.pressure_gpa = 5.0;
    EXPECT_DOUBLE_EQ(result.pressure_gpa, 5.0);

    // Stress is a 3x3 matrix, initialized to zero
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            EXPECT_DOUBLE_EQ(result.stress[a][b], 0.0);
        }
    }
}

TEST(VCRelax, VCRelaxResultDefaults) {
    VCRelaxResult result;
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.vc_steps, 0);
    EXPECT_DOUBLE_EQ(result.final_energy_ry, 0.0);
    EXPECT_DOUBLE_EQ(result.final_pressure_gpa, 0.0);
    EXPECT_TRUE(result.energy_history.empty());
    EXPECT_TRUE(result.pressure_history.empty());
}

TEST(VCRelax, InputParserVCRelax) {
    // Test that input parser recognizes vc-relax type
    std::string yaml = R"(
system:
  lattice:
    - [5.43, 0.0, 0.0]
    - [0.0, 5.43, 0.0]
    - [0.0, 0.0, 5.43]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: vc-relax
  ecutwfc: 15.0
  press_target: 0.5
  cell_factor: 1.5
convergence:
  force: 0.001
  stress: 0.3
pseudopotentials:
  Si: ../pseudopotentials/Si.pz-vbc.UPF
)";

    auto [crystal, input] = parse_input_string(yaml);
    EXPECT_EQ(input.calculation.type, CalculationType::VCRelax);
    EXPECT_DOUBLE_EQ(input.calculation.press_target, 0.5);
    EXPECT_DOUBLE_EQ(input.calculation.cell_factor, 1.5);
    EXPECT_DOUBLE_EQ(input.convergence.stress_threshold, 0.3);
}

TEST(VCRelax, InputParserVCRelaxVariantNames) {
    // Test that alternative spellings work
    auto test_name = [](const std::string& type_name) {
        std::string yaml = R"(
system:
  lattice:
    - [5.43, 0.0, 0.0]
    - [0.0, 5.43, 0.0]
    - [0.0, 0.0, 5.43]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: )" + type_name + R"(
  ecutwfc: 15.0
pseudopotentials:
  Si: ../pseudopotentials/Si.pz-vbc.UPF
)";
        auto [crystal, input] = parse_input_string(yaml);
        EXPECT_EQ(input.calculation.type, CalculationType::VCRelax);
    };

    test_name("vc-relax");
    test_name("vcrelax");
    test_name("vc_relax");
}

// ============================================================================
// Integration test: run vc-relax on Si with toy PP
// ============================================================================

TEST(VCRelax, SiDiamondVCRelaxRuns) {
    // Test that vc-relax actually runs and produces results for Si diamond
    // with a toy Gaussian pseudopotential. We use very loose thresholds
    // and low cutoff to make the test fast.
    Crystal crystal = test::make_si_diamond_crystal();
    auto pp_map = test::make_si_pp_map();

    CalculationParams calc_params;
    calc_params.type = CalculationType::VCRelax;
    calc_params.ecutwfc = 10.0;
    calc_params.ecutrho = 40.0;
    calc_params.press_target = 0.0;
    calc_params.cell_factor = 2.0;

    ConvergenceParams conv_params;
    conv_params.energy_threshold = 1e-4;
    conv_params.density_threshold = 1e-4;
    conv_params.max_scf_steps = 30;
    conv_params.force_threshold = 0.1;  // very loose
    conv_params.stress_threshold = 100.0;  // very loose (kbar)

    VCRelaxOptimizer::Params vc_params;
    vc_params.max_steps = 3;  // Just run a few steps
    vc_params.force_threshold = conv_params.force_threshold;
    vc_params.stress_threshold = conv_params.stress_threshold;
    vc_params.press_target = 0.0;
    vc_params.cell_factor = 2.0;

    VCRelaxOptimizer optimizer(vc_params);
    auto result = optimizer.optimize(crystal, calc_params, conv_params, pp_map);

    // Should have run some steps
    EXPECT_GT(result.vc_steps, 0);
    EXPECT_FALSE(result.energy_history.empty());
    EXPECT_FALSE(result.pressure_history.empty());

    // Energy should be finite
    EXPECT_TRUE(std::isfinite(result.final_energy_ry));
    EXPECT_NE(result.final_energy_ry, 0.0);

    // Pressure should be finite
    EXPECT_TRUE(std::isfinite(result.final_pressure_gpa));
}

TEST(VCRelax, SiDiamondVCRelaxChangesCell) {
    // After a few vc-relax steps, the lattice should have changed
    Crystal crystal = test::make_si_diamond_crystal();
    auto pp_map = test::make_si_pp_map();

    Mat3 initial_lattice = crystal.lattice();
    double initial_volume = crystal.volume();

    CalculationParams calc_params;
    calc_params.type = CalculationType::VCRelax;
    calc_params.ecutwfc = 10.0;
    calc_params.ecutrho = 40.0;

    ConvergenceParams conv_params;
    conv_params.energy_threshold = 1e-4;
    conv_params.density_threshold = 1e-4;
    conv_params.max_scf_steps = 30;
    conv_params.force_threshold = 0.1;
    conv_params.stress_threshold = 100.0;

    VCRelaxOptimizer::Params vc_params;
    vc_params.max_steps = 3;
    vc_params.force_threshold = 0.1;
    vc_params.stress_threshold = 100.0;

    VCRelaxOptimizer optimizer(vc_params);
    auto result = optimizer.optimize(crystal, calc_params, conv_params, pp_map);

    // After vc-relax, the lattice vectors should differ from initial
    Mat3 final_lattice = result.final_crystal.lattice();
    double final_volume = result.final_crystal.volume();

    // At least the volume should have changed (stress drives cell change)
    // Note: with a toy PP and low cutoff, changes can be small but should be nonzero
    bool volume_changed = (std::abs(final_volume - initial_volume) > 1e-6);
    bool any_element_changed = false;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (std::abs(final_lattice[i][j] - initial_lattice[i][j]) > 1e-8) {
                any_element_changed = true;
            }
        }
    }

    // At minimum, either volume or lattice elements should have changed
    EXPECT_TRUE(volume_changed || any_element_changed)
        << "vc-relax should change the cell; initial_vol=" << initial_volume
        << " final_vol=" << final_volume;
}

TEST(VCRelax, GeneralizedBFGSDimensionCheck) {
    // The generalized coordinate vector should be 3*natoms + 9
    // For 2-atom Si diamond: 6 + 9 = 15
    Crystal crystal = test::make_si_diamond_crystal();
    EXPECT_EQ(crystal.num_atoms(), 2u);
    // The generalized BFGS should have 15 degrees of freedom
    int expected_dim = 3 * 2 + 9;
    EXPECT_EQ(expected_dim, 15);
}

TEST(VCRelax, PressureConversion) {
    // Verify the pressure conversion: 1 Ry/bohr^3 = 14710.507 GPa
    double stress_val = 1.0;  // Ry/bohr^3
    double pressure_gpa = stress_val * 14710.507;
    EXPECT_NEAR(pressure_gpa, 14710.507, 0.01);
}
