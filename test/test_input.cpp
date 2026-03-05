#include <gtest/gtest.h>
#include "io/input_parser.hpp"

using namespace kronos;

// ---------------------------------------------------------------------------
// Full Si bulk example
// ---------------------------------------------------------------------------
TEST(InputParser, ParsesSiBulk) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
    - {symbol: Si, position: [0.25, 0.25, 0.25]}
calculation:
  type: scf
  ecutwfc: 60.0
  ecutrho: 240.0
  kpoints: [8, 8, 8, 0, 0, 0]
  xc: PBE
  smearing: marzari-vanderbilt
  degauss: 0.01
  spin: false
pseudopotentials:
  Si: Si.ONCVPSP.upf
convergence:
  energy: 1e-8
  density: 1e-9
  max_scf_steps: 100
hardware:
  use_gpu: true
  gpu_backend: cuda
  mpi_tasks: 4
)";

    ParsedInput parsed = parse_input_string(yaml);

    // Crystal checks
    EXPECT_EQ(parsed.crystal.num_atoms(), 2u);
    EXPECT_EQ(parsed.crystal.atoms()[0].symbol, "Si");
    EXPECT_EQ(parsed.crystal.atoms()[0].atomic_number, 14);
    EXPECT_EQ(parsed.crystal.atoms()[1].symbol, "Si");
    EXPECT_DOUBLE_EQ(parsed.crystal.atoms()[1].position[0], 0.25);

    // Calculation checks
    EXPECT_EQ(parsed.input.calculation.type, CalculationType::SCF);
    EXPECT_DOUBLE_EQ(parsed.input.calculation.ecutwfc, 60.0);
    EXPECT_DOUBLE_EQ(parsed.input.calculation.ecutrho, 240.0);
    EXPECT_EQ(parsed.input.calculation.kpoints.grid[0], 8);
    EXPECT_EQ(parsed.input.calculation.kpoints.grid[1], 8);
    EXPECT_EQ(parsed.input.calculation.kpoints.grid[2], 8);
    EXPECT_EQ(parsed.input.calculation.xc_functional, "PBE");
    EXPECT_EQ(parsed.input.calculation.smearing, SmearingType::MarzariVanderbilt);
    EXPECT_DOUBLE_EQ(parsed.input.calculation.degauss, 0.01);
    EXPECT_FALSE(parsed.input.calculation.spin_polarized);

    // Pseudopotentials
    EXPECT_EQ(parsed.input.pseudopotentials.at("Si"), "Si.ONCVPSP.upf");

    // Convergence
    EXPECT_DOUBLE_EQ(parsed.input.convergence.energy_threshold, 1e-8);
    EXPECT_DOUBLE_EQ(parsed.input.convergence.density_threshold, 1e-9);
    EXPECT_EQ(parsed.input.convergence.max_scf_steps, 100);

    // Hardware
    EXPECT_TRUE(parsed.input.hardware.use_gpu);
    EXPECT_EQ(parsed.input.hardware.gpu_backend, "cuda");
    EXPECT_EQ(parsed.input.hardware.mpi_tasks, 4);
}

// ---------------------------------------------------------------------------
// ecutwfc below minimum
// ---------------------------------------------------------------------------
TEST(InputParser, RejectsLowCutoff) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 5.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";

    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

// ---------------------------------------------------------------------------
// Negative lattice determinant (left-handed)
// ---------------------------------------------------------------------------
TEST(InputParser, RejectsNegativeDeterminant) {
    // Swap first two lattice vectors to make determinant negative
    const char* yaml = R"(
system:
  lattice: [[0, 5.43, 0], [5.43, 0, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";

    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

// ---------------------------------------------------------------------------
// ecutrho auto-set to 4*ecutwfc when omitted
// ---------------------------------------------------------------------------
TEST(InputParser, AutoSetsEcutrho) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 40.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";

    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_DOUBLE_EQ(parsed.input.calculation.ecutrho, 160.0);
}

// ---------------------------------------------------------------------------
// Unknown top-level key
// ---------------------------------------------------------------------------
TEST(InputParser, RejectsUnknownKeys) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
bogus_section:
  foo: bar
)";

    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

// ============================================================================
// ecutrho validation: must be >= 4 * ecutwfc
// ============================================================================

TEST(InputParser, RejectsEcutrhoTooSmall) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  ecutrho: 200.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    // ecutrho (200) < 4 * ecutwfc (240)
    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

// ============================================================================
// ecutwfc out of range: too high
// ============================================================================

TEST(InputParser, RejectsEcutwfcTooHigh) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 600.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    // ecutwfc > 500 Ry
    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

// ============================================================================
// Missing required fields
// ============================================================================

TEST(InputParser, RejectsMissingSystem) {
    const char* yaml = R"(
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

TEST(InputParser, RejectsMissingCalculation) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
pseudopotentials:
  Si: Si.upf
)";
    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

TEST(InputParser, RejectsMissingAtoms) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

TEST(InputParser, RejectsMissingLattice) {
    const char* yaml = R"(
system:
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

TEST(InputParser, RejectsMissingEcutwfc) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

// ============================================================================
// Edge cases: empty / malformed YAML
// ============================================================================

TEST(InputParser, RejectsEmptyInput) {
    EXPECT_THROW(parse_input_string(""), InputValidationError);
}

TEST(InputParser, RejectsMalformedYAML) {
    const char* yaml = R"(
system:
  lattice: [[[invalid
)";
    EXPECT_THROW(parse_input_string(yaml), std::exception);
}

// ============================================================================
// Calculation types
// ============================================================================

TEST(InputParser, ParsesRelaxType) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: relax
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.input.calculation.type, CalculationType::Relax);
}

TEST(InputParser, ParsesBandsType) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: bands
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.input.calculation.type, CalculationType::Bands);
}

TEST(InputParser, ParsesDOSType) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: dos
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.input.calculation.type, CalculationType::DOS);
}

// ============================================================================
// Smearing types
// ============================================================================

TEST(InputParser, ParsesGaussianSmearing) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
  smearing: gaussian
  degauss: 0.02
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.input.calculation.smearing, SmearingType::Gaussian);
    EXPECT_DOUBLE_EQ(parsed.input.calculation.degauss, 0.02);
}

TEST(InputParser, ParsesFermiDiracSmearing) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
  smearing: fermi-dirac
  degauss: 0.03
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.input.calculation.smearing, SmearingType::FermiDirac);
}

// ============================================================================
// XC functional strings
// ============================================================================

TEST(InputParser, ParsesLDAFunctional) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: LDA_PZ
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.input.calculation.xc_functional, "LDA_PZ");
}

// ============================================================================
// K-point grid parsing
// ============================================================================

TEST(InputParser, ParsesKPointsWithShift) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
  kpoints: [4, 4, 4, 1, 1, 1]
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.input.calculation.kpoints.grid[0], 4);
    EXPECT_EQ(parsed.input.calculation.kpoints.grid[1], 4);
    EXPECT_EQ(parsed.input.calculation.kpoints.grid[2], 4);
    EXPECT_EQ(parsed.input.calculation.kpoints.shift[0], 1);
    EXPECT_EQ(parsed.input.calculation.kpoints.shift[1], 1);
    EXPECT_EQ(parsed.input.calculation.kpoints.shift[2], 1);
}

// ============================================================================
// Spin polarization
// ============================================================================

TEST(InputParser, ParsesSpinTrue) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
  spin: true
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_TRUE(parsed.input.calculation.spin_polarized);
}

// ============================================================================
// Convergence parameters
// ============================================================================

TEST(InputParser, ParsesConvergenceParams) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
convergence:
  energy: 1e-6
  density: 1e-7
  max_scf_steps: 50
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_DOUBLE_EQ(parsed.input.convergence.energy_threshold, 1e-6);
    EXPECT_DOUBLE_EQ(parsed.input.convergence.density_threshold, 1e-7);
    EXPECT_EQ(parsed.input.convergence.max_scf_steps, 50);
}

// ============================================================================
// Unknown calculation subkey
// ============================================================================

TEST(InputParser, RejectsUnknownTopLevelKey2) {
    // Additional test: unknown top-level key with different name
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
extra_section:
  value: 123
)";
    EXPECT_THROW(parse_input_string(yaml), InputValidationError);
}

// ============================================================================
// Multiple atoms in input
// ============================================================================

TEST(InputParser, ParsesMultipleAtoms) {
    const char* yaml = R"(
system:
  lattice: [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]
  atoms:
    - {symbol: Na, position: [0.0, 0.0, 0.0]}
    - {symbol: Cl, position: [0.5, 0.5, 0.5]}
calculation:
  type: scf
  ecutwfc: 40.0
  xc: LDA_PZ
pseudopotentials:
  Na: Na.upf
  Cl: Cl.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.crystal.num_atoms(), 2u);
    EXPECT_EQ(parsed.crystal.atoms()[0].symbol, "Na");
    EXPECT_EQ(parsed.crystal.atoms()[1].symbol, "Cl");
}

// ============================================================================
// Hardware defaults
// ============================================================================

TEST(InputParser, DefaultHardwareParams) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 60.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    // Without explicit hardware section, defaults apply
    EXPECT_FALSE(parsed.input.hardware.use_gpu);
    EXPECT_EQ(parsed.input.hardware.mpi_tasks, 1);
}

// ============================================================================
// Exact ecutrho boundary
// ============================================================================

TEST(InputParser, AcceptsEcutrhoExact4x) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 30.0
  ecutrho: 120.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    // ecutrho = 4 * ecutwfc exactly — should be accepted
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_DOUBLE_EQ(parsed.input.calculation.ecutrho, 120.0);
}

// ============================================================================
// Large ecutrho (PAW-like)
// ============================================================================

TEST(InputParser, AcceptsLargeEcutrho) {
    const char* yaml = R"(
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 30.0
  ecutrho: 360.0
  xc: PBE
pseudopotentials:
  Si: Si.upf
)";
    // ecutrho = 12 * ecutwfc (PAW), should be fine
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_DOUBLE_EQ(parsed.input.calculation.ecutrho, 360.0);
}

// ============================================================================
// Multiple pseudopotential species
// ============================================================================

TEST(InputParser, ParsesMultiplePseudopotentials) {
    const char* yaml = R"(
system:
  lattice: [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]
  atoms:
    - {symbol: Na, position: [0.0, 0.0, 0.0]}
    - {symbol: Cl, position: [0.5, 0.5, 0.5]}
calculation:
  type: scf
  ecutwfc: 40.0
  xc: PBE
pseudopotentials:
  Na: Na.ONCVPSP.upf
  Cl: Cl.ONCVPSP.upf
)";
    ParsedInput parsed = parse_input_string(yaml);
    EXPECT_EQ(parsed.input.pseudopotentials.at("Na"), "Na.ONCVPSP.upf");
    EXPECT_EQ(parsed.input.pseudopotentials.at("Cl"), "Cl.ONCVPSP.upf");
}
