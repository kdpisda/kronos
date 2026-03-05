// ============================================================================
// KRONOS  test/test_output.cpp
// Tests for output writing: JSON format, atomic writes.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "io/output_writer.hpp"
#include "solver/scf.hpp"

#include <filesystem>
#include <fstream>
#include <string>

using namespace kronos;

namespace {

// Build a minimal SCFResult for testing
SCFResult make_dummy_scf_result() {
    SCFResult r;
    r.converged = true;
    r.scf_steps = 10;
    r.total_energy_ry = -15.123456;
    r.total_energy_ev = r.total_energy_ry * constants::rydberg_to_ev;
    r.fermi_energy_ev = -3.5;
    r.kinetic_energy = 5.0;
    r.hartree_energy = 3.0;
    r.xc_energy = -4.0;
    r.local_pp_energy = -12.0;
    r.nonlocal_pp_energy = -1.5;
    r.ewald_energy = -5.6;
    r.eigenvalues = {{-0.5, -0.2, 0.3, 0.8}};
    r.forces = {{0.01, -0.02, 0.0}, {-0.01, 0.02, 0.0}};
    return r;
}

} // anonymous namespace

// ============================================================================
// JSON string output
// ============================================================================

TEST(OutputWriter, JSONContainsRequiredFields) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string json = OutputWriter::to_json_string(result, crystal, "scf");

    // Check for required fields
    EXPECT_NE(json.find("total_energy"), std::string::npos)
        << "JSON should contain total_energy";
    EXPECT_NE(json.find("converged"), std::string::npos)
        << "JSON should contain converged";
    EXPECT_NE(json.find("eigenvalues"), std::string::npos)
        << "JSON should contain eigenvalues";
}

TEST(OutputWriter, JSONContainsEnergyComponents) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string json = OutputWriter::to_json_string(result, crystal, "scf");

    EXPECT_NE(json.find("kinetic"), std::string::npos);
    EXPECT_NE(json.find("hartree"), std::string::npos);
    EXPECT_NE(json.find("xc"), std::string::npos);
    EXPECT_NE(json.find("ewald"), std::string::npos);
}

TEST(OutputWriter, JSONIsValid) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string json = OutputWriter::to_json_string(result, crystal, "scf");

    // Basic JSON validation: should start with { and end with }
    ASSERT_FALSE(json.empty());
    // Trim whitespace
    size_t start = json.find_first_not_of(" \t\n\r");
    size_t end = json.find_last_not_of(" \t\n\r");
    ASSERT_NE(start, std::string::npos);
    EXPECT_EQ(json[start], '{');
    EXPECT_EQ(json[end], '}');

    // Balanced braces
    int brace_count = 0;
    int bracket_count = 0;
    for (char c : json) {
        if (c == '{') ++brace_count;
        if (c == '}') --brace_count;
        if (c == '[') ++bracket_count;
        if (c == ']') --bracket_count;
    }
    EXPECT_EQ(brace_count, 0) << "JSON braces are not balanced";
    EXPECT_EQ(bracket_count, 0) << "JSON brackets are not balanced";
}

TEST(OutputWriter, JSONContainsCalculationType) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string json = OutputWriter::to_json_string(result, crystal, "scf");
    EXPECT_NE(json.find("scf"), std::string::npos);
}

// ============================================================================
// File output
// ============================================================================

TEST(OutputWriter, WriteJSONToFile) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string tmpfile = std::filesystem::temp_directory_path().string()
                         + "/kronos_test_output.json";

    OutputWriter::write_json(tmpfile, result, crystal, "scf");

    // File should exist
    EXPECT_TRUE(std::filesystem::exists(tmpfile));

    // File should be non-empty
    auto fsize = std::filesystem::file_size(tmpfile);
    EXPECT_GT(fsize, 0u);

    // File should contain valid JSON
    std::ifstream ifs(tmpfile);
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("total_energy"), std::string::npos);

    std::filesystem::remove(tmpfile);
}

TEST(OutputWriter, AtomicWriteNoPartialOutput) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string tmpfile = std::filesystem::temp_directory_path().string()
                         + "/kronos_test_atomic.json";

    // Write multiple times — file should never be partial
    for (int i = 0; i < 3; ++i) {
        result.scf_steps = i + 1;
        OutputWriter::write_json(tmpfile, result, crystal, "scf");

        // Verify file is valid after each write
        std::ifstream ifs(tmpfile);
        std::string content((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
        ASSERT_FALSE(content.empty());
        size_t start = content.find_first_not_of(" \t\n\r");
        ASSERT_NE(start, std::string::npos);
        EXPECT_EQ(content[start], '{');
    }

    std::filesystem::remove(tmpfile);
}

// ============================================================================
// Non-converged output
// ============================================================================

TEST(OutputWriter, NonConvergedOutput) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();
    result.converged = false;

    std::string json = OutputWriter::to_json_string(result, crystal, "scf");

    // Should indicate non-convergence
    EXPECT_NE(json.find("false"), std::string::npos)
        << "Non-converged result should contain 'false'";
}

// ============================================================================
// Forces in output
// ============================================================================

TEST(OutputWriter, JSONContainsForces) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string json = OutputWriter::to_json_string(result, crystal, "scf");
    EXPECT_NE(json.find("forces"), std::string::npos)
        << "JSON should contain forces";
}

// ============================================================================
// Crystal info in output
// ============================================================================

TEST(OutputWriter, JSONContainsCrystalInfo) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string json = OutputWriter::to_json_string(result, crystal, "scf");

    // Should contain lattice or atom information
    EXPECT_NE(json.find("Si"), std::string::npos)
        << "JSON should contain element symbol";
}

// ============================================================================
// Fermi energy in output
// ============================================================================

TEST(OutputWriter, JSONContainsFermiEnergy) {
    auto crystal = test::make_si_diamond_crystal();
    auto result = make_dummy_scf_result();

    std::string json = OutputWriter::to_json_string(result, crystal, "scf");
    EXPECT_NE(json.find("fermi"), std::string::npos)
        << "JSON should contain Fermi energy";
}
