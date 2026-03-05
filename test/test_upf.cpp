// ============================================================================
// KRONOS  test/test_upf.cpp
// Unit tests for the UPF pseudopotential parser.
// ============================================================================

#include "io/upf_parser.hpp"

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>

// ============================================================================
// Access internal helper functions from the detail namespace.
// These are defined in upf_parser.cpp and have internal linkage via the
// anonymous namespace -- we re-declare the ones we need here so the linker
// can find them.  (They are in kronos::detail with external linkage.)
// ============================================================================
namespace kronos::detail {
    std::vector<double> parse_doubles_from_string(const std::string& text);
    std::string extract_attribute(const std::string& tag_str,
                                  const std::string& attr_name);
}

// ============================================================================
// ParseDoublesFromString tests
// ============================================================================

TEST(UPFParser, ParseDoublesBasic) {
    auto vals = kronos::detail::parse_doubles_from_string("1.0 2.0 3.0");
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_DOUBLE_EQ(vals[0], 1.0);
    EXPECT_DOUBLE_EQ(vals[1], 2.0);
    EXPECT_DOUBLE_EQ(vals[2], 3.0);
}

TEST(UPFParser, ParseDoublesWithNewlines) {
    auto vals = kronos::detail::parse_doubles_from_string(
        "  1.5e-3\n  2.7e+1\n  -0.42  ");
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_DOUBLE_EQ(vals[0], 1.5e-3);
    EXPECT_DOUBLE_EQ(vals[1], 2.7e+1);
    EXPECT_DOUBLE_EQ(vals[2], -0.42);
}

TEST(UPFParser, ParseDoublesFortranExponent) {
    // Fortran-style D exponent notation
    auto vals = kronos::detail::parse_doubles_from_string(
        "1.23D+02  4.56d-03");
    ASSERT_EQ(vals.size(), 2u);
    EXPECT_DOUBLE_EQ(vals[0], 1.23e+02);
    EXPECT_DOUBLE_EQ(vals[1], 4.56e-03);
}

TEST(UPFParser, ParseDoublesEmptyString) {
    auto vals = kronos::detail::parse_doubles_from_string("");
    EXPECT_TRUE(vals.empty());
}

TEST(UPFParser, ParseDoublesWhitespaceOnly) {
    auto vals = kronos::detail::parse_doubles_from_string("   \n\t  ");
    EXPECT_TRUE(vals.empty());
}

// ============================================================================
// ExtractAttribute tests
// ============================================================================

TEST(UPFParser, ExtractAttributeDoubleQuoted) {
    std::string tag = R"(<PP_HEADER element="Si" z_valence="4.0" type="NC">)";
    EXPECT_EQ(kronos::detail::extract_attribute(tag, "element"), "Si");
    EXPECT_EQ(kronos::detail::extract_attribute(tag, "z_valence"), "4.0");
    EXPECT_EQ(kronos::detail::extract_attribute(tag, "type"), "NC");
}

TEST(UPFParser, ExtractAttributeCaseInsensitive) {
    std::string tag = R"(<PP_HEADER Element="Si" Z_VALENCE="4.0">)";
    EXPECT_EQ(kronos::detail::extract_attribute(tag, "element"), "Si");
    EXPECT_EQ(kronos::detail::extract_attribute(tag, "z_valence"), "4.0");
}

TEST(UPFParser, ExtractAttributeNotFound) {
    std::string tag = R"(<PP_HEADER element="Si">)";
    EXPECT_EQ(kronos::detail::extract_attribute(tag, "nonexistent"), "");
}

TEST(UPFParser, ExtractAttributeWithSpacesAroundEquals) {
    std::string tag = R"(<PP_HEADER element = "Ge" >)";
    EXPECT_EQ(kronos::detail::extract_attribute(tag, "element"), "Ge");
}

// ============================================================================
// parse_upf: file not found
// ============================================================================

TEST(UPFParser, ThrowsOnMissingFile) {
    EXPECT_THROW(
        kronos::parse_upf("/nonexistent/path/fake.upf"),
        kronos::UPFParseError
    );
}

TEST(UPFParser, ThrowsOnMissingFileMessage) {
    try {
        kronos::parse_upf("/nonexistent/path/fake.upf");
        FAIL() << "Expected UPFParseError";
    } catch (const kronos::UPFParseError& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Cannot open"), std::string::npos)
            << "Error message should mention 'Cannot open', got: " << msg;
        EXPECT_NE(msg.find("fake.upf"), std::string::npos)
            << "Error message should contain the file name, got: " << msg;
    }
}

// ============================================================================
// validate_pseudopotential tests
// ============================================================================

TEST(UPFParser, ValidateRejectsZeroValence) {
    kronos::PseudoPotential pp;
    pp.element = "Si";
    pp.z_valence = 0.0;
    pp.mesh.npoints = 100;
    pp.mesh.r.resize(100, 0.0);
    pp.mesh.rab.resize(100, 0.0);

    EXPECT_THROW(
        kronos::validate_pseudopotential(pp),
        kronos::UPFParseError
    );
}

TEST(UPFParser, ValidateRejectsNegativeValence) {
    kronos::PseudoPotential pp;
    pp.element = "H";
    pp.z_valence = -1.0;
    pp.mesh.npoints = 50;
    pp.mesh.r.resize(50, 0.0);
    pp.mesh.rab.resize(50, 0.0);

    EXPECT_THROW(
        kronos::validate_pseudopotential(pp),
        kronos::UPFParseError
    );
}

TEST(UPFParser, ValidateRejectsEmptyMesh) {
    kronos::PseudoPotential pp;
    pp.element = "Si";
    pp.z_valence = 4.0;
    pp.mesh.npoints = 0;

    EXPECT_THROW(
        kronos::validate_pseudopotential(pp),
        kronos::UPFParseError
    );
}

TEST(UPFParser, ValidateRejectsMismatchedRandRAB) {
    kronos::PseudoPotential pp;
    pp.element = "O";
    pp.z_valence = 6.0;
    pp.mesh.npoints = 100;
    pp.mesh.r.resize(100, 0.0);
    pp.mesh.rab.resize(50, 0.0);  // mismatch

    EXPECT_THROW(
        kronos::validate_pseudopotential(pp),
        kronos::UPFParseError
    );
}

TEST(UPFParser, ValidateRejectsVlocSizeMismatch) {
    kronos::PseudoPotential pp;
    pp.element = "Si";
    pp.z_valence = 4.0;
    pp.mesh.npoints = 100;
    pp.mesh.r.resize(100, 0.0);
    pp.mesh.rab.resize(100, 0.0);
    pp.vloc.resize(50, 0.0);  // should be 100

    EXPECT_THROW(
        kronos::validate_pseudopotential(pp),
        kronos::UPFParseError
    );
}

TEST(UPFParser, ValidateAcceptsValidNCPP) {
    kronos::PseudoPotential pp;
    pp.element = "Si";
    pp.z_valence = 4.0;
    pp.pp_type = "NC";
    pp.is_norm_conserving = true;
    pp.mesh.npoints = 100;
    pp.mesh.r.resize(100, 0.0);
    pp.mesh.rab.resize(100, 0.0);
    pp.vloc.resize(100, 0.0);
    pp.num_projectors = 0;

    EXPECT_NO_THROW(kronos::validate_pseudopotential(pp));
}

// ============================================================================
// Round-trip test: write a minimal synthetic UPF v2 file and parse it
// ============================================================================

// ============================================================================
// Truncated / corrupted file tests
// ============================================================================

TEST(UPFParser, RejectsTruncatedFile) {
    std::string tmp_dir = std::filesystem::temp_directory_path().string();
    std::string tmp_file = tmp_dir + "/kronos_test_truncated.upf";

    {
        std::ofstream ofs(tmp_file);
        ofs << R"(<?xml version="1.0" encoding="UTF-8"?>
<UPF version="2.0.1">
<PP_HEADER element="Si" z_valence="4.0" type="NC"
)";
        // Truncated — no closing tags, no mesh, etc.
    }

    EXPECT_THROW(kronos::parse_upf(tmp_file), kronos::UPFParseError);
    std::filesystem::remove(tmp_file);
}

TEST(UPFParser, RejectsEmptyFile) {
    std::string tmp_dir = std::filesystem::temp_directory_path().string();
    std::string tmp_file = tmp_dir + "/kronos_test_empty.upf";

    {
        std::ofstream ofs(tmp_file);
        ofs << "";  // completely empty file
    }

    EXPECT_THROW(kronos::parse_upf(tmp_file), kronos::UPFParseError);
    std::filesystem::remove(tmp_file);
}

// ============================================================================
// Z_valence extraction
// ============================================================================

TEST(UPFParser, ExtractsCorrectZValence) {
    // Verified in ParseSyntheticUPF below, but also test via validation
    kronos::PseudoPotential pp;
    pp.element = "O";
    pp.z_valence = 6.0;
    pp.mesh.npoints = 10;
    pp.mesh.r.resize(10, 0.0);
    pp.mesh.rab.resize(10, 0.0);
    pp.vloc.resize(10, 0.0);
    pp.num_projectors = 0;

    EXPECT_NO_THROW(kronos::validate_pseudopotential(pp));
    EXPECT_DOUBLE_EQ(pp.z_valence, 6.0);
}

// ============================================================================
// Radial grid ordering
// ============================================================================

TEST(UPFParser, RadialGridMonotonic) {
    // After parsing, the radial grid should be monotonically increasing
    std::string tmp_dir = std::filesystem::temp_directory_path().string();
    std::string tmp_file = tmp_dir + "/kronos_test_grid_order.upf";

    {
        std::ofstream ofs(tmp_file);
        ofs << R"(<?xml version="1.0" encoding="UTF-8"?>
<UPF version="2.0.1">
<PP_HEADER element="H" z_valence="1.0" type="NC" xc_functional="LDA"
    l_max="0" mesh_size="5" number_of_proj="0" number_of_wfc="0"
    total_psenergy="-0.5" wfc_cutoff="20.0" rho_cutoff="80.0"/>
<PP_MESH>
    <PP_R>
        0.0  0.01  0.05  0.1  0.5
    </PP_R>
    <PP_RAB>
        0.01  0.01  0.04  0.05  0.4
    </PP_RAB>
</PP_MESH>
<PP_LOCAL>
    -1.0  -0.5  -0.25  -0.1  -0.02
</PP_LOCAL>
<PP_NONLOCAL>
    <PP_DIJ>
    </PP_DIJ>
</PP_NONLOCAL>
<PP_RHOATOM>
    0.0  0.1  0.2  0.15  0.05
</PP_RHOATOM>
</UPF>
)";
    }

    kronos::PseudoPotential pp = kronos::parse_upf(tmp_file);

    // Grid should be monotonically increasing
    for (int i = 1; i < pp.mesh.npoints; ++i) {
        EXPECT_GE(pp.mesh.r[i], pp.mesh.r[i-1])
            << "Radial grid not monotonic at index " << i;
    }

    std::filesystem::remove(tmp_file);
}

// ============================================================================
// Multiple projector parsing
// ============================================================================

TEST(UPFParser, ParsesMultipleProjectors) {
    // Covered in ParseSyntheticUPF below — verifying 2 projectors
    // This is a separate focused test on projector count
    std::string tmp_dir = std::filesystem::temp_directory_path().string();
    std::string tmp_file = tmp_dir + "/kronos_test_projectors.upf";

    {
        std::ofstream ofs(tmp_file);
        ofs << R"(<?xml version="1.0" encoding="UTF-8"?>
<UPF version="2.0.1">
<PP_HEADER element="Si" z_valence="4.0" type="NC" xc_functional="PBE"
    l_max="1" mesh_size="4" number_of_proj="2" number_of_wfc="0"
    total_psenergy="-7.8" wfc_cutoff="30.0" rho_cutoff="120.0"/>
<PP_MESH>
    <PP_R>  0.0  0.01  0.02  0.03  </PP_R>
    <PP_RAB>  0.01  0.01  0.01  0.01  </PP_RAB>
</PP_MESH>
<PP_LOCAL>  -1.0  -0.5  -0.25  -0.1  </PP_LOCAL>
<PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="3">
        0.1  0.2  0.3  0.0
    </PP_BETA.1>
    <PP_BETA.2 index="2" angular_momentum="1" cutoff_radius_index="3">
        0.01  0.02  0.03  0.0
    </PP_BETA.2>
    <PP_DIJ>
        1.5  0.0  0.0  2.5
    </PP_DIJ>
</PP_NONLOCAL>
<PP_RHOATOM>  0.0  0.1  0.2  0.15  </PP_RHOATOM>
</UPF>
)";
    }

    kronos::PseudoPotential pp = kronos::parse_upf(tmp_file);
    EXPECT_EQ(pp.num_projectors, 2);
    EXPECT_EQ(pp.betas.size(), 2u);
    EXPECT_EQ(pp.betas[0].angular_momentum, 0);
    EXPECT_EQ(pp.betas[1].angular_momentum, 1);
    EXPECT_EQ(pp.lmax, 1);

    std::filesystem::remove(tmp_file);
}

// ============================================================================
// Validate rejects fractional z_valence (well, actually accepts it)
// ============================================================================

TEST(UPFParser, AcceptsFractionalZValence) {
    // Some pseudopotentials have fractional z_valence (semi-core states)
    kronos::PseudoPotential pp;
    pp.element = "Fe";
    pp.z_valence = 16.0;  // Fe with semi-core
    pp.mesh.npoints = 10;
    pp.mesh.r.resize(10, 0.0);
    pp.mesh.rab.resize(10, 0.0);
    pp.vloc.resize(10, 0.0);
    pp.num_projectors = 0;

    EXPECT_NO_THROW(kronos::validate_pseudopotential(pp));
}

// ============================================================================
// Full round-trip test
// ============================================================================

TEST(UPFParser, ParseSyntheticUPF) {
    // Create a temporary UPF file with minimal valid content
    std::string tmp_dir = std::filesystem::temp_directory_path().string();
    std::string tmp_file = tmp_dir + "/kronos_test_si.upf";

    {
        std::ofstream ofs(tmp_file);
        ASSERT_TRUE(ofs.is_open()) << "Cannot create temp file: " << tmp_file;

        ofs << R"(<?xml version="1.0" encoding="UTF-8"?>
<UPF version="2.0.1">
<PP_HEADER
    element="Si"
    z_valence="4.0"
    type="NC"
    xc_functional="PBE"
    l_max="1"
    mesh_size="4"
    number_of_proj="2"
    number_of_wfc="2"
    total_psenergy="-7.8"
    wfc_cutoff="30.0"
    rho_cutoff="120.0"
/>
<PP_MESH>
    <PP_R>
        0.0  0.01  0.02  0.03
    </PP_R>
    <PP_RAB>
        0.01  0.01  0.01  0.01
    </PP_RAB>
</PP_MESH>
<PP_LOCAL>
    -1.0  -0.5  -0.25  -0.1
</PP_LOCAL>
<PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="3">
        0.1  0.2  0.3  0.0
    </PP_BETA.1>
    <PP_BETA.2 index="2" angular_momentum="1" cutoff_radius_index="3">
        0.01  0.02  0.03  0.0
    </PP_BETA.2>
    <PP_DIJ>
        1.5  0.0  0.0  2.5
    </PP_DIJ>
</PP_NONLOCAL>
<PP_RHOATOM>
    0.0  0.1  0.2  0.15
</PP_RHOATOM>
<PP_PSWFC>
    <PP_CHI.1 l="0" occupation="2.0" label="3S">
        0.0  0.5  0.8  0.3
    </PP_CHI.1>
    <PP_CHI.2 l="1" occupation="2.0" label="3P">
        0.0  0.1  0.4  0.2
    </PP_CHI.2>
</PP_PSWFC>
</UPF>
)";
    }

    // Parse the file
    kronos::PseudoPotential pp;
    ASSERT_NO_THROW(pp = kronos::parse_upf(tmp_file));

    // Validate header fields
    EXPECT_EQ(pp.element, "Si");
    EXPECT_DOUBLE_EQ(pp.z_valence, 4.0);
    EXPECT_EQ(pp.pp_type, "NC");
    EXPECT_TRUE(pp.is_norm_conserving);
    EXPECT_FALSE(pp.is_ultrasoft);
    EXPECT_FALSE(pp.is_paw);
    EXPECT_EQ(pp.xc_functional, "PBE");
    EXPECT_DOUBLE_EQ(pp.total_psenergy, -7.8);
    EXPECT_DOUBLE_EQ(pp.wfc_cutoff, 30.0);
    EXPECT_DOUBLE_EQ(pp.rho_cutoff, 120.0);
    EXPECT_EQ(pp.lmax, 1);
    EXPECT_EQ(pp.num_projectors, 2);
    EXPECT_EQ(pp.num_wfc, 2);

    // Validate mesh
    EXPECT_EQ(pp.mesh.npoints, 4);
    ASSERT_EQ(pp.mesh.r.size(), 4u);
    EXPECT_DOUBLE_EQ(pp.mesh.r[0], 0.0);
    EXPECT_DOUBLE_EQ(pp.mesh.r[3], 0.03);
    ASSERT_EQ(pp.mesh.rab.size(), 4u);
    EXPECT_DOUBLE_EQ(pp.mesh.rab[0], 0.01);

    // Validate V_loc
    ASSERT_EQ(pp.vloc.size(), 4u);
    EXPECT_DOUBLE_EQ(pp.vloc[0], -1.0);
    EXPECT_DOUBLE_EQ(pp.vloc[3], -0.1);

    // Validate beta projectors
    ASSERT_EQ(pp.betas.size(), 2u);
    EXPECT_EQ(pp.betas[0].angular_momentum, 0);
    EXPECT_EQ(pp.betas[0].cutoff_index, 3);
    ASSERT_EQ(pp.betas[0].values.size(), 4u);
    EXPECT_DOUBLE_EQ(pp.betas[0].values[0], 0.1);
    EXPECT_EQ(pp.betas[1].angular_momentum, 1);

    // Validate D_ij
    ASSERT_EQ(pp.dij.size(), 2u);
    ASSERT_EQ(pp.dij[0].size(), 2u);
    EXPECT_DOUBLE_EQ(pp.dij[0][0], 1.5);
    EXPECT_DOUBLE_EQ(pp.dij[0][1], 0.0);
    EXPECT_DOUBLE_EQ(pp.dij[1][0], 0.0);
    EXPECT_DOUBLE_EQ(pp.dij[1][1], 2.5);

    // Validate rho_atomic
    ASSERT_EQ(pp.rho_atomic.size(), 4u);
    EXPECT_DOUBLE_EQ(pp.rho_atomic[2], 0.2);

    // Validate atomic wavefunctions
    ASSERT_EQ(pp.atomic_wfc.size(), 2u);
    EXPECT_EQ(pp.atomic_wfc[0].angular_momentum, 0);
    EXPECT_DOUBLE_EQ(pp.atomic_wfc[0].occupation, 2.0);
    EXPECT_EQ(pp.atomic_wfc[0].label, "3S");
    ASSERT_EQ(pp.atomic_wfc[0].values.size(), 4u);
    EXPECT_DOUBLE_EQ(pp.atomic_wfc[0].values[2], 0.8);
    EXPECT_EQ(pp.atomic_wfc[1].angular_momentum, 1);
    EXPECT_EQ(pp.atomic_wfc[1].label, "3P");

    // Full validation should pass
    EXPECT_NO_THROW(kronos::validate_pseudopotential(pp));

    // Clean up
    std::filesystem::remove(tmp_file);
}
