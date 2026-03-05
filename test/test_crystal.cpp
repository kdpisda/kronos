// ============================================================================
// KRONOS  test/test_crystal.cpp
// Crystal structure tests: lattice, volume, coordinate transforms, symmetry.
// ============================================================================

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "core/element_data.hpp"

#include <cmath>

using namespace kronos;

// ============================================================================
// Lattice construction
// ============================================================================

TEST(Crystal, CubicLatticeConstruction) {
    auto crystal = test::make_cubic_crystal(5.0, "Si");
    EXPECT_EQ(crystal.num_atoms(), 1u);
    EXPECT_EQ(crystal.atoms()[0].symbol, "Si");
    EXPECT_EQ(crystal.atoms()[0].atomic_number, 14);
}

TEST(Crystal, DiamondLatticeConstruction) {
    auto crystal = test::make_si_diamond_crystal();
    EXPECT_EQ(crystal.num_atoms(), 2u);
    EXPECT_EQ(crystal.atoms()[0].symbol, "Si");
    EXPECT_EQ(crystal.atoms()[1].symbol, "Si");
    EXPECT_DOUBLE_EQ(crystal.atoms()[1].position[0], 0.25);
    EXPECT_DOUBLE_EQ(crystal.atoms()[1].position[1], 0.25);
    EXPECT_DOUBLE_EQ(crystal.atoms()[1].position[2], 0.25);
}

// ============================================================================
// Volume computation
// ============================================================================

TEST(Crystal, CubicVolumeCorrect) {
    double a_ang = 5.0;
    auto crystal = test::make_cubic_crystal(a_ang);
    double a_bohr = a_ang * constants::angstrom_to_bohr;
    double expected_volume = a_bohr * a_bohr * a_bohr;
    EXPECT_NEAR(crystal.volume(), expected_volume, expected_volume * 1e-10);
}

TEST(Crystal, FCCVolumeCorrect) {
    auto crystal = test::make_si_diamond_crystal();
    double a_ang = 5.43;
    double a_bohr = a_ang * constants::angstrom_to_bohr;
    // FCC primitive cell volume = a^3/4
    double expected_volume = a_bohr * a_bohr * a_bohr / 4.0;
    EXPECT_NEAR(crystal.volume(), expected_volume, expected_volume * 1e-10);
}

TEST(Crystal, VolumeIsPositive) {
    auto crystal = test::make_cubic_crystal(3.0);
    EXPECT_GT(crystal.volume(), 0.0);
}

// ============================================================================
// Coordinate transforms
// ============================================================================

TEST(Crystal, FracToCartRoundtrip) {
    auto crystal = test::make_si_diamond_crystal();
    Vec3 frac = {0.25, 0.5, 0.75};
    Vec3 cart = crystal.frac_to_cart(frac);
    Vec3 frac_back = crystal.cart_to_frac(cart);

    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(frac_back[d], frac[d], 1e-12)
            << "Fractional coordinate roundtrip failed for component " << d;
    }
}

TEST(Crystal, CartToFracRoundtrip) {
    auto crystal = test::make_cubic_crystal(5.0);
    double a_bohr = 5.0 * constants::angstrom_to_bohr;
    Vec3 cart = {a_bohr * 0.3, a_bohr * 0.6, a_bohr * 0.1};
    Vec3 frac = crystal.cart_to_frac(cart);
    Vec3 cart_back = crystal.frac_to_cart(frac);

    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(cart_back[d], cart[d], 1e-12)
            << "Cartesian coordinate roundtrip failed for component " << d;
    }
}

TEST(Crystal, OriginMapsToZero) {
    auto crystal = test::make_cubic_crystal(5.0);
    Vec3 frac = {0.0, 0.0, 0.0};
    Vec3 cart = crystal.frac_to_cart(frac);
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(cart[d], 0.0, 1e-15);
    }
}

// ============================================================================
// Multi-atom crystals
// ============================================================================

TEST(Crystal, NaClHasEightAtoms) {
    auto crystal = test::make_nacl_crystal();
    EXPECT_EQ(crystal.num_atoms(), 8u);
}

TEST(Crystal, AtomAccessBoundsChecked) {
    auto crystal = test::make_si_diamond_crystal();
    EXPECT_NO_THROW(crystal.atom(0));
    EXPECT_NO_THROW(crystal.atom(1));
    EXPECT_THROW(crystal.atom(2), std::out_of_range);
}

TEST(Crystal, TotalElectrons) {
    auto crystal = test::make_si_diamond_crystal();
    // Two Si atoms, Z=14 each
    EXPECT_EQ(crystal.total_electrons(), 28);
}

// ============================================================================
// Reciprocal lattice
// ============================================================================

TEST(Crystal, ReciprocalLatticeOrthogonal) {
    auto crystal = test::make_cubic_crystal(5.0);
    auto recip = crystal.reciprocal_lattice();
    auto lat_bohr = crystal.lattice_bohr();

    // a_i . b_j = 2*pi * delta_ij
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double dot = 0.0;
            for (int k = 0; k < 3; ++k) {
                dot += lat_bohr[i][k] * recip[j][k];
            }
            double expected = (i == j) ? constants::two_pi : 0.0;
            EXPECT_NEAR(dot, expected, 1e-10)
                << "a_" << i << " . b_" << j << " should be "
                << expected << ", got " << dot;
        }
    }
}

TEST(Crystal, ReciprocalLatticeForFCC) {
    auto crystal = test::make_si_diamond_crystal();
    auto recip = crystal.reciprocal_lattice();
    auto lat_bohr = crystal.lattice_bohr();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double dot = 0.0;
            for (int k = 0; k < 3; ++k) {
                dot += lat_bohr[i][k] * recip[j][k];
            }
            double expected = (i == j) ? constants::two_pi : 0.0;
            EXPECT_NEAR(dot, expected, 1e-10);
        }
    }
}

// ============================================================================
// Triclinic cell
// ============================================================================

TEST(Crystal, TriclinicCell) {
    // A triclinic cell with no symmetry
    Mat3 lattice = {{{4.0, 0.0, 0.0},
                     {0.5, 5.0, 0.0},
                     {0.3, 0.4, 6.0}}};
    std::vector<Atom> atoms = {{"H", 1, {0.0, 0.0, 0.0}}};
    Crystal crystal(lattice, std::move(atoms));

    EXPECT_GT(crystal.volume(), 0.0);

    // Reciprocal lattice orthogonality
    auto recip = crystal.reciprocal_lattice();
    auto lat_bohr = crystal.lattice_bohr();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double dot = 0.0;
            for (int k = 0; k < 3; ++k) {
                dot += lat_bohr[i][k] * recip[j][k];
            }
            double expected = (i == j) ? constants::two_pi : 0.0;
            EXPECT_NEAR(dot, expected, 1e-10);
        }
    }
}

// ============================================================================
// Lattice in bohr
// ============================================================================

TEST(Crystal, LatticeBohrvAngstrom) {
    double a_ang = 5.43;
    auto crystal = test::make_cubic_crystal(a_ang);
    auto lat_bohr = crystal.lattice_bohr();
    auto lat_ang = crystal.lattice();
    double conversion = constants::angstrom_to_bohr;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(lat_bohr[i][j], lat_ang[i][j] * conversion, 1e-12);
        }
    }
}

// ============================================================================
// Element data
// ============================================================================

TEST(ElementData, AtomicNumberFromSymbol) {
    EXPECT_EQ(atomic_number_from_symbol("H"), 1);
    EXPECT_EQ(atomic_number_from_symbol("Si"), 14);
    EXPECT_EQ(atomic_number_from_symbol("Fe"), 26);
    EXPECT_EQ(atomic_number_from_symbol("Au"), 79);
}

TEST(ElementData, SymbolFromAtomicNumber) {
    EXPECT_EQ(symbol_from_atomic_number(1), "H");
    EXPECT_EQ(symbol_from_atomic_number(14), "Si");
    EXPECT_EQ(symbol_from_atomic_number(26), "Fe");
}

TEST(ElementData, InvalidSymbolThrows) {
    EXPECT_THROW(atomic_number_from_symbol("Xx"), std::exception);
}

TEST(ElementData, InvalidAtomicNumberThrows) {
    EXPECT_THROW(symbol_from_atomic_number(0), std::exception);
    EXPECT_THROW(symbol_from_atomic_number(200), std::exception);
}
