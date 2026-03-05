#pragma once

#include <numbers>

namespace kronos::constants {

// --------------------------------------------------------------------------
// Fundamental conversion factors (CODATA 2018)
// --------------------------------------------------------------------------

/// Bohr radius in angstrom
constexpr double bohr_to_angstrom = 0.529177249;

/// Angstrom to bohr conversion
constexpr double angstrom_to_bohr = 1.0 / bohr_to_angstrom;

/// 1 Hartree in eV
constexpr double hartree_to_ev = 27.211386245988;

/// 1 Rydberg in eV  (1 Ry = 0.5 Hartree)
constexpr double rydberg_to_ev = 13.605693122994;

/// 1 eV in Rydberg
constexpr double ev_to_rydberg = 1.0 / rydberg_to_ev;

/// 1 eV in Hartree
constexpr double ev_to_hartree = 1.0 / hartree_to_ev;

// --------------------------------------------------------------------------
// Mathematical constants
// --------------------------------------------------------------------------

/// pi (from C++20 <numbers>)
constexpr double pi = std::numbers::pi;

/// 2 * pi
constexpr double two_pi = 2.0 * std::numbers::pi;

/// 4 * pi
constexpr double four_pi = 4.0 * std::numbers::pi;

/// sqrt(2)
constexpr double sqrt2 = std::numbers::sqrt2;

// --------------------------------------------------------------------------
// Physical constants in atomic units (Hartree atomic units)
// --------------------------------------------------------------------------

/// Electron mass in atomic units (= 1 by definition)
constexpr double electron_mass_au = 1.0;

/// Electron charge in atomic units (= 1 by definition)
constexpr double electron_charge_au = 1.0;

/// Reduced Planck constant in atomic units (= 1 by definition)
constexpr double hbar_au = 1.0;

/// Boltzmann constant in Hartree/K
constexpr double kboltzmann_hartree_per_K = 3.1668115634556e-6;

/// Boltzmann constant in Ry/K  (1 Ry = 0.5 Ha => k_B in Ry = 2 * k_B in Ha)
constexpr double kboltzmann_ry_per_K = 2.0 * kboltzmann_hartree_per_K;

/// Speed of light in atomic units (= 1/alpha ~ 137.036)
constexpr double speed_of_light_au = 137.035999084;

/// Fine structure constant
constexpr double fine_structure_constant = 1.0 / speed_of_light_au;

/// Proton-to-electron mass ratio
constexpr double proton_electron_mass_ratio = 1836.15267343;

// --------------------------------------------------------------------------
// Derived / convenience
// --------------------------------------------------------------------------

/// Rydberg in Hartree
constexpr double rydberg_to_hartree = 0.5;

/// Hartree in Rydberg
constexpr double hartree_to_rydberg = 2.0;

/// 1 Bohr in metres
constexpr double bohr_to_meter = 0.529177249e-10;

/// 1 Angstrom in metres
constexpr double angstrom_to_meter = 1.0e-10;

} // namespace kronos::constants
