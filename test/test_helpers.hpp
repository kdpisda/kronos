#pragma once
// ============================================================================
// KRONOS  test/test_helpers.hpp
// Shared test fixtures and helper functions.
// ============================================================================

#include "core/types.hpp"
#include "core/constants.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "io/upf_parser.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace kronos::test {

// ============================================================================
// Tolerance constants
// ============================================================================

constexpr double ENERGY_TOL   = 1e-6;   // Ry
constexpr double FORCE_TOL    = 1e-4;   // Ry/bohr
constexpr double FFT_TOL      = 1e-10;  // FFT round-trip
constexpr double TIGHT_TOL    = 1e-12;  // exact math
constexpr double LOOSE_TOL    = 1e-3;   // discretization effects
constexpr double PERCENT_1    = 0.01;   // 1% relative tolerance
constexpr double PERCENT_5    = 0.05;   // 5% relative tolerance

// ============================================================================
// Crystal construction helpers
// ============================================================================

/// Simple cubic crystal with one atom at origin
inline Crystal make_cubic_crystal(double a_ang,
                                  const std::string& element = "Si",
                                  int Z = 14) {
    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};
    std::vector<Atom> atoms = {{element, Z, {0.0, 0.0, 0.0}}};
    return Crystal(lattice, std::move(atoms));
}

/// Si diamond at a=5.43 angstrom with FCC lattice and 2-atom basis
inline Crystal make_si_diamond_crystal() {
    const double a = 5.43;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.00, 0.00, 0.00}},
        {"Si", 14, {0.25, 0.25, 0.25}},
    };
    return Crystal(lattice, std::move(atoms));
}

/// NaCl rocksalt conventional cell (8 atoms: 4 Na + 4 Cl)
inline Crystal make_nacl_crystal(double a_ang = 5.64) {
    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};
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
    return Crystal(lattice, std::move(atoms));
}

// ============================================================================
// Pseudopotential helpers
// ============================================================================

/// Minimal analytic Si pseudopotential with Gaussian local part.
/// V_loc(r) = -Z_val * erf(r / r_loc) / r
inline PseudoPotential make_si_pseudopotential(double z_val = 4.0,
                                                int npts = 500,
                                                double rmax = 10.0) {
    PseudoPotential pp;
    pp.element = "Si";
    pp.atomic_number = 14;
    pp.z_valence = z_val;
    pp.pp_type = "NC";
    pp.is_norm_conserving = true;
    pp.is_ultrasoft = false;
    pp.is_paw = false;
    pp.xc_functional = "LDA_PZ";
    pp.lmax = 0;
    pp.num_projectors = 0;
    pp.num_wfc = 0;

    pp.mesh.npoints = npts;
    pp.mesh.r.resize(npts);
    pp.mesh.rab.resize(npts);
    pp.vloc.resize(npts);

    double dr = rmax / (npts - 1);
    double r_loc = 0.5;

    for (int i = 0; i < npts; ++i) {
        double r = i * dr;
        pp.mesh.r[i] = r;
        pp.mesh.rab[i] = dr;
        if (r < 1e-30) {
            pp.vloc[i] = -z_val * 2.0 / (std::sqrt(constants::pi) * r_loc);
        } else {
            pp.vloc[i] = -z_val * std::erf(r / r_loc) / r;
        }
    }

    // Atomic density: normalized Gaussian
    pp.rho_atomic.resize(npts);
    double norm = 0.0;
    double sigma = 1.0;
    for (int i = 0; i < npts; ++i) {
        double r = pp.mesh.r[i];
        pp.rho_atomic[i] = std::exp(-r * r / (2.0 * sigma * sigma));
        norm += r * r * pp.rho_atomic[i] * pp.mesh.rab[i];
    }
    norm *= constants::four_pi;
    for (int i = 0; i < npts; ++i) {
        pp.rho_atomic[i] *= z_val / norm;
    }

    return pp;
}

/// Map of pseudopotentials for Si-only calculations
inline std::map<std::string, PseudoPotential> make_si_pp_map(double z_val = 4.0) {
    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = make_si_pseudopotential(z_val);
    return pps;
}

/// Trivial empty pseudopotential (no projectors, zero local potential)
inline PseudoPotential make_empty_pp(const std::string& element = "X",
                                      double z_val = 0.0) {
    PseudoPotential pp;
    pp.element = element;
    pp.z_valence = z_val;
    pp.mesh.npoints = 2;
    pp.mesh.r = {0.0, 1.0};
    pp.mesh.rab = {1.0, 1.0};
    pp.vloc = {0.0, 0.0};
    return pp;
}

// ============================================================================
// Basis helpers
// ============================================================================

/// Convenience: build a PlaneWaveBasis from a crystal and cutoff
inline PlaneWaveBasis make_plane_wave_basis(const Crystal& crystal,
                                             double ecutwfc) {
    return PlaneWaveBasis(crystal, ecutwfc);
}

} // namespace kronos::test
