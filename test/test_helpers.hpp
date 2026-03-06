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

/// Minimal analytic Si pseudopotential with Coulomb-tailed local part.
/// V_loc(r) = -2*Z_val * erf(r / r_loc) / r   (Rydberg units: tail is -2Z/r)
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

    // Factor of 2 for Rydberg units: V_loc -> -2Z/r at large r
    for (int i = 0; i < npts; ++i) {
        double r = i * dr;
        pp.mesh.r[i] = r;
        pp.mesh.rab[i] = dr;
        if (r < 1e-30) {
            pp.vloc[i] = -2.0 * z_val * 2.0 / (std::sqrt(constants::pi) * r_loc);
        } else {
            pp.vloc[i] = -2.0 * z_val * std::erf(r / r_loc) / r;
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

// ============================================================================
// Nonlocal pseudopotential helpers
// ============================================================================

/// Si PP with one p-type KB projector: β(r) = r·exp(-r²/(2σ²)), D_ij = -2.0 Ry
inline PseudoPotential make_si_pseudopotential_nonlocal(double z_val = 4.0,
                                                         int npts = 500,
                                                         double rmax = 10.0) {
    PseudoPotential pp = make_si_pseudopotential(z_val, npts, rmax);
    pp.lmax = 1;
    pp.num_projectors = 1;

    double sigma = 1.0;
    BetaProjector beta;
    beta.index = 0;
    beta.angular_momentum = 1;  // p-type
    beta.cutoff_index = npts - 1;
    beta.values.resize(npts);
    for (int i = 0; i < npts; ++i) {
        double r = pp.mesh.r[i];
        beta.values[i] = r * std::exp(-r * r / (2.0 * sigma * sigma));
    }
    pp.betas.push_back(beta);

    // D_ij matrix: single element = -2.0 Ry
    pp.dij = {{-2.0}};

    return pp;
}

/// PP map wrapper for nonlocal Si
inline std::map<std::string, PseudoPotential> make_si_pp_map_nonlocal(double z_val = 4.0) {
    std::map<std::string, PseudoPotential> pps;
    pps["Si"] = make_si_pseudopotential_nonlocal(z_val);
    return pps;
}

/// Si diamond with one atom displaced by delta (fractional) along direction dir
inline Crystal make_si_diamond_displaced(double delta, int atom_idx = 0, int dir = 0) {
    const double a = 5.43;
    Mat3 lattice = {{{0, a/2, a/2}, {a/2, 0, a/2}, {a/2, a/2, 0}}};
    std::vector<Atom> atoms = {
        {"Si", 14, {0.00, 0.00, 0.00}},
        {"Si", 14, {0.25, 0.25, 0.25}},
    };
    atoms[atom_idx].position[dir] += delta;
    return Crystal(lattice, std::move(atoms));
}

/// CsCl structure (primitive cell: Cs at origin, Cl at body center)
inline Crystal make_cscl_crystal(double a_ang = 4.12) {
    Mat3 lattice = {{{a_ang, 0, 0}, {0, a_ang, 0}, {0, 0, a_ang}}};
    std::vector<Atom> atoms = {
        {"Cs", 55, {0.0, 0.0, 0.0}},
        {"Cl", 17, {0.5, 0.5, 0.5}},
    };
    return Crystal(lattice, std::move(atoms));
}

/// Minimal H pseudopotential (z_val=1, Coulomb-tailed local, Rydberg units)
inline PseudoPotential make_h_pseudopotential(int npts = 500, double rmax = 10.0) {
    PseudoPotential pp;
    pp.element = "H";
    pp.atomic_number = 1;
    pp.z_valence = 1.0;
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

    // Factor of 2 for Rydberg units: V_loc -> -2Z/r at large r
    for (int i = 0; i < npts; ++i) {
        double r = i * dr;
        pp.mesh.r[i] = r;
        pp.mesh.rab[i] = dr;
        if (r < 1e-30) {
            pp.vloc[i] = -2.0 * 2.0 / (std::sqrt(constants::pi) * r_loc);
        } else {
            pp.vloc[i] = -2.0 * std::erf(r / r_loc) / r;
        }
    }

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
        pp.rho_atomic[i] *= 1.0 / norm;
    }

    return pp;
}

// ============================================================================
// Generic analytic pseudopotential helper
// ============================================================================

/// Generic analytic PP: V_loc(r) = -2*z_val*erf(r/r_loc)/r (Ry units)
inline PseudoPotential make_analytic_pp(const std::string& element, int Z,
                                         double z_val, double r_loc = 0.5,
                                         int npts = 500, double rmax = 10.0) {
    PseudoPotential pp;
    pp.element = element;
    pp.atomic_number = Z;
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
    for (int i = 0; i < npts; ++i) {
        double r = i * dr;
        pp.mesh.r[i] = r;
        pp.mesh.rab[i] = dr;
        if (r < 1e-30) {
            pp.vloc[i] = -2.0 * z_val * 2.0 / (std::sqrt(constants::pi) * r_loc);
        } else {
            pp.vloc[i] = -2.0 * z_val * std::erf(r / r_loc) / r;
        }
    }

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

// ============================================================================
// H2O molecule in a cubic box
// ============================================================================

/// H2O in cubic box: O at center, 2 H displaced (bond ~1.8 bohr, angle ~104.5 deg)
/// box_ang = box side length in Angstrom (default 7.938 Ang = 15 bohr)
inline Crystal make_h2o_crystal(double box_ang = 7.938) {
    // O at (0.5, 0.5, 0.5), H displaced in fractional coords
    // O-H bond ~0.96 Ang = 1.81 bohr; in fractional: 0.96/7.938 ~ 0.121
    // H-O-H angle 104.5 deg: use symmetric displacement in xy plane
    // H1 at (+dx, +dy, 0), H2 at (-dx, +dy, 0) relative to O
    double bond_ang = 0.96;
    double half_angle = 52.25 * constants::pi / 180.0;
    double dx = bond_ang * std::sin(half_angle) / box_ang;
    double dy = bond_ang * std::cos(half_angle) / box_ang;

    Mat3 lattice = {{{box_ang, 0, 0}, {0, box_ang, 0}, {0, 0, box_ang}}};
    std::vector<Atom> atoms = {
        {"O", 8,  {0.5, 0.5, 0.5}},
        {"H", 1,  {0.5 + dx, 0.5 + dy, 0.5}},
        {"H", 1,  {0.5 - dx, 0.5 + dy, 0.5}},
    };
    return Crystal(lattice, std::move(atoms));
}

/// H2O PP map using analytic PPs (toy Gaussian)
/// O uses r_loc=1.0 (soft) to ensure SCF stability in vacuum box
inline std::map<std::string, PseudoPotential> make_h2o_pp_map() {
    std::map<std::string, PseudoPotential> pps;
    pps["O"] = make_analytic_pp("O", 8, 6.0, 1.0);
    pps["H"] = make_h_pseudopotential();
    return pps;
}

/// H2O PP map using real UPF pseudopotentials
inline std::map<std::string, PseudoPotential> make_h2o_pp_map_real() {
    std::map<std::string, PseudoPotential> pps;
    pps["O"] = parse_upf("../pseudopotentials/O.pz-mt.UPF");
    pps["H"] = parse_upf("../pseudopotentials/H.pz-vbc.UPF");
    return pps;
}

// ============================================================================
// MgO rocksalt primitive cell
// ============================================================================

/// MgO rocksalt FCC primitive cell (2 atoms): a = 4.21 Ang
inline Crystal make_mgo_crystal(double a_ang = 4.21) {
    Mat3 lattice = {{{0, a_ang/2, a_ang/2},
                     {a_ang/2, 0, a_ang/2},
                     {a_ang/2, a_ang/2, 0}}};
    std::vector<Atom> atoms = {
        {"Mg", 12, {0.0, 0.0, 0.0}},
        {"O",  8,  {0.5, 0.5, 0.5}},
    };
    return Crystal(lattice, std::move(atoms));
}

/// MgO PP map using analytic PPs (toy Gaussian)
inline std::map<std::string, PseudoPotential> make_mgo_pp_map() {
    std::map<std::string, PseudoPotential> pps;
    pps["Mg"] = make_analytic_pp("Mg", 12, 2.0, 0.6);
    pps["O"]  = make_analytic_pp("O",  8,  6.0, 0.4);
    return pps;
}

/// MgO PP map using real UPF pseudopotentials
inline std::map<std::string, PseudoPotential> make_mgo_pp_map_real() {
    std::map<std::string, PseudoPotential> pps;
    pps["Mg"] = parse_upf("../pseudopotentials/Mg.pz-n-vbc.UPF");
    pps["O"]  = parse_upf("../pseudopotentials/O.pz-mt.UPF");
    return pps;
}

// ============================================================================
// Graphene hexagonal cell with vacuum
// ============================================================================

/// Graphene hexagonal unit cell: a = 2.461 Ang, c = 10.583 Ang (20 bohr vacuum)
/// 2 C atoms at (0,0,0) and (1/3, 2/3, 0)
inline Crystal make_graphene_crystal(double a_ang = 2.461, double c_ang = 10.583) {
    // Hexagonal lattice: a1 = (a, 0, 0), a2 = (-a/2, a*sqrt(3)/2, 0), a3 = (0, 0, c)
    double a2x = -a_ang / 2.0;
    double a2y = a_ang * std::sqrt(3.0) / 2.0;
    Mat3 lattice = {{{a_ang, 0, 0}, {a2x, a2y, 0}, {0, 0, c_ang}}};
    std::vector<Atom> atoms = {
        {"C", 6, {0.0, 0.0, 0.0}},
        {"C", 6, {1.0/3.0, 2.0/3.0, 0.0}},
    };
    return Crystal(lattice, std::move(atoms));
}

/// Graphene PP map using analytic PPs (toy Gaussian)
inline std::map<std::string, PseudoPotential> make_graphene_pp_map() {
    std::map<std::string, PseudoPotential> pps;
    pps["C"] = make_analytic_pp("C", 6, 4.0, 0.4);
    return pps;
}

/// Graphene PP map using real UPF pseudopotentials
inline std::map<std::string, PseudoPotential> make_graphene_pp_map_real() {
    std::map<std::string, PseudoPotential> pps;
    pps["C"] = parse_upf("../pseudopotentials/C.pz-vbc.UPF");
    return pps;
}

// ============================================================================
// Fe BCC primitive cell (spin-polarized)
// ============================================================================

/// Fe BCC primitive cell: a = 2.87 Ang, BCC primitive vectors
/// 1 Fe atom at origin
inline Crystal make_fe_bcc_crystal(double a_ang = 2.87) {
    double a2 = a_ang / 2.0;
    Mat3 lattice = {{{-a2, a2, a2}, {a2, -a2, a2}, {a2, a2, -a2}}};
    std::vector<Atom> atoms = {
        {"Fe", 26, {0.0, 0.0, 0.0}},
    };
    return Crystal(lattice, std::move(atoms));
}

/// Fe BCC PP map using analytic PP (toy Gaussian, Z_val=8 for 3d+4s)
inline std::map<std::string, PseudoPotential> make_fe_bcc_pp_map() {
    std::map<std::string, PseudoPotential> pps;
    pps["Fe"] = make_analytic_pp("Fe", 26, 8.0, 0.5);
    return pps;
}

/// Fe BCC PP map using real UPF pseudopotential
inline std::map<std::string, PseudoPotential> make_fe_bcc_pp_map_real() {
    std::map<std::string, PseudoPotential> pps;
    pps["Fe"] = parse_upf("../pseudopotentials/Fe.pz-hgh.UPF");
    return pps;
}

} // namespace kronos::test
