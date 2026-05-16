#pragma once
#include "core/types.hpp"
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>

namespace kronos {

class UPFParseError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Radial grid for pseudopotential data
struct RadialGrid {
    int npoints{0};
    std::vector<double> r;      // radial points
    std::vector<double> rab;    // dr for integration: integral = sum(f*rab)
};

// Beta projector for non-local PP
struct BetaProjector {
    int index{0};
    int angular_momentum{0};     // l quantum number
    int cutoff_index{0};         // index beyond which beta=0
    std::vector<double> values;  // beta(r) on radial grid
};

// Atomic wavefunction (for initial density guess)
struct AtomicWavefunction {
    int angular_momentum{0};
    double occupation{0.0};
    std::string label;
    std::vector<double> values;  // chi(r) on radial grid
};

// PAW augmentation charge Q_ij^l(r)
struct PAWAugmentation {
    int i{0};                      // First projector index
    int j{0};                      // Second projector index
    int l{0};                      // Angular momentum of augmentation
    std::vector<double> qfunc;     // Q_ij^l(r) on radial grid
    double q_integral{0.0};        // ∫ Q_ij(r) dr for overlap
};

// PAW-specific data from UPF file
struct PAWData {
    double core_energy{0.0};                          // One-center core energy (Ry)
    std::vector<std::vector<double>> ae_wfc;          // All-electron partial waves φ_i(r)
    std::vector<std::vector<double>> ps_wfc;          // Pseudo partial waves φ~_i(r)
    std::vector<double> ae_core_charge;               // All-electron core charge density
    std::vector<double> ps_core_charge;               // Pseudo core charge density (NLCC)
    std::vector<PAWAugmentation> augmentation;        // Q_ij^l(r) augmentation charges
    double r_paw{0.0};                                // PAW sphere radius (bohr)
    std::vector<std::vector<double>> ae_vloc;         // AE local potential (optional)
};

// Complete pseudopotential data
struct PseudoPotential {
    // Header info
    std::string element;
    int atomic_number{0};
    double z_valence{0.0};       // number of valence electrons
    std::string pp_type;         // "NC" for norm-conserving, "US" for ultrasoft, "PAW"
    bool is_norm_conserving{false};
    bool is_ultrasoft{false};
    bool is_paw{false};
    std::string xc_functional;
    double total_psenergy{0.0};  // total pseudo-energy
    double wfc_cutoff{0.0};      // suggested wavefunction cutoff (Ry)
    double rho_cutoff{0.0};      // suggested density cutoff (Ry)
    int lmax{0};                  // max angular momentum
    int num_projectors{0};        // number of beta projectors
    int num_wfc{0};               // number of atomic wavefunctions

    // Radial grid
    RadialGrid mesh;

    // Local potential V_loc(r) in Ry
    std::vector<double> vloc;

    // Beta projectors for non-local part
    std::vector<BetaProjector> betas;

    // D_ij matrix (num_projectors x num_projectors)
    // V_NL = sum_ij D_ij |beta_i><beta_j|
    std::vector<std::vector<double>> dij;

    // Atomic charge density rho_atom(r) (for initial guess)
    std::vector<double> rho_atomic;

    // Atomic wavefunctions (for initial density)
    std::vector<AtomicWavefunction> atomic_wfc;

    // PAW data (present only for PAW pseudopotentials)
    std::optional<PAWData> paw;
};

// Parse a UPF v2 pseudopotential file
// Throws UPFParseError on parse failure with file path and description
PseudoPotential parse_upf(const std::string& filepath);

// Validate the loaded PP: check norm conservation, z_valence > 0, etc.
// Throws UPFParseError if validation fails
void validate_pseudopotential(const PseudoPotential& pp);

} // namespace kronos
