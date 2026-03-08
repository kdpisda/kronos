#include "io/input_parser.hpp"
#include "core/element_data.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace kronos {

// ============================================================================
// Helpers
// ============================================================================

namespace {

/// Compute the determinant of a 3x3 matrix.
double determinant(const Mat3& m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
         - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
         + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

/// Convert a string to lower-case.
std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

/// Validate that the YAML node contains only the allowed top-level keys.
void check_unknown_keys(const YAML::Node& root,
                        const std::set<std::string>& allowed,
                        const std::string& context) {
    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string key = it->first.as<std::string>();
        if (allowed.find(key) == allowed.end()) {
            throw InputValidationError(
                context + ": unknown key '" + key + "'");
        }
    }
}

// ============================================================================
// Section parsers
// ============================================================================

/// Parse the system section: lattice + atoms.
/// Returns (lattice, atoms) pair.
std::pair<Mat3, std::vector<Atom>> parse_system(const YAML::Node& sys) {
    if (!sys) {
        throw InputValidationError("'system' section is required");
    }
    if (!sys.IsMap()) {
        throw InputValidationError("'system' must be a mapping");
    }

    // -- lattice -----------------------------------------------------------
    if (!sys["lattice"]) {
        throw InputValidationError("system.lattice is required");
    }
    const auto& lat_node = sys["lattice"];
    if (!lat_node.IsSequence() || lat_node.size() != 3) {
        throw InputValidationError(
            "system.lattice must be a 3x3 matrix (sequence of 3 rows)");
    }

    Mat3 lattice{};
    for (int i = 0; i < 3; ++i) {
        const auto& row = lat_node[i];
        if (!row.IsSequence() || row.size() != 3) {
            throw InputValidationError(
                "system.lattice[" + std::to_string(i) +
                "] must be a sequence of 3 numbers");
        }
        for (int j = 0; j < 3; ++j) {
            lattice[i][j] = row[j].as<double>();
        }
    }

    double det = determinant(lattice);
    if (det <= 0.0) {
        throw InputValidationError(
            "system.lattice: determinant must be positive (right-handed), "
            "got " + std::to_string(det));
    }

    // -- atoms -------------------------------------------------------------
    if (!sys["atoms"]) {
        throw InputValidationError("system.atoms is required");
    }
    const auto& atoms_node = sys["atoms"];
    if (!atoms_node.IsSequence() || atoms_node.size() == 0) {
        throw InputValidationError(
            "system.atoms must be a non-empty sequence");
    }

    std::vector<Atom> atoms;
    atoms.reserve(atoms_node.size());

    for (size_t i = 0; i < atoms_node.size(); ++i) {
        const auto& a = atoms_node[i];
        if (!a["symbol"]) {
            throw InputValidationError(
                "system.atoms[" + std::to_string(i) + "].symbol is required");
        }
        if (!a["position"]) {
            throw InputValidationError(
                "system.atoms[" + std::to_string(i) + "].position is required");
        }

        std::string symbol = a["symbol"].as<std::string>();
        int Z = atomic_number_from_symbol(symbol);

        const auto& pos_node = a["position"];
        if (!pos_node.IsSequence() || pos_node.size() != 3) {
            throw InputValidationError(
                "system.atoms[" + std::to_string(i) +
                "].position must be [x, y, z]");
        }

        Vec3 pos{};
        for (int j = 0; j < 3; ++j) {
            pos[j] = pos_node[j].as<double>();
        }

        // Warn if fractional coords are outside [0,1)
        for (int j = 0; j < 3; ++j) {
            if (pos[j] < 0.0 || pos[j] >= 1.0) {
                std::cerr << "Warning: system.atoms[" << i
                          << "].position[" << j << "] = " << pos[j]
                          << " is outside [0, 1) fractional range\n";
            }
        }

        atoms.push_back(Atom{std::move(symbol), Z, pos});
    }

    return {lattice, atoms};
}

/// Parse the calculation section.
CalculationParams parse_calculation(const YAML::Node& calc) {
    CalculationParams params{};

    if (!calc) {
        throw InputValidationError("'calculation' section is required");
    }
    if (!calc.IsMap()) {
        throw InputValidationError("'calculation' must be a mapping");
    }

    // -- type --------------------------------------------------------------
    if (calc["type"]) {
        std::string type_str = to_lower(calc["type"].as<std::string>());
        if (type_str == "scf") {
            params.type = CalculationType::SCF;
        } else if (type_str == "relax") {
            params.type = CalculationType::Relax;
        } else if (type_str == "bands") {
            params.type = CalculationType::Bands;
        } else if (type_str == "vc-relax" || type_str == "vcrelax" || type_str == "vc_relax") {
            params.type = CalculationType::VCRelax;
        } else if (type_str == "dos") {
            params.type = CalculationType::DOS;
        } else {
            throw InputValidationError(
                "calculation.type: must be one of scf, relax, vc-relax, bands, dos; "
                "got '" + type_str + "'");
        }
    }

    // -- ecutwfc -----------------------------------------------------------
    if (!calc["ecutwfc"]) {
        throw InputValidationError("calculation.ecutwfc is required");
    }
    params.ecutwfc = calc["ecutwfc"].as<double>();
    if (params.ecutwfc < 10.0 || params.ecutwfc > 500.0) {
        throw InputValidationError(
            "calculation.ecutwfc: must be in range [10, 500] Ry, got " +
            std::to_string(params.ecutwfc));
    }

    // -- ecutrho -----------------------------------------------------------
    if (calc["ecutrho"]) {
        params.ecutrho = calc["ecutrho"].as<double>();
        if (params.ecutrho < 4.0 * params.ecutwfc) {
            throw InputValidationError(
                "calculation.ecutrho: must be >= 4*ecutwfc (" +
                std::to_string(4.0 * params.ecutwfc) + "), got " +
                std::to_string(params.ecutrho));
        }
    } else {
        // Auto-set to 4 * ecutwfc (norm-conserving default)
        params.ecutrho = 4.0 * params.ecutwfc;
    }

    // -- kpoints -----------------------------------------------------------
    if (calc["kpoints"]) {
        const auto& kp = calc["kpoints"];
        if (!kp.IsSequence() || kp.size() != 6) {
            throw InputValidationError(
                "calculation.kpoints: must be a 6-element array "
                "[nk1, nk2, nk3, sk1, sk2, sk3]");
        }
        for (int i = 0; i < 3; ++i) {
            params.kpoints.grid[i] = kp[i].as<int>();
            if (params.kpoints.grid[i] < 1) {
                throw InputValidationError(
                    "calculation.kpoints: grid dimensions must be >= 1, "
                    "got " + std::to_string(params.kpoints.grid[i]) +
                    " at index " + std::to_string(i));
            }
        }
        for (int i = 0; i < 3; ++i) {
            params.kpoints.shift[i] = kp[3 + i].as<int>();
        }
    }
    // else: defaults to {1,1,1},{0,0,0}

    // -- xc ----------------------------------------------------------------
    if (calc["xc"]) {
        std::string xc = calc["xc"].as<std::string>();
        static const std::set<std::string> valid_xc = {
            "LDA_PZ", "LDA_PW", "PBE", "PBEsol"
        };
        if (valid_xc.find(xc) == valid_xc.end()) {
            throw InputValidationError(
                "calculation.xc: must be one of LDA_PZ, LDA_PW, PBE, PBEsol; "
                "got '" + xc + "'");
        }
        params.xc_functional = xc;
    }

    // -- smearing ----------------------------------------------------------
    if (calc["smearing"]) {
        std::string sm = to_lower(calc["smearing"].as<std::string>());
        if (sm == "none") {
            params.smearing = SmearingType::None;
        } else if (sm == "gaussian") {
            params.smearing = SmearingType::Gaussian;
        } else if (sm == "marzari-vanderbilt") {
            params.smearing = SmearingType::MarzariVanderbilt;
        } else if (sm == "fermi-dirac") {
            params.smearing = SmearingType::FermiDirac;
        } else {
            throw InputValidationError(
                "calculation.smearing: must be one of none, gaussian, "
                "marzari-vanderbilt, fermi-dirac; got '" + sm + "'");
        }
    }

    // -- degauss -----------------------------------------------------------
    if (calc["degauss"]) {
        params.degauss = calc["degauss"].as<double>();
    }

    // -- spin --------------------------------------------------------------
    if (calc["spin"]) {
        params.spin_polarized = calc["spin"].as<bool>();
    }

    // -- nspin -------------------------------------------------------------
    if (calc["nspin"]) {
        params.nspin = calc["nspin"].as<int>();
        if (params.nspin != 1 && params.nspin != 2) {
            throw InputValidationError(
                "calculation.nspin: must be 1 or 2, got " +
                std::to_string(params.nspin));
        }
        if (params.nspin == 2) {
            params.spin_polarized = true;
        }
    } else if (params.spin_polarized) {
        params.nspin = 2;
    }

    // -- starting_magnetization -------------------------------------------
    if (calc["starting_magnetization"]) {
        const auto& mag = calc["starting_magnetization"];
        if (!mag.IsMap()) {
            throw InputValidationError(
                "calculation.starting_magnetization must be a mapping");
        }
        for (auto it = mag.begin(); it != mag.end(); ++it) {
            std::string elem = it->first.as<std::string>();
            double m = it->second.as<double>();
            if (m < -1.0 || m > 1.0) {
                throw InputValidationError(
                    "calculation.starting_magnetization[" + elem +
                    "]: must be in [-1, 1], got " + std::to_string(m));
            }
            params.starting_magnetization[elem] = m;
        }
    }

    // -- checkpoint --------------------------------------------------------
    if (calc["checkpoint_every"]) {
        params.checkpoint_every = calc["checkpoint_every"].as<int>();
        if (params.checkpoint_every < 0) {
            throw InputValidationError(
                "calculation.checkpoint_every: must be >= 0, got " +
                std::to_string(params.checkpoint_every));
        }
    }
    if (calc["checkpoint_file"]) {
        params.checkpoint_file = calc["checkpoint_file"].as<std::string>();
    }
    if (calc["restart"]) {
        params.restart_from_checkpoint = calc["restart"].as<bool>();
    }

    // -- vc-relax parameters -----------------------------------------------
    if (calc["press_target"]) {
        params.press_target = calc["press_target"].as<double>();
    }
    if (calc["cell_factor"]) {
        params.cell_factor = calc["cell_factor"].as<double>();
        if (params.cell_factor <= 0.0) {
            throw InputValidationError(
                "calculation.cell_factor: must be positive, got " +
                std::to_string(params.cell_factor));
        }
    }

    return params;
}

/// Parse the convergence section.
ConvergenceParams parse_convergence(const YAML::Node& conv) {
    ConvergenceParams params{};

    if (!conv) {
        return params;  // all defaults
    }
    if (!conv.IsMap()) {
        throw InputValidationError("'convergence' must be a mapping");
    }

    if (conv["energy"]) {
        params.energy_threshold = conv["energy"].as<double>();
    }
    if (conv["density"]) {
        params.density_threshold = conv["density"].as<double>();
    }
    if (conv["max_scf_steps"]) {
        params.max_scf_steps = conv["max_scf_steps"].as<int>();
    }
    if (conv["force"]) {
        params.force_threshold = conv["force"].as<double>();
    }
    if (conv["stress"]) {
        params.stress_threshold = conv["stress"].as<double>();
    }

    return params;
}

/// Parse the hardware section.
HardwareParams parse_hardware(const YAML::Node& hw) {
    HardwareParams params{};

    if (!hw) {
        return params;  // all defaults
    }
    if (!hw.IsMap()) {
        throw InputValidationError("'hardware' must be a mapping");
    }

    if (hw["use_gpu"]) {
        params.use_gpu = hw["use_gpu"].as<bool>();
    }
    if (hw["gpu_backend"]) {
        params.gpu_backend = hw["gpu_backend"].as<std::string>();
    }
    if (hw["mpi_tasks"]) {
        params.mpi_tasks = hw["mpi_tasks"].as<int>();
    }

    return params;
}

/// Parse pseudopotentials section and validate against atom symbols.
std::map<std::string, std::string>
parse_pseudopotentials(const YAML::Node& pp,
                       const std::vector<Atom>& atoms) {
    std::map<std::string, std::string> result;

    if (!pp) {
        throw InputValidationError("'pseudopotentials' section is required");
    }
    if (!pp.IsMap()) {
        throw InputValidationError("'pseudopotentials' must be a mapping");
    }

    for (auto it = pp.begin(); it != pp.end(); ++it) {
        result[it->first.as<std::string>()] = it->second.as<std::string>();
    }

    // Every atom symbol in the structure must have a PP entry
    std::set<std::string> symbols;
    for (const auto& atom : atoms) {
        symbols.insert(atom.symbol);
    }
    for (const auto& sym : symbols) {
        if (result.find(sym) == result.end()) {
            throw InputValidationError(
                "pseudopotentials: missing entry for element '" + sym + "'");
        }
    }

    return result;
}

/// Parse from a YAML::Node (shared implementation for file and string).
ParsedInput parse_impl(const YAML::Node& root) {
    if (!root.IsMap()) {
        throw InputValidationError("YAML root must be a mapping");
    }

    // Strict schema: reject unknown top-level keys
    static const std::set<std::string> allowed_top_keys = {
        "system", "calculation", "convergence", "hardware", "pseudopotentials"
    };
    check_unknown_keys(root, allowed_top_keys, "top-level");

    // Parse system (lattice + atoms)
    auto [lattice, atoms] = parse_system(root["system"]);

    // Parse calculation parameters
    CalculationParams calc_params = parse_calculation(root["calculation"]);

    // Parse convergence (optional)
    ConvergenceParams conv_params = parse_convergence(root["convergence"]);

    // Parse hardware (optional)
    HardwareParams hw_params = parse_hardware(root["hardware"]);

    // Parse pseudopotentials
    auto pp = parse_pseudopotentials(root["pseudopotentials"], atoms);

    // Build the Crystal object
    Crystal crystal(lattice, std::move(atoms));

    // Assemble InputData
    InputData input{};
    input.calculation = calc_params;
    input.convergence = conv_params;
    input.hardware = hw_params;
    input.pseudopotentials = std::move(pp);

    return ParsedInput{std::move(crystal), std::move(input)};
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

ParsedInput parse_input(const std::string& filepath) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(filepath);
    } catch (const YAML::Exception& e) {
        throw InputValidationError(
            "Failed to parse YAML file '" + filepath + "': " + e.what());
    }
    return parse_impl(root);
}

ParsedInput parse_input_string(const std::string& yaml_content) {
    YAML::Node root;
    try {
        root = YAML::Load(yaml_content);
    } catch (const YAML::Exception& e) {
        throw InputValidationError(
            std::string("Failed to parse YAML string: ") + e.what());
    }
    return parse_impl(root);
}

} // namespace kronos
