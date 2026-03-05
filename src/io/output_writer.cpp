#include "io/output_writer.hpp"
#include "utils/logger.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace kronos {

// Helper: escape a string for JSON (handles quotes and backslashes)
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

std::string OutputWriter::to_json_string(const SCFResult& result,
                                         const Crystal& crystal,
                                         const std::string& calculation_type) {
    std::ostringstream ss;
    ss << std::setprecision(10);

    ss << "{\n";

    // Top-level fields
    ss << "  \"calculation_type\": \"" << json_escape(calculation_type) << "\",\n";
    ss << "  \"converged\": " << (result.converged ? "true" : "false") << ",\n";
    ss << "  \"scf_steps\": " << result.scf_steps << ",\n";
    ss << "  \"total_energy_ry\": " << result.total_energy_ry << ",\n";
    ss << "  \"total_energy_ev\": " << result.total_energy_ev << ",\n";
    ss << "  \"fermi_energy_ev\": " << result.fermi_energy_ev << ",\n";

    // Energy components
    ss << "  \"energy_components\": {\n";
    ss << "    \"kinetic_energy\": " << result.kinetic_energy << ",\n";
    ss << "    \"hartree_energy\": " << result.hartree_energy << ",\n";
    ss << "    \"xc_energy\": " << result.xc_energy << ",\n";
    ss << "    \"local_pp_energy\": " << result.local_pp_energy << ",\n";
    ss << "    \"nonlocal_pp_energy\": " << result.nonlocal_pp_energy << ",\n";
    ss << "    \"ewald_energy\": " << result.ewald_energy << "\n";
    ss << "  },\n";

    // Crystal info
    ss << "  \"crystal\": {\n";
    ss << "    \"num_atoms\": " << crystal.num_atoms() << ",\n";
    ss << "    \"volume_bohr3\": " << crystal.volume() << ",\n";
    ss << "    \"atoms\": [\n";
    for (size_t i = 0; i < crystal.num_atoms(); ++i) {
        const auto& atom = crystal.atom(i);
        ss << "      {\"symbol\": \"" << json_escape(atom.symbol) << "\", "
           << "\"position\": [" << atom.position[0] << ", "
           << atom.position[1] << ", " << atom.position[2] << "]}";
        if (i + 1 < crystal.num_atoms()) ss << ",";
        ss << "\n";
    }
    ss << "    ]\n";
    ss << "  },\n";

    // Forces
    ss << "  \"forces\": [\n";
    for (size_t i = 0; i < result.forces.size(); ++i) {
        ss << "    [" << result.forces[i][0] << ", "
           << result.forces[i][1] << ", "
           << result.forces[i][2] << "]";
        if (i + 1 < result.forces.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ],\n";

    // Eigenvalues per k-point
    ss << "  \"eigenvalues\": [\n";
    for (size_t ik = 0; ik < result.eigenvalues.size(); ++ik) {
        ss << "    {\n";
        ss << "      \"kpoint_index\": " << ik << ",\n";
        ss << "      \"values_ry\": [";
        for (size_t n = 0; n < result.eigenvalues[ik].size(); ++n) {
            if (n > 0) ss << ", ";
            ss << result.eigenvalues[ik][n];
        }
        ss << "]\n";
        ss << "    }";
        if (ik + 1 < result.eigenvalues.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ],\n";

    // Timing breakdown
    ss << "  \"timing\": {\n";
    size_t t_count = 0;
    for (const auto& [name, seconds] : result.timing) {
        ss << "    \"" << json_escape(name) << "\": " << seconds;
        if (++t_count < result.timing.size()) ss << ",";
        ss << "\n";
    }
    ss << "  }\n";

    ss << "}\n";

    return ss.str();
}

void OutputWriter::write_json(const std::string& filepath,
                              const SCFResult& result,
                              const Crystal& crystal,
                              const std::string& calculation_type) {
    std::string json = to_json_string(result, crystal, calculation_type);

    // Atomic write: write to a temporary file, then rename
    std::string tmp_path = filepath + ".tmp";

    {
        std::ofstream ofs(tmp_path);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open output file: " + tmp_path);
        }
        ofs << json;
        ofs.flush();
        if (!ofs.good()) {
            throw std::runtime_error("Failed to write output file: " + tmp_path);
        }
    }

    // Rename for atomic write
    if (std::rename(tmp_path.c_str(), filepath.c_str()) != 0) {
        // Rename failed; try to remove temp and report error
        std::remove(tmp_path.c_str());
        throw std::runtime_error("Failed to rename output file from "
                                 + tmp_path + " to " + filepath);
    }

    Logger::instance().info("output", "JSON output written",
        {{"filepath", filepath}});
}

} // namespace kronos
