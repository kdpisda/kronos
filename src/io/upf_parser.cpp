// ============================================================================
// KRONOS  src/io/upf_parser.cpp
// Lightweight UPF v2 pseudopotential parser (XML-based, no external XML lib).
// ============================================================================

#include "io/upf_parser.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace kronos {

// ============================================================================
// Internal helper utilities
// ============================================================================
namespace detail {

/// Read an entire file into a string. Throws UPFParseError on failure.
static std::string read_file(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        throw UPFParseError("Cannot open UPF file: " + filepath);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

/// Trim leading and trailing whitespace from a string.
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return {};
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

/// Convert a string to lower case (for case-insensitive comparisons).
static std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

/// Parse a sequence of whitespace-separated doubles from a string.
/// Public so tests can reach it via the free-function wrapper below.
std::vector<double> parse_doubles_from_string(const std::string& text) {
    std::vector<double> values;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        // Fortran-style D exponent: replace 'D' or 'd' with 'E'
        for (auto& c : token) {
            if (c == 'D' || c == 'd') c = 'E';
        }
        char* end = nullptr;
        double v = std::strtod(token.c_str(), &end);
        if (end == token.c_str()) {
            // Could not parse -- skip non-numeric tokens silently
            continue;
        }
        values.push_back(v);
    }
    return values;
}

/// Find the opening tag that matches `tag_name` and return its full tag
/// string (everything between '<' and '>').  Searches from `start_pos`.
/// Returns std::string::npos in the size_t out-param if not found.
static std::string find_open_tag(const std::string& content,
                                 const std::string& tag_name,
                                 size_t start_pos,
                                 size_t& tag_end_pos) {
    // We search for "<TAG_NAME" possibly followed by whitespace, '>', or '/>'
    std::string pattern = "<" + tag_name;
    size_t pos = content.find(pattern, start_pos);

    // Retry with case-insensitive search if not found
    if (pos == std::string::npos) {
        // Manual case-insensitive search
        std::string lower_content = to_lower(content);
        std::string lower_pattern = to_lower(pattern);
        pos = lower_content.find(lower_pattern, start_pos);
    }

    if (pos == std::string::npos) {
        tag_end_pos = std::string::npos;
        return {};
    }

    // The character after the tag name must be whitespace, '>', or '/'
    size_t after = pos + pattern.size();
    if (after < content.size()) {
        char c = content[after];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r' &&
            c != '>' && c != '/') {
            // Not actually our tag (e.g. PP_R vs PP_RHOATOM). Search further.
            return find_open_tag(content, tag_name, after, tag_end_pos);
        }
    }

    // Find closing '>'
    size_t close = content.find('>', pos);
    if (close == std::string::npos) {
        tag_end_pos = std::string::npos;
        return {};
    }

    tag_end_pos = close + 1;
    return content.substr(pos, close - pos + 1);
}

/// Extract the body text between <tag_name ...> and </tag_name>.
/// Returns empty string if the tag is not found.
static std::string extract_tag_body(const std::string& content,
                                    const std::string& tag_name,
                                    size_t search_from = 0) {
    size_t tag_end = 0;
    std::string open_tag = find_open_tag(content, tag_name, search_from, tag_end);
    if (tag_end == std::string::npos) return {};

    // Check for self-closing tag (ends with "/>")
    if (open_tag.size() >= 2 && open_tag.substr(open_tag.size() - 2) == "/>") {
        return {};
    }

    // Find the closing tag
    std::string close_pattern = "</" + tag_name + ">";
    size_t close_pos = content.find(close_pattern, tag_end);
    if (close_pos == std::string::npos) {
        // Try case-insensitive
        std::string lower_content = to_lower(content);
        std::string lower_close = to_lower(close_pattern);
        close_pos = lower_content.find(lower_close, tag_end);
    }
    if (close_pos == std::string::npos) return {};

    return content.substr(tag_end, close_pos - tag_end);
}

/// Extract an attribute value from a tag string.
/// Handles both double-quoted and unquoted attribute values.
/// e.g. extract_attribute("<PP_HEADER z_valence=\"4.0\">", "z_valence") => "4.0"
std::string extract_attribute(const std::string& tag_str,
                              const std::string& attr_name) {
    // Try: attr_name="value" or attr_name = "value"
    // First, find the attribute name (case-insensitive)
    std::string lower_tag = to_lower(tag_str);
    std::string lower_attr = to_lower(attr_name);

    size_t attr_pos = lower_tag.find(lower_attr);
    if (attr_pos == std::string::npos) return {};

    // Skip past the attribute name
    size_t pos = attr_pos + lower_attr.size();

    // Skip whitespace
    while (pos < tag_str.size() && std::isspace(static_cast<unsigned char>(tag_str[pos])))
        ++pos;

    // Expect '='
    if (pos >= tag_str.size() || tag_str[pos] != '=') return {};
    ++pos;

    // Skip whitespace after '='
    while (pos < tag_str.size() && std::isspace(static_cast<unsigned char>(tag_str[pos])))
        ++pos;

    if (pos >= tag_str.size()) return {};

    // Check if value is quoted
    if (tag_str[pos] == '"') {
        ++pos;
        size_t end = tag_str.find('"', pos);
        if (end == std::string::npos) return {};
        return trim(tag_str.substr(pos, end - pos));
    } else if (tag_str[pos] == '\'') {
        ++pos;
        size_t end = tag_str.find('\'', pos);
        if (end == std::string::npos) return {};
        return trim(tag_str.substr(pos, end - pos));
    } else {
        // Unquoted value: read until whitespace or '>' or '/'
        size_t end = pos;
        while (end < tag_str.size() && !std::isspace(static_cast<unsigned char>(tag_str[end]))
               && tag_str[end] != '>' && tag_str[end] != '/') {
            ++end;
        }
        return trim(tag_str.substr(pos, end - pos));
    }
}

/// Safely parse an integer from a string, returning default_val on failure.
static int safe_stoi(const std::string& s, int default_val = 0) {
    if (s.empty()) return default_val;
    try {
        return std::stoi(s);
    } catch (...) {
        return default_val;
    }
}

/// Safely parse a double from a string, returning default_val on failure.
static double safe_stod(const std::string& s, double default_val = 0.0) {
    if (s.empty()) return default_val;
    // Handle Fortran D exponent
    std::string sc = s;
    for (auto& c : sc) {
        if (c == 'D' || c == 'd') c = 'E';
    }
    try {
        return std::stod(sc);
    } catch (...) {
        return default_val;
    }
}

/// Parse a boolean from a UPF attribute string.
/// UPF uses "T"/"F" or ".TRUE."/".FALSE." or "true"/"false".
static bool parse_bool(const std::string& s) {
    std::string lower = to_lower(trim(s));
    return (lower == "t" || lower == "true" || lower == ".true.");
}

/// Try to extract an attribute from the PP_HEADER tag string first;
/// if that fails, look for a child element <attr_name>value</attr_name>
/// within the PP_HEADER body. This handles both UPF v2 styles.
static std::string get_header_value(const std::string& header_tag,
                                    const std::string& header_body,
                                    const std::string& attr_name) {
    // First try as attribute
    std::string val = extract_attribute(header_tag, attr_name);
    if (!val.empty()) return val;

    // Then try as child element
    val = trim(extract_tag_body(header_body, attr_name));
    return val;
}

} // namespace detail

// ============================================================================
// Free-function wrappers exposed for testing
// ============================================================================

// These are declared in an internal detail header or accessed via the
// namespace for unit testing; they are not part of the public API but are
// accessible because the test file includes this translation unit's header
// and links against it.

namespace detail {
    // Already defined above; we just ensure the declarations are visible.
}

// ============================================================================
// parse_upf implementation
// ============================================================================

PseudoPotential parse_upf(const std::string& filepath) {
    std::string content = detail::read_file(filepath);
    PseudoPotential pp;

    // ------------------------------------------------------------------
    // PP_HEADER
    // ------------------------------------------------------------------
    size_t header_tag_end = 0;
    std::string header_tag = detail::find_open_tag(content, "PP_HEADER", 0, header_tag_end);
    if (header_tag_end == std::string::npos) {
        throw UPFParseError(filepath + ": missing <PP_HEADER> tag");
    }
    std::string header_body = detail::extract_tag_body(content, "PP_HEADER");

    pp.element         = detail::trim(detail::get_header_value(header_tag, header_body, "element"));
    pp.z_valence       = detail::safe_stod(detail::get_header_value(header_tag, header_body, "z_valence"));
    pp.pp_type         = detail::trim(detail::get_header_value(header_tag, header_body, "type"));
    pp.xc_functional   = detail::trim(detail::get_header_value(header_tag, header_body, "functional"));
    pp.total_psenergy  = detail::safe_stod(detail::get_header_value(header_tag, header_body, "total_psenergy"));
    pp.wfc_cutoff      = detail::safe_stod(detail::get_header_value(header_tag, header_body, "wfc_cutoff"));
    pp.rho_cutoff      = detail::safe_stod(detail::get_header_value(header_tag, header_body, "rho_cutoff"));
    pp.lmax            = detail::safe_stoi(detail::get_header_value(header_tag, header_body, "l_max"));
    pp.num_projectors  = detail::safe_stoi(detail::get_header_value(header_tag, header_body, "number_of_proj"));
    pp.num_wfc         = detail::safe_stoi(detail::get_header_value(header_tag, header_body, "number_of_wfc"));

    int mesh_size      = detail::safe_stoi(detail::get_header_value(header_tag, header_body, "mesh_size"));

    // Determine PP type flags
    std::string pp_type_lower = detail::to_lower(pp.pp_type);
    pp.is_norm_conserving = (pp_type_lower.find("nc") != std::string::npos ||
                             pp_type_lower.find("norm") != std::string::npos);
    pp.is_ultrasoft       = (pp_type_lower.find("us") != std::string::npos ||
                             detail::parse_bool(detail::get_header_value(header_tag, header_body, "is_ultrasoft")));
    pp.is_paw             = (pp_type_lower.find("paw") != std::string::npos ||
                             detail::parse_bool(detail::get_header_value(header_tag, header_body, "is_paw")));

    // If type string was not set but boolean flags are, deduce type
    if (pp.pp_type.empty()) {
        if (pp.is_paw) pp.pp_type = "PAW";
        else if (pp.is_ultrasoft) pp.pp_type = "US";
        else pp.pp_type = "NC";
    }

    // Atomic number: some UPF files have it as "atomic_number"
    pp.atomic_number = detail::safe_stoi(
        detail::get_header_value(header_tag, header_body, "atomic_number"));

    // If xc_functional was empty, also try "xc_functional" key
    if (pp.xc_functional.empty()) {
        pp.xc_functional = detail::trim(
            detail::get_header_value(header_tag, header_body, "xc_functional"));
    }

    // ------------------------------------------------------------------
    // PP_MESH / PP_R / PP_RAB
    // ------------------------------------------------------------------
    std::string r_body   = detail::extract_tag_body(content, "PP_R");
    std::string rab_body = detail::extract_tag_body(content, "PP_RAB");

    pp.mesh.r   = detail::parse_doubles_from_string(r_body);
    pp.mesh.rab = detail::parse_doubles_from_string(rab_body);
    pp.mesh.npoints = static_cast<int>(pp.mesh.r.size());

    // Cross-check with header mesh_size if available
    if (mesh_size > 0 && pp.mesh.npoints != mesh_size) {
        std::cerr << "KRONOS warning [" << filepath << "]: PP_HEADER mesh_size="
                  << mesh_size << " but PP_R has " << pp.mesh.npoints
                  << " points. Using actual data.\n";
    }

    // ------------------------------------------------------------------
    // PP_LOCAL
    // ------------------------------------------------------------------
    std::string vloc_body = detail::extract_tag_body(content, "PP_LOCAL");
    pp.vloc = detail::parse_doubles_from_string(vloc_body);

    // ------------------------------------------------------------------
    // PP_NONLOCAL / PP_BETA.n
    // ------------------------------------------------------------------
    pp.betas.resize(static_cast<size_t>(pp.num_projectors));
    for (int ib = 1; ib <= pp.num_projectors; ++ib) {
        std::string beta_tag_name = "PP_BETA." + std::to_string(ib);

        size_t beta_tag_end = 0;
        std::string beta_tag = detail::find_open_tag(content, beta_tag_name, 0, beta_tag_end);
        std::string beta_body = detail::extract_tag_body(content, beta_tag_name);

        auto& beta = pp.betas[static_cast<size_t>(ib - 1)];
        beta.index = ib;
        beta.angular_momentum = detail::safe_stoi(
            detail::extract_attribute(beta_tag, "angular_momentum"));
        beta.cutoff_index = detail::safe_stoi(
            detail::extract_attribute(beta_tag, "cutoff_radius_index"));
        beta.values = detail::parse_doubles_from_string(beta_body);
    }

    // ------------------------------------------------------------------
    // PP_DIJ
    // ------------------------------------------------------------------
    std::string dij_body = detail::extract_tag_body(content, "PP_DIJ");
    std::vector<double> dij_flat = detail::parse_doubles_from_string(dij_body);

    // Build the D_ij matrix (num_projectors x num_projectors)
    int np = pp.num_projectors;
    pp.dij.assign(static_cast<size_t>(np),
                  std::vector<double>(static_cast<size_t>(np), 0.0));
    if (static_cast<int>(dij_flat.size()) >= np * np) {
        for (int i = 0; i < np; ++i) {
            for (int j = 0; j < np; ++j) {
                pp.dij[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    dij_flat[static_cast<size_t>(i * np + j)];
            }
        }
    } else if (!dij_flat.empty()) {
        // Some files store only the upper triangle or a subset -- fill what we
        // can and warn.
        std::cerr << "KRONOS warning [" << filepath << "]: PP_DIJ has "
                  << dij_flat.size() << " values but expected "
                  << np * np << ". Filling available entries.\n";
        for (size_t k = 0; k < dij_flat.size() && k < static_cast<size_t>(np * np); ++k) {
            int i = static_cast<int>(k) / np;
            int j = static_cast<int>(k) % np;
            pp.dij[static_cast<size_t>(i)][static_cast<size_t>(j)] = dij_flat[k];
        }
    }

    // ------------------------------------------------------------------
    // PP_RHOATOM
    // ------------------------------------------------------------------
    std::string rho_body = detail::extract_tag_body(content, "PP_RHOATOM");
    pp.rho_atomic = detail::parse_doubles_from_string(rho_body);

    // ------------------------------------------------------------------
    // PP_PSWFC / PP_CHI.n  (atomic wavefunctions)
    // ------------------------------------------------------------------
    pp.atomic_wfc.resize(static_cast<size_t>(pp.num_wfc));
    for (int iw = 1; iw <= pp.num_wfc; ++iw) {
        std::string chi_tag_name = "PP_CHI." + std::to_string(iw);

        size_t chi_tag_end = 0;
        std::string chi_tag = detail::find_open_tag(content, chi_tag_name, 0, chi_tag_end);
        std::string chi_body = detail::extract_tag_body(content, chi_tag_name);

        auto& wfc = pp.atomic_wfc[static_cast<size_t>(iw - 1)];
        wfc.angular_momentum = detail::safe_stoi(
            detail::extract_attribute(chi_tag, "l"));
        wfc.occupation = detail::safe_stod(
            detail::extract_attribute(chi_tag, "occupation"));
        wfc.label = detail::extract_attribute(chi_tag, "label");
        wfc.values = detail::parse_doubles_from_string(chi_body);
    }

    return pp;
}

// ============================================================================
// validate_pseudopotential implementation
// ============================================================================

void validate_pseudopotential(const PseudoPotential& pp) {
    // z_valence must be positive
    if (pp.z_valence <= 0.0) {
        throw UPFParseError(
            "Invalid pseudopotential: z_valence=" +
            std::to_string(pp.z_valence) + " (must be > 0)");
    }

    // Mesh must have points
    if (pp.mesh.npoints <= 0) {
        throw UPFParseError(
            "Invalid pseudopotential for " + pp.element +
            ": radial mesh has no points");
    }

    // r and rab must have consistent sizes
    if (pp.mesh.r.size() != pp.mesh.rab.size()) {
        throw UPFParseError(
            "Invalid pseudopotential for " + pp.element +
            ": PP_R size (" + std::to_string(pp.mesh.r.size()) +
            ") != PP_RAB size (" + std::to_string(pp.mesh.rab.size()) + ")");
    }

    // V_loc must match mesh size
    if (!pp.vloc.empty() &&
        static_cast<int>(pp.vloc.size()) != pp.mesh.npoints) {
        throw UPFParseError(
            "Invalid pseudopotential for " + pp.element +
            ": V_loc size (" + std::to_string(pp.vloc.size()) +
            ") != mesh size (" + std::to_string(pp.mesh.npoints) + ")");
    }

    // Beta projector sizes must not exceed mesh size
    for (const auto& beta : pp.betas) {
        if (static_cast<int>(beta.values.size()) > pp.mesh.npoints) {
            throw UPFParseError(
                "Invalid pseudopotential for " + pp.element +
                ": beta projector " + std::to_string(beta.index) +
                " has more values (" + std::to_string(beta.values.size()) +
                ") than mesh points (" + std::to_string(pp.mesh.npoints) + ")");
        }
    }

    // D_ij matrix dimensions
    if (pp.num_projectors > 0) {
        if (static_cast<int>(pp.dij.size()) != pp.num_projectors) {
            throw UPFParseError(
                "Invalid pseudopotential for " + pp.element +
                ": D_ij row count (" + std::to_string(pp.dij.size()) +
                ") != num_projectors (" + std::to_string(pp.num_projectors) + ")");
        }
        for (const auto& row : pp.dij) {
            if (static_cast<int>(row.size()) != pp.num_projectors) {
                throw UPFParseError(
                    "Invalid pseudopotential for " + pp.element +
                    ": D_ij column count (" + std::to_string(row.size()) +
                    ") != num_projectors (" + std::to_string(pp.num_projectors) + ")");
            }
        }
    }

    // Warn (but do not throw) for non-norm-conserving PP in v0.1
    if (!pp.is_norm_conserving) {
        std::cerr << "KRONOS warning: pseudopotential for " << pp.element
                  << " is type '" << pp.pp_type
                  << "' (not norm-conserving). "
                  << "KRONOS v0.1 only supports norm-conserving PP.\n";
    }
}

} // namespace kronos
