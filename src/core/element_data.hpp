#pragma once

#include <array>
#include <string>
#include <string_view>
#include <stdexcept>

namespace kronos {

namespace detail {

/// Element symbols indexed by atomic number (0 = unused placeholder).
/// Covers Z = 1 (H) through Z = 86 (Rn).
inline constexpr std::array<std::string_view, 87> element_symbols = {
    "",                                                                 // 0
    "H",  "He",                                                        // 1-2
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",                  // 3-10
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",                  // 11-18
    "K",  "Ca",                                                        // 19-20
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",     // 21-30
    "Ga", "Ge", "As", "Se", "Br", "Kr",                              // 31-36
    "Rb", "Sr",                                                        // 37-38
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",    // 39-48
    "In", "Sn", "Sb", "Te", "I",  "Xe",                              // 49-54
    "Cs", "Ba",                                                        // 55-56
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",                 // 57-64
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",                       // 65-71
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",          // 72-80
    "Tl", "Pb", "Bi", "Po", "At", "Rn"                               // 81-86
};

} // namespace detail

/// Return the atomic number for a given element symbol.
/// Throws std::invalid_argument if the symbol is not recognised.
inline int atomic_number_from_symbol(const std::string& symbol) {
    for (int z = 1; z < static_cast<int>(detail::element_symbols.size()); ++z) {
        if (detail::element_symbols[z] == symbol) {
            return z;
        }
    }
    throw std::invalid_argument(
        "atomic_number_from_symbol: unknown element symbol '" + symbol + "'");
}

/// Return the element symbol for a given atomic number.
/// Throws std::out_of_range if Z is outside [1, 86].
inline std::string symbol_from_atomic_number(int z) {
    if (z < 1 || z >= static_cast<int>(detail::element_symbols.size())) {
        throw std::out_of_range(
            "symbol_from_atomic_number: Z=" + std::to_string(z)
            + " is outside the supported range [1, 86]");
    }
    return std::string(detail::element_symbols[z]);
}

} // namespace kronos
