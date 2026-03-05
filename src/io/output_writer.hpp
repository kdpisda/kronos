#pragma once
#include "solver/scf.hpp"
#include "core/crystal.hpp"
#include <string>

namespace kronos {

class OutputWriter {
public:
    // Write JSON summary to file
    // Uses atomic write (write to temp, then rename) to avoid partial output
    static void write_json(const std::string& filepath,
                           const SCFResult& result,
                           const Crystal& crystal,
                           const std::string& calculation_type);

    // Write JSON to string (for testing / stdout)
    static std::string to_json_string(const SCFResult& result,
                                      const Crystal& crystal,
                                      const std::string& calculation_type);
};

} // namespace kronos
