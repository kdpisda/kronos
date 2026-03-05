#pragma once
#include "core/types.hpp"
#include "core/crystal.hpp"
#include <string>
#include <stdexcept>

namespace kronos {

class InputValidationError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ParsedInput {
    Crystal crystal;
    InputData input;
};

// Parse YAML input file, validate all fields, return ParsedInput
// Throws InputValidationError on validation failure
ParsedInput parse_input(const std::string& filepath);

// Parse YAML from string (useful for testing)
ParsedInput parse_input_string(const std::string& yaml_content);

} // namespace kronos
