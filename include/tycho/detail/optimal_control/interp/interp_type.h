#pragma once

#include <stdexcept>
#include <string>

namespace tycho {

/// Interpolation method for InterpTable1D through InterpTable4D.
enum class InterpType { Cubic, Linear };

/// Parse a string to InterpType. Accepts "cubic"/"Cubic" and "linear"/"Linear".
inline InterpType parse_interp_type(const std::string &kind) {
    if (kind == "cubic" || kind == "Cubic")
        return InterpType::Cubic;
    if (kind == "linear" || kind == "Linear")
        return InterpType::Linear;
    throw std::invalid_argument("Unknown interp type: '" + kind +
                                "' (expected 'cubic' or 'linear')");
}

} // namespace tycho
