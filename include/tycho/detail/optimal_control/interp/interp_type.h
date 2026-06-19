#pragma once

#include <stdexcept>
#include <string>

namespace tycho {

/// @ingroup optimal_control
/// @brief Interpolation method for InterpTable1D through InterpTable4D.
enum class InterpType {
    Cubic,  ///< Cubic (spline) interpolation.
    Linear, ///< Piecewise-linear interpolation.
};

/// @brief Parse a string representation to an @ref InterpType value.
///
/// Accepted values (case-insensitive first letter): @c "cubic" / @c "Cubic"
/// and @c "linear" / @c "Linear".
///
/// @param kind  String naming the interpolation method.
/// @return The corresponding @ref InterpType enumerator.
/// @throws std::invalid_argument if @p kind is not a recognised method name.
inline InterpType parse_interp_type(const std::string &kind) {
    if (kind == "cubic" || kind == "Cubic")
        return InterpType::Cubic;
    if (kind == "linear" || kind == "Linear")
        return InterpType::Linear;
    throw std::invalid_argument("Unknown interp type: '" + kind +
                                "' (expected 'cubic' or 'linear')");
}

} // namespace tycho
