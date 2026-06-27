// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <exception>
#include <string>

namespace tycho::utils {

/// @brief Extract the `what()` message from a `std::exception_ptr`.
///
/// Returns `"<non-std::exception>"` when the captured exception does not
/// derive from `std::exception`, and `"<no exception>"` when @p p is empty.
/// Localises the rethrow-and-catch pattern so surrounding code does not have
/// to open a try/catch just to read an exception message.
///
/// @param p  The exception pointer to inspect (may be null).
/// @return  The exception message string, or a placeholder if unavailable.
inline std::string exception_what(std::exception_ptr p) {
    if (!p) {
        return "<no exception>";
    }
    try {
        std::rethrow_exception(p);
    } catch (const std::exception &e) {
        return e.what();
    } catch (...) {
        return "<non-std::exception>";
    }
}

} // namespace tycho::utils
