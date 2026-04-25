// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <exception>
#include <string>

namespace tycho::utils {

/// Extract `what()` from a std::exception_ptr, returning "<non-std::exception>"
/// when the captured exception does not derive from std::exception. Localises
/// the rethrow-and-catch pattern so the surrounding code does not have to
/// open a try/catch just to read a message.
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
