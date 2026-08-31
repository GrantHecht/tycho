// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/detail/solvers/solve_types.h"

#include <cctype>
#include <stdexcept>

#include <fmt/format.h>

namespace {

std::string solve_types_to_lower(const std::string &s) {
    std::string lowered;
    lowered.reserve(s.size());
    for (unsigned char c : s) {
        lowered.push_back(static_cast<char>(std::tolower(c)));
    }
    return lowered;
}

} // namespace

const char *tycho::solvers::to_string(Mode m) {
    switch (m) {
    case Mode::Optimal:
        return "Optimal";
    case Mode::Feasible:
        return "Feasible";
    }
    throw std::invalid_argument("Unknown tycho::solvers::Mode value");
}

tycho::solvers::Mode tycho::solvers::mode_from_string(const std::string &s) {
    const std::string lowered = solve_types_to_lower(s);
    if (lowered == "optimal") {
        return Mode::Optimal;
    }
    if (lowered == "feasible") {
        return Mode::Feasible;
    }
    throw std::invalid_argument(
        fmt::format("Unknown solve mode \"{}\" (expected \"optimal\" or \"feasible\")", s));
}

const tycho::solvers::StageResult &tycho::solvers::SolveResult::final_stage() const {
    if (stages_.empty()) {
        throw std::logic_error("SolveResult::final_stage(): stages_ is empty -- no stage ran");
    }
    return stages_.back();
}
