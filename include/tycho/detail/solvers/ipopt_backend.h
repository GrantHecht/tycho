// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Backend-neutral declarations for the optional Ipopt NLP solver backend.
// This header deliberately contains no Ipopt includes and no Ipopt types, so it
// is safe to include in every build regardless of how the project was
// configured. The Ipopt dependency is confined to the translation unit that
// implements the backend.

#pragma once

#include <string>

#include "tycho/detail/solvers/psiopt_fwd.h"

namespace tycho::solvers {

/// Outcome of one Ipopt run on the transcribed NLP. Sentinel values (-1 /
/// empty / ran_ == false) mean no Ipopt solve has run on this problem object.
struct IpoptRunInfo {
    bool ran_ = false;
    std::string status_ = "";     ///< Raw Ipopt ApplicationReturnStatus name.
    std::string normalized_ = ""; ///< converged/acceptable/infeasible/failed/diverged.
    ConvergenceFlags converge_flag_ = ConvergenceFlags::NOTCONVERGED;
    int iterations_ = -1;
    double objective_ = 0.0;
    double constraint_violation_ = -1.0;
    double wall_time_s_ = -1.0;
};

namespace ipopt_backend {

/// True when this build was configured with Ipopt support linked in.
bool available();

// The solve entry point is declared at the bottom of optimization_problem_base.h
// rather than here: its return type is nested in OptimizationProblemBase, which
// is only forward-declared at this point. Keeping the declaration there also
// keeps this header free of any heavy solver includes.

} // namespace ipopt_backend
} // namespace tycho::solvers
