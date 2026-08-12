// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Fallback implementation of the Ipopt backend entry points for builds without
// Ipopt. Selecting the ipopt backend in such a build is a configuration error,
// reported as a std::runtime_error naming the build option that enables it.

#include "tycho/detail/solvers/nlp_backend.h"

#include <stdexcept>

namespace tycho::solvers::ipopt_backend {

bool available() { return false; }

BackendProblemBase::NlpSolveOutput solve(BackendProblemBase &prob,
                                         BackendProblemBase::JetJobModes mode,
                                         const Eigen::VectorXd &input) {
    (void)prob;
    (void)mode;
    (void)input;
    throw std::runtime_error("Tycho was built without Ipopt support; configure with "
                             "-DENABLE_IPOPT=ON (requires an installed Ipopt discoverable "
                             "via pkg-config) to use nlp_solver = ipopt");
}

} // namespace tycho::solvers::ipopt_backend
