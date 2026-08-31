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

IpoptSolveOutput solve(const std::shared_ptr<NonLinearProgram> &nlp, const Eigen::VectorXd &x0,
                       const std::map<std::string, std::string> &ipopt_options,
                       const InteriorPointSolver::Settings &tolerance_baseline) {
    (void)nlp;
    (void)x0;
    (void)ipopt_options;
    (void)tolerance_baseline;
    throw std::runtime_error("Tycho was built without Ipopt support; configure with "
                             "-DENABLE_IPOPT=ON (requires an installed Ipopt discoverable "
                             "via pkg-config) to construct an IpoptSolver engine");
}

} // namespace tycho::solvers::ipopt_backend
