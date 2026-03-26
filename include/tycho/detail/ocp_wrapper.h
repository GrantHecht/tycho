// =============================================================================
// Tycho — Builder API: OCP wrapper
//
// Thin wrapper around OptimalControlProblem that accepts Phase wrappers
// directly, eliminating base_ptr() calls and enabling named-variable link
// constraints.  For methods not wrapped here, use base() to access the
// underlying OptimalControlProblem.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/OptimalControlProblem.h"
#include "tycho/detail/phase_wrapper.h"

namespace Tycho {

/// Thin wrapper around OptimalControlProblem with Phase-aware overloads.
///
/// Accepts Phase wrapper objects directly (no base_ptr() needed) and
/// supports named-variable link constraints.  Use base() for any
/// OptimalControlProblem method not wrapped here.
class OCP {
  public:
    OCP() = default;

    // ── Phase management ────────────────────────────────────────────────

    int addPhase(Phase &p) { return ocp_.addPhase(p.base_ptr()); }

    int addPhase(Phase &p, const std::string &name) { return ocp_.addPhase(p.base_ptr(), name); }

    // ── Forward link constraints ────────────────────────────────────────

    /// Link two phases with named variables (resolves via first phase's registry).
    int addForwardLinkEqualCon(Phase &p1, Phase &p2,
                               std::initializer_list<std::string> var_names) {
        return ocp_.addForwardLinkEqualCon(p1.base_ptr(), p2.base_ptr(),
                                           p1.registry().resolve(var_names));
    }

    /// Link two phases with an index vector.
    int addForwardLinkEqualCon(Phase &p1, Phase &p2, const Eigen::VectorXi &vars) {
        return ocp_.addForwardLinkEqualCon(p1.base_ptr(), p2.base_ptr(), vars);
    }

    // ── Solve ───────────────────────────────────────────────────────────

    PSIOPT::ConvergenceFlags solve() { return ocp_.solve(); }
    PSIOPT::ConvergenceFlags optimize() { return ocp_.optimize(); }
    PSIOPT::ConvergenceFlags solve_optimize() { return ocp_.solve_optimize(); }
    PSIOPT::ConvergenceFlags optimize_solve() { return ocp_.optimize_solve(); }

    // ── Settings ────────────────────────────────────────────────────────

    void setAutoScaling(bool autoscale, bool applytophases = true) {
        ocp_.setAutoScaling(autoscale, applytophases);
    }
    void setAdaptiveMesh(bool amesh, bool applytophases = true) {
        ocp_.setAdaptiveMesh(amesh, applytophases);
    }
    void setMeshTol(double t) { ocp_.setMeshTol(t); }
    void setNumPartitions(int n) { ocp_.setNumPartitions(n); }
    void setNumPartitions(int n, int qp_threads) { ocp_.setNumPartitions(n, qp_threads); }

    // ── Optimizer / base access ─────────────────────────────────────────

    PSIOPT &optimizer() { return *ocp_.optimizer; }

    OptimalControlProblem &base() { return ocp_; }
    const OptimalControlProblem &base() const { return ocp_; }

  private:
    OptimalControlProblem ocp_;
};

} // namespace Tycho
