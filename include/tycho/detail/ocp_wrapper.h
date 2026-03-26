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

    /// Link phases with named variables.
    /// Creates equality constraints between consecutive phase pairs from p1
    /// through p2 (using phase ordering within the OCP).  Both phases must
    /// have registries; throws if either is empty or if names resolve to
    /// different indices.
    int addForwardLinkEqualCon(Phase &p1, Phase &p2,
                               std::initializer_list<std::string> var_names) {
        if (p1.registry().empty() || p2.registry().empty())
            throw std::invalid_argument(
                "OCP::addForwardLinkEqualCon: both phases must have variable "
                "names registered when using the named-variable overload — "
                "register names via ODEBuilder::var_names() or use the "
                "index-based overload");
        auto idx1 = p1.registry().resolve(var_names);
        auto idx2 = p2.registry().resolve(var_names);
        if (idx1.size() != idx2.size() ||
            (idx1.array() != idx2.array()).any()) {
            throw std::invalid_argument(
                "OCP::addForwardLinkEqualCon: variable names resolve to "
                "different indices in p1 vs p2 — use index-based overload "
                "for heterogeneous phase layouts");
        }
        return ocp_.addForwardLinkEqualCon(p1.base_ptr(), p2.base_ptr(), idx1);
    }

    /// Link two phases with an index vector.
    int addForwardLinkEqualCon(Phase &p1, Phase &p2, const Eigen::VectorXi &vars) {
        return ocp_.addForwardLinkEqualCon(p1.base_ptr(), p2.base_ptr(), vars);
    }

    // ── Solve ───────────────────────────────────────────────────────────

    PSIOPT::ConvergenceFlags solve() {
        check_has_phases("solve");
        return ocp_.solve();
    }
    PSIOPT::ConvergenceFlags optimize() {
        check_has_phases("optimize");
        return ocp_.optimize();
    }
    PSIOPT::ConvergenceFlags solve_optimize() {
        check_has_phases("solve_optimize");
        return ocp_.solve_optimize();
    }
    PSIOPT::ConvergenceFlags optimize_solve() {
        check_has_phases("optimize_solve");
        return ocp_.optimize_solve();
    }

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
    void check_has_phases(const char *method) const {
        if (ocp_.phases.empty())
            throw std::invalid_argument(
                fmt::format("OCP::{}: no phases added — call addPhase() before solving", method));
    }

    OptimalControlProblem ocp_;
};

} // namespace Tycho
