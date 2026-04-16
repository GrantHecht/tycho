// =============================================================================
// Tycho — Builder API: OptimalControlProblem wrapper
//
// Thin wrapper around OptimalControlProblemBase that accepts Phase wrappers
// directly, eliminating base_ptr() calls and enabling named-variable link
// constraints.  For methods not wrapped here, use base() to access the
// underlying OptimalControlProblemBase.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/builder/phase_wrapper.h"
#include "tycho/detail/optimal_control/phase/optimal_control_problem.h"

namespace tycho {

using oc::OptimalControlProblemBase;
// Solvers types
using tycho::solvers::PSIOPT;

/// Thin wrapper around OptimalControlProblemBase with Phase-aware overloads.
///
/// Accepts Phase wrapper objects directly (no base_ptr() needed) and
/// supports named-variable link constraints.  Use base() for any
/// OptimalControlProblemBase method not wrapped here.
class OptimalControlProblem {
  public:
    OptimalControlProblem() = default;

    // The OCP stores a raw pointer to each Phase passed to add_phase. The
    // caller must keep every added Phase alive (and not relocated, e.g. not
    // inside a std::vector<Phase> that may reallocate) for the lifetime of
    // the OCP. Name-resolution overloads (add_direct_link_equal_con by int
    // phase index) read the caller's live VarRegistry and sp-name map, so
    // names registered on the Phase after add_phase are visible to the OCP.
    int add_phase(Phase &p) {
        phases_.push_back(&p);
        return ocp_.add_phase(p.base_ptr());
    }

    int add_phase(Phase &p, const std::string &name) {
        phases_.push_back(&p);
        return ocp_.add_phase(p.base_ptr(), name);
    }

    /// Link phases with named variables.
    /// Creates equality constraints between consecutive phase pairs from p1
    /// through p2 (using phase ordering within the OCP).  Both phases must
    /// have registries; throws if either is empty or if names resolve to
    /// different indices.
    int add_forward_link_equal_con(Phase &p1, Phase &p2,
                                   std::initializer_list<std::string> var_names) {
        bool p1_empty = p1.registry().empty();
        bool p2_empty = p2.registry().empty();
        if (p1_empty || p2_empty) {
            std::string which = p1_empty && p2_empty ? "both phases"
                                : p1_empty           ? "phase p1"
                                                     : "phase p2";
            throw std::invalid_argument(fmt::format(
                "OptimalControlProblem::add_forward_link_equal_con: {} {} no variable names "
                "registered -- register names via ODEBuilder::var_names() "
                "or use the index-based overload",
                which, (p1_empty && p2_empty) ? "have" : "has"));
        }
        auto idx1 = p1.registry().resolve(var_names);
        auto idx2 = p2.registry().resolve(var_names);
        if (idx1.size() != idx2.size() || (idx1.array() != idx2.array()).any()) {
            auto fmt_idx = [](const Eigen::VectorXi &v) {
                std::string s = "[";
                for (int i = 0; i < v.size(); ++i) {
                    if (i > 0)
                        s += ", ";
                    s += std::to_string(v[i]);
                }
                return s + "]";
            };
            throw std::invalid_argument(fmt::format(
                "OptimalControlProblem::add_forward_link_equal_con: variable names resolve to "
                "different indices in p1 {} vs p2 {} -- use the "
                "index-based overload for heterogeneous phase layouts",
                fmt_idx(idx1), fmt_idx(idx2)));
        }
        return ocp_.add_forward_link_equal_con(p1.base_ptr(), p2.base_ptr(), idx1);
    }

    /// Link two phases with an index vector.
    int add_forward_link_equal_con(Phase &p1, Phase &p2, const Eigen::VectorXi &vars) {
        return ocp_.add_forward_link_equal_con(p1.base_ptr(), p2.base_ptr(), vars);
    }

    /// Index-based (int phase indices).
    int add_direct_link_equal_con(int phase_a, PhaseRegionFlags region_a,
                                  const Eigen::VectorXi &vars_a, int phase_b,
                                  PhaseRegionFlags region_b, const Eigen::VectorXi &vars_b) {
        return ocp_.add_direct_link_equal_con(phase_a, region_a, vars_a, phase_b, region_b, vars_b);
    }

    /// Index-based (Phase& references).
    int add_direct_link_equal_con(Phase &phase_a, PhaseRegionFlags region_a,
                                  const Eigen::VectorXi &vars_a, Phase &phase_b,
                                  PhaseRegionFlags region_b, const Eigen::VectorXi &vars_b) {
        return ocp_.add_direct_link_equal_con(phase_a.base_ptr(), region_a, vars_a,
                                              phase_b.base_ptr(), region_b, vars_b);
    }

    /// Named (int phase indices) — resolves names via stored Phase references.
    int add_direct_link_equal_con(int phase_a, PhaseRegionFlags region_a,
                                  const std::vector<std::string> &vars_a, int phase_b,
                                  PhaseRegionFlags region_b,
                                  const std::vector<std::string> &vars_b) {
        auto idx_a =
            resolve_link_vars(*phases_.at(static_cast<std::size_t>(phase_a)), region_a, vars_a);
        auto idx_b =
            resolve_link_vars(*phases_.at(static_cast<std::size_t>(phase_b)), region_b, vars_b);
        return ocp_.add_direct_link_equal_con(phase_a, region_a, idx_a, phase_b, region_b, idx_b);
    }

    /// Named (Phase& references).
    int add_direct_link_equal_con(Phase &phase_a, PhaseRegionFlags region_a,
                                  const std::vector<std::string> &vars_a, Phase &phase_b,
                                  PhaseRegionFlags region_b,
                                  const std::vector<std::string> &vars_b) {
        auto idx_a = resolve_link_vars(phase_a, region_a, vars_a);
        auto idx_b = resolve_link_vars(phase_b, region_b, vars_b);
        return ocp_.add_direct_link_equal_con(phase_a.base_ptr(), region_a, idx_a,
                                              phase_b.base_ptr(), region_b, idx_b);
    }

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

    void set_auto_scaling(bool autoscale, bool applytophases = true) {
        ocp_.set_auto_scaling(autoscale, applytophases);
    }
    void set_adaptive_mesh(bool amesh, bool applytophases = true) {
        ocp_.set_adaptive_mesh(amesh, applytophases);
    }
    void set_mesh_tol(double t) { ocp_.set_mesh_tol(t); }
    void set_num_partitions(int n) { ocp_.set_num_partitions(n); }
    void set_num_partitions(int n, int qp_threads) { ocp_.set_num_partitions(n, qp_threads); }

    PSIOPT &optimizer() { return *ocp_.optimizer_; }

    OptimalControlProblemBase &base() { return ocp_; }
    const OptimalControlProblemBase &base() const { return ocp_; }

  private:
    void check_has_phases(const char *method) const {
        if (ocp_.phases.empty())
            throw std::invalid_argument(fmt::format(
                "OptimalControlProblem::{}: no phases added — call add_phase() before solving",
                method));
    }

    /// Resolve variable names for a link constraint, respecting region.
    /// ODEParams names get P-relative translation; StaticParams use the SP registry.
    Eigen::VectorXi resolve_link_vars(Phase &p, PhaseRegionFlags region,
                                      const std::vector<std::string> &names) const {
        Eigen::VectorXi idx(static_cast<int>(names.size()));
        for (int i = 0; i < static_cast<int>(names.size()); ++i) {
            if (region == PhaseRegionFlags::StaticParams) {
                idx[i] = p.resolve_sp_single(names[static_cast<std::size_t>(i)],
                                             "add_direct_link_equal_con");
            } else {
                idx[i] = p.resolve_for_region(region, names[static_cast<std::size_t>(i)],
                                              "add_direct_link_equal_con");
            }
        }
        return idx;
    }

    OptimalControlProblemBase ocp_;
    /// Raw pointers to caller-owned Phase wrappers, used for int-indexed
    /// name resolution in add_direct_link_equal_con. The caller must keep
    /// each referenced Phase alive (and not relocated) for the lifetime of
    /// the OCP — see add_phase() for the contract.
    std::vector<Phase *> phases_;
};

} // namespace tycho
