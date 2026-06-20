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
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/phase/optimal_control_problem.h"

namespace tycho {

using oc::OptimalControlProblemBase;
// Solvers types
using tycho::solvers::PSIOPT;

/// @ingroup optimal_control
/// @brief Builder-API multi-phase optimal control problem over @ref Phase wrappers.
///
/// Thin wrapper around @ref tycho::oc::OptimalControlProblemBase that accepts
/// @ref Phase wrapper objects directly (no @c base_ptr() needed) and supports
/// named-variable link constraints. Use @ref base() for any base method not
/// wrapped here.
/// @note Phases are stored by raw pointer; the caller must keep every added
///       @ref Phase alive and non-relocated for the lifetime of this problem.
class OptimalControlProblem {
  public:
    /// @brief Construct an empty problem with no phases.
    OptimalControlProblem() = default;

    // The OCP stores a raw pointer to each Phase passed to add_phase. The
    // caller must keep every added Phase alive (and not relocated, e.g. not
    // inside a std::vector<Phase> that may reallocate) for the lifetime of
    // the OCP. Name-resolution overloads (add_direct_link_equal_con by int
    // phase index) read the caller's live VarRegistry and sp-name map, so
    // names registered on the Phase after add_phase are visible to the OCP.
    /// @brief Add a phase to the problem.
    /// @param p  The phase to add (must outlive the problem).
    /// @return The index assigned to the phase.
    int add_phase(Phase &p) {
        phases_.push_back(&p);
        return ocp_.add_phase(p.base_ptr());
    }

    /// @brief Add a phase to the problem with an explicit name.
    /// @param p     The phase to add (must outlive the problem).
    /// @param name  Unique name for the phase.
    /// @return The index assigned to the phase.
    int add_phase(Phase &p, const std::string &name) {
        phases_.push_back(&p);
        return ocp_.add_phase(p.base_ptr(), name);
    }

    /// @brief Add forward-link equality constraints between two phases using named variables.
    ///
    /// Creates equality constraints that link the final state of @p p1 to the
    /// initial state of @p p2 for each variable name in @p var_names.  Both
    /// phases must have a populated variable registry (populated via
    /// @ref ODEBuilder::var_names() or @ref ODE::var_names()); the names must
    /// resolve to the same XtUP indices in both phases.
    ///
    /// @param p1         The first (departing) phase; its terminal state is constrained.
    /// @param p2         The second (arriving) phase; its initial state is constrained.
    /// @param var_names  Names of the variables to link.
    /// @return A vector of one constraint index per consecutive phase pair created.
    /// @throws std::invalid_argument if either phase has no registered variable names,
    ///         or if @p var_names resolve to different indices in @p p1 and @p p2.
    std::vector<int> add_forward_link_equal_con(Phase &p1, Phase &p2,
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
        return ocp_.add_forward_link_equal_con(p1.base_ptr(), p2.base_ptr(), idx1,
                                               ScaleModes::AUTO);
    }

    /// @brief Add forward-link equality constraints between two phases using an explicit index
    /// vector.
    ///
    /// Creates equality constraints that link the final state of @p p1 to the
    /// initial state of @p p2 for each XtUP index in @p vars.
    ///
    /// @param p1    The first (departing) phase; its terminal state is constrained.
    /// @param p2    The second (arriving) phase; its initial state is constrained.
    /// @param vars  XtUP-space indices of the variables to link.
    /// @return A vector of one constraint index per consecutive phase pair created.
    std::vector<int> add_forward_link_equal_con(Phase &p1, Phase &p2, const Eigen::VectorXi &vars) {
        return ocp_.add_forward_link_equal_con(p1.base_ptr(), p2.base_ptr(), vars,
                                               ScaleModes::AUTO);
    }

    /// @brief Add a direct link equality constraint between two phases using integer indices and
    /// explicit variable index vectors.
    ///
    /// Links the variables @p vars_a at region @p region_a of phase @p phase_a to
    /// @p vars_b at region @p region_b of phase @p phase_b.
    ///
    /// @param phase_a   Zero-based index of the first phase (as returned by @ref add_phase()).
    /// @param region_a  Phase region (e.g. @c Front, @c Back) for the first phase.
    /// @param vars_a    XtUP-space variable indices for the first phase.
    /// @param phase_b   Zero-based index of the second phase.
    /// @param region_b  Phase region for the second phase.
    /// @param vars_b    XtUP-space variable indices for the second phase.
    /// @return The constraint index assigned to the created link constraint.
    int add_direct_link_equal_con(int phase_a, PhaseRegionFlags region_a,
                                  const Eigen::VectorXi &vars_a, int phase_b,
                                  PhaseRegionFlags region_b, const Eigen::VectorXi &vars_b) {
        return ocp_.add_direct_link_equal_con(phase_a, region_a, vars_a, phase_b, region_b, vars_b,
                                              ScaleModes::AUTO);
    }

    /// @brief Add a direct link equality constraint between two phases using @ref Phase references
    /// and explicit variable index vectors.
    ///
    /// Links the variables @p vars_a at region @p region_a of @p phase_a to
    /// @p vars_b at region @p region_b of @p phase_b.
    ///
    /// @param phase_a   Reference to the first phase.
    /// @param region_a  Phase region for the first phase.
    /// @param vars_a    XtUP-space variable indices for the first phase.
    /// @param phase_b   Reference to the second phase.
    /// @param region_b  Phase region for the second phase.
    /// @param vars_b    XtUP-space variable indices for the second phase.
    /// @return The constraint index assigned to the created link constraint.
    int add_direct_link_equal_con(Phase &phase_a, PhaseRegionFlags region_a,
                                  const Eigen::VectorXi &vars_a, Phase &phase_b,
                                  PhaseRegionFlags region_b, const Eigen::VectorXi &vars_b) {
        return ocp_.add_direct_link_equal_con(phase_a.base_ptr(), region_a, vars_a,
                                              phase_b.base_ptr(), region_b, vars_b,
                                              ScaleModes::AUTO);
    }

    /// @brief Add a direct link equality constraint between two phases using integer indices and
    /// named variables.
    ///
    /// Resolves @p vars_a and @p vars_b to XtUP indices via the @ref VarRegistry
    /// of the corresponding stored @ref Phase references (indexed by @p phase_a
    /// and @p phase_b respectively), then delegates to the index-based overload.
    ///
    /// @param phase_a   Zero-based index of the first phase (as returned by @ref add_phase()).
    /// @param region_a  Phase region for the first phase.
    /// @param vars_a    Variable names for the first phase (must be registered in that phase).
    /// @param phase_b   Zero-based index of the second phase.
    /// @param region_b  Phase region for the second phase.
    /// @param vars_b    Variable names for the second phase (must be registered in that phase).
    /// @return The constraint index assigned to the created link constraint.
    /// @throws std::out_of_range if @p phase_a or @p phase_b exceed the number of added phases.
    /// @throws std::invalid_argument if any name in @p vars_a or @p vars_b is not registered
    ///         in the respective phase registry.
    int add_direct_link_equal_con(int phase_a, PhaseRegionFlags region_a,
                                  const std::vector<std::string> &vars_a, int phase_b,
                                  PhaseRegionFlags region_b,
                                  const std::vector<std::string> &vars_b) {
        auto idx_a =
            resolve_link_vars(*phases_.at(static_cast<std::size_t>(phase_a)), region_a, vars_a);
        auto idx_b =
            resolve_link_vars(*phases_.at(static_cast<std::size_t>(phase_b)), region_b, vars_b);
        return ocp_.add_direct_link_equal_con(phase_a, region_a, idx_a, phase_b, region_b, idx_b,
                                              ScaleModes::AUTO);
    }

    /// @brief Add a direct link equality constraint between two phases using @ref Phase references
    /// and named variables.
    ///
    /// Resolves @p vars_a via @p phase_a's registry and @p vars_b via
    /// @p phase_b's registry, then delegates to the index-based overload.
    ///
    /// @param phase_a   Reference to the first phase.
    /// @param region_a  Phase region for the first phase.
    /// @param vars_a    Variable names for the first phase (must be registered in that phase).
    /// @param phase_b   Reference to the second phase.
    /// @param region_b  Phase region for the second phase.
    /// @param vars_b    Variable names for the second phase (must be registered in that phase).
    /// @return The constraint index assigned to the created link constraint.
    /// @throws std::invalid_argument if any name in @p vars_a or @p vars_b is not registered
    ///         in the respective phase registry.
    int add_direct_link_equal_con(Phase &phase_a, PhaseRegionFlags region_a,
                                  const std::vector<std::string> &vars_a, Phase &phase_b,
                                  PhaseRegionFlags region_b,
                                  const std::vector<std::string> &vars_b) {
        auto idx_a = resolve_link_vars(phase_a, region_a, vars_a);
        auto idx_b = resolve_link_vars(phase_b, region_b, vars_b);
        return ocp_.add_direct_link_equal_con(phase_a.base_ptr(), region_a, idx_a,
                                              phase_b.base_ptr(), region_b, idx_b,
                                              ScaleModes::AUTO);
    }

    /// @brief Solve the problem for feasibility (no optimization).
    /// @return The solver convergence flag.
    /// @throws std::invalid_argument if no phases have been added.
    PSIOPT::ConvergenceFlags solve() {
        check_has_phases("solve");
        return ocp_.solve();
    }
    /// @brief Optimize the problem (minimize the objective subject to constraints).
    /// @return The solver convergence flag.
    /// @throws std::invalid_argument if no phases have been added.
    PSIOPT::ConvergenceFlags optimize() {
        check_has_phases("optimize");
        return ocp_.optimize();
    }
    /// @brief Solve for feasibility, then optimize.
    /// @return The solver convergence flag.
    /// @throws std::invalid_argument if no phases have been added.
    PSIOPT::ConvergenceFlags solve_optimize() {
        check_has_phases("solve_optimize");
        return ocp_.solve_optimize();
    }
    /// @brief Optimize, then solve to tighten feasibility.
    /// @return The solver convergence flag.
    /// @throws std::invalid_argument if no phases have been added.
    PSIOPT::ConvergenceFlags optimize_solve() {
        check_has_phases("optimize_solve");
        return ocp_.optimize_solve();
    }

    /// @brief Enable/disable automatic scaling, optionally propagating to phases.
    /// @param autoscale     Whether to enable automatic scaling.
    /// @param applytophases Whether to also set the flag on each phase.
    void set_auto_scaling(bool autoscale, bool applytophases = true) {
        ocp_.set_auto_scaling(autoscale, applytophases);
    }
    /// @brief Enable/disable adaptive mesh refinement, optionally propagating to phases.
    /// @param amesh         Whether to enable adaptive mesh refinement.
    /// @param applytophases Whether to also set the flag on each phase.
    void set_adaptive_mesh(bool amesh, bool applytophases = true) {
        ocp_.set_adaptive_mesh(amesh, applytophases);
    }
    /// @brief Set the mesh-error tolerance on every phase.
    /// @param t  Tolerance.
    void set_mesh_tol(double t) { ocp_.set_mesh_tol(t); }
    /// @brief Set the number of solver partitions.
    /// @param n  Number of partitions.
    void set_num_partitions(int n) { ocp_.set_num_partitions(n); }
    /// @brief Set the number of solver partitions and QP threads.
    /// @param n           Number of partitions.
    /// @param qp_threads  Number of QP (linear-solver) threads.
    void set_num_partitions(int n, int qp_threads) { ocp_.set_num_partitions(n, qp_threads); }

    /// @brief Access the underlying PSIOPT optimizer.
    /// @return Reference to the optimizer.
    PSIOPT &optimizer() { return *ocp_.optimizer_; }

    /// @brief Access the wrapped base problem.
    /// @return Reference to the underlying @ref tycho::oc::OptimalControlProblemBase.
    OptimalControlProblemBase &base() { return ocp_; }
    /// @brief Access the wrapped base problem (const).
    /// @return Const reference to the underlying base problem.
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
