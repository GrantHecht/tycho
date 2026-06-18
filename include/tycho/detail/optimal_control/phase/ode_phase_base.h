// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/core/interface_types.h"
#include "tycho/detail/optimal_control/core/ode_sizes.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/core/state_function.h"
#include "tycho/detail/optimal_control/phase/mesh_iterate_info.h"
#include "tycho/detail/optimal_control/phase/phase_indexer.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_table.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/optimization_problem_base.h"
#include "tycho/detail/solvers/psiopt.h"
#include "tycho/detail/vf/scaling/io_scaled.h"
#include "tycho/vector_functions.h"
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

// Solvers types
using tycho::solvers::NonLinearProgram;
using tycho::solvers::OptimizationProblemBase;
using tycho::solvers::PSIOPT;

/// @internal
/// @brief Forward declaration of the multi-phase problem base (friend of @ref ODEPhaseBase).
/// @endinternal
struct OptimalControlProblemBase;

/// @ingroup optimal_control
/// @brief Type-erased base class for a single optimal control phase.
///
/// An @c ODEPhaseBase carries one ODE's discretized trajectory plus the
/// boundary values, path/integral constraints, objectives, control mode, mesh,
/// and scaling that define an optimal control phase. It transcribes the
/// continuous problem into an NLP and drives PSIOPT to solve or optimize it, and
/// owns the adaptive-mesh-refinement loop. The typed @ref ODEPhase derives from
/// this; @ref OptimalControlProblem links multiple phases together.
struct ODEPhaseBase : ODESize<-1, -1, -1>, OptimizationProblemBase {
    using VectorXi = Eigen::VectorXi; ///< @brief Convenience alias for an integer vector.
    using MatrixXi = Eigen::MatrixXi; ///< @brief Convenience alias for an integer matrix.

    using VectorXd = Eigen::VectorXd; ///< @brief Convenience alias for a double vector.
    using MatrixXd = Eigen::MatrixXd; ///< @brief Convenience alias for a double matrix.

    using VectorFunctionalX =
        GenericFunction<-1, -1>; ///< @brief Type-erased vector-valued function.
    using ScalarFunctionalX =
        GenericFunction<-1, 1>; ///< @brief Type-erased scalar-valued function.

    using StateConstraint =
        StateFunction<VectorFunctionalX>; ///< @brief A constraint bound to a region.
    using StateObjective =
        StateFunction<ScalarFunctionalX>; ///< @brief An objective bound to a region.

    /// @brief Convenience alias for a compile-time integer constant.
    /// @tparam V  The integer value.
    template <int V> using int_const = std::integral_constant<int, V>;

    /// @internal
    /// @brief Grants the multi-phase problem base access to this phase's internals.
    /// @endinternal
    friend OptimalControlProblemBase;

  protected:
    PhaseIndexer indexer_;         ///< Maps this phase's variables/constraints into the NLP.
    bool do_transcription_ = true; ///< Whether the phase must be (re-)transcribed before solving.
    bool enable_vectorization_ = true; ///< Whether SuperScalar evaluation is enabled.

    int num_defects_ = 0;      ///< Number of defect (collocation) intervals.
    int num_stat_params_ = 0;  ///< Number of static (non-ODE) parameters.
    VectorXd def_bin_spacing_; ///< Normalized bin boundaries for defect distribution.
    VectorXi defs_per_bin_;    ///< Number of defects per mesh bin.

    TranscriptionModes transcription_mode_ =
        TranscriptionModes::LGL3;                                ///< Active transcription scheme.
    ControlModes control_mode_ = ControlModes::FirstOrderSpline; ///< Active control representation.
    IntegralModes integral_mode_ = IntegralModes::BaseIntegral;  ///< Active integral quadrature.
    int num_tran_card_states_ = 2; ///< Cardinal states per defect interval for the transcription.
    double order_ = 3;             ///< Polynomial order of the transcription.

    LGLInterpTable table_; ///< LGL interpolation table backing trajectory queries.

    bool trajectory_loaded_ = false;    ///< Whether an initial/active trajectory has been set.
    std::vector<VectorXd> active_traj_; ///< Current discretized trajectory (per state).
    VectorXd active_ode_params_;        ///< Current ODE-parameter values.
    VectorXd active_static_params_;     ///< Current static-parameter values.

    bool multipliers_loaded_ = false;  ///< Whether Lagrange multipliers are available.
    bool post_opt_info_valid_ = false; ///< Whether post-solve constraint/multiplier data is valid.

    VectorXd active_eq_lmults_; ///< Equality-constraint Lagrange multipliers.
    VectorXd active_iq_lmults_; ///< Inequality-constraint Lagrange multipliers.
    VectorXd active_eq_cons_;   ///< Equality-constraint residual values.
    VectorXd active_iq_cons_;   ///< Inequality-constraint residual values.

    std::map<int, StateConstraint> user_equalities_; ///< User-added equality constraints by index.
    std::map<int, StateConstraint>
        user_inequalities_; ///< User-added inequality constraints by index.
    std::map<int, StateObjective> user_state_objectives_; ///< User-added state objectives by index.
    std::map<int, StateObjective> user_integrands_; ///< User-added integral objectives by index.
    std::map<int, StateObjective>
        user_param_integrands_; ///< User-added integral-parameter functions.

    int dynamics_func_index_ = 0;        ///< Global index of the dynamics defect function.
    int control_funcs_index_ = -1;       ///< Global index of the control-spline constraint, or -1.
    VectorXi node_spacing_func_indices_; ///< Global indices of node-spacing constraints.
    int tran_spacing_func_indices_ = 0;  ///< Global index of the transcription-spacing constraint.
    int constraint_order_ = 0;           ///< Ordering tag for constraint registration.

    //////////////////////////

    Eigen::VectorXd xtup_units_; ///< Scaling units for the state/time/control/ODE-param block.
    Eigen::VectorXd sp_units_;   ///< Scaling units for the static-parameter block.

    std::map<std::string, Eigen::VectorXi> sp_idxs_; ///< Named static-parameter variable groups.

    ///////////////////////
  public:
    bool auto_scaling_ = false;         ///< Whether automatic variable/equation scaling is enabled.
    bool sync_objective_scales_ = true; ///< Whether objective scales track the auto-scaling.

    bool adaptive_mesh_ = false;  ///< Whether adaptive mesh refinement is enabled.
    bool print_mesh_info_ = true; ///< Whether to print mesh-refinement diagnostics.
    int max_mesh_iters_ = 10;     ///< Maximum mesh-refinement iterations.
    int max_segments_ = 10000;    ///< Upper bound on the number of mesh segments.
    int min_segments_ = 4;        ///< Lower bound on the number of mesh segments.

    int num_extra_segs_ = 4;           ///< Extra segments added when refining.
    double mesh_red_factor_ = .5;      ///< Factor by which segment counts may be reduced.
    double mesh_inc_factor_ = 5.0;     ///< Factor by which segment counts may be increased.
    double mesh_err_factor_ = 10.0;    ///< Error-overshoot factor triggering refinement.
    bool force_one_mesh_iter_ = false; ///< Force at least one mesh iteration.
    bool solve_only_first_ = true;     ///< Only solve (not optimize) on the first mesh iteration.
    bool new_error_ = false;           ///< Use the newer mesh-error formulation.
    bool detect_control_switches_ = false; ///< Detect and refine around control switches.

    double rel_switch_tol_ = .3;     ///< Relative tolerance for control-switch detection.
    double abs_switch_tol_ = 1.0e-6; ///< Absolute tolerance for control-switch detection.

    double mesh_tol_ = 1.0e-6; ///< Target mesh-error tolerance.

    MeshErrorEstimators mesh_error_estimator_ =
        MeshErrorEstimators::DEBOOR; ///< Mesh-error estimator.
    MeshErrorAggregation mesh_error_criteria_ =
        MeshErrorAggregation::MAX; ///< Convergence aggregation.
    MeshErrorAggregation mesh_error_distributor_ =
        MeshErrorAggregation::AVG; ///< Distribution aggregation.
    PSIOPT::ConvergenceFlags mesh_abort_flag_ =
        PSIOPT::ConvergenceFlags::DIVERGING; ///< Solver flag that aborts the mesh loop.

    bool mesh_converged_ = false; ///< Whether the mesh has met the tolerance.

    std::vector<MeshIterateInfo> mesh_iters_; ///< Per-iteration mesh-refinement diagnostics.

    /// @brief Enable or disable automatic scaling and invalidate cached transcription.
    /// @param autoscale  Whether to enable automatic scaling.
    void set_auto_scaling(bool autoscale) {
        this->auto_scaling_ = autoscale;
        this->reset_transcription();
        this->invalidate_post_opt_info();
    }

    /// @brief Enable or disable adaptive mesh refinement.
    /// @param amesh  Whether to enable adaptive mesh refinement.
    void set_adaptive_mesh(bool amesh) { this->adaptive_mesh_ = amesh; }
    /// @brief Set the target mesh-error tolerance.
    /// @param t  Tolerance (its absolute value is used).
    void set_mesh_tol(double t) { this->mesh_tol_ = abs(t); }
    /// @brief Set the segment-count reduction factor.
    /// @param t  Reduction factor (its absolute value is used).
    void set_mesh_red_factor(double t) { this->mesh_red_factor_ = abs(t); }
    /// @brief Set the segment-count increase factor.
    /// @param t  Increase factor (its absolute value is used).
    void set_mesh_inc_factor(double t) { this->mesh_inc_factor_ = abs(t); }
    /// @brief Set the error-overshoot factor that triggers refinement.
    /// @param t  Error factor (its absolute value is used).
    void set_mesh_err_factor(double t) { this->mesh_err_factor_ = abs(t); }
    /// @brief Set the maximum number of mesh-refinement iterations.
    /// @param it  Iteration cap (its absolute value is used).
    void set_max_mesh_iters(int it) { this->max_mesh_iters_ = abs(it); }
    /// @brief Set the minimum number of mesh segments.
    /// @param it  Minimum segment count (its absolute value is used).
    void set_min_segments(int it) { this->min_segments_ = abs(it); }
    /// @brief Set the maximum number of mesh segments.
    /// @param it  Maximum segment count (its absolute value is used).
    void set_max_segments(int it) { this->max_segments_ = abs(it); }
    /// @brief Set the mesh-error convergence aggregation.
    /// @param m  Aggregation mode.
    void set_mesh_error_criteria(MeshErrorAggregation m) { this->mesh_error_criteria_ = m; }
    /// @brief Set the mesh-error convergence aggregation by name.
    /// @param m  Aggregation name (see @c strto_MeshErrorAggregation).
    void set_mesh_error_criteria(const std::string &m) {
        this->mesh_error_criteria_ = strto_MeshErrorAggregation(m);
    }
    /// @brief Set the mesh-error estimator.
    /// @param m  Estimator mode.
    void set_mesh_error_estimator(MeshErrorEstimators m) { this->mesh_error_estimator_ = m; }
    /// @brief Set the mesh-error estimator by name.
    /// @param m  Estimator name (see @c strto_MeshErrorEstimator).
    void set_mesh_error_estimator(const std::string &m) {
        this->mesh_error_estimator_ = strto_MeshErrorEstimator(m);
    }
    /// @brief Set the aggregation used to distribute the mesh-error density.
    /// @param m  Aggregation mode.
    void set_mesh_error_distributor(MeshErrorAggregation m) { this->mesh_error_distributor_ = m; }
    /// @brief Set the mesh-error distribution aggregation by name.
    /// @param m  Aggregation name (see @c strto_MeshErrorAggregation).
    void set_mesh_error_distributor(const std::string &m) {
        this->mesh_error_distributor_ = strto_MeshErrorAggregation(m);
    }

    /// @brief Return the per-iteration mesh-refinement diagnostics.
    /// @return Vector of @ref MeshIterateInfo, one per mesh iteration.
    std::vector<MeshIterateInfo> get_mesh_iters() const { return this->mesh_iters_; }

    ///////////////////

  public:
    /////////////////////////////////////////////////////////////////////////////

    /// @brief Enable or disable SuperScalar (vectorized) function evaluation.
    /// @param b  Whether to enable vectorization.
    void enable_vectorization(bool b) { this->enable_vectorization_ = b; }
    /// @brief Mark the phase as needing (re-)transcription before the next solve.
    void reset_transcription() { this->do_transcription_ = true; };

    /// @brief Invalidate cached post-solve constraint/multiplier data.
    void invalidate_post_opt_info() { this->post_opt_info_valid_ = false; };

    /// @brief Default constructor; dimensions must be set before use.
    ODEPhaseBase() {}
    /// @brief Construct with the phase's ODE dimensions.
    /// @param Xv  Number of state variables.
    /// @param Uv  Number of control variables.
    /// @param Pv  Number of ODE-parameter variables.
    ODEPhaseBase(int Xv, int Uv, int Pv) {
        this->set_xvars(Xv);
        this->set_uvars(Uv);
        this->set_pvars(Pv);

        this->xtup_units_ = Eigen::VectorXd::Ones(this->xtu_p_vars());
    }
    /// @brief Virtual destructor.
    virtual ~ODEPhaseBase() = default;

    //////////////////////////////////////////////////

    /// @brief Set the scaling units for the state/time/control/ODE-param block.
    /// @param xtup_units  Per-variable scaling units.
    virtual void set_units(const Eigen::VectorXd &xtup_units) = 0;

    /// @brief Set the control representation mode.
    /// @param m  Control mode (e.g. spline order or block-constant).
    virtual void set_control_mode(ControlModes m) {
        this->reset_transcription();
        this->invalidate_post_opt_info();
        this->control_mode_ = m;
        if (this->control_mode_ == ControlModes::BlockConstant) {
            this->table_.blocked_controls_ = true;
        } else {
            this->table_.blocked_controls_ = false;
        }
    }

    /// @brief Set the control representation mode by name.
    /// @param m  Control-mode name (see @c strto_ControlMode).
    void set_control_mode(std::string m) { this->set_control_mode(strto_ControlMode(m)); }

    /// @brief Set the integral quadrature mode.
    /// @param m  Integral mode.
    virtual void set_integral_mode(IntegralModes m) {
        this->reset_transcription();
        this->integral_mode_ = m;
    }
    /// @brief Set the transcription (collocation) scheme.
    /// @param m  Transcription mode.
    virtual void set_transcription_mode(TranscriptionModes m) = 0;

    /// @brief Switch transcription mode and re-load the trajectory onto a new mesh.
    /// @param m    New transcription mode.
    /// @param DBS  Defect-bin spacing for the new mesh.
    /// @param DPB  Defects per bin for the new mesh.
    void switch_transcription_mode(TranscriptionModes m, VectorXd DBS, VectorXi DPB) {
        this->set_transcription_mode(m);
        this->set_traj(this->active_traj_, DBS, DPB);
    }
    /// @brief Switch transcription mode, keeping the current mesh spacing.
    /// @param m  New transcription mode.
    void switch_transcription_mode(TranscriptionModes m) {
        this->switch_transcription_mode(m, this->def_bin_spacing_, this->defs_per_bin_);
    }

    /// @brief Switch transcription mode by name onto a new mesh.
    /// @param m    New transcription-mode name (see @c strto_TranscriptionMode).
    /// @param DBS  Defect-bin spacing for the new mesh.
    /// @param DPB  Defects per bin for the new mesh.
    void switch_transcription_mode(std::string m, VectorXd DBS, VectorXi DPB) {
        this->switch_transcription_mode(strto_TranscriptionMode(m), DBS, DPB);
    }
    /// @brief Switch transcription mode by name, keeping the current mesh spacing.
    /// @param m  New transcription-mode name (see @c strto_TranscriptionMode).
    void switch_transcription_mode(std::string m) {
        this->switch_transcription_mode(strto_TranscriptionMode(m));
    }

    ///////////////////////////////////////////////////

    /// @brief Replace all named static-parameter variable groups.
    /// @param spidxs  Map of group name to static-parameter indices.
    void set_static_param_vgroups(std::map<std::string, Eigen::VectorXi> spidxs) {
        this->sp_idxs_ = spidxs;
    }
    /// @brief Merge additional named static-parameter variable groups.
    /// @param spidxs  Map of group name to static-parameter indices to add.
    void add_static_param_vgroups(std::map<std::string, Eigen::VectorXi> spidxs) {
        for (auto &[key, value] : spidxs) {
            this->sp_idxs_[key] = value;
        }
    }
    /// @brief Add a named static-parameter variable group.
    /// @param idx  Static-parameter indices in the group.
    /// @param key  Group name.
    void add_static_param_vgroup(Eigen::VectorXi idx, std::string key) {
        this->sp_idxs_[key] = idx;
    }
    /// @brief Add a named static-parameter variable group of a single index.
    /// @param idx  The single static-parameter index.
    /// @param key  Group name.
    void add_static_param_vgroup(int idx, std::string key) {
        VectorXi tmp(1);
        tmp << idx;
        this->sp_idxs_[key] = tmp;
    }

    /// @brief Look up a named static-parameter variable group.
    /// @param key  Group name.
    /// @return The static-parameter indices in the group.
    /// @throws std::invalid_argument if no group named @p key exists.
    VectorXi get_sp_idx(std::string key) const {
        if (sp_idxs_.count(key) == 0) {
            throw std::invalid_argument(
                fmt::format("No StaticParam variable index group with name: {0:} exists.", key));
        }
        return this->sp_idxs_.at(key);
    }

    /////////////////////////////////////////////////
    /// @internal
    /// @brief Remove a registered function from a function map and invalidate transcription.
    /// @tparam FuncMap  The function-map type.
    /// @param map      The map to remove from.
    /// @param index    Function index to remove, or -1 for the most recently added.
    /// @param funcstr  Human-readable function kind, for error messages.
    /// @throws std::invalid_argument if no function with @p index exists.
    /// @endinternal
    template <class FuncMap>
    void remove_func_impl(FuncMap &map, int index, const std::string &funcstr) {
        // Validate before any mutation: an out-of-range index throws, and
        // pre-flipping reset_transcription_/post_opt_info_ would force a
        // wasteful re-transcription on a subsequent solve() under the
        // exception path.
        if (index == -1 && map.size() > 0) {
            index = map.rbegin()->first;
        }

        if (map.count(index) == 0) {
            throw std::invalid_argument(
                fmt::format("No {0:} with index {1:} exists in phase.", funcstr, index));
        }

        this->reset_transcription();
        this->invalidate_post_opt_info();
        map.erase(index);
    }

    /// @internal
    /// @brief Add a function to a function map after validating its IO size.
    /// @tparam FuncType  The function type.
    /// @tparam FuncMap   The function-map type.
    /// @param func     The function to add.
    /// @param map      The map to add to.
    /// @param funcstr  Human-readable function kind, for error messages.
    /// @return The index assigned to the added function.
    /// @endinternal
    template <class FuncType, class FuncMap>
    int add_func_impl(FuncType func, FuncMap &map, const std::string &funcstr) {
        // Validate before any mutation: check_function_size can throw, and
        // any state we mutate before the throw would leak into the phase
        // and duplicate on retry.
        int index = map.size() == 0 ? 0 : map.rbegin()->first + 1;
        func.storage_index_ = index;
        check_function_size(func, funcstr);

        this->reset_transcription();
        this->invalidate_post_opt_info();
        map[index] = std::move(func);
        return index;
    }

    /// @internal
    /// @brief Resolve a @ref RegionType selector to a concrete @ref PhaseRegionFlags.
    /// @param reg_t  Region as an enum value or its string name.
    /// @return The corresponding region enumerator.
    /// @endinternal
    PhaseRegionFlags get_region(RegionType reg_t) const {
        PhaseRegionFlags reg;

        if (std::holds_alternative<PhaseRegionFlags>(reg_t)) {
            reg = std::get<PhaseRegionFlags>(reg_t);
        } else if (std::holds_alternative<std::string>(reg_t)) {
            reg = strto_PhaseRegionFlag(std::get<std::string>(reg_t));
        }
        return reg;
    }

    /// @internal
    /// @brief Resolve a state/time/control/param variable selector to absolute indices.
    /// @param reg          The region the indices apply to.
    /// @param xtup_vars_t  Variable selector (index, vector, or named group(s)).
    /// @return Absolute (or zero-based, for @c ODEParams) variable indices.
    /// @endinternal
    VectorXi get_xt_up_vars(PhaseRegionFlags reg, VarIndexType xtup_vars_t) const {

        VectorXi xtup_vars;

        /////////////////////////////////////////////////
        if (std::holds_alternative<int>(xtup_vars_t)) {
            xtup_vars.resize(1);
            xtup_vars[0] = std::get<int>(xtup_vars_t);
        } else if (std::holds_alternative<VectorXi>(xtup_vars_t)) {
            xtup_vars = std::get<VectorXi>(xtup_vars_t);
        } else if (std::holds_alternative<std::string>(xtup_vars_t)) {
            if (reg != PhaseRegionFlags::StaticParams) {
                xtup_vars = this->idx(std::get<std::string>(xtup_vars_t));
            } else {
                xtup_vars = this->get_sp_idx(std::get<std::string>(xtup_vars_t));
            }
            if (reg == PhaseRegionFlags::ODEParams) {
                // Convert to 0 based index
                for (int i = 0; i < xtup_vars.size(); i++) {
                    xtup_vars[i] -= this->xtu_vars();
                }
            }
        } else if (std::holds_alternative<std::vector<std::string>>(xtup_vars_t)) {

            std::vector<VectorXi> varvec;
            int size = 0;

            auto tmpvars = std::get<std::vector<std::string>>(xtup_vars_t);

            for (auto tmpv : tmpvars) {
                if (reg != PhaseRegionFlags::StaticParams) {
                    varvec.push_back(this->idx(tmpv));
                } else {
                    varvec.push_back(this->get_sp_idx(tmpv));
                }

                size += varvec.back().size();
            }
            xtup_vars.resize(size);

            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    xtup_vars[next] = varv[i];
                    next++;
                }
            }

            if (reg == PhaseRegionFlags::ODEParams) {
                // Convert to 0 based index
                for (int i = 0; i < xtup_vars.size(); i++) {
                    xtup_vars[i] -= this->xtu_vars();
                }
            }
        }
        return xtup_vars;
    }

    /// @internal
    /// @brief Resolve an ODE-parameter variable selector to zero-based indices.
    /// @param reg       The region the indices apply to.
    /// @param OPvars_t  ODE-parameter selector (index, vector, or named group(s)).
    /// @return Zero-based ODE-parameter indices.
    /// @endinternal
    VectorXi get_op_vars(PhaseRegionFlags reg, VarIndexType OPvars_t) const {

        VectorXi OPvars;

        if (std::holds_alternative<int>(OPvars_t)) {
            OPvars.resize(1);
            OPvars[0] = std::get<int>(OPvars_t);
        } else if (std::holds_alternative<VectorXi>(OPvars_t)) {
            OPvars = std::get<VectorXi>(OPvars_t);
        } else if (std::holds_alternative<std::string>(OPvars_t)) {
            OPvars = this->idx(std::get<std::string>(OPvars_t));

            for (int i = 0; i < OPvars.size(); i++) {
                // Convert to 0 based index
                OPvars[i] -= this->xtu_vars();
            }
        } else if (std::holds_alternative<std::vector<std::string>>(OPvars_t)) {
            std::vector<VectorXi> varvec;
            int size = 0;
            auto tmpvars = std::get<std::vector<std::string>>(OPvars_t);
            for (auto tmpv : tmpvars) {
                varvec.push_back(this->idx(tmpv));
                size += varvec.back().size();
            }
            OPvars.resize(size);

            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    // Convert to 0 based index
                    OPvars[next] = varv[i] - this->xtu_vars();
                    next++;
                }
            }
        }

        return OPvars;
    }

    /// @internal
    /// @brief Resolve a static-parameter variable selector to absolute indices.
    /// @param reg       The region the indices apply to.
    /// @param SPvars_t  Static-parameter selector (index, vector, or named group(s)).
    /// @return Static-parameter indices.
    /// @endinternal
    VectorXi get_sp_vars(PhaseRegionFlags reg, VarIndexType SPvars_t) const {

        VectorXi SPvars;

        if (std::holds_alternative<int>(SPvars_t)) {
            SPvars.resize(1);
            SPvars[0] = std::get<int>(SPvars_t);
        } else if (std::holds_alternative<VectorXi>(SPvars_t)) {
            SPvars = std::get<VectorXi>(SPvars_t);
        } else if (std::holds_alternative<std::string>(SPvars_t)) {
            SPvars = this->get_sp_idx(std::get<std::string>(SPvars_t));
        } else if (std::holds_alternative<std::vector<std::string>>(SPvars_t)) {
            std::vector<VectorXi> varvec;
            int size = 0;
            auto tmpvars = std::get<std::vector<std::string>>(SPvars_t);
            for (auto tmpv : tmpvars) {
                varvec.push_back(this->get_sp_idx(tmpv));
                size += varvec.back().size();
            }
            SPvars.resize(size);
            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    SPvars[next] = varv[i];
                    next++;
                }
            }
        }

        return SPvars;
    }

    /// @internal
    /// @brief Build a region-bound @ref StateFunction from user selectors.
    /// @tparam FuncHolder  The StateFunction holder type to produce.
    /// @tparam FuncType    The wrapped function type.
    /// @param reg_t        Region selector.
    /// @param fun          The user function.
    /// @param xtup_vars_t  State/time/control/param variable selector (or "All").
    /// @param OPvars_t     ODE-parameter selector.
    /// @param SPvars_t     Static-parameter selector.
    /// @param scale_t      Output-scale selector.
    /// @return The constructed region-bound function holder.
    /// @endinternal
    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(RegionType reg_t, FuncType fun, VarIndexType xtup_vars_t,
                              VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        PhaseRegionFlags reg = get_region(reg_t);
        FuncHolder func;

        if (std::holds_alternative<std::string>(xtup_vars_t)) {
            std::string vars = std::get<std::string>(xtup_vars_t);
            if (vars == "All" || vars == "all" || vars == "XtUP") {
                // Default case where the function has same inputs and order as ODE
                VectorXi xtup_vars;
                VectorXi OPvars;
                VectorXi SPvars;
                xtup_vars.setLinSpaced(this->xtu_vars(), 0, this->xtu_vars() - 1);
                if (this->p_vars() > 0) {
                    OPvars.setLinSpaced(this->p_vars(), 0, this->p_vars() - 1);
                }
                func = FuncHolder(fun, reg, xtup_vars, OPvars, SPvars, scale_t);
                return func; // return early
            }
        }

        VectorXi xtup_vars = this->get_xt_up_vars(reg, xtup_vars_t);
        VectorXi OPvars;
        VectorXi SPvars;
        /////////////////////////////////////////////////
        if (reg != PhaseRegionFlags::ODEParams && reg != PhaseRegionFlags::StaticParams) {
            // If region is Params then the indices are held in xtup_vars_t and the others are emtpy
            OPvars = this->get_op_vars(reg, OPvars_t);
            SPvars = this->get_sp_vars(reg, SPvars_t);
            func = FuncHolder(fun, reg, xtup_vars, OPvars, SPvars, scale_t);

        } else {
            func = FuncHolder(fun, reg, xtup_vars, scale_t);
        }

        return func;
    }

    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    /// @brief Add a pre-built equality constraint to the phase.
    /// @param con  The region-bound equality constraint.
    /// @return The index assigned to the constraint.
    int add_equal_con(StateConstraint con) {
        return add_func_impl(con, this->user_equalities_, "Equality Constraint");
    }

    /// @brief Add an equality constraint @f$f=0@f$ over a region with full bindings.
    /// @param reg_t       Region selector.
    /// @param fun         The constraint function.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param OPvars_t    ODE-parameter selector.
    /// @param SPvars_t    Static-parameter selector.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_equal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType xtup_vars_t,
                      VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                      OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_equalities_, "Equality Constraint");
    }

    /// @brief Add an equality constraint @f$f=0@f$ over a region (state vars only).
    /// @param reg_t       Region selector.
    /// @param fun         The constraint function.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_equal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType xtup_vars_t,
                      ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                      empty, empty, scale_t);
        return add_func_impl(con, this->user_equalities_, "Equality Constraint");
    }

    /// @brief Fix selected region variables to a given value (boundary-value constraint).
    /// @param reg     Region selector.
    /// @param args    Variable selector.
    /// @param value   Target value (scalar broadcast, or per-variable vector).
    /// @param scale_t Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_boundary_value(RegionType reg, VarIndexType args,
                           const std::variant<double, VectorXd> &value, ScaleType scale_t);
    /// @brief Constrain the front-to-back change of a variable to a given value.
    /// @param var      Variable selector.
    /// @param value    Target change (back minus front).
    /// @param scale    Constraint scaling.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_delta_var_equal_con(VarIndexType var, double value, double scale, ScaleType scale_t);
    /// @brief Constrain the total elapsed time of the phase to a given value.
    /// @param value    Target elapsed time.
    /// @param scale    Constraint scaling.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_delta_time_equal_con(double value, double scale, ScaleType scale_t) {
        return this->add_delta_var_equal_con(this->t_var(), value, scale, scale_t);
    }
    /// @brief Lock selected region variables to their current trajectory values.
    /// @param reg      Region selector.
    /// @param args     Variable selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_value_lock(RegionType reg, VarIndexType args, ScaleType scale_t);
    /// @brief Enforce periodicity (front == back) on selected variables.
    /// @param args     Variable selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_periodicity_con(VarIndexType args, ScaleType scale_t);

    /////////////////////////////////////////////////

    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    /// @brief Add a pre-built inequality constraint to the phase.
    /// @param con  The region-bound inequality constraint.
    /// @return The index assigned to the constraint.
    int add_inequal_con(StateConstraint con) {
        return add_func_impl(con, this->user_inequalities_, "Inequality Constraint");
    }
    /// @brief Add an inequality constraint @f$f\le 0@f$ over a region with full bindings.
    /// @param reg_t       Region selector.
    /// @param fun         The constraint function.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param OPvars_t    ODE-parameter selector.
    /// @param SPvars_t    Static-parameter selector.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_inequal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType xtup_vars_t,
                        VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                      OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_inequalities_, "Inequality Constraint");
    }

    /// @brief Add an inequality constraint @f$f\le 0@f$ over a region (state vars only).
    /// @param reg_t       Region selector.
    /// @param fun         The constraint function.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_inequal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType xtup_vars_t,
                        ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                      empty, empty, scale_t);
        return add_func_impl(con, this->user_inequalities_, "Inequality Constraint");
    }

    ////////////////////////
    /// @brief Bound a variable below and above with separate bound scales.
    /// @param reg         Region selector.
    /// @param var         Variable selector.
    /// @param lowerbound  Lower bound.
    /// @param upperbound  Upper bound.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_var_bound(RegionType reg, VarIndexType var, double lowerbound, double upperbound,
                         double lbscale, double ubscale, ScaleType scale_t);

    /// @brief Bound a variable below and above with a shared bound scale.
    /// @param reg         Region selector.
    /// @param var         Variable selector.
    /// @param lowerbound  Lower bound.
    /// @param upperbound  Upper bound.
    /// @param scale       Shared bound scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_var_bound(RegionType reg, VarIndexType var, double lowerbound, double upperbound,
                         double scale, ScaleType scale_t) {
        return this->add_lu_var_bound(reg, var, lowerbound, upperbound, scale, scale, scale_t);
    }
    /// @brief Bound a variable below and above with unit bound scales.
    /// @param reg         Region selector.
    /// @param var         Variable selector.
    /// @param lowerbound  Lower bound.
    /// @param upperbound  Upper bound.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_var_bound(RegionType reg, VarIndexType var, double lowerbound, double upperbound,
                         ScaleType scale_t) {
        return this->add_lu_var_bound(reg, var, lowerbound, upperbound, 1.0, 1.0, scale_t);
    }

    /// @brief Bound a variable from below.
    /// @param reg         Region selector.
    /// @param var         Variable selector.
    /// @param lowerbound  Lower bound.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_lower_var_bound(RegionType reg, VarIndexType var, double lowerbound, double lbscale,
                            ScaleType scale_t);

    /// @brief Bound a variable from above.
    /// @param reg         Region selector.
    /// @param var         Variable selector.
    /// @param upperbound  Upper bound.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_upper_var_bound(RegionType reg, VarIndexType var, double upperbound, double ubscale,
                            ScaleType scale_t);

    /// @brief Bound a scalar function of region variables below and above.
    /// @param reg         Region selector.
    /// @param func        The scalar function to bound.
    /// @param xtup_vars   State/time/control variable selector.
    /// @param OPvars      ODE-parameter selector.
    /// @param SPvars      Static-parameter selector.
    /// @param lowerbound  Lower bound on the function value.
    /// @param upperbound  Upper bound on the function value.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                          VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                          double upperbound, double lbscale, double ubscale, ScaleType scale_t);

    /// @brief Bound a scalar function below and above (state vars only).
    /// @param reg         Region selector.
    /// @param func        The scalar function to bound.
    /// @param xtup_vars   State/time/control variable selector.
    /// @param lowerbound  Lower bound on the function value.
    /// @param upperbound  Upper bound on the function value.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                          double lowerbound, double upperbound, double lbscale, double ubscale,
                          ScaleType scale_t) {

        VectorXi empty;

        return add_lu_func_bound(reg, func, xtup_vars, empty, empty, lowerbound, upperbound,
                                 lbscale, ubscale, scale_t);
    }

    /// @brief Bound a scalar function below and above with a shared bound scale.
    /// @param reg         Region selector.
    /// @param func        The scalar function to bound.
    /// @param xtup_vars   State/time/control variable selector.
    /// @param OPvars      ODE-parameter selector.
    /// @param SPvars      Static-parameter selector.
    /// @param lowerbound  Lower bound on the function value.
    /// @param upperbound  Upper bound on the function value.
    /// @param scale       Shared bound scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                          VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                          double upperbound, double scale, ScaleType scale_t) {
        return add_lu_func_bound(reg, func, xtup_vars, OPvars, SPvars, lowerbound, upperbound,
                                 scale, scale, scale_t);
    }

    /// @brief Bound a scalar function below and above with a shared scale (state vars only).
    /// @param reg         Region selector.
    /// @param func        The scalar function to bound.
    /// @param xtup_vars   State/time/control variable selector.
    /// @param lowerbound  Lower bound on the function value.
    /// @param upperbound  Upper bound on the function value.
    /// @param scale       Shared bound scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                          double lowerbound, double upperbound, double scale, ScaleType scale_t) {
        VectorXi empty;
        return add_lu_func_bound(reg, func, xtup_vars, empty, empty, lowerbound, upperbound, scale,
                                 scale, scale_t);
    }

    /// @brief Bound a scalar function of region variables from below.
    /// @param reg         Region selector.
    /// @param func        The scalar function to bound.
    /// @param xtup_vars   State/time/control variable selector.
    /// @param OPvars      ODE-parameter selector.
    /// @param SPvars      Static-parameter selector.
    /// @param lowerbound  Lower bound on the function value.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_lower_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                             VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                             double lbscale, ScaleType scale_t);

    /// @brief Bound a scalar function from below (state vars only).
    /// @param reg         Region selector.
    /// @param func        The scalar function to bound.
    /// @param xtup_vars   State/time/control variable selector.
    /// @param lowerbound  Lower bound on the function value.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_lower_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                             double lowerbound, double lbscale, ScaleType scale_t) {
        VectorXi empty;

        return this->add_lower_func_bound(reg, func, xtup_vars, empty, empty, lowerbound, lbscale,
                                          scale_t);
    }

    /// @brief Bound a scalar function of region variables from above.
    /// @param reg         Region selector.
    /// @param func        The scalar function to bound.
    /// @param xtup_vars   State/time/control variable selector.
    /// @param OPvars      ODE-parameter selector.
    /// @param SPvars      Static-parameter selector.
    /// @param upperbound  Upper bound on the function value.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_upper_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                             VarIndexType OPvars, VarIndexType SPvars, double upperbound,
                             double ubscale, ScaleType scale_t);

    /// @brief Bound a scalar function from above (state vars only).
    /// @param reg         Region selector.
    /// @param func        The scalar function to bound.
    /// @param xtup_vars   State/time/control variable selector.
    /// @param upperbound  Upper bound on the function value.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_upper_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                             double upperbound, double ubscale, ScaleType scale_t) {
        VectorXi empty;

        return this->add_upper_func_bound(reg, func, xtup_vars, empty, empty, upperbound, ubscale,
                                          scale_t);
    }

    /// @brief Bound the Euclidean norm of selected variables below and above.
    /// @param reg         Region selector.
    /// @param xtup_vars   Variable selector whose norm is bounded.
    /// @param lowerbound  Lower bound on the norm.
    /// @param upperbound  Upper bound on the norm.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                          double upperbound, double lbscale, double ubscale, ScaleType scale_t);

    /// @brief Bound the Euclidean norm below and above with a shared bound scale.
    /// @param reg         Region selector.
    /// @param xtup_vars   Variable selector whose norm is bounded.
    /// @param lowerbound  Lower bound on the norm.
    /// @param upperbound  Upper bound on the norm.
    /// @param scale       Shared bound scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                          double upperbound, double scale, ScaleType scale_t) {
        return this->add_lu_norm_bound(reg, xtup_vars, lowerbound, upperbound, scale, scale,
                                       scale_t);
    }

    /// @brief Bound the squared Euclidean norm of selected variables below and above.
    /// @param reg         Region selector.
    /// @param xtup_vars   Variable selector whose squared norm is bounded.
    /// @param lowerbound  Lower bound on the squared norm.
    /// @param upperbound  Upper bound on the squared norm.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_squared_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                                  double upperbound, double lbscale, double ubscale,
                                  ScaleType scale_t);

    /// @brief Bound the squared norm below and above with a shared bound scale.
    /// @param reg         Region selector.
    /// @param xtup_vars   Variable selector whose squared norm is bounded.
    /// @param lowerbound  Lower bound on the squared norm.
    /// @param upperbound  Upper bound on the squared norm.
    /// @param scale       Shared bound scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_squared_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                                  double upperbound, double scale, ScaleType scale_t) {
        return this->add_lu_squared_norm_bound(reg, xtup_vars, lowerbound, upperbound, scale, scale,
                                               scale_t);
    }

    /// @brief Bound the Euclidean norm of selected variables from below.
    /// @param reg         Region selector.
    /// @param xtup_vars   Variable selector whose norm is bounded.
    /// @param lowerbound  Lower bound on the norm.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_lower_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                             double lbscale, ScaleType scale_t);

    /// @brief Bound the squared Euclidean norm of selected variables from below.
    /// @param reg         Region selector.
    /// @param xtup_vars   Variable selector whose squared norm is bounded.
    /// @param lowerbound  Lower bound on the squared norm.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_lower_squared_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                                     double lbscale, ScaleType scale_t);

    /// @brief Bound the Euclidean norm of selected variables from above.
    /// @param reg         Region selector.
    /// @param xtup_vars   Variable selector whose norm is bounded.
    /// @param upperbound  Upper bound on the norm.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_upper_norm_bound(RegionType reg, VarIndexType xtup_vars, double upperbound,
                             double ubscale, ScaleType scale_t);

    /// @brief Bound the squared Euclidean norm of selected variables from above.
    /// @param reg         Region selector.
    /// @param xtup_vars   Variable selector whose squared norm is bounded.
    /// @param upperbound  Upper bound on the squared norm.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_upper_squared_norm_bound(RegionType reg, VarIndexType xtup_vars, double upperbound,
                                     double ubscale, ScaleType scale_t);
    //
    /// @brief Bound the front-to-back change of a variable from below.
    /// @param reg         Region selector.
    /// @param var         Variable selector.
    /// @param lowerbound  Lower bound on the change.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_lower_delta_var_bound(RegionType reg, VarIndexType var, double lowerbound,
                                  double lbscale, ScaleType scale_t);
    /// @brief Bound the front-to-back change of a variable from below (front-and-back region).
    /// @param var         Variable selector.
    /// @param lowerbound  Lower bound on the change.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_lower_delta_var_bound(VarIndexType var, double lowerbound, double lbscale,
                                  ScaleType scale_t) {
        return this->add_lower_delta_var_bound(PhaseRegionFlags::FrontandBack, var, lowerbound,
                                               lbscale, scale_t);
    }
    /// @brief Bound the elapsed time of the phase from below.
    /// @param lowerbound  Lower bound on the elapsed time.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_lower_delta_time_bound(double lowerbound, double lbscale, ScaleType scale_t) {
        return this->add_lower_delta_var_bound(this->t_var(), lowerbound, lbscale, scale_t);
    }
    ///
    /// @brief Bound the front-to-back change of a variable from above.
    /// @param reg         Region selector.
    /// @param var         Variable selector.
    /// @param upperbound  Upper bound on the change.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_upper_delta_var_bound(RegionType reg, VarIndexType var, double upperbound,
                                  double ubscale, ScaleType scale_t);
    /// @brief Bound the front-to-back change of a variable from above (front-and-back region).
    /// @param var         Variable selector.
    /// @param upperbound  Upper bound on the change.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_upper_delta_var_bound(VarIndexType var, double upperbound, double ubscale,
                                  ScaleType scale_t) {
        return this->add_upper_delta_var_bound(PhaseRegionFlags::FrontandBack, var, upperbound,
                                               ubscale, scale_t);
    }
    /// @brief Bound the elapsed time of the phase from above.
    /// @param upperbound  Upper bound on the elapsed time.
    /// @param ubscale     Upper-bound constraint scale.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the bound constraint.
    int add_upper_delta_time_bound(double upperbound, double ubscale, ScaleType scale_t) {
        return this->add_upper_delta_var_bound(this->t_var(), upperbound, ubscale, scale_t);
    }

    ///////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    /// @brief Bound a variable below and above using concrete enum/index arguments.
    /// @param reg         The phase region.
    /// @param var         Variable index within the region.
    /// @param lowerbound  Lower bound.
    /// @param upperbound  Upper bound.
    /// @param lbscale     Lower-bound constraint scale.
    /// @param ubscale     Upper-bound constraint scale.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_var_bound(PhaseRegionFlags reg, int var, double lowerbound, double upperbound,
                         double lbscale, double ubscale);

    /// @brief Bound a variable below and above with a shared scale (enum/index arguments).
    /// @param reg         The phase region.
    /// @param var         Variable index within the region.
    /// @param lowerbound  Lower bound.
    /// @param upperbound  Upper bound.
    /// @param scale       Shared bound scale.
    /// @return The index assigned to the bound constraint(s).
    int add_lu_var_bound(PhaseRegionFlags reg, int var, double lowerbound, double upperbound,
                         double scale) {
        return this->add_lu_var_bound(reg, var, lowerbound, upperbound, scale, scale);
    }

    /// @brief Bound several variables below and above with a shared scale.
    /// @param reg         The phase region.
    /// @param vars        Variable indices within the region.
    /// @param lowerbound  Lower bound applied to each.
    /// @param upperbound  Upper bound applied to each.
    /// @param scale       Shared bound scale.
    /// @return Vector of indices assigned to the bound constraints.
    Eigen::VectorXi add_lu_var_bounds(PhaseRegionFlags reg, Eigen::VectorXi vars, double lowerbound,
                                      double upperbound, double scale) {
        Eigen::VectorXi cnums(vars.size());
        for (int i = 0; i < cnums.size(); i++) {
            cnums[i] = this->add_lu_var_bound(reg, vars[i], lowerbound, upperbound, scale, scale);
        }

        return cnums;
    }

    /// @brief Bound several variables below and above with a shared scale (region by name).
    /// @param reg         The phase region name.
    /// @param vars        Variable indices within the region.
    /// @param lowerbound  Lower bound applied to each.
    /// @param upperbound  Upper bound applied to each.
    /// @param scale       Shared bound scale.
    /// @return Vector of indices assigned to the bound constraints.
    Eigen::VectorXi add_lu_var_bounds(std::string reg, Eigen::VectorXi vars, double lowerbound,
                                      double upperbound, double scale) {
        return add_lu_var_bounds(strto_PhaseRegionFlag(reg), vars, lowerbound, upperbound, scale);
    }

    ////////////////////////////////////////////////////

    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    /// @brief Add a pre-built state objective to the phase.
    /// @param obj  The region-bound scalar objective.
    /// @return The index assigned to the objective.
    int add_state_objective(StateObjective obj) {
        return add_func_impl(obj, this->user_state_objectives_, "State Objective");
    }
    /// @brief Add a scalar objective over a region with full variable bindings.
    /// @param reg_t       Region selector.
    /// @param fun         The scalar objective function.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param OPvars_t    ODE-parameter selector.
    /// @param SPvars_t    Static-parameter selector.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the objective.
    int add_state_objective(RegionType reg_t, ScalarFunctionalX fun, VarIndexType xtup_vars_t,
                            VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                     OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_state_objectives_, "State Objective");
    }

    /// @brief Add a scalar objective over a region (state vars only).
    /// @param reg_t       Region selector.
    /// @param fun         The scalar objective function.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the objective.
    int add_state_objective(RegionType reg_t, ScalarFunctionalX fun, VarIndexType xtup_vars_t,
                            ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(reg_t, fun, xtup_vars_t, empty,
                                                                     empty, scale_t);
        return add_func_impl(con, this->user_state_objectives_, "State Objective");
    }
    /// @brief Add an objective equal to a scaled variable value.
    /// @param reg      Region selector.
    /// @param var      Variable selector.
    /// @param scale    Objective scale (sign sets minimize/maximize).
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_value_objective(RegionType reg, VarIndexType var, double scale, ScaleType scale_t);
    /// @brief Add an objective on the front-to-back change of a variable.
    /// @param var      Variable selector.
    /// @param scale    Objective scale.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_delta_var_objective(VarIndexType var, double scale, ScaleType scale_t);
    /// @brief Add an objective on the total elapsed time of the phase.
    /// @param scale    Objective scale.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_delta_time_objective(double scale, ScaleType scale_t) {
        return this->add_delta_var_objective(this->t_var(), scale, scale_t);
    }

    ///////////////////////////////////////////////

    ///////////////////////////////////////////////////
    /// @brief Add a pre-built integral (running-cost) objective to the phase.
    /// @param obj  The integrand objective.
    /// @return The index assigned to the integral objective.
    int add_integral_objective(StateObjective obj) {
        return add_func_impl(obj, this->user_integrands_, "Integral Objective");
    }

    /// @brief Add an integral objective @f$\int g\,dt@f$ with full variable bindings.
    /// @param fun         The scalar integrand.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param OPvars_t    ODE-parameter selector.
    /// @param SPvars_t    Static-parameter selector.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the integral objective.
    int add_integral_objective(ScalarFunctionalX fun, VarIndexType xtup_vars_t,
                               VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(
            PhaseRegionFlags::Path, fun, xtup_vars_t, OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_integrands_, "Integral Objective");
    }

    /// @brief Add an integral objective @f$\int g\,dt@f$ (state vars only).
    /// @param fun         The scalar integrand.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the integral objective.
    int add_integral_objective(ScalarFunctionalX fun, VarIndexType xtup_vars_t, ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(
            PhaseRegionFlags::Path, fun, xtup_vars_t, empty, empty, scale_t);
        return add_func_impl(con, this->user_integrands_, "Integral Objective");
    }

    ///////////////////////////////////////////////////
    /// @brief Add a pre-built integral-parameter function accumulating into a parameter.
    /// @param con  The integrand objective.
    /// @param pv   Index of the parameter the integral accumulates into.
    /// @return The index assigned to the integral-parameter function.
    int add_integral_param_function(StateObjective con, int pv) {
        VectorXi epv(1);
        epv[0] = pv;
        int index = add_func_impl(con, this->user_param_integrands_, "Integral Parameter Function");
        this->user_param_integrands_[index].ext_vars_ = epv;
        return index;
    }

    /// @brief Add an integral that accumulates into a parameter, with full bindings.
    /// @param fun         The scalar integrand.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param OPvars_t    ODE-parameter selector.
    /// @param SPvars_t    Static-parameter selector.
    /// @param accum_parm  Index of the parameter the integral accumulates into.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the integral-parameter function.
    int add_integral_param_function(ScalarFunctionalX fun, VarIndexType xtup_vars_t,
                                    VarIndexType OPvars_t, VarIndexType SPvars_t, int accum_parm,
                                    ScaleType scale_t) {

        VectorXi epv(1);
        epv[0] = accum_parm;

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(
            PhaseRegionFlags::Path, fun, xtup_vars_t, OPvars_t, SPvars_t, scale_t);
        int index = add_func_impl(con, this->user_param_integrands_, "Integral Parameter Function");
        this->user_param_integrands_[index].ext_vars_ = epv;
        return index;
    }

    /// @brief Add an integral that accumulates into a parameter (state vars only).
    /// @param fun         The scalar integrand.
    /// @param xtup_vars_t State/time/control variable selector.
    /// @param accum_parm  Index of the parameter the integral accumulates into.
    /// @param scale_t     Output-scale selector.
    /// @return The index assigned to the integral-parameter function.
    int add_integral_param_function(ScalarFunctionalX fun, VarIndexType xtup_vars_t, int accum_parm,
                                    ScaleType scale_t) {
        VectorXi empty;
        return add_integral_param_function(fun, xtup_vars_t, empty, empty, accum_parm, scale_t);
    }

    /////////////////////////////////////////////////

    /// @brief Remove a previously added equality constraint by index.
    /// @param index  Constraint index, or -1 for the most recently added.
    void remove_equal_con(int index) {
        this->remove_func_impl(this->user_equalities_, index, "Equality Constraint");
    }
    /// @brief Remove a previously added inequality constraint by index.
    /// @param index  Constraint index, or -1 for the most recently added.
    void remove_inequal_con(int index) {
        this->remove_func_impl(this->user_inequalities_, index, "Inequality Constraint");
    }
    /// @brief Remove a previously added state objective by index.
    /// @param index  Objective index, or -1 for the most recently added.
    void remove_state_objective(int index) {
        this->remove_func_impl(this->user_state_objectives_, index, "State Objective");
    }
    /// @brief Remove a previously added integral objective by index.
    /// @param index  Objective index, or -1 for the most recently added.
    void remove_integral_objective(int index) {
        this->remove_func_impl(this->user_integrands_, index, "Integral Objective");
    }
    /// @brief Remove a previously added integral-parameter function by index.
    /// @param index  Function index, or -1 for the most recently added.
    void remove_integral_param_function(int index) {
        this->remove_func_impl(this->user_param_integrands_, index, "Integral Parameter Function");
    }

    /////////////////////////////////////////////////

    /// @brief Set the initial trajectory and mesh, optionally interpolating onto it.
    /// @param mesh      Initial trajectory (one packed state vector per node).
    /// @param DBS       Normalized defect-bin spacing.
    /// @param DPB       Defects per bin.
    /// @param LerpTraj  Whether to linearly interpolate the mesh onto the new layout.
    virtual void set_traj(const std::vector<Eigen::VectorXd> &mesh, Eigen::VectorXd DBS,
                          Eigen::VectorXi DPB, bool LerpTraj);

    /// @brief Set the initial trajectory, inferring the mesh from the node count.
    /// @param mesh  Initial trajectory (one packed state vector per node).
    void set_traj(const std::vector<Eigen::VectorXd> &mesh);

    /// @brief Set the initial trajectory and mesh without interpolation.
    /// @param mesh  Initial trajectory (one packed state vector per node).
    /// @param DBS   Normalized defect-bin spacing.
    /// @param DPB   Defects per bin.
    void set_traj(const std::vector<Eigen::VectorXd> &mesh, Eigen::VectorXd DBS,
                  Eigen::VectorXi DPB) {
        this->set_traj(mesh, DBS, DPB, false);
    }

    /// @brief Set the initial trajectory on a uniform mesh of @p ndef defects.
    /// @param mesh      Initial trajectory (one packed state vector per node).
    /// @param ndef      Number of defect intervals in a single uniform bin.
    /// @param LerpTraj  Whether to linearly interpolate the mesh onto the new layout.
    void set_traj(const std::vector<Eigen::VectorXd> &mesh, int ndef, bool LerpTraj) {
        VectorXd dbs(2);
        dbs[0] = 0.0;
        dbs[1] = 1.0;
        VectorXi dpb(1);
        dpb[0] = ndef;
        this->set_traj(mesh, dbs, dpb, LerpTraj);
    }
    /// @brief Set the initial trajectory on a uniform mesh of @p ndef defects (no interpolation).
    /// @param mesh  Initial trajectory (one packed state vector per node).
    /// @param ndef  Number of defect intervals in a single uniform bin.
    void set_traj(const std::vector<Eigen::VectorXd> &mesh, int ndef) {
        this->set_traj(mesh, ndef, false);
    }

    /// @brief Re-mesh the current trajectory onto a manually specified distribution.
    /// @param DBS  Normalized defect-bin spacing.
    /// @param DPB  Defects per bin.
    void refine_traj_manual(VectorXd DBS, VectorXi DPB);

    /// @brief Re-mesh the current trajectory onto a uniform mesh of @p ndef defects.
    /// @param ndef  Number of defect intervals in a single uniform bin.
    void refine_traj_manual(int ndef) {
        VectorXd dbs(2);
        dbs[0] = 0.0;
        dbs[1] = 1.0;
        VectorXi dpb(1);
        dpb[0] = ndef;
        this->refine_traj_manual(dbs, dpb);
    }
    /// @brief Re-mesh the current trajectory onto @p n equally spaced defect intervals.
    /// @param n  Number of equally spaced defect intervals.
    /// @return The re-interpolated trajectory.
    std::vector<Eigen::VectorXd> refine_traj_equal(int n);

    /// @brief Run one automatic mesh-refinement step on the current trajectory.
    void refine_traj_auto();

    /// @brief Set the static (non-ODE) parameters and their scaling units.
    /// @param parm   Static-parameter values.
    /// @param units  Per-parameter scaling units (must match @p parm in size).
    /// @throws std::invalid_argument if @p units and @p parm differ in size.
    void set_static_params(VectorXd parm, VectorXd units) {
        if (units.size() != parm.size()) {
            throw std::invalid_argument(
                "Size of static parameter vector and scaling units vector must match");
        }

        this->active_static_params_ = parm;
        this->num_stat_params_ = parm.size();
        this->reset_transcription();
        this->sp_units_ = units;
    }
    /// @brief Set the static parameters with unit scaling.
    /// @param parm  Static-parameter values.
    void set_static_params(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->set_static_params(parm, units);
    }

    /// @brief Append additional static parameters and their scaling units.
    /// @param parm   Static-parameter values to append.
    /// @param units  Per-parameter scaling units for the appended values.
    void add_static_params(VectorXd parm, VectorXd units) {
        if (this->num_stat_params_ == 0) {
            this->set_static_params(parm, units);
        } else {
            VectorXd parmstmp(this->active_static_params_.size() + parm.size());
            parmstmp << this->active_static_params_, parm;
            VectorXd unitstmp(this->sp_units_.size() + units.size());
            unitstmp << this->sp_units_, units;
            this->set_static_params(parmstmp, unitstmp);
        }
    }
    /// @brief Append additional static parameters with unit scaling.
    /// @param parm  Static-parameter values to append.
    void add_static_params(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->add_static_params(parm, units);
    }

    /// @brief Update static-parameter values in place if the count matches; else replace.
    /// @param parm  New static-parameter values.
    void sub_static_params(VectorXd parm) {
        if (this->num_stat_params_ == parm.size()) {
            this->active_static_params_ = parm;
        } else {
            this->set_static_params(parm);
        }
    }

    /// @brief Substitute fixed values into selected region variables of the trajectory.
    /// @param reg      The phase region.
    /// @param indices  Variable indices to substitute.
    /// @param vals     Values to assign (one per index).
    void sub_variables(PhaseRegionFlags reg, VectorXi indices, VectorXd vals);
    /// @brief Substitute a fixed value into a single region variable.
    /// @param reg  The phase region.
    /// @param var  Variable index to substitute.
    /// @param val  Value to assign.
    void sub_variable(PhaseRegionFlags reg, int var, double val) {
        VectorXi indices(1);
        indices[0] = var;
        VectorXd vals(1);
        vals[0] = val;
        this->sub_variables(reg, indices, vals);
    }

    /// @brief Substitute fixed values into selected region variables (region by name).
    /// @param reg      The phase region name.
    /// @param indices  Variable indices to substitute.
    /// @param vals     Values to assign (one per index).
    void sub_variables(std::string reg, VectorXi indices, VectorXd vals) {
        this->sub_variables(strto_PhaseRegionFlag(reg), indices, vals);
    }
    /// @brief Substitute a fixed value into a single region variable (region by name).
    /// @param reg  The phase region name.
    /// @param var  Variable index to substitute.
    /// @param val  Value to assign.
    void sub_variable(std::string reg, int var, double val) {
        this->sub_variable(strto_PhaseRegionFlag(reg), var, val);
    }

    /// @brief Return the current discretized trajectory.
    /// @return One packed state vector per node.
    std::vector<Eigen::VectorXd> return_traj() const { return this->active_traj_; }

    /// @brief Return the trajectory resampled to @p num points over a time range.
    /// @param num  Number of output samples.
    /// @param tl   Lower time bound.
    /// @param th   Upper time bound.
    /// @return Interpolated trajectory samples.
    std::vector<Eigen::VectorXd> return_traj_range(int num, double tl, double th) {
        this->table_.load_regular_data(this->defs_per_bin_.sum(), this->active_traj_);
        return this->table_.interp_range(num, tl, th);
    }
    /// @brief Return the trajectory non-dimensionally equidistributed over a time range.
    /// @param num  Number of output samples.
    /// @param tl   Lower time bound.
    /// @param th   Upper time bound.
    /// @return Equidistributed trajectory samples.
    std::vector<Eigen::VectorXd> return_traj_range_nd(int num, double tl, double th) {
        this->table_.load_regular_data(this->defs_per_bin_.sum(), this->active_traj_);
        return this->table_.nd_equidist(num, tl, th);
    }
    /// @brief Return an interpolation table built from the current trajectory.
    /// @return An @ref LGLInterpTable loaded with the exact trajectory data.
    LGLInterpTable return_traj_table() {
        LGLInterpTable tabt = this->table_;
        tabt.load_exact_data(this->active_traj_);
        return tabt;
    }

    /// @brief Return the current static-parameter values.
    /// @return The static-parameter vector.
    Eigen::VectorXd return_static_params() const { return this->active_static_params_; }

    /// @brief Return the control-spline constraint multipliers from the last solve.
    /// @return Per-node multiplier vectors, or empty if no control-spline constraint exists.
    /// @throws std::invalid_argument if no solve/optimize has been run.
    std::vector<Eigen::VectorXd> return_u_spline_con_lmults() const {

        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No multipliers to return, a solve or optimize call must be made "
                "before returning constraint multipliers ");
        }

        if (this->control_funcs_index_ < 0) {
            return std::vector<Eigen::VectorXd>();
        } else {
            return this->indexer_.get_func_eq_multipliers(this->control_funcs_index_,
                                                          this->active_eq_lmults_);
        }
    }

    /// @brief Return the control-spline constraint residual values from the last solve.
    /// @return Per-node residual vectors, or empty if no control-spline constraint exists.
    /// @throws std::invalid_argument if no solve/optimize has been run.
    std::vector<Eigen::VectorXd> return_u_spline_con_vals() const {

        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No constraints to return, a solve or optimize call must be made "
                "before returning constraint values ");
        }
        if (this->control_funcs_index_ < 0) {
            return std::vector<Eigen::VectorXd>();
        } else {
            return this->indexer_.get_func_eq_multipliers(this->control_funcs_index_,
                                                          this->active_eq_cons_);
        }
    }

    /// @brief Return the Lagrange multipliers of an equality constraint from the last solve.
    /// @param index  Equality-constraint index.
    /// @return Per-application multiplier vectors for the constraint.
    /// @throws std::invalid_argument if no solve/optimize has been run.
    std::vector<Eigen::VectorXd> return_equal_con_lmults(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No multipliers to return, a solve or optimize call must be made "
                "before returning constraint multipliers ");
        }
        int Gindex = this->user_equalities_.at(index).global_index_;
        return this->indexer_.get_func_eq_multipliers(Gindex, this->active_eq_lmults_);
    }
    /// @brief Return the residual values of an equality constraint from the last solve.
    /// @param index  Equality-constraint index.
    /// @return Per-application residual vectors for the constraint.
    /// @throws std::invalid_argument if no solve/optimize has been run.
    std::vector<Eigen::VectorXd> return_equal_con_vals(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No constraints to return, a solve or optimize call must be made "
                "before returning constraint values ");
        }
        int Gindex = this->user_equalities_.at(index).global_index_;
        return this->indexer_.get_func_eq_multipliers(Gindex, this->active_eq_cons_);
    }
    /// @brief Return the output scales of an equality constraint.
    /// @param index  Equality-constraint index.
    /// @return The constraint's per-output scale vector.
    Eigen::VectorXd return_equal_con_scales(int index) const {
        return this->user_equalities_.at(index).output_scales_;
    }

    /// @brief Return the Lagrange multipliers of an inequality constraint from the last solve.
    /// @param index  Inequality-constraint index.
    /// @return Per-application multiplier vectors for the constraint.
    /// @throws std::invalid_argument if no solve/optimize has been run.
    std::vector<Eigen::VectorXd> return_inequal_con_lmults(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No multipliers to return, a solve or optimize call must be made "
                "before returning constraint multipliers ");
        }
        int Gindex = this->user_inequalities_.at(index).global_index_;
        return this->indexer_.get_func_iq_multipliers(Gindex, this->active_iq_lmults_);
    }

    /// @brief Return the residual values of an inequality constraint from the last solve.
    /// @param index  Inequality-constraint index.
    /// @return Per-application residual vectors for the constraint.
    /// @throws std::invalid_argument if no solve/optimize has been run.
    std::vector<Eigen::VectorXd> return_inequal_con_vals(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No constraints to return, a solve or optimize call must be made "
                "before returning constraint values ");
        }
        int Gindex = this->user_inequalities_.at(index).global_index_;
        return this->indexer_.get_func_iq_multipliers(Gindex, this->active_iq_cons_);
    }
    /// @brief Return the output scales of an inequality constraint.
    /// @param index  Inequality-constraint index.
    /// @return The constraint's per-output scale vector.
    Eigen::VectorXd return_inequal_con_scales(int index) const {
        return this->user_inequalities_.at(index).output_scales_;
    }

    /// @brief Return the costate (adjoint) trajectory recovered from the multipliers.
    /// @return One packed costate vector per node.
    std::vector<Eigen::VectorXd> return_costate_traj() const;
    /// @brief Return the per-node discretization-error estimate of the trajectory.
    /// @return One error vector per node.
    std::vector<Eigen::VectorXd> return_traj_error() const;

    /// @brief Return the output scales of an integral objective.
    /// @param index  Integral-objective index.
    /// @return The objective's per-output scale vector.
    Eigen::VectorXd return_integral_objective_scales(int index) const {
        return this->user_integrands_.at(index).output_scales_;
    }
    /// @brief Return the output scales of an integral-parameter function.
    /// @param index  Integral-parameter-function index.
    /// @return The function's per-output scale vector.
    Eigen::VectorXd return_integral_param_function_scales(int index) const {
        return this->user_param_integrands_.at(index).output_scales_;
    }
    /// @brief Return the output scales of a state objective.
    /// @param index  State-objective index.
    /// @return The objective's per-output scale vector.
    Eigen::VectorXd return_state_objective_scales(int index) const {
        return this->user_state_objectives_.at(index).output_scales_;
    }
    /// @brief Return the output scales applied to the ODE dynamics (defect) function.
    /// @return The per-state-derivative output scale vector.
    Eigen::VectorXd return_ode_output_scales() const {
        VectorXd output_scales =
            xtup_units_.head(this->x_vars()).cwiseInverse() * this->xtup_units_[this->x_vars()];
        return output_scales;
    }

    /////////////////////////////////////////////////
  protected:
    /// @internal
    /// @brief Transcribe the ODE dynamics into defect constraints (subclass-specific).
    /// @endinternal
    virtual void transcribe_dynamics() = 0;
    /// @internal
    /// @brief Transcribe axis (per-state) functions into the NLP.
    /// @endinternal
    virtual void transcribe_axis_funcs();
    /// @internal
    /// @brief Transcribe control-spline constraints into the NLP.
    /// @endinternal
    virtual void transcribe_control_funcs();
    /// @internal
    /// @brief Transcribe integral objectives/functions into the NLP.
    /// @endinternal
    virtual void transcribe_integrals();
    /// @internal
    /// @brief Transcribe boundary values, bounds, and other basic constraints.
    /// @endinternal
    virtual void transcribe_basic_funcs();

    /// @internal
    /// @brief (Re)build the @ref PhaseIndexer from the current dimensions and mesh.
    /// @endinternal
    void init_indexing() {
        this->indexer_ =
            PhaseIndexer(this->x_vars(), this->u_vars(), this->p_vars(), this->num_stat_params_);
        this->indexer_.set_dimensions(this->num_tran_card_states_, this->num_defects_,
                                      this->control_mode_ == ControlModes::BlockConstant);
    }
    /// @internal
    /// @brief Validate the IO sizes of all registered functions for a phase.
    /// @param pnum  The phase index, for diagnostics.
    /// @endinternal
    void check_functions(int pnum);

    /// @internal
    /// @brief Validate that a function's input size matches its region bindings.
    /// @tparam T  The region-bound function holder type.
    /// @param func   The function to check.
    /// @param ftype  Human-readable function kind, for error messages.
    /// @throws std::invalid_argument if the input size is inconsistent.
    /// @endinternal
    template <class T> static void check_function_size(const T &func, std::string ftype) {
        int irows = func.func_.input_rows();
        switch (func.region_flag_) {
        case PhaseRegionFlags::Front:
        case PhaseRegionFlags::Back:
        case PhaseRegionFlags::Path:
        case PhaseRegionFlags::Params:
        case PhaseRegionFlags::ODEParams:
        case PhaseRegionFlags::StaticParams:
        case PhaseRegionFlags::InnerPath:
        case PhaseRegionFlags::NodalPath: {
            int isize = func.xtu_vars_.size() + func.op_vars_.size() + func.sp_vars_.size();
            if (irows != isize) {

                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Input size of {0:} (IRows = {1:}) does not match that implied by "
                           "indexing parameters "
                           "(IRows = {2:}).\n",
                           ftype, irows, isize);

                throw std::invalid_argument("");
            }
            break;
        }
        case PhaseRegionFlags::FrontandBack:
        case PhaseRegionFlags::BackandFront:
        case PhaseRegionFlags::PairWisePath: {
            int isize = func.xtu_vars_.size() * 2 + func.op_vars_.size() + func.sp_vars_.size();
            if (irows != isize) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Input size of {0:} (IRows = {1:}) does not match that implied by "
                           "indexing parameters "
                           "(IRows = {2:}).\n",
                           ftype, irows, isize);
                throw std::invalid_argument("");
            }
            break;
        }
        default: {
            break;
        }
        }
    }

  public:
    /// @internal
    /// @brief Compute the input scaling vector for a function over a region.
    /// @param flag  The phase region.
    /// @param XtUV  State/time/control variable indices.
    /// @param OPV   ODE-parameter variable indices.
    /// @param SPV   Static-parameter variable indices.
    /// @return The per-input scale vector.
    /// @endinternal
    Eigen::VectorXd get_input_scale(PhaseRegionFlags flag, VectorXi XtUV, VectorXi OPV,
                                    VectorXi SPV) const;

  protected:
    /// @internal
    /// @brief Sample representative inputs for a region, used to seed auto-scaling.
    /// @param flag  The phase region.
    /// @param XtUV  State/time/control variable indices.
    /// @param OPV   ODE-parameter variable indices.
    /// @param SPV   Static-parameter variable indices.
    /// @return Representative input vectors over the region.
    /// @endinternal
    std::vector<Eigen::VectorXd> get_test_inputs(PhaseRegionFlags flag, VectorXi XtUV, VectorXi OPV,
                                                 VectorXi SPV) const;

    /// @internal
    /// @brief Compute the automatic variable/equation scales from the trajectory.
    /// @endinternal
    void calc_auto_scales();

    /// @internal
    /// @brief Collect the per-objective scale factors.
    /// @return The objective scale factors.
    /// @endinternal
    std::vector<double> get_objective_scales();
    /// @internal
    /// @brief Apply a uniform scale to all objectives.
    /// @param scale  The scale to apply.
    /// @endinternal
    void update_objective_scales(double scale);

    /// @internal
    /// @brief Validate that a lower-bound scale is strictly positive.
    /// @param lbscale  The lower-bound scale to check.
    /// @throws std::invalid_argument if @p lbscale is not strictly positive.
    /// @endinternal
    static void check_lbscale(double lbscale) {
        if (lbscale <= 0.0) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "Lower-bound scale ({0:.3e}) must be a strictly positive number.\n",
                       lbscale);
            throw std::invalid_argument("");
        }
    }
    /// @internal
    /// @brief Validate that an upper-bound scale is strictly positive.
    /// @param ubscale  The upper-bound scale to check.
    /// @throws std::invalid_argument if @p ubscale is not strictly positive.
    /// @endinternal
    static void check_ubscale(double ubscale) {
        if (ubscale <= 0.0) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "Upper-bound scale ({0:.3e}) must be a strictly positive number.\n",
                       ubscale);
            throw std::invalid_argument("");
        }
    }

    /// @internal
    /// @brief Transcribe this phase's variables and constraints into a shared NLP.
    /// @param vo    Decision-variable offset of this phase.
    /// @param eqo   Equality-constraint-row offset.
    /// @param iqo   Inequality-constraint-row offset.
    /// @param np    The shared NLP to transcribe into.
    /// @param pnum  The phase index, for diagnostics.
    /// @endinternal
    void transcribe_phase(int vo, int eqo, int iqo, std::shared_ptr<NonLinearProgram> np, int pnum);

    /// @internal
    /// @brief Pack the active trajectory and static params into the solver input vector.
    /// @return The packed decision-variable vector (scaled if auto-scaling is on).
    /// @endinternal
    Eigen::VectorXd make_solver_input() const {

        if (this->auto_scaling_) {
            auto active_traj_tmp = this->active_traj_;

            for (auto &T : active_traj_tmp) {
                T = T.cwiseQuotient(this->xtup_units_);
            }

            VectorXd static_params_tmp;
            if (this->active_static_params_.size() > 0 && this->sp_units_.size() > 0) {
                static_params_tmp = this->active_static_params_.cwiseQuotient(this->sp_units_);
            }
            return this->indexer_.make_solver_input(active_traj_tmp, static_params_tmp);

        } else {
            return this->indexer_.make_solver_input(this->active_traj_,
                                                    this->active_static_params_);
        }
    }
    /// @internal
    /// @brief Unpack the solver output back into the active trajectory and static params.
    /// @param Vars  The solved decision-variable vector.
    /// @endinternal
    void collect_solver_output(const VectorXd &Vars) {
        this->indexer_.collect_solver_output(Vars, this->active_traj_, this->active_static_params_);

        if (this->auto_scaling_) {
            for (auto &T : this->active_traj_) {
                T = T.cwiseProduct(this->xtup_units_);
            }
            if (this->active_static_params_.size() > 0 && this->sp_units_.size() > 0) {
                this->active_static_params_ =
                    this->active_static_params_.cwiseProduct(this->sp_units_);
            }
        }
    }
    /// @internal
    /// @brief Store the equality/inequality Lagrange multipliers from a solve.
    /// @param EM  Equality-constraint multipliers.
    /// @param IM  Inequality-constraint multipliers.
    /// @endinternal
    void collect_solver_multipliers(const VectorXd &EM, const VectorXd &IM) {
        this->multipliers_loaded_ = true;
        this->active_eq_lmults_ = EM;
        this->active_iq_lmults_ = IM;
    }

    /// @internal
    /// @brief Store post-optimization constraint values and multipliers.
    /// @param EC  Equality-constraint residuals.
    /// @param EM  Equality-constraint multipliers.
    /// @param IC  Inequality-constraint residuals.
    /// @param IM  Inequality-constraint multipliers.
    /// @endinternal
    void collect_post_opt_info(const VectorXd &EC, const VectorXd &EM, const VectorXd &IC,
                               const VectorXd &IM) {

        this->post_opt_info_valid_ = true;
        this->multipliers_loaded_ = true;

        this->active_eq_cons_ = EC;
        this->active_iq_cons_ = IC;
        this->active_eq_lmults_ = EM;
        this->active_iq_lmults_ = IM;
    }

    /// @internal
    /// @brief Drive PSIOPT for a single solve/optimize pass (no mesh loop).
    /// @param mode  The solve/optimize job mode.
    /// @return The solver convergence flag.
    /// @endinternal
    PSIOPT::ConvergenceFlags psipot_call_impl(JetJobModes mode);

    /// @internal
    /// @brief Run the full phase solve, including the adaptive-mesh-refinement loop.
    /// @param mode  The solve/optimize job mode.
    /// @return The solver convergence flag.
    /// @endinternal
    PSIOPT::ConvergenceFlags phase_call_impl(JetJobModes mode);

  public:
    /// @brief Transcribe the phase into its NLP, optionally printing diagnostics.
    /// @param showstats  Whether to print problem-size statistics.
    /// @param showfuns   Whether to print per-function details.
    void transcribe(bool showstats, bool showfuns);

    /// @brief Transcribe the phase into its NLP with diagnostics suppressed.
    void transcribe() { this->transcribe(false, false); }

    /// @internal
    /// @brief Test the threaded function partitioning over a range of partition counts.
    /// @param i  First partition count to test.
    /// @param j  Last partition count to test.
    /// @param n  Number of repetitions per count.
    /// @endinternal
    void test_partitions(int i, int j, int n);

    /// @internal
    /// @brief Prepare the phase for a "jet" (batched) solve.
    /// @endinternal
    void jet_initialize() {
        this->set_num_partitions(1, 1);
        this->optimizer_->set_print_level(10);
        this->print_mesh_info_ = false;

        this->transcribe();
    }
    /// @internal
    /// @brief Tear down jet-solve state and restore the default phase configuration.
    /// @endinternal
    void jet_release() {
        this->indexer_ = PhaseIndexer();
        this->optimizer_->release();
        this->init_partitions();
        this->optimizer_->set_print_level(0);
        this->print_mesh_info_ = true;
        this->nlp_ = std::shared_ptr<NonLinearProgram>();
        this->reset_transcription();
        this->invalidate_post_opt_info();
    }

    /// @brief Solve the phase for feasibility (satisfy constraints, no optimization).
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags solve() { return phase_call_impl(JetJobModes::Solve); }
    /// @brief Optimize the phase (minimize the objective subject to the constraints).
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags optimize() { return phase_call_impl(JetJobModes::Optimize); }
    /// @brief Solve for feasibility, then optimize.
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags solve_optimize() {
        return phase_call_impl(JetJobModes::SolveOptimize);
    }
    /// @brief Solve, optimize, then solve again to tighten feasibility.
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags solve_optimize_solve() {
        return phase_call_impl(JetJobModes::SolveOptimizeSolve);
    }
    /// @brief Optimize, then solve to tighten feasibility.
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags optimize_solve() {
        return phase_call_impl(JetJobModes::OptimizeSolve);
    }

    /////////////////////////////////////////////////////////////////

    /// @internal
    /// @brief Compute mesh error/distribution using a reference integrator.
    /// @param tsnd         Output non-dimensional node times.
    /// @param mesh_errors  Output per-segment error matrix.
    /// @param mesh_dist    Output per-segment error-distribution matrix.
    /// @endinternal
    virtual void get_meshinfo_integrator(Eigen::VectorXd &tsnd, Eigen::MatrixXd &mesh_errors,
                                         Eigen::MatrixXd &mesh_dist) const = 0;
    /// @internal
    /// @brief Compute mesh error/distribution using the de Boor estimator.
    /// @param tsnd         Output non-dimensional node times.
    /// @param mesh_errors  Output per-segment error matrix.
    /// @param mesh_dist    Output per-segment error-distribution matrix.
    /// @endinternal
    virtual void get_meshinfo_deboor(Eigen::VectorXd &tsnd, Eigen::MatrixXd &mesh_errors,
                                     Eigen::MatrixXd &mesh_dist) const = 0;
    /// @internal
    /// @brief Compute the end-to-end (global) error of the current trajectory.
    /// @return The per-state global error vector.
    /// @endinternal
    virtual Eigen::VectorXd calc_global_error() const = 0;

    /// @internal
    /// @brief Reset mesh-refinement convergence state before a new solve.
    /// @endinternal
    virtual void init_mesh_refinement() {
        this->mesh_converged_ = false;
        this->mesh_iters_.resize(0);
    }

    /// @internal
    /// @brief Check whether the current mesh meets the error tolerance.
    /// @return True if the mesh has converged.
    /// @endinternal
    virtual bool check_mesh();
    /// @internal
    /// @brief Redistribute and refine the mesh based on the latest error estimate.
    /// @endinternal
    virtual void update_mesh();

    /// @internal
    /// @brief Locate control switches in the current trajectory.
    /// @return Non-dimensional times of detected control switches.
    /// @endinternal
    virtual Eigen::VectorXd calc_switches();

    /// @internal
    /// @brief Compute node times, equidistributing bins, and per-segment error.
    /// @param integ  Whether to use the integrator (true) or de Boor (false) estimator.
    /// @param n      Number of equidistributing bins to produce.
    /// @return Tuple of {node times, bin boundaries, per-segment error}.
    /// @endinternal
    auto get_mesh_info(bool integ, int n) {

        Eigen::VectorXd tsnd;
        Eigen::MatrixXd mesh_errors;
        Eigen::MatrixXd mesh_dist;

        this->table_.load_exact_data(this->active_traj_);

        if (integ) {
            this->get_meshinfo_integrator(tsnd, mesh_errors, mesh_dist);
        } else {
            this->get_meshinfo_deboor(tsnd, mesh_errors, mesh_dist);
        }

        Eigen::VectorXd error = mesh_errors.colwise().lpNorm<Eigen::Infinity>();
        Eigen::VectorXd dist = mesh_dist.colwise().lpNorm<Eigen::Infinity>();

        Eigen::VectorXd distint(dist.size());
        distint[0] = 0;

        for (int i = 0; i < dist.size() - 1; i++) {
            distint[i + 1] = distint[i] + (dist[i]) * (tsnd[i + 1] - tsnd[i]);
        }

        distint = distint / distint[distint.size() - 1];

        Eigen::VectorXd bins;
        bins.setLinSpaced(n + 1, 0.0, 1.0);
        int elem = 0;
        for (int i = 1; i < n; i++) {
            double di = double(i) / double(n);
            auto it = std::upper_bound(distint.cbegin() + elem, distint.cend(), di);
            elem = int(it - distint.cbegin()) - 1;

            double t0 = tsnd[elem];
            double t1 = tsnd[elem + 1];
            double d0 = distint[elem];
            double d1 = distint[elem + 1];
            double slope = (d1 - d0) / (t1 - t0);
            bins[i] = (di - d0) / slope + t0;
        }

        return std::tuple{tsnd, bins, error};
    }
};

} // namespace tycho::oc
