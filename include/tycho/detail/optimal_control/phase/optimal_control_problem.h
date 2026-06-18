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
#include "tycho/detail/optimal_control/core/link_function.h"
#include "tycho/detail/optimal_control/phase/ode_phase_base.h"
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

// Solvers types
using tycho::solvers::NonLinearProgram;
using tycho::solvers::OptimizationProblemBase;
using tycho::solvers::PSIOPT;

// VF types
using vf::Arguments;
using vf::GenericFunction;
using vf::StackedOutputs;

/// @ingroup optimal_control
/// @brief A multi-phase optimal control problem linking several phases.
///
/// Owns a collection of @ref ODEPhase objects and the link constraints,
/// inequalities, and objectives that couple them (plus a shared link-parameter
/// vector). It transcribes all phases and links into one shared NLP and drives
/// PSIOPT to solve or optimize the combined problem, coordinating adaptive mesh
/// refinement across phases.
struct OptimalControlProblemBase : OptimizationProblemBase {
    using VectorXi = Eigen::VectorXi; ///< @brief Convenience alias for an integer vector.
    using MatrixXi = Eigen::MatrixXi; ///< @brief Convenience alias for an integer matrix.

    using VectorXd = Eigen::VectorXd; ///< @brief Convenience alias for a double vector.
    using MatrixXd = Eigen::MatrixXd; ///< @brief Convenience alias for a double matrix.

    using VectorFunctionalX =
        GenericFunction<-1, -1>; ///< @brief Type-erased vector-valued function.
    using ScalarFunctionalX =
        GenericFunction<-1, 1>; ///< @brief Type-erased scalar-valued function.

    using LinkConstraint =
        LinkFunction<VectorFunctionalX>; ///< @brief A link constraint across phases.
    using LinkObjective =
        LinkFunction<ScalarFunctionalX>; ///< @brief A link objective across phases.
    using StateConstraint = StateFunction<VectorFunctionalX>; ///< @brief A region-bound constraint.
    using StateObjective = StateFunction<ScalarFunctionalX>;  ///< @brief A region-bound objective.
    using StateIntegral = StateFunction<ScalarFunctionalX>;   ///< @brief A region-bound integrand.
    using PhasePtr = std::shared_ptr<ODEPhaseBase>;           ///< @brief Shared pointer to a phase.

    /// @brief Packed link target: phase index, region name, and three index vectors.
    using PhaseIndexPack =
        std::tuple<int, std::string, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>;
    /// @brief Packed link target keyed by phase pointer instead of index.
    using PhaseIndexPackPtr =
        std::tuple<PhasePtr, std::string, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>;

    /// @brief Flexible phase reference: index, pointer, or name.
    using PhaseRefType = std::variant<int, PhasePtr, std::string>;

    /// @brief A link target: phase reference, region, and state/ODE/static selectors.
    using PhasePack =
        std::tuple<PhaseRefType, RegionType, VarIndexType, VarIndexType, VarIndexType>;

    std::vector<PhasePtr> phases;         ///< The phases in this problem.
    std::vector<std::string> phase_names; ///< Names of the phases (unique).

    bool do_transcription_ = true; ///< Whether the problem needs (re-)transcription.
    /// @brief Mark the problem as needing (re-)transcription before the next solve.
    void reset_transcription() { this->do_transcription_ = true; };

    std::map<int, LinkConstraint> link_equalities_;   ///< Link equality constraints by index.
    std::map<int, LinkConstraint> link_inequalities_; ///< Link inequality constraints by index.
    std::map<int, LinkObjective> link_objectives_;    ///< Link objectives by index.

    VectorXd active_link_params_;       ///< Current shared link-parameter values.
    Eigen::VectorXd lp_units_;          ///< Scaling units for the link parameters.
    bool auto_scaling_ = false;         ///< Whether automatic scaling is enabled.
    bool sync_objective_scales_ = true; ///< Whether objective scales track auto-scaling.

    std::map<std::string, Eigen::VectorXi> LPidxs; ///< Named link-parameter variable groups.

    /// @brief Set the shared link parameters and their scaling units.
    /// @param parm   Link-parameter values.
    /// @param units  Per-parameter scaling units (must match @p parm in size).
    /// @throws std::invalid_argument if @p units and @p parm differ in size.
    void set_link_params(VectorXd parm, VectorXd units) {
        if (units.size() != parm.size()) {
            throw std::invalid_argument(
                "Size of link parameter vector and scaling units vector must match");
        }

        this->active_link_params_ = parm;
        this->num_link_params_ = parm.size();
        this->reset_transcription();
        this->lp_units_ = units;
    }
    /// @brief Set the shared link parameters with unit scaling.
    /// @param parm  Link-parameter values.
    void set_link_params(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->set_link_params(parm, units);
    }

    /// @brief Replace all named link-parameter variable groups.
    /// @param lpidxs  Map of group name to link-parameter indices.
    void set_link_param_vgroups(std::map<std::string, Eigen::VectorXi> lpidxs) {
        this->LPidxs = lpidxs;
    }
    /// @brief Merge additional named link-parameter variable groups.
    /// @param lpidxs  Map of group name to link-parameter indices to add.
    void add_link_param_vgroups(std::map<std::string, Eigen::VectorXi> lpidxs) {
        for (auto &[key, value] : lpidxs) {
            this->LPidxs[key] = value;
        }
    }
    /// @brief Add a named link-parameter variable group.
    /// @param idx  Link-parameter indices in the group.
    /// @param key  Group name.
    void add_link_param_vgroup(Eigen::VectorXi idx, std::string key) { this->LPidxs[key] = idx; }
    /// @brief Add a named link-parameter variable group of a single index.
    /// @param idx  The single link-parameter index.
    /// @param key  Group name.
    void add_link_param_vgroup(int idx, std::string key) {
        VectorXi tmp(1);
        tmp << idx;
        this->LPidxs[key] = tmp;
    }
    /// @brief Look up a named link-parameter variable group.
    /// @param key  Group name.
    /// @return The link-parameter indices in the group.
    /// @throws std::invalid_argument if no group named @p key exists.
    VectorXi get_lp_idx(std::string key) const {
        if (LPidxs.count(key) == 0) {
            throw std::invalid_argument(
                fmt::format("No LinkParam variable index group with name: {0:} exists.", key));
        }
        return this->LPidxs.at(key);
    }

    /// @brief Return the current shared link-parameter values.
    /// @return The link-parameter vector.
    VectorXd return_link_params() { return this->active_link_params_; }

    bool multipliers_loaded_ = false;  ///< Whether Lagrange multipliers are available.
    bool post_opt_info_valid_ = false; ///< Whether post-solve constraint/multiplier data is valid.

    /// @brief Invalidate cached post-solve constraint/multiplier data.
    void invalidate_post_opt_info() { this->post_opt_info_valid_ = false; };

    VectorXd active_eq_lmults_; ///< Link equality-constraint Lagrange multipliers.
    VectorXd active_iq_lmults_; ///< Link inequality-constraint Lagrange multipliers.
    VectorXd active_eq_cons_;   ///< Link equality-constraint residual values.
    VectorXd active_iq_cons_;   ///< Link inequality-constraint residual values.

    VectorXi link_param_locs_; ///< NLP locations of the link-parameter variables.

    VectorXi num_phase_vars_;    ///< Decision-variable count of each phase.
    VectorXi num_phase_eq_cons_; ///< Equality-constraint-row count of each phase.
    VectorXi num_phase_iq_cons_; ///< Inequality-constraint-row count of each phase.

    int num_link_params_ = 0;  ///< Number of shared link parameters.
    int num_link_eq_cons_ = 0; ///< Number of link equality-constraint rows.
    int num_link_iq_cons_ = 0; ///< Number of link inequality-constraint rows.

    int start_obj_ = 0; ///< Index of the first link objective in the NLP.
    int start_eq_ = 0;  ///< Index of the first link equality function in the NLP.
    int start_iq_ = 0;  ///< Index of the first link inequality function in the NLP.

    int num_obj_funs_ = 0; ///< Number of link objective functions.
    int num_eq_funs_ = 0;  ///< Number of link equality functions.
    int num_iq_funs_ = 0;  ///< Number of link inequality functions.

    int num_prob_vars_ = 0;    ///< Total decision-variable count of the problem.
    int num_prob_eq_cons_ = 0; ///< Total equality-constraint-row count of the problem.
    int num_prob_iq_cons_ = 0; ///< Total inequality-constraint-row count of the problem.

    ///////////////////////////////
    bool adaptive_mesh_ = false;   ///< Whether adaptive mesh refinement is enabled problem-wide.
    bool print_mesh_info_ = true;  ///< Whether to print mesh-refinement diagnostics.
    bool solve_only_first_ = true; ///< Only solve (not optimize) on the first mesh iteration.

    int max_mesh_iters_ = 10; ///< Maximum mesh-refinement iterations.
    PSIOPT::ConvergenceFlags mesh_abort_flag_ =
        PSIOPT::ConvergenceFlags::DIVERGING; ///< Solver flag that aborts the mesh loop.

    bool mesh_converged_ = false; ///< Whether all adaptive phases have converged.

    /// @brief Enable/disable automatic scaling, optionally propagating to phases.
    /// @param autoscale     Whether to enable automatic scaling.
    /// @param applytophases Whether to also set the flag on each phase.
    void set_auto_scaling(bool autoscale, bool applytophases) {
        this->auto_scaling_ = autoscale;
        if (applytophases) {
            for (auto phase : this->phases) {
                phase->set_auto_scaling(autoscale);
            }
        }
        this->reset_transcription();
        this->invalidate_post_opt_info();
    }

    /// @brief Enable/disable adaptive mesh refinement, optionally propagating to phases.
    /// @param amesh         Whether to enable adaptive mesh refinement.
    /// @param applytophases Whether to also set the flag on each phase.
    void set_adaptive_mesh(bool amesh, bool applytophases) {
        this->adaptive_mesh_ = amesh;
        if (applytophases) {
            for (auto phase : this->phases) {
                phase->set_adaptive_mesh(amesh);
            }
        }
    }

    /// @brief Set the mesh-error tolerance on every phase.
    /// @param t  Tolerance.
    void set_mesh_tol(double t) {
        for (auto phase : this->phases) {
            phase->set_mesh_tol(t);
        }
    }
    /// @brief Set the segment-count reduction factor on every phase.
    /// @param t  Reduction factor.
    void set_mesh_red_factor(double t) {
        for (auto phase : this->phases) {
            phase->set_mesh_red_factor(t);
        }
    }
    /// @brief Set the segment-count increase factor on every phase.
    /// @param t  Increase factor.
    void set_mesh_inc_factor(double t) {
        for (auto phase : this->phases) {
            phase->set_mesh_inc_factor(t);
        }
    }
    /// @brief Set the error-overshoot factor on every phase.
    /// @param t  Error factor.
    void set_mesh_err_factor(double t) {
        for (auto phase : this->phases) {
            phase->set_mesh_err_factor(t);
        }
    }
    /// @brief Set the maximum number of problem-level mesh-refinement iterations.
    /// @param it  Iteration cap.
    void set_max_mesh_iters(int it) { this->max_mesh_iters_ = it; }
    /// @brief Set the minimum mesh-segment count on every phase.
    /// @param it  Minimum segment count.
    void set_min_segments(int it) {
        for (auto phase : this->phases) {
            phase->set_min_segments(it);
        }
    }
    /// @brief Set the maximum mesh-segment count on every phase.
    /// @param it  Maximum segment count.
    void set_max_segments(int it) {
        for (auto phase : this->phases) {
            phase->set_max_segments(it);
        }
    }
    /// @brief Set the mesh-error convergence aggregation on every phase.
    /// @param m  Aggregation mode.
    void set_mesh_error_criteria(MeshErrorAggregation m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_criteria(m);
        }
    }
    /// @brief Set the mesh-error convergence aggregation on every phase by name.
    /// @param m  Aggregation name.
    void set_mesh_error_criteria(const std::string &m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_criteria(m);
        }
    }
    /// @brief Set the mesh-error estimator on every phase.
    /// @param m  Estimator mode.
    void set_mesh_error_estimator(MeshErrorEstimators m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_estimator(m);
        }
    }
    /// @brief Set the mesh-error estimator on every phase by name.
    /// @param m  Estimator name.
    void set_mesh_error_estimator(const std::string &m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_estimator(m);
        }
    }
    /// @brief Set the mesh-error distribution aggregation on every phase.
    /// @param m  Aggregation mode.
    void set_mesh_error_distributor(MeshErrorAggregation m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_distributor(m);
        }
    }
    /// @brief Set the mesh-error distribution aggregation on every phase by name.
    /// @param m  Aggregation name.
    void set_mesh_error_distributor(const std::string &m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_distributor(m);
        }
    }

    ///////////////////////////////
    /// @brief Default constructor; create an empty problem with no phases.
    OptimalControlProblemBase() {}
    /// @brief Construct from a list of phases.
    /// @param ps  The phases to add.
    OptimalControlProblemBase(std::vector<PhasePtr> ps) { this->add_phases(ps); }

    /// @brief Add a phase, naming it by its index.
    /// @param p  The phase to add.
    /// @return The index assigned to the phase.
    int add_phase(PhasePtr p) {
        this->reset_transcription();
        this->phases.push_back(p);
        int index = int(this->phases.size()) - 1;
        this->phase_names.push_back(std::to_string(index));
        check_dupilcate_phases();
        return index;
    }

    /// @brief Add several phases.
    /// @param ps  The phases to add.
    /// @return The indices assigned to the phases.
    std::vector<int> add_phases(std::vector<PhasePtr> ps) {
        std::vector<int> idxs;
        for (auto p : ps) {
            idxs.push_back(this->add_phase(p));
        }
        return idxs;
    }

    /// @brief Add a phase with an explicit name.
    /// @param p     The phase to add.
    /// @param name  Unique name for the phase.
    /// @return The index assigned to the phase.
    int add_phase(PhasePtr p, const std::string &name) {
        this->reset_transcription();
        this->phases.push_back(p);
        int index = int(this->phases.size()) - 1;
        this->phase_names.push_back(name);
        check_dupilcate_phases();
        return index;
    }

    /// @brief Look up a phase index by name.
    /// @param name  The phase name.
    /// @return The phase index.
    /// @throws std::invalid_argument if no phase with @p name exists.
    int get_phase_num(const std::string &name) {
        auto nameit = std::find(phase_names.begin(), phase_names.end(), name);
        if (nameit == phase_names.end()) {
            throw std::invalid_argument(
                fmt::format("Transcription Error!!!\n"
                            "No phase with name '{}' exists in OptimalControlProblem.\n",
                            name));
        }
        return int(nameit - phase_names.begin());
    }

    /// @brief Look up a phase index by pointer.
    /// @param p  Shared pointer to the phase.
    /// @return The phase index.
    /// @throws std::invalid_argument if @p p is not part of this problem.
    int get_phase_num(PhasePtr p) {
        auto ptrit = std::find_if(phases.begin(), phases.end(),
                                  [&](PhasePtr pt) { return pt.get() == p.get(); });
        if (ptrit == phases.end()) {
            throw std::invalid_argument(
                "Transcription Error!!!\n"
                "The requested phase does not exist in OptimalControlProblem\n");
        }
        return int(ptrit - phases.begin());
    }

    /// @brief Resolve a flexible phase reference to its index.
    /// @param phase_t  Phase reference (index, pointer, or name).
    /// @return The phase index.
    int get_phase_num(PhaseRefType phase_t) {
        int phasenum;

        if (std::holds_alternative<int>(phase_t)) {
            phasenum = std::get<int>(phase_t);
        } else if (std::holds_alternative<PhasePtr>(phase_t)) {
            phasenum = this->get_phase_num(std::get<PhasePtr>(phase_t));
        } else if (std::holds_alternative<std::string>(phase_t)) {
            phasenum = this->get_phase_num(std::get<std::string>(phase_t));
        }
        return phasenum;
    }

    /// @brief Convert nested phase-name lists into phase-index vectors.
    /// @param ptlnamevec  Per-application lists of phase names to link.
    /// @return Per-application phase-index vectors.
    std::vector<VectorXi> ptl_from_phase_names(std::vector<std::vector<std::string>> ptlnamevec) {
        std::vector<VectorXi> ptl;
        for (auto &appl : ptlnamevec) {
            VectorXi ptlv(appl.size());
            for (int i = 0; i < appl.size(); i++) {
                ptlv[i] = this->get_phase_num(appl[i]);
            }
            ptl.push_back(ptlv);
        }
        return ptl;
    }

    /// @brief Remove a phase by index.
    /// @param ith  Phase index (negative counts from the end).
    void remove_phase(int ith) {
        this->reset_transcription();
        if (ith < 0)
            ith = (this->phases.size() + ith);
        this->phases.erase(this->phases.begin() + ith);
        this->phase_names.erase(this->phase_names.begin() + ith);
    }
    /// @brief Access a phase by index.
    /// @param ith  Phase index (negative counts from the end).
    /// @return Shared pointer to the phase.
    PhasePtr phase(int ith) {
        if (ith < 0)
            ith = (this->phases.size() + ith);
        return this->phases[ith];
    }

    /// @internal
    /// @brief Resolve a link-parameter selector to absolute indices.
    /// @param LPvars_t  Link-parameter selector (index, vector, or named group(s)).
    /// @return The resolved link-parameter indices.
    /// @endinternal
    VectorXi get_lp_vars(VarIndexType LPvars_t) const {

        VectorXi LPvars;

        if (std::holds_alternative<int>(LPvars_t)) {
            LPvars.resize(1);
            LPvars[0] = std::get<int>(LPvars_t);
        } else if (std::holds_alternative<VectorXi>(LPvars_t)) {
            LPvars = std::get<VectorXi>(LPvars_t);
        } else if (std::holds_alternative<std::string>(LPvars_t)) {
            LPvars = this->get_lp_idx(std::get<std::string>(LPvars_t));
        } else if (std::holds_alternative<std::vector<std::string>>(LPvars_t)) {
            std::vector<VectorXi> varvec;
            int size = 0;
            auto tmpvars = std::get<std::vector<std::string>>(LPvars_t);
            for (auto tmpv : tmpvars) {
                varvec.push_back(this->get_lp_idx(tmpv));
                size += varvec.back().size();
            }
            LPvars.resize(size);
            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    LPvars[next] = varv[i];
                    next++;
                }
            }
        }

        return LPvars;
    }

    /// @internal
    /// @brief Build a @ref LinkFunction from a list of phase targets and a link-param selector.
    /// @tparam FuncHolder  The LinkFunction holder type to produce.
    /// @tparam FuncType    The wrapped function type.
    /// @param fun      The user function.
    /// @param packs    Per-phase link targets (phase, region, variable selectors).
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The constructed link-function holder.
    /// @throws std::invalid_argument if a pack references a nonexistent phase.
    /// @endinternal
    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(FuncType fun, std::vector<PhasePack> packs, VarIndexType lv,
                              ScaleType scale_t) {

        int npacks = packs.size();
        std::vector<Eigen::VectorXi> PTL;
        VectorXi phasenums(npacks);
        Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags(npacks);
        std::vector<Eigen::VectorXi> xtvs(npacks);
        std::vector<Eigen::VectorXi> opvs(npacks);
        std::vector<Eigen::VectorXi> spvs(npacks);

        for (int i = 0; i < npacks; i++) {

            auto phase_t = std::get<0>(packs[i]); // Name of phase, either phaseptr,int, or string
            int phasenum = get_phase_num(phase_t);

            if (phasenum < 0 || phasenum >= this->phases.size()) {
                throw std::invalid_argument(
                    fmt::format("Function references non - existent phase : {0:}\n", phasenum));
            }

            phasenums[i] = phasenum;

            RegFlags[i] = this->phases[phasenum]->get_region(std::get<1>(packs[i]));
            xtvs[i] = this->phases[phasenum]->get_xt_up_vars(RegFlags[i], std::get<2>(packs[i]));
            opvs[i] = this->phases[phasenum]->get_op_vars(RegFlags[i], std::get<3>(packs[i]));
            spvs[i] = this->phases[phasenum]->get_sp_vars(RegFlags[i], std::get<4>(packs[i]));
        }

        std::vector<Eigen::VectorXi> lvs;
        lvs.push_back(get_lp_vars(lv));
        PTL.push_back(phasenums);

        auto func = FuncHolder(fun, RegFlags, PTL, xtvs, opvs, spvs, lvs, scale_t);
        return func;
    }

    /// @internal
    /// @brief Build a two-phase @ref LinkFunction with full per-phase bindings.
    /// @tparam FuncHolder  The LinkFunction holder type to produce.
    /// @tparam FuncType    The wrapped function type.
    /// @param fun      The user function.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param XtUV0    First phase state/time/control selector.
    /// @param OPV0     First phase ODE-parameter selector.
    /// @param SPV0     First phase static-parameter selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param XtUV1    Second phase state/time/control selector.
    /// @param OPV1     Second phase ODE-parameter selector.
    /// @param SPV1     Second phase static-parameter selector.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The constructed link-function holder.
    /// @endinternal
    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(FuncType fun, PhaseRefType p0, RegionType reg0, VarIndexType XtUV0,
                              VarIndexType OPV0, VarIndexType SPV0, PhaseRefType p1,
                              RegionType reg1, VarIndexType XtUV1, VarIndexType OPV1,
                              VarIndexType SPV1, VarIndexType lv, ScaleType scale_t) {

        auto pack0 = PhasePack{p0, reg0, XtUV0, OPV0, SPV0};
        auto pack1 = PhasePack{p1, reg1, XtUV1, OPV1, SPV1};

        auto packs = std::vector{pack0, pack1};
        return make_func_impl<FuncHolder, FuncType>(fun, packs, lv, scale_t);
    }

    /// @internal
    /// @brief Build a two-phase @ref LinkFunction from a single selector per phase.
    ///
    /// Each phase's selector is routed to the state, ODE-parameter, or
    /// static-parameter slot based on its region.
    /// @tparam FuncHolder  The LinkFunction holder type to produce.
    /// @tparam FuncType    The wrapped function type.
    /// @param fun      The user function.
    /// @param p0       First phase reference.
    /// @param reg0_t   First phase region selector.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1_t   Second phase region selector.
    /// @param v1       Second phase variable selector.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The constructed link-function holder.
    /// @endinternal
    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(FuncType fun, PhaseRefType p0, RegionType reg0_t, VarIndexType v0,
                              PhaseRefType p1, RegionType reg1_t, VarIndexType v1, VarIndexType lv,
                              ScaleType scale_t) {

        VarIndexType xtv0, opv0, spv0, xtv1, opv1, spv1;

        xtv0 = VectorXi();
        opv0 = VectorXi();
        spv0 = VectorXi();

        xtv1 = VectorXi();
        opv1 = VectorXi();
        spv1 = VectorXi();

        PhaseRegionFlags reg0, reg1;

        if (std::holds_alternative<PhaseRegionFlags>(reg0_t)) {
            reg0 = std::get<PhaseRegionFlags>(reg0_t);
        } else if (std::holds_alternative<std::string>(reg0_t)) {
            reg0 = strto_PhaseRegionFlag(std::get<std::string>(reg0_t));
        }

        if (std::holds_alternative<PhaseRegionFlags>(reg1_t)) {
            reg1 = std::get<PhaseRegionFlags>(reg1_t);
        } else if (std::holds_alternative<std::string>(reg1_t)) {
            reg1 = strto_PhaseRegionFlag(std::get<std::string>(reg1_t));
        }

        if (reg0 == PhaseRegionFlags::ODEParams)
            opv0 = v0;
        else if (reg0 == PhaseRegionFlags::StaticParams)
            spv0 = v0;
        else
            xtv0 = v0;

        if (reg1 == PhaseRegionFlags::ODEParams)
            opv1 = v1;
        else if (reg1 == PhaseRegionFlags::StaticParams)
            spv1 = v1;
        else
            xtv1 = v1;

        auto pack0 = PhasePack{p0, reg0, xtv0, opv0, spv0};
        auto pack1 = PhasePack{p1, reg1, xtv1, opv1, spv1};
        auto packs = std::vector{pack0, pack1};

        return make_func_impl<FuncHolder, FuncType>(fun, packs, lv, scale_t);
    }

    /// @internal
    /// @brief Two-phase @ref make_func_impl overload with no link parameters.
    /// @tparam FuncHolder  The LinkFunction holder type to produce.
    /// @tparam FuncType    The wrapped function type.
    /// @param fun      The user function.
    /// @param p0       First phase reference.
    /// @param reg0_t   First phase region selector.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1_t   Second phase region selector.
    /// @param v1       Second phase variable selector.
    /// @param scale_t  Output-scale selector.
    /// @return The constructed link-function holder.
    /// @endinternal
    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(FuncType fun, PhaseRefType p0, RegionType reg0_t, VarIndexType v0,
                              PhaseRefType p1, RegionType reg1_t, VarIndexType v1,
                              ScaleType scale_t) {

        VectorXi empty;

        return make_func_impl<FuncHolder, FuncType>(fun, p0, reg0_t, v0, p1, reg1_t, v1, empty,
                                                    scale_t);
    }

    /////////////////////////////////////////////////

    /// @internal
    /// @brief Remove a registered link function from a map and invalidate transcription.
    /// @tparam FuncMap  The function-map type.
    /// @param map      The map to remove from.
    /// @param index    Function index, or -1 for the most recently added.
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
            throw std::invalid_argument(fmt::format(
                "No {0:} with index {1:} exists in Optimal Control Problem.", funcstr, index));
        }

        this->reset_transcription();
        this->invalidate_post_opt_info();
        map.erase(index);
    }

    /// @internal
    /// @brief Add a link function to a map after validating its IO size.
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
        // any state we mutate before the throw would leak into the OCP and
        // duplicate on retry.
        int index = map.size() == 0 ? 0 : map.rbegin()->first + 1;
        func.storage_index_ = index;
        check_function_size(func, funcstr);

        this->reset_transcription();
        this->invalidate_post_opt_info();
        map[index] = std::move(func);
        return index;
    }

    ////////////////////////////////////////////////

    /// @brief Add a link equality constraint @f$f=0@f$ over a list of phase targets.
    /// @param lc       The link constraint function.
    /// @param packs    Per-phase link targets.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_equal_con(VectorFunctionalX lc, std::vector<PhasePack> packs, VarIndexType lv,
                           ScaleType scale_t) {
        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, packs, lv, scale_t);
        return add_func_impl(Func, this->link_equalities_, "Link Equality Constraint");
    }

    /// @brief Add a two-phase link equality constraint with full per-phase bindings.
    /// @param lc       The link constraint function.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param XtUV0    First phase state/time/control selector.
    /// @param OPV0     First phase ODE-parameter selector.
    /// @param SPV0     First phase static-parameter selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param XtUV1    Second phase state/time/control selector.
    /// @param OPV1     Second phase ODE-parameter selector.
    /// @param SPV1     Second phase static-parameter selector.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_equal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0,
                           VarIndexType XtUV0, VarIndexType OPV0, VarIndexType SPV0,
                           PhaseRefType p1, RegionType reg1, VarIndexType XtUV1, VarIndexType OPV1,
                           VarIndexType SPV1, VarIndexType lv, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(
            lc, p0, reg0, XtUV0, OPV0, SPV0, p1, reg1, XtUV1, OPV1, SPV1, lv, scale_t);
        return add_func_impl(Func, this->link_equalities_, "Link Equality Constraint");
    }

    /// @brief Add a two-phase link equality constraint with one selector per phase.
    /// @param lc       The link constraint function.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param v1       Second phase variable selector.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_equal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                           PhaseRefType p1, RegionType reg1, VarIndexType v1, VarIndexType lv,
                           ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, p0, reg0, v0, p1,
                                                                            reg1, v1, lv, scale_t);
        return add_func_impl(Func, this->link_equalities_, "Link Equality Constraint");
    }
    /// @brief Add a two-phase link equality constraint with no link parameters.
    /// @param lc       The link constraint function.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param v1       Second phase variable selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_equal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                           PhaseRefType p1, RegionType reg1, VarIndexType v1, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, p0, reg0, v0, p1,
                                                                            reg1, v1, scale_t);
        return add_func_impl(Func, this->link_equalities_, "Link Equality Constraint");
    }

    /// @brief Chain consecutive phases with back-to-front continuity constraints.
    ///
    /// Adds, for each consecutive phase pair from @p iphase_t to @p fphase_t, an
    /// equality enforcing that the selected variables match across the join.
    /// @param iphase_t  Starting phase reference.
    /// @param fphase_t  Ending phase reference (must be after the start).
    /// @param vars      Variable selector to make continuous.
    /// @param scale_t   Output-scale selector.
    /// @return The indices of the added constraints.
    /// @throws std::invalid_argument if a phase reference is invalid or @p iphase >= @p fphase.
    std::vector<int> add_forward_link_equal_con(PhaseRefType iphase_t, PhaseRefType fphase_t,
                                                VarIndexType vars, ScaleType scale_t) {

        int iphase = get_phase_num(iphase_t);
        int fphase = get_phase_num(fphase_t);

        if (iphase < 0)
            iphase = (this->phases.size() + iphase);
        if (fphase < 0)
            fphase = (this->phases.size() + fphase);

        // Validate phase indices up-front. The insertion loop calls add_func_impl
        // per iteration, so a mid-loop *index* failure (e.g. fphase out of range)
        // would leave earlier link constraints already inserted, producing
        // duplicates on retry. Body failures inside add_link_equal_con (e.g.
        // check_function_size on a heterogeneous phase var layout) remain
        // possible but are caught atomically per insertion by add_func_impl's
        // validate-before-mutate ordering.
        if (iphase < 0 || iphase >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent starting phase:{0:}\n", iphase));
        }
        if (fphase < 0 || fphase >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent ending phase:{0:}\n", fphase));
        }
        if (iphase >= fphase) {
            throw std::invalid_argument(
                fmt::format("add_forward_link_equal_con requires iphase < fphase; got iphase={0:}, "
                            "fphase={1:}\n",
                            iphase, fphase));
        }

        int vsize = this->phases[iphase]->get_xt_up_vars(PhaseRegionFlags::Front, vars).size();

        auto args = Arguments<-1>(2 * vsize);
        auto func = args.head<-1>(vsize) - args.tail<-1>(vsize);

        std::vector<int> idxs;
        for (int i = iphase; i < fphase; i++) {

            int idx =
                this->add_link_equal_con(func, i, "Last", vars, i + 1, "First", vars, scale_t);

            idxs.push_back(idx);
        }

        return idxs;
    }

    /// @brief Constrain selected variables of two phases to be equal (difference == 0).
    /// @param p0       First phase reference.
    /// @param reg0_t   First phase region selector.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1_t   Second phase region selector.
    /// @param v1       Second phase variable selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    /// @throws std::invalid_argument if either phase reference is invalid.
    int add_direct_link_equal_con(PhaseRefType p0, RegionType reg0_t, VarIndexType v0,
                                  PhaseRefType p1, RegionType reg1_t, VarIndexType v1,
                                  ScaleType scale_t) {

        int phase = get_phase_num(p0);
        int phase1 = get_phase_num(p1);

        if (phase < 0)
            phase = (this->phases.size() + phase);
        if (phase1 < 0)
            phase1 = (this->phases.size() + phase1);

        // Validate phase indices up-front so that an out-of-range phase
        // throws std::invalid_argument with the bad phase number, instead
        // of std::out_of_range from the deeper phases[phase] access in
        // get_xt_up_vars below. Single-shot insertion (no loop), so body
        // failures further down are caught atomically by add_func_impl.
        // Mirrors add_forward_link_equal_con.
        if (phase < 0 || phase >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent phase:{0:}\n", phase));
        }
        if (phase1 < 0 || phase1 >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent phase:{0:}\n", phase1));
        }

        PhaseRegionFlags reg0 = get_PhaseRegion(reg0_t);

        int vsize = this->phases[phase]->get_xt_up_vars(reg0, v0).size();

        auto args = Arguments<-1>(2 * vsize);
        auto func = args.head<-1>(vsize) - args.tail<-1>(vsize);

        // Pass the wrapped int indices through, not the original p0/p1: the
        // inner add_link_equal_con eventually reaches make_func_impl, which
        // re-resolves the phase via get_phase_num and rejects negative values
        // with its own "Function references non-existent phase" throw. Passing
        // the already-wrapped positive ints keeps the negative-index wrap
        // protection at the top of this function meaningful.
        return this->add_link_equal_con(func, phase, reg0_t, v0, phase1, reg1_t, v1, scale_t);
    }

    /// @brief Chain consecutive phases with equal-parameter continuity constraints.
    /// @param iphase_t  Starting phase reference.
    /// @param fphase_t  Ending phase reference (must be after the start).
    /// @param reg0_t    Parameter region selector (must be @c ODEParams or @c StaticParams).
    /// @param vars      Parameter selector to make continuous.
    /// @param scale_t   Output-scale selector.
    /// @return The indices of the added constraints.
    /// @throws std::invalid_argument if the region is not a parameter region, a phase
    ///         reference is invalid, or @p iphase >= @p fphase.
    std::vector<int> add_param_link_equal_con(PhaseRefType iphase_t, PhaseRefType fphase_t,
                                              RegionType reg0_t, VarIndexType vars,
                                              ScaleType scale_t) {

        PhaseRegionFlags reg0 = get_PhaseRegion(reg0_t);

        if (reg0 != PhaseRegionFlags::ODEParams && reg0 != PhaseRegionFlags::StaticParams) {
            throw std::invalid_argument("Phase Region must be ODEParams or StaticParams");
        }

        int iphase = get_phase_num(iphase_t);
        int fphase = get_phase_num(fphase_t);

        if (iphase < 0)
            iphase = (this->phases.size() + iphase);
        if (fphase < 0)
            fphase = (this->phases.size() + fphase);

        // Validate phase indices up-front so a mid-loop *index* failure cannot
        // leave earlier link constraints partially inserted; see
        // add_forward_link_equal_con for the same rationale and scope.
        if (iphase < 0 || iphase >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent starting phase:{0:}\n", iphase));
        }
        if (fphase < 0 || fphase >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent ending phase:{0:}\n", fphase));
        }
        if (iphase >= fphase) {
            throw std::invalid_argument(
                fmt::format("add_param_link_equal_con requires iphase < fphase; got iphase={0:}, "
                            "fphase={1:}\n",
                            iphase, fphase));
        }

        std::vector<int> idxs;
        for (int i = iphase; i < fphase; i++) {

            int idx = this->add_direct_link_equal_con(i, reg0, vars, i + 1, reg0, vars, scale_t);

            idxs.push_back(idx);
        }

        return idxs;
    }

    /// @brief Add an equality constraint @f$f=0@f$ on the shared link parameters.
    /// @param lc       The link constraint function.
    /// @param lpvs     Per-application link-parameter index vectors.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_param_equal_con(VectorFunctionalX lc, std::vector<VectorXi> lpvs,
                                 ScaleType scale_t) {
        std::vector<Eigen::VectorXi> empty;
        auto LC =
            LinkConstraint(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs, scale_t);
        return add_func_impl(LC, this->link_equalities_, "Link Equality Constraint");
    }
    /// @brief Add an equality constraint on a single set of link parameters.
    /// @param lc       The link constraint function.
    /// @param lpv      Link-parameter index vector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_param_equal_con(VectorFunctionalX lc, VectorXi lpv, ScaleType scale_t) {
        std::vector<Eigen::VectorXi> lpvs;
        lpvs.push_back(lpv);
        return this->add_link_param_equal_con(lc, lpvs, scale_t);
    }

    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////

    /// @brief Add a link inequality constraint @f$f\le 0@f$ over a list of phase targets.
    /// @param lc       The link constraint function.
    /// @param packs    Per-phase link targets.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_inequal_con(VectorFunctionalX lc, std::vector<PhasePack> packs, VarIndexType lv,
                             ScaleType scale_t) {
        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, packs, lv, scale_t);
        return add_func_impl(Func, this->link_inequalities_, "Link Inequality Constraint");
    }

    /// @brief Add a two-phase link inequality constraint with full per-phase bindings.
    /// @param lc       The link constraint function.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param XtUV0    First phase state/time/control selector.
    /// @param OPV0     First phase ODE-parameter selector.
    /// @param SPV0     First phase static-parameter selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param XtUV1    Second phase state/time/control selector.
    /// @param OPV1     Second phase ODE-parameter selector.
    /// @param SPV1     Second phase static-parameter selector.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_inequal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0,
                             VarIndexType XtUV0, VarIndexType OPV0, VarIndexType SPV0,
                             PhaseRefType p1, RegionType reg1, VarIndexType XtUV1,
                             VarIndexType OPV1, VarIndexType SPV1, VarIndexType lv,
                             ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(
            lc, p0, reg0, XtUV0, OPV0, SPV0, p1, reg1, XtUV1, OPV1, SPV1, lv, scale_t);
        return add_func_impl(Func, this->link_inequalities_, "Link Inequality Constraint");
    }

    /// @brief Add a two-phase link inequality constraint with one selector per phase.
    /// @param lc       The link constraint function.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param v1       Second phase variable selector.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_inequal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0,
                             VarIndexType v0, PhaseRefType p1, RegionType reg1, VarIndexType v1,
                             VarIndexType lv, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, p0, reg0, v0, p1,
                                                                            reg1, v1, lv, scale_t);
        return add_func_impl(Func, this->link_inequalities_, "Link Inequality Constraint");
    }
    /// @brief Add a two-phase link inequality constraint with no link parameters.
    /// @param lc       The link constraint function.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param v1       Second phase variable selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_inequal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0,
                             VarIndexType v0, PhaseRefType p1, RegionType reg1, VarIndexType v1,
                             ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, p0, reg0, v0, p1,
                                                                            reg1, v1, scale_t);
        return add_func_impl(Func, this->link_inequalities_, "Link Inequality Constraint");
    }
    /// @brief Add an inequality constraint @f$f\le 0@f$ on the shared link parameters.
    /// @param lc       The link constraint function.
    /// @param lpvs     Per-application link-parameter index vectors.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_param_inequal_con(VectorFunctionalX lc, std::vector<VectorXi> lpvs,
                                   ScaleType scale_t) {
        std::vector<Eigen::VectorXi> empty;
        auto LC =
            LinkConstraint(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs, scale_t);
        return add_func_impl(LC, this->link_inequalities_, "Link Inequality Constraint");
    }
    /// @brief Add an inequality constraint on a single set of link parameters.
    /// @param lc       The link constraint function.
    /// @param lpv      Link-parameter index vector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the constraint.
    int add_link_param_inequal_con(VectorFunctionalX lc, VectorXi lpv, ScaleType scale_t) {
        std::vector<Eigen::VectorXi> lpvs;
        lpvs.push_back(lpv);
        return this->add_link_param_inequal_con(lc, lpvs, scale_t);
    }
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////

    /// @brief Add a link objective over a list of phase targets.
    /// @param lc       The scalar link objective.
    /// @param packs    Per-phase link targets.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_link_objective(ScalarFunctionalX lc, std::vector<PhasePack> packs, VarIndexType lv,
                           ScaleType scale_t) {
        auto Func = this->make_func_impl<LinkObjective, ScalarFunctionalX>(lc, packs, lv, scale_t);
        return add_func_impl(Func, this->link_objectives_, "Link Objective");
    }

    /// @brief Add a two-phase link objective with full per-phase bindings.
    /// @param lc       The scalar link objective.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param XtUV0    First phase state/time/control selector.
    /// @param OPV0     First phase ODE-parameter selector.
    /// @param SPV0     First phase static-parameter selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param XtUV1    Second phase state/time/control selector.
    /// @param OPV1     Second phase ODE-parameter selector.
    /// @param SPV1     Second phase static-parameter selector.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_link_objective(ScalarFunctionalX lc, PhaseRefType p0, RegionType reg0,
                           VarIndexType XtUV0, VarIndexType OPV0, VarIndexType SPV0,
                           PhaseRefType p1, RegionType reg1, VarIndexType XtUV1, VarIndexType OPV1,
                           VarIndexType SPV1, VarIndexType lv, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkObjective, ScalarFunctionalX>(
            lc, p0, reg0, XtUV0, OPV0, SPV0, p1, reg1, XtUV1, OPV1, SPV1, lv, scale_t);
        return add_func_impl(Func, this->link_objectives_, "Link Objective");
    }

    /// @brief Add a two-phase link objective with one selector per phase.
    /// @param lc       The scalar link objective.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param v1       Second phase variable selector.
    /// @param lv       Link-parameter selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_link_objective(ScalarFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                           PhaseRefType p1, RegionType reg1, VarIndexType v1, VarIndexType lv,
                           ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkObjective, ScalarFunctionalX>(lc, p0, reg0, v0, p1,
                                                                           reg1, v1, lv, scale_t);
        return add_func_impl(Func, this->link_objectives_, "Link Objective");
    }
    /// @brief Add a two-phase link objective with no link parameters.
    /// @param lc       The scalar link objective.
    /// @param p0       First phase reference.
    /// @param reg0     First phase region.
    /// @param v0       First phase variable selector.
    /// @param p1       Second phase reference.
    /// @param reg1     Second phase region.
    /// @param v1       Second phase variable selector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_link_objective(ScalarFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                           PhaseRefType p1, RegionType reg1, VarIndexType v1, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkObjective, ScalarFunctionalX>(lc, p0, reg0, v0, p1,
                                                                           reg1, v1, scale_t);
        return add_func_impl(Func, this->link_objectives_, "Link Objective");
    }
    /// @brief Add a link objective on the shared link parameters.
    /// @param lc       The scalar link objective.
    /// @param lpvs     Per-application link-parameter index vectors.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_link_param_objective(ScalarFunctionalX lc, std::vector<VectorXi> lpvs,
                                 ScaleType scale_t) {
        std::vector<Eigen::VectorXi> empty;
        auto LC =
            LinkObjective(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs, scale_t);
        return add_func_impl(LC, this->link_objectives_, "Link Objective");
    }
    /// @brief Add a link objective on a single set of link parameters.
    /// @param lc       The scalar link objective.
    /// @param lpv      Link-parameter index vector.
    /// @param scale_t  Output-scale selector.
    /// @return The index assigned to the objective.
    int add_link_param_objective(ScalarFunctionalX lc, VectorXi lpv, ScaleType scale_t) {
        std::vector<Eigen::VectorXi> lpvs;
        lpvs.push_back(lpv);
        return this->add_link_param_objective(lc, lpvs, scale_t);
    }

    ///////////////////////////////////////////////////

    /// @brief Remove a previously added link equality constraint by index.
    /// @param index  Constraint index, or -1 for the most recently added.
    void remove_link_equal_con(int index) {
        this->remove_func_impl(this->link_equalities_, index, "Equality Constraint");
    }
    /// @brief Remove a previously added link inequality constraint by index.
    /// @param index  Constraint index, or -1 for the most recently added.
    void remove_link_inequal_con(int index) {
        this->remove_func_impl(this->link_inequalities_, index, "Inequality Constraint");
    }
    /// @brief Remove a previously added link objective by index.
    /// @param index  Objective index, or -1 for the most recently added.
    void remove_link_objective(int index) {
        this->remove_func_impl(this->link_objectives_, index, "Link Objective");
    }
    ///////////////////////////////////////////////////

    /// @brief Return the residual values of a link equality constraint from the last solve.
    /// @param index  Link-equality-constraint index.
    /// @return Per-application residual vectors.
    /// @throws std::invalid_argument if no solve has been run or the index is unknown.
    std::vector<Eigen::VectorXd> return_link_equal_con_vals(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(" Post optimization info unavailable.");
        }
        if (this->link_equalities_.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No Equality Constraint with index {0:} exists in Optimal Control Problem.",
                index));
        }

        int Gindex = this->link_equalities_.at(index).global_index_;
        auto c_index_ = this->nlp_->equality_constraints_[Gindex].index_data_.c_index_;
        int offset = this->num_phase_eq_cons_.sum();

        std::vector<Eigen::VectorXd> Allvals;
        for (int i = 0; i < c_index_.cols(); i++) {
            VectorXd vals(c_index_.rows());
            for (int j = 0; j < c_index_.rows(); j++) {
                int idx = c_index_(j, i) - offset;
                vals[j] = this->active_eq_cons_[idx];
            }
            Allvals.push_back(vals);
        }
        return Allvals;
    }

    /// @brief Return the Lagrange multipliers of a link equality constraint from the last solve.
    /// @param index  Link-equality-constraint index.
    /// @return Per-application multiplier vectors.
    /// @throws std::invalid_argument if no solve has been run or the index is unknown.
    std::vector<Eigen::VectorXd> return_link_equal_con_lmults(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(" Post optimization info unavailable.");
        }
        if (this->link_equalities_.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No Equality Constraint with index {0:} exists in Optimal Control Problem.",
                index));
        }

        int Gindex = this->link_equalities_.at(index).global_index_;
        auto c_index_ = this->nlp_->equality_constraints_[Gindex].index_data_.c_index_;
        int offset = this->num_phase_eq_cons_.sum();

        std::vector<Eigen::VectorXd> Allvals;
        for (int i = 0; i < c_index_.cols(); i++) {
            VectorXd vals(c_index_.rows());
            for (int j = 0; j < c_index_.rows(); j++) {
                int idx = c_index_(j, i) - offset;
                vals[j] = this->active_eq_lmults_[idx];
            }
            Allvals.push_back(vals);
        }
        return Allvals;
    }

    /// @brief Return the residual values of a link inequality constraint from the last solve.
    /// @param index  Link-inequality-constraint index.
    /// @return Per-application residual vectors.
    /// @throws std::invalid_argument if no solve has been run or the index is unknown.
    std::vector<Eigen::VectorXd> return_link_inequal_con_vals(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(" Post optimization info unavailable.");
        }
        if (this->link_inequalities_.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No Inequality Constraint with index {0:} exists in Optimal Control Problem.",
                index));
        }
        int Gindex = this->link_inequalities_.at(index).global_index_;
        auto c_index_ = this->nlp_->inequality_constraints_[Gindex].index_data_.c_index_;
        int offset = this->num_phase_iq_cons_.sum();

        std::vector<Eigen::VectorXd> Allvals;
        for (int i = 0; i < c_index_.cols(); i++) {
            VectorXd vals(c_index_.rows());
            for (int j = 0; j < c_index_.rows(); j++) {
                int idx = c_index_(j, i) - offset;
                vals[j] = this->active_iq_cons_[idx];
            }
            Allvals.push_back(vals);
        }
        return Allvals;
    }

    /// @brief Return the Lagrange multipliers of a link inequality constraint from the last solve.
    /// @param index  Link-inequality-constraint index.
    /// @return Per-application multiplier vectors.
    /// @throws std::invalid_argument if no solve has been run or the index is unknown.
    std::vector<Eigen::VectorXd> return_link_inequal_con_lmults(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(" Post optimization info unavailable.");
        }
        if (this->link_inequalities_.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No Inequality Constraint with index {0:} exists in Optimal Control Problem.",
                index));
        }

        int Gindex = this->link_inequalities_.at(index).global_index_;
        auto c_index_ = this->nlp_->inequality_constraints_[Gindex].index_data_.c_index_;
        int offset = this->num_phase_iq_cons_.sum();

        std::vector<Eigen::VectorXd> Allvals;
        for (int i = 0; i < c_index_.cols(); i++) {
            VectorXd vals(c_index_.rows());
            for (int j = 0; j < c_index_.rows(); j++) {
                int idx = c_index_(j, i) - offset;
                vals[j] = this->active_iq_lmults_[idx];
            }
            Allvals.push_back(vals);
        }
        return Allvals;
    }

    /// @brief Return the output scales of a link equality constraint.
    /// @param index  Link-equality-constraint index.
    /// @return The constraint's per-output scale vector.
    Eigen::VectorXd return_link_equal_con_scales(int index) {
        return this->link_equalities_.at(index).output_scales_;
    }
    /// @brief Return the output scales of a link inequality constraint.
    /// @param index  Link-inequality-constraint index.
    /// @return The constraint's per-output scale vector.
    Eigen::VectorXd return_link_inequal_con_scales(int index) {
        return this->link_inequalities_.at(index).output_scales_;
    }
    /// @brief Return the output scales of a link objective.
    /// @param index  Link-objective index.
    /// @return The objective's per-output scale vector.
    Eigen::VectorXd return_link_objective_scales(int index) {
        return this->link_objectives_.at(index).output_scales_;
    }

    ///////////////////////////////////////////////////

    /// @internal
    /// @brief Compute the input scaling vector for a link function.
    /// @param lflag           The link pairing.
    /// @param regs            Region of each linked phase.
    /// @param phases_to_link  Phase indices per link application.
    /// @param XtUVars         Per-phase state/time/control indices.
    /// @param OPVars          Per-phase ODE-parameter indices.
    /// @param SPVars          Per-phase static-parameter indices.
    /// @param LVars           Link-parameter indices.
    /// @return The per-input scale vector.
    /// @endinternal
    Eigen::VectorXd get_input_scale(LinkFlags lflag, Eigen::Vector<PhaseRegionFlags, -1> regs,
                                    std::vector<VectorXi> phases_to_link,
                                    std::vector<VectorXi> XtUVars, std::vector<VectorXi> OPVars,
                                    std::vector<VectorXi> SPVars, std::vector<VectorXi> LVars);

    /// @internal
    /// @brief Sample representative inputs for a link function, used to seed auto-scaling.
    /// @param lflag           The link pairing.
    /// @param regs            Region of each linked phase.
    /// @param phases_to_link  Phase indices per link application.
    /// @param XtUVars         Per-phase state/time/control indices.
    /// @param OPVars          Per-phase ODE-parameter indices.
    /// @param SPVars          Per-phase static-parameter indices.
    /// @param LVars           Link-parameter indices.
    /// @return Representative input vectors for the link function.
    /// @endinternal
    std::vector<Eigen::VectorXd>
    get_test_inputs(LinkFlags lflag, Eigen::Vector<PhaseRegionFlags, -1> regs,
                    std::vector<VectorXi> phases_to_link, std::vector<VectorXi> XtUVars,
                    std::vector<VectorXi> OPVars, std::vector<VectorXi> SPVars,
                    std::vector<VectorXi> LVars);

    /// @internal
    /// @brief Propagate any phase's pending-transcription flag up to the problem.
    /// @endinternal
    void check_transcriptions() {
        for (int i = 0; i < this->phases.size(); i++) {
            if (this->phases[i]->do_transcription_) {
                this->do_transcription_ = true;
            }
        }
    }

    /// @internal
    /// @brief Transcribe all phases into the shared NLP.
    /// @endinternal
    void transcribe_phases();

    /// @internal
    /// @brief Validate that no two phases share a name or pointer.
    /// @throws std::invalid_argument if a duplicate phase name or pointer is found.
    /// @endinternal
    void check_dupilcate_phases() {
        for (int i = 0; i < this->phases.size(); i++) {
            for (int j = 0; j < this->phases.size(); j++) {
                if (j != i) {

                    if (this->phase_names[i] == this->phase_names[j]) {
                        throw std::invalid_argument(
                            "Transcription Error!!!\n"
                            "OptimalControlProblem contains two phases with identical names\n");
                    }

                    if (this->phases[i].get() == this->phases[j].get()) {

                        throw std::invalid_argument(
                            "Transcription Error!!!\n"
                            "Same phase detected more than once in optimal control problem.\n");
                    }
                }
            }
        }
    }

    /// @internal
    /// @brief Validate the IO sizes of all registered link functions.
    /// @endinternal
    void check_functions();

    /// @internal
    /// @brief Validate that a link function's input size matches its phase/region bindings.
    /// @tparam T  The link-function holder type.
    /// @param func   The link function to check.
    /// @param ftype  Human-readable function kind, for error messages.
    /// @throws std::invalid_argument if the input size or binding sizes are inconsistent.
    /// @endinternal
    template <class T> void check_function_size(const T &func, std::string ftype) {
        int irows = func.func_.input_rows();
        switch (func.link_flag_) {
        case LinkFlags::BackToFront:
        case LinkFlags::FrontToBack:
        case LinkFlags::FrontToFront:
        case LinkFlags::BackToBack:
        case LinkFlags::ParamsToParams:
        case LinkFlags::LinkParams:
        case LinkFlags::PathToPath:
        case LinkFlags::ReadRegions: {

            if (func.link_params_.size() != func.phases_to_link_.size() &&
                func.phases_to_link_.size() > 0) {
                throw std::invalid_argument(
                    "Transcription Error!!!\n"
                    "LinkParam Vector Must be same size as PTL Vector "
                    "even if each element of LinkParam Vector is empty (See Docs).");
            }

            if (func.link_params_.size() > 0 && func.phases_to_link_.size() == 0) {
                for (int i = 0; i < func.link_params_.size(); i++) {
                    int isize = func.link_params_[i].size();
                    if (irows != isize) {
                        throw std::invalid_argument(
                            fmt::format("Transcription Error!!!\n"
                                        "Input size of {} (IRows = {}) does not match that implied "
                                        "by indexing parameters (IRows = {}).\n",
                                        ftype, irows, isize));
                    }
                }
            } else {
                for (int i = 0; i < func.phases_to_link_.size(); i++) {
                    int isize = func.link_params_[i].size();

                    if (func.phases_to_link_[i].size() != func.xtu_vars_.size()) {
                        throw std::invalid_argument(
                            "Transcription Error!!!\n"
                            "Size of PTL vector element must equal size of Phase State "
                            "Variables Vector");
                    }
                    if (func.phases_to_link_[i].size() != func.op_vars_.size()) {
                        throw std::invalid_argument(
                            "Transcription Error!!!\n"
                            "Size of PTL vector element must equal size of Phase ODEParam "
                            "Variables Vector");
                    }
                    if (func.phases_to_link_[i].size() != func.sp_vars_.size()) {
                        throw std::invalid_argument(
                            "Transcription Error!!!\n"
                            "Size of PTL vector element must equal size of Phase "
                            "StaticParam Variables Vector");
                    }
                    if (func.phases_to_link_[i].size() != func.phase_reg_flags_.size()) {
                        throw std::invalid_argument(
                            "Transcription Error!!!\n"
                            "Size of PTL vector element must equal size of Phase Region "
                            "Flag Vector");
                    }
                    for (int j = 0; j < func.phases_to_link_[i].size(); j++) {
                        auto flag = func.phase_reg_flags_[j];
                        int xmult = 1;
                        switch (flag) {
                        case PhaseRegionFlags::Front:
                        case PhaseRegionFlags::Back:
                        case PhaseRegionFlags::Path:
                        case PhaseRegionFlags::ODEParams:
                        case PhaseRegionFlags::StaticParams:
                        case PhaseRegionFlags::Params:
                            xmult = 1;
                            break;
                        case PhaseRegionFlags::FrontandBack:
                        case PhaseRegionFlags::BackandFront:
                            xmult = 2;
                            break;
                        default: {
                            throw std::invalid_argument(
                                "Transcription Error!!!\n"
                                "Invalid Phase Region requested in link function\n"
                                "Only the following regions are supported\n"
                                "    Front, Back, Path, ODEParams, StaticParams, Params, "
                                "FrontandBack, BackAndFront\n");
                        }
                        }
                        isize += func.xtu_vars_[j].size() * xmult + func.op_vars_[j].size() +
                                 func.sp_vars_[j].size();
                    }
                    if (irows != isize) {
                        throw std::invalid_argument(
                            fmt::format("Transcription Error!!!\n"
                                        "Input size of {} (IRows = {}) does not match that implied "
                                        "by indexing parameters (IRows = {}).\n",
                                        ftype, irows, isize));
                    }
                }
            }

            break;
        }
        default: {
            break;
        }
        }
    }

    /// @internal
    /// @brief Transcribe all link constraints and objectives into the shared NLP.
    /// @endinternal
    void transcribe_links();

    /// @internal
    /// @brief Compute the automatic problem-level scales from the trajectories.
    /// @endinternal
    void calc_auto_scales();

    /// @internal
    /// @brief Collect the per-objective scale factors across the problem.
    /// @return The objective scale factors.
    /// @endinternal
    std::vector<double> get_objective_scales();
    /// @internal
    /// @brief Apply a uniform scale to all link objectives.
    /// @param scale  The scale to apply.
    /// @endinternal
    void update_objective_scales(double scale);

    /// @brief Transcribe the problem into its NLP, optionally printing diagnostics.
    /// @param showstats  Whether to print problem-size statistics.
    /// @param showfuns   Whether to print per-function details.
    void transcribe(bool showstats, bool showfuns);

    /// @brief Transcribe the problem into its NLP with diagnostics suppressed.
    void transcribe() { this->transcribe(false, false); }

    /// @internal
    /// @brief Prepare the problem for a "jet" (batched) solve.
    /// @endinternal
    void jet_initialize() {
        this->set_num_partitions(1, 1);
        this->optimizer_->set_print_level(10);
        this->print_mesh_info_ = false;
        this->transcribe();
    }
    /// @internal
    /// @brief Tear down jet-solve state and restore the default configuration.
    /// @endinternal
    void jet_release() {
        this->optimizer_->release();
        this->init_partitions();
        this->optimizer_->set_print_level(0);
        this->print_mesh_info_ = true;
        this->nlp_ = std::shared_ptr<NonLinearProgram>();
        for (auto &phase : this->phases)
            phase->jet_release();
        this->reset_transcription();
        this->invalidate_post_opt_info();
    }

    ////////////////////////////////////////////////////////////
  protected:
    /// @internal
    /// @brief Reset mesh-refinement state on every adaptive phase before a new solve.
    /// @endinternal
    void init_meshs() {
        this->mesh_converged_ = false;
        for (auto &phase : this->phases) {
            if (phase->adaptive_mesh_) {
                phase->init_mesh_refinement();
            }
        }
    }

    /// @internal
    /// @brief Check whether every adaptive phase's mesh has converged.
    /// @param printinfo  Whether to print per-phase mesh info.
    /// @return True if all adaptive phases have converged.
    /// @endinternal
    bool check_meshs(bool printinfo) {
        this->mesh_converged_ = true;
        for (auto &phase : this->phases) {
            if (phase->adaptive_mesh_) {
                if (!phase->check_mesh())
                    mesh_converged_ = false;
            }
        }

        return this->mesh_converged_;
    }
    /// @internal
    /// @brief Refine the mesh of every adaptive phase that has not yet converged.
    /// @param printinfo  Whether to print per-phase mesh info.
    /// @endinternal
    void update_meshs(bool printinfo) {
        for (auto &phase : this->phases) {
            if (phase->adaptive_mesh_) {
                if (!phase->mesh_converged_) {
                    phase->update_mesh();
                }
            }
        }
    }

    /// @internal
    /// @brief Print the mesh-refinement table for one problem-level iteration.
    /// @param iter  The mesh-iteration number.
    /// @endinternal
    void print_meshs(int iter) {
        MeshIterateInfo::print_header(iter);
        for (int i = 0; i < this->phases.size(); i++) {
            if (this->phases[i]->adaptive_mesh_) {
                this->phases[i]->mesh_iters_.back().print(i);
            }
        }
    }

    /// @internal
    /// @brief Build the variable-index and constraint-index matrices for a link function.
    /// @param Reg        The link pairing.
    /// @param PhaseRegs  Region of each linked phase.
    /// @param PTL        Phases-to-link index groups.
    /// @param xtv        Per-phase state/time/control indices.
    /// @param opv        Per-phase ODE-parameter indices.
    /// @param spv        Per-phase static-parameter indices.
    /// @param lv         Link-parameter indices.
    /// @param orows      Number of output (constraint) rows per evaluation.
    /// @param NextCLoc   In/out next free constraint-row location; advanced by this call.
    /// @return Array of {variable-index matrix, constraint-index matrix}.
    /// @endinternal
    std::array<Eigen::MatrixXi, 2> make_link_Vindex_Cindex(
        LinkFlags Reg, const Eigen::Matrix<PhaseRegionFlags, -1, 1> &PhaseRegs,
        const std::vector<Eigen::VectorXi> &PTL, const std::vector<Eigen::VectorXi> &xtv,
        const std::vector<Eigen::VectorXi> &opv, const std::vector<Eigen::VectorXi> &spv,
        const std::vector<Eigen::VectorXi> &lv, int orows, int &NextCLoc) const;

    /// @internal
    /// @brief Drive PSIOPT for a single solve/optimize pass (no mesh loop).
    /// @param mode  The solve/optimize job mode.
    /// @return The solver convergence flag.
    /// @endinternal
    PSIOPT::ConvergenceFlags psipot_call_impl(JetJobModes mode);

    /// @internal
    /// @brief Run the full problem solve, including the multi-phase mesh-refinement loop.
    /// @param mode  The solve/optimize job mode.
    /// @return The solver convergence flag.
    /// @endinternal
    PSIOPT::ConvergenceFlags ocp_call_impl(JetJobModes mode);

    /// @internal
    /// @brief Assemble the full problem decision vector from all phases and link params.
    /// @return The packed decision-variable vector (scaled if auto-scaling is on).
    /// @endinternal
    VectorXd make_solver_input() const {
        VectorXd Vars(this->num_prob_vars_);

        for (int i = 0; i < this->phases.size(); i++) {
            int Start = 0;
            if (i > 0)
                Start = this->num_phase_vars_.segment(0, i).sum();
            Vars.segment(Start, this->num_phase_vars_[i]) = this->phases[i]->make_solver_input();
        }
        if (this->auto_scaling_ && this->lp_units_.size() > 0) {
            Vars.tail(this->num_link_params_) =
                this->active_link_params_.cwiseQuotient(this->lp_units_);
        } else {
            Vars.tail(this->num_link_params_) = this->active_link_params_;
        }

        return Vars;
    }

    /// @internal
    /// @brief Distribute the solved decision vector back into the phases and link params.
    /// @param Vars  The solved decision-variable vector.
    /// @endinternal
    void collect_solver_output(const VectorXd &Vars) {
        for (int i = 0; i < this->phases.size(); i++) {
            int Start = 0;
            if (i > 0)
                Start = this->num_phase_vars_.segment(0, i).sum();
            this->phases[i]->collect_solver_output(Vars.segment(Start, this->num_phase_vars_[i]));
        }
        if (this->auto_scaling_ && this->lp_units_.size() > 0) {
            this->active_link_params_ =
                Vars.tail(this->num_link_params_).cwiseProduct(this->lp_units_);
        } else {
            this->active_link_params_ = Vars.tail(this->num_link_params_);
        }
    }
    /// @internal
    /// @brief Distribute the solved multipliers into the phases and the link slots.
    /// @param EM  Equality-constraint multipliers for the whole problem.
    /// @param IM  Inequality-constraint multipliers for the whole problem.
    /// @endinternal
    void collect_solver_multipliers(const VectorXd &EM, const VectorXd &IM) {
        this->multipliers_loaded_ = true;
        for (int i = 0; i < this->phases.size(); i++) {
            int EStart = 0;
            if (i > 0)
                EStart = this->num_phase_eq_cons_.segment(0, i).sum();
            int IStart = 0;
            if (i > 0)
                IStart = this->num_phase_iq_cons_.segment(0, i).sum();
            this->phases[i]->collect_solver_multipliers(
                EM.segment(EStart, this->num_phase_eq_cons_[i]),
                IM.segment(IStart, this->num_phase_iq_cons_[i]));
        }
        this->active_eq_lmults_ = EM.tail(this->num_link_eq_cons_);
        this->active_iq_lmults_ = IM.tail(this->num_link_iq_cons_);
    }

    /// @internal
    /// @brief Distribute post-optimization constraint values and multipliers.
    /// @param EC  Equality-constraint residuals for the whole problem.
    /// @param EM  Equality-constraint multipliers for the whole problem.
    /// @param IC  Inequality-constraint residuals for the whole problem.
    /// @param IM  Inequality-constraint multipliers for the whole problem.
    /// @endinternal
    void collect_post_opt_info(const VectorXd &EC, const VectorXd &EM, const VectorXd &IC,
                               const VectorXd &IM) {
        this->multipliers_loaded_ = true;
        this->post_opt_info_valid_ = true;

        for (int i = 0; i < this->phases.size(); i++) {
            int EStart = 0;
            if (i > 0)
                EStart = this->num_phase_eq_cons_.segment(0, i).sum();
            int IStart = 0;
            if (i > 0)
                IStart = this->num_phase_iq_cons_.segment(0, i).sum();
            this->phases[i]->collect_post_opt_info(EC.segment(EStart, this->num_phase_eq_cons_[i]),
                                                   EM.segment(EStart, this->num_phase_eq_cons_[i]),
                                                   IC.segment(IStart, this->num_phase_iq_cons_[i]),
                                                   IM.segment(IStart, this->num_phase_iq_cons_[i]));
        }
        this->active_eq_lmults_ = EM.tail(this->num_link_eq_cons_);
        this->active_iq_lmults_ = IM.tail(this->num_link_iq_cons_);
        this->active_eq_cons_ = EC.tail(this->num_link_eq_cons_);
        this->active_iq_cons_ = IC.tail(this->num_link_iq_cons_);
    }

  public:
    /// @brief Solve the multi-phase problem for feasibility (no optimization).
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags solve() { return ocp_call_impl(JetJobModes::Solve); }
    /// @brief Optimize the multi-phase problem (minimize the objective subject to constraints).
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags optimize() { return ocp_call_impl(JetJobModes::Optimize); }
    /// @brief Solve for feasibility, then optimize.
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags solve_optimize() { return ocp_call_impl(JetJobModes::SolveOptimize); }
    /// @brief Solve, optimize, then solve again to tighten feasibility.
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags solve_optimize_solve() {
        return ocp_call_impl(JetJobModes::SolveOptimizeSolve);
    }
    /// @brief Optimize, then solve to tighten feasibility.
    /// @return The solver convergence flag.
    PSIOPT::ConvergenceFlags optimize_solve() { return ocp_call_impl(JetJobModes::OptimizeSolve); }

    /// @brief Print the problem-size statistics.
    /// @param showfuns  Whether to also print per-function details.
    void print_stats(bool showfuns);
};

} // namespace tycho::oc
