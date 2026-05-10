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

struct OptimalControlProblemBase;

struct ODEPhaseBase : ODESize<-1, -1, -1>, OptimizationProblemBase {
    using VectorXi = Eigen::VectorXi;
    using MatrixXi = Eigen::MatrixXi;

    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    using VectorFunctionalX = GenericFunction<-1, -1>;
    using ScalarFunctionalX = GenericFunction<-1, 1>;

    using StateConstraint = StateFunction<VectorFunctionalX>;
    using StateObjective = StateFunction<ScalarFunctionalX>;

    template <int V> using int_const = std::integral_constant<int, V>;

    friend OptimalControlProblemBase;

  protected:
    PhaseIndexer indexer_;
    bool do_transcription_ = true;
    bool enable_vectorization_ = true;

    int num_defects_ = 0;
    int num_stat_params_ = 0;
    VectorXd def_bin_spacing_;
    VectorXi defs_per_bin_;

    TranscriptionModes transcription_mode_ = TranscriptionModes::LGL3;
    ControlModes control_mode_ = ControlModes::FirstOrderSpline;
    IntegralModes integral_mode_ = IntegralModes::BaseIntegral;
    int num_tran_card_states_ = 2;
    double order_ = 3;

    LGLInterpTable table_;

    bool trajectory_loaded_ = false;
    std::vector<VectorXd> active_traj_;
    VectorXd active_ode_params_;
    VectorXd active_static_params_;

    bool multipliers_loaded_ = false;
    bool post_opt_info_valid_ = false;

    VectorXd active_eq_lmults_;
    VectorXd active_iq_lmults_;
    VectorXd active_eq_cons_;
    VectorXd active_iq_cons_;

    std::map<int, StateConstraint> user_equalities_;
    std::map<int, StateConstraint> user_inequalities_;
    std::map<int, StateObjective> user_state_objectives_;
    std::map<int, StateObjective> user_integrands_;
    std::map<int, StateObjective> user_param_integrands_;

    int dynamics_func_index_ = 0;
    int control_funcs_index_ = -1;
    VectorXi node_spacing_func_indices_;
    int tran_spacing_func_indices_ = 0;
    int constraint_order_ = 0;

    //////////////////////////

    Eigen::VectorXd xtup_units_;
    Eigen::VectorXd sp_units_;

    std::map<std::string, Eigen::VectorXi> sp_idxs_;

    ///////////////////////
  public:
    bool auto_scaling_ = false;
    bool sync_objective_scales_ = true;

    bool adaptive_mesh_ = false;
    bool print_mesh_info_ = true;
    int max_mesh_iters_ = 10;
    int max_segments_ = 10000;
    int min_segments_ = 4;

    int num_extra_segs_ = 4;
    double mesh_red_factor_ = .5;
    double mesh_inc_factor_ = 5.0;
    double mesh_err_factor_ = 10.0;
    bool force_one_mesh_iter_ = false;
    bool solve_only_first_ = true;
    bool new_error_ = false;
    bool detect_control_switches_ = false;

    double rel_switch_tol_ = .3;
    double abs_switch_tol_ = 1.0e-6;

    double mesh_tol_ = 1.0e-6;

    MeshErrorEstimators mesh_error_estimator_ = MeshErrorEstimators::DEBOOR;
    MeshErrorAggregation mesh_error_criteria_ = MeshErrorAggregation::MAX;
    MeshErrorAggregation mesh_error_distributor_ = MeshErrorAggregation::AVG;
    PSIOPT::ConvergenceFlags mesh_abort_flag_ = PSIOPT::ConvergenceFlags::DIVERGING;

    bool mesh_converged_ = false;

    std::vector<MeshIterateInfo> mesh_iters_;

    void set_auto_scaling(bool autoscale) {
        this->auto_scaling_ = autoscale;
        this->reset_transcription();
        this->invalidate_post_opt_info();
    }

    void set_adaptive_mesh(bool amesh) { this->adaptive_mesh_ = amesh; }
    void set_mesh_tol(double t) { this->mesh_tol_ = abs(t); }
    void set_mesh_red_factor(double t) { this->mesh_red_factor_ = abs(t); }
    void set_mesh_inc_factor(double t) { this->mesh_inc_factor_ = abs(t); }
    void set_mesh_err_factor(double t) { this->mesh_err_factor_ = abs(t); }
    void set_max_mesh_iters(int it) { this->max_mesh_iters_ = abs(it); }
    void set_min_segments(int it) { this->min_segments_ = abs(it); }
    void set_max_segments(int it) { this->max_segments_ = abs(it); }
    void set_mesh_error_criteria(MeshErrorAggregation m) { this->mesh_error_criteria_ = m; }
    void set_mesh_error_criteria(const std::string &m) {
        this->mesh_error_criteria_ = strto_MeshErrorAggregation(m);
    }
    void set_mesh_error_estimator(MeshErrorEstimators m) { this->mesh_error_estimator_ = m; }
    void set_mesh_error_estimator(const std::string &m) {
        this->mesh_error_estimator_ = strto_MeshErrorEstimator(m);
    }
    void set_mesh_error_distributor(MeshErrorAggregation m) { this->mesh_error_distributor_ = m; }
    void set_mesh_error_distributor(const std::string &m) {
        this->mesh_error_distributor_ = strto_MeshErrorAggregation(m);
    }

    std::vector<MeshIterateInfo> get_mesh_iters() const { return this->mesh_iters_; }

    ///////////////////

  public:
    /////////////////////////////////////////////////////////////////////////////

    void enable_vectorization(bool b) { this->enable_vectorization_ = b; }
    void reset_transcription() { this->do_transcription_ = true; };

    void invalidate_post_opt_info() { this->post_opt_info_valid_ = false; };

    ODEPhaseBase() {}
    ODEPhaseBase(int Xv, int Uv, int Pv) {
        this->set_xvars(Xv);
        this->set_uvars(Uv);
        this->set_pvars(Pv);

        this->xtup_units_ = Eigen::VectorXd::Ones(this->xtu_p_vars());
    }
    virtual ~ODEPhaseBase() = default;

    //////////////////////////////////////////////////

    virtual void set_units(const Eigen::VectorXd &xtup_units) = 0;

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

    void set_control_mode(std::string m) { this->set_control_mode(strto_ControlMode(m)); }

    virtual void set_integral_mode(IntegralModes m) {
        this->reset_transcription();
        this->integral_mode_ = m;
    }
    virtual void set_transcription_mode(TranscriptionModes m) = 0;

    void switch_transcription_mode(TranscriptionModes m, VectorXd DBS, VectorXi DPB) {
        this->set_transcription_mode(m);
        this->set_traj(this->active_traj_, DBS, DPB);
    }
    void switch_transcription_mode(TranscriptionModes m) {
        this->switch_transcription_mode(m, this->def_bin_spacing_, this->defs_per_bin_);
    }

    void switch_transcription_mode(std::string m, VectorXd DBS, VectorXi DPB) {
        this->switch_transcription_mode(strto_TranscriptionMode(m), DBS, DPB);
    }
    void switch_transcription_mode(std::string m) {
        this->switch_transcription_mode(strto_TranscriptionMode(m));
    }

    ///////////////////////////////////////////////////

    void set_static_param_vgroups(std::map<std::string, Eigen::VectorXi> spidxs) {
        this->sp_idxs_ = spidxs;
    }
    void add_static_param_vgroups(std::map<std::string, Eigen::VectorXi> spidxs) {
        for (auto &[key, value] : spidxs) {
            this->sp_idxs_[key] = value;
        }
    }
    void add_static_param_vgroup(Eigen::VectorXi idx, std::string key) {
        this->sp_idxs_[key] = idx;
    }
    void add_static_param_vgroup(int idx, std::string key) {
        VectorXi tmp(1);
        tmp << idx;
        this->sp_idxs_[key] = tmp;
    }

    VectorXi get_sp_idx(std::string key) const {
        if (sp_idxs_.count(key) == 0) {
            throw std::invalid_argument(
                fmt::format("No StaticParam variable index group with name: {0:} exists.", key));
        }
        return this->sp_idxs_.at(key);
    }

    /////////////////////////////////////////////////
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

    PhaseRegionFlags get_region(RegionType reg_t) const {
        PhaseRegionFlags reg;

        if (std::holds_alternative<PhaseRegionFlags>(reg_t)) {
            reg = std::get<PhaseRegionFlags>(reg_t);
        } else if (std::holds_alternative<std::string>(reg_t)) {
            reg = strto_PhaseRegionFlag(std::get<std::string>(reg_t));
        }
        return reg;
    }

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
    int add_equal_con(StateConstraint con) {
        return add_func_impl(con, this->user_equalities_, "Equality Constraint");
    }

    int add_equal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType xtup_vars_t,
                      VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                      OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_equalities_, "Equality Constraint");
    }

    int add_equal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType xtup_vars_t,
                      ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                      empty, empty, scale_t);
        return add_func_impl(con, this->user_equalities_, "Equality Constraint");
    }

    int add_boundary_value(RegionType reg, VarIndexType args,
                           const std::variant<double, VectorXd> &value, ScaleType scale_t);
    int add_delta_var_equal_con(VarIndexType var, double value, double scale, ScaleType scale_t);
    int add_delta_time_equal_con(double value, double scale, ScaleType scale_t) {
        return this->add_delta_var_equal_con(this->t_var(), value, scale, scale_t);
    }
    int add_value_lock(RegionType reg, VarIndexType args, ScaleType scale_t);
    int add_periodicity_con(VarIndexType args, ScaleType scale_t);

    /////////////////////////////////////////////////

    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int add_inequal_con(StateConstraint con) {
        return add_func_impl(con, this->user_inequalities_, "Inequality Constraint");
    }
    int add_inequal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType xtup_vars_t,
                        VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                      OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_inequalities_, "Inequality Constraint");
    }

    int add_inequal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType xtup_vars_t,
                        ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                      empty, empty, scale_t);
        return add_func_impl(con, this->user_inequalities_, "Inequality Constraint");
    }

    ////////////////////////
    int add_lu_var_bound(RegionType reg, VarIndexType var, double lowerbound, double upperbound,
                         double lbscale, double ubscale, ScaleType scale_t);

    int add_lu_var_bound(RegionType reg, VarIndexType var, double lowerbound, double upperbound,
                         double scale, ScaleType scale_t) {
        return this->add_lu_var_bound(reg, var, lowerbound, upperbound, scale, scale, scale_t);
    }
    int add_lu_var_bound(RegionType reg, VarIndexType var, double lowerbound, double upperbound,
                         ScaleType scale_t) {
        return this->add_lu_var_bound(reg, var, lowerbound, upperbound, 1.0, 1.0, scale_t);
    }

    int add_lower_var_bound(RegionType reg, VarIndexType var, double lowerbound, double lbscale,
                            ScaleType scale_t);

    int add_upper_var_bound(RegionType reg, VarIndexType var, double upperbound, double ubscale,
                            ScaleType scale_t);

    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                          VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                          double upperbound, double lbscale, double ubscale, ScaleType scale_t);

    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                          double lowerbound, double upperbound, double lbscale, double ubscale,
                          ScaleType scale_t) {

        VectorXi empty;

        return add_lu_func_bound(reg, func, xtup_vars, empty, empty, lowerbound, upperbound,
                                 lbscale, ubscale, scale_t);
    }

    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                          VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                          double upperbound, double scale, ScaleType scale_t) {
        return add_lu_func_bound(reg, func, xtup_vars, OPvars, SPvars, lowerbound, upperbound,
                                 scale, scale, scale_t);
    }

    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                          double lowerbound, double upperbound, double scale, ScaleType scale_t) {
        VectorXi empty;
        return add_lu_func_bound(reg, func, xtup_vars, empty, empty, lowerbound, upperbound, scale,
                                 scale, scale_t);
    }

    int add_lower_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                             VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                             double lbscale, ScaleType scale_t);

    int add_lower_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                             double lowerbound, double lbscale, ScaleType scale_t) {
        VectorXi empty;

        return this->add_lower_func_bound(reg, func, xtup_vars, empty, empty, lowerbound, lbscale,
                                          scale_t);
    }

    int add_upper_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                             VarIndexType OPvars, VarIndexType SPvars, double upperbound,
                             double ubscale, ScaleType scale_t);

    int add_upper_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType xtup_vars,
                             double upperbound, double ubscale, ScaleType scale_t) {
        VectorXi empty;

        return this->add_upper_func_bound(reg, func, xtup_vars, empty, empty, upperbound, ubscale,
                                          scale_t);
    }

    int add_lu_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                          double upperbound, double lbscale, double ubscale, ScaleType scale_t);

    int add_lu_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                          double upperbound, double scale, ScaleType scale_t) {
        return this->add_lu_norm_bound(reg, xtup_vars, lowerbound, upperbound, scale, scale,
                                       scale_t);
    }

    int add_lu_squared_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                                  double upperbound, double lbscale, double ubscale,
                                  ScaleType scale_t);

    int add_lu_squared_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                                  double upperbound, double scale, ScaleType scale_t) {
        return this->add_lu_squared_norm_bound(reg, xtup_vars, lowerbound, upperbound, scale, scale,
                                               scale_t);
    }

    int add_lower_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                             double lbscale, ScaleType scale_t);

    int add_lower_squared_norm_bound(RegionType reg, VarIndexType xtup_vars, double lowerbound,
                                     double lbscale, ScaleType scale_t);

    int add_upper_norm_bound(RegionType reg, VarIndexType xtup_vars, double upperbound,
                             double ubscale, ScaleType scale_t);

    int add_upper_squared_norm_bound(RegionType reg, VarIndexType xtup_vars, double upperbound,
                                     double ubscale, ScaleType scale_t);
    //
    int add_lower_delta_var_bound(RegionType reg, VarIndexType var, double lowerbound,
                                  double lbscale, ScaleType scale_t);
    int add_lower_delta_var_bound(VarIndexType var, double lowerbound, double lbscale,
                                  ScaleType scale_t) {
        return this->add_lower_delta_var_bound(PhaseRegionFlags::FrontandBack, var, lowerbound,
                                               lbscale, scale_t);
    }
    int add_lower_delta_time_bound(double lowerbound, double lbscale, ScaleType scale_t) {
        return this->add_lower_delta_var_bound(this->t_var(), lowerbound, lbscale, scale_t);
    }
    ///
    int add_upper_delta_var_bound(RegionType reg, VarIndexType var, double upperbound,
                                  double ubscale, ScaleType scale_t);
    int add_upper_delta_var_bound(VarIndexType var, double upperbound, double ubscale,
                                  ScaleType scale_t) {
        return this->add_upper_delta_var_bound(PhaseRegionFlags::FrontandBack, var, upperbound,
                                               ubscale, scale_t);
    }
    int add_upper_delta_time_bound(double upperbound, double ubscale, ScaleType scale_t) {
        return this->add_upper_delta_var_bound(this->t_var(), upperbound, ubscale, scale_t);
    }

    ///////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    int add_lu_var_bound(PhaseRegionFlags reg, int var, double lowerbound, double upperbound,
                         double lbscale, double ubscale);

    int add_lu_var_bound(PhaseRegionFlags reg, int var, double lowerbound, double upperbound,
                         double scale) {
        return this->add_lu_var_bound(reg, var, lowerbound, upperbound, scale, scale);
    }

    Eigen::VectorXi add_lu_var_bounds(PhaseRegionFlags reg, Eigen::VectorXi vars, double lowerbound,
                                      double upperbound, double scale) {
        Eigen::VectorXi cnums(vars.size());
        for (int i = 0; i < cnums.size(); i++) {
            cnums[i] = this->add_lu_var_bound(reg, vars[i], lowerbound, upperbound, scale, scale);
        }

        return cnums;
    }

    Eigen::VectorXi add_lu_var_bounds(std::string reg, Eigen::VectorXi vars, double lowerbound,
                                      double upperbound, double scale) {
        return add_lu_var_bounds(strto_PhaseRegionFlag(reg), vars, lowerbound, upperbound, scale);
    }

    ////////////////////////////////////////////////////

    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int add_state_objective(StateObjective obj) {
        return add_func_impl(obj, this->user_state_objectives_, "State Objective");
    }
    int add_state_objective(RegionType reg_t, ScalarFunctionalX fun, VarIndexType xtup_vars_t,
                            VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(reg_t, fun, xtup_vars_t,
                                                                     OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_state_objectives_, "State Objective");
    }

    int add_state_objective(RegionType reg_t, ScalarFunctionalX fun, VarIndexType xtup_vars_t,
                            ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(reg_t, fun, xtup_vars_t, empty,
                                                                     empty, scale_t);
        return add_func_impl(con, this->user_state_objectives_, "State Objective");
    }
    int add_value_objective(RegionType reg, VarIndexType var, double scale, ScaleType scale_t);
    int add_delta_var_objective(VarIndexType var, double scale, ScaleType scale_t);
    int add_delta_time_objective(double scale, ScaleType scale_t) {
        return this->add_delta_var_objective(this->t_var(), scale, scale_t);
    }

    ///////////////////////////////////////////////

    ///////////////////////////////////////////////////
    int add_integral_objective(StateObjective obj) {
        return add_func_impl(obj, this->user_integrands_, "Integral Objective");
    }

    int add_integral_objective(ScalarFunctionalX fun, VarIndexType xtup_vars_t,
                               VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(
            PhaseRegionFlags::Path, fun, xtup_vars_t, OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_integrands_, "Integral Objective");
    }

    int add_integral_objective(ScalarFunctionalX fun, VarIndexType xtup_vars_t, ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(
            PhaseRegionFlags::Path, fun, xtup_vars_t, empty, empty, scale_t);
        return add_func_impl(con, this->user_integrands_, "Integral Objective");
    }

    ///////////////////////////////////////////////////
    int add_integral_param_function(StateObjective con, int pv) {
        VectorXi epv(1);
        epv[0] = pv;
        int index = add_func_impl(con, this->user_param_integrands_, "Integral Parameter Function");
        this->user_param_integrands_[index].ext_vars_ = epv;
        return index;
    }

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

    int add_integral_param_function(ScalarFunctionalX fun, VarIndexType xtup_vars_t, int accum_parm,
                                    ScaleType scale_t) {
        VectorXi empty;
        return add_integral_param_function(fun, xtup_vars_t, empty, empty, accum_parm, scale_t);
    }

    /////////////////////////////////////////////////

    void remove_equal_con(int index) {
        this->remove_func_impl(this->user_equalities_, index, "Equality Constraint");
    }
    void remove_inequal_con(int index) {
        this->remove_func_impl(this->user_inequalities_, index, "Inequality Constraint");
    }
    void remove_state_objective(int index) {
        this->remove_func_impl(this->user_state_objectives_, index, "State Objective");
    }
    void remove_integral_objective(int index) {
        this->remove_func_impl(this->user_integrands_, index, "Integral Objective");
    }
    void remove_integral_param_function(int index) {
        this->remove_func_impl(this->user_param_integrands_, index, "Integral Parameter Function");
    }

    /////////////////////////////////////////////////

    virtual void set_traj(const std::vector<Eigen::VectorXd> &mesh, Eigen::VectorXd DBS,
                          Eigen::VectorXi DPB, bool LerpTraj);

    void set_traj(const std::vector<Eigen::VectorXd> &mesh);

    void set_traj(const std::vector<Eigen::VectorXd> &mesh, Eigen::VectorXd DBS,
                  Eigen::VectorXi DPB) {
        this->set_traj(mesh, DBS, DPB, false);
    }

    void set_traj(const std::vector<Eigen::VectorXd> &mesh, int ndef, bool LerpTraj) {
        VectorXd dbs(2);
        dbs[0] = 0.0;
        dbs[1] = 1.0;
        VectorXi dpb(1);
        dpb[0] = ndef;
        this->set_traj(mesh, dbs, dpb, LerpTraj);
    }
    void set_traj(const std::vector<Eigen::VectorXd> &mesh, int ndef) {
        this->set_traj(mesh, ndef, false);
    }

    void refine_traj_manual(VectorXd DBS, VectorXi DPB);

    void refine_traj_manual(int ndef) {
        VectorXd dbs(2);
        dbs[0] = 0.0;
        dbs[1] = 1.0;
        VectorXi dpb(1);
        dpb[0] = ndef;
        this->refine_traj_manual(dbs, dpb);
    }
    std::vector<Eigen::VectorXd> refine_traj_equal(int n);

    void refine_traj_auto();

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
    void set_static_params(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->set_static_params(parm, units);
    }

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
    void add_static_params(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->add_static_params(parm, units);
    }

    void sub_static_params(VectorXd parm) {
        if (this->num_stat_params_ == parm.size()) {
            this->active_static_params_ = parm;
        } else {
            this->set_static_params(parm);
        }
    }

    void sub_variables(PhaseRegionFlags reg, VectorXi indices, VectorXd vals);
    void sub_variable(PhaseRegionFlags reg, int var, double val) {
        VectorXi indices(1);
        indices[0] = var;
        VectorXd vals(1);
        vals[0] = val;
        this->sub_variables(reg, indices, vals);
    }

    void sub_variables(std::string reg, VectorXi indices, VectorXd vals) {
        this->sub_variables(strto_PhaseRegionFlag(reg), indices, vals);
    }
    void sub_variable(std::string reg, int var, double val) {
        this->sub_variable(strto_PhaseRegionFlag(reg), var, val);
    }

    std::vector<Eigen::VectorXd> return_traj() const { return this->active_traj_; }

    std::vector<Eigen::VectorXd> return_traj_range(int num, double tl, double th) {
        this->table_.load_regular_data(this->defs_per_bin_.sum(), this->active_traj_);
        return this->table_.interp_range(num, tl, th);
    }
    std::vector<Eigen::VectorXd> return_traj_range_nd(int num, double tl, double th) {
        this->table_.load_regular_data(this->defs_per_bin_.sum(), this->active_traj_);
        return this->table_.nd_equidist(num, tl, th);
    }
    LGLInterpTable return_traj_table() {
        LGLInterpTable tabt = this->table_;
        tabt.load_exact_data(this->active_traj_);
        return tabt;
    }

    Eigen::VectorXd return_static_params() const { return this->active_static_params_; }

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

    std::vector<Eigen::VectorXd> return_equal_con_lmults(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No multipliers to return, a solve or optimize call must be made "
                "before returning constraint multipliers ");
        }
        int Gindex = this->user_equalities_.at(index).global_index_;
        return this->indexer_.get_func_eq_multipliers(Gindex, this->active_eq_lmults_);
    }
    std::vector<Eigen::VectorXd> return_equal_con_vals(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No constraints to return, a solve or optimize call must be made "
                "before returning constraint values ");
        }
        int Gindex = this->user_equalities_.at(index).global_index_;
        return this->indexer_.get_func_eq_multipliers(Gindex, this->active_eq_cons_);
    }
    Eigen::VectorXd return_equal_con_scales(int index) const {
        return this->user_equalities_.at(index).output_scales_;
    }

    std::vector<Eigen::VectorXd> return_inequal_con_lmults(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No multipliers to return, a solve or optimize call must be made "
                "before returning constraint multipliers ");
        }
        int Gindex = this->user_inequalities_.at(index).global_index_;
        return this->indexer_.get_func_iq_multipliers(Gindex, this->active_iq_lmults_);
    }

    std::vector<Eigen::VectorXd> return_inequal_con_vals(int index) const {
        if (!this->post_opt_info_valid_) {
            throw std::invalid_argument(
                "No constraints to return, a solve or optimize call must be made "
                "before returning constraint values ");
        }
        int Gindex = this->user_inequalities_.at(index).global_index_;
        return this->indexer_.get_func_iq_multipliers(Gindex, this->active_iq_cons_);
    }
    Eigen::VectorXd return_inequal_con_scales(int index) const {
        return this->user_inequalities_.at(index).output_scales_;
    }

    std::vector<Eigen::VectorXd> return_costate_traj() const;
    std::vector<Eigen::VectorXd> return_traj_error() const;

    Eigen::VectorXd return_integral_objective_scales(int index) const {
        return this->user_integrands_.at(index).output_scales_;
    }
    Eigen::VectorXd return_integral_param_function_scales(int index) const {
        return this->user_param_integrands_.at(index).output_scales_;
    }
    Eigen::VectorXd return_state_objective_scales(int index) const {
        return this->user_state_objectives_.at(index).output_scales_;
    }
    Eigen::VectorXd return_ode_output_scales() const {
        VectorXd output_scales =
            xtup_units_.head(this->x_vars()).cwiseInverse() * this->xtup_units_[this->x_vars()];
        return output_scales;
    }

    /////////////////////////////////////////////////
  protected:
    virtual void transcribe_dynamics() = 0;
    virtual void transcribe_axis_funcs();
    virtual void transcribe_control_funcs();
    virtual void transcribe_integrals();
    virtual void transcribe_basic_funcs();

    void init_indexing() {
        this->indexer_ =
            PhaseIndexer(this->x_vars(), this->u_vars(), this->p_vars(), this->num_stat_params_);
        this->indexer_.set_dimensions(this->num_tran_card_states_, this->num_defects_,
                                      this->control_mode_ == ControlModes::BlockConstant);
    }
    void check_functions(int pnum);

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
    Eigen::VectorXd get_input_scale(PhaseRegionFlags flag, VectorXi XtUV, VectorXi OPV,
                                    VectorXi SPV) const;

  protected:
    std::vector<Eigen::VectorXd> get_test_inputs(PhaseRegionFlags flag, VectorXi XtUV, VectorXi OPV,
                                                 VectorXi SPV) const;

    void calc_auto_scales();

    std::vector<double> get_objective_scales();
    void update_objective_scales(double scale);

    static void check_lbscale(double lbscale) {
        if (lbscale <= 0.0) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "Lower-bound scale ({0:.3e}) must be a strictly positive number.\n",
                       lbscale);
            throw std::invalid_argument("");
        }
    }
    static void check_ubscale(double ubscale) {
        if (ubscale <= 0.0) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "Upper-bound scale ({0:.3e}) must be a strictly positive number.\n",
                       ubscale);
            throw std::invalid_argument("");
        }
    }

    void transcribe_phase(int vo, int eqo, int iqo, std::shared_ptr<NonLinearProgram> np, int pnum);

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
    void collect_solver_multipliers(const VectorXd &EM, const VectorXd &IM) {
        this->multipliers_loaded_ = true;
        this->active_eq_lmults_ = EM;
        this->active_iq_lmults_ = IM;
    }

    void collect_post_opt_info(const VectorXd &EC, const VectorXd &EM, const VectorXd &IC,
                               const VectorXd &IM) {

        this->post_opt_info_valid_ = true;
        this->multipliers_loaded_ = true;

        this->active_eq_cons_ = EC;
        this->active_iq_cons_ = IC;
        this->active_eq_lmults_ = EM;
        this->active_iq_lmults_ = IM;
    }

    PSIOPT::ConvergenceFlags psipot_call_impl(JetJobModes mode);

    PSIOPT::ConvergenceFlags phase_call_impl(JetJobModes mode);

  public:
    void transcribe(bool showstats, bool showfuns);

    void transcribe() { this->transcribe(false, false); }

    void test_partitions(int i, int j, int n);

    void jet_initialize() {
        this->set_num_partitions(1, 1);
        this->optimizer_->set_print_level(10);
        this->print_mesh_info_ = false;

        this->transcribe();
    }
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

    PSIOPT::ConvergenceFlags solve() { return phase_call_impl(JetJobModes::Solve); }
    PSIOPT::ConvergenceFlags optimize() { return phase_call_impl(JetJobModes::Optimize); }
    PSIOPT::ConvergenceFlags solve_optimize() {
        return phase_call_impl(JetJobModes::SolveOptimize);
    }
    PSIOPT::ConvergenceFlags solve_optimize_solve() {
        return phase_call_impl(JetJobModes::SolveOptimizeSolve);
    }
    PSIOPT::ConvergenceFlags optimize_solve() {
        return phase_call_impl(JetJobModes::OptimizeSolve);
    }

    /////////////////////////////////////////////////////////////////

    virtual void get_meshinfo_integrator(Eigen::VectorXd &tsnd, Eigen::MatrixXd &mesh_errors,
                                         Eigen::MatrixXd &mesh_dist) const = 0;
    virtual void get_meshinfo_deboor(Eigen::VectorXd &tsnd, Eigen::MatrixXd &mesh_errors,
                                     Eigen::MatrixXd &mesh_dist) const = 0;
    virtual Eigen::VectorXd calc_global_error() const = 0;

    virtual void init_mesh_refinement() {
        this->mesh_converged_ = false;
        this->mesh_iters_.resize(0);
    }

    virtual bool check_mesh();
    virtual void update_mesh();

    virtual Eigen::VectorXd calc_switches();

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
