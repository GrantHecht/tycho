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

#include "tycho/detail/vf/scaling/io_scaled.h"
#include "tycho/detail/optimal_control/core/interface_types.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_table.h"
#include "tycho/detail/optimal_control/phase/mesh_iterate_info.h"
#include "tycho/detail/optimal_control/core/ode_sizes.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/phase/phase_indexer.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/optimization_problem_base.h"
#include "tycho/detail/solvers/psiopt.h"
#include "tycho/detail/optimal_control/core/state_function.h"
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
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/crtp_base.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_SUM;
using utils::SZ_MAX;
using utils::SZ_PROD;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::VectorExpression;
using vf::VectorFunction;
using vf::ThreadingFlags;

// Solvers types
using tycho::solvers::OptimizationProblemBase;
using tycho::solvers::NonLinearProgram;
using tycho::solvers::PSIOPT;

struct OptimalControlProblem;

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

    friend OptimalControlProblem;

  protected:
    PhaseIndexer indexer;
    bool do_transcription = true;
    bool EnableVectorization = true;

    int num_defects = 0;
    int num_stat_params = 0;
    VectorXd DefBinSpacing;
    VectorXi DefsPerBin;

    TranscriptionModes TranscriptionMode = TranscriptionModes::LGL3;
    ControlModes ControlMode = ControlModes::FirstOrderSpline;
    IntegralModes IntegralMode = IntegralModes::BaseIntegral;
    int num_tran_card_states = 2;
    double Order = 3;

    LGLInterpTable Table;

    bool TrajectoryLoaded = false;
    std::vector<VectorXd> ActiveTraj;
    VectorXd ActiveODEParams;
    VectorXd ActiveStaticParams;

    bool MultipliersLoaded = false;
    bool PostOptInfoValid = false;

    VectorXd ActiveEqLmults;
    VectorXd ActiveIqLmults;
    VectorXd ActiveEqCons;
    VectorXd ActiveIqCons;

    std::map<int, StateConstraint> user_equalities;
    std::map<int, StateConstraint> user_inequalities;
    std::map<int, StateObjective> user_state_objectives;
    std::map<int, StateObjective> user_integrands;
    std::map<int, StateObjective> user_param_integrands;

    int DynamicsFuncIndex = 0;
    int ControlFuncsIndex = -1;
    VectorXi NodeSpacingFuncIndices;
    int TranSpacingFuncIndices = 0;
    int ConstraintOrder = 0;

    //////////////////////////

    Eigen::VectorXd XtUPUnits;
    Eigen::VectorXd SPUnits;

    std::map<std::string, Eigen::VectorXi> SPidxs;

    ///////////////////////
  public:
    bool AutoScaling = false;
    bool SyncObjectiveScales = true;

    bool AdaptiveMesh = false;
    bool PrintMeshInfo = true;
    int MaxMeshIters = 10;
    int MaxSegments = 10000;
    int MinSegments = 4;

    int NumExtraSegs = 4;
    double MeshRedFactor = .5;
    double MeshIncFactor = 5.0;
    double MeshErrFactor = 10.0;
    bool ForceOneMeshIter = false;
    bool SolveOnlyFirst = true;
    bool NewError = false;
    bool DetectControlSwitches = false;

    double RelSwitchTol = .3;
    double AbsSwitchTol = 1.0e-6;

    double MeshTol = 1.0e-6;

    MeshErrorEstimators MeshErrorEstimator = MeshErrorEstimators::DEBOOR;
    MeshErrorAggregation MeshErrorCriteria = MeshErrorAggregation::MAX;
    MeshErrorAggregation MeshErrorDistributor = MeshErrorAggregation::AVG;
    PSIOPT::ConvergenceFlags MeshAbortFlag = PSIOPT::ConvergenceFlags::DIVERGING;

    bool MeshConverged = false;

    std::vector<MeshIterateInfo> MeshIters;

    void set_auto_scaling(bool autoscale) {
        this->AutoScaling = autoscale;
        this->reset_transcription();
        this->invalidate_post_opt_info();
    }

    void set_adaptive_mesh(bool amesh) { this->AdaptiveMesh = amesh; }
    void set_mesh_tol(double t) { this->MeshTol = abs(t); }
    void set_mesh_red_factor(double t) { this->MeshRedFactor = abs(t); }
    void set_mesh_inc_factor(double t) { this->MeshIncFactor = abs(t); }
    void set_mesh_err_factor(double t) { this->MeshErrFactor = abs(t); }
    void set_max_mesh_iters(int it) { this->MaxMeshIters = abs(it); }
    void set_min_segments(int it) { this->MinSegments = abs(it); }
    void set_max_segments(int it) { this->MaxSegments = abs(it); }
    void set_mesh_error_criteria(MeshErrorAggregation m) { this->MeshErrorCriteria = m; }
    void set_mesh_error_criteria(const std::string &m) {
        this->MeshErrorCriteria = strto_MeshErrorAggregation(m);
    }
    void set_mesh_error_estimator(MeshErrorEstimators m) { this->MeshErrorEstimator = m; }
    void set_mesh_error_estimator(const std::string &m) {
        this->MeshErrorEstimator = strto_MeshErrorEstimator(m);
    }
    void set_mesh_error_distributor(MeshErrorAggregation m) { this->MeshErrorDistributor = m; }
    void set_mesh_error_distributor(const std::string &m) {
        this->MeshErrorDistributor = strto_MeshErrorAggregation(m);
    }

    std::vector<MeshIterateInfo> get_mesh_iters() const { return this->MeshIters; }

    ///////////////////

  public:
    /////////////////////////////////////////////////////////////////////////////

    void enable_vectorization(bool b) { this->EnableVectorization = b; }
    void reset_transcription() { this->do_transcription = true; };

    void invalidate_post_opt_info() { this->PostOptInfoValid = false; };

    ODEPhaseBase() {}
    ODEPhaseBase(int Xv, int Uv, int Pv) {
        this->set_xvars(Xv);
        this->set_uvars(Uv);
        this->set_pvars(Pv);

        this->XtUPUnits = Eigen::VectorXd::Ones(this->XtUPVars());
    }
    virtual ~ODEPhaseBase() = default;

    //////////////////////////////////////////////////

    virtual void set_units(const Eigen::VectorXd &XtUPUnits_) = 0;

    virtual void set_control_mode(ControlModes m) {
        this->reset_transcription();
        this->invalidate_post_opt_info();
        this->ControlMode = m;
        if (this->ControlMode == ControlModes::BlockConstant) {
            this->Table.BlockedControls = true;
        } else {
            this->Table.BlockedControls = false;
        }
    }

    void set_control_mode(std::string m) { this->set_control_mode(strto_ControlMode(m)); }

    virtual void set_integral_mode(IntegralModes m) {
        this->reset_transcription();
        this->IntegralMode = m;
    }
    virtual void set_transcription_mode(TranscriptionModes m) = 0;

    void switch_transcription_mode(TranscriptionModes m, VectorXd DBS, VectorXi DPB) {
        this->set_transcription_mode(m);
        this->set_traj(this->ActiveTraj, DBS, DPB);
    }
    void switch_transcription_mode(TranscriptionModes m) {
        this->switch_transcription_mode(m, this->DefBinSpacing, this->DefsPerBin);
    }

    void switch_transcription_mode(std::string m, VectorXd DBS, VectorXi DPB) {
        this->switch_transcription_mode(strto_TranscriptionMode(m), DBS, DPB);
    }
    void switch_transcription_mode(std::string m) {
        this->switch_transcription_mode(strto_TranscriptionMode(m));
    }

    ///////////////////////////////////////////////////

    void set_static_param_vgroups(std::map<std::string, Eigen::VectorXi> spidxs) {
        this->SPidxs = spidxs;
    }
    void add_static_param_vgroups(std::map<std::string, Eigen::VectorXi> spidxs) {
        for (auto &[key, value] : spidxs) {
            this->SPidxs[key] = value;
        }
    }
    void add_static_param_vgroup(Eigen::VectorXi idx, std::string key) { this->SPidxs[key] = idx; }
    void add_static_param_vgroup(int idx, std::string key) {
        VectorXi tmp(1);
        tmp << idx;
        this->SPidxs[key] = tmp;
    }

    VectorXi get_sp_idx(std::string key) const {
        if (SPidxs.count(key) == 0) {
            throw std::invalid_argument(
                fmt::format("No StaticParam variable index group with name: {0:} exists.", key));
        }
        return this->SPidxs.at(key);
    }

    /////////////////////////////////////////////////
    template <class FuncMap>
    void remove_func_impl(FuncMap &map, int index, const std::string &funcstr) {
        this->reset_transcription();
        this->invalidate_post_opt_info();
        if (index == -1 && map.size() > 0) {
            index = map.rbegin()->first;
        }

        if (map.count(index) == 0) {
            throw std::invalid_argument(
                fmt::format("No {0:} with index {1:} exists in phase.", funcstr, index));
        }
        map.erase(index);
    }

    template <class FuncType, class FuncMap>
    int add_func_impl(FuncType func, FuncMap &map, const std::string &funcstr) {
        this->reset_transcription();
        this->invalidate_post_opt_info();
        int index = map.size() == 0 ? 0 : map.rbegin()->first + 1;
        map[index] = func;
        map[index].StorageIndex = index;
        check_function_size(map.at(index), funcstr);
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

    VectorXi get_xt_up_vars(PhaseRegionFlags reg, VarIndexType XtUPvars_t) const {

        VectorXi XtUPvars;

        /////////////////////////////////////////////////
        if (std::holds_alternative<int>(XtUPvars_t)) {
            XtUPvars.resize(1);
            XtUPvars[0] = std::get<int>(XtUPvars_t);
        } else if (std::holds_alternative<VectorXi>(XtUPvars_t)) {
            XtUPvars = std::get<VectorXi>(XtUPvars_t);
        } else if (std::holds_alternative<std::string>(XtUPvars_t)) {
            if (reg != PhaseRegionFlags::StaticParams) {
                XtUPvars = this->idx(std::get<std::string>(XtUPvars_t));
            } else {
                XtUPvars = this->get_sp_idx(std::get<std::string>(XtUPvars_t));
            }
            if (reg == PhaseRegionFlags::ODEParams) {
                // Convert to 0 based index
                for (int i = 0; i < XtUPvars.size(); i++) {
                    XtUPvars[i] -= this->XtUVars();
                }
            }
        } else if (std::holds_alternative<std::vector<std::string>>(XtUPvars_t)) {

            std::vector<VectorXi> varvec;
            int size = 0;

            auto tmpvars = std::get<std::vector<std::string>>(XtUPvars_t);

            for (auto tmpv : tmpvars) {
                if (reg != PhaseRegionFlags::StaticParams) {
                    varvec.push_back(this->idx(tmpv));
                } else {
                    varvec.push_back(this->get_sp_idx(tmpv));
                }

                size += varvec.back().size();
            }
            XtUPvars.resize(size);

            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    XtUPvars[next] = varv[i];
                    next++;
                }
            }

            if (reg == PhaseRegionFlags::ODEParams) {
                // Convert to 0 based index
                for (int i = 0; i < XtUPvars.size(); i++) {
                    XtUPvars[i] -= this->XtUVars();
                }
            }
        }
        return XtUPvars;
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
                OPvars[i] -= this->XtUVars();
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
                    OPvars[next] = varv[i] - this->XtUVars();
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
    FuncHolder make_func_impl(RegionType reg_t, FuncType fun, VarIndexType XtUPvars_t,
                            VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        PhaseRegionFlags reg = get_region(reg_t);
        FuncHolder func;

        if (std::holds_alternative<std::string>(XtUPvars_t)) {
            std::string vars = std::get<std::string>(XtUPvars_t);
            if (vars == "All" || vars == "all" || vars == "XtUP") {
                // Default case where the function has same inputs and order as ODE
                VectorXi XtUPvars;
                VectorXi OPvars;
                VectorXi SPvars;
                XtUPvars.setLinSpaced(this->XtUVars(), 0, this->XtUVars() - 1);
                if (this->PVars() > 0) {
                    OPvars.setLinSpaced(this->PVars(), 0, this->PVars() - 1);
                }
                func = FuncHolder(fun, reg, XtUPvars, OPvars, SPvars, scale_t);
                return func; // return early
            }
        }

        VectorXi XtUPvars = this->get_xt_up_vars(reg, XtUPvars_t);
        VectorXi OPvars;
        VectorXi SPvars;
        /////////////////////////////////////////////////
        if (reg != PhaseRegionFlags::ODEParams && reg != PhaseRegionFlags::StaticParams) {
            // If region is Params then the indices are held in XtUPvars_t and the others are emtpy
            OPvars = this->get_op_vars(reg, OPvars_t);
            SPvars = this->get_sp_vars(reg, SPvars_t);
            func = FuncHolder(fun, reg, XtUPvars, OPvars, SPvars, scale_t);

        } else {
            func = FuncHolder(fun, reg, XtUPvars, scale_t);
        }

        return func;
    }

    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int add_equal_con(StateConstraint con) {
        return add_func_impl(con, this->user_equalities, "Equality Constraint");
    }

    int add_equal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType XtUPvars_t,
                    VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, XtUPvars_t,
                                                                    OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_equalities, "Equality Constraint");
    }

    int add_equal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType XtUPvars_t,
                    ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, XtUPvars_t, empty,
                                                                    empty, scale_t);
        return add_func_impl(con, this->user_equalities, "Equality Constraint");
    }

    int add_boundary_value(RegionType reg, VarIndexType args,
                         const std::variant<double, VectorXd> &value, ScaleType scale_t);
    int add_delta_var_equal_con(VarIndexType var, double value, double scale, ScaleType scale_t);
    int add_delta_time_equal_con(double value, double scale, ScaleType scale_t) {
        return this->add_delta_var_equal_con(this->TVar(), value, scale, scale_t);
    }
    int add_value_lock(RegionType reg, VarIndexType args, ScaleType scale_t);
    int add_periodicity_con(VarIndexType args, ScaleType scale_t);

    /////////////////////////////////////////////////

    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int add_inequal_con(StateConstraint con) {
        return add_func_impl(con, this->user_inequalities, "Inequality Constraint");
    }
    int add_inequal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType XtUPvars_t,
                      VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, XtUPvars_t,
                                                                    OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_inequalities, "Inequality Constraint");
    }

    int add_inequal_con(RegionType reg_t, VectorFunctionalX fun, VarIndexType XtUPvars_t,
                      ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateConstraint, VectorFunctionalX>(reg_t, fun, XtUPvars_t, empty,
                                                                    empty, scale_t);
        return add_func_impl(con, this->user_inequalities, "Inequality Constraint");
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

    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
                       VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                       double upperbound, double lbscale, double ubscale, ScaleType scale_t);

    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
                       double lowerbound, double upperbound, double lbscale, double ubscale,
                       ScaleType scale_t) {

        VectorXi empty;

        return add_lu_func_bound(reg, func, XtUPvars, empty, empty, lowerbound, upperbound, lbscale,
                              ubscale, scale_t);
    }

    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
                       VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                       double upperbound, double scale, ScaleType scale_t) {
        return add_lu_func_bound(reg, func, XtUPvars, OPvars, SPvars, lowerbound, upperbound, scale,
                              scale, scale_t);
    }

    int add_lu_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
                       double lowerbound, double upperbound, double scale, ScaleType scale_t) {
        VectorXi empty;
        return add_lu_func_bound(reg, func, XtUPvars, empty, empty, lowerbound, upperbound, scale,
                              scale, scale_t);
    }

    int add_lower_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
                          VarIndexType OPvars, VarIndexType SPvars, double lowerbound,
                          double lbscale, ScaleType scale_t);

    int add_lower_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
                          double lowerbound, double lbscale, ScaleType scale_t) {
        VectorXi empty;

        return this->add_lower_func_bound(reg, func, XtUPvars, empty, empty, lowerbound, lbscale,
                                       scale_t);
    }

    int add_upper_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
                          VarIndexType OPvars, VarIndexType SPvars, double upperbound,
                          double ubscale, ScaleType scale_t);

    int add_upper_func_bound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
                          double upperbound, double ubscale, ScaleType scale_t) {
        VectorXi empty;

        return this->add_upper_func_bound(reg, func, XtUPvars, empty, empty, upperbound, ubscale,
                                       scale_t);
    }

    int add_lu_norm_bound(RegionType reg, VarIndexType XtUPvars, double lowerbound, double upperbound,
                       double lbscale, double ubscale, ScaleType scale_t);

    int add_lu_norm_bound(RegionType reg, VarIndexType XtUPvars, double lowerbound, double upperbound,
                       double scale, ScaleType scale_t) {
        return this->add_lu_norm_bound(reg, XtUPvars, lowerbound, upperbound, scale, scale, scale_t);
    }

    int add_lu_squared_norm_bound(RegionType reg, VarIndexType XtUPvars, double lowerbound,
                              double upperbound, double lbscale, double ubscale, ScaleType scale_t);

    int add_lu_squared_norm_bound(RegionType reg, VarIndexType XtUPvars, double lowerbound,
                              double upperbound, double scale, ScaleType scale_t) {
        return this->add_lu_squared_norm_bound(reg, XtUPvars, lowerbound, upperbound, scale, scale,
                                           scale_t);
    }

    int add_lower_norm_bound(RegionType reg, VarIndexType XtUPvars, double lowerbound, double lbscale,
                          ScaleType scale_t);

    int add_lower_squared_norm_bound(RegionType reg, VarIndexType XtUPvars, double lowerbound,
                                 double lbscale, ScaleType scale_t);

    int add_upper_norm_bound(RegionType reg, VarIndexType XtUPvars, double upperbound, double ubscale,
                          ScaleType scale_t);

    int add_upper_squared_norm_bound(RegionType reg, VarIndexType XtUPvars, double upperbound,
                                 double ubscale, ScaleType scale_t);
    //
    int add_lower_delta_var_bound(RegionType reg, VarIndexType var, double lowerbound, double lbscale,
                              ScaleType scale_t);
    int add_lower_delta_var_bound(VarIndexType var, double lowerbound, double lbscale,
                              ScaleType scale_t) {
        return this->add_lower_delta_var_bound(PhaseRegionFlags::FrontandBack, var, lowerbound, lbscale,
                                           scale_t);
    }
    int add_lower_delta_time_bound(double lowerbound, double lbscale, ScaleType scale_t) {
        return this->add_lower_delta_var_bound(this->TVar(), lowerbound, lbscale, scale_t);
    }
    ///
    int add_upper_delta_var_bound(RegionType reg, VarIndexType var, double upperbound, double ubscale,
                              ScaleType scale_t);
    int add_upper_delta_var_bound(VarIndexType var, double upperbound, double ubscale,
                              ScaleType scale_t) {
        return this->add_upper_delta_var_bound(PhaseRegionFlags::FrontandBack, var, upperbound, ubscale,
                                           scale_t);
    }
    int add_upper_delta_time_bound(double upperbound, double ubscale, ScaleType scale_t) {
        return this->add_upper_delta_var_bound(this->TVar(), upperbound, ubscale, scale_t);
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
        return add_func_impl(obj, this->user_state_objectives, "State Objective");
    }
    int add_state_objective(RegionType reg_t, ScalarFunctionalX fun, VarIndexType XtUPvars_t,
                          VarIndexType OPvars_t, VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(reg_t, fun, XtUPvars_t, OPvars_t,
                                                                   SPvars_t, scale_t);
        return add_func_impl(con, this->user_state_objectives, "State Objective");
    }

    int add_state_objective(RegionType reg_t, ScalarFunctionalX fun, VarIndexType XtUPvars_t,
                          ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(reg_t, fun, XtUPvars_t, empty,
                                                                   empty, scale_t);
        return add_func_impl(con, this->user_state_objectives, "State Objective");
    }
    int add_value_objective(RegionType reg, VarIndexType var, double scale, ScaleType scale_t);
    int add_delta_var_objective(VarIndexType var, double scale, ScaleType scale_t);
    int add_delta_time_objective(double scale, ScaleType scale_t) {
        return this->add_delta_var_objective(this->TVar(), scale, scale_t);
    }

    ///////////////////////////////////////////////

    ///////////////////////////////////////////////////
    int add_integral_objective(StateObjective obj) {
        return add_func_impl(obj, this->user_integrands, "Integral Objective");
    }

    int add_integral_objective(ScalarFunctionalX fun, VarIndexType XtUPvars_t, VarIndexType OPvars_t,
                             VarIndexType SPvars_t, ScaleType scale_t) {

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(
            PhaseRegionFlags::Path, fun, XtUPvars_t, OPvars_t, SPvars_t, scale_t);
        return add_func_impl(con, this->user_integrands, "Integral Objective");
    }

    int add_integral_objective(ScalarFunctionalX fun, VarIndexType XtUPvars_t, ScaleType scale_t) {

        VectorXi empty;

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(
            PhaseRegionFlags::Path, fun, XtUPvars_t, empty, empty, scale_t);
        return add_func_impl(con, this->user_integrands, "Integral Objective");
    }

    ///////////////////////////////////////////////////
    int add_integral_param_function(StateObjective con, int pv) {
        VectorXi epv(1);
        epv[0] = pv;
        int index = add_func_impl(con, this->user_param_integrands, "Integral Parameter Function");
        this->user_param_integrands[index].EXTVars = epv;
        return index;
    }

    int add_integral_param_function(ScalarFunctionalX fun, VarIndexType XtUPvars_t,
                                 VarIndexType OPvars_t, VarIndexType SPvars_t, int accum_parm,
                                 ScaleType scale_t) {

        VectorXi epv(1);
        epv[0] = accum_parm;

        auto con = make_func_impl<StateObjective, ScalarFunctionalX>(
            PhaseRegionFlags::Path, fun, XtUPvars_t, OPvars_t, SPvars_t, scale_t);
        int index = add_func_impl(con, this->user_param_integrands, "Integral Parameter Function");
        this->user_param_integrands[index].EXTVars = epv;
        return index;
    }

    int add_integral_param_function(ScalarFunctionalX fun, VarIndexType XtUPvars_t, int accum_parm,
                                 ScaleType scale_t) {
        VectorXi empty;
        return add_integral_param_function(fun, XtUPvars_t, empty, empty, accum_parm, scale_t);
    }

    /////////////////////////////////////////////////

    void remove_equal_con(int index) {
        this->remove_func_impl(this->user_equalities, index, "Equality Constraint");
    }
    void remove_inequal_con(int index) {
        this->remove_func_impl(this->user_inequalities, index, "Inequality Constraint");
    }
    void remove_state_objective(int index) {
        this->remove_func_impl(this->user_state_objectives, index, "State Objective");
    }
    void remove_integral_objective(int index) {
        this->remove_func_impl(this->user_integrands, index, "Integral Objective");
    }
    void remove_integral_param_function(int index) {
        this->remove_func_impl(this->user_param_integrands, index, "Integral Parameter Function");
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

        this->ActiveStaticParams = parm;
        this->num_stat_params = parm.size();
        this->reset_transcription();
        this->SPUnits = units;
    }
    void set_static_params(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->set_static_params(parm, units);
    }

    void add_static_params(VectorXd parm, VectorXd units) {
        if (this->num_stat_params == 0) {
            this->set_static_params(parm, units);
        } else {
            VectorXd parmstmp(this->ActiveStaticParams.size() + parm.size());
            parmstmp << this->ActiveStaticParams, parm;
            VectorXd unitstmp(this->SPUnits.size() + units.size());
            unitstmp << this->SPUnits, units;
            this->set_static_params(parmstmp, unitstmp);
        }
    }
    void add_static_params(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->add_static_params(parm, units);
    }

    void sub_static_params(VectorXd parm) {
        if (this->num_stat_params == parm.size()) {
            this->ActiveStaticParams = parm;
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

    std::vector<Eigen::VectorXd> return_traj() const { return this->ActiveTraj; }

    std::vector<Eigen::VectorXd> return_traj_range(int num, double tl, double th) {
        this->Table.load_regular_data(this->DefsPerBin.sum(), this->ActiveTraj);
        return this->Table.InterpRange(num, tl, th);
    }
    std::vector<Eigen::VectorXd> return_traj_range_nd(int num, double tl, double th) {
        this->Table.load_regular_data(this->DefsPerBin.sum(), this->ActiveTraj);
        return this->Table.NDequidist(num, tl, th);
    }
    LGLInterpTable return_traj_table() {
        LGLInterpTable tabt = this->Table;
        tabt.load_exact_data(this->ActiveTraj);
        return tabt;
    }

    Eigen::VectorXd return_static_params() const { return this->ActiveStaticParams; }

    std::vector<Eigen::VectorXd> return_u_spline_con_lmults() const {

        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(
                "No multipliers to return, a solve or optimize call must be made "
                "before returning constraint multipliers ");
        }

        if (this->ControlFuncsIndex < 0) {
            return std::vector<Eigen::VectorXd>();
        } else {
            return this->indexer.get_func_eq_multipliers(this->ControlFuncsIndex,
                                                      this->ActiveEqLmults);
        }
    }

    std::vector<Eigen::VectorXd> return_u_spline_con_vals() const {

        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(
                "No constraints to return, a solve or optimize call must be made "
                "before returning constraint values ");
        }
        if (this->ControlFuncsIndex < 0) {
            return std::vector<Eigen::VectorXd>();
        } else {
            return this->indexer.get_func_eq_multipliers(this->ControlFuncsIndex, this->ActiveEqCons);
        }
    }

    std::vector<Eigen::VectorXd> return_equal_con_lmults(int index) const {
        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(
                "No multipliers to return, a solve or optimize call must be made "
                "before returning constraint multipliers ");
        }
        int Gindex = this->user_equalities.at(index).GlobalIndex;
        return this->indexer.get_func_eq_multipliers(Gindex, this->ActiveEqLmults);
    }
    std::vector<Eigen::VectorXd> return_equal_con_vals(int index) const {
        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(
                "No constraints to return, a solve or optimize call must be made "
                "before returning constraint values ");
        }
        int Gindex = this->user_equalities.at(index).GlobalIndex;
        return this->indexer.get_func_eq_multipliers(Gindex, this->ActiveEqCons);
    }
    Eigen::VectorXd return_equal_con_scales(int index) const {
        return this->user_equalities.at(index).OutputScales;
    }

    std::vector<Eigen::VectorXd> return_inequal_con_lmults(int index) const {
        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(
                "No multipliers to return, a solve or optimize call must be made "
                "before returning constraint multipliers ");
        }
        int Gindex = this->user_inequalities.at(index).GlobalIndex;
        return this->indexer.get_func_iq_multipliers(Gindex, this->ActiveIqLmults);
    }

    std::vector<Eigen::VectorXd> return_inequal_con_vals(int index) const {
        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(
                "No constraints to return, a solve or optimize call must be made "
                "before returning constraint values ");
        }
        int Gindex = this->user_inequalities.at(index).GlobalIndex;
        return this->indexer.get_func_iq_multipliers(Gindex, this->ActiveIqCons);
    }
    Eigen::VectorXd return_inequal_con_scales(int index) const {
        return this->user_inequalities.at(index).OutputScales;
    }

    std::vector<Eigen::VectorXd> return_costate_traj() const;
    std::vector<Eigen::VectorXd> return_traj_error() const;

    Eigen::VectorXd return_integral_objective_scales(int index) const {
        return this->user_integrands.at(index).OutputScales;
    }
    Eigen::VectorXd return_integral_param_function_scales(int index) const {
        return this->user_param_integrands.at(index).OutputScales;
    }
    Eigen::VectorXd return_state_objective_scales(int index) const {
        return this->user_state_objectives.at(index).OutputScales;
    }
    Eigen::VectorXd return_ode_output_scales() const {
        VectorXd output_scales =
            XtUPUnits.head(this->XVars()).cwiseInverse() * this->XtUPUnits[this->XVars()];
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
        this->indexer =
            PhaseIndexer(this->XVars(), this->UVars(), this->PVars(), this->num_stat_params);
        this->indexer.set_dimensions(this->num_tran_card_states, this->num_defects,
                                     this->ControlMode == ControlModes::BlockConstant);
    }
    void check_functions(int pnum);

    template <class T> static void check_function_size(const T &func, std::string ftype) {
        int irows = func.Func.input_rows();
        switch (func.RegionFlag) {
        case PhaseRegionFlags::Front:
        case PhaseRegionFlags::Back:
        case PhaseRegionFlags::Path:
        case PhaseRegionFlags::Params:
        case PhaseRegionFlags::ODEParams:
        case PhaseRegionFlags::StaticParams:
        case PhaseRegionFlags::InnerPath:
        case PhaseRegionFlags::NodalPath: {
            int isize = func.XtUVars.size() + func.OPVars.size() + func.SPVars.size();
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
            int isize = func.XtUVars.size() * 2 + func.OPVars.size() + func.SPVars.size();
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

        if (this->AutoScaling) {
            auto ActiveTrajTmp = this->ActiveTraj;

            for (auto &T : ActiveTrajTmp) {
                T = T.cwiseQuotient(this->XtUPUnits);
            }

            VectorXd StaticParamsTmp;
            if (this->ActiveStaticParams.size() > 0 && this->SPUnits.size() > 0) {
                StaticParamsTmp = this->ActiveStaticParams.cwiseQuotient(this->SPUnits);
            }
            return this->indexer.make_solver_input(ActiveTrajTmp, StaticParamsTmp);

        } else {
            return this->indexer.make_solver_input(this->ActiveTraj, this->ActiveStaticParams);
        }
    }
    void collect_solver_output(const VectorXd &Vars) {
        this->indexer.collect_solver_output(Vars, this->ActiveTraj, this->ActiveStaticParams);

        if (this->AutoScaling) {
            for (auto &T : this->ActiveTraj) {
                T = T.cwiseProduct(this->XtUPUnits);
            }
            if (this->ActiveStaticParams.size() > 0 && this->SPUnits.size() > 0) {
                this->ActiveStaticParams = this->ActiveStaticParams.cwiseProduct(this->SPUnits);
            }
        }
    }
    void collect_solver_multipliers(const VectorXd &EM, const VectorXd &IM) {
        this->MultipliersLoaded = true;
        this->ActiveEqLmults = EM;
        this->ActiveIqLmults = IM;
    }

    void collect_post_opt_info(const VectorXd &EC, const VectorXd &EM, const VectorXd &IC,
                            const VectorXd &IM) {

        this->PostOptInfoValid = true;
        this->MultipliersLoaded = true;

        this->ActiveEqCons = EC;
        this->ActiveIqCons = IC;
        this->ActiveEqLmults = EM;
        this->ActiveIqLmults = IM;
    }

    PSIOPT::ConvergenceFlags psipot_call_impl(JetJobModes mode);

    PSIOPT::ConvergenceFlags phase_call_impl(JetJobModes mode);

  public:
    void transcribe(bool showstats, bool showfuns);

    void transcribe() { this->transcribe(false, false); }

    void test_partitions(int i, int j, int n);

    void jet_initialize() {
        this->setNumPartitions(1, 1);
        this->optimizer->PrintLevel = 10;
        this->PrintMeshInfo = false;

        this->transcribe();
    }
    void jet_release() {
        this->indexer = PhaseIndexer();
        this->optimizer->release();
        this->initPartitions();
        this->optimizer->PrintLevel = 0;
        this->PrintMeshInfo = true;
        this->nlp = std::shared_ptr<NonLinearProgram>();
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
        this->MeshConverged = false;
        this->MeshIters.resize(0);
    }

    virtual bool check_mesh();
    virtual void update_mesh();

    virtual Eigen::VectorXd calcSwitches();

    auto get_mesh_info(bool integ, int n) {

        Eigen::VectorXd tsnd;
        Eigen::MatrixXd mesh_errors;
        Eigen::MatrixXd mesh_dist;

        this->Table.load_exact_data(this->ActiveTraj);

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

