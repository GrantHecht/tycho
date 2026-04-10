// =============================================================================
// Tycho — Builder API: Phase wrapper
//
// Wraps a shared_ptr<ODEPhaseBase> and a VarRegistry to provide named-variable
// overloads on top of the index-based phase API.  All named overloads resolve
// names to indices via the registry and then delegate to ODEPhaseBase.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/builder/var_registry.h"
#include "tycho/detail/optimal_control/core/interface_types.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/phase/ode_phase_base.h"
#include "tycho/detail/solvers/psiopt.h"
#include <fmt/format.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace tycho {

using oc::ODEPhaseBase;
using oc::ScaleType;
using oc::VarIndexType;
using vf::GenericFunction;
// Solvers types
using tycho::solvers::PSIOPT;

/// High-level Phase wrapper with named-variable overloads.
///
/// Holds a shared_ptr to the underlying ODEPhaseBase (which is actually an
/// ODEPhase<GenericODE<...>>) plus a VarRegistry.  All constraint/objective
/// methods accept string variable names that are resolved through the
/// registry before forwarding to the base phase.
class Phase {
  public:
    Phase(std::shared_ptr<ODEPhaseBase> phase, VarRegistry registry)
        : phase_(std::move(phase)), registry_(std::move(registry)) {
        if (!phase_)
            throw std::invalid_argument("Phase: null phase pointer (construct via ODE::phase())");
        if (registry_.xvars() != phase_->x_vars() || registry_.uvars() != phase_->u_vars() ||
            registry_.pvars() != phase_->p_vars()) {
            throw std::invalid_argument(fmt::format(
                "Phase: registry dimensions ({},{},{}) do not match phase dimensions ({},{},{})",
                registry_.xvars(), registry_.uvars(), registry_.pvars(), phase_->x_vars(),
                phase_->u_vars(), phase_->p_vars()));
        }
    }

    // ── Named-variable constraint overloads ─────────────────────────────

    int add_boundary_value(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                           const Eigen::VectorXd &values, ScaleType scale = ScaleModes::AUTO) {
        auto idx = resolve_for_region(flag, var_names, "add_boundary_value");
        if (idx.size() != values.size()) {
            throw std::invalid_argument(fmt::format(
                "Phase::add_boundary_value: resolved {} indices from names but values has size {}",
                idx.size(), values.size()));
        }
        return phase_->add_boundary_value(flag, idx, values, scale);
    }

    int add_boundary_value(PhaseRegionFlags flag, const std::string &var_name, double value,
                           ScaleType scale = ScaleModes::AUTO) {
        int i = resolve_for_region(flag, var_name, "add_boundary_value");
        Eigen::VectorXi idx(1);
        idx[0] = i;
        Eigen::VectorXd v(1);
        v[0] = value;
        return phase_->add_boundary_value(flag, idx, v, scale);
    }

    int add_lu_var_bound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                         double upper, double scale = 1.0) {
        return phase_->add_lu_var_bound(
            flag, resolve_for_region(flag, var_name, "add_lu_var_bound"), lower, upper, scale);
    }

    int add_lower_var_bound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                            double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_var_bound(
            flag, resolve_for_region(flag, var_name, "add_lower_var_bound"), lower, scale, scale_t);
    }

    int add_upper_var_bound(PhaseRegionFlags flag, const std::string &var_name, double upper,
                            double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_var_bound(
            flag, resolve_for_region(flag, var_name, "add_upper_var_bound"), upper, scale, scale_t);
    }

    int add_value_objective(PhaseRegionFlags flag, const std::string &var_name, double scale,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_value_objective(
            flag, resolve_for_region(flag, var_name, "add_value_objective"), scale, scale_t);
    }

    int add_value_lock(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                       ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_value_lock(flag, resolve_for_region(flag, var_names, "add_value_lock"),
                                      scale);
    }

    int add_periodicity_con(std::initializer_list<std::string> var_names,
                            ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_periodicity_con(resolve(var_names), scale);
    }

    int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      std::initializer_list<std::string> var_names,
                      ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_equal_con(flag, func,
                                     resolve_for_region(flag, var_names, "add_equal_con"), scale);
    }

    int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                        std::initializer_list<std::string> var_names,
                        ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_inequal_con(flag, func,
                                       resolve_for_region(flag, var_names, "add_inequal_con"),
                                       scale);
    }

    int add_state_objective(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                            std::initializer_list<std::string> var_names,
                            ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_state_objective(
            flag, func, resolve_for_region(flag, var_names, "add_state_objective"), scale);
    }

    int add_integral_objective(GenericFunction<-1, 1> func,
                               std::initializer_list<std::string> var_names,
                               ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_objective(func, resolve(var_names), scale);
    }

    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          std::initializer_list<std::string> var_names, double lower, double upper,
                          double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_func_bound(
            flag, func, resolve_for_region(flag, var_names, "add_lu_func_bound"), lower, upper,
            scale, scale_t);
    }

    int add_lu_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                          double lower, double upper, double scale = 1.0,
                          ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_lu_norm_bound"), lower, upper, scale,
            scale_t);
    }

    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             std::initializer_list<std::string> var_names, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_func_bound(
            flag, func, resolve_for_region(flag, var_names, "add_lower_func_bound"), lower, scale,
            scale_t);
    }

    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             std::initializer_list<std::string> var_names, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_func_bound(
            flag, func, resolve_for_region(flag, var_names, "add_upper_func_bound"), upper, scale,
            scale_t);
    }

    int add_lu_squared_norm_bound(PhaseRegionFlags flag,
                                  std::initializer_list<std::string> var_names, double lower,
                                  double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_squared_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_lu_squared_norm_bound"), lower, upper,
            scale, scale_t);
    }

    int add_lower_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                             double lower, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_lower_norm_bound"), lower, scale,
            scale_t);
    }

    int add_upper_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                             double upper, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_upper_norm_bound"), upper, scale,
            scale_t);
    }

    int add_lower_squared_norm_bound(PhaseRegionFlags flag,
                                     std::initializer_list<std::string> var_names, double lower,
                                     double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_squared_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_lower_squared_norm_bound"), lower, scale,
            scale_t);
    }

    int add_upper_squared_norm_bound(PhaseRegionFlags flag,
                                     std::initializer_list<std::string> var_names, double upper,
                                     double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_squared_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_upper_squared_norm_bound"), upper, scale,
            scale_t);
    }

    int add_integral_param_function(GenericFunction<-1, 1> func,
                                    std::initializer_list<std::string> var_names, int accum_parm,
                                    ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_param_function(func, resolve(var_names), accum_parm, scale);
    }

    // ── Index-based overloads (direct passthrough) ──────────────────────

    int add_boundary_value(PhaseRegionFlags flag, const Eigen::VectorXi &indices,
                           const Eigen::VectorXd &values, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_boundary_value(flag, indices, values, scale);
    }

    int add_boundary_value(PhaseRegionFlags flag, int index, double value,
                           ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_boundary_value(flag, index, value, scale);
    }

    int add_lu_var_bound(PhaseRegionFlags flag, int var, double lower, double upper,
                         double scale = 1.0) {
        return phase_->add_lu_var_bound(flag, var, lower, upper, scale);
    }

    int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_equal_con(flag, func, VarIndexType(vars), scale);
    }

    int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                        const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_inequal_con(flag, func, VarIndexType(vars), scale);
    }

    int add_state_objective(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                            const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_state_objective(flag, func, VarIndexType(vars), scale);
    }

    int add_integral_objective(GenericFunction<-1, 1> func, const Eigen::VectorXi &vars,
                               ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_objective(func, VarIndexType(vars), scale);
    }

    int add_value_objective(PhaseRegionFlags flag, int var, double scale,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_value_objective(flag, var, scale, scale_t);
    }

    int add_lower_var_bound(PhaseRegionFlags flag, int var, double lower, double scale = 1.0,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_var_bound(flag, var, lower, scale, scale_t);
    }

    int add_upper_var_bound(PhaseRegionFlags flag, int var, double upper, double scale = 1.0,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_var_bound(flag, var, upper, scale, scale_t);
    }

    int add_value_lock(PhaseRegionFlags flag, const Eigen::VectorXi &vars,
                       ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_value_lock(flag, VarIndexType(vars), scale);
    }

    int add_periodicity_con(const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_periodicity_con(VarIndexType(vars), scale);
    }

    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          const Eigen::VectorXi &vars, double lower, double upper,
                          double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_func_bound(flag, func, VarIndexType(vars), lower, upper, scale,
                                         scale_t);
    }

    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const Eigen::VectorXi &vars, double lower, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_func_bound(flag, func, VarIndexType(vars), lower, scale, scale_t);
    }

    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const Eigen::VectorXi &vars, double upper, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_func_bound(flag, func, VarIndexType(vars), upper, scale, scale_t);
    }

    int add_lu_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars, double lower,
                          double upper, double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_norm_bound(flag, VarIndexType(vars), lower, upper, scale, scale_t);
    }

    int add_lu_squared_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars, double lower,
                                  double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_squared_norm_bound(flag, VarIndexType(vars), lower, upper, scale,
                                                 scale_t);
    }

    int add_lower_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_norm_bound(flag, VarIndexType(vars), lower, scale, scale_t);
    }

    int add_upper_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_norm_bound(flag, VarIndexType(vars), upper, scale, scale_t);
    }

    int add_lower_squared_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars,
                                     double lower, double scale = 1.0,
                                     ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_squared_norm_bound(flag, VarIndexType(vars), lower, scale,
                                                    scale_t);
    }

    int add_upper_squared_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars,
                                     double upper, double scale = 1.0,
                                     ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_squared_norm_bound(flag, VarIndexType(vars), upper, scale,
                                                    scale_t);
    }

    int add_integral_param_function(GenericFunction<-1, 1> func, const Eigen::VectorXi &vars,
                                    int accum_parm, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_param_function(func, VarIndexType(vars), accum_parm, scale);
    }

    // ── Delta-time / delta-var objectives ───────────────────────────────

    int add_delta_time_objective(double scale, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_time_objective(scale, scale_t);
    }

    int add_delta_var_objective(const std::string &var_name, double scale,
                                ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_var_objective(resolve_single(var_name, "add_delta_var_objective"),
                                               scale, scale_t);
    }

    int add_delta_var_objective(int var, double scale, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_var_objective(var, scale, scale_t);
    }

    int add_delta_var_equal_con(const std::string &var_name, double value, double scale = 1.0,
                                ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_var_equal_con(resolve_single(var_name, "add_delta_var_equal_con"),
                                               value, scale, scale_t);
    }

    int add_delta_var_equal_con(int var, double value, double scale = 1.0,
                                ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_var_equal_con(var, value, scale, scale_t);
    }

    int add_delta_time_equal_con(double value, double scale = 1.0,
                                 ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_time_equal_con(value, scale, scale_t);
    }

    int add_lower_delta_var_bound(const std::string &var_name, double lower, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_delta_var_bound(
            resolve_single(var_name, "add_lower_delta_var_bound"), lower, scale, scale_t);
    }

    int add_lower_delta_var_bound(int var, double lower, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_delta_var_bound(var, lower, scale, scale_t);
    }

    int add_lower_delta_time_bound(double lower, double scale = 1.0,
                                   ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_delta_time_bound(lower, scale, scale_t);
    }

    int add_upper_delta_var_bound(const std::string &var_name, double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_delta_var_bound(
            resolve_single(var_name, "add_upper_delta_var_bound"), upper, scale, scale_t);
    }

    int add_upper_delta_var_bound(int var, double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_delta_var_bound(var, upper, scale, scale_t);
    }

    int add_upper_delta_time_bound(double upper, double scale = 1.0,
                                   ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_delta_time_bound(upper, scale, scale_t);
    }

    // ── Phase configuration ─────────────────────────────────────────────

    void set_auto_scaling(bool enable) { phase_->set_auto_scaling(enable); }
    void set_adaptive_mesh(bool enable) { phase_->set_adaptive_mesh(enable); }
    void set_control_mode(ControlModes mode) { phase_->set_control_mode(mode); }
    void set_mesh_tol(double tol) { phase_->set_mesh_tol(tol); }
    void set_max_mesh_iters(int iters) { phase_->set_max_mesh_iters(iters); }
    void set_mesh_error_estimator(MeshErrorEstimators est) {
        phase_->set_mesh_error_estimator(est);
    }
    void set_mesh_error_criteria(MeshErrorAggregation crit) {
        phase_->set_mesh_error_criteria(crit);
    }
    void set_mesh_inc_factor(double factor) { phase_->set_mesh_inc_factor(factor); }
    void set_mesh_red_factor(double factor) { phase_->set_mesh_red_factor(factor); }
    void set_mesh_err_factor(double factor) { phase_->set_mesh_err_factor(factor); }

    void set_num_partitions(int nparts) { phase_->set_num_partitions(nparts); }
    void set_num_partitions(int nparts, int qp_threads) {
        phase_->set_num_partitions(nparts, qp_threads);
    }

    void set_units(const Eigen::VectorXd &units) { phase_->set_units(units); }
    void set_units(std::initializer_list<std::pair<std::string, double>> unit_vals) {
        phase_->set_units(registry_.make_units(unit_vals));
    }

    void set_traj(const std::vector<Eigen::VectorXd> &traj) { phase_->set_traj(traj); }
    void set_traj(const std::vector<Eigen::VectorXd> &traj, int ndef) {
        phase_->set_traj(traj, ndef);
    }
    void set_traj(const std::vector<Eigen::VectorXd> &traj, int ndef, bool lerp_ig) {
        phase_->set_traj(traj, ndef, lerp_ig);
    }

    void set_static_params(const Eigen::VectorXd &parm) { phase_->set_static_params(parm); }
    void set_static_params(const Eigen::VectorXd &parm, const Eigen::VectorXd &units) {
        phase_->set_static_params(parm, units);
    }

    // ── Static param naming ────────────────────────────────────────────

    void set_static_param_names(
        std::initializer_list<std::pair<std::string, int>> names) {
        sp_names_.clear();
        for (const auto &[name, idx] : names)
            add_static_param_name(name, idx);
    }

    void add_static_param_name(const std::string &name, int sp_index) {
        if (name.empty())
            throw std::invalid_argument("Phase::add_static_param_name: name must not be empty");
        if (sp_names_.count(name) > 0)
            throw std::invalid_argument(
                fmt::format("Phase::add_static_param_name: '{}' already registered", name));
        Eigen::VectorXi idx(1);
        idx[0] = sp_index;
        sp_names_.emplace(name, std::move(idx));
    }

    void add_static_param_group(const std::string &name, int start, int count) {
        if (name.empty())
            throw std::invalid_argument("Phase::add_static_param_group: name must not be empty");
        if (sp_names_.count(name) > 0)
            throw std::invalid_argument(
                fmt::format("Phase::add_static_param_group: '{}' already registered", name));
        if (count <= 0)
            throw std::invalid_argument(fmt::format(
                "Phase::add_static_param_group: count must be positive (got {})", count));
        Eigen::VectorXi idx(count);
        for (int i = 0; i < count; ++i)
            idx[i] = start + i;
        sp_names_.emplace(name, std::move(idx));
    }

    // ── Trajectory refinement ──────────────────────────────────────────

    void refine_traj_manual(int ndef) { phase_->refine_traj_manual(ndef); }
    void refine_traj_manual(const Eigen::VectorXd &dbs, const Eigen::VectorXi &dpb) {
        phase_->refine_traj_manual(dbs, dpb);
    }

    // ── Variable substitution ──────────────────────────────────────────

    void sub_variable(PhaseRegionFlags region, int var, double val) {
        phase_->sub_variable(region, var, val);
    }
    void sub_variables(PhaseRegionFlags region, const Eigen::VectorXi &vars,
                       const Eigen::VectorXd &vals) {
        phase_->sub_variables(region, vars, vals);
    }

    // Named overloads
    void sub_variable(PhaseRegionFlags region, const std::string &var, double val) {
        if (region == PhaseRegionFlags::StaticParams) {
            phase_->sub_variable(region, resolve_sp_single(var, "sub_variable"), val);
        } else {
            phase_->sub_variable(region, resolve_for_region(region, var, "sub_variable"), val);
        }
    }
    void sub_variables(PhaseRegionFlags region, const std::vector<std::string> &vars,
                       const Eigen::VectorXd &vals) {
        Eigen::VectorXi idx(static_cast<int>(vars.size()));
        if (region == PhaseRegionFlags::StaticParams) {
            for (int i = 0; i < static_cast<int>(vars.size()); ++i)
                idx[i] = resolve_sp_single(vars[i], "sub_variables");
        } else {
            for (int i = 0; i < static_cast<int>(vars.size()); ++i)
                idx[i] = resolve_for_region(region, vars[i], "sub_variables");
        }
        phase_->sub_variables(region, idx, vals);
    }

    // ── Constraint/objective removal ───────────────────────────────────

    void remove_state_objective(int idx) { phase_->remove_state_objective(idx); }
    void remove_integral_objective(int idx) { phase_->remove_integral_objective(idx); }
    void remove_equal_con(int idx) { phase_->remove_equal_con(idx); }
    void remove_inequal_con(int idx) { phase_->remove_inequal_con(idx); }

    // ── Mixed variable source constraints (index-based) ────────────────

    int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      const Eigen::VectorXi &xtup_vars,
                      const Eigen::VectorXi &ode_param_vars,
                      const Eigen::VectorXi &static_param_vars,
                      ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_equal_con(flag, func, VarIndexType(xtup_vars),
                                     VarIndexType(ode_param_vars),
                                     VarIndexType(static_param_vars), scale);
    }

    int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                        const Eigen::VectorXi &xtup_vars,
                        const Eigen::VectorXi &ode_param_vars,
                        const Eigen::VectorXi &static_param_vars,
                        ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_inequal_con(flag, func, VarIndexType(xtup_vars),
                                       VarIndexType(ode_param_vars),
                                       VarIndexType(static_param_vars), scale);
    }

    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          const Eigen::VectorXi &xtup_vars,
                          const Eigen::VectorXi &ode_param_vars,
                          const Eigen::VectorXi &static_param_vars, double lower, double upper,
                          double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_func_bound(flag, func, VarIndexType(xtup_vars),
                                         VarIndexType(ode_param_vars),
                                         VarIndexType(static_param_vars), lower, upper, scale,
                                         scale_t);
    }

    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const Eigen::VectorXi &xtup_vars,
                             const Eigen::VectorXi &ode_param_vars,
                             const Eigen::VectorXi &static_param_vars, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_func_bound(flag, func, VarIndexType(xtup_vars),
                                            VarIndexType(ode_param_vars),
                                            VarIndexType(static_param_vars), lower, scale, scale_t);
    }

    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const Eigen::VectorXi &xtup_vars,
                             const Eigen::VectorXi &ode_param_vars,
                             const Eigen::VectorXi &static_param_vars, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_func_bound(flag, func, VarIndexType(xtup_vars),
                                            VarIndexType(ode_param_vars),
                                            VarIndexType(static_param_vars), upper, scale, scale_t);
    }

    // ── Mixed variable source constraints (named) ──────────────────────

    int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      const std::vector<std::string> &xtup_names,
                      const std::vector<std::string> &ode_param_names,
                      const std::vector<std::string> &static_param_names,
                      ScaleType scale = ScaleModes::AUTO) {
        auto xtup_idx = resolve_names_to_indices(xtup_names, "add_equal_con");
        auto op_idx = resolve_ode_param_names(ode_param_names, "add_equal_con");
        auto sp_idx = resolve_sp_names(static_param_names, "add_equal_con");
        return phase_->add_equal_con(flag, func, VarIndexType(xtup_idx), VarIndexType(op_idx),
                                     VarIndexType(sp_idx), scale);
    }

    int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                        const std::vector<std::string> &xtup_names,
                        const std::vector<std::string> &ode_param_names,
                        const std::vector<std::string> &static_param_names,
                        ScaleType scale = ScaleModes::AUTO) {
        auto xtup_idx = resolve_names_to_indices(xtup_names, "add_inequal_con");
        auto op_idx = resolve_ode_param_names(ode_param_names, "add_inequal_con");
        auto sp_idx = resolve_sp_names(static_param_names, "add_inequal_con");
        return phase_->add_inequal_con(flag, func, VarIndexType(xtup_idx), VarIndexType(op_idx),
                                       VarIndexType(sp_idx), scale);
    }

    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          const std::vector<std::string> &xtup_names,
                          const std::vector<std::string> &ode_param_names,
                          const std::vector<std::string> &static_param_names, double lower,
                          double upper, double scale = 1.0,
                          ScaleType scale_t = ScaleModes::AUTO) {
        auto xtup_idx = resolve_names_to_indices(xtup_names, "add_lu_func_bound");
        auto op_idx = resolve_ode_param_names(ode_param_names, "add_lu_func_bound");
        auto sp_idx = resolve_sp_names(static_param_names, "add_lu_func_bound");
        return phase_->add_lu_func_bound(flag, func, VarIndexType(xtup_idx), VarIndexType(op_idx),
                                         VarIndexType(sp_idx), lower, upper, scale, scale_t);
    }

    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const std::vector<std::string> &xtup_names,
                             const std::vector<std::string> &ode_param_names,
                             const std::vector<std::string> &static_param_names, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        auto xtup_idx = resolve_names_to_indices(xtup_names, "add_lower_func_bound");
        auto op_idx = resolve_ode_param_names(ode_param_names, "add_lower_func_bound");
        auto sp_idx = resolve_sp_names(static_param_names, "add_lower_func_bound");
        return phase_->add_lower_func_bound(flag, func, VarIndexType(xtup_idx), VarIndexType(op_idx),
                                            VarIndexType(sp_idx), lower, scale, scale_t);
    }

    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const std::vector<std::string> &xtup_names,
                             const std::vector<std::string> &ode_param_names,
                             const std::vector<std::string> &static_param_names, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        auto xtup_idx = resolve_names_to_indices(xtup_names, "add_upper_func_bound");
        auto op_idx = resolve_ode_param_names(ode_param_names, "add_upper_func_bound");
        auto sp_idx = resolve_sp_names(static_param_names, "add_upper_func_bound");
        return phase_->add_upper_func_bound(flag, func, VarIndexType(xtup_idx), VarIndexType(op_idx),
                                            VarIndexType(sp_idx), upper, scale, scale_t);
    }

    // ── Solve ───────────────────────────────────────────────────────────

    PSIOPT::ConvergenceFlags solve() { return phase_->solve(); }
    PSIOPT::ConvergenceFlags optimize() { return phase_->optimize(); }
    PSIOPT::ConvergenceFlags solve_optimize() { return phase_->solve_optimize(); }
    PSIOPT::ConvergenceFlags optimize_solve() { return phase_->optimize_solve(); }
    PSIOPT::ConvergenceFlags solve_optimize_solve() { return phase_->solve_optimize_solve(); }

    // ── Results ─────────────────────────────────────────────────────────

    std::vector<Eigen::VectorXd> return_traj() const { return phase_->return_traj(); }
    Eigen::VectorXd return_static_params() const { return phase_->return_static_params(); }
    std::vector<Eigen::VectorXd> return_costate_traj() const {
        return phase_->return_costate_traj();
    }
    std::vector<Eigen::VectorXd> return_traj_error() const { return phase_->return_traj_error(); }
    bool mesh_converged() const { return phase_->mesh_converged_; }

    // ── Access underlying objects ───────────────────────────────────────

    ODEPhaseBase &base() { return *phase_; }
    const ODEPhaseBase &base() const { return *phase_; }
    std::shared_ptr<ODEPhaseBase> base_ptr() { return phase_; }
    PSIOPT &optimizer() { return *phase_->optimizer_; }
    const VarRegistry &registry() const { return registry_; }

    // ── Name resolution (public for OCP wrapper) ────────────────────────

    /// Translate a XtUP-space index to the region-relative index that
    /// ODEPhaseBase expects.  For ODEParams, subtract the P-block offset.
    /// For StaticParams, pass through as-is (already SP-relative).
    /// For all other regions, return as-is (XtUP is what base expects).
    int to_region_index(PhaseRegionFlags region, int xtup_index) const {
        if (region == PhaseRegionFlags::ODEParams) {
            int p_offset = registry_.xvars() + 1 + registry_.uvars();
            int result = xtup_index - p_offset;
            if (result < 0 || result >= registry_.pvars()) {
                throw std::invalid_argument(fmt::format(
                    "Phase::to_region_index: XtUP index {} does not fall in the P-block "
                    "[{}, {}) for ODEParams region",
                    xtup_index, p_offset, p_offset + registry_.pvars()));
            }
            return result;
        }
        // StaticParams indices are already SP-relative (resolved via SP registry)
        // All other regions use XtUP indices directly
        return xtup_index;
    }

    /// Resolve a name and translate to region-relative index (single var).
    int resolve_for_region(PhaseRegionFlags region, const std::string &var_name,
                           const char *method) const {
        int xtup_idx = resolve_single(var_name, method);
        return to_region_index(region, xtup_idx);
    }

    /// Resolve names and translate to region-relative indices (multi var).
    Eigen::VectorXi resolve_for_region(PhaseRegionFlags region,
                                       std::initializer_list<std::string> var_names,
                                       const char *method) const {
        auto idx = resolve(var_names);
        if (region == PhaseRegionFlags::ODEParams) {
            for (int i = 0; i < idx.size(); ++i)
                idx[i] = to_region_index(region, idx[i]);
        }
        return idx;
    }

    /// Resolve a single static param name to its SP-relative index.
    int resolve_sp_single(const std::string &name, const char *method) const {
        auto idx = resolve_sp(name);
        if (idx.size() != 1)
            throw std::invalid_argument(fmt::format(
                "Phase::{}: static param '{}' maps to {} indices, expected 1", method, name,
                idx.size()));
        return idx[0];
    }

  private:
    std::shared_ptr<ODEPhaseBase> phase_;
    VarRegistry registry_;
    std::unordered_map<std::string, Eigen::VectorXi> sp_names_;

    void check_registry(const char *context) const {
        if (registry_.empty()) {
            throw std::invalid_argument(
                fmt::format("Phase::{}: no variable names registered. Use the index-based overload "
                            "or register names via ODEBuilder::var_names()",
                            context));
        }
    }

    Eigen::VectorXi resolve(const std::string &name) const {
        check_registry("resolve");
        return registry_.resolve(name);
    }

    Eigen::VectorXi resolve(std::initializer_list<std::string> names) const {
        check_registry("resolve");
        return registry_.resolve(names);
    }

    Eigen::VectorXi resolve(const std::vector<std::string> &names) const {
        check_registry("resolve");
        return registry_.resolve(names);
    }

    int resolve_single(const std::string &var_name, const char *method) const {
        auto idx = resolve(var_name);
        if (idx.size() != 1) {
            throw std::invalid_argument(fmt::format(
                "Phase::{}: '{}' maps to {} indices, expected 1", method, var_name, idx.size()));
        }
        return idx[0];
    }

    Eigen::VectorXi resolve_sp(const std::string &name) const {
        auto it = sp_names_.find(name);
        if (it == sp_names_.end())
            throw std::invalid_argument(
                fmt::format("Phase::resolve_sp: unknown static param name '{}'", name));
        return it->second;
    }

    /// Resolve XtUP var names to indices (no region translation — these are
    /// always XtUP-relative for mixed-var constraints).
    Eigen::VectorXi resolve_names_to_indices(const std::vector<std::string> &names,
                                             const char *method) const {
        check_registry(method);
        Eigen::VectorXi idx(static_cast<int>(names.size()));
        for (int i = 0; i < static_cast<int>(names.size()); ++i) {
            auto resolved = registry_.resolve(names[i]);
            if (resolved.size() != 1)
                throw std::invalid_argument(fmt::format(
                    "Phase::{}: XtUP var '{}' maps to {} indices, expected 1", method, names[i],
                    resolved.size()));
            idx[i] = resolved[0];
        }
        return idx;
    }

    /// Resolve ODE param names to P-relative indices.
    Eigen::VectorXi resolve_ode_param_names(const std::vector<std::string> &names,
                                            const char *method) const {
        check_registry(method);
        int p_offset = registry_.xvars() + 1 + registry_.uvars();
        Eigen::VectorXi idx(static_cast<int>(names.size()));
        for (int i = 0; i < static_cast<int>(names.size()); ++i) {
            auto resolved = registry_.resolve(names[i]);
            if (resolved.size() != 1)
                throw std::invalid_argument(fmt::format(
                    "Phase::{}: ODE param '{}' maps to {} indices, expected 1", method, names[i],
                    resolved.size()));
            idx[i] = resolved[0] - p_offset;
            if (idx[i] < 0 || idx[i] >= registry_.pvars())
                throw std::invalid_argument(fmt::format(
                    "Phase::{}: '{}' (XtUP index {}) is not in the P-block", method, names[i],
                    resolved[0]));
        }
        return idx;
    }

    /// Resolve static param names to SP-relative indices.
    Eigen::VectorXi resolve_sp_names(const std::vector<std::string> &names,
                                     const char *method) const {
        Eigen::VectorXi idx(static_cast<int>(names.size()));
        for (int i = 0; i < static_cast<int>(names.size()); ++i)
            idx[i] = resolve_sp_single(names[i], method);
        return idx;
    }
};

} // namespace tycho
