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
            throw std::invalid_argument(
                "Phase: null phase pointer (construct via RuntimeODE::phase())");
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
        auto idx = resolve(var_names);
        if (idx.size() != values.size()) {
            throw std::invalid_argument(fmt::format(
                "Phase::add_boundary_value: resolved {} indices from names but values has size {}",
                idx.size(), values.size()));
        }
        return phase_->add_boundary_value(flag, idx, values, scale);
    }

    int add_boundary_value(PhaseRegionFlags flag, const std::string &var_name, double value,
                           ScaleType scale = ScaleModes::AUTO) {
        int i = resolve_single(var_name, "add_boundary_value");
        Eigen::VectorXi idx(1);
        idx[0] = i;
        Eigen::VectorXd v(1);
        v[0] = value;
        return phase_->add_boundary_value(flag, idx, v, scale);
    }

    int add_lu_var_bound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                         double upper, double scale = 1.0) {
        return phase_->add_lu_var_bound(flag, resolve_single(var_name, "add_lu_var_bound"), lower,
                                        upper, scale);
    }

    int add_lower_var_bound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                            double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_var_bound(flag, resolve_single(var_name, "add_lower_var_bound"),
                                           lower, scale, scale_t);
    }

    int add_upper_var_bound(PhaseRegionFlags flag, const std::string &var_name, double upper,
                            double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_var_bound(flag, resolve_single(var_name, "add_upper_var_bound"),
                                           upper, scale, scale_t);
    }

    int add_value_objective(PhaseRegionFlags flag, const std::string &var_name, double scale,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_value_objective(flag, resolve_single(var_name, "add_value_objective"),
                                           scale, scale_t);
    }

    int add_value_lock(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                       ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_value_lock(flag, resolve(var_names), scale);
    }

    int add_periodicity_con(std::initializer_list<std::string> var_names,
                            ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_periodicity_con(resolve(var_names), scale);
    }

    int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      std::initializer_list<std::string> var_names,
                      ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_equal_con(flag, func, resolve(var_names), scale);
    }

    int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                        std::initializer_list<std::string> var_names,
                        ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_inequal_con(flag, func, resolve(var_names), scale);
    }

    int add_state_objective(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                            std::initializer_list<std::string> var_names,
                            ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_state_objective(flag, func, resolve(var_names), scale);
    }

    int add_integral_objective(GenericFunction<-1, 1> func,
                               std::initializer_list<std::string> var_names,
                               ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_objective(func, resolve(var_names), scale);
    }

    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          std::initializer_list<std::string> var_names, double lower, double upper,
                          double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_func_bound(flag, func, resolve(var_names), lower, upper, scale,
                                         scale_t);
    }

    int add_lu_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                          double lower, double upper, double scale = 1.0,
                          ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_norm_bound(flag, resolve(var_names), lower, upper, scale, scale_t);
    }

    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             std::initializer_list<std::string> var_names, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_func_bound(flag, func, resolve(var_names), lower, scale, scale_t);
    }

    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             std::initializer_list<std::string> var_names, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_func_bound(flag, func, resolve(var_names), upper, scale, scale_t);
    }

    int add_lu_squared_norm_bound(PhaseRegionFlags flag,
                                  std::initializer_list<std::string> var_names, double lower,
                                  double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_squared_norm_bound(flag, resolve(var_names), lower, upper, scale,
                                                 scale_t);
    }

    int add_lower_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                             double lower, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_norm_bound(flag, resolve(var_names), lower, scale, scale_t);
    }

    int add_upper_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                             double upper, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_norm_bound(flag, resolve(var_names), upper, scale, scale_t);
    }

    int add_lower_squared_norm_bound(PhaseRegionFlags flag,
                                     std::initializer_list<std::string> var_names, double lower,
                                     double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_squared_norm_bound(flag, resolve(var_names), lower, scale,
                                                    scale_t);
    }

    int add_upper_squared_norm_bound(PhaseRegionFlags flag,
                                     std::initializer_list<std::string> var_names, double upper,
                                     double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_squared_norm_bound(flag, resolve(var_names), upper, scale,
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

    void set_num_partitions(int nparts) { phase_->setNumPartitions(nparts); }
    void set_num_partitions(int nparts, int qp_threads) {
        phase_->setNumPartitions(nparts, qp_threads);
    }

    void set_units(const Eigen::VectorXd &units) { phase_->set_units(units); }
    void set_units(std::initializer_list<std::pair<std::string, double>> unit_vals) {
        phase_->set_units(registry_.make_units(unit_vals));
    }

    void set_traj(const std::vector<Eigen::VectorXd> &traj) { phase_->set_traj(traj); }
    void set_traj(const std::vector<Eigen::VectorXd> &traj, int ndef) {
        phase_->set_traj(traj, ndef);
    }

    void set_static_params(const Eigen::VectorXd &parm) { phase_->set_static_params(parm); }
    void set_static_params(const Eigen::VectorXd &parm, const Eigen::VectorXd &units) {
        phase_->set_static_params(parm, units);
    }

    // ── Solve ───────────────────────────────────────────────────────────

    PSIOPT::ConvergenceFlags solve() { return phase_->solve(); }
    PSIOPT::ConvergenceFlags optimize() { return phase_->optimize(); }
    PSIOPT::ConvergenceFlags solve_optimize() { return phase_->solve_optimize(); }
    PSIOPT::ConvergenceFlags optimize_solve() { return phase_->optimize_solve(); }

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
    PSIOPT &optimizer() { return *phase_->optimizer; }
    const VarRegistry &registry() const { return registry_; }

  private:
    std::shared_ptr<ODEPhaseBase> phase_;
    VarRegistry registry_;

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
};

} // namespace tycho
