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

/// @ingroup optimal_control
/// @brief Builder-API optimal control phase with named-variable overloads.
///
/// Holds a shared pointer to the underlying @ref tycho::oc::ODEPhaseBase plus a
/// @ref VarRegistry. Its constraint/objective/bound methods accept string
/// variable names that are resolved through the registry before forwarding to
/// the base phase; index-based overloads forward directly. Use @ref base() for
/// any base method not wrapped here.
class Phase {
  public:
    /// @brief Construct a phase wrapper from a base phase and variable registry.
    /// @param phase     The underlying base phase (must be non-null).
    /// @param registry  The variable registry; its dimensions must match the phase.
    /// @throws std::invalid_argument if @p phase is null or the dimensions disagree.
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

    /// @brief Fix selected region variables to given values (boundary-value constraint).
    /// @param flag      The phase region (e.g. Front, Back, FrontandBack).
    /// @param var_names Named variables to fix; each name must resolve to exactly one index.
    /// @param values    Target values, one per resolved variable.
    /// @param scale     Output-scale selector (default AUTO).
    /// @return The index assigned to the boundary-value constraint.
    /// @throws std::invalid_argument if the number of resolved indices differs from
    ///         the size of @p values, or if any name is unknown.
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

    /// @brief Fix selected region variables to given values (boundary-value constraint).
    /// @param flag     The phase region (e.g. Front, Back, FrontandBack).
    /// @param var_name Name of the single variable to fix.
    /// @param value    Target value for the variable.
    /// @param scale    Output-scale selector (default AUTO).
    /// @return The index assigned to the boundary-value constraint.
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_boundary_value(PhaseRegionFlags flag, const std::string &var_name, double value,
                           ScaleType scale = ScaleModes::AUTO) {
        int i = resolve_for_region(flag, var_name, "add_boundary_value");
        Eigen::VectorXi idx(1);
        idx[0] = i;
        Eigen::VectorXd v(1);
        v[0] = value;
        return phase_->add_boundary_value(flag, idx, v, scale);
    }

    /// @brief Bound a variable below and above.
    /// @param flag     The phase region.
    /// @param var_name Name of the variable to bound.
    /// @param lower    Lower bound value.
    /// @param upper    Upper bound value.
    /// @param scale    Shared bound-constraint scale (default 1.0).
    /// @return The index assigned to the bound constraint(s).
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_lu_var_bound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                         double upper, double scale = 1.0) {
        return phase_->add_lu_var_bound(
            flag, resolve_for_region(flag, var_name, "add_lu_var_bound"), lower, upper, scale);
    }

    /// @brief Bound a variable from below.
    /// @param flag     The phase region.
    /// @param var_name Name of the variable to bound.
    /// @param lower    Lower bound value.
    /// @param scale    Lower-bound constraint scale (default 1.0).
    /// @param scale_t  Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_lower_var_bound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                            double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_var_bound(
            flag, resolve_for_region(flag, var_name, "add_lower_var_bound"), lower, scale, scale_t);
    }

    /// @brief Bound a variable from above.
    /// @param flag     The phase region.
    /// @param var_name Name of the variable to bound.
    /// @param upper    Upper bound value.
    /// @param scale    Upper-bound constraint scale (default 1.0).
    /// @param scale_t  Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_upper_var_bound(PhaseRegionFlags flag, const std::string &var_name, double upper,
                            double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_var_bound(
            flag, resolve_for_region(flag, var_name, "add_upper_var_bound"), upper, scale, scale_t);
    }

    /// @brief Add an objective equal to a scaled variable value.
    /// @param flag     The phase region.
    /// @param var_name Name of the variable whose value becomes the objective.
    /// @param scale    Objective scale (sign determines minimize vs. maximize).
    /// @param scale_t  Output-scale selector (default AUTO).
    /// @return The index assigned to the objective.
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_value_objective(PhaseRegionFlags flag, const std::string &var_name, double scale,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_value_objective(
            flag, resolve_for_region(flag, var_name, "add_value_objective"), scale, scale_t);
    }

    /// @brief Lock selected region variables to their current trajectory values.
    /// @param flag      The phase region.
    /// @param var_names Names of the variables to lock.
    /// @param scale     Output-scale selector (default AUTO).
    /// @return The index assigned to the value-lock constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_value_lock(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                       ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_value_lock(flag, resolve_for_region(flag, var_names, "add_value_lock"),
                                      scale);
    }

    /// @brief Enforce periodicity (front == back) on selected variables.
    /// @param var_names Names of the variables that must be equal at the front and back.
    /// @param scale     Output-scale selector (default AUTO).
    /// @return The index assigned to the periodicity constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_periodicity_con(std::initializer_list<std::string> var_names,
                            ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_periodicity_con(resolve(var_names), scale);
    }

    /// @brief Add an equality constraint over a phase region.
    /// @param flag      The phase region.
    /// @param func      The constraint function @f$f@f$; enforces @f$f=0@f$.
    /// @param var_names Names of the variables passed to @p func.
    /// @param scale     Output-scale selector (default AUTO).
    /// @return The index assigned to the equality constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      std::initializer_list<std::string> var_names,
                      ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_equal_con(flag, func,
                                     resolve_for_region(flag, var_names, "add_equal_con"), scale);
    }

    /// @brief Add an inequality constraint over a phase region.
    /// @param flag      The phase region.
    /// @param func      The constraint function @f$f@f$; enforces @f$f\le 0@f$.
    /// @param var_names Names of the variables passed to @p func.
    /// @param scale     Output-scale selector (default AUTO).
    /// @return The index assigned to the inequality constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                        std::initializer_list<std::string> var_names,
                        ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_inequal_con(
            flag, func, resolve_for_region(flag, var_names, "add_inequal_con"), scale);
    }

    /// @brief Add a scalar state objective over a phase region.
    /// @param flag      The phase region.
    /// @param func      The scalar objective function.
    /// @param var_names Names of the variables passed to @p func.
    /// @param scale     Output-scale selector (default AUTO).
    /// @return The index assigned to the state objective.
    /// @throws std::invalid_argument if any name is unknown.
    int add_state_objective(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                            std::initializer_list<std::string> var_names,
                            ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_state_objective(
            flag, func, resolve_for_region(flag, var_names, "add_state_objective"), scale);
    }

    /// @brief Add an integral (running-cost) objective.
    /// @param func      The scalar integrand @f$g@f$; contributes @f$\int g\,dt@f$.
    /// @param var_names Names of the variables passed to @p func.
    /// @param scale     Output-scale selector (default AUTO).
    /// @return The index assigned to the integral objective.
    /// @throws std::invalid_argument if any name is unknown.
    int add_integral_objective(GenericFunction<-1, 1> func,
                               std::initializer_list<std::string> var_names,
                               ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_objective(func, resolve(var_names), scale);
    }

    /// @brief Bound a scalar function of region variables below and above.
    /// @param flag      The phase region.
    /// @param func      The scalar function to bound.
    /// @param var_names Names of the variables passed to @p func.
    /// @param lower     Lower bound on the function value.
    /// @param upper     Upper bound on the function value.
    /// @param scale     Shared bound-constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint(s).
    /// @throws std::invalid_argument if any name is unknown.
    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          std::initializer_list<std::string> var_names, double lower, double upper,
                          double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_func_bound(flag, func,
                                         resolve_for_region(flag, var_names, "add_lu_func_bound"),
                                         lower, upper, scale, scale_t);
    }

    /// @brief Bound the Euclidean norm of selected variables below and above.
    /// @param flag      The phase region.
    /// @param var_names Names of the variables whose Euclidean norm is bounded.
    /// @param lower     Lower bound on the norm.
    /// @param upper     Upper bound on the norm.
    /// @param scale     Shared bound-constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint(s).
    /// @throws std::invalid_argument if any name is unknown.
    int add_lu_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                          double lower, double upper, double scale = 1.0,
                          ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_norm_bound(flag,
                                         resolve_for_region(flag, var_names, "add_lu_norm_bound"),
                                         lower, upper, scale, scale_t);
    }

    /// @brief Bound a scalar function of region variables from below.
    /// @param flag      The phase region.
    /// @param func      The scalar function to bound.
    /// @param var_names Names of the variables passed to @p func.
    /// @param lower     Lower bound on the function value.
    /// @param scale     Lower-bound constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             std::initializer_list<std::string> var_names, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_func_bound(
            flag, func, resolve_for_region(flag, var_names, "add_lower_func_bound"), lower, scale,
            scale_t);
    }

    /// @brief Bound a scalar function of region variables from above.
    /// @param flag      The phase region.
    /// @param func      The scalar function to bound.
    /// @param var_names Names of the variables passed to @p func.
    /// @param upper     Upper bound on the function value.
    /// @param scale     Upper-bound constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             std::initializer_list<std::string> var_names, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_func_bound(
            flag, func, resolve_for_region(flag, var_names, "add_upper_func_bound"), upper, scale,
            scale_t);
    }

    /// @brief Bound the squared Euclidean norm below and above.
    /// @param flag      The phase region.
    /// @param var_names Names of the variables whose squared norm is bounded.
    /// @param lower     Lower bound on the squared norm.
    /// @param upper     Upper bound on the squared norm.
    /// @param scale     Shared bound-constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint(s).
    /// @throws std::invalid_argument if any name is unknown.
    int add_lu_squared_norm_bound(PhaseRegionFlags flag,
                                  std::initializer_list<std::string> var_names, double lower,
                                  double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_squared_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_lu_squared_norm_bound"), lower, upper,
            scale, scale_t);
    }

    /// @brief Bound the Euclidean norm from below.
    /// @param flag      The phase region.
    /// @param var_names Names of the variables whose norm is bounded.
    /// @param lower     Lower bound on the norm.
    /// @param scale     Lower-bound constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_lower_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                             double lower, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_lower_norm_bound"), lower, scale,
            scale_t);
    }

    /// @brief Bound the Euclidean norm from above.
    /// @param flag      The phase region.
    /// @param var_names Names of the variables whose norm is bounded.
    /// @param upper     Upper bound on the norm.
    /// @param scale     Upper-bound constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_upper_norm_bound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                             double upper, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_upper_norm_bound"), upper, scale,
            scale_t);
    }

    /// @brief Bound the squared Euclidean norm from below.
    /// @param flag      The phase region.
    /// @param var_names Names of the variables whose squared norm is bounded.
    /// @param lower     Lower bound on the squared norm.
    /// @param scale     Lower-bound constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_lower_squared_norm_bound(PhaseRegionFlags flag,
                                     std::initializer_list<std::string> var_names, double lower,
                                     double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_squared_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_lower_squared_norm_bound"), lower, scale,
            scale_t);
    }

    /// @brief Bound the squared Euclidean norm from above.
    /// @param flag      The phase region.
    /// @param var_names Names of the variables whose squared norm is bounded.
    /// @param upper     Upper bound on the squared norm.
    /// @param scale     Upper-bound constraint scale (default 1.0).
    /// @param scale_t   Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if any name is unknown.
    int add_upper_squared_norm_bound(PhaseRegionFlags flag,
                                     std::initializer_list<std::string> var_names, double upper,
                                     double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_squared_norm_bound(
            flag, resolve_for_region(flag, var_names, "add_upper_squared_norm_bound"), upper, scale,
            scale_t);
    }

    /// @brief Add an integral that accumulates into a parameter.
    /// @param func       The scalar integrand @f$g@f$; its running integral accumulates into
    ///                   the ODE parameter at index @p accum_parm.
    /// @param var_names  Names of the variables passed to @p func.
    /// @param accum_parm Zero-based index of the ODE parameter that receives the integral value.
    /// @param scale      Output-scale selector (default AUTO).
    /// @return The index assigned to the integral-parameter function.
    /// @throws std::invalid_argument if any name is unknown.
    int add_integral_param_function(GenericFunction<-1, 1> func,
                                    std::initializer_list<std::string> var_names, int accum_parm,
                                    ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_param_function(func, resolve(var_names), accum_parm, scale);
    }

    // ── Index-based overloads ─────────────────────────────────────────────────

    /// @brief Fix selected region variables to given values (boundary-value constraint).
    /// @param flag    The phase region.
    /// @param indices Variable indices within the region.
    /// @param values  Target values, one per index.
    /// @param scale   Output-scale selector (default AUTO).
    /// @return The index assigned to the boundary-value constraint.
    int add_boundary_value(PhaseRegionFlags flag, const Eigen::VectorXi &indices,
                           const Eigen::VectorXd &values, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_boundary_value(flag, indices, values, scale);
    }

    /// @brief Fix selected region variables to given values (boundary-value constraint).
    /// @param flag  The phase region.
    /// @param index Variable index within the region.
    /// @param value Target value for the variable.
    /// @param scale Output-scale selector (default AUTO).
    /// @return The index assigned to the boundary-value constraint.
    int add_boundary_value(PhaseRegionFlags flag, int index, double value,
                           ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_boundary_value(flag, index, value, scale);
    }

    /// @brief Bound a variable below and above.
    /// @param flag  The phase region.
    /// @param var   Variable index within the region.
    /// @param lower Lower bound value.
    /// @param upper Upper bound value.
    /// @param scale Shared bound-constraint scale (default 1.0).
    /// @return The index assigned to the bound constraint(s).
    int add_lu_var_bound(PhaseRegionFlags flag, int var, double lower, double upper,
                         double scale = 1.0) {
        return phase_->add_lu_var_bound(flag, var, lower, upper, scale);
    }

    /// @brief Add an equality constraint over a phase region.
    /// @param flag  The phase region.
    /// @param func  The constraint function @f$f@f$; enforces @f$f=0@f$.
    /// @param vars  Variable indices (XtUP-space) passed to @p func.
    /// @param scale Output-scale selector (default AUTO).
    /// @return The index assigned to the equality constraint.
    int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_equal_con(flag, func, VarIndexType(vars), scale);
    }

    /// @brief Add an inequality constraint over a phase region.
    /// @param flag  The phase region.
    /// @param func  The constraint function @f$f@f$; enforces @f$f\le 0@f$.
    /// @param vars  Variable indices (XtUP-space) passed to @p func.
    /// @param scale Output-scale selector (default AUTO).
    /// @return The index assigned to the inequality constraint.
    int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                        const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_inequal_con(flag, func, VarIndexType(vars), scale);
    }

    /// @brief Add a scalar state objective over a phase region.
    /// @param flag  The phase region.
    /// @param func  The scalar objective function.
    /// @param vars  Variable indices (XtUP-space) passed to @p func.
    /// @param scale Output-scale selector (default AUTO).
    /// @return The index assigned to the state objective.
    int add_state_objective(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                            const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_state_objective(flag, func, VarIndexType(vars), scale);
    }

    /// @brief Add an integral (running-cost) objective.
    /// @param func  The scalar integrand @f$g@f$; contributes @f$\int g\,dt@f$.
    /// @param vars  Variable indices (XtUP-space) passed to @p func.
    /// @param scale Output-scale selector (default AUTO).
    /// @return The index assigned to the integral objective.
    int add_integral_objective(GenericFunction<-1, 1> func, const Eigen::VectorXi &vars,
                               ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_objective(func, VarIndexType(vars), scale);
    }

    /// @brief Add an objective equal to a scaled variable value.
    /// @param flag    The phase region.
    /// @param var     Variable index within the region.
    /// @param scale   Objective scale (sign determines minimize vs. maximize).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the objective.
    int add_value_objective(PhaseRegionFlags flag, int var, double scale,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_value_objective(flag, var, scale, scale_t);
    }

    /// @brief Bound a variable from below.
    /// @param flag    The phase region.
    /// @param var     Variable index within the region.
    /// @param lower   Lower bound value.
    /// @param scale   Lower-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_lower_var_bound(PhaseRegionFlags flag, int var, double lower, double scale = 1.0,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_var_bound(flag, var, lower, scale, scale_t);
    }

    /// @brief Bound a variable from above.
    /// @param flag    The phase region.
    /// @param var     Variable index within the region.
    /// @param upper   Upper bound value.
    /// @param scale   Upper-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_upper_var_bound(PhaseRegionFlags flag, int var, double upper, double scale = 1.0,
                            ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_var_bound(flag, var, upper, scale, scale_t);
    }

    /// @brief Lock selected region variables to their current trajectory values.
    /// @param flag  The phase region.
    /// @param vars  Variable indices within the region to lock.
    /// @param scale Output-scale selector (default AUTO).
    /// @return The index assigned to the value-lock constraint.
    int add_value_lock(PhaseRegionFlags flag, const Eigen::VectorXi &vars,
                       ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_value_lock(flag, VarIndexType(vars), scale);
    }

    /// @brief Enforce periodicity (front == back) on selected variables.
    /// @param vars  Variable indices (XtUP-space) that must be equal at front and back.
    /// @param scale Output-scale selector (default AUTO).
    /// @return The index assigned to the periodicity constraint.
    int add_periodicity_con(const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_periodicity_con(VarIndexType(vars), scale);
    }

    /// @brief Bound a scalar function of region variables below and above.
    /// @param flag    The phase region.
    /// @param func    The scalar function to bound.
    /// @param vars    Variable indices (XtUP-space) passed to @p func.
    /// @param lower   Lower bound on the function value.
    /// @param upper   Upper bound on the function value.
    /// @param scale   Shared bound-constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint(s).
    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          const Eigen::VectorXi &vars, double lower, double upper,
                          double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_func_bound(flag, func, VarIndexType(vars), lower, upper, scale,
                                         scale_t);
    }

    /// @brief Bound a scalar function of region variables from below.
    /// @param flag    The phase region.
    /// @param func    The scalar function to bound.
    /// @param vars    Variable indices (XtUP-space) passed to @p func.
    /// @param lower   Lower bound on the function value.
    /// @param scale   Lower-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const Eigen::VectorXi &vars, double lower, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_func_bound(flag, func, VarIndexType(vars), lower, scale, scale_t);
    }

    /// @brief Bound a scalar function of region variables from above.
    /// @param flag    The phase region.
    /// @param func    The scalar function to bound.
    /// @param vars    Variable indices (XtUP-space) passed to @p func.
    /// @param upper   Upper bound on the function value.
    /// @param scale   Upper-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const Eigen::VectorXi &vars, double upper, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_func_bound(flag, func, VarIndexType(vars), upper, scale, scale_t);
    }

    /// @brief Bound the Euclidean norm of selected variables below and above.
    /// @param flag    The phase region.
    /// @param vars    Variable indices (XtUP-space) whose norm is bounded.
    /// @param lower   Lower bound on the norm.
    /// @param upper   Upper bound on the norm.
    /// @param scale   Shared bound-constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint(s).
    int add_lu_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars, double lower,
                          double upper, double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_norm_bound(flag, VarIndexType(vars), lower, upper, scale, scale_t);
    }

    /// @brief Bound the squared Euclidean norm below and above.
    /// @param flag    The phase region.
    /// @param vars    Variable indices (XtUP-space) whose squared norm is bounded.
    /// @param lower   Lower bound on the squared norm.
    /// @param upper   Upper bound on the squared norm.
    /// @param scale   Shared bound-constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint(s).
    int add_lu_squared_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars, double lower,
                                  double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_squared_norm_bound(flag, VarIndexType(vars), lower, upper, scale,
                                                 scale_t);
    }

    /// @brief Bound the Euclidean norm from below.
    /// @param flag    The phase region.
    /// @param vars    Variable indices (XtUP-space) whose norm is bounded.
    /// @param lower   Lower bound on the norm.
    /// @param scale   Lower-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_lower_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_norm_bound(flag, VarIndexType(vars), lower, scale, scale_t);
    }

    /// @brief Bound the Euclidean norm from above.
    /// @param flag    The phase region.
    /// @param vars    Variable indices (XtUP-space) whose norm is bounded.
    /// @param upper   Upper bound on the norm.
    /// @param scale   Upper-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_upper_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_norm_bound(flag, VarIndexType(vars), upper, scale, scale_t);
    }

    /// @brief Bound the squared Euclidean norm from below.
    /// @param flag    The phase region.
    /// @param vars    Variable indices (XtUP-space) whose squared norm is bounded.
    /// @param lower   Lower bound on the squared norm.
    /// @param scale   Lower-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_lower_squared_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars,
                                     double lower, double scale = 1.0,
                                     ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_squared_norm_bound(flag, VarIndexType(vars), lower, scale,
                                                    scale_t);
    }

    /// @brief Bound the squared Euclidean norm from above.
    /// @param flag    The phase region.
    /// @param vars    Variable indices (XtUP-space) whose squared norm is bounded.
    /// @param upper   Upper bound on the squared norm.
    /// @param scale   Upper-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_upper_squared_norm_bound(PhaseRegionFlags flag, const Eigen::VectorXi &vars,
                                     double upper, double scale = 1.0,
                                     ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_squared_norm_bound(flag, VarIndexType(vars), upper, scale,
                                                    scale_t);
    }

    /// @brief Add an integral that accumulates into a parameter.
    /// @param func       The scalar integrand @f$g@f$; its running integral accumulates into
    ///                   the ODE parameter at index @p accum_parm.
    /// @param vars       Variable indices (XtUP-space) passed to @p func.
    /// @param accum_parm Zero-based index of the ODE parameter that receives the integral value.
    /// @param scale      Output-scale selector (default AUTO).
    /// @return The index assigned to the integral-parameter function.
    int add_integral_param_function(GenericFunction<-1, 1> func, const Eigen::VectorXi &vars,
                                    int accum_parm, ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_integral_param_function(func, VarIndexType(vars), accum_parm, scale);
    }

    // ── Delta-time / delta-variable objectives and constraints ───────────────

    /// @brief Add an objective on the total elapsed time.
    /// @param scale   Objective scale (sign determines minimize vs. maximize).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the objective.
    int add_delta_time_objective(double scale, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_time_objective(scale, scale_t);
    }

    /// @brief Add an objective on the front-to-back change of a variable.
    /// @param var_name Name of the variable (must resolve to a single index).
    /// @param scale    Objective scale (sign determines minimize vs. maximize).
    /// @param scale_t  Output-scale selector (default AUTO).
    /// @return The index assigned to the objective.
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_delta_var_objective(const std::string &var_name, double scale,
                                ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_var_objective(resolve_single(var_name, "add_delta_var_objective"),
                                               scale, scale_t);
    }

    /// @brief Add an objective on the front-to-back change of a variable.
    /// @param var     Variable index (XtUP-space).
    /// @param scale   Objective scale (sign determines minimize vs. maximize).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the objective.
    int add_delta_var_objective(int var, double scale, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_var_objective(var, scale, scale_t);
    }

    /// @brief Constrain the front-to-back change of a variable to a value.
    /// @param var_name Name of the variable (must resolve to a single index).
    /// @param value    Target change (back minus front).
    /// @param scale    Constraint scale (default 1.0).
    /// @param scale_t  Output-scale selector (default AUTO).
    /// @return The index assigned to the equality constraint.
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_delta_var_equal_con(const std::string &var_name, double value, double scale = 1.0,
                                ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_var_equal_con(resolve_single(var_name, "add_delta_var_equal_con"),
                                               value, scale, scale_t);
    }

    /// @brief Constrain the front-to-back change of a variable to a value.
    /// @param var     Variable index (XtUP-space).
    /// @param value   Target change (back minus front).
    /// @param scale   Constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the equality constraint.
    int add_delta_var_equal_con(int var, double value, double scale = 1.0,
                                ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_var_equal_con(var, value, scale, scale_t);
    }

    /// @brief Constrain the total elapsed time to a value.
    /// @param value   Target elapsed time (back minus front time coordinate).
    /// @param scale   Constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the equality constraint.
    int add_delta_time_equal_con(double value, double scale = 1.0,
                                 ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_delta_time_equal_con(value, scale, scale_t);
    }

    /// @brief Bound the front-to-back change of a variable from below.
    /// @param var_name Name of the variable (must resolve to a single index).
    /// @param lower    Lower bound on the change (back minus front).
    /// @param scale    Lower-bound constraint scale (default 1.0).
    /// @param scale_t  Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_lower_delta_var_bound(const std::string &var_name, double lower, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_delta_var_bound(
            resolve_single(var_name, "add_lower_delta_var_bound"), lower, scale, scale_t);
    }

    /// @brief Bound the front-to-back change of a variable from below.
    /// @param var     Variable index (XtUP-space).
    /// @param lower   Lower bound on the change (back minus front).
    /// @param scale   Lower-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_lower_delta_var_bound(int var, double lower, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_delta_var_bound(var, lower, scale, scale_t);
    }

    /// @brief Bound the elapsed time from below.
    /// @param lower   Lower bound on the elapsed time.
    /// @param scale   Lower-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_lower_delta_time_bound(double lower, double scale = 1.0,
                                   ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_delta_time_bound(lower, scale, scale_t);
    }

    /// @brief Bound the front-to-back change of a variable from above.
    /// @param var_name Name of the variable (must resolve to a single index).
    /// @param upper    Upper bound on the change (back minus front).
    /// @param scale    Upper-bound constraint scale (default 1.0).
    /// @param scale_t  Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if @p var_name is unknown or maps to multiple indices.
    int add_upper_delta_var_bound(const std::string &var_name, double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_delta_var_bound(
            resolve_single(var_name, "add_upper_delta_var_bound"), upper, scale, scale_t);
    }

    /// @brief Bound the front-to-back change of a variable from above.
    /// @param var     Variable index (XtUP-space).
    /// @param upper   Upper bound on the change (back minus front).
    /// @param scale   Upper-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_upper_delta_var_bound(int var, double upper, double scale = 1.0,
                                  ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_delta_var_bound(var, upper, scale, scale_t);
    }

    /// @brief Bound the elapsed time from above.
    /// @param upper   Upper bound on the elapsed time.
    /// @param scale   Upper-bound constraint scale (default 1.0).
    /// @param scale_t Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_upper_delta_time_bound(double upper, double scale = 1.0,
                                   ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_delta_time_bound(upper, scale, scale_t);
    }

    // ── Configuration setters ─────────────────────────────────────────────────

    /// @brief Enable or disable automatic scaling.
    /// @param enable  Whether to enable automatic variable/equation scaling.
    void set_auto_scaling(bool enable) { phase_->set_auto_scaling(enable); }
    /// @brief Enable or disable adaptive mesh refinement.
    /// @param enable  Whether to enable adaptive mesh refinement.
    void set_adaptive_mesh(bool enable) { phase_->set_adaptive_mesh(enable); }
    /// @brief Set the control representation mode.
    /// @param mode  The control mode (e.g. spline order or block-constant).
    void set_control_mode(ControlModes mode) { phase_->set_control_mode(mode); }
    /// @brief Set the target mesh-error tolerance.
    /// @param tol  Tolerance; its absolute value is used.
    void set_mesh_tol(double tol) { phase_->set_mesh_tol(tol); }
    /// @brief Set the maximum number of mesh-refinement iterations.
    /// @param iters  Iteration cap; its absolute value is used.
    void set_max_mesh_iters(int iters) { phase_->set_max_mesh_iters(iters); }
    /// @brief Set the mesh-error estimator.
    /// @param est  The estimator mode.
    void set_mesh_error_estimator(MeshErrorEstimators est) {
        phase_->set_mesh_error_estimator(est);
    }
    /// @brief Set the mesh-error convergence aggregation.
    /// @param crit  The aggregation mode used to measure convergence across the mesh.
    void set_mesh_error_criteria(MeshErrorAggregation crit) {
        phase_->set_mesh_error_criteria(crit);
    }
    /// @brief Set the segment-count increase factor.
    /// @param factor  Multiplicative factor applied when increasing segment counts; its absolute
    ///                value is used.
    void set_mesh_inc_factor(double factor) { phase_->set_mesh_inc_factor(factor); }
    /// @brief Set the segment-count reduction factor.
    /// @param factor  Multiplicative factor applied when reducing segment counts; its absolute
    ///                value is used.
    void set_mesh_red_factor(double factor) { phase_->set_mesh_red_factor(factor); }
    /// @brief Set the error-overshoot factor that triggers refinement.
    /// @param factor  If the estimated error exceeds @p factor times the tolerance, refinement
    ///                is triggered; its absolute value is used.
    void set_mesh_err_factor(double factor) { phase_->set_mesh_err_factor(factor); }

    /// @brief Set the number of solver partitions.
    /// @param nparts  Number of NLP partitions for the parallel PSIOPT solver.
    void set_num_partitions(int nparts) { phase_->set_num_partitions(nparts); }
    /// @brief Set the number of solver partitions.
    /// @param nparts     Number of NLP partitions for the parallel PSIOPT solver.
    /// @param qp_threads Number of threads used per partition for linear-system solves.
    void set_num_partitions(int nparts, int qp_threads) {
        phase_->set_num_partitions(nparts, qp_threads);
    }

    /// @brief Set the jet (batched) solve job mode.
    /// @param mode  The jet job mode controlling how batched Jacobian/Hessian evaluations
    ///              are dispatched.
    void set_jet_job_mode(solvers::OptimizationProblemBase::JetJobModes mode) {
        phase_->set_jet_job_mode(mode);
    }

    /// @brief Set the variable scaling units.
    /// @param units  Per-variable scaling units for the XtUP block; size must match
    ///               the full XtUP dimension.
    void set_units(const Eigen::VectorXd &units) { phase_->set_units(units); }
    /// @brief Set the variable scaling units.
    /// @param unit_vals  Named pairs of (variable name, scale value).  Unset entries
    ///                   default to 1.0.  Names are resolved through the variable registry.
    void set_units(std::initializer_list<std::pair<std::string, double>> unit_vals) {
        phase_->set_units(registry_.make_units(unit_vals));
    }

    // ── Trajectory setters ────────────────────────────────────────────────────

    /// @brief Set the initial trajectory and mesh.
    /// @param traj  Initial trajectory; one packed state/time/control/param vector per node.
    ///              The mesh distribution is inferred from the node count.
    void set_traj(const std::vector<Eigen::VectorXd> &traj) { phase_->set_traj(traj); }
    /// @brief Set the initial trajectory and mesh.
    /// @param traj  Initial trajectory; one packed state/time/control/param vector per node.
    /// @param ndef  Number of defect intervals in a single uniform mesh bin.
    void set_traj(const std::vector<Eigen::VectorXd> &traj, int ndef) {
        phase_->set_traj(traj, ndef);
    }
    /// @brief Set the initial trajectory and mesh.
    /// @param traj     Initial trajectory; one packed state/time/control/param vector per node.
    /// @param ndef     Number of defect intervals in a single uniform mesh bin.
    /// @param lerp_ig  Whether to linearly interpolate the supplied nodes onto the new mesh.
    void set_traj(const std::vector<Eigen::VectorXd> &traj, int ndef, bool lerp_ig) {
        phase_->set_traj(traj, ndef, lerp_ig);
    }

    // ── Static-parameter setters ──────────────────────────────────────────────

    /// @brief Set the static (non-ODE) parameters.
    /// @param parm  Static-parameter values; the number of parameters is inferred from the
    ///              vector size and scaling units default to 1.
    void set_static_params(const Eigen::VectorXd &parm) { phase_->set_static_params(parm); }
    /// @brief Set the static (non-ODE) parameters.
    /// @param parm   Static-parameter values.
    /// @param units  Per-parameter scaling units; must have the same size as @p parm.
    /// @throws std::invalid_argument if @p parm and @p units differ in size.
    void set_static_params(const Eigen::VectorXd &parm, const Eigen::VectorXd &units) {
        phase_->set_static_params(parm, units);
    }

    /// @brief Replace the named static-parameter index map.
    /// @param names  Pairs of (parameter name, zero-based static-parameter index).  Any
    ///               previously registered names are cleared before registering the new ones.
    /// @throws std::invalid_argument if any name is empty or a name is registered twice.
    void set_static_param_names(std::initializer_list<std::pair<std::string, int>> names) {
        sp_names_.clear();
        for (const auto &[name, idx] : names)
            add_static_param_name(name, idx);
    }

    /// @brief Register a single named static parameter.
    /// @param name      The name to register; must not be empty or already registered.
    /// @param sp_index  Zero-based index of the static parameter this name maps to.
    /// @throws std::invalid_argument if @p name is empty or already registered.
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

    /// @brief Register a named static-parameter group as a contiguous range.
    /// @param name   The group name; must not be empty or already registered.
    /// @param start  Zero-based index of the first static parameter in the group.
    /// @param count  Number of consecutive parameters in the group (must be positive).
    /// @throws std::invalid_argument if @p name is empty or already registered, or if
    ///         @p count is not positive.
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

    // ── Mesh refinement ───────────────────────────────────────────────────────

    /// @brief Re-mesh the current trajectory onto a manual distribution.
    /// @param ndef  Number of defect intervals in a single uniform mesh bin.
    void refine_traj_manual(int ndef) { phase_->refine_traj_manual(ndef); }
    /// @brief Re-mesh the current trajectory onto a manual distribution.
    /// @param dbs  Normalized defect-bin boundary positions (size = num_bins + 1).
    /// @param dpb  Number of defect intervals per bin (size = num_bins).
    void refine_traj_manual(const Eigen::VectorXd &dbs, const Eigen::VectorXi &dpb) {
        phase_->refine_traj_manual(dbs, dpb);
    }

    // ── Variable substitution ─────────────────────────────────────────────────

    /// @brief Substitute a fixed value into a single region variable.
    /// @param region  The phase region.
    /// @param var     Variable index within the region.
    /// @param val     Value to fix the variable to throughout the trajectory.
    void sub_variable(PhaseRegionFlags region, int var, double val) {
        phase_->sub_variable(region, var, val);
    }
    /// @brief Substitute fixed values into selected region variables.
    /// @param region  The phase region.
    /// @param vars    Variable indices within the region.
    /// @param vals    Values to assign, one per index in @p vars.
    void sub_variables(PhaseRegionFlags region, const Eigen::VectorXi &vars,
                       const Eigen::VectorXd &vals) {
        phase_->sub_variables(region, vars, vals);
    }

    // Named overloads
    /// @brief Substitute a fixed value into a single region variable.
    /// @param region  The phase region.  If @c StaticParams, the name is resolved through
    ///                the static-parameter name map; otherwise through the XtUP registry.
    /// @param var     Variable name.
    /// @param val     Value to fix the variable to throughout the trajectory.
    /// @throws std::invalid_argument if @p var is unknown or maps to multiple indices.
    void sub_variable(PhaseRegionFlags region, const std::string &var, double val) {
        if (region == PhaseRegionFlags::StaticParams) {
            phase_->sub_variable(region, resolve_sp_single(var, "sub_variable"), val);
        } else {
            phase_->sub_variable(region, resolve_for_region(region, var, "sub_variable"), val);
        }
    }
    /// @brief Substitute fixed values into selected region variables.
    /// @param region  The phase region.  If @c StaticParams, names are resolved through
    ///                the static-parameter name map; otherwise through the XtUP registry.
    /// @param vars    Variable names (one per value in @p vals).
    /// @param vals    Values to assign, one per name in @p vars.
    /// @throws std::invalid_argument if any name is unknown or maps to multiple indices.
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

    // ── Remove methods ────────────────────────────────────────────────────────

    /// @brief Remove a previously added state objective by index.
    /// @param idx  State-objective index, or -1 for the most recently added.
    void remove_state_objective(int idx) { phase_->remove_state_objective(idx); }
    /// @brief Remove a previously added integral objective by index.
    /// @param idx  Integral-objective index, or -1 for the most recently added.
    void remove_integral_objective(int idx) { phase_->remove_integral_objective(idx); }
    /// @brief Remove a previously added equality constraint by index.
    /// @param idx  Equality-constraint index, or -1 for the most recently added.
    void remove_equal_con(int idx) { phase_->remove_equal_con(idx); }
    /// @brief Remove a previously added inequality constraint by index.
    /// @param idx  Inequality-constraint index, or -1 for the most recently added.
    void remove_inequal_con(int idx) { phase_->remove_inequal_con(idx); }

    // ── Mixed-variable (split-index) constraint overloads ────────────────────

    /// @brief Add an equality constraint over a phase region.
    /// @param flag              The phase region.
    /// @param func              The constraint function @f$f@f$; enforces @f$f=0@f$.
    /// @param xtup_vars         State/time/control (XtUP-space) variable indices.
    /// @param ode_param_vars    ODE-parameter variable indices (zero-based in P-block).
    /// @param static_param_vars Static-parameter variable indices (zero-based in SP-block).
    /// @param scale             Output-scale selector (default AUTO).
    /// @return The index assigned to the equality constraint.
    int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      const Eigen::VectorXi &xtup_vars, const Eigen::VectorXi &ode_param_vars,
                      const Eigen::VectorXi &static_param_vars,
                      ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_equal_con(flag, func, VarIndexType(xtup_vars),
                                     VarIndexType(ode_param_vars), VarIndexType(static_param_vars),
                                     scale);
    }

    /// @brief Add an inequality constraint over a phase region.
    /// @param flag              The phase region.
    /// @param func              The constraint function @f$f@f$; enforces @f$f\le 0@f$.
    /// @param xtup_vars         State/time/control (XtUP-space) variable indices.
    /// @param ode_param_vars    ODE-parameter variable indices (zero-based in P-block).
    /// @param static_param_vars Static-parameter variable indices (zero-based in SP-block).
    /// @param scale             Output-scale selector (default AUTO).
    /// @return The index assigned to the inequality constraint.
    int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                        const Eigen::VectorXi &xtup_vars, const Eigen::VectorXi &ode_param_vars,
                        const Eigen::VectorXi &static_param_vars,
                        ScaleType scale = ScaleModes::AUTO) {
        return phase_->add_inequal_con(flag, func, VarIndexType(xtup_vars),
                                       VarIndexType(ode_param_vars),
                                       VarIndexType(static_param_vars), scale);
    }

    /// @brief Bound a scalar function of region variables below and above.
    /// @param flag              The phase region.
    /// @param func              The scalar function to bound.
    /// @param xtup_vars         State/time/control (XtUP-space) variable indices.
    /// @param ode_param_vars    ODE-parameter variable indices (zero-based in P-block).
    /// @param static_param_vars Static-parameter variable indices (zero-based in SP-block).
    /// @param lower             Lower bound on the function value.
    /// @param upper             Upper bound on the function value.
    /// @param scale             Shared bound-constraint scale (default 1.0).
    /// @param scale_t           Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint(s).
    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          const Eigen::VectorXi &xtup_vars, const Eigen::VectorXi &ode_param_vars,
                          const Eigen::VectorXi &static_param_vars, double lower, double upper,
                          double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lu_func_bound(
            flag, func, VarIndexType(xtup_vars), VarIndexType(ode_param_vars),
            VarIndexType(static_param_vars), lower, upper, scale, scale_t);
    }

    /// @brief Bound a scalar function of region variables from below.
    /// @param flag              The phase region.
    /// @param func              The scalar function to bound.
    /// @param xtup_vars         State/time/control (XtUP-space) variable indices.
    /// @param ode_param_vars    ODE-parameter variable indices (zero-based in P-block).
    /// @param static_param_vars Static-parameter variable indices (zero-based in SP-block).
    /// @param lower             Lower bound on the function value.
    /// @param scale             Lower-bound constraint scale (default 1.0).
    /// @param scale_t           Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const Eigen::VectorXi &xtup_vars,
                             const Eigen::VectorXi &ode_param_vars,
                             const Eigen::VectorXi &static_param_vars, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_lower_func_bound(flag, func, VarIndexType(xtup_vars),
                                            VarIndexType(ode_param_vars),
                                            VarIndexType(static_param_vars), lower, scale, scale_t);
    }

    /// @brief Bound a scalar function of region variables from above.
    /// @param flag              The phase region.
    /// @param func              The scalar function to bound.
    /// @param xtup_vars         State/time/control (XtUP-space) variable indices.
    /// @param ode_param_vars    ODE-parameter variable indices (zero-based in P-block).
    /// @param static_param_vars Static-parameter variable indices (zero-based in SP-block).
    /// @param upper             Upper bound on the function value.
    /// @param scale             Upper-bound constraint scale (default 1.0).
    /// @param scale_t           Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const Eigen::VectorXi &xtup_vars,
                             const Eigen::VectorXi &ode_param_vars,
                             const Eigen::VectorXi &static_param_vars, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->add_upper_func_bound(flag, func, VarIndexType(xtup_vars),
                                            VarIndexType(ode_param_vars),
                                            VarIndexType(static_param_vars), upper, scale, scale_t);
    }

    // ── Named mixed-variable overloads ────────────────────────────────────────

    /// @brief Add an equality constraint over a phase region.
    /// @param flag               The phase region.
    /// @param func               The constraint function @f$f@f$; enforces @f$f=0@f$.
    /// @param xtup_names         Names of state/time/control (XtUP-space) variables.
    /// @param ode_param_names    Names of ODE-parameter variables (must be in the P-block).
    /// @param static_param_names Names of static-parameter variables (from the SP name map).
    /// @param scale              Output-scale selector (default AUTO).
    /// @return The index assigned to the equality constraint.
    /// @throws std::invalid_argument if any name is unknown or maps to multiple indices.
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

    /// @brief Add an inequality constraint over a phase region.
    /// @param flag               The phase region.
    /// @param func               The constraint function @f$f@f$; enforces @f$f\le 0@f$.
    /// @param xtup_names         Names of state/time/control (XtUP-space) variables.
    /// @param ode_param_names    Names of ODE-parameter variables (must be in the P-block).
    /// @param static_param_names Names of static-parameter variables (from the SP name map).
    /// @param scale              Output-scale selector (default AUTO).
    /// @return The index assigned to the inequality constraint.
    /// @throws std::invalid_argument if any name is unknown or maps to multiple indices.
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

    /// @brief Bound a scalar function of region variables below and above.
    /// @param flag               The phase region.
    /// @param func               The scalar function to bound.
    /// @param xtup_names         Names of state/time/control (XtUP-space) variables.
    /// @param ode_param_names    Names of ODE-parameter variables (must be in the P-block).
    /// @param static_param_names Names of static-parameter variables (from the SP name map).
    /// @param lower              Lower bound on the function value.
    /// @param upper              Upper bound on the function value.
    /// @param scale              Shared bound-constraint scale (default 1.0).
    /// @param scale_t            Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint(s).
    /// @throws std::invalid_argument if any name is unknown or maps to multiple indices.
    int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          const std::vector<std::string> &xtup_names,
                          const std::vector<std::string> &ode_param_names,
                          const std::vector<std::string> &static_param_names, double lower,
                          double upper, double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        auto xtup_idx = resolve_names_to_indices(xtup_names, "add_lu_func_bound");
        auto op_idx = resolve_ode_param_names(ode_param_names, "add_lu_func_bound");
        auto sp_idx = resolve_sp_names(static_param_names, "add_lu_func_bound");
        return phase_->add_lu_func_bound(flag, func, VarIndexType(xtup_idx), VarIndexType(op_idx),
                                         VarIndexType(sp_idx), lower, upper, scale, scale_t);
    }

    /// @brief Bound a scalar function of region variables from below.
    /// @param flag               The phase region.
    /// @param func               The scalar function to bound.
    /// @param xtup_names         Names of state/time/control (XtUP-space) variables.
    /// @param ode_param_names    Names of ODE-parameter variables (must be in the P-block).
    /// @param static_param_names Names of static-parameter variables (from the SP name map).
    /// @param lower              Lower bound on the function value.
    /// @param scale              Lower-bound constraint scale (default 1.0).
    /// @param scale_t            Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if any name is unknown or maps to multiple indices.
    int add_lower_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const std::vector<std::string> &xtup_names,
                             const std::vector<std::string> &ode_param_names,
                             const std::vector<std::string> &static_param_names, double lower,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        auto xtup_idx = resolve_names_to_indices(xtup_names, "add_lower_func_bound");
        auto op_idx = resolve_ode_param_names(ode_param_names, "add_lower_func_bound");
        auto sp_idx = resolve_sp_names(static_param_names, "add_lower_func_bound");
        return phase_->add_lower_func_bound(flag, func, VarIndexType(xtup_idx),
                                            VarIndexType(op_idx), VarIndexType(sp_idx), lower,
                                            scale, scale_t);
    }

    /// @brief Bound a scalar function of region variables from above.
    /// @param flag               The phase region.
    /// @param func               The scalar function to bound.
    /// @param xtup_names         Names of state/time/control (XtUP-space) variables.
    /// @param ode_param_names    Names of ODE-parameter variables (must be in the P-block).
    /// @param static_param_names Names of static-parameter variables (from the SP name map).
    /// @param upper              Upper bound on the function value.
    /// @param scale              Upper-bound constraint scale (default 1.0).
    /// @param scale_t            Output-scale selector (default AUTO).
    /// @return The index assigned to the bound constraint.
    /// @throws std::invalid_argument if any name is unknown or maps to multiple indices.
    int add_upper_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                             const std::vector<std::string> &xtup_names,
                             const std::vector<std::string> &ode_param_names,
                             const std::vector<std::string> &static_param_names, double upper,
                             double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        auto xtup_idx = resolve_names_to_indices(xtup_names, "add_upper_func_bound");
        auto op_idx = resolve_ode_param_names(ode_param_names, "add_upper_func_bound");
        auto sp_idx = resolve_sp_names(static_param_names, "add_upper_func_bound");
        return phase_->add_upper_func_bound(flag, func, VarIndexType(xtup_idx),
                                            VarIndexType(op_idx), VarIndexType(sp_idx), upper,
                                            scale, scale_t);
    }

    // ── Solve / optimize ──────────────────────────────────────────────────────

    /// @brief Solve the phase for feasibility (no optimization).
    /// @return Convergence status from PSIOPT.
    PSIOPT::ConvergenceFlags solve() { return phase_->solve(); }
    /// @brief Optimize the phase (minimize the objective subject to constraints).
    /// @return Convergence status from PSIOPT.
    PSIOPT::ConvergenceFlags optimize() { return phase_->optimize(); }
    /// @brief Solve for feasibility, then optimize.
    /// @return Convergence status from the final PSIOPT call.
    PSIOPT::ConvergenceFlags solve_optimize() { return phase_->solve_optimize(); }
    /// @brief Optimize, then solve to tighten feasibility.
    /// @return Convergence status from the final PSIOPT call.
    PSIOPT::ConvergenceFlags optimize_solve() { return phase_->optimize_solve(); }
    /// @brief Solve, optimize, then solve again.
    /// @return Convergence status from the final PSIOPT call.
    PSIOPT::ConvergenceFlags solve_optimize_solve() { return phase_->solve_optimize_solve(); }

    // ── Result accessors ──────────────────────────────────────────────────────

    /// @brief Return the current discretized trajectory.
    /// @return One packed state/time/control/param vector per trajectory node.
    std::vector<Eigen::VectorXd> return_traj() const { return phase_->return_traj(); }
    /// @brief Return the current static-parameter values.
    /// @return The static-parameter vector.
    Eigen::VectorXd return_static_params() const { return phase_->return_static_params(); }
    /// @brief Return the costate (adjoint) trajectory.
    /// @return One packed costate vector per trajectory node, recovered from the
    ///         Lagrange multipliers of the last solve or optimize call.
    std::vector<Eigen::VectorXd> return_costate_traj() const {
        return phase_->return_costate_traj();
    }
    /// @brief Return the per-node discretization-error estimate.
    /// @return One error vector per trajectory node.
    std::vector<Eigen::VectorXd> return_traj_error() const { return phase_->return_traj_error(); }
    /// @brief Whether the phase mesh has converged.
    /// @return True if the adaptive mesh-refinement loop met the tolerance on its last run.
    bool mesh_converged() const { return phase_->mesh_converged_; }

    // ── Internal accessors ────────────────────────────────────────────────────

    /// @brief Access the wrapped base phase.
    /// @return Reference to the underlying @ref tycho::oc::ODEPhaseBase.
    ODEPhaseBase &base() { return *phase_; }
    /// @brief Access the wrapped base phase (const).
    /// @return Const reference to the underlying base phase.
    const ODEPhaseBase &base() const { return *phase_; }
    /// @brief Access the wrapped base phase as a shared pointer.
    /// @return Shared pointer to the underlying base phase.
    std::shared_ptr<ODEPhaseBase> base_ptr() { return phase_; }
    /// @brief Access the underlying PSIOPT optimizer.
    /// @return Reference to the optimizer.
    PSIOPT &optimizer() { return *phase_->optimizer_; }
    /// @brief Access the variable registry.
    /// @return Const reference to the variable registry.
    const VarRegistry &registry() const { return registry_; }

    // ── Resolution helpers (public for external use / testing) ────────────────

    /// @brief Translate a XtUP-space index to the region-relative index that
    ///        ODEPhaseBase expects.
    ///
    /// For @c ODEParams, subtracts the P-block offset so the result is zero-based
    /// within the parameter block.  For @c StaticParams, passes the index through
    /// unchanged (already SP-relative).  For all other regions the XtUP index is
    /// returned as-is because ODEPhaseBase accepts XtUP-space directly.
    ///
    /// @param region      The target phase region.
    /// @param xtup_index  Variable index in XtUP-space (i.e. in the concatenated
    ///                    @c [X, t, U, P] layout).
    /// @return Region-relative index suitable for passing to ODEPhaseBase.
    /// @throws std::invalid_argument if @p region is @c ODEParams and @p xtup_index
    ///         does not fall within the P-block.
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

    /// @brief Resolve a name and translate to region-relative index (single variable).
    ///
    /// For @c StaticParams, routes through @ref resolve_sp_single so that SP names
    /// are looked up in the static-parameter name map rather than the XtUP registry.
    /// For all other regions, resolves through the XtUP registry and then calls
    /// @ref to_region_index.
    ///
    /// @param region    The target phase region.
    /// @param var_name  Variable name to resolve (single-index names only).
    /// @param method    Caller name used in error messages.
    /// @return Region-relative index.
    /// @throws std::invalid_argument if @p var_name is unknown, maps to multiple
    ///         indices, or (for ODEParams) does not fall within the P-block.
    int resolve_for_region(PhaseRegionFlags region, const std::string &var_name,
                           const char *method) const {
        if (region == PhaseRegionFlags::StaticParams) {
            int sp_idx = resolve_sp_single(var_name, method);
            return to_region_index(region, sp_idx);
        }
        int xtup_idx = resolve_single(var_name, method);
        return to_region_index(region, xtup_idx);
    }

    /// @brief Resolve names and translate to region-relative indices (multiple variables).
    ///
    /// For @c StaticParams, routes through @c resolve_sp_single per name.  For all other
    /// regions, resolves through the XtUP registry and applies @ref to_region_index
    /// to each result, keeping both single- and multi-name overloads symmetric.
    ///
    /// @param region    The target phase region.
    /// @param var_names Variable names to resolve.
    /// @param method    Caller name used in error messages.
    /// @return Integer vector of region-relative indices, one per name.
    /// @throws std::invalid_argument if any name is unknown or (for ODEParams) does
    ///         not fall within the P-block.
    Eigen::VectorXi resolve_for_region(PhaseRegionFlags region,
                                       std::initializer_list<std::string> var_names,
                                       const char *method) const {
        if (region == PhaseRegionFlags::StaticParams) {
            std::vector<std::string> as_vec(var_names.begin(), var_names.end());
            auto idx = resolve_sp_names(as_vec, method);
            for (int i = 0; i < idx.size(); ++i)
                idx[i] = to_region_index(region, idx[i]);
            return idx;
        }
        auto idx = resolve(var_names);
        for (int i = 0; i < idx.size(); ++i)
            idx[i] = to_region_index(region, idx[i]);
        return idx;
    }

    /// @brief Resolve a single static-parameter name to its SP-relative index.
    ///
    /// Looks the name up in the @c sp_names_ map and asserts that it maps to
    /// exactly one index.
    ///
    /// @param name    Static-parameter name registered via @ref add_static_param_name
    ///                or @ref add_static_param_group.
    /// @param method  Caller name used in error messages.
    /// @return Zero-based static-parameter index.
    /// @throws std::invalid_argument if @p name is unknown or maps to more than one index.
    int resolve_sp_single(const std::string &name, const char *method) const {
        auto idx = resolve_sp(name);
        if (idx.size() != 1)
            throw std::invalid_argument(
                fmt::format("Phase::{}: static param '{}' maps to {} indices, expected 1", method,
                            name, idx.size()));
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
                throw std::invalid_argument(
                    fmt::format("Phase::{}: XtUP var '{}' maps to {} indices, expected 1", method,
                                names[i], resolved.size()));
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
                throw std::invalid_argument(
                    fmt::format("Phase::{}: ODE param '{}' maps to {} indices, expected 1", method,
                                names[i], resolved.size()));
            idx[i] = resolved[0] - p_offset;
            if (idx[i] < 0 || idx[i] >= registry_.pvars())
                throw std::invalid_argument(
                    fmt::format("Phase::{}: '{}' (XtUP index {}) is not in the P-block", method,
                                names[i], resolved[0]));
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
