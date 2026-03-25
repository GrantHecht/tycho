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

#include "tycho/detail/InterfaceTypes.h"
#include "tycho/detail/ODEPhaseBase.h"
#include "tycho/detail/OptimalControlFlags.h"
#include "tycho/detail/PSIOPT.h"
#include "tycho/detail/var_registry.h"
#include <fmt/format.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace Tycho {

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
        if (registry_.xvars() != phase_->XVars() || registry_.uvars() != phase_->UVars() ||
            registry_.pvars() != phase_->PVars()) {
            throw std::invalid_argument(fmt::format(
                "Phase: registry dimensions ({},{},{}) do not match phase dimensions ({},{},{})",
                registry_.xvars(), registry_.uvars(), registry_.pvars(), phase_->XVars(),
                phase_->UVars(), phase_->PVars()));
        }
    }

    // ── Named-variable constraint overloads ─────────────────────────────

    int addBoundaryValue(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                         const Eigen::VectorXd &values, ScaleType scale = ScaleModes::AUTO) {
        return phase_->addBoundaryValue(flag, resolve(var_names), values, scale);
    }

    int addBoundaryValue(PhaseRegionFlags flag, const std::string &var_name, double value,
                         ScaleType scale = ScaleModes::AUTO) {
        auto idx = resolve(var_name);
        if (idx.size() != 1) {
            throw std::invalid_argument(
                fmt::format("Phase::addBoundaryValue (scalar): '{}' maps to {} indices, expected 1",
                            var_name, idx.size()));
        }
        Eigen::VectorXd v(1);
        v[0] = value;
        return phase_->addBoundaryValue(flag, idx, v, scale);
    }

    int addLUVarBound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                      double upper, double scale = 1.0) {
        auto idx = resolve(var_name);
        if (idx.size() != 1) {
            throw std::invalid_argument(fmt::format(
                "Phase::addLUVarBound: '{}' maps to {} indices, expected 1", var_name, idx.size()));
        }
        return phase_->addLUVarBound(flag, idx[0], lower, upper, scale);
    }

    int addLowerVarBound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                         double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addLowerVarBound(flag, resolve(var_name), lower, scale, scale_t);
    }

    int addUpperVarBound(PhaseRegionFlags flag, const std::string &var_name, double upper,
                         double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addUpperVarBound(flag, resolve(var_name), upper, scale, scale_t);
    }

    int addValueObjective(PhaseRegionFlags flag, const std::string &var_name, double scale,
                          ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addValueObjective(flag, resolve(var_name), scale, scale_t);
    }

    int addValueLock(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                     ScaleType scale = ScaleModes::AUTO) {
        return phase_->addValueLock(flag, resolve(var_names), scale);
    }

    int addPeriodicityCon(std::initializer_list<std::string> var_names,
                          ScaleType scale = ScaleModes::AUTO) {
        return phase_->addPeriodicityCon(resolve(var_names), scale);
    }

    int addEqualCon(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                    std::initializer_list<std::string> var_names,
                    ScaleType scale = ScaleModes::AUTO) {
        return phase_->addEqualCon(flag, func, resolve(var_names), scale);
    }

    int addInequalCon(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      std::initializer_list<std::string> var_names,
                      ScaleType scale = ScaleModes::AUTO) {
        return phase_->addInequalCon(flag, func, resolve(var_names), scale);
    }

    int addStateObjective(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          std::initializer_list<std::string> var_names,
                          ScaleType scale = ScaleModes::AUTO) {
        return phase_->addStateObjective(flag, func, resolve(var_names), scale);
    }

    int addIntegralObjective(GenericFunction<-1, 1> func,
                             std::initializer_list<std::string> var_names,
                             ScaleType scale = ScaleModes::AUTO) {
        return phase_->addIntegralObjective(func, resolve(var_names), scale);
    }

    int addLUFuncBound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                       std::initializer_list<std::string> var_names, double lower, double upper,
                       double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addLUFuncBound(flag, func, resolve(var_names), lower, upper, scale, scale_t);
    }

    int addLUNormBound(PhaseRegionFlags flag, std::initializer_list<std::string> var_names,
                       double lower, double upper, double scale = 1.0,
                       ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addLUNormBound(flag, resolve(var_names), lower, upper, scale, scale_t);
    }

    // ── Index-based overloads (direct passthrough) ──────────────────────

    int addBoundaryValue(PhaseRegionFlags flag, const Eigen::VectorXi &indices,
                         const Eigen::VectorXd &values, ScaleType scale = ScaleModes::AUTO) {
        return phase_->addBoundaryValue(flag, indices, values, scale);
    }

    int addBoundaryValue(PhaseRegionFlags flag, int index, double value,
                         ScaleType scale = ScaleModes::AUTO) {
        return phase_->addBoundaryValue(flag, index, value, scale);
    }

    int addLUVarBound(PhaseRegionFlags flag, int var, double lower, double upper,
                      double scale = 1.0) {
        return phase_->addLUVarBound(flag, var, lower, upper, scale);
    }

    int addEqualCon(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                    const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->addEqualCon(flag, func, VarIndexType(vars), scale);
    }

    int addInequalCon(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                      const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->addInequalCon(flag, func, VarIndexType(vars), scale);
    }

    int addStateObjective(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                          const Eigen::VectorXi &vars, ScaleType scale = ScaleModes::AUTO) {
        return phase_->addStateObjective(flag, func, VarIndexType(vars), scale);
    }

    int addIntegralObjective(GenericFunction<-1, 1> func, const Eigen::VectorXi &vars,
                             ScaleType scale = ScaleModes::AUTO) {
        return phase_->addIntegralObjective(func, VarIndexType(vars), scale);
    }

    int addValueObjective(PhaseRegionFlags flag, int var, double scale,
                          ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addValueObjective(flag, var, scale, scale_t);
    }

    // ── Delta-time / delta-var objectives ───────────────────────────────

    int addDeltaTimeObjective(double scale, ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addDeltaTimeObjective(scale, scale_t);
    }

    int addDeltaVarObjective(const std::string &var_name, double scale,
                             ScaleType scale_t = ScaleModes::AUTO) {
        auto idx = resolve(var_name);
        if (idx.size() != 1) {
            throw std::invalid_argument(
                fmt::format("Phase::addDeltaVarObjective: '{}' maps to {} indices, expected 1",
                            var_name, idx.size()));
        }
        return phase_->addDeltaVarObjective(idx, scale, scale_t);
    }

    int addDeltaTimeEqualCon(double value, double scale = 1.0,
                             ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addDeltaTimeEqualCon(value, scale, scale_t);
    }

    int addLowerDeltaTimeBound(double lower, double scale = 1.0,
                               ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addLowerDeltaTimeBound(lower, scale, scale_t);
    }

    int addUpperDeltaTimeBound(double upper, double scale = 1.0,
                               ScaleType scale_t = ScaleModes::AUTO) {
        return phase_->addUpperDeltaTimeBound(upper, scale, scale_t);
    }

    // ── Phase configuration ─────────────────────────────────────────────

    void setAutoScaling(bool enable) { phase_->setAutoScaling(enable); }
    void setAdaptiveMesh(bool enable) { phase_->setAdaptiveMesh(enable); }
    void setControlMode(ControlModes mode) { phase_->setControlMode(mode); }
    void setMeshTol(double tol) { phase_->setMeshTol(tol); }
    void setMaxMeshIters(int iters) { phase_->setMaxMeshIters(iters); }
    void setMeshErrorEstimator(MeshErrorEstimators est) { phase_->setMeshErrorEstimator(est); }
    void setMeshErrorCriteria(MeshErrorAggregation crit) { phase_->setMeshErrorCriteria(crit); }
    void setMeshIncFactor(double factor) { phase_->setMeshIncFactor(factor); }
    void setMeshRedFactor(double factor) { phase_->setMeshRedFactor(factor); }
    void setMeshErrFactor(double factor) { phase_->setMeshErrFactor(factor); }

    void setNumPartitions(int nparts) { phase_->setNumPartitions(nparts); }
    void setNumPartitions(int nparts, int qp_threads) {
        phase_->setNumPartitions(nparts, qp_threads);
    }

    void setUnits(const Eigen::VectorXd &units) { phase_->setUnits(units); }
    void setUnits(std::initializer_list<std::pair<std::string, double>> unit_vals) {
        phase_->setUnits(registry_.make_units(unit_vals));
    }

    void setTraj(const std::vector<Eigen::VectorXd> &traj) { phase_->setTraj(traj); }
    void setTraj(const std::vector<Eigen::VectorXd> &traj, int ndef) {
        phase_->setTraj(traj, ndef);
    }

    void setStaticParams(const Eigen::VectorXd &parm) { phase_->setStaticParams(parm); }
    void setStaticParams(const Eigen::VectorXd &parm, const Eigen::VectorXd &units) {
        phase_->setStaticParams(parm, units);
    }

    // ── Solve ───────────────────────────────────────────────────────────

    PSIOPT::ConvergenceFlags solve() { return phase_->solve(); }
    PSIOPT::ConvergenceFlags optimize() { return phase_->optimize(); }
    PSIOPT::ConvergenceFlags solve_optimize() { return phase_->solve_optimize(); }
    PSIOPT::ConvergenceFlags optimize_solve() { return phase_->optimize_solve(); }

    // ── Results ─────────────────────────────────────────────────────────

    std::vector<Eigen::VectorXd> returnTraj() const { return phase_->returnTraj(); }
    Eigen::VectorXd returnStaticParams() const { return phase_->returnStaticParams(); }
    std::vector<Eigen::VectorXd> returnCostateTraj() const { return phase_->returnCostateTraj(); }
    std::vector<Eigen::VectorXd> returnTrajError() const { return phase_->returnTrajError(); }
    bool meshConverged() const { return phase_->MeshConverged; }

    // ── Access underlying objects ───────────────────────────────────────

    ODEPhaseBase &base() { return *phase_; }
    const ODEPhaseBase &base() const { return *phase_; }
    std::shared_ptr<ODEPhaseBase> base_ptr() { return phase_; }
    PSIOPT &optimizer() { return *phase_->optimizer; }
    const VarRegistry &registry() const { return registry_; }

  private:
    std::shared_ptr<ODEPhaseBase> phase_;
    VarRegistry registry_;

    Eigen::VectorXi resolve(const std::string &name) const { return registry_.resolve(name); }

    Eigen::VectorXi resolve(std::initializer_list<std::string> names) const {
        return registry_.resolve(names);
    }

    Eigen::VectorXi resolve(const std::vector<std::string> &names) const {
        return registry_.resolve(names);
    }
};

} // namespace Tycho
