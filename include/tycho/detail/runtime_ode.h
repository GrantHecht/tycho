// =============================================================================
// Tycho — Builder API: RuntimeODE
//
// Runtime-defined ODE that stores a type-erased GenericFunction plus size
// metadata.  Produces ODEPhase objects via a fully-dynamic GenericODE.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/GenericFunction.h"
#include "tycho/detail/ODE.h"
#include "tycho/detail/OptimalControlFlags.h"
#include "tycho/detail/var_registry.h"
#include <string>
#include <utility>
#include <vector>

namespace Tycho {

class Phase; // forward

/// A runtime-defined ODE wrapping a type-erased VectorFunction and optional
/// named-variable registry.  Construct via ODEBuilder or directly from a
/// pre-built VectorFunction expression.
class RuntimeODE {
  public:
    using DynODE = GenericODE<GenericFunction<-1, -1>, -1, -1, -1>;

    RuntimeODE() = default;

    /// Construct from any VectorFunction expression and explicit sizes.
    template <typename Func>
    RuntimeODE(const Func &ode_func, int xvars, int uvars, int pvars = 0)
        : func_(ode_func), xvars_(xvars), uvars_(uvars), pvars_(pvars) {
        validate();
    }

    /// Construct from an already type-erased GenericFunction and sizes.
    RuntimeODE(GenericFunction<-1, -1> func, int xvars, int uvars, int pvars = 0)
        : func_(std::move(func)), xvars_(xvars), uvars_(uvars), pvars_(pvars) {
        validate();
    }

    // ── Named variable registration (fluent) ────────────────────────────

    RuntimeODE &var_names(std::initializer_list<std::pair<std::string, int>> names) {
        ensure_registry();
        for (const auto &[name, idx] : names)
            registry_.add_name(name, idx);
        return *this;
    }

    RuntimeODE &var_group(const std::string &name, int start, int count) {
        ensure_registry();
        registry_.add_group(name, start, count);
        return *this;
    }

    RuntimeODE &var_group(const std::string &name, std::initializer_list<std::string> members) {
        ensure_registry();
        registry_.add_group(name, members);
        return *this;
    }

    // ── Phase construction ──────────────────────────────────────────────

    /// Create a Phase from this ODE with a trajectory guess and segment count.
    Phase phase(TranscriptionModes mode, const std::vector<Eigen::VectorXd> &traj,
                int num_segments) const;

    // ── Convenience (delegate to VarRegistry) ───────────────────────────

    Eigen::VectorXd make_input(std::initializer_list<std::pair<std::string, double>> vals) const {
        check_registry();
        return registry_.make_input(vals);
    }

    Eigen::VectorXd make_units(std::initializer_list<std::pair<std::string, double>> vals) const {
        check_registry();
        return registry_.make_units(vals);
    }

    // ── Accessors ───────────────────────────────────────────────────────

    int xvars() const { return xvars_; }
    int uvars() const { return uvars_; }
    int pvars() const { return pvars_; }
    int xtup_size() const { return xvars_ + 1 + uvars_ + pvars_; }

    bool has_registry() const { return has_registry_; }
    const VarRegistry &registry() const {
        check_registry();
        return registry_;
    }

    /// Access the underlying GenericFunction (for advanced use).
    const GenericFunction<-1, -1> &function() const { return func_; }

    /// Build the fully-dynamic GenericODE.
    DynODE generic_ode() const { return DynODE(func_, xvars_, uvars_, pvars_); }

  private:
    GenericFunction<-1, -1> func_;
    int xvars_ = 0;
    int uvars_ = 0;
    int pvars_ = 0;
    VarRegistry registry_;
    bool has_registry_ = false;

    void validate() const {
        int expected_ir = xvars_ + 1 + uvars_ + pvars_;
        if (func_.IRows() != expected_ir) {
            throw std::invalid_argument(
                fmt::format("RuntimeODE: function input size {} does not match XtUP size {} "
                            "(xv={}, uv={}, pv={})",
                            func_.IRows(), expected_ir, xvars_, uvars_, pvars_));
        }
        if (func_.ORows() != xvars_) {
            throw std::invalid_argument(fmt::format(
                "RuntimeODE: function output size {} does not match XV={}", func_.ORows(), xvars_));
        }
    }

    void ensure_registry() {
        if (!has_registry_) {
            registry_ = VarRegistry(xvars_, uvars_, pvars_);
            has_registry_ = true;
        }
    }

    void check_registry() const {
        if (!has_registry_) {
            throw std::invalid_argument(
                "RuntimeODE: no variable names registered (call var_names() first)");
        }
    }
};

} // namespace Tycho
