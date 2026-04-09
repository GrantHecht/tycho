// =============================================================================
// Tycho — Builder API: ODE
//
// Runtime-defined ODE that stores a type-erased GenericFunction plus size
// metadata.  Produces ODEPhase objects via a fully-dynamic GenericODE.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/builder/var_registry.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/phase/ode.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tycho {

using oc::GenericODE;
using vf::GenericFunction;

class Phase; // forward

/// A runtime-defined ODE wrapping a type-erased VectorFunction and optional
/// named-variable registry.  Construct via ODEBuilder, from a VectorFunction
/// expression (auto-erased), or from a pre-built GenericFunction.  Call
/// phase() to produce a Phase object for setting up constraints, objectives,
/// and solving.
class ODE {
  public:
    using DynODE = GenericODE<GenericFunction<-1, -1>, -1, -1, -1>;

    /// Construct from any VectorFunction expression and explicit sizes.
    template <typename Func>
    ODE(const Func &ode_func, int xvars, int uvars, int pvars = 0)
        : func_(ode_func), xvars_(xvars), uvars_(uvars), pvars_(pvars) {
        validate_sizes();
        validate();
    }

    /// Construct from an already type-erased GenericFunction and sizes.
    ODE(GenericFunction<-1, -1> func, int xvars, int uvars, int pvars = 0)
        : func_(std::move(func)), xvars_(xvars), uvars_(uvars), pvars_(pvars) {
        validate_sizes();
        validate();
    }

    // ── Named variable registration (fluent) ────────────────────────────

    ODE &var_names(std::initializer_list<std::pair<std::string, int>> names) {
        ensure_registry();
        for (const auto &[name, idx] : names)
            registry_->add_name(name, idx);
        return *this;
    }

    ODE &var_group(const std::string &name, int start, int count) {
        ensure_registry();
        registry_->add_group(name, start, count);
        return *this;
    }

    ODE &var_group(const std::string &name, std::initializer_list<std::string> members) {
        ensure_registry();
        registry_->add_group(name, members);
        return *this;
    }

    // ── Phase construction ──────────────────────────────────────────────

    /// Create a Phase from this ODE with a trajectory guess and segment count.
    Phase phase(TranscriptionModes mode, const std::vector<Eigen::VectorXd> &traj,
                int num_segments) const;

    // ── Convenience (delegate to VarRegistry) ───────────────────────────

    Eigen::VectorXd make_input(std::initializer_list<std::pair<std::string, double>> vals) const {
        check_registry();
        return registry_->make_input(vals);
    }

    Eigen::VectorXd make_units(std::initializer_list<std::pair<std::string, double>> vals) const {
        check_registry();
        return registry_->make_units(vals);
    }

    // ── Accessors ───────────────────────────────────────────────────────

    int xvars() const { return xvars_; }
    int uvars() const { return uvars_; }
    int pvars() const { return pvars_; }
    int xtup_size() const { return xvars_ + 1 + uvars_ + pvars_; }

    bool has_registry() const { return registry_.has_value(); }
    const VarRegistry &registry() const {
        check_registry();
        return *registry_;
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
    std::optional<VarRegistry> registry_;

    void validate_sizes() const {
        if (xvars_ <= 0)
            throw std::invalid_argument(
                fmt::format("ODE: xvars must be positive (got {})", xvars_));
        if (uvars_ < 0)
            throw std::invalid_argument(
                fmt::format("ODE: uvars must be non-negative (got {})", uvars_));
        if (pvars_ < 0)
            throw std::invalid_argument(
                fmt::format("ODE: pvars must be non-negative (got {})", pvars_));
    }

    void validate() const {
        int expected_ir = xvars_ + 1 + uvars_ + pvars_;
        if (func_.input_rows() != expected_ir) {
            throw std::invalid_argument(
                fmt::format("ODE: function input size {} does not match XtUP size {} "
                            "(xv={}, uv={}, pv={})",
                            func_.input_rows(), expected_ir, xvars_, uvars_, pvars_));
        }
        if (func_.output_rows() != xvars_) {
            throw std::invalid_argument(
                fmt::format("ODE: function output size {} does not match XV={}",
                            func_.output_rows(), xvars_));
        }
    }

    void ensure_registry() {
        if (!registry_)
            registry_.emplace(xvars_, uvars_, pvars_);
    }

    void check_registry() const {
        if (!registry_) {
            throw std::invalid_argument(
                "ODE: no variable names registered (call var_names() first)");
        }
    }
};

/// Deprecated alias — use ODE instead.
[[deprecated("Use ODE instead")]] typedef ODE RuntimeODE;

} // namespace tycho
