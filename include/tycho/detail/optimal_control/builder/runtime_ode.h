// =============================================================================
// Tycho — Builder API: ODE
//
// Runtime-defined ODE that stores a type-erased GenericFunction plus size
// metadata.  Produces ODEPhase objects via a fully-dynamic GenericODE.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/integrators/integrator.h"
#include "tycho/detail/optimal_control/builder/var_registry.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/phase/ode.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tycho {

using oc::GenericODE;
using vf::GenericFunction;

class Phase;            // forward
class IntegratorBuilder; // forward

/// A runtime-defined ODE wrapping a type-erased VectorFunction and optional
/// named-variable registry.  Construct via ODEBuilder, from a VectorFunction
/// expression (auto-erased), or from a pre-built GenericFunction.  Call
/// phase() to produce a Phase object for setting up constraints, objectives,
/// and solving.
class ODE {
  public:
    using DynODE = GenericODE<GenericFunction<-1, -1>, -1, -1, -1>;
    using DynIntegrator = integrators::Integrator<DynODE>;

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

    // ── Integrator construction ────────────────────────────────────────

    /// Returns a fluent builder for constructing a DynIntegrator.
    /// Configure the step size (required) via .step(double), and optionally
    /// the method via .method(IVPAlg), and a control specification via
    /// .control(...). Call .build() to materialize the integrator. See
    /// IntegratorBuilder below for the full API.
    IntegratorBuilder integrator() const;

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

    // ── Internal (used by IntegratorBuilder) ────────────────────────────
    DynODE make_dyn_ode_public() const { return make_dyn_ode(); }
    void require_registry() const { check_registry(); }
    const VarRegistry *registry_ptr() const {
        return registry_ ? &*registry_ : nullptr;
    }

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
            throw std::invalid_argument(fmt::format(
                "ODE: function output size {} does not match XV={}", func_.output_rows(), xvars_));
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

    /// Build a DynODE with FlatMap populated from VarRegistry.
    DynODE make_dyn_ode() const;
};

/// Fluent builder for DynIntegrator. Obtained via ODE::integrator().
///
/// Configuration:
///   - .step(defstep)                 [required]
///   - .method(IVPAlg)                [optional, defaults to DOPRI87]
///   - exactly one of:
///       .control()                                  [no control — default]
///       .control(ulaw, Eigen::VectorXi varlocs)     [indexed]
///       .control(ulaw, {"u1","u2"})                 [named; requires registry]
///       .control(const Eigen::VectorXd& u_const)    [constant control]
///       .control(tab, Eigen::VectorXi ulocs)        [LGL table + indexed]
///       .control(tab)                               [LGL table, auto ulocs]
/// Call .build() to materialize the integrator. Calling .control(...) more
/// than once throws.
class IntegratorBuilder {
  public:
    explicit IntegratorBuilder(const ODE &ode) : ode_(&ode) {}

    IntegratorBuilder &method(IVPAlg m) {
        method_ = m;
        return *this;
    }
    IntegratorBuilder &step(double s) {
        step_ = s;
        return *this;
    }

    IntegratorBuilder &control(GenericFunction<-1, -1> ulaw, const Eigen::VectorXi &varlocs) {
        set_kind(ControlKind::IndexedLaw);
        ulaw_ = std::move(ulaw);
        varlocs_ = varlocs;
        return *this;
    }

    IntegratorBuilder &control(GenericFunction<-1, -1> ulaw,
                               std::initializer_list<std::string> names) {
        set_kind(ControlKind::NamedLaw);
        ulaw_ = std::move(ulaw);
        name_varlocs_.assign(names.begin(), names.end());
        return *this;
    }

    IntegratorBuilder &control(const Eigen::VectorXd &u_const) {
        set_kind(ControlKind::Const);
        u_const_ = u_const;
        return *this;
    }

    IntegratorBuilder &control(std::shared_ptr<oc::LGLInterpTable> tab,
                               const Eigen::VectorXi &ulocs) {
        set_kind(ControlKind::TableIndexed);
        tab_ = std::move(tab);
        varlocs_ = ulocs;
        return *this;
    }

    IntegratorBuilder &control(std::shared_ptr<oc::LGLInterpTable> tab) {
        set_kind(ControlKind::TableAuto);
        tab_ = std::move(tab);
        return *this;
    }

    /// Construct the integrator. Throws if .step() was never called.
    ODE::DynIntegrator build() const;

  private:
    enum class ControlKind { None, IndexedLaw, NamedLaw, Const, TableIndexed, TableAuto };

    const ODE *ode_;
    double step_ = -1.0;
    IVPAlg method_ = IVPAlg::DOPRI87;

    ControlKind kind_ = ControlKind::None;
    GenericFunction<-1, -1> ulaw_;
    Eigen::VectorXi varlocs_;
    std::vector<std::string> name_varlocs_;
    Eigen::VectorXd u_const_;
    std::shared_ptr<oc::LGLInterpTable> tab_;

    void set_kind(ControlKind k) {
        if (kind_ != ControlKind::None) {
            throw std::logic_error(
                "IntegratorBuilder: .control(...) already configured; call at most once");
        }
        kind_ = k;
    }
};

} // namespace tycho
