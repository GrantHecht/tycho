// =============================================================================
// Tycho — Builder API: ODEBuilder
//
// Fluent builder for constructing RuntimeODE objects.  Accepts dynamics via
// a lambda (receiving a proxy with XVar/UVar/TVar/PVar/XVec/UVec/PVec accessors) or
// a pre-built VectorFunction expression.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/vf/type_erasure/generic_function.h"
#include "tycho/detail/optimal_control/core/ode_arguments.h"
#include "tycho/detail/optimal_control/builder/runtime_ode.h"
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Tycho {

/// Proxy passed to the ODEBuilder::define() lambda.  Provides simplified
/// XVar/UVar/TVar/PVar/XVec/UVec/PVec accessors for building VectorFunction
/// expressions from the XtUP input layout.
class ODEArgsProxy {
  public:
    ODEArgsProxy(int xv, int uv, int pv) : args_(xv, uv, pv) {}

    /// i-th state variable (0-based within X).
    auto XVar(int i) const {
        if (args_.XVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::XVar: no state variables declared (xvars=0)");
        if (i < 0 || i >= args_.XVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::XVar: index {} out of range [0, {})", i, args_.XVars()));
        return args_.segment(0, args_.XVars()).coeff(i);
    }

    /// Full state vector segment.
    auto XVec() const {
        if (args_.XVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::XVec: no state variables declared (xvars=0)");
        return args_.segment(0, args_.XVars());
    }

    /// Sub-segment of the state vector (0-based within X).
    auto XVec(int start, int count) const {
        if (count <= 0)
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::XVec: count must be positive (got {})", count));
        if (start < 0 || start + count > args_.XVars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::XVec: range [{}, {}) out of X range [0, {})", start, start + count,
                args_.XVars()));
        return args_.segment(start, count);
    }

    /// Time variable.
    auto TVar() const { return args_.coeff(args_.TVar()); }

    /// i-th control variable (0-based within U).
    auto UVar(int i) const {
        if (args_.UVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::UVar: no control variables declared (uvars=0)");
        if (i < 0 || i >= args_.UVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::UVar: index {} out of range [0, {})", i, args_.UVars()));
        return args_.segment(args_.XtVars(), args_.UVars()).coeff(i);
    }

    /// Full control vector segment.
    auto UVec() const {
        if (args_.UVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::UVec: no control variables declared (uvars=0)");
        return args_.segment(args_.XtVars(), args_.UVars());
    }

    /// Sub-segment of the control vector (0-based within U).
    auto UVec(int start, int count) const {
        if (count <= 0)
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::UVec: count must be positive (got {})", count));
        if (start < 0 || start + count > args_.UVars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::UVec: range [{}, {}) out of U range [0, {})", start, start + count,
                args_.UVars()));
        return args_.segment(args_.XtVars() + start, count);
    }

    /// i-th parameter variable (0-based within P).
    auto PVar(int i) const {
        if (args_.PVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::PVar: no parameter variables declared (pvars=0)");
        if (i < 0 || i >= args_.PVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::PVar: index {} out of range [0, {})", i, args_.PVars()));
        return args_.segment(args_.XtUVars(), args_.PVars()).coeff(i);
    }

    /// Full parameter vector segment.
    auto PVec() const {
        if (args_.PVars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::PVec: no parameter variables declared (pvars=0)");
        return args_.segment(args_.XtUVars(), args_.PVars());
    }

    /// Sub-segment of the parameter vector (0-based within P).
    auto PVec(int start, int count) const {
        if (count <= 0)
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::PVec: count must be positive (got {})", count));
        if (start < 0 || start + count > args_.PVars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::PVec: range [{}, {}) out of P range [0, {})", start, start + count,
                args_.PVars()));
        return args_.segment(args_.XtUVars() + start, count);
    }

    /// Sub-segment of the state vector with compile-time size (0-based within X).
    template <int SZ>
    auto XVec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::XVec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.XVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::XVec<{}>: range [{}, {}) out of X range [0, {})", SZ,
                            start, start + SZ, args_.XVars()));
        return args_.segment<SZ>(start);
    }

    /// Sub-segment of the control vector with compile-time size (0-based within U).
    template <int SZ>
    auto UVec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::UVec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.UVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::UVec<{}>: range [{}, {}) out of U range [0, {})", SZ,
                            start, start + SZ, args_.UVars()));
        return args_.segment<SZ>(args_.XtVars() + start);
    }

    /// Sub-segment of the parameter vector with compile-time size (0-based within P).
    template <int SZ>
    auto PVec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::PVec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.PVars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::PVec<{}>: range [{}, {}) out of P range [0, {})", SZ,
                            start, start + SZ, args_.PVars()));
        return args_.segment<SZ>(args_.XtUVars() + start);
    }

    int xvars() const { return args_.XVars(); }
    int uvars() const { return args_.UVars(); }
    int pvars() const { return args_.PVars(); }

  private:
    ODEArguments<> args_;
};

/// Fluent builder for RuntimeODE.
///
/// Usage:
///   auto ode = ODEBuilder(3, 1)
///       .define([](const auto& args) {
///           auto v = args.XVar(2);
///           auto theta = args.UVar(0);
///           return stack(sin(theta)*v, cos(theta)*v*(-1.0), 9.81*cos(theta));
///       })
///       .var_names({{"x",0}, {"y",1}, {"v",2}, {"t",3}, {"theta",4}})
///       .build();
class ODEBuilder {
  public:
    ODEBuilder(int xvars, int uvars = 0, int pvars = 0)
        : xvars_(xvars), uvars_(uvars), pvars_(pvars) {
        if (xvars <= 0)
            throw std::invalid_argument(
                fmt::format("ODEBuilder: xvars must be positive (got {})", xvars));
        if (uvars < 0)
            throw std::invalid_argument(
                fmt::format("ODEBuilder: uvars must be non-negative (got {})", uvars));
        if (pvars < 0)
            throw std::invalid_argument(
                fmt::format("ODEBuilder: pvars must be non-negative (got {})", pvars));
    }

    ODEBuilder(const ODEBuilder &) = delete;
    ODEBuilder &operator=(const ODEBuilder &) = delete;
    ODEBuilder(ODEBuilder &&) = delete;
    ODEBuilder &operator=(ODEBuilder &&) = delete;

    /// Define ODE dynamics via a lambda.
    /// The lambda receives an ODEArgsProxy and must return a VectorFunction
    /// expression (via stack(), StackedOutputs, or a single expression for
    /// 1-state ODEs).
    template <typename F> ODEBuilder &define(F &&func) {
        if (state_ == State::Built)
            throw std::invalid_argument("ODEBuilder: cannot modify after build()");
        if (state_ != State::Initial)
            throw std::invalid_argument("ODEBuilder: dynamics already defined");
        const ODEArgsProxy proxy(xvars_, uvars_, pvars_);
        auto expr = func(proxy);
        func_ = GenericFunction<-1, -1>(expr);
        validate_func();
        state_ = State::Defined;
        return *this;
    }

    /// Define ODE from a pre-built VectorFunction expression.
    template <typename Func> ODEBuilder &from(const Func &ode_expr) {
        if (state_ == State::Built)
            throw std::invalid_argument("ODEBuilder: cannot modify after build()");
        if (state_ != State::Initial)
            throw std::invalid_argument("ODEBuilder: dynamics already defined");
        func_ = GenericFunction<-1, -1>(ode_expr);
        validate_func();
        state_ = State::Defined;
        return *this;
    }

    /// Register named variables (name -> XtUP index).
    ODEBuilder &var_names(std::initializer_list<std::pair<std::string, int>> names) {
        if (state_ == State::Built)
            throw std::invalid_argument("ODEBuilder: cannot modify after build()");
        int xtup = xvars_ + 1 + uvars_ + pvars_;
        for (const auto &[name, idx] : names) {
            if (idx < 0 || idx >= xtup)
                throw std::invalid_argument(fmt::format(
                    "ODEBuilder::var_names: index {} for '{}' out of XtUP range [0, {})", idx, name,
                    xtup));
        }
        for (const auto &p : names)
            pending_names_.push_back(p);
        return *this;
    }

    /// Register a named variable group (contiguous range).
    ODEBuilder &var_group(const std::string &name, int start, int count) {
        if (state_ == State::Built)
            throw std::invalid_argument("ODEBuilder: cannot modify after build()");
        const int xtup = xvars_ + 1 + uvars_ + pvars_;
        if (start < 0)
            throw std::invalid_argument(
                fmt::format("ODEBuilder::var_group: start must be non-negative (got {})", start));
        if (count <= 0)
            throw std::invalid_argument(
                fmt::format("ODEBuilder::var_group: count must be positive (got {})", count));
        if (start + count > xtup)
            throw std::invalid_argument(fmt::format(
                "ODEBuilder::var_group: range [{}, {}) exceeds XtUP size {}", start,
                start + count, xtup));
        pending_groups_.emplace_back(name, start, count);
        return *this;
    }

    /// Build the RuntimeODE.
    RuntimeODE build() {
        if (state_ == State::Built)
            throw std::invalid_argument(
                "ODEBuilder: build() already called; create a new ODEBuilder");
        if (state_ == State::Initial) {
            throw std::invalid_argument(
                "ODEBuilder: no dynamics defined (call define() or from())");
        }

        RuntimeODE ode(func_, xvars_, uvars_, pvars_);

        if (!pending_names_.empty() || !pending_groups_.empty()) {
            for (const auto &[name, idx] : pending_names_)
                ode.var_names({{name, idx}});
            for (const auto &[name, start, count] : pending_groups_)
                ode.var_group(name, start, count);
        }

        state_ = State::Built;
        return ode;
    }

  private:
    int xvars_;
    int uvars_;
    int pvars_;
    GenericFunction<-1, -1> func_;
    enum class State { Initial, Defined, Built };
    State state_ = State::Initial;
    std::vector<std::pair<std::string, int>> pending_names_;
    std::vector<std::tuple<std::string, int, int>> pending_groups_;

    void validate_func() const {
        int expected_ir = xvars_ + 1 + uvars_ + pvars_;
        if (func_.IRows() != expected_ir)
            throw std::invalid_argument(
                fmt::format("ODEBuilder: function input size {} does not match XtUP size {} "
                            "(xv={}, uv={}, pv={})",
                            func_.IRows(), expected_ir, xvars_, uvars_, pvars_));
        if (func_.ORows() != xvars_)
            throw std::invalid_argument(fmt::format(
                "ODEBuilder: function output size {} does not match XV={}", func_.ORows(), xvars_));
    }
};

} // namespace Tycho
