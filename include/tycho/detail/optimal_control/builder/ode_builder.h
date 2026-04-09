// =============================================================================
// Tycho — Builder API: ODEBuilder
//
// Fluent builder for constructing ODE objects.  Accepts dynamics via
// a lambda (receiving a proxy with x_var/u_var/t_var/p_var/x_vec/u_vec/p_vec accessors) or
// a pre-built VectorFunction expression.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/builder/runtime_ode.h"
#include "tycho/detail/optimal_control/core/ode_arguments.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tycho {

using oc::ODEArguments;
using vf::GenericFunction;

/// Proxy passed to the ODEBuilder::define() lambda.  Provides simplified
/// x_var/u_var/t_var/p_var/x_vec/u_vec/p_vec accessors for building VectorFunction
/// expressions from the XtUP input layout.
class ODEArgsProxy {
  public:
    ODEArgsProxy(int xv, int uv, int pv) : args_(xv, uv, pv) {}

    /// i-th state variable (0-based within X).
    auto x_var(int i) const {
        if (args_.x_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::x_var: no state variables declared (xvars=0)");
        if (i < 0 || i >= args_.x_vars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::x_var: index {} out of range [0, {})", i, args_.x_vars()));
        return args_.segment(0, args_.x_vars()).coeff(i);
    }

    /// Full state vector segment.
    auto x_vec() const {
        if (args_.x_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::x_vec: no state variables declared (xvars=0)");
        return args_.segment(0, args_.x_vars());
    }

    /// Sub-segment of the state vector (0-based within X).
    auto x_vec(int start, int count) const {
        if (count <= 0)
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::x_vec: count must be positive (got {})", count));
        if (start < 0 || start + count > args_.x_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::x_vec: range [{}, {}) out of X range [0, {})", start,
                            start + count, args_.x_vars()));
        return args_.segment(start, count);
    }

    /// Time variable.
    auto t_var() const { return args_.coeff(args_.t_var()); }

    /// i-th control variable (0-based within U).
    auto u_var(int i) const {
        if (args_.u_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::u_var: no control variables declared (uvars=0)");
        if (i < 0 || i >= args_.u_vars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::u_var: index {} out of range [0, {})", i, args_.u_vars()));
        return args_.segment(args_.xt_vars(), args_.u_vars()).coeff(i);
    }

    /// Full control vector segment.
    auto u_vec() const {
        if (args_.u_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::u_vec: no control variables declared (uvars=0)");
        return args_.segment(args_.xt_vars(), args_.u_vars());
    }

    /// Sub-segment of the control vector (0-based within U).
    auto u_vec(int start, int count) const {
        if (count <= 0)
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::u_vec: count must be positive (got {})", count));
        if (start < 0 || start + count > args_.u_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::u_vec: range [{}, {}) out of U range [0, {})", start,
                            start + count, args_.u_vars()));
        return args_.segment(args_.xt_vars() + start, count);
    }

    /// i-th parameter variable (0-based within P).
    auto p_var(int i) const {
        if (args_.p_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::p_var: no parameter variables declared (pvars=0)");
        if (i < 0 || i >= args_.p_vars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::p_var: index {} out of range [0, {})", i, args_.p_vars()));
        return args_.segment(args_.xtu_vars(), args_.p_vars()).coeff(i);
    }

    /// Full parameter vector segment.
    auto p_vec() const {
        if (args_.p_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::p_vec: no parameter variables declared (pvars=0)");
        return args_.segment(args_.xtu_vars(), args_.p_vars());
    }

    /// Sub-segment of the parameter vector (0-based within P).
    auto p_vec(int start, int count) const {
        if (count <= 0)
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::p_vec: count must be positive (got {})", count));
        if (start < 0 || start + count > args_.p_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::p_vec: range [{}, {}) out of P range [0, {})", start,
                            start + count, args_.p_vars()));
        return args_.segment(args_.xtu_vars() + start, count);
    }

    /// Sub-segment of the state vector with compile-time size (0-based within X).
    template <int SZ> auto x_vec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::x_vec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.x_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::x_vec<{}>: range [{}, {}) out of X range [0, {})", SZ,
                            start, start + SZ, args_.x_vars()));
        return args_.segment<SZ>(start);
    }

    /// Sub-segment of the control vector with compile-time size (0-based within U).
    template <int SZ> auto u_vec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::u_vec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.u_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::u_vec<{}>: range [{}, {}) out of U range [0, {})", SZ,
                            start, start + SZ, args_.u_vars()));
        return args_.segment<SZ>(args_.xt_vars() + start);
    }

    /// Sub-segment of the parameter vector with compile-time size (0-based within P).
    template <int SZ> auto p_vec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::p_vec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.p_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::p_vec<{}>: range [{}, {}) out of P range [0, {})", SZ,
                            start, start + SZ, args_.p_vars()));
        return args_.segment<SZ>(args_.xtu_vars() + start);
    }

    int xvars() const { return args_.x_vars(); }
    int uvars() const { return args_.u_vars(); }
    int pvars() const { return args_.p_vars(); }

  private:
    ODEArguments<> args_;
};

/// Fluent builder for ODE.
///
/// Usage:
///   auto ode = ODEBuilder(3, 1)
///       .define([](const auto& args) {
///           auto v = args.x_var(2);
///           auto theta = args.u_var(0);
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
            throw std::invalid_argument(
                fmt::format("ODEBuilder::var_group: range [{}, {}) exceeds XtUP size {}", start,
                            start + count, xtup));
        pending_groups_.emplace_back(name, start, count);
        return *this;
    }

    /// Build the ODE.
    ODE build() {
        if (state_ == State::Built)
            throw std::invalid_argument(
                "ODEBuilder: build() already called; create a new ODEBuilder");
        if (state_ == State::Initial) {
            throw std::invalid_argument(
                "ODEBuilder: no dynamics defined (call define() or from())");
        }

        ODE ode(func_, xvars_, uvars_, pvars_);

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
        if (func_.input_rows() != expected_ir)
            throw std::invalid_argument(
                fmt::format("ODEBuilder: function input size {} does not match XtUP size {} "
                            "(xv={}, uv={}, pv={})",
                            func_.input_rows(), expected_ir, xvars_, uvars_, pvars_));
        if (func_.output_rows() != xvars_)
            throw std::invalid_argument(
                fmt::format("ODEBuilder: function output size {} does not match XV={}",
                            func_.output_rows(), xvars_));
    }
};

} // namespace tycho
