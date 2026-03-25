// =============================================================================
// Tycho — Builder API: ODEBuilder
//
// Fluent builder for constructing RuntimeODE objects.  Accepts dynamics via
// a lambda (receiving a proxy with XVar/UVar/TVar/XVec/UVec accessors) or
// a pre-built VectorFunction expression.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include "tycho/detail/GenericFunction.h"
#include "tycho/detail/ODEArguments.h"
#include "tycho/detail/runtime_ode.h"
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Tycho {

/// Proxy passed to the `define()` lambda.  Provides XVar/UVar/TVar/PVar/XVec/
/// UVec/PVec accessors that mirror the Python ODEArguments API.
///
/// Wraps a fully-dynamic ODEArguments<-1,-1,-1>.
class ODEArgsProxy {
  public:
    ODEArgsProxy(int xv, int uv, int pv) : args_(xv, uv, pv) {}

    /// i-th state variable (0-based within X).
    auto XVar(int i) const { return args_.segment(0, args_.XVars()).coeff(i); }

    /// State vector segment.
    auto XVec() const { return args_.segment(0, args_.XVars()); }

    /// Time variable.
    auto TVar() const { return args_.coeff(args_.TVar()); }

    /// i-th control variable (0-based within U).
    auto UVar(int i) const { return args_.segment(args_.XtVars(), args_.UVars()).coeff(i); }

    /// Control vector segment.
    auto UVec() const { return args_.segment(args_.XtVars(), args_.UVars()); }

    /// i-th parameter variable (0-based within P).
    auto PVar(int i) const { return args_.segment(args_.XtUVars(), args_.PVars()).coeff(i); }

    /// Parameter vector segment.
    auto PVec() const { return args_.segment(args_.XtUVars(), args_.PVars()); }

    int xvars() const { return args_.XVars(); }
    int uvars() const { return args_.UVars(); }
    int pvars() const { return args_.PVars(); }

  private:
    ODEArguments<-1, -1, -1> args_;
};

/// Fluent builder for RuntimeODE.
///
/// Usage:
///   auto ode = ODEBuilder(3, 1)
///       .define([](auto& args) {
///           auto v = args.XVar(2);
///           auto theta = args.UVar(0);
///           return stack(sin(theta)*v, -cos(theta)*v, 9.81*cos(theta));
///       })
///       .var_names({{"x",0}, {"y",1}, {"v",2}, {"theta",4}})
///       .build();
class ODEBuilder {
  public:
    ODEBuilder(int xvars, int uvars = 0, int pvars = 0)
        : xvars_(xvars), uvars_(uvars), pvars_(pvars) {}

    /// Define ODE dynamics via a lambda.
    /// The lambda receives an ODEArgsProxy and must return a VectorFunction
    /// expression (typically via stack() or StackedOutputs).
    template <typename F> ODEBuilder &define(F &&func) {
        if (func_set_) {
            throw std::invalid_argument("ODEBuilder: dynamics already defined");
        }
        ODEArgsProxy proxy(xvars_, uvars_, pvars_);
        auto expr = func(proxy);
        func_ = GenericFunction<-1, -1>(expr);
        func_set_ = true;
        return *this;
    }

    /// Define ODE from a pre-built VectorFunction expression.
    template <typename Func> ODEBuilder &from(const Func &ode_expr) {
        if (func_set_) {
            throw std::invalid_argument("ODEBuilder: dynamics already defined");
        }
        func_ = GenericFunction<-1, -1>(ode_expr);
        func_set_ = true;
        return *this;
    }

    /// Register named variables (name -> XtUP index).
    ODEBuilder &var_names(std::initializer_list<std::pair<std::string, int>> names) {
        for (const auto &p : names)
            pending_names_.push_back(p);
        return *this;
    }

    /// Register a named variable group (contiguous range).
    ODEBuilder &var_group(const std::string &name, int start, int count) {
        pending_groups_.emplace_back(name, start, count);
        return *this;
    }

    /// Build the RuntimeODE.
    RuntimeODE build() {
        if (!func_set_) {
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

        return ode;
    }

  private:
    int xvars_;
    int uvars_;
    int pvars_;
    GenericFunction<-1, -1> func_;
    bool func_set_ = false;
    std::vector<std::pair<std::string, int>> pending_names_;
    std::vector<std::tuple<std::string, int, int>> pending_groups_;
};

} // namespace Tycho
