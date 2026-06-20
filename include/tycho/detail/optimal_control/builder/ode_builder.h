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

/// @ingroup optimal_control
/// @brief Proxy passed to the @ref ODEBuilder::define() lambda for building dynamics.
///
/// Provides simplified @c x_var / @c u_var / @c t_var / @c p_var / @c x_vec /
/// @c u_vec / @c p_vec accessors for building VectorFunction expressions from the
/// XtUP input layout.
class ODEArgsProxy {
  public:
    /// @brief Construct a proxy for the given ODE dimensions.
    /// @param xv  Number of state variables.
    /// @param uv  Number of control variables.
    /// @param pv  Number of parameter variables.
    ODEArgsProxy(int xv, int uv, int pv) : args_(xv, uv, pv) {}

    /// @brief Return the i-th state variable as a VectorFunction scalar expression (0-based within
    /// X).
    /// @param i  Zero-based state index in @c [0, xvars).
    /// @return A scalar VectorFunction expression representing @c X[i].
    /// @throws std::invalid_argument if no state variables were declared or @p i is out of range.
    auto x_var(int i) const {
        if (args_.x_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::x_var: no state variables declared (xvars=0)");
        if (i < 0 || i >= args_.x_vars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::x_var: index {} out of range [0, {})", i, args_.x_vars()));
        return args_.segment(0, args_.x_vars()).coeff(i);
    }

    /// @brief Return the full state vector as a VectorFunction segment expression.
    /// @return A segment expression spanning all @c xvars state variables.
    /// @throws std::invalid_argument if no state variables were declared.
    auto x_vec() const {
        if (args_.x_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::x_vec: no state variables declared (xvars=0)");
        return args_.segment(0, args_.x_vars());
    }

    /// @brief Return a runtime-sized sub-segment of the state vector (0-based within X).
    /// @param start  Zero-based starting index within @c X.
    /// @param count  Number of elements to select (must be positive).
    /// @return A segment expression spanning @c X[start..start+count).
    /// @throws std::invalid_argument if @p count is not positive or the range exceeds @c X.
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

    /// @brief Return the time variable as a VectorFunction scalar expression.
    /// @return A scalar expression representing the time @c t in the XtUP input layout.
    auto t_var() const { return args_.coeff(args_.t_var()); }

    /// @brief Return the i-th control variable as a VectorFunction scalar expression (0-based
    /// within U).
    /// @param i  Zero-based control index in @c [0, uvars).
    /// @return A scalar VectorFunction expression representing @c U[i].
    /// @throws std::invalid_argument if no control variables were declared or @p i is out of range.
    auto u_var(int i) const {
        if (args_.u_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::u_var: no control variables declared (uvars=0)");
        if (i < 0 || i >= args_.u_vars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::u_var: index {} out of range [0, {})", i, args_.u_vars()));
        return args_.segment(args_.xt_vars(), args_.u_vars()).coeff(i);
    }

    /// @brief Return the full control vector as a VectorFunction segment expression.
    /// @return A segment expression spanning all @c uvars control variables.
    /// @throws std::invalid_argument if no control variables were declared.
    auto u_vec() const {
        if (args_.u_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::u_vec: no control variables declared (uvars=0)");
        return args_.segment(args_.xt_vars(), args_.u_vars());
    }

    /// @brief Return a runtime-sized sub-segment of the control vector (0-based within U).
    /// @param start  Zero-based starting index within @c U.
    /// @param count  Number of elements to select (must be positive).
    /// @return A segment expression spanning @c U[start..start+count).
    /// @throws std::invalid_argument if @p count is not positive or the range exceeds @c U.
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

    /// @brief Return the i-th parameter variable as a VectorFunction scalar expression (0-based
    /// within P).
    /// @param i  Zero-based parameter index in @c [0, pvars).
    /// @return A scalar VectorFunction expression representing @c P[i].
    /// @throws std::invalid_argument if no parameter variables were declared or @p i is out of
    /// range.
    auto p_var(int i) const {
        if (args_.p_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::p_var: no parameter variables declared (pvars=0)");
        if (i < 0 || i >= args_.p_vars())
            throw std::invalid_argument(fmt::format(
                "ODEArgsProxy::p_var: index {} out of range [0, {})", i, args_.p_vars()));
        return args_.segment(args_.xtu_vars(), args_.p_vars()).coeff(i);
    }

    /// @brief Return the full parameter vector as a VectorFunction segment expression.
    /// @return A segment expression spanning all @c pvars parameter variables.
    /// @throws std::invalid_argument if no parameter variables were declared.
    auto p_vec() const {
        if (args_.p_vars() == 0)
            throw std::invalid_argument(
                "ODEArgsProxy::p_vec: no parameter variables declared (pvars=0)");
        return args_.segment(args_.xtu_vars(), args_.p_vars());
    }

    /// @brief Return a runtime-sized sub-segment of the parameter vector (0-based within P).
    /// @param start  Zero-based starting index within @c P.
    /// @param count  Number of elements to select (must be positive).
    /// @return A segment expression spanning @c P[start..start+count).
    /// @throws std::invalid_argument if @p count is not positive or the range exceeds @c P.
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

    /// @brief Return a compile-time-sized sub-segment of the state vector (0-based within X).
    /// @tparam SZ   Number of elements (must be positive, checked at compile time).
    /// @param start Zero-based starting index within @c X (defaults to 0).
    /// @return A fixed-size segment expression spanning @c X[start..start+SZ).
    /// @throws std::invalid_argument if the range @c [start, start+SZ) exceeds @c X.
    template <int SZ> auto x_vec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::x_vec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.x_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::x_vec<{}>: range [{}, {}) out of X range [0, {})", SZ,
                            start, start + SZ, args_.x_vars()));
        return args_.segment<SZ>(start);
    }

    /// @brief Return a compile-time-sized sub-segment of the control vector (0-based within U).
    /// @tparam SZ   Number of elements (must be positive, checked at compile time).
    /// @param start Zero-based starting index within @c U (defaults to 0).
    /// @return A fixed-size segment expression spanning @c U[start..start+SZ).
    /// @throws std::invalid_argument if the range @c [start, start+SZ) exceeds @c U.
    template <int SZ> auto u_vec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::u_vec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.u_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::u_vec<{}>: range [{}, {}) out of U range [0, {})", SZ,
                            start, start + SZ, args_.u_vars()));
        return args_.segment<SZ>(args_.xt_vars() + start);
    }

    /// @brief Return a compile-time-sized sub-segment of the parameter vector (0-based within P).
    /// @tparam SZ   Number of elements (must be positive, checked at compile time).
    /// @param start Zero-based starting index within @c P (defaults to 0).
    /// @return A fixed-size segment expression spanning @c P[start..start+SZ).
    /// @throws std::invalid_argument if the range @c [start, start+SZ) exceeds @c P.
    template <int SZ> auto p_vec(int start = 0) const {
        static_assert(SZ > 0, "ODEArgsProxy::p_vec<SZ>: SZ must be positive");
        if (start < 0 || start + SZ > args_.p_vars())
            throw std::invalid_argument(
                fmt::format("ODEArgsProxy::p_vec<{}>: range [{}, {}) out of P range [0, {})", SZ,
                            start, start + SZ, args_.p_vars()));
        return args_.segment<SZ>(args_.xtu_vars() + start);
    }

    /// @brief Number of state variables. @return The state-variable count.
    int xvars() const { return args_.x_vars(); }
    /// @brief Number of control variables. @return The control-variable count.
    int uvars() const { return args_.u_vars(); }
    /// @brief Number of parameter variables. @return The parameter-variable count.
    int pvars() const { return args_.p_vars(); }

  private:
    ODEArguments<> args_;
};

/// @ingroup optimal_control
/// @brief Fluent builder for an @ref ODE from a lambda or expression.
///
/// Accepts dynamics via a lambda (receiving an @ref ODEArgsProxy) or a pre-built
/// VectorFunction expression, optionally registers named variables, and produces
/// a runtime @ref ODE.
/// @code{.cpp}
/// auto ode = ODEBuilder(3, 1)
///     .define([](const auto& args) {
///         auto v = args.x_var(2);
///         auto theta = args.u_var(0);
///         return stack(sin(theta)*v, cos(theta)*v*(-1.0), 9.81*cos(theta));
///     })
///     .var_names({{"x",0}, {"y",1}, {"v",2}, {"t",3}, {"theta",4}})
///     .build();
/// @endcode
class ODEBuilder {
  public:
    /// @brief Construct a builder for the given ODE dimensions.
    /// @param xvars  Number of state variables (must be positive).
    /// @param uvars  Number of control variables (must be non-negative).
    /// @param pvars  Number of parameter variables (must be non-negative).
    /// @throws std::invalid_argument if @p xvars is not positive or any other dimension is
    /// negative.
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

    /// @brief Define ODE dynamics via a lambda.
    ///
    /// The lambda receives an @ref ODEArgsProxy and must return a VectorFunction
    /// expression (via @c stack(), @c StackedOutputs, or a single expression for
    /// 1-state ODEs).  The expression is validated against the declared dimensions
    /// immediately.
    ///
    /// @tparam F  Callable type with signature @c (const ODEArgsProxy&) -> VFExpr.
    /// @param func  Lambda that constructs and returns the dynamics expression.
    /// @return Reference to this builder for chaining.
    /// @throws std::invalid_argument if dynamics were already defined, if the
    ///         builder has already been used to build, or if the returned
    ///         expression has the wrong IO dimensions.
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

    /// @brief Define ODE dynamics from a pre-built VectorFunction expression.
    ///
    /// The expression is type-erased into a @c GenericFunction and validated
    /// against the declared dimensions immediately.
    ///
    /// @tparam Func  A concrete VectorFunction type.
    /// @param ode_expr  The pre-built dynamics expression.
    /// @return Reference to this builder for chaining.
    /// @throws std::invalid_argument if dynamics were already defined, if the
    ///         builder has already been used to build, or if @p ode_expr has
    ///         the wrong IO dimensions.
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

    /// @brief Register named variables mapped to XtUP-space indices.
    ///
    /// Each pair maps a variable name to a zero-based index in the flat
    /// @c [X, t, U, P] input vector.  May be called multiple times; entries
    /// accumulate.
    ///
    /// @param names  Brace-enclosed list of @c {name, xtup_index} pairs.
    /// @return Reference to this builder for chaining.
    /// @throws std::invalid_argument if the builder has already been used to
    ///         build, or if any index is outside @c [0, XtUP).
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

    /// @brief Register a named variable group covering a contiguous range in XtUP space.
    ///
    /// The group name can later be passed to @c VarRegistry::resolve() to
    /// obtain all indices at once.  May be called multiple times; groups
    /// accumulate.
    ///
    /// @param name   Group name (must be non-empty and not already registered).
    /// @param start  Zero-based start index in @c [0, XtUP) (must be non-negative).
    /// @param count  Number of variables in the group (must be positive).
    /// @return Reference to this builder for chaining.
    /// @throws std::invalid_argument if the builder has already been used to
    ///         build, if @p start is negative, if @p count is not positive, or
    ///         if the range @c [start, start+count) exceeds the XtUP size.
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

    /// @brief Consume the builder and produce the configured @ref ODE.
    ///
    /// Transfers all registered names and groups to the new @ref ODE.
    /// The builder must not be used again after this call.
    ///
    /// @return The constructed @ref ODE.
    /// @throws std::invalid_argument if dynamics were never defined (neither
    ///         @c define() nor @c from() was called) or if @c build() was
    ///         already called on this builder.
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
