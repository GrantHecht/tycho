// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/core/ode_sizes.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"
#include "tycho/vector_functions.h"

namespace tycho::oc {

// Import cross-namespace types used by ODE definitions.
using integrators::Integrator;
using utils::SZ_SUM;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::FunctionHolder;
using vf::GenericFunction;
using vf::VecRef;
using vf::VectorExpression;
using vf::VectorFunction;

/// @internal
/// @brief CRTP base shared by all statically-typed ODE definitions; declared here, defined below.
/// @tparam BaseType  The VectorFunction (or expression) base the ODE derives from.
/// @tparam Derived   The concrete ODE type (CRTP host).
/// @tparam _XV       State-vector dimension.
/// @tparam _UV       Control-vector dimension.
/// @tparam _PV       Parameter-vector dimension.
/// @endinternal
template <class BaseType, class Derived, int _XV, int _UV, int _PV> struct ODEBase;

/// @ingroup optimal_control
/// @brief CRTP base for hand-written, compile-time-sized ODE dynamics.
///
/// Derive from this to define a custom ODE @f$\dot{x}=f(x,t,u,p)@f$ as a
/// VectorFunction whose `compute_impl` is written by hand. The state, control,
/// and parameter dimensions are template parameters.
/// @tparam Derived  The concrete ODE type (CRTP host).
/// @tparam _XV      State-vector dimension.
/// @tparam _UV      Control-vector dimension.
/// @tparam _PV      Parameter-vector dimension.
/// @tparam Jm       Jacobian derivative mode.
/// @tparam Hm       Hessian derivative mode.
template <class Derived, int _XV, int _UV, int _PV,
          DenseDerivativeMode Jm = DenseDerivativeMode::Analytic,
          DenseDerivativeMode Hm = DenseDerivativeMode::Analytic>
struct StaticODE : ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>,
                           Derived, _XV, _UV, _PV> {
    /// @brief Convenience alias for the ODEBase CRTP base class.
    using Base = ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>,
                         Derived, _XV, _UV, _PV>;

    /// @brief Set the runtime ODE dimensions and the derived IO row counts.
    /// @param xv  Number of state variables.
    /// @param uv  Number of control variables.
    /// @param pv  Number of parameter variables.
    void set_ode_size(int xv, int uv, int pv) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);
        this->set_input_rows(this->xtu_p_vars());
        this->set_output_rows(this->x_vars());
    }
};

/// @ingroup optimal_control
/// @brief CRTP base for ODEs whose dynamics are a VectorFunction expression tree.
///
/// Used by the @c BUILD_ODE_FROM_EXPRESSION macro: the ODE's `compute_impl` is
/// the analytic VectorFunction expression, so derivatives come from the
/// expression tree.
/// @tparam Derived   The concrete ODE type (CRTP host).
/// @tparam ExprImpl  The expression-implementation type defining the dynamics.
/// @tparam Ts        Extra type parameters forwarded to the expression base.
template <class Derived, class ExprImpl, class... Ts>
struct StaticODE_Expression : ODEBase<VectorExpression<Derived, ExprImpl, Ts...>, Derived,
                                      ExprImpl::XV, ExprImpl::UV, ExprImpl::PV> {
    /// @brief Convenience alias for the ODEBase CRTP base class.
    using Base = ODEBase<VectorExpression<Derived, ExprImpl, Ts...>, Derived, ExprImpl::XV,
                         ExprImpl::UV, ExprImpl::PV>;
    using Base::Base;

    /// @brief Set the runtime ODE dimensions.
    /// @param xv  Number of state variables.
    /// @param uv  Number of control variables.
    /// @param pv  Number of parameter variables.
    void set_ode_size(int xv, int uv, int pv) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);
    }
};

/// @brief Define a named ODE type from a VectorFunction expression implementation.
/// @param NAME  Name of the generated ODE struct.
/// @param IMPL  The expression-implementation type defining the dynamics.
/// @param ...   Optional extra type parameters forwarded to the expression base.
/// @hideinitializer
#define BUILD_ODE_FROM_EXPRESSION(NAME, IMPL, ...)                                                 \
    struct NAME : StaticODE_Expression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__> {                    \
        using Base = StaticODE_Expression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__>;                  \
        using Base::Base;                                                                          \
    };

/// @ingroup optimal_control
/// @brief Wraps an ODE so its derivatives use a different (e.g. FD) mode.
///
/// Delegates `compute_impl` to an inner ODE but computes the Jacobian and
/// Hessian with the requested @p Jm / @p Hm modes. Used with expression-based
/// ODEs to avoid instantiating the expression tree's Jacobian/Hessian
/// templates, computing them by finite difference instead.
/// @tparam Derived   The concrete wrapper type (CRTP host).
/// @tparam InnerODE  The wrapped ODE type providing the primal `compute`.
/// @tparam Jm        Jacobian derivative mode.
/// @tparam Hm        Hessian derivative mode.
template <class Derived, class InnerODE, DenseDerivativeMode Jm = DenseDerivativeMode::FDiffFwd,
          DenseDerivativeMode Hm = DenseDerivativeMode::FDiffFwd>
struct StaticODE_DerivModeWrapper
    : ODEBase<VectorFunction<Derived, InnerODE::IRC, InnerODE::ORC, Jm, Hm>, Derived, InnerODE::XV,
              InnerODE::UV, InnerODE::PV> {
    /// @brief Convenience alias for the ODEBase CRTP base class.
    using Base = ODEBase<VectorFunction<Derived, InnerODE::IRC, InnerODE::ORC, Jm, Hm>, Derived,
                         InnerODE::XV, InnerODE::UV, InnerODE::PV>;
    VF_TYPE_ALIASES(Base);

    /// @brief Construct the inner ODE from forwarded arguments and size the wrapper.
    /// @tparam Args  Argument types forwarded to the @p InnerODE constructor.
    /// @param args   Arguments forwarded to the @p InnerODE constructor.
    template <class... Args>
        requires std::is_constructible_v<InnerODE, Args...>
    StaticODE_DerivModeWrapper(Args &&...args) : inner_(std::forward<Args>(args)...) {
        this->set_xvars(inner_.x_vars());
        this->set_uvars(inner_.u_vars());
        this->set_pvars(inner_.p_vars());
        this->set_idxs(inner_.get_idxs());
        this->set_io_rows(inner_.input_rows(), inner_.output_rows());
        // Re-initialize FD step-size vectors now that IO rows are known.
        // The FD base constructor ran before set_io_rows, so for dynamic-size
        // inner ODEs (IRC == -1) the step vectors were initialized to length 0.
        // For compile-time-sized ODEs this is a no-op that reassigns same-length vectors.
        if constexpr (Jm == DenseDerivativeMode::FDiffFwd) {
            this->set_jac_fd_steps(1.0e-7);
        } else if constexpr (Jm == DenseDerivativeMode::FDiffCentArray) {
            this->set_jac_fd_steps(1.0e-5);
        }
        if constexpr (Hm == DenseDerivativeMode::FDiffFwd) {
            this->set_hess_fd_steps(1.0e-7);
        } else if constexpr (Hm == DenseDerivativeMode::FDiffCentArray) {
            this->set_hess_fd_steps(1.0e-5);
        }
        // Other modes (Analytic, EnzymeAD) don't use FD step vectors;
        // add new FD-like modes here with an if constexpr branch calling
        // set_jac_fd_steps / set_hess_fd_steps.
        static_assert(Jm == DenseDerivativeMode::Analytic || Jm == DenseDerivativeMode::FDiffFwd ||
                          Jm == DenseDerivativeMode::FDiffCentArray ||
                          Jm == DenseDerivativeMode::EnzymeAD,
                      "Unhandled Jacobian derivative mode — check FD step reinitialization");
        static_assert(Hm == DenseDerivativeMode::Analytic || Hm == DenseDerivativeMode::FDiffFwd ||
                          Hm == DenseDerivativeMode::FDiffCentArray ||
                          Hm == DenseDerivativeMode::EnzymeAD,
                      "Unhandled Hessian derivative mode — check FD step reinitialization");
    }

    /// @brief Access the wrapped inner ODE.
    /// @return Const reference to the inner ODE.
    const InnerODE &inner() const { return inner_; }

    /// @internal
    /// @brief Delegate primal evaluation to the inner ODE.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input vector @c [x,t,u,p].
    /// @param fx_  Output state-derivative vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        inner_.compute(x, fx_);
    }

  private:
    InnerODE inner_; ///< The wrapped inner ODE supplying the primal `compute`.
};

/// @brief Define a named ODE that evaluates an expression but uses forward-FD derivatives.
///
/// Generates an intermediate analytic-mode ODE (`NAME##_AnalyticBase_`) for the
/// expression and wraps it in a @c StaticODE_DerivModeWrapper that computes
/// the Jacobian and Hessian by forward finite differences.
/// @param NAME  Name of the generated ODE struct.
/// @param IMPL  The expression-implementation type defining the dynamics.
/// @param ...   Optional extra type parameters forwarded to the expression base.
/// @hideinitializer
#define BUILD_ODE_FROM_EXPRESSION_FD(NAME, IMPL, ...)                                              \
    struct NAME##_AnalyticBase_                                                                    \
        : StaticODE_Expression<NAME##_AnalyticBase_, IMPL __VA_OPT__(, ) __VA_ARGS__> {            \
        using Base = StaticODE_Expression<NAME##_AnalyticBase_, IMPL __VA_OPT__(, ) __VA_ARGS__>;  \
        using Base::Base;                                                                          \
    };                                                                                             \
    struct NAME                                                                                    \
        : StaticODE_DerivModeWrapper<NAME, NAME##_AnalyticBase_, DenseDerivativeMode::FDiffFwd,    \
                                     DenseDerivativeMode::FDiffFwd> {                              \
        using Base =                                                                               \
            StaticODE_DerivModeWrapper<NAME, NAME##_AnalyticBase_, DenseDerivativeMode::FDiffFwd,  \
                                       DenseDerivativeMode::FDiffFwd>;                             \
        using Base::Base;                                                                          \
    };

/// @ingroup optimal_control
/// @brief CRTP base shared by all statically-typed ODE definitions.
///
/// Combines a VectorFunction (or expression) base with the @c ODESize trait
/// stack and adds the @ref integrator factory.
/// @tparam BaseType  The VectorFunction (or expression) base the ODE derives from.
/// @tparam Derived   The concrete ODE type (CRTP host).
/// @tparam _XV       State-vector dimension.
/// @tparam _UV       Control-vector dimension.
/// @tparam _PV       Parameter-vector dimension.
template <class BaseType, class Derived, int _XV, int _UV, int _PV>
struct ODEBase : BaseType, ODESize<_XV, _UV, _PV> {
    /// @brief Convenience alias for the VectorFunction (or expression) base class.
    using Base = BaseType;
    using Base::Base;
    static constexpr bool IsGenericODE = false; ///< This ODE is statically typed, not type-erased.

    /// @brief Build a fixed-step integrator for this ODE.
    /// @param dstep  Default integration step size.
    /// @return An @c Integrator bound to this ODE.
    Integrator<Derived> integrator(double dstep) const {
        return Integrator<Derived>(this->derived(), dstep);
    }
};

/// @ingroup optimal_control
/// @brief Type-erased ODE wrapping a runtime-sized @c tycho::vf::GenericFunction.
///
/// Used for ODEs defined at runtime (notably from Python), where the dynamics
/// are held as a type-erased function and the dimensions are supplied as
/// constructor arguments.
/// @tparam BaseType  The type-erased function type held by the ODE.
/// @tparam _XV       State-vector dimension.
/// @tparam _UV       Control-vector dimension.
/// @tparam _PV       Parameter-vector dimension.
template <class BaseType, int _XV, int _UV, int _PV>
struct GenericODE : FunctionHolder<GenericODE<BaseType, _XV, _UV, _PV>, BaseType,
                                   SZ_SUM<_XV, _UV, _PV, 1>::value, _XV>,
                    ODESize<_XV, _UV, _PV> {
    /// @brief Convenience alias for the FunctionHolder base class.
    using Base = FunctionHolder<GenericODE<BaseType, _XV, _UV, _PV>, BaseType,
                                SZ_SUM<_XV, _UV, _PV, 1>::value, _XV>;
    using Base::Base;

    static constexpr bool IsGenericODE = true; ///< This ODE is type-erased.

    /// @brief Construct from a function and explicit state, control, and param sizes.
    /// @param f   The type-erased dynamics function.
    /// @param xv  Number of state variables.
    /// @param uv  Number of control variables.
    /// @param pv  Number of parameter variables.
    /// @throws std::invalid_argument if @p f's output size != @p xv or its input
    ///         size != @c xv+uv+pv+1.
    GenericODE(BaseType f, int xv, int uv, int pv) : Base(f) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);

        if (this->output_rows() != xv) {
            throw std::invalid_argument(
                "Output Size of Generic ODE Expression does not match the specified "
                "model size");
        }
        if (this->input_rows() != (xv + uv + pv + 1)) {
            throw std::invalid_argument(
                "Input Size of Generic ODE Expression does not match the specified "
                "model size");
        }
    }

    /// @brief Construct with state and control sizes (zero parameters).
    /// @param f   The type-erased dynamics function.
    /// @param xv  Number of state variables.
    /// @param uv  Number of control variables.
    GenericODE(BaseType f, int xv, int uv) : GenericODE(f, xv, uv, 0) {}

    /// @brief Construct with a state size only (zero controls and parameters).
    /// @param f   The type-erased dynamics function.
    /// @param xv  Number of state variables.
    GenericODE(BaseType f, int xv) : GenericODE(f, xv, 0, 0) {}
    /// @brief Construct using the compile-time @p _XV, @p _UV, @p _PV sizes.
    /// @param f  The type-erased dynamics function.
    GenericODE(BaseType f) : GenericODE(f, _XV, _UV, _PV) {}
};

/// @ingroup optimal_control
/// @brief Type-erased ODE alias backed by a Python-facing @c GenericFunction.
/// @tparam XV  State-vector dimension.
/// @tparam UV  Control-vector dimension.
/// @tparam PV  Parameter-vector dimension.
template <int XV, int UV, int PV>
using PythonGenericODE = GenericODE<GenericFunction<-1, -1>, XV, UV, PV>;

} // namespace tycho::oc
