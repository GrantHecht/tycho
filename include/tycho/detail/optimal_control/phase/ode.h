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
using vf::DenseDerivativeMode;
using vf::FunctionHolder;
using vf::GenericFunction;
using vf::VectorExpression;
using vf::VectorFunction;

template <class BaseType, class Derived, int _XV, int _UV, int _PV> struct ODEBase;

template <class Derived, int _XV, int _UV, int _PV,
          DenseDerivativeMode Jm = DenseDerivativeMode::Analytic,
          DenseDerivativeMode Hm = DenseDerivativeMode::Analytic>
struct ODE : ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>, Derived,
                     _XV, _UV, _PV> {
    using Base = ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>,
                         Derived, _XV, _UV, _PV>;

    void set_ode_size(int xv, int uv, int pv) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);
        this->set_input_rows(this->xtu_p_vars());
        this->set_output_rows(this->x_vars());
    }
};

template <class Derived, class ExprImpl, class... Ts>
struct ODE_Expression : ODEBase<VectorExpression<Derived, ExprImpl, Ts...>, Derived, ExprImpl::XV,
                                ExprImpl::UV, ExprImpl::PV> {
    using Base = ODEBase<VectorExpression<Derived, ExprImpl, Ts...>, Derived, ExprImpl::XV,
                         ExprImpl::UV, ExprImpl::PV>;
    using Base::Base;

    void set_ode_size(int xv, int uv, int pv) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);
    }
};

#define BUILD_ODE_FROM_EXPRESSION(NAME, IMPL, ...)                                                 \
    struct NAME : ODE_Expression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__> {                          \
        using Base = ODE_Expression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__>;                        \
        using Base::Base;                                                                          \
    };

// Wraps an expression-based ODE with a different derivative mode.
// The inner ODE is built from the expression tree (Analytic mode),
// but compute_jacobian uses the specified mode (FD/forward-AD),
// avoiding instantiation of the expression tree's Jacobian templates.
template <class Derived, class InnerODE,
          DenseDerivativeMode Jm = DenseDerivativeMode::FDiffFwd,
          DenseDerivativeMode Hm = DenseDerivativeMode::FDiffFwd>
struct ODE_DerivModeWrapper
    : ODEBase<VectorFunction<Derived, InnerODE::IRC, InnerODE::ORC, Jm, Hm>, Derived, InnerODE::XV,
              InnerODE::UV, InnerODE::PV> {
    using Base = ODEBase<VectorFunction<Derived, InnerODE::IRC, InnerODE::ORC, Jm, Hm>, Derived,
                         InnerODE::XV, InnerODE::UV, InnerODE::PV>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    InnerODE inner_;

    template <class... Args, std::enable_if_t<std::is_constructible_v<InnerODE, Args...>, int> = 0>
    ODE_DerivModeWrapper(Args &&...args) : inner_(std::forward<Args>(args)...) {
        this->set_xvars(inner_.x_vars());
        this->set_uvars(inner_.u_vars());
        this->set_pvars(inner_.p_vars());
        this->set_io_rows(inner_.input_rows(), inner_.output_rows());
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x,
                             ConstVectorBaseRef<OutType> fx_) const {
        inner_.compute(x, fx_);
    }
};

#define BUILD_ODE_FROM_EXPRESSION_FD(NAME, IMPL, ...)                                              \
    struct NAME##_AnalyticBase_                                                                     \
        : ODE_Expression<NAME##_AnalyticBase_, IMPL __VA_OPT__(, ) __VA_ARGS__> {                  \
        using Base = ODE_Expression<NAME##_AnalyticBase_, IMPL __VA_OPT__(, ) __VA_ARGS__>;        \
        using Base::Base;                                                                          \
    };                                                                                             \
    struct NAME                                                                                    \
        : ODE_DerivModeWrapper<NAME, NAME##_AnalyticBase_, DenseDerivativeMode::FDiffFwd,          \
                               DenseDerivativeMode::FDiffFwd> {                                    \
        using Base = ODE_DerivModeWrapper<NAME, NAME##_AnalyticBase_,                              \
                                          DenseDerivativeMode::FDiffFwd,                           \
                                          DenseDerivativeMode::FDiffFwd>;                          \
        using Base::Base;                                                                          \
    };

#define BUILD_ODE_FROM_EXPRESSION_FWAD(NAME, IMPL, ...)                                            \
    struct NAME##_AnalyticBase_                                                                     \
        : ODE_Expression<NAME##_AnalyticBase_, IMPL __VA_OPT__(, ) __VA_ARGS__> {                  \
        using Base = ODE_Expression<NAME##_AnalyticBase_, IMPL __VA_OPT__(, ) __VA_ARGS__>;        \
        using Base::Base;                                                                          \
    };                                                                                             \
    struct NAME                                                                                    \
        : ODE_DerivModeWrapper<NAME, NAME##_AnalyticBase_, DenseDerivativeMode::AutodiffFwd,       \
                               DenseDerivativeMode::AutodiffFwd> {                                 \
        using Base = ODE_DerivModeWrapper<NAME, NAME##_AnalyticBase_,                              \
                                          DenseDerivativeMode::AutodiffFwd,                        \
                                          DenseDerivativeMode::AutodiffFwd>;                       \
        using Base::Base;                                                                          \
    };

template <class BaseType, class Derived, int _XV, int _UV, int _PV>
struct ODEBase : BaseType, ODESize<_XV, _UV, _PV> {
    using Base = BaseType;
    using Base::Base;
    static const bool IsGenericODE = false;

    Integrator<Derived> integrator(double dstep) const {
        return Integrator<Derived>(this->derived(), dstep);
    }
};

template <class BaseType, int _XV, int _UV, int _PV>
struct GenericODE : FunctionHolder<GenericODE<BaseType, _XV, _UV, _PV>, BaseType,
                                   SZ_SUM<_XV, _UV, _PV, 1>::value, _XV>,
                    ODESize<_XV, _UV, _PV> {
    using Base = FunctionHolder<GenericODE<BaseType, _XV, _UV, _PV>, BaseType,
                                SZ_SUM<_XV, _UV, _PV, 1>::value, _XV>;
    using Base::Base;

    static const bool IsGenericODE = true;

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

    GenericODE(BaseType f, int xv, int uv) : GenericODE(f, xv, uv, 0) {}

    GenericODE(BaseType f, int xv) : GenericODE(f, xv, 0, 0) {}
    GenericODE(BaseType f) : GenericODE(f, _XV, _UV, _PV) {}
};

template <int XV, int UV, int PV>
using PythonGenericODE = GenericODE<GenericFunction<-1, -1>, XV, UV, PV>;

} // namespace tycho::oc
