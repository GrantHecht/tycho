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

template <class BaseType, class Derived, int _XV, int _UV, int _PV> struct ODEBase;

template <class Derived, int _XV, int _UV, int _PV,
          DenseDerivativeMode Jm = DenseDerivativeMode::Analytic,
          DenseDerivativeMode Hm = DenseDerivativeMode::Analytic>
struct StaticODE : ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>,
                           Derived, _XV, _UV, _PV> {
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
struct StaticODE_Expression : ODEBase<VectorExpression<Derived, ExprImpl, Ts...>, Derived,
                                      ExprImpl::XV, ExprImpl::UV, ExprImpl::PV> {
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
    struct NAME : StaticODE_Expression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__> {                    \
        using Base = StaticODE_Expression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__>;                  \
        using Base::Base;                                                                          \
    };

// Wraps any ODE with a different derivative mode.
// The wrapper delegates compute to the inner ODE but uses the specified
// modes (e.g., FD or forward-AD) for Jacobian and Hessian computation.
// When used with expression-based ODEs (via the macros below), this avoids
// instantiation of the expression tree's Jacobian and Hessian templates.
template <class Derived, class InnerODE, DenseDerivativeMode Jm = DenseDerivativeMode::FDiffFwd,
          DenseDerivativeMode Hm = DenseDerivativeMode::FDiffFwd>
struct StaticODE_DerivModeWrapper
    : ODEBase<VectorFunction<Derived, InnerODE::IRC, InnerODE::ORC, Jm, Hm>, Derived, InnerODE::XV,
              InnerODE::UV, InnerODE::PV> {
    using Base = ODEBase<VectorFunction<Derived, InnerODE::IRC, InnerODE::ORC, Jm, Hm>, Derived,
                         InnerODE::XV, InnerODE::UV, InnerODE::PV>;
    VF_TYPE_ALIASES(Base);

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

    const InnerODE &inner() const { return inner_; }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        inner_.compute(x, fx_);
    }

  private:
    InnerODE inner_;
};

// Defines an ODE that uses the expression tree for compute_impl only, with
// Jacobian/Hessian computed via forward finite differences. Generates an
// intermediate analytic-mode ODE (NAME##_AnalyticBase_) and wraps it.
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


template <class BaseType, class Derived, int _XV, int _UV, int _PV>
struct ODEBase : BaseType, ODESize<_XV, _UV, _PV> {
    using Base = BaseType;
    using Base::Base;
    static constexpr bool IsGenericODE = false;

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

    static constexpr bool IsGenericODE = true;

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
