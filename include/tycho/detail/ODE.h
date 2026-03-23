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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/ODEPhase.h"
#include "tycho/detail/ODESizes.h"
#include "VectorFunctions/Tycho_VectorFunctions.h"

namespace Tycho {

template <class BaseType, class Derived, int _XV, int _UV, int _PV> struct ODEBase;

template <class Derived, int _XV, int _UV, int _PV,
          DenseDerivativeMode Jm = DenseDerivativeMode::Analytic,
          DenseDerivativeMode Hm = DenseDerivativeMode::Analytic>
struct ODE : ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>, Derived,
                     _XV, _UV, _PV> {
    using Base = ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>,
                         Derived, _XV, _UV, _PV>;

    void setODESize(int xv, int uv, int pv) {
        this->setXVars(xv);
        this->setUVars(uv);
        this->setPVars(pv);
        this->setInputRows(this->XtUPVars());
        this->setOutputRows(this->XVars());
    }
};

template <class Derived, class ExprImpl, class... Ts>
struct ODE_Expression : ODEBase<VectorExpression<Derived, ExprImpl, Ts...>, Derived, ExprImpl::XV,
                                ExprImpl::UV, ExprImpl::PV> {
    using Base = ODEBase<VectorExpression<Derived, ExprImpl, Ts...>, Derived, ExprImpl::XV,
                         ExprImpl::UV, ExprImpl::PV>;
    using Base::Base;

    void setODESize(int xv, int uv, int pv) {
        this->setXVars(xv);
        this->setUVars(uv);
        this->setPVars(pv);
    }
};

#define BUILD_ODE_FROM_EXPRESSION(NAME, IMPL, ...)                                                 \
    struct NAME : ODE_Expression<NAME, IMPL, __VA_ARGS__> {                                        \
        using Base = ODE_Expression<NAME, IMPL, __VA_ARGS__>;                                      \
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
        this->setXVars(xv);
        this->setUVars(uv);
        this->setPVars(pv);

        if (this->ORows() != xv) {
            throw std::invalid_argument(
                "Output Size of Generic ODE Expression does not match the specified "
                "model size");
        }
        if (this->IRows() != (xv + uv + pv + 1)) {
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

} // namespace Tycho

