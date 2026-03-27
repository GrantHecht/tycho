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
//   - pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/vf/core/vector_function.h"

namespace Tycho {

template <int IR, int OR, class Func, class Data = std::integral_constant<bool, false>>
struct LambdaFunction
    : VectorFunction<LambdaFunction<IR, OR, Func, Data>, IR, OR, DenseDerivativeMode::AutodiffFwd,
                     DenseDerivativeMode::AutodiffFwd>,
      Data {
    using Base = VectorFunction<LambdaFunction<IR, OR, Func, Data>, IR, OR,
                                DenseDerivativeMode::AutodiffFwd, DenseDerivativeMode::AutodiffFwd>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::compute;

    std::shared_ptr<Func> compute_func;
    // LambdaFunction() = default;
    LambdaFunction(InputOutputSize<IR, OR> io, Func f) : compute_func(std::make_shared<Func>(f)) {
        this->setIORows(io.InputRows, io.OutputRows);
    }
    LambdaFunction(Data dat, InputOutputSize<IR, OR> io, Func f)
        : compute_func(std::make_shared<Func>(f)), Data(dat) {
        this->setIORows(io.InputRows, io.OutputRows);
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        this->compute_func->operator()(this, x, fx);
    }
};

template <int IR, int OR, class Func, class JacFunc,
          class Data = std::integral_constant<bool, false>>
struct LambdaFunction2
    : VectorFunction<LambdaFunction2<IR, OR, Func, JacFunc, Data>, IR, OR,
                     DenseDerivativeMode::Analytic, DenseDerivativeMode::AutodiffFwd>,
      Data {
    using Base = VectorFunction<LambdaFunction2<IR, OR, Func, JacFunc, Data>, IR, OR,
                                DenseDerivativeMode::Analytic, DenseDerivativeMode::AutodiffFwd>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::compute;

    std::shared_ptr<Func> compute_func;
    std::shared_ptr<JacFunc> compute_jacobian_func;
    // LambdaFunction2() = default;

    LambdaFunction2(InputOutputSize<IR, OR> io, Func f, JacFunc jf)
        : compute_func(std::make_shared<Func>(f)),
          compute_jacobian_func(std::make_shared<JacFunc>(jf)) {
        this->setIORows(io.InputRows, io.OutputRows);
    }
    LambdaFunction2(Data dat, InputOutputSize<IR, OR> io, Func f, JacFunc jf)
        : compute_func(std::make_shared<Func>(f)),
          compute_jacobian_func(std::make_shared<JacFunc>(jf)), Data(dat) {
        this->setIORows(io.InputRows, io.OutputRows);
    }
    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        this->compute_func->operator()(this, x, fx);
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        this->compute_jacobian_func->operator()(this, x, fx, jx);
    }
};

template <int IR, int OR, class Func> auto make_lambda_func(InputOutputSize<IR, OR> io, Func f) {
    return LambdaFunction{std::integral_constant<bool, false>(), io, f};
}
template <class Data, int IR, int OR, class Func>
auto make_lambda_func(Data dat, InputOutputSize<IR, OR> io, Func f) {
    return LambdaFunction{dat, io, f};
}
template <int IR, int OR, class Func, class JacFunc>
auto make_lambda_func(InputOutputSize<IR, OR> io, Func f, JacFunc jf) {
    return LambdaFunction2{std::integral_constant<bool, false>(), io, f, jf};
}
template <class Data, int IR, int OR, class Func, class JacFunc>
auto make_lambda_func(Data dat, InputOutputSize<IR, OR> io, Func f, JacFunc jf) {
    return LambdaFunction2{dat, io, f, jf};
}

} // namespace Tycho
