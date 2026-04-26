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

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

template <int IR, int OR, class Func, class Data = std::integral_constant<bool, false>>
struct LambdaFunction
    : VectorFunction<LambdaFunction<IR, OR, Func, Data>, IR, OR, DenseDerivativeMode::FDiffCentArray,
                     DenseDerivativeMode::FDiffFwd>,
      Data {
    using Base = VectorFunction<LambdaFunction<IR, OR, Func, Data>, IR, OR,
                                DenseDerivativeMode::FDiffCentArray, DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base);
    using Base::compute;

    std::shared_ptr<Func> compute_func_;
    // LambdaFunction() = default;
    LambdaFunction(InputOutputSize<IR, OR> io, Func f) : compute_func_(std::make_shared<Func>(f)) {
        this->set_io_rows(io.input_rows_val, io.output_rows_val);
    }
    LambdaFunction(Data dat, InputOutputSize<IR, OR> io, Func f)
        : compute_func_(std::make_shared<Func>(f)), Data(dat) {
        this->set_io_rows(io.input_rows_val, io.output_rows_val);
    }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->compute_func_->operator()(this, x, fx);
    }
};

template <int IR, int OR, class Func, class JacFunc,
          class Data = std::integral_constant<bool, false>>
struct LambdaFunction2
    : VectorFunction<LambdaFunction2<IR, OR, Func, JacFunc, Data>, IR, OR,
                     DenseDerivativeMode::Analytic, DenseDerivativeMode::FDiffFwd>,
      Data {
    using Base = VectorFunction<LambdaFunction2<IR, OR, Func, JacFunc, Data>, IR, OR,
                                DenseDerivativeMode::Analytic, DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base);
    using Base::compute;

    std::shared_ptr<Func> compute_func_;
    std::shared_ptr<JacFunc> compute_jacobian_func_;
    // LambdaFunction2() = default;

    LambdaFunction2(InputOutputSize<IR, OR> io, Func f, JacFunc jf)
        : compute_func_(std::make_shared<Func>(f)),
          compute_jacobian_func_(std::make_shared<JacFunc>(jf)) {
        this->set_io_rows(io.input_rows_val, io.output_rows_val);
    }
    LambdaFunction2(Data dat, InputOutputSize<IR, OR> io, Func f, JacFunc jf)
        : compute_func_(std::make_shared<Func>(f)),
          compute_jacobian_func_(std::make_shared<JacFunc>(jf)), Data(dat) {
        this->set_io_rows(io.input_rows_val, io.output_rows_val);
    }
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->compute_func_->operator()(this, x, fx);
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        this->compute_jacobian_func_->operator()(this, x, fx, jx);
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

} // namespace tycho::vf
