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

/// @brief VectorFunction adapter wrapping a user callable as the value function.
///
/// Wraps an arbitrary callable @p Func (invoked as `f(this, x, fx)`) so it can
/// participate in the VectorFunction DSL. Derivatives are taken numerically:
/// the Jacobian by central finite differences over SuperScalar arrays and the
/// Hessian by forward finite differences. An optional @p Data base lets callers
/// mix in extra static traits.
///
/// @tparam IR    Compile-time input-row count (Eigen::Dynamic if runtime).
/// @tparam OR    Compile-time output-row count (Eigen::Dynamic if runtime).
/// @tparam Func  Callable computing the value; invoked as `f(this, x, fx)`.
/// @tparam Data  Optional mix-in base carrying extra compile-time traits.
/// @ingroup vf
template <int IR, int OR, class Func, class Data = std::integral_constant<bool, false>>
struct LambdaFunction
    : VectorFunction<LambdaFunction<IR, OR, Func, Data>, IR, OR,
                     DenseDerivativeMode::FDiffCentArray, DenseDerivativeMode::FDiffFwd>,
      Data {
    /// @brief VectorFunction base: central-FD Jacobian, forward-FD Hessian.
    using Base = VectorFunction<LambdaFunction<IR, OR, Func, Data>, IR, OR,
                                DenseDerivativeMode::FDiffCentArray, DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base);
    using Base::compute;

    std::shared_ptr<Func> compute_func_; ///< @brief Shared handle to the value callable.
    // LambdaFunction() = default;
    /// @brief Construct from explicit I/O sizes and a value callable.
    /// @param io  Input/output row sizes.
    /// @param f   Callable invoked as `f(this, x, fx)`.
    LambdaFunction(InputOutputSize<IR, OR> io, Func f) : compute_func_(std::make_shared<Func>(f)) {
        this->set_io_rows(io.input_rows_val, io.output_rows_val);
    }
    /// @brief Construct with a @p Data mix-in, I/O sizes, and a value callable.
    /// @param dat  Mix-in trait base value.
    /// @param io   Input/output row sizes.
    /// @param f    Callable invoked as `f(this, x, fx)`.
    LambdaFunction(Data dat, InputOutputSize<IR, OR> io, Func f)
        : compute_func_(std::make_shared<Func>(f)), Data(dat) {
        this->set_io_rows(io.input_rows_val, io.output_rows_val);
    }

    /// @brief Evaluate the wrapped callable: `fx = compute_func_(x)`.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->compute_func_->operator()(this, x, fx);
    }
};

/// @brief VectorFunction adapter wrapping a value callable plus a Jacobian callable.
///
/// Like @ref LambdaFunction but with a user-supplied analytic Jacobian callable
/// (invoked as `jf(this, x, fx, jx)`); the Hessian is still taken by forward
/// finite differences. An optional @p Data base mixes in extra static traits.
///
/// @tparam IR       Compile-time input-row count (Eigen::Dynamic if runtime).
/// @tparam OR       Compile-time output-row count (Eigen::Dynamic if runtime).
/// @tparam Func     Callable computing the value; invoked as `f(this, x, fx)`.
/// @tparam JacFunc  Callable computing the Jacobian; `jf(this, x, fx, jx)`.
/// @tparam Data     Optional mix-in base carrying extra compile-time traits.
/// @ingroup vf
template <int IR, int OR, class Func, class JacFunc,
          class Data = std::integral_constant<bool, false>>
struct LambdaFunction2
    : VectorFunction<LambdaFunction2<IR, OR, Func, JacFunc, Data>, IR, OR,
                     DenseDerivativeMode::Analytic, DenseDerivativeMode::FDiffFwd>,
      Data {
    /// @brief VectorFunction base: analytic Jacobian, forward-FD Hessian.
    using Base = VectorFunction<LambdaFunction2<IR, OR, Func, JacFunc, Data>, IR, OR,
                                DenseDerivativeMode::Analytic, DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base);
    using Base::compute;

    std::shared_ptr<Func> compute_func_; ///< @brief Shared handle to the value callable.
    std::shared_ptr<JacFunc>
        compute_jacobian_func_; ///< @brief Shared handle to the Jacobian callable.
    // LambdaFunction2() = default;

    /// @brief Construct from I/O sizes, a value callable, and a Jacobian callable.
    /// @param io  Input/output row sizes.
    /// @param f   Value callable invoked as `f(this, x, fx)`.
    /// @param jf  Jacobian callable invoked as `jf(this, x, fx, jx)`.
    LambdaFunction2(InputOutputSize<IR, OR> io, Func f, JacFunc jf)
        : compute_func_(std::make_shared<Func>(f)),
          compute_jacobian_func_(std::make_shared<JacFunc>(jf)) {
        this->set_io_rows(io.input_rows_val, io.output_rows_val);
    }
    /// @brief Construct with a @p Data mix-in plus value and Jacobian callables.
    /// @param dat  Mix-in trait base value.
    /// @param io   Input/output row sizes.
    /// @param f    Value callable invoked as `f(this, x, fx)`.
    /// @param jf   Jacobian callable invoked as `jf(this, x, fx, jx)`.
    LambdaFunction2(Data dat, InputOutputSize<IR, OR> io, Func f, JacFunc jf)
        : compute_func_(std::make_shared<Func>(f)),
          compute_jacobian_func_(std::make_shared<JacFunc>(jf)), Data(dat) {
        this->set_io_rows(io.input_rows_val, io.output_rows_val);
    }
    /// @brief Evaluate the wrapped value callable: `fx = compute_func_(x)`.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->compute_func_->operator()(this, x, fx);
    }

    /// @brief Evaluate value and analytic Jacobian via the wrapped callables.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        this->compute_jacobian_func_->operator()(this, x, fx, jx);
    }
};

/// @brief Build a @ref LambdaFunction from I/O sizes and a value callable.
/// @tparam IR    Compile-time input-row count.
/// @tparam OR    Compile-time output-row count.
/// @tparam Func  Value callable type.
/// @param  io    Input/output row sizes.
/// @param  f     Value callable invoked as `f(this, x, fx)`.
/// @return A @ref LambdaFunction wrapping @p f.
template <int IR, int OR, class Func> auto make_lambda_func(InputOutputSize<IR, OR> io, Func f) {
    return LambdaFunction{std::integral_constant<bool, false>(), io, f};
}
/// @brief Build a @ref LambdaFunction with a @p Data mix-in.
/// @tparam Data  Mix-in trait base type.
/// @tparam IR    Compile-time input-row count.
/// @tparam OR    Compile-time output-row count.
/// @tparam Func  Value callable type.
/// @param  dat   Mix-in trait base value.
/// @param  io    Input/output row sizes.
/// @param  f     Value callable invoked as `f(this, x, fx)`.
/// @return A @ref LambdaFunction wrapping @p f and carrying @p dat.
template <class Data, int IR, int OR, class Func>
auto make_lambda_func(Data dat, InputOutputSize<IR, OR> io, Func f) {
    return LambdaFunction{dat, io, f};
}
/// @brief Build a @ref LambdaFunction2 from value and Jacobian callables.
/// @tparam IR       Compile-time input-row count.
/// @tparam OR       Compile-time output-row count.
/// @tparam Func     Value callable type.
/// @tparam JacFunc  Jacobian callable type.
/// @param  io       Input/output row sizes.
/// @param  f        Value callable invoked as `f(this, x, fx)`.
/// @param  jf       Jacobian callable invoked as `jf(this, x, fx, jx)`.
/// @return A @ref LambdaFunction2 wrapping @p f and @p jf.
template <int IR, int OR, class Func, class JacFunc>
auto make_lambda_func(InputOutputSize<IR, OR> io, Func f, JacFunc jf) {
    return LambdaFunction2{std::integral_constant<bool, false>(), io, f, jf};
}
/// @brief Build a @ref LambdaFunction2 with a @p Data mix-in.
/// @tparam Data     Mix-in trait base type.
/// @tparam IR       Compile-time input-row count.
/// @tparam OR       Compile-time output-row count.
/// @tparam Func     Value callable type.
/// @tparam JacFunc  Jacobian callable type.
/// @param  dat      Mix-in trait base value.
/// @param  io       Input/output row sizes.
/// @param  f        Value callable invoked as `f(this, x, fx)`.
/// @param  jf       Jacobian callable invoked as `jf(this, x, fx, jx)`.
/// @return A @ref LambdaFunction2 wrapping @p f and @p jf and carrying @p dat.
template <class Data, int IR, int OR, class Func, class JacFunc>
auto make_lambda_func(Data dat, InputOutputSize<IR, OR> io, Func f, JacFunc jf) {
    return LambdaFunction2{dat, io, f, jf};
}

} // namespace tycho::vf
