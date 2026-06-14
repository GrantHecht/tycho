// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines all Vector Functionals that apply linear scalings to
// the outputs of other functions.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

/// @file
/// @brief VectorFunctions applying linear (scalar, vector, or matrix) scalings
///        to the outputs of other functions.
#pragma once
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @cond INTERNAL
//! Declaration of \struct StaticScaled_Impl
template <class Derived, class Func, class Value> struct StaticScaled_Impl;

//! Declaration of \struct Scaled_Impl
template <class Derived, class Func> struct Scaled_Impl;

//! Declaration of \struct RowScaled_Impl
template <class Derived, class Func> struct RowScaled_Impl;

//! Declaration of \struct MatrixScaled_Impl
template <class Derived, class Func, int MRows> struct MatrixScaled_Impl;
/// @endcond

/// @ingroup vf
/// @brief Scales a function's output by a compile-time constant value.
///
/// Computes @f$ g(x) = c \cdot f(x) @f$ where @f$ c @f$ is the compile-time
/// constant @p Value::value; the Jacobian, adjoint gradient, and adjoint Hessian
/// are uniformly scaled by the same factor at zero runtime overhead.
/// Typically produced by the `*` operator with a `std::integral_constant` or
/// similar tag on a VectorFunction.
///
/// @tparam Func   Wrapped VectorFunction type.
/// @tparam Value  Compile-time constant providing the scale via `Value::value`.
template <class Func, class Value>
struct StaticScaled : StaticScaled_Impl<StaticScaled<Func, Value>, Func, Value> {
    /// @brief Convenience alias for the underlying scaled implementation.
    using Base = StaticScaled_Impl<StaticScaled<Func, Value>, Func, Value>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @ingroup vf
/// @brief Scales a function's output by a single runtime scalar.
///
/// Computes @f$ g(x) = s \cdot f(x) @f$ where @f$ s @f$ is a `double` supplied
/// at construction; the Jacobian and adjoint quantities are scaled by the same
/// factor. Produced by the `*` operator on a VectorFunction and a `double`, or
/// by calling `f.scale(s)`.
///
/// @tparam Func  Wrapped VectorFunction type.
template <class Func> struct Scaled : Scaled_Impl<Scaled<Func>, Func> {
    /// @brief Convenience alias for the underlying scaled implementation.
    using Base = Scaled_Impl<Scaled<Func>, Func>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @ingroup vf
/// @brief Scales each output row of a function by its own constant factor.
///
/// Computes @f$ g_i(x) = s_i \cdot f_i(x) @f$ for each output row @f$ i @f$,
/// where @f$ \mathbf{s} @f$ is a per-row `Output<double>` vector; the Jacobian
/// rows and adjoint variables are pre-scaled by the same factors.
/// Produced by calling `f.row_scale(s)` on a VectorFunction.
///
/// @tparam Func  Wrapped VectorFunction type.
template <class Func> struct RowScaled : RowScaled_Impl<RowScaled<Func>, Func> {
    /// @brief Convenience alias for the underlying row-scaled implementation.
    using Base = RowScaled_Impl<RowScaled<Func>, Func>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @ingroup vf
/// @brief Left-multiplies a function's output by a constant matrix.
///
/// Computes @f$ g(x) = A \, f(x) @f$ where @f$ A @f$ is an @p MRows
/// @f$ \times @f$ `Func::ORC` constant `double` matrix stored at construction
/// time; the Jacobian is similarly left-multiplied and the adjoint variables are
/// pre-multiplied by @f$ A^\top @f$. Produced by calling `f.matrix_scale(A)` on
/// a VectorFunction.
///
/// @tparam Func   Wrapped VectorFunction type.
/// @tparam MRows  Compile-time row count of the resulting (matrix-scaled) output.
template <class Func, int MRows>
struct MatrixScaled : MatrixScaled_Impl<MatrixScaled<Func, MRows>, Func, MRows> {
    /// @brief Convenience alias for the underlying matrix-scaled implementation.
    using Base = MatrixScaled_Impl<MatrixScaled<Func, MRows>, Func, MRows>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Shared implementation of @ref StaticScaled (compile-time scalar scale).
/// @tparam Derived  CRTP host type (@ref StaticScaled).
/// @tparam Func     Wrapped VectorFunction type.
/// @tparam Value    Compile-time constant providing the scale via `Value::value`.
/// @endinternal
template <class Derived, class Func, class Value>
struct StaticScaled_Impl
    : VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);
    /// @brief The wrapped function whose output is scaled.
    Func func_;
    /// @brief Input-domain sparsity inherited from the wrapped function.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    /// @brief Linearity inherited from the wrapped function (scaling is linear).
    static constexpr bool is_linear_function = Func::is_linear_function;

    /// @brief Constructs an empty wrapper (configured later).
    StaticScaled_Impl() {}
    /// @brief Constructs a static-scaled wrapper around a function.
    /// @param f  Function to wrap.
    StaticScaled_Impl(Func f) : func_(std::move(f)) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
    }
    /// @internal
    /// @brief Evaluates the wrapped function and scales the output by `Value::value`.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector receiving the scaled result.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->func_.compute(x, fx_);
        fx *= Value::value;
    }
    /// @internal
    /// @brief Evaluates value and Jacobian, both scaled by `Value::value`.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector receiving the scaled result.
    /// @param jx_  Jacobian buffer receiving the scaled Jacobian.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        this->func_.compute_jacobian(x, fx_, jx_);

        fx *= Value::value;
        this->func_.scale_jacobian(jx, Value::value);
    }
    /// @internal
    /// @brief Evaluates value, Jacobian, adjoint gradient, and Hessian, all scaled.
    ///
    /// The adjoint variables are pre-scaled by `Value::value` so the returned
    /// adjoint gradient and Hessian are consistent with the scaled output.
    /// @tparam InType       Eigen input vector expression type.
    /// @tparam OutType      Eigen output vector expression type.
    /// @tparam JacType      Eigen Jacobian matrix expression type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector expression type.
    /// @param x        Input vector.
    /// @param fx_      Output vector receiving the scaled result.
    /// @param jx_      Jacobian buffer receiving the scaled Jacobian.
    /// @param adjgrad_ Adjoint-gradient buffer.
    /// @param adjhess_ Adjoint-Hessian buffer.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        // VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        // MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        Output<Scalar> adjv_scaled = adjvars * Value::value;

        this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                    adjv_scaled);
        fx *= Value::value;
        this->func_.scale_jacobian(jx, Value::value);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Forwards a right-Jacobian product to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Left Left-operand type.
    /// @tparam Right  Eigen right-operand type. @tparam Assignment Assignment tag.
    /// @tparam Aliased Whether operands may alias the target.
    /// @param target_ Destination matrix. @param left Left operand.
    /// @param right Right operand. @param assign Assignment policy.
    /// @param aliased Aliasing flag. @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        this->func_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal
    /// @brief Forwards a symmetric-Jacobian product to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Left Left-operand type.
    /// @tparam Right  Eigen right-operand type. @tparam Assignment Assignment tag.
    /// @tparam Aliased Whether operands may alias the target.
    /// @param target_ Destination matrix. @param left Left operand.
    /// @param right Right operand. @param assign Assignment policy.
    /// @param aliased Aliasing flag. @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                          CEigRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func_.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal
    /// @brief Forwards a Jacobian accumulation to the wrapped function.
    /// @tparam Target Eigen target type. @tparam JacType Source type.
    /// @tparam Assignment Assignment tag. @param target_ Destination.
    /// @param right Source contribution. @param assign Assignment policy. @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_jacobian(target_, right, assign);
    }
    /// @internal
    /// @brief Forwards a gradient accumulation to the wrapped function.
    /// @tparam Target Eigen target type. @tparam JacType Source type.
    /// @tparam Assignment Assignment tag. @param target_ Destination.
    /// @param right Source contribution. @param assign Assignment policy. @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_gradient(target_, right, assign);
    }
    /// @internal
    /// @brief Forwards a Hessian accumulation to the wrapped function.
    /// @tparam Target Eigen target type. @tparam JacType Source type.
    /// @tparam Assignment Assignment tag. @param target_ Destination.
    /// @param right Source contribution. @param assign Assignment policy. @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        this->func_.accumulate_hessian(target_, right, assign);
    }
    /// @internal
    /// @brief Forwards a Jacobian scaling to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Scalar Scalar factor type.
    /// @param target_ Jacobian to scale. @param s Scalar factor. @endinternal
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_jacobian(target_, s);
    }
    /// @internal
    /// @brief Forwards a gradient scaling to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Scalar Scalar factor type.
    /// @param target_ Gradient to scale. @param s Scalar factor. @endinternal
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_gradient(target_, s);
    }
    /// @internal
    /// @brief Forwards a Hessian scaling to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Scalar Scalar factor type.
    /// @param target_ Hessian to scale. @param s Scalar factor. @endinternal
    template <class Target, class Scalar>
    inline void scale_hessian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_hessian(target_, s);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

/// @internal
/// @brief Shared implementation of @ref Scaled (runtime scalar scale).
/// @tparam Derived  CRTP host type (@ref Scaled).
/// @tparam Func     Wrapped VectorFunction type.
/// @endinternal
template <class Derived, class Func>
struct Scaled_Impl : VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);
    /// @brief The wrapped function whose output is scaled.
    Func func_;
    /// @brief Runtime scalar factor applied to the output and all derivatives.
    double scale_value_ = 1.0;

    /// @brief Input-domain sparsity inherited from the wrapped function.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    /// @brief Linearity inherited from the wrapped function (scaling is linear).
    static constexpr bool is_linear_function = Func::is_linear_function;
    /// @brief Vectorizability inherited from the wrapped function.
    static constexpr bool is_vectorizable = Func::is_vectorizable;

    /// @brief Constructs an empty wrapper (configured later).
    Scaled_Impl() {}
    /// @brief Constructs a scalar-scaled wrapper around a function.
    /// @param f  Function to wrap.
    /// @param s  Scalar factor applied to the output.
    Scaled_Impl(Func f, double s) : func_(std::move(f)), scale_value_(s) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }

    /// @brief Returns whether the wrapped function is currently linear.
    /// @return True if the wrapped function reports itself linear.
    bool is_linear() const { return func_.is_linear(); }

    /// @internal
    /// @brief Evaluate the wrapped function and scale its value by the factor.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->func_.compute(x, fx_);
        fx *= Scalar(this->scale_value_);
    }
    /// @internal
    /// @brief Evaluate value and Jacobian, scaling both by the factor.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        this->func_.compute_jacobian(x, fx_, jx_);
        fx *= Scalar(this->scale_value_);
        this->func_.scale_jacobian(jx, Scalar(this->scale_value_));
    }
    /// @internal
    /// @brief Evaluate value, Jacobian, adjoint gradient, and adjoint Hessian, scaled.
    ///
    /// The adjoint variables are pre-scaled so the wrapped function's reverse
    /// pass yields already-scaled gradient and Hessian; the value and Jacobian
    /// are then scaled directly.
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @param  adjgrad_ Adjoint-gradient accumulator (size = `input_rows()`).
    /// @param  adjhess_ Adjoint-Hessian accumulator (`input_rows()` square).
    /// @param  adjvars  Adjoint variables seeding the reverse pass.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &adjv_scaled) {
            adjv_scaled = adjvars * Scalar(this->scale_value_);
            this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                        adjhess_, adjv_scaled);
            fx *= Scalar(this->scale_value_);
            this->func_.scale_jacobian(jx_, Scalar(this->scale_value_));
        };

        const int orows = this->func_.output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Forward a right-Jacobian product to the wrapped function.
    ///
    /// Resolves the left operand to a dense or diagonal reference before
    /// delegating, so a diagonal scale matrix is multiplied efficiently.
    ///
    /// @tparam Target      Concrete Eigen target buffer type.
    /// @tparam Left        Concrete Eigen left-operand type.
    /// @tparam Right       Concrete Eigen right-operand type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @tparam Aliased     True when the operands alias the target.
    /// @param  target_     Destination buffer for the product.
    /// @param  left        Left operand of the product.
    /// @param  right       Right operand of the product.
    /// @param  assign      Assignment policy instance.
    /// @param  aliased     Aliasing flag instance.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            CMatRef<Right> right_ref(right.derived());
            CDiagRef<Left> left_ref(left.derived());
            this->func_.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        } else {
            CMatRef<Right> right_ref(right.derived());
            CMatRef<Left> left_ref(left.derived());
            this->func_.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        }
    }
    /// @internal
    /// @brief Forward a symmetric Jacobian product to the wrapped function.
    /// @tparam Target      Concrete Eigen target buffer type.
    /// @tparam Left        Concrete Eigen left-operand type.
    /// @tparam Right       Concrete Eigen right-operand type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @tparam Aliased     True when the operands alias the target.
    /// @param  target_     Destination buffer for the product.
    /// @param  left        Left operand of the product.
    /// @param  right       Right operand of the product.
    /// @param  assign      Assignment policy instance.
    /// @param  aliased     Aliasing flag instance.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                          CEigRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func_.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal
    /// @brief Forward a Jacobian accumulation to the wrapped function.
    /// @tparam Target      Concrete Eigen target buffer type.
    /// @tparam JacType     Concrete Eigen Jacobian source type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @param  target_     Destination buffer.
    /// @param  right       Jacobian contribution to accumulate.
    /// @param  assign      Assignment policy instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_jacobian(target_, right, assign);
    }
    /// @internal
    /// @brief Forward a gradient accumulation to the wrapped function.
    /// @tparam Target      Concrete Eigen target buffer type.
    /// @tparam JacType     Concrete Eigen gradient source type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @param  target_     Destination buffer.
    /// @param  right       Gradient contribution to accumulate.
    /// @param  assign      Assignment policy instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_gradient(target_, right, assign);
    }
    /// @internal
    /// @brief Forward a Hessian accumulation to the wrapped function.
    /// @tparam Target      Concrete Eigen target buffer type.
    /// @tparam JacType     Concrete Eigen Hessian source type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @param  target_     Destination buffer.
    /// @param  right       Hessian contribution to accumulate.
    /// @param  assign      Assignment policy instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        this->func_.accumulate_hessian(target_, right, assign);
    }
    /// @internal
    /// @brief Forward a scalar Jacobian scaling to the wrapped function.
    /// @tparam Target  Concrete Eigen target buffer type.
    /// @tparam Scalar  Scalar factor type.
    /// @param  target_ Jacobian buffer to scale in place.
    /// @param  s       Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_jacobian(target_, s);
    }
    /// @internal
    /// @brief Forward a scalar gradient scaling to the wrapped function.
    /// @tparam Target  Concrete Eigen target buffer type.
    /// @tparam Scalar  Scalar factor type.
    /// @param  target_ Gradient buffer to scale in place.
    /// @param  s       Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_gradient(target_, s);
    }
    /// @internal
    /// @brief Forward a scalar Hessian scaling to the wrapped function.
    /// @tparam Target  Concrete Eigen target buffer type.
    /// @tparam Scalar  Scalar factor type.
    /// @param  target_ Hessian buffer to scale in place.
    /// @param  s       Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_hessian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_hessian(target_, s);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

/// @internal
/// @brief Shared implementation of @ref RowScaled (per-row vector scale).
/// @tparam Derived  CRTP host type (@ref RowScaled).
/// @tparam Func     Wrapped VectorFunction type.
/// @endinternal
template <class Derived, class Func>
struct RowScaled_Impl
    : VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);
    /// @brief The wrapped function whose output rows are scaled.
    Func func_;
    /// @brief Per-row output scale factors.
    Output<double> row_scale_values_;
    /// @brief Linearity inherited from the wrapped function (scaling is linear).
    static constexpr bool is_linear_function = Func::is_linear_function;
    /// @brief Vectorizability inherited from the wrapped function.
    static constexpr bool is_vectorizable = Func::is_vectorizable;
    /// @brief Input-domain sparsity inherited from the wrapped function.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    /// @brief Constructs an empty wrapper with unit row scales.
    RowScaled_Impl() { this->row_scale_values_.setOnes(); }

    /// @brief Constructs a row-scaled wrapper around a function.
    /// @tparam OutType  Eigen vector expression type of the scale factors.
    /// @param f  Function to wrap.
    /// @param s  Per-row output scale factors.
    template <class OutType> RowScaled_Impl(Func f, CVecRef<OutType> s) : func_(std::move(f)) {
        this->row_scale_values_ = s;
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }
    /// @brief Returns whether the wrapped function is currently linear.
    /// @return True if the wrapped function reports itself linear.
    bool is_linear() const { return func_.is_linear(); }

    /// @internal
    /// @brief Evaluates the wrapped function and scales each output row.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector receiving the row-scaled result.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &scales) {
            scales = this->row_scale_values_.template cast<Scalar>();
            this->func_.compute(x, fx_);
            fx = fx.cwiseProduct(scales);
        };

        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }
    /// @internal
    /// @brief Evaluates value and Jacobian with per-row scaling applied.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector receiving the row-scaled result.
    /// @param jx_  Jacobian buffer receiving the row-scaled Jacobian.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &scales) {
            scales = this->row_scale_values_.template cast<Scalar>();

            this->func_.compute_jacobian(x, fx_, jx_);
            fx = fx.cwiseProduct(scales);
            this->func_.right_jacobian_product(jx, scales.asDiagonal(), jx, DirectAssignment(),
                                               std::bool_constant<true>());
        };

        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }
    /// @internal
    /// @brief Evaluates value, Jacobian, adjoint gradient, and Hessian, row-scaled.
    ///
    /// The adjoint variables are pre-scaled per row so the returned adjoint
    /// quantities are consistent with the row-scaled output.
    /// @tparam InType       Eigen input vector expression type.
    /// @tparam OutType      Eigen output vector expression type.
    /// @tparam JacType      Eigen Jacobian matrix expression type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector expression type.
    /// @param x        Input vector.
    /// @param fx_      Output vector receiving the row-scaled result.
    /// @param jx_      Jacobian buffer receiving the row-scaled Jacobian.
    /// @param adjgrad_ Adjoint-gradient buffer.
    /// @param adjhess_ Adjoint-Hessian buffer.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        auto Impl = [&](auto &scales, auto &adjv_scaled) {
            scales = this->row_scale_values_.template cast<Scalar>();
            adjv_scaled = adjvars.cwiseProduct(scales);

            this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                        adjhess_, adjv_scaled);
            fx = fx.cwiseProduct(scales);
            this->func_.right_jacobian_product(jx, scales.asDiagonal(), jx, DirectAssignment(),
                                               std::bool_constant<true>());
        };

        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  tycho::utils::TempSpec<Output<Scalar>>(orows, 1),
                                                  tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Forwards a right-Jacobian product to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Left Left-operand type.
    /// @tparam Right  Eigen right-operand type. @tparam Assignment Assignment tag.
    /// @tparam Aliased Whether operands may alias the target.
    /// @param target_ Destination matrix. @param left Left operand.
    /// @param right Right operand. @param assign Assignment policy.
    /// @param aliased Aliasing flag. @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {

        if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            CMatRef<Right> right_ref(right.derived());
            CDiagRef<Left> left_ref(left.derived());
            this->func_.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        } else {
            CMatRef<Right> right_ref(right.derived());
            CMatRef<Left> left_ref(left.derived());
            this->func_.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        }
    }
    /// @internal
    /// @brief Forwards a symmetric-Jacobian product to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Left Left-operand type.
    /// @tparam Right  Eigen right-operand type. @tparam Assignment Assignment tag.
    /// @tparam Aliased Whether operands may alias the target.
    /// @param target_ Destination matrix. @param left Left operand.
    /// @param right Right operand. @param assign Assignment policy.
    /// @param aliased Aliasing flag. @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                          CEigRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func_.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal
    /// @brief Forwards a Jacobian accumulation to the wrapped function.
    /// @tparam Target Eigen target type. @tparam JacType Source type.
    /// @tparam Assignment Assignment tag. @param target_ Destination.
    /// @param right Source contribution. @param assign Assignment policy. @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_jacobian(target_, right, assign);
    }
    /// @internal
    /// @brief Forwards a gradient accumulation to the wrapped function.
    /// @tparam Target Eigen target type. @tparam JacType Source type.
    /// @tparam Assignment Assignment tag. @param target_ Destination.
    /// @param right Source contribution. @param assign Assignment policy. @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_gradient(target_, right, assign);
    }
    /// @internal
    /// @brief Forwards a Hessian accumulation to the wrapped function.
    /// @tparam Target Eigen target type. @tparam JacType Source type.
    /// @tparam Assignment Assignment tag. @param target_ Destination.
    /// @param right Source contribution. @param assign Assignment policy. @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        this->func_.accumulate_hessian(target_, right, assign);
    }
    /// @internal
    /// @brief Forwards a Jacobian scaling to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Scalar Scalar factor type.
    /// @param target_ Jacobian to scale. @param s Scalar factor. @endinternal
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_jacobian(target_, s);
    }
    /// @internal
    /// @brief Forwards a gradient scaling to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Scalar Scalar factor type.
    /// @param target_ Gradient to scale. @param s Scalar factor. @endinternal
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_gradient(target_, s);
    }
    /// @internal
    /// @brief Forwards a Hessian scaling to the wrapped function.
    /// @tparam Target Eigen target type. @tparam Scalar Scalar factor type.
    /// @param target_ Hessian to scale. @param s Scalar factor. @endinternal
    template <class Target, class Scalar>
    inline void scale_hessian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_hessian(target_, s);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

/// @internal
/// @brief Shared implementation of @ref MatrixScaled (constant matrix product).
/// @tparam Derived  CRTP host type (@ref MatrixScaled).
/// @tparam Func     Wrapped VectorFunction type.
/// @tparam MRows    Compile-time row count of the matrix-scaled output.
/// @endinternal
template <class Derived, class Func, int MRows>
struct MatrixScaled_Impl
    : VectorFunction<Derived, Func::IRC, MRows, DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base = VectorFunction<Derived, Func::IRC, MRows, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    /// @brief The wrapped function whose output is matrix-multiplied.
    Func func_;

    /// @brief Scaling-matrix type for a given scalar (rows = MRows, cols = Func::ORC).
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using MatType = Eigen::Matrix<Scalar, MRows, Func::ORC>;

    /// @brief The constant left-multiplying matrix.
    MatType<double> mat;
    /// @brief True when the matrix is square, allowing in-place evaluation.
    bool NoTemp = false;

    /// @brief Linearity inherited from the wrapped function (scaling is linear).
    static constexpr bool is_linear_function = Func::is_linear_function;
    /// @brief Vectorizability inherited from the wrapped function.
    static constexpr bool is_vectorizable = Func::is_vectorizable;
    /// @brief Input-domain sparsity inherited from the wrapped function.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    /// @brief Constructs an empty wrapper (configured later).
    MatrixScaled_Impl() {}

    /// @brief Constructs a matrix-scaled wrapper around a function.
    /// @tparam OutType  Eigen expression type providing the scaling matrix.
    /// @param f  Function to wrap.
    /// @param s  Scaling matrix; its column count must equal `f.output_rows()`.
    template <class OutType> MatrixScaled_Impl(Func f, CVecRef<OutType> s) : func_(std::move(f)) {
        this->mat = s;
        this->set_io_rows(this->func_.input_rows(), this->mat.rows());
        this->set_input_domain(this->input_rows(), {this->func_.input_domain()});

        if (mat.cols() != func_.output_rows()) {
            throw std::invalid_argument(
                "Matrix Must have same number of cols as RHS vector function.");
        }

        if (mat.rows() == mat.cols()) {
            NoTemp = true;
        }
    }
    /// @brief Returns whether the wrapped function is currently linear.
    /// @return True if the wrapped function reports itself linear.
    bool is_linear() const { return this->func_.is_linear(); }

    /// @internal
    /// @brief Evaluates the wrapped function and left-multiplies by the matrix.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector receiving the matrix-scaled result.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        if (NoTemp) {
            auto Impl = [&](auto &mattmp) {
                this->func_.compute(x, fx_);
                mattmp = this->mat.template cast<Scalar>();
                fx = (mattmp * fx).eval();
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(),
                                                              this->func_.output_rows()));
        } else {

            auto Impl = [&](auto &mattmp, auto &fxtmp) {
                this->func_.compute(x, fxtmp);
                mattmp = this->mat.template cast<Scalar>();
                fx = (mattmp * fxtmp);
            };

            tycho::utils::BumpAllocator::allocate_run(
                Impl,
                tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(),
                                                        this->func_.output_rows()),
                tycho::utils::TempSpec<typename Func::template Output<Scalar>>(
                    this->func_.output_rows(), 1));
        }
    }
    /// @internal
    /// @brief Evaluates value and Jacobian, both left-multiplied by the matrix.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector receiving the matrix-scaled result.
    /// @param jx_  Jacobian buffer receiving the matrix-scaled Jacobian.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        if (NoTemp) {
            auto Impl = [&](auto &mattmp) {
                this->func_.compute_jacobian(x, fx_, jx_);
                mattmp = this->mat.template cast<Scalar>();
                fx = (mattmp * fx).eval();
                this->func_.right_jacobian_product(jx, mattmp, jx, DirectAssignment(),
                                                   std::bool_constant<true>());
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(),
                                                              this->func_.output_rows()));
        } else {

            auto Impl = [&](auto &mattmp, auto &fxtmp, auto &jxtmp) {
                this->func_.compute_jacobian(x, fxtmp, jxtmp);
                mattmp = this->mat.template cast<Scalar>();
                fx = (mattmp * fxtmp);
                this->func_.right_jacobian_product(jx, mattmp, jxtmp, DirectAssignment(),
                                                   std::bool_constant<false>());
            };

            tycho::utils::BumpAllocator::allocate_run(
                Impl,
                tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(),
                                                        this->func_.output_rows()),
                tycho::utils::TempSpec<typename Func::template Output<Scalar>>(
                    this->func_.output_rows(), 1),
                tycho::utils::TempSpec<typename Func::template Jacobian<Scalar>>(
                    this->func_.output_rows(), this->func_.input_rows()));
        }
    }
    /// @internal
    /// @brief Evaluates value, Jacobian, adjoint gradient, and Hessian (matrix-scaled).
    ///
    /// The adjoint variables are pre-multiplied by the matrix transpose so the
    /// returned adjoint quantities are consistent with the matrix-scaled output.
    /// @tparam InType       Eigen input vector expression type.
    /// @tparam OutType      Eigen output vector expression type.
    /// @tparam JacType      Eigen Jacobian matrix expression type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector expression type.
    /// @param x        Input vector.
    /// @param fx_      Output vector receiving the matrix-scaled result.
    /// @param jx_      Jacobian buffer receiving the matrix-scaled Jacobian.
    /// @param adjgrad_ Adjoint-gradient buffer.
    /// @param adjhess_ Adjoint-Hessian buffer.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        if (NoTemp) {
            auto Impl = [&](auto &mattmp, auto &adjv_scaled) {
                mattmp = this->mat.template cast<Scalar>();
                adjv_scaled = (adjvars.transpose() * mattmp).transpose();
                this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                            adjhess_, adjv_scaled);

                fx = (mattmp * fx).eval();
                this->func_.right_jacobian_product(jx, mattmp, jx, DirectAssignment(),
                                                   std::bool_constant<true>());
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl,
                tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(),
                                                        this->func_.output_rows()),
                tycho::utils::TempSpec<typename Func::template Output<Scalar>>(
                    this->func_.output_rows(), 1)

            );
        } else {

            auto Impl = [&](auto &mattmp, auto &fxtmp, auto &jxtmp, auto &adjv_scaled) {
                mattmp = this->mat.template cast<Scalar>();
                adjv_scaled = (adjvars.transpose() * mattmp).transpose();
                this->func_.compute_jacobian_adjointgradient_adjointhessian(
                    x, fxtmp, jxtmp, adjgrad_, adjhess_, adjv_scaled);
                fx = (mattmp * fxtmp);
                this->func_.right_jacobian_product(jx, mattmp, jxtmp, DirectAssignment(),
                                                   std::bool_constant<false>());
            };

            tycho::utils::BumpAllocator::allocate_run(
                Impl,
                tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(),
                                                        this->func_.output_rows()),
                tycho::utils::TempSpec<typename Func::template Output<Scalar>>(
                    this->func_.output_rows(), 1),
                tycho::utils::TempSpec<typename Func::template Jacobian<Scalar>>(
                    this->func_.output_rows(), this->func_.input_rows()),
                tycho::utils::TempSpec<typename Func::template Output<Scalar>>(
                    this->func_.output_rows(), 1));
        }
    }
};

} // namespace tycho::vf
