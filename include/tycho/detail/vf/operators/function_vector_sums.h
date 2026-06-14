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
#include "tycho/detail/vf/derivatives/detect_diagonal.h"

namespace tycho::vf {

template <class Derived, class Func> struct FunctionVectorSum_Impl;

/// @ingroup vf
/// @brief VectorFunction adding a constant offset vector to a function @f$f(x)+v@f$.
/// @tparam Func  Wrapped VectorFunction whose output the offset is added to.
template <class Func>
struct FunctionPlusVector : FunctionVectorSum_Impl<FunctionPlusVector<Func>, Func> {
    /// @brief Convenience alias for the underlying function-vector-sum implementation.
    using Base = FunctionVectorSum_Impl<FunctionPlusVector<Func>, Func>;
    using Base::Base;
};

/// @internal
/// @brief Shared implementation of function-plus-constant-vector VectorFunctions.
///
/// Forwards all derivative work to the wrapped function and adds a constant
/// offset vector to the value; derivatives are unchanged by the constant.
/// @tparam Derived  CRTP host type (@ref FunctionPlusVector).
/// @tparam Func     Wrapped VectorFunction.
/// @endinternal
template <class Derived, class Func>
struct FunctionVectorSum_Impl
    : VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);
    Func func_; ///< The wrapped VectorFunction.
    Output<double>
        offset_vector; ///< The constant offset vector added to (or subtracted from) the value.
    /// @brief Input-domain descriptor inherited from the wrapped function.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static constexpr bool is_linear_function =
        Func::is_linear_function; ///< Linear iff the wrapped function is linear.
    static constexpr bool is_vectorizable =
        Func::is_vectorizable; ///< Vectorizable iff the wrapped function is.

    /// @brief Default constructor; leaves the wrapped function and offset unset.
    FunctionVectorSum_Impl() {}
    /// @brief Construct from a wrapped function and a constant offset vector.
    /// @tparam OutType  Eigen output-vector type of the offset.
    /// @param f  The VectorFunction to wrap.
    /// @param v  The constant offset vector.
    template <class OutType>
    FunctionVectorSum_Impl(Func f, CVecRef<OutType> v) : func_(std::move(f)) {
        this->offset_vector = v;
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }
    /// @internal
    /// @brief Evaluate the wrapped function and add the constant offset.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->func_.compute(x, fx_);
        fx += this->offset_vector.template cast<Scalar>();
    }
    /// @internal
    /// @brief Evaluate value and Jacobian, adding the constant offset to the value.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        // MatRef<JacType> jx = jx_.const_cast_derived();
        this->func_.compute_jacobian(x, fx_, jx_);
        fx += this->offset_vector.template cast<Scalar>();
    }
    /// @internal
    /// @brief Evaluate value, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input vector.
    /// @param fx_      Output vector to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjhess_ Output adjoint Hessian to write.
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
        // MatRef<JacType> jx = jx_.const_cast_derived();
        // VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        // MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();
        this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                    adjvars);
        fx += this->offset_vector.template cast<Scalar>();
    }

    /// @internal
    /// @brief Forward a right Jacobian product to the wrapped function.
    /// @tparam Target      Eigen target-matrix type.
    /// @tparam Left        Eigen left-operand type.
    /// @tparam Right       Eigen right-operand type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @tparam Aliased     Whether the operands alias the target.
    /// @param target_  Output matrix to accumulate into.
    /// @param left     Left operand of the product.
    /// @param right    Right operand of the product.
    /// @param assign   Assignment-policy tag.
    /// @param aliased  Aliasing flag.
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
    /// @tparam Target      Eigen target-matrix type.
    /// @tparam Left        Eigen left-operand type.
    /// @tparam Right       Eigen right-operand type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @tparam Aliased     Whether the operands alias the target.
    /// @param target_  Output matrix to accumulate into.
    /// @param left     Left operand of the product.
    /// @param right    Right operand of the product.
    /// @param assign   Assignment-policy tag.
    /// @param aliased  Aliasing flag.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                          CEigRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func_.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal
    /// @brief Forward a Jacobian accumulation to the wrapped function.
    /// @tparam Target      Eigen target-matrix type.
    /// @tparam JacType     Eigen Jacobian-matrix type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @param target_  Output matrix to accumulate into.
    /// @param right    Jacobian block to accumulate.
    /// @param assign   Assignment-policy tag.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_jacobian(target_, right, assign);
    }
    /// @internal
    /// @brief Forward a gradient accumulation to the wrapped function.
    /// @tparam Target      Eigen target-matrix type.
    /// @tparam JacType     Eigen gradient block type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @param target_  Output matrix to accumulate into.
    /// @param right    Gradient block to accumulate.
    /// @param assign   Assignment-policy tag.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_gradient(target_, right, assign);
    }
    /// @internal
    /// @brief Forward a Hessian accumulation to the wrapped function.
    /// @tparam Target      Eigen target-matrix type.
    /// @tparam JacType     Eigen Hessian block type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @param target_  Output matrix to accumulate into.
    /// @param right    Hessian block to accumulate.
    /// @param assign   Assignment-policy tag.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        this->func_.accumulate_hessian(target_, right, assign);
    }
    /// @internal
    /// @brief Forward a Jacobian scaling to the wrapped function.
    /// @tparam Target  Eigen target-matrix type.
    /// @tparam Scalar  Arithmetic scalar type.
    /// @param target_  Jacobian block to scale.
    /// @param s        Scaling factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_jacobian(target_, s);
    }
    /// @internal
    /// @brief Forward a gradient scaling to the wrapped function.
    /// @tparam Target  Eigen target-matrix type.
    /// @tparam Scalar  Arithmetic scalar type.
    /// @param target_  Gradient block to scale.
    /// @param s        Scaling factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_gradient(target_, s);
    }
    /// @internal
    /// @brief Forward a Hessian scaling to the wrapped function.
    /// @tparam Target  Eigen target-matrix type.
    /// @tparam Scalar  Arithmetic scalar type.
    /// @param target_  Hessian block to scale.
    /// @param s        Scaling factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_hessian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_hessian(target_, s);
    }
};
} // namespace tycho::vf
