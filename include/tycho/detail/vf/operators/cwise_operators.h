// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines all Vector Functions which simply apply a common
// coefficient (Cwise) operation/functio to all elements of input vectors.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @brief Forward declaration of the shared base for element-wise function operators.
/// @tparam Derived  CRTP-derived Cwise operator type.
/// @tparam Func     Wrapped operand function the operation is applied to.
/// @ingroup vf
template <class Derived, class Func> struct CwiseFunctionOperator;

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Element-wise sine of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseSin : CwiseFunctionOperator<CwiseSin<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseSin<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().sin();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CVecRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().sin();
        jx = x.array().cos();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CVecRef<JacType> jx_, CVecRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();
        VecRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().sin();
        hx = -fx;
        jx = x.array().cos();
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Element-wise cosine of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseCos : CwiseFunctionOperator<CwiseCos<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseCos<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().cos();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().cos();
        jx.derived() = -x.array().sin();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().cos();
        hx.derived() = -fx;
        jx.derived() = -x.array().sin();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Element-wise tangent of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseTan : CwiseFunctionOperator<CwiseTan<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseTan<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().tan();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().tan();
        jx.derived() = x.array().cos().square().inverse();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().tan();
        jx.derived() = x.array().cos().square().inverse();
        hx.derived() = fx.derived().cwiseProduct(jx.derived()) * Scalar(2.0);
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Element-wise arcsine of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseArcSin : CwiseFunctionOperator<CwiseArcSin<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseArcSin<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);
    /// @brief Disables SIMD SuperScalar evaluation for this operator.
    static constexpr bool is_vectorizable = false;

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().asin();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().asin();
        jx.derived().setOnes();
        jx.derived() = jx.derived() - x.cwiseProduct(x);
        jx.derived() = (jx.derived().cwiseSqrt()).cwiseInverse();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().asin();
        jx.derived().setOnes();
        jx.derived() = jx.derived() - x.cwiseProduct(x);
        jx.derived() = (jx.derived().cwiseSqrt()).cwiseInverse();
        hx.derived() = (jx.derived().array().cube()) * x.array();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Element-wise arccosine of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseArcCos : CwiseFunctionOperator<CwiseArcCos<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseArcCos<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);
    /// @brief Disables SIMD SuperScalar evaluation for this operator.
    static constexpr bool is_vectorizable = false;

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().acos();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().acos();
        jx.derived().setOnes();
        jx.derived() = jx.derived() - x.cwiseProduct(x);
        jx.derived() = -(jx.derived().cwiseSqrt()).cwiseInverse();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().acos();
        jx.derived().setOnes();
        jx.derived() = jx.derived() - x.cwiseProduct(x);

        jx.derived() = -(jx.derived().cwiseSqrt()).cwiseInverse();
        hx.derived() = (jx.derived().array().cube()) * x.array();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Element-wise arctangent of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseArcTan : CwiseFunctionOperator<CwiseArcTan<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseArcTan<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);
    /// @brief Enables SIMD SuperScalar evaluation for this operator.
    static constexpr bool is_vectorizable = true;

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().atan();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().atan();
        jx.derived().setOnes();
        jx.derived() = (jx.derived() + x.cwiseProduct(x)).cwiseInverse();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().atan();
        jx.derived().setOnes();
        jx.derived() = (jx.derived() + x.cwiseProduct(x)).cwiseInverse();
        hx.derived() = (jx.derived().array().square()) * x.array() * Scalar(-2.0);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Element-wise square of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseSquare : CwiseFunctionOperator<CwiseSquare<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseSquare<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().square();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().square();
        jx.derived() = Scalar(2.0) * x;
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().square();
        jx.derived() = Scalar(2.0) * x;
        hx.derived().setConstant(Scalar(2.0));
    }
};

/// @brief Element-wise reciprocal of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseInverse : CwiseFunctionOperator<CwiseInverse<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseInverse<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().inverse();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().inverse();
        jx.derived() = -fx.cwiseProduct(fx);
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().inverse();
        jx.derived() = -fx.cwiseProduct(fx);
        hx.derived() = -Scalar(2.0) * jx.derived().cwiseProduct(fx);
    }
};

/// @brief Element-wise exponential of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseExp : CwiseFunctionOperator<CwiseExp<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseExp<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().exp();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().exp();
        jx.derived() = fx;
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().exp();
        jx.derived() = fx;
        hx.derived() = fx;
    }
};

/// @brief Element-wise natural logarithm of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseLog : CwiseFunctionOperator<CwiseLog<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseLog<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().log();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().log();
        jx.derived() = x.cwiseInverse();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().log();
        jx.derived() = x.cwiseInverse();
        hx.derived() = -(jx.derived().array().square());
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Element-wise hyperbolic sine of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseSinH : CwiseFunctionOperator<CwiseSinH<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseSinH<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().sinh();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CVecRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().sinh();
        jx = x.array().cosh();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CVecRef<JacType> jx_, CVecRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();
        VecRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().sinh();
        hx = fx;
        jx = x.array().cosh();
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Element-wise hyperbolic cosine of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseCosH : CwiseFunctionOperator<CwiseCosH<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseCosH<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().cosh();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().cosh();
        jx.derived() = x.array().sinh();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().cosh();
        hx.derived() = fx;
        jx.derived() = x.array().sinh();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Element-wise hyperbolic tangent of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseTanH : CwiseFunctionOperator<CwiseTanH<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseTanH<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().tanh();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().tanh();
        jx.derived().setOnes();
        jx.derived() = jx.derived().array() - fx.derived().array().square();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().tanh();
        jx.derived().setOnes();
        jx.derived() = jx.derived().array() - fx.derived().array().square();
        hx.derived() = Scalar(-2.0) * (fx.derived().cwiseProduct(jx.derived()));
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Element-wise inverse hyperbolic sine of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseArcSinH : CwiseFunctionOperator<CwiseArcSinH<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseArcSinH<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().asinh();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CVecRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();

        typedef typename InType::Scalar Scalar;

        fx = x.array().asinh();
        jx.setOnes();
        jx += x.cwiseProduct(x);

        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            if constexpr (Base::ORC == 1)
                jx[0] = sqrt(jx[0]);
            else {
                const int sz = x.size();
                for (int i = 0; i < sz; i++)
                    jx[i] = sqrt(jx[i]);
            }
        } else {
            jx = jx.cwiseSqrt();
        }

        jx = jx.cwiseInverse();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CVecRef<JacType> jx_, CVecRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();
        VecRef<HessType> hx = hx_.const_cast_derived();

        typedef typename InType::Scalar Scalar;

        fx = x.array().asinh();
        jx.setOnes();
        jx += x.cwiseProduct(x);

        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            if constexpr (Base::ORC == 1)
                jx[0] = sqrt(jx[0]);
            else {
                const int sz = x.size();
                for (int i = 0; i < sz; i++)
                    jx[i] = sqrt(jx[i]);
            }
        } else {
            jx = jx.cwiseSqrt();
        }

        jx = jx.cwiseInverse();
        hx = Scalar(-1.0) * jx.array().cube() * x.array();
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Element-wise inverse hyperbolic cosine of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseArcCosH : CwiseFunctionOperator<CwiseArcCosH<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseArcCosH<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().acosh();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CVecRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();

        typedef typename InType::Scalar Scalar;

        fx = x.array().acosh();
        jx.setOnes();
        jx *= Scalar(-1.0);
        jx += x.cwiseProduct(x);

        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            if constexpr (Base::ORC == 1)
                jx[0] = sqrt(jx[0]);
            else {
                const int sz = x.size();
                for (int i = 0; i < sz; i++)
                    jx[i] = sqrt(jx[i]);
            }
        } else {
            jx = jx.cwiseSqrt();
        }

        jx = jx.cwiseInverse();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CVecRef<JacType> jx_, CVecRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();
        VecRef<HessType> hx = hx_.const_cast_derived();

        typedef typename InType::Scalar Scalar;

        fx = x.array().acosh();
        jx.setOnes();
        jx *= Scalar(-1.0);
        jx += x.cwiseProduct(x);

        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            if constexpr (Base::ORC == 1)
                jx[0] = sqrt(jx[0]);
            else {
                const int sz = x.size();
                for (int i = 0; i < sz; i++)
                    jx[i] = sqrt(jx[i]);
            }
        } else {
            jx = jx.cwiseSqrt();
        }

        jx = jx.cwiseInverse();
        hx = Scalar(-1.0) * jx.array().cube() * x.array();
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Element-wise inverse hyperbolic tangent of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseArcTanH : CwiseFunctionOperator<CwiseArcTanH<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseArcTanH<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().atanh();
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CVecRef<JacType> jx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();

        typedef typename InType::Scalar Scalar;

        fx = x.array().atanh();
        jx.setOnes();
        jx -= x.cwiseProduct(x);
        jx = jx.cwiseInverse();
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CVecRef<JacType> jx_, CVecRef<HessType> hx_) {
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();
        VecRef<HessType> hx = hx_.const_cast_derived();

        typedef typename InType::Scalar Scalar;

        fx = x.array().atanh();
        jx.setOnes();
        jx -= x.cwiseProduct(x);
        jx = jx.cwiseInverse();

        hx = Scalar(2.0) * jx.array().square() * x.array();
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Element-wise square root of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseSqrt : CwiseFunctionOperator<CwiseSqrt<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseSqrt<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Enables SIMD SuperScalar evaluation for this operator.
    static constexpr bool is_vectorizable = true;

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    static void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            if constexpr (Base::ORC == 1)
                fx[0] = sqrt(x[0]);
            else {
                const int sz = x.size();
                for (int i = 0; i < sz; i++)
                    fx[i] = sqrt(x[i]);
            }
        } else {
            fx = x.array().sqrt();
        }
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                       CEigRef<JacType> jx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            if constexpr (Base::ORC == 1)
                fx[0] = sqrt(x[0]);
            else {
                const int sz = x.size();
                for (int i = 0; i < sz; i++)
                    fx[i] = sqrt(x[i]);
            }
        } else {
            fx = x.array().sqrt();
        }
        jx.derived() = fx.array().inverse() * Scalar(0.5);
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                               CEigRef<JacType> jx_, CEigRef<HessType> hx_) {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();

        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            if constexpr (Base::ORC == 1)
                fx[0] = sqrt(x[0]);
            else {
                const int sz = x.size();
                for (int i = 0; i < sz; i++)
                    fx[i] = sqrt(x[i]);
            }
        } else {
            fx = x.array().sqrt();
        }

        jx.derived() = fx.array().inverse() * Scalar(0.5);
        hx.derived() = Scalar(-0.5) * jx.derived().array() / x.array();
    }
};

/// @brief Element-wise power of a VectorFunction, @f$ x^p @f$.
///
/// The exponent @p power is a runtime double supplied at construction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwisePow : CwiseFunctionOperator<CwisePow<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwisePow<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Enables SIMD SuperScalar evaluation for this operator.
    static constexpr bool is_vectorizable = true;
    /// @brief Runtime exponent applied element-wise.
    double power = 1;

    /// @brief Construct from an operand function and an exponent.
    /// @param f      Operand function to wrap.
    /// @param power  Exponent applied element-wise.
    CwisePow(Func f, double power) : Base(f), power(power) {}
    /// @brief Default constructor; leaves the operand default-constructed and exponent at 1.
    CwisePow() {}
    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().pow(Scalar(this->power));
    }
    /// @brief Apply the element-wise operation and its first derivative (diagonal Jacobian).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                CEigRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        fx = x.array().pow(Scalar(this->power));
        jx.derived() = Scalar(this->power) * x.array().pow(Scalar(this->power - 1.0));
    }
    /// @brief Apply the element-wise operation with its first and second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                        CEigRef<JacType> jx_, CEigRef<HessType> hx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        EigRef<JacType> jx = jx_.const_cast_derived();
        EigRef<HessType> hx = hx_.const_cast_derived();
        fx = x.array().pow(Scalar(this->power));
        jx.derived() = Scalar(this->power) * x.array().pow(Scalar(this->power - 1.0));
        hx.derived() =
            Scalar(this->power * (this->power - 1.0)) * x.array().pow(Scalar(this->power - 2.0));
        ;
    }
};

/// @brief Element-wise absolute value of a VectorFunction.
/// @tparam Func  Wrapped operand function.
/// @ingroup vf
template <class Func> struct CwiseAbs : CwiseFunctionOperator<CwiseAbs<Func>, Func> {
    /// @brief Shared base implementing the operator's compute/derivative dispatch.
    using Base = CwiseFunctionOperator<CwiseAbs<Func>, Func>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);

    /// @brief Apply the element-wise operation to the operand output.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    template <class InType, class OutType>
    void cwise_compute(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        fx = x.cwiseAbs();
    }
    /// @brief Apply the element-wise operation and its first derivative (the sign function).
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    template <class InType, class OutType, class JacType>
    void cwise_compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                CEigRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();

        fx = x.cwiseAbs();
        if constexpr (Is_SuperScalar<Scalar>::value) {
            for (int i = 0; i < x.size(); i++) {
                jx[i] = x[i].sign();
            }
        } else {
            jx = x.array().sign();
        }
    }
    /// @brief Apply the element-wise operation with its first and (zero) second derivatives.
    /// @tparam InType   Eigen type of the operand output.
    /// @tparam OutType  Eigen type of this operator's output.
    /// @tparam JacType  Eigen type of the per-element first-derivative vector.
    /// @tparam HessType Eigen type of the per-element second-derivative vector.
    /// @param x    Operand output values.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Reference receiving the per-element first derivatives.
    /// @param hx_  Reference receiving the per-element second derivatives.
    template <class InType, class OutType, class JacType, class HessType>
    void cwise_compute_jacobian_hessian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                        CEigRef<JacType> jx_, CEigRef<HessType> hx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        VecRef<JacType> jx = jx_.const_cast_derived();
        VecRef<HessType> hx = hx_.const_cast_derived();

        fx = x.cwiseAbs();
        if constexpr (Is_SuperScalar<Scalar>::value) {
            for (int i = 0; i < x.size(); i++) {
                jx[i] = x[i].sign();
                // hx[i] = jx[i];
            }
        } else {
            jx = x.array().sign();
            // hx = jx;
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Shared base for element-wise function operators that wrap one operand.
///
/// Defines the operations common to all Cwise operators: it composes a wrapped
/// operand function @p Func with a per-element transform, providing compute and
/// derivative methods for any derived class that implements the static
/// `cwise_compute`, `cwise_compute_jacobian`, and `cwise_compute_jacobian_hessian`
/// hooks. Derivatives are propagated by chaining the wrapped operand's Jacobian
/// with the diagonal per-element derivative supplied by the derived class.
/// @tparam Derived  CRTP-derived Cwise operator type.
/// @tparam Func     Wrapped operand function the operation is applied to.
/// @ingroup vf
template <class Derived, class Func>
struct CwiseFunctionOperator : VectorFunction<Derived, Func::IRC, Func::ORC> {
    /// @brief VectorFunction base inheriting the operand's input and output sizes.
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    /// @internal
    /// @brief Input domain inherited from the wrapped operand function.
    /// @endinternal
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    /// @internal
    /// @brief The wrapped operand function the element-wise transform is applied to.
    /// @endinternal
    Func func_;
    /// @internal
    /// @brief True when the wrapped operand is SIMD-vectorizable.
    /// @endinternal
    static constexpr bool is_vectorizable = Func::is_vectorizable;
    /// @internal
    /// @brief Default constructor; leaves the operand default-constructed.
    /// @endinternal
    CwiseFunctionOperator() {}
    /// @internal
    /// @brief Construct from an operand function and configure I/O rows and input domain.
    /// @param f  Operand function to wrap.
    /// @endinternal
    CwiseFunctionOperator(Func f) : func_(std::move(f)) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }

    /// @internal
    /// @brief Evaluate the operand and apply the derived class's element-wise transform.
    /// @tparam InType   Eigen type of the input vector.
    /// @tparam OutType  Eigen type of the output.
    /// @param x    Input vector.
    /// @param fx_  Output reference receiving the transformed values.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fxt;

        if constexpr (Func::OutputIsDynamic) {
            fxt.resize(this->func_.output_rows());
        }

        this->func_.compute(x, fxt);
        this->derived().cwise_compute(fxt, fx_);
    }
    /// @internal
    /// @brief Evaluate the transform and its Jacobian via the chain rule.
    /// @tparam InType   Eigen type of the input vector.
    /// @tparam OutType  Eigen type of the output.
    /// @tparam JacType  Eigen type of the Jacobian.
    /// @param x    Input vector.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Jacobian reference.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;

        Output<Scalar> fxt;
        Output<Scalar> jxdiag;

        if constexpr (Func::OutputIsDynamic) {
            fxt.resize(this->func_.output_rows());
            jxdiag.resize(this->func_.output_rows());
        }

        this->func_.compute_jacobian(x, fxt, jx_);
        this->derived().cwise_compute_jacobian(fxt, fx_, jxdiag);
        if constexpr (Func::ORC == 1) {
            this->func_.right_jacobian_product(jx_, jxdiag, jx_, DirectAssignment(),
                                               std::bool_constant<true>());
        } else {
            this->func_.right_jacobian_product(jx_, jxdiag.asDiagonal(), jx_, DirectAssignment(),
                                               std::bool_constant<true>());
        }
    }
    /// @internal
    /// @brief Evaluate the transform, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen type of the input vector.
    /// @tparam OutType      Eigen type of the output.
    /// @tparam JacType      Eigen type of the Jacobian.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient.
    /// @tparam AdjHessType  Eigen type of the adjoint Hessian.
    /// @tparam AdjVarType   Eigen type of the adjoint (dual) variables.
    /// @param x        Input vector.
    /// @param fx_      Output reference receiving the transformed values.
    /// @param jx_      Jacobian reference.
    /// @param adjgrad_ Adjoint gradient reference.
    /// @param adjhess_ Adjoint Hessian reference.
    /// @param adjvars  Adjoint (dual) variables seeding the reverse pass.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        MatRef<JacType> jx = jx_.const_cast_derived();
        // VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        Output<Scalar> fxt;
        Output<Scalar> jxdiag;
        Output<Scalar> hxdiag;

        if constexpr (Func::OutputIsDynamic) {
            fxt.resize(this->func_.output_rows());
            jxdiag.resize(this->func_.output_rows());
            hxdiag.resize(this->func_.output_rows());
        }

        this->func_.compute(x, fxt);
        this->derived().cwise_compute_jacobian_hessian(fxt, fx_, jxdiag, hxdiag);

        fxt.setZero();
        Output<Scalar> adjtemp = jxdiag.cwiseProduct(adjvars);
        hxdiag = hxdiag.cwiseProduct(adjvars);
        this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fxt, jx_, adjgrad_, adjhess_,
                                                                    adjtemp);
        if constexpr (Func::ORC == 1) {

            this->func_.symetric_jacobian_product(adjhess, hxdiag, jx, PlusEqualsAssignment(),
                                                  std::bool_constant<false>());

            this->func_.right_jacobian_product(jx_, jxdiag, jx, DirectAssignment(),
                                               std::bool_constant<true>());
        } else {
            this->func_.symetric_jacobian_product(adjhess, hxdiag.asDiagonal(), jx,
                                                  PlusEqualsAssignment(),
                                                  std::bool_constant<false>());
            this->func_.right_jacobian_product(jx_, jxdiag.asDiagonal(), jx, DirectAssignment(),
                                               std::bool_constant<true>());
        }
    }
};

/// @brief Base for square element-wise operators that act directly on the input vector.
///
/// Unlike CwiseFunctionOperator (which wraps another function), this base applies the
/// derived class's `cwise_compute*` hooks straight to the input, producing a square
/// (IR-by-IR) Jacobian whose only non-zeros lie on the diagonal.
/// @tparam Derived  CRTP-derived operator type.
/// @tparam IR       Compile-time input and output rows.
/// @ingroup vf
template <class Derived, int IR> struct CwiseOperator : VectorFunction<Derived, IR, IR> {
    /// @brief Square VectorFunction base with equal input and output sizes.
    using Base = VectorFunction<Derived, IR, IR>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    /// @internal
    /// @brief Apply the derived class's element-wise transform to the input directly.
    /// @tparam InType   Eigen type of the input vector.
    /// @tparam OutType  Eigen type of the output.
    /// @param x    Input vector.
    /// @param fx_  Output reference receiving the transformed values.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        this->derived().cwise_compute(x, fx_);
    }
    /// @internal
    /// @brief Apply the transform and write its derivative onto the Jacobian diagonal.
    /// @tparam InType   Eigen type of the input vector.
    /// @tparam OutType  Eigen type of the output.
    /// @tparam JacType  Eigen type of the Jacobian.
    /// @param x    Input vector.
    /// @param fx_  Output reference receiving the transformed values.
    /// @param jx_  Jacobian reference (diagonal populated).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                 CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        MatRef<JacType> jx = jx_.const_cast_derived();
        this->derived().cwise_compute_jacobian(x, fx_, jx.diagonal());
    }
    /// @internal
    /// @brief Apply the transform with diagonal Jacobian, adjoint gradient, and Hessian.
    /// @tparam InType       Eigen type of the input vector.
    /// @tparam OutType      Eigen type of the output.
    /// @tparam JacType      Eigen type of the Jacobian.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient.
    /// @tparam AdjHessType  Eigen type of the adjoint Hessian.
    /// @tparam AdjVarType   Eigen type of the adjoint (dual) variables.
    /// @param x        Input vector.
    /// @param fx_      Output reference receiving the transformed values.
    /// @param jx_      Jacobian reference (diagonal populated).
    /// @param adjgrad_ Adjoint gradient reference.
    /// @param adjhess_ Adjoint Hessian reference (diagonal populated).
    /// @param adjvars  Adjoint (dual) variables seeding the reverse pass.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian(CVecRef<InType> x,
                                                                CVecRef<OutType> fx_,
                                                                CMatRef<JacType> jx_,
                                                                CVecRef<AdjGradType> adjgrad_,
                                                                CMatRef<AdjHessType> adjhess_,
                                                                CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();
        this->derived().cwise_compute_jacobian_hessian(x, fx_, jx.diagonal(), adjhess.diagonal());
        adjgrad = jx.diagonal().cwiseProduct(adjvars);
        adjhess.diagonal() = adjhess.diagonal().cwiseProduct(adjvars);
    }
};

} // namespace tycho::vf
