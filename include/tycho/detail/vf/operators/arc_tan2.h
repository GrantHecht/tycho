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

/// @ingroup vf
/// @brief Two-input VectorFunction computing @f$\operatorname{atan2}(y, x)@f$ with analytic
/// derivatives.
///
/// Takes a 2-vector @f$(y, x)@f$ and returns the quadrant-aware arctangent as a
/// scalar output. Both first and second derivatives are analytic.
struct ArcTan2Op : VectorFunction<ArcTan2Op, 2, 1, DenseDerivativeMode::Analytic,
                                  DenseDerivativeMode::Analytic> {

    /// @brief Convenience alias for the VectorFunction CRTP base.
    using Base = VectorFunction<ArcTan2Op, 2, 1, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);
    static constexpr bool is_vectorizable = true; ///< @c atan2 supports SuperScalar evaluation.

    /// @internal
    /// @brief Compute @f$\operatorname{atan2}(yy, xx)@f$, looping per lane for SuperScalar types.
    /// @tparam Scalar  Arithmetic scalar type (plain double or a SuperScalar pack).
    /// @param yy  Numerator (the @f$y@f$ argument).
    /// @param xx  Denominator (the @f$x@f$ argument).
    /// @return The quadrant-aware arctangent @f$\operatorname{atan2}(yy, xx)@f$.
    /// @endinternal
    template <class Scalar> static Scalar calc_arc_tan2(Scalar yy, Scalar xx) {
        Scalar fx;
        if constexpr (Is_SuperScalar<Scalar>::value) {
            for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
                fx[i] = atan2(yy[i], xx[i]);
            }
        } else {
            fx = atan2(yy, xx);
        }
        return fx;
    }

    /// @internal
    /// @brief Evaluate @f$\operatorname{atan2}(x_0, x_1)@f$ into @p fx_.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input 2-vector @f$(y, x)@f$.
    /// @param fx_  Output scalar (1-vector) to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        Scalar yy = x[0];
        Scalar xx = x[1];
        fx[0] = calc_arc_tan2(yy, xx);
    }
    /// @internal
    /// @brief Evaluate @f$\operatorname{atan2}@f$ and its analytic Jacobian.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input 2-vector @f$(y, x)@f$.
    /// @param fx_  Output scalar (1-vector) to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        Scalar yy = x[0];
        Scalar xx = x[1];
        fx[0] = calc_arc_tan2(yy, xx);

        Scalar denom = xx * xx + yy * yy;

        jx(0, 0) = xx / denom;
        jx(0, 1) = -yy / denom;
    }
    /// @internal
    /// @brief Evaluate @f$\operatorname{atan2}@f$, its Jacobian, adjoint gradient, and adjoint
    /// Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input 2-vector @f$(y, x)@f$.
    /// @param fx_      Output scalar (1-vector) to write.
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
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        Scalar yy = x[0];
        Scalar xx = x[1];
        fx[0] = calc_arc_tan2(yy, xx);

        Scalar denom = xx * xx + yy * yy;

        jx(0, 0) = xx / denom;
        jx(0, 1) = -yy / denom;

        adjgrad[0] = adjvars[0] * xx / denom;
        adjgrad[1] = -adjvars[0] * yy / denom;

        adjhess(0, 0) = -2 * xx * yy / (denom * denom);
        adjhess(1, 1) = 2 * xx * yy / (denom * denom);

        adjhess(0, 1) = 1 / denom - 2 * xx * xx / (denom * denom);
        adjhess(1, 0) = -1 / denom + 2 * yy * yy / (denom * denom);

        adjhess *= adjvars[0];
    }
};

/// @internal
/// @brief Builder that composes @ref ArcTan2Op over two scalar operand functions.
/// @tparam YFunc  Numerator operand function (the @f$y@f$ argument).
/// @tparam XFunc  Denominator operand function (the @f$x@f$ argument).
/// @endinternal
template <class YFunc, class XFunc> struct ArcTan2Impl {
    /// @internal
    /// @brief Stack @p yf and @p xf and feed the 2-vector through @ref ArcTan2Op.
    /// @param yf  Numerator operand function.
    /// @param xf  Denominator operand function.
    /// @return A VectorFunction evaluating @f$\operatorname{atan2}(yf, xf)@f$.
    /// @endinternal
    static auto Definition(YFunc yf, XFunc xf) { return ArcTan2Op().eval(StackedOutputs{yf, xf}); }
};

} // namespace tycho::vf