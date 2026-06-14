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

/// @file
/// @brief Constant-valued VectorFunction primitive.
#pragma once

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @ingroup vf
/// @brief A VectorFunction returning a fixed vector for any input.
///
/// The output value is independent of the input, so the Jacobian and adjoint
/// Hessian are identically zero. The function still carries an input dimension
/// so it can be composed with input-sized expressions.
/// @tparam IR  Compile-time input row count (Eigen::Dynamic for runtime size).
/// @tparam OR  Compile-time output row count (Eigen::Dynamic for runtime size).
template <int IR, int OR> struct Constant : VectorFunction<Constant<IR, OR>, IR, OR> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base = VectorFunction<Constant<IR, OR>, IR, OR>;
    VF_TYPE_ALIASES(Base)

    /// @brief Marks the function as linear (a constant is trivially affine).
    static constexpr bool is_linear_function = true;
    /// @brief Marks the function as SuperScalar-vectorizable.
    static constexpr bool is_vectorizable = true;

    /// @brief The fixed output vector returned for every input.
    Output<double> value_;

    /// @brief Constructs a constant function with a given input size and value.
    /// @param ir   Number of input rows.
    /// @param val  Fixed output vector; its size sets the output dimension.
    Constant(int ir, Output<double> val) {
        this->set_io_rows(ir, val.size());
        value_ = val;

        DomainMatrix dmn(2, 1);
        dmn(0, 0) = 0;
        dmn(1, 0) = 0;
        this->set_input_domain(this->input_rows(), {dmn});
    }

    /// @brief Constructs an empty constant function (sizes set later).
    Constant() {}

    /// @internal
    /// @brief Writes the fixed value into the output, ignoring the input.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param x    Input vector (unused).
    /// @param fx_  Output vector to fill with the constant value.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value_.template cast<Scalar>();
    }
    /// @internal
    /// @brief Computes value and Jacobian; the Jacobian is identically zero.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param x    Input vector (unused).
    /// @param fx_  Output vector to fill with the constant value.
    /// @param jx_  Jacobian buffer; not written (a constant's Jacobian is zero).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value_.template cast<Scalar>();
    }

    /// @internal
    /// @brief Computes value, Jacobian, adjoint gradient, and adjoint Hessian.
    ///
    /// For a constant function the Jacobian, adjoint gradient, and adjoint
    /// Hessian are all zero, so only the value is written.
    /// @tparam InType       Eigen input vector expression type.
    /// @tparam OutType      Eigen output vector expression type.
    /// @tparam JacType      Eigen Jacobian matrix expression type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector expression type.
    /// @param x        Input vector (unused).
    /// @param fx_      Output vector to fill with the constant value.
    /// @param jx_      Jacobian buffer; not written (zero for a constant).
    /// @param adjgrad_ Adjoint-gradient buffer; not written (zero for a constant).
    /// @param adjhess_ Adjoint-Hessian buffer; not written (zero for a constant).
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector (unused).
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value_.template cast<Scalar>();
    }
};

} // namespace tycho::vf
