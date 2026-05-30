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
/// @brief Elementwise sign of a child function, @f$\operatorname{sgn}(f(x))\in\{-1,0,1\}@f$.
///
/// Evaluates the wrapped function @p func_ and replaces each output component by
/// its sign. The map is piecewise constant, so its Jacobian and Hessian are zero
/// almost everywhere; the derivative methods only recompute the value.
/// @tparam Func  The wrapped child VectorFunction whose output sign is taken.
template <class Func>
struct SignFunction : VectorFunction<SignFunction<Func>, Func::IRC, Func::ORC> {
    /// @brief Convenience alias for the VectorFunction CRTP base.
    using Base = VectorFunction<SignFunction<Func>, Func::IRC, Func::ORC>;

    Func func_; ///< The wrapped child function whose output sign is taken.
    VF_TYPE_ALIASES(Base);

    // using INPUT_DOMAIN = SingleDomain<Func::IRC, 0, 0>;

    /// @brief Default constructor; leaves the wrapped function default-constructed.
    SignFunction() {}
    /// @brief Construct from the child function whose sign is to be taken.
    /// @param func  The wrapped child VectorFunction.
    SignFunction(Func func) : func_(std::move(func)) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        DomainMatrix dmn(2, 1);
        dmn(0, 0) = 0;
        dmn(1, 0) = 0;
        this->set_input_domain(this->input_rows(), {dmn});
    }

    /// @internal
    /// @brief Replace each component of @p fx by its sign in place.
    /// @tparam OutType  Eigen output-vector type.
    /// @param fx  Vector overwritten with the elementwise sign of its entries.
    /// @endinternal
    template <class OutType> void sign_impl(OutType &fx) const {
        typedef typename OutType::Scalar Scalar;
        if constexpr (Is_SuperScalar<Scalar>::value) {
            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] = fx[i].sign();
            }
        } else {
            fx = fx.array().sign();
        }
    }

    /// @internal
    /// @brief Evaluate the child function and overwrite its output with the sign.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        this->func_.compute(x, fx);
        this->sign_impl(fx);
    }
    /// @internal
    /// @brief Evaluate the sign value; the Jacobian is zero and left unchanged.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to write.
    /// @param jx_  Output Jacobian (left unchanged; the sign Jacobian is zero).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        this->func_.compute(x, fx);
        this->sign_impl(fx);
    }
    /// @internal
    /// @brief Evaluate the sign value; Jacobian, gradient, and Hessian are zero and left unchanged.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input vector.
    /// @param fx_      Output vector to write.
    /// @param jx_      Output Jacobian (left unchanged).
    /// @param adjgrad_ Output adjoint gradient (left unchanged).
    /// @param adjhess_ Output adjoint Hessian (left unchanged).
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

        this->func_.compute(x, fx);
        this->sign_impl(fx);
    }
};

} // namespace tycho::vf