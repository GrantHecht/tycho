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

#include "tycho/detail/optimal_control/transcription/lgl_coeffs.h"
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

/// @ingroup optimal_control
/// @brief Linear constraint fixing one interior node at a given cardinal spacing.
///
/// Scalar-output equality constraint enforcing that the middle of three node
/// times sits at the prescribed fractional spacing within the interval (used to
/// pin mesh-node positions). Linear, so its Hessian vanishes.
struct SingleMeshSpacing : VectorFunction<SingleMeshSpacing, 3, 1> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<SingleMeshSpacing, 3, 1>;
    /// @brief Output-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    /// @brief Input-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    /// @brief Jacobian type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>;
    /// @brief Hessian type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Hessian = typename Base::template Hessian<Scalar>;

    double cardinal_spacing_; ///< Target fractional spacing of the interior node.
    double scale_ = 1.0;      ///< Constraint output scale.
    static constexpr bool IsLinearFunction = true; ///< The constraint is linear.

    /// @brief Default constructor; leaves the spacing unset.
    SingleMeshSpacing() {}
    /// @brief Construct with a target cardinal spacing.
    /// @param cs  Target fractional spacing in @f$[0,1]@f$.
    SingleMeshSpacing(double cs) { cardinal_spacing_ = cs; }

    /// @brief Set the target cardinal spacing.
    /// @param cs  Target fractional spacing in @f$[0,1]@f$.
    void set_spacing(double cs) { cardinal_spacing_ = cs; }
    /// @internal
    /// @brief Evaluate the spacing residual.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input @c [t0, t_mid, t1].
    /// @param fx_  Output residual to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Scalar h = x[2] - x[0];
        fx[0] = cardinal_spacing_ * h - (x[1] - x[0]);
        fx[0] *= scale_;
    }

    /// @internal
    /// @brief Evaluate the spacing residual and its Jacobian.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input @c [t0, t_mid, t1].
    /// @param fx_  Output residual to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Scalar h = x[2] - x[0];
        fx[0] = cardinal_spacing_ * h - (x[1] - x[0]);
        fx[0] *= scale_;

        jx(0, 0) = (1.0 - cardinal_spacing_);
        jx(0, 1) = -1.0;
        jx(0, 2) = cardinal_spacing_;
        jx *= scale_;
    }

    /// @internal
    /// @brief Evaluate the residual, Jacobian, and adjoint gradient.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input node times.
    /// @param fx_      Output residual to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        this->compute_jacobian(x, fx_, jx_);
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        adjgrad = (adjvars.transpose() * jx_).transpose();
    }

    /// @internal
    /// @brief Evaluate the residual, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input node times.
    /// @param fx_      Output residual to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjhess_ Output adjoint Hessian to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();

        Scalar h = x[2] - x[0];
        fx[0] = cardinal_spacing_ * h - (x[1] - x[0]);
        fx[0] *= scale_;

        jx(0, 0) = (1.0 - cardinal_spacing_);
        jx(0, 1) = -1.0;
        jx(0, 2) = cardinal_spacing_;
        jx *= scale_;
        adjgrad = (adjvars.transpose() * jx_).transpose();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @ingroup optimal_control
/// @brief Constraint pinning all interior LGL nodes to their cardinal spacings.
///
/// Equality constraint VectorFunction enforcing that each interior cardinal node
/// of an LGL interval sits at its prescribed fractional spacing, given the full
/// set of @c CSC node times.
/// @tparam CSC  Number of cardinal states per interval.
template <int CSC> struct LGLMeshSpacing : VectorFunction<LGLMeshSpacing<CSC>, CSC, CSC - 2> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<LGLMeshSpacing<CSC>, CSC, CSC - 2>;
    /// @brief Output-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    /// @brief Input-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    /// @brief Jacobian type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>;
    /// @brief Hessian type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Hessian = typename Base::template Hessian<Scalar>;
    using Coeffs = LGLCoeffs<CSC>; ///< @brief The LGL coefficient table for this order.

    /// @internal
    /// @brief Evaluate the per-interior-node spacing residuals.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input node times of the interval.
    /// @param fx_  Output residual vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute(const Eigen::MatrixBase<InType> &x,
                        Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Scalar h = x[CSC - 1] - x[0];
        for (int i = 0; i < (CSC - 2); i++) {
            fx[i] = Coeffs::CardinalSpacings[i + 1] - (x[1 + i] - x[0]) / h;
        }
    }

    /// @internal
    /// @brief Evaluate the spacing residuals and their Jacobian.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input node times of the interval.
    /// @param fx_  Output residual vector to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Scalar h = x[CSC - 1] - x[0];
        for (int i = 0; i < (CSC - 2); i++) {
            fx[i] = Coeffs::CardinalSpacings[i + 1] - (x[1 + i] - x[0]) / h;
            jx(i, i + 1) = -1.0 / h;
            jx(i, 0) = 1.0 / h - (x[1 + i] - x[0]) / (h * h);
            jx(i, CSC - 1) = (x[1 + i] - x[0]) / (h * h);
        }
    }

    /// @internal
    /// @brief Evaluate the residual, Jacobian, and adjoint gradient.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input node times.
    /// @param fx_      Output residual to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        this->compute_jacobian(x, fx_, jx_);
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        adjgrad = (adjvars.transpose() * jx_).transpose();
    }

    /// @internal
    /// @brief Evaluate the residual, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input node times.
    /// @param fx_      Output residual to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjhess_ Output adjoint Hessian to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

        Scalar h = x[CSC - 1] - x[0];
        Scalar h2 = h * h;
        Scalar h3 = h2 * h;
        for (int i = 0; i < (CSC - 2); i++) {
            fx[i] = Coeffs::CardinalSpacings[i + 1] - (x[1 + i] - x[0]) / h;
            jx(i, i + 1) = -1.0 / h;
            jx(i, 0) = 1.0 / h - (x[1 + i] - x[0]) / (h2);
            jx(i, CSC - 1) = (x[1 + i] - x[0]) / (h2);
            adjhess(0, i + 1) += -adjvars[i] * 1.0 / (h2);
            adjhess(i + 1, 0) += -adjvars[i] * 1.0 / (h2);

            adjhess(CSC - 1, i + 1) += adjvars[i] * 1.0 / (h2);
            adjhess(i + 1, CSC - 1) += adjvars[i] * 1.0 / (h2);

            adjhess(0, 0) += adjvars[i] * 2.0 / (h2)-adjvars[i] * 2.0 * (x[1 + i] - x[0]) / (h3);
            adjhess(CSC - 1, CSC - 1) += -adjvars[i] * 2.0 * (x[1 + i] - x[0]) / (h3);

            adjhess(0, CSC - 1) += adjvars[i] * (2.0 * (x[1 + i] - x[0]) / (h3)-1.0 / h2);
            adjhess(CSC - 1, 0) += adjvars[i] * (2.0 * (x[1 + i] - x[0]) / (h3)-1.0 / h2);
        }
        adjgrad = (adjvars.transpose() * jx_).transpose();
    }
};

} // namespace tycho::oc
