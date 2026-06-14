// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines all VectorFunction for computing the root of a Scalar function
// inside of an expression.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

template <class Derived, class FX, class DFX> struct ScalarRootFinder_Impl;

/// @ingroup vf
/// @brief VectorFunction returning the root of a scalar function found by Newton iteration.
///
/// The first input is the scalar iteration variable (initialised to the guess);
/// the remaining inputs are differentiated through. @c DFX optionally supplies the
/// derivative of @c FX with respect to the iteration variable.
/// @tparam FX   Scalar VectorFunction whose root is sought.
/// @tparam DFX  Derivative of @c FX (or @c std::false_type to derive from the Jacobian).
template <class FX, class DFX>
struct ScalarRootFinder : ScalarRootFinder_Impl<ScalarRootFinder<FX, DFX>, FX, DFX> {
    /// @brief Convenience alias for the underlying root-finder implementation.
    using Base = ScalarRootFinder_Impl<ScalarRootFinder<FX, DFX>, FX, DFX>;
    using Base::Base;
};

/// <summary>
/// Defines a Vector Function for computing the root of Scalar Function FX, with derivative DFX.
/// By definition, the first input to FX is the Scalar parameter that we can adjust to find the
/// root, and should be set to your desired initial guess. The rest of the inputs to FX are
/// parameters that the output will be differentiated wrt to. You can also provide a second function
/// scalar DFX that is the derivative of the FX wrt to only the iteration variable. Otherwise it
/// will be computed from the jacobian of FX
/// </summary>
/// <typeparam name="Derived"></typeparam>
/// <typeparam name="FX"></typeparam>
/// <typeparam name="DFX"></typeparam>
/// @internal
/// @brief Shared implementation of @ref ScalarRootFinder.
///
/// Runs Newton iteration on the scalar function @c FX to locate a root, then
/// differentiates the converged root with respect to the remaining inputs via
/// the implicit-function theorem.
/// @tparam Derived  CRTP host type (@ref ScalarRootFinder).
/// @tparam FX       Scalar VectorFunction whose root is sought.
/// @tparam DFX      Derivative of @c FX (or @c std::false_type).
/// @endinternal
template <class Derived, class FX, class DFX>
struct ScalarRootFinder_Impl : VectorFunction<Derived, FX::IRC, 1> {

    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<Derived, FX::IRC, 1>;
    VF_TYPE_ALIASES(Base);

    double tol = 1.0e-9; ///< Convergence tolerance on the residual.
    int MaxIters = 10;   ///< Maximum number of Newton iterations.

    FX fxfunc;   ///< The scalar function whose root is sought.
    DFX dfxfunc; ///< Optional derivative of @c fxfunc w.r.t. the iteration variable.

    /// @brief Default constructor; leaves the iteration parameters at their defaults.
    ScalarRootFinder_Impl() {}

    /// @brief Construct from the function, its derivative, and iteration controls.
    /// @param f     The scalar function whose root is sought.
    /// @param df    Derivative of @p f w.r.t. the iteration variable (or @c std::false_type).
    /// @param iter  Maximum number of Newton iterations.
    /// @param tol   Convergence tolerance on the residual.
    ScalarRootFinder_Impl(FX f, DFX df, int iter, double tol)
        : fxfunc(f), dfxfunc(df), MaxIters(iter), tol(tol) {

        if constexpr (!std::is_same<DFX, std::false_type>::value) {
            if (f.input_rows() != df.input_rows()) {
                throw std::invalid_argument(
                    "Root and First Derivative functions must have same number of Input Rows");
            }
        }
        this->set_io_rows(f.input_rows(), 1);
    }

    /// @internal
    /// @brief Run the Newton iteration in place, leaving the root in @p x[0].
    /// @tparam VecType  Eigen working-vector type.
    /// @tparam JacType  Eigen working-Jacobian type.
    /// @param x   Working vector; @c x[0] is the iteration variable, updated in place.
    /// @param jx  Scratch Jacobian used when no explicit derivative is supplied.
    /// @endinternal
    template <class VecType, class JacType> void find_root(VecType &x, JacType &jx) const {
        typedef typename VecType::Scalar Scalar;
        Vector1<Scalar> fx;
        Vector1<Scalar> dfx;

        for (int i = 0; i < this->MaxIters; i++) {

            if constexpr (!std::is_same<DFX, std::false_type>::value) {
                this->fxfunc.compute(x, fx);
                if (abs(fx[0]) < tol)
                    break;
                this->dfxfunc.compute(x, dfx);
                x[0] -= fx[0] / dfx[0];
                fx[0] = 0;
                dfx[0] = 0;
            } else {
                this->fxfunc.compute_jacobian(x, fx, jx);
                if (abs(fx[0]) < tol)
                    break;
                x[0] -= fx[0] / jx(0, 0);
                fx[0] = 0;
                jx.setZero();
            }
        }
    }

    /// @internal
    /// @brief Evaluate the root-finder, writing the converged root into @p fx_.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input vector (initial guess and parameters).
    /// @param fx_  Output scalar (1-vector) to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        const int irows = this->input_rows();

        auto Impl = [&](auto &xtmp, auto &jtmp) {
            xtmp = x;
            this->find_root(xtmp, jtmp);
            fx[0] = xtmp[0];
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<Scalar>>(irows, 1),
            tycho::utils::TempSpec<Jacobian<Scalar>>(1, irows));
    }

    /// @internal
    /// @brief Evaluate the converged root and its Jacobian via implicit differentiation.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input vector (initial guess and parameters).
    /// @param fx_  Output scalar (1-vector) to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        const int irows = this->input_rows();

        auto Impl = [&](auto &xtmp, auto &jtmp) {
            xtmp = x;
            this->find_root(xtmp, jtmp);

            fx[0] = xtmp[0];
            Vector1<Scalar> fxtmp;
            this->fxfunc.compute_jacobian(xtmp, fxtmp, jx);
            jx /= -jx(0, 0);
            jx(0, 0) = 0;
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<Scalar>>(irows, 1),
            tycho::utils::TempSpec<Jacobian<Scalar>>(1, irows));
    }
    /// @internal
    /// @brief Evaluate the converged root, its Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input vector (initial guess and parameters).
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

        const int irows = this->input_rows();

        auto Impl = [&](auto &xtmp, auto &jtmp, auto &htmp) {
            xtmp = x;
            this->find_root(xtmp, jtmp);

            fx[0] = xtmp[0];
            Vector1<Scalar> fxtmp;
            this->fxfunc.compute_jacobian_adjointgradient_adjointhessian(xtmp, fxtmp, jx, adjgrad,
                                                                         htmp, adjvars);
            htmp /= -jx(0, 0);
            adjhess.noalias() =
                htmp +
                (jx.transpose() * (jx * htmp(0, 0) / jx(0, 0) - htmp.row(0)) - htmp.col(0) * jx) /
                    jx(0, 0);

            jx /= -jx(0, 0);
            jx(0, 0) = 0;
            adjgrad = jx.transpose() * adjvars[0];
            adjhess.row(0).setZero();
            adjhess.col(0).setZero();
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<Scalar>>(irows, 1),
            tycho::utils::TempSpec<Jacobian<Scalar>>(1, irows),
            tycho::utils::TempSpec<Hessian<Scalar>>(irows, irows));
    }
};

} // namespace tycho::vf