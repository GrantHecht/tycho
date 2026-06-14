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

/// @cond INTERNAL
//! Declaration of \struct NormalizedPower_Impl
template <class Derived, int IR, int PW> struct NormalizedPower_Impl;
/// @endcond

/// @ingroup vf
/// @brief Vector-normalizing VectorFunction @f$x/\|x\|_2@f$ returning the unit direction.
///
/// Maps an @f$n@f$-vector to the unit vector pointing in the same direction;
/// output size equals input size. Analytic Jacobian and adjoint Hessian are
/// provided. Produced in C++ via `f.normalized()`; in Python via `f.normalized()`
/// or `vf.normalized(f)`.
/// @tparam IR  Compile-time input/output row count (Eigen::Dynamic for runtime size).
template <int IR> struct Normalized : NormalizedPower_Impl<Normalized<IR>, IR, 1> {
    /// @brief Convenience alias for the underlying normalized-power implementation.
    using Base = NormalizedPower_Impl<Normalized<IR>, IR, 1>;

    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @ingroup vf
/// @brief Vector divided by an integer power of its norm, @f$x/\|x\|_2^{PW}@f$.
///
/// Generalises @ref Normalized to arbitrary positive integer denominators; for
/// example @f$PW=3@f$ yields the @f$x/\|x\|_2^3@f$ factor found in the
/// two-body gravitational acceleration. Produced in C++ via
/// `f.normalized_power<PW>()`; Python exposes fixed-power shorthands, namely
/// `f.normalized_power2()` through `f.normalized_power5()`.
/// @tparam IR  Compile-time input/output row count (Eigen::Dynamic for runtime size).
/// @tparam PW  Integer power of the norm in the denominator.
template <int IR, int PW>
struct NormalizedPower : NormalizedPower_Impl<NormalizedPower<IR, PW>, IR, PW> {
    /// @brief Convenience alias for the underlying normalized-power implementation.
    using Base = NormalizedPower_Impl<NormalizedPower<IR, PW>, IR, PW>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @internal
/// @brief Shared implementation of @ref Normalized and @ref NormalizedPower.
///
/// Computes @f$x/\|x\|_2^{PW}@f$ with analytic first and second derivatives,
/// specialising small integer powers for speed.
/// @tparam Derived  CRTP host type (@ref Normalized or @ref NormalizedPower).
/// @tparam IR       Compile-time input/output row count.
/// @tparam PW       Integer power of the norm in the denominator.
/// @endinternal
template <class Derived, int IR, int PW>
struct NormalizedPower_Impl : VectorFunction<Derived, IR, IR> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<Derived, IR, IR>;
    VF_TYPE_ALIASES(Base)

    static constexpr int power = PW;      ///< The integer power applied to the norm.
    static constexpr int pp2 = power + 2; ///< Helper exponent @f$PW+2@f$.
    static constexpr int pp4 = power + 4; ///< Helper exponent @f$PW+4@f$.

    using Base::compute;
    static constexpr bool is_vectorizable =
        true; ///< Normalised powers support SuperScalar evaluation.

    /// @brief Default constructor; leaves the row count unset.
    NormalizedPower_Impl() {}
    /// @brief Construct with a runtime input/output row count.
    /// @param irows  Number of input (and output) rows.
    NormalizedPower_Impl(int irows) { this->set_io_rows(irows, irows); }

    /// @internal
    /// @brief Compute @f$n^{PW}@f$, specialising small integer powers.
    /// @tparam Scalar  Arithmetic scalar type.
    /// @param n  The Euclidean norm value to raise to @c power.
    /// @return @f$n^{PW}@f$.
    /// @endinternal
    template <class Scalar> inline Scalar calc_pow_n(Scalar n) const {
        Scalar pow_n;
        if constexpr (power == 1)
            pow_n = n;
        else if constexpr (power == 2)
            pow_n = n * n;
        else if constexpr (power == 3)
            pow_n = n * n * n;
        else if constexpr (power == 4)
            pow_n = n * n * n * n;
        else if constexpr (power == 5)
            pow_n = n * n * n * n * n;
        else
            pow_n = pow(n, power);
        return pow_n;
    }

    /// @internal
    /// @brief Evaluate @f$x/\|x\|_2^{PW}@f$ into @p fx_.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        using std::sqrt;
        Scalar n;
        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            n = sqrt(x.dot(x));
        } else {
            n = x.norm();
        }
        Scalar pow_n = this->calc_pow_n(n);

        Scalar np = 1.0 / (pow_n);
        fx = x * np;
    }
    /// @internal
    /// @brief Evaluate @f$x/\|x\|_2^{PW}@f$ and its Jacobian.
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
        MatRef<JacType> jx = jx_.const_cast_derived();

        using std::sqrt;
        Scalar n;
        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            n = sqrt(x.dot(x));
        } else {
            n = x.norm();
        }

        Scalar pow_n = this->calc_pow_n(n);

        Scalar pow_n_2 = pow_n * n * n;

        Scalar np = 1.0 / (pow_n);
        Scalar npd = -power / pow_n_2;
        fx = x * (np);

        jx.diagonal().setConstant(np);

        if constexpr (InType::RowsAtCompileTime == Eigen::Dynamic) {
            for (int i = 0; i < this->input_rows(); i++) {
                jx.col(i) += x * (npd * x[i]);
            }
        } else {
            jx.noalias() += (x * npd) * x.transpose();
        }
    }

    //! DenseDerivativeMode::Analytic second derivative
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

        const int irows = this->input_rows();
        auto Impl = [&](auto &xxt) {
            using std::sqrt;
            Scalar n;
            if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
                n = sqrt(x.dot(x));
            } else {
                n = x.norm();
            }

            Scalar pow_n = this->calc_pow_n(n);

            Scalar pow_n_2 = pow_n * n * n;
            Scalar pow_n_4 = pow_n_2 * n * n;

            Scalar np = 1.0 / (pow_n);
            Scalar npd = -power / pow_n_2;
            Scalar npdd = power * pp2 / pow_n_4;
            Scalar K = x.dot(adjvars);

            if constexpr (InType::RowsAtCompileTime == Eigen::Dynamic) {
                for (int i = 0; i < irows; i++) {
                    xxt.col(i) = x * x[i];
                }
            } else {
                xxt.noalias() = x * x.transpose();
            }

            fx = x * (np);

            jx.diagonal().setConstant((np));
            jx += xxt * (npd);

            adjgrad.noalias() = (adjvars.transpose() * jx).transpose();
            adjhess.diagonal().setConstant(npd);
            adjhess.noalias() += xxt * npdd;
            adjhess *= K;

            if constexpr (InType::RowsAtCompileTime == Eigen::Dynamic) {
                for (int i = 0; i < irows; i++) {
                    adjhess.col(i) += (x * adjvars[i] + adjvars * x[i]) * npd;
                }
            } else {
                adjhess.noalias() += (x * adjvars.transpose() + adjvars * x.transpose()) * npd;
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Jacobian<Scalar>>(irows, irows));
    }
};

} // namespace tycho::vf
