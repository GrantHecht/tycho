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

template <class Derived, int USZ, int Power> struct IntegralNorm_Impl;

/// @ingroup vf
/// @brief Euclidean norm VectorFunction @f$\|x\|_2=\sqrt{x^T x}@f$.
///
/// Scalar-output (1×1) expression that computes the @f$\ell_2@f$ length of its
/// input vector with analytic first and second derivatives. Produced in C++ via
/// `f.norm()`; in Python via `f.norm()` or the free function `vf.norm(f)`.
/// @tparam IR  Compile-time input row count (Eigen::Dynamic for runtime size).
template <int IR> struct Norm : IntegralNorm_Impl<Norm<IR>, IR, 1> {
    /// @brief Convenience alias for the underlying integral-power-norm implementation.
    using Base = IntegralNorm_Impl<Norm<IR>, IR, 1>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
/// @ingroup vf
/// @brief Squared Euclidean norm VectorFunction @f$\|x\|_2^2=x^T x@f$.
///
/// Scalar-output expression returning the sum of squares of the input entries,
/// avoiding the square-root of @ref Norm and therefore producing simpler
/// derivatives for unconstrained-magnitude objectives. Produced in C++ via
/// `f.squared_norm()`; in Python via `f.squared_norm()` or `vf.squared_norm(f)`.
/// @tparam IR  Compile-time input row count (Eigen::Dynamic for runtime size).
template <int IR> struct SquaredNorm : IntegralNorm_Impl<SquaredNorm<IR>, IR, 2> {
    /// @brief Convenience alias for the underlying integral-power-norm implementation.
    using Base = IntegralNorm_Impl<SquaredNorm<IR>, IR, 2>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
/// @ingroup vf
/// @brief Reciprocal Euclidean norm VectorFunction @f$\|x\|_2^{-1}@f$.
///
/// Scalar-output expression computing @f$1/\|x\|_2@f$, which arises frequently
/// in gravitational and electrostatic potential terms. Produced in C++ via
/// `f.inverse_norm()`; in Python via `f.inverse_norm()` or `vf.inverse_norm(f)`.
/// @tparam IR  Compile-time input row count (Eigen::Dynamic for runtime size).
template <int IR> struct InverseNorm : IntegralNorm_Impl<InverseNorm<IR>, IR, -1> {
    /// @brief Convenience alias for the underlying integral-power-norm implementation.
    using Base = IntegralNorm_Impl<InverseNorm<IR>, IR, -1>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
/// @ingroup vf
/// @brief Reciprocal squared Euclidean norm VectorFunction @f$\|x\|_2^{-2}@f$.
///
/// Scalar-output expression computing @f$1/\|x\|_2^2@f$; the @f$r^{-2}@f$
/// factor common in gravitational force/acceleration magnitude and inverse-square-law constraints.
/// Produced in C++ via `f.inverse_squared_norm()`; in Python via
/// `f.inverse_squared_norm()` or `vf.inverse_squared_norm(f)`.
/// @tparam IR  Compile-time input row count (Eigen::Dynamic for runtime size).
template <int IR> struct InverseSquaredNorm : IntegralNorm_Impl<InverseSquaredNorm<IR>, IR, -2> {
    /// @brief Convenience alias for the underlying integral-power-norm implementation.
    using Base = IntegralNorm_Impl<InverseSquaredNorm<IR>, IR, -2>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
/// @ingroup vf
/// @brief Integer power of the Euclidean norm VectorFunction @f$\|x\|_2^{PW}@f$.
///
/// General scalar-output expression for any compile-time integer exponent; common
/// values (±1 through ±4) are specialised for speed. Produced in C++ via
/// `f.norm_power<PW>()`; Python exposes `cubed_norm` (`f.cubed_norm()`) as the
/// only named positive-power shorthand — higher powers use the C++ member directly.
/// @tparam IR  Compile-time input row count (Eigen::Dynamic for runtime size).
/// @tparam PW  Integer power applied to the norm.
template <int IR, int PW> struct NormPower : IntegralNorm_Impl<NormPower<IR, PW>, IR, PW> {
    /// @brief Convenience alias for the underlying integral-power-norm implementation.
    using Base = IntegralNorm_Impl<NormPower<IR, PW>, IR, PW>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
/// @ingroup vf
/// @brief Reciprocal integer power of the Euclidean norm VectorFunction @f$\|x\|_2^{-PW}@f$.
///
/// Scalar-output expression computing @f$\|x\|_2^{-PW}@f$ for any compile-time
/// positive integer @p PW; the @f$r^{-3}@f$ factor (PW=3) appears, for instance,
/// in the gravitational acceleration vector @f$\mu r / \|r\|^3@f$. Produced in
/// C++ via `f.inverse_norm_power<PW>()`; Python exposes `inverse_cubed_norm` and
/// `inverse_four_norm` as named shorthands.
/// @tparam IR  Compile-time input row count (Eigen::Dynamic for runtime size).
/// @tparam PW  Integer power whose negation is applied to the norm.
template <int IR, int PW>
struct InverseNormPower : IntegralNorm_Impl<InverseNormPower<IR, PW>, IR, -PW> {
    /// @brief Convenience alias for the underlying integral-power-norm implementation.
    using Base = IntegralNorm_Impl<InverseNormPower<IR, PW>, IR, -PW>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @internal
/// @brief Shared implementation of all integer-power-of-norm VectorFunctions.
///
/// Computes @f$\|x\|_2^{Power}@f$ with analytic first and second derivatives,
/// specialising small integer powers for speed.
/// @tparam Derived  CRTP host type (one of @ref Norm, @ref SquaredNorm, etc.).
/// @tparam USZ      Compile-time input row count.
/// @tparam Power    Integer power applied to the Euclidean norm.
/// @endinternal
template <class Derived, int USZ, int Power>
struct IntegralNorm_Impl : VectorFunction<Derived, USZ, 1> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<Derived, USZ, 1>;
    VF_TYPE_ALIASES(Base);
    using Base::compute;

    static constexpr int power = Power;              ///< The integer power applied to the norm.
    static constexpr int pp2 = power - 2;            ///< Helper exponent @f$Power-2@f$.
    static constexpr int pp4 = power - 4;            ///< Helper exponent @f$Power-4@f$.
    static constexpr int ppm2 = power * (power - 2); ///< Helper coefficient @f$Power(Power-2)@f$.
    // double IntegralNorm_Scale = 1.0;
    static constexpr bool is_vectorizable = true; ///< Norm powers support SuperScalar evaluation.

    /// @brief Default constructor; leaves the row count unset.
    IntegralNorm_Impl() {}
    /// @brief Construct with a runtime input row count.
    /// @param ir  Number of input rows.
    IntegralNorm_Impl(int ir) { this->set_io_rows(ir, 1); }

    /// @internal
    /// @brief Compute @f$n^{Power}@f$, specialising small integer powers.
    /// @tparam Scalar  Arithmetic scalar type.
    /// @param n  The Euclidean norm value to raise to @c Power.
    /// @return @f$n^{Power}@f$.
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
        else if constexpr (power == -1)
            pow_n = Scalar(1.0) / n;
        else if constexpr (power == -2)
            pow_n = Scalar(1.0) / (n * n);
        else if constexpr (power == -3)
            pow_n = Scalar(1.0) / (n * n * n);
        else if constexpr (power == -4)
            pow_n = Scalar(1.0) / (n * n * n * n);
        else
            pow_n = pow(n, power);
        return pow_n;
    }

    /// @internal
    /// @brief Evaluate @f$\|x\|_2^{Power}@f$ into @p fx_.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input vector.
    /// @param fx_  Output scalar (1-vector) to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        // using std::pow;
        using std::sqrt;
        Scalar n;
        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            n = sqrt(x.dot(x));
        } else {
            n = x.norm();
        }
        Scalar pow_n = this->calc_pow_n(n);

        fx[0] = pow_n;
    }

    /// @internal
    /// @brief Evaluate @f$\|x\|_2^{Power}@f$ and its Jacobian.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input vector.
    /// @param fx_  Output scalar (1-vector) to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        // using std::pow;

        using std::sqrt;
        Scalar n;
        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            n = sqrt(x.dot(x));
        } else {
            n = x.norm();
        }
        Scalar pow_n = this->calc_pow_n(n);

        Scalar np = pow_n;
        Scalar npd = power * np / (n * n);
        fx[0] = np;
        jx = npd * x.transpose();
    }

    /// @internal
    /// @brief Evaluate @f$\|x\|_2^{Power}@f$, its Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input vector.
    /// @param fx_      Output scalar (1-vector) to write.
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

        using std::sqrt;
        Scalar n;
        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
            n = sqrt(x.dot(x));
        } else {
            n = x.norm();
        }

        Scalar pow_n = this->calc_pow_n(n);

        Scalar np = pow_n;
        Scalar npd = power * np / (n * n);
        Scalar npdd = adjvars[0] * ppm2 * np / (n * n * n * n);

        fx[0] = np;
        jx = npd * x.transpose();
        adjgrad = jx.transpose() * adjvars[0];
        adjhess.diagonal().setConstant(npd * adjvars[0]);

        if constexpr (InType::RowsAtCompileTime == Eigen::Dynamic) {
            for (int i = 0; i < this->input_rows(); i++) {
                adjhess.col(i) += x * (npdd * x[i]);
            }
        } else {
            adjhess.noalias() += (x * npdd) * x.transpose();
        }
    }
};

} // namespace tycho::vf
