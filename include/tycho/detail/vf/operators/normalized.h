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

//! Declaration of \struct NormalizedPower_Impl
template <class Derived, int IR, int PW> struct NormalizedPower_Impl;

template <int IR> struct Normalized : NormalizedPower_Impl<Normalized<IR>, IR, 1> {
    using Base = NormalizedPower_Impl<Normalized<IR>, IR, 1>;

    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

template <int IR, int PW>
struct NormalizedPower : NormalizedPower_Impl<NormalizedPower<IR, PW>, IR, PW> {
    using Base = NormalizedPower_Impl<NormalizedPower<IR, PW>, IR, PW>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

template <class Derived, int IR, int PW>
struct NormalizedPower_Impl : VectorFunction<Derived, IR, IR> {
    using Base = VectorFunction<Derived, IR, IR>;
    VF_TYPE_ALIASES(Base)

    static constexpr int power = PW;
    static constexpr int pp2 = power + 2;
    static constexpr int pp4 = power + 4;

    using Base::compute;
    static constexpr bool is_vectorizable = true;

    NormalizedPower_Impl() {}
    NormalizedPower_Impl(int irows) { this->set_io_rows(irows, irows); }

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
