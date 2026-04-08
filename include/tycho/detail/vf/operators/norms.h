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

template <int IR> struct Norm : IntegralNorm_Impl<Norm<IR>, IR, 1> {
    using Base = IntegralNorm_Impl<Norm<IR>, IR, 1>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
template <int IR> struct SquaredNorm : IntegralNorm_Impl<SquaredNorm<IR>, IR, 2> {
    using Base = IntegralNorm_Impl<SquaredNorm<IR>, IR, 2>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
template <int IR> struct InverseNorm : IntegralNorm_Impl<InverseNorm<IR>, IR, -1> {
    using Base = IntegralNorm_Impl<InverseNorm<IR>, IR, -1>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
template <int IR> struct InverseSquaredNorm : IntegralNorm_Impl<InverseSquaredNorm<IR>, IR, -2> {
    using Base = IntegralNorm_Impl<InverseSquaredNorm<IR>, IR, -2>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
template <int IR, int PW> struct NormPower : IntegralNorm_Impl<NormPower<IR, PW>, IR, PW> {
    using Base = IntegralNorm_Impl<NormPower<IR, PW>, IR, PW>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};
template <int IR, int PW>
struct InverseNormPower : IntegralNorm_Impl<InverseNormPower<IR, PW>, IR, -PW> {
    using Base = IntegralNorm_Impl<InverseNormPower<IR, PW>, IR, -PW>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

template <class Derived, int USZ, int Power>
struct IntegralNorm_Impl : VectorFunction<Derived, USZ, 1> {
    using Base = VectorFunction<Derived, USZ, 1>;
    VF_TYPE_ALIASES(Base);
    using Base::compute;

    static constexpr int power = Power;
    static constexpr int pp2 = power - 2;
    static constexpr int pp4 = power - 4;
    static constexpr int ppm2 = power * (power - 2);
    // double IntegralNorm_Scale = 1.0;
    static constexpr bool is_vectorizable = true;

    IntegralNorm_Impl() {}
    IntegralNorm_Impl(int ir) { this->set_io_rows(ir, 1); }

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
