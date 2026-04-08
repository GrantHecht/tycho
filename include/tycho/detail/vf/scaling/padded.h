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

template <int St> struct UpperPadHolder {
    static constexpr int u_pad_ = St;
    UpperPadHolder(int st) {};
    UpperPadHolder() {};
};

template <> struct UpperPadHolder<-1> {
    int u_pad_ = 0;
    UpperPadHolder(int st) { u_pad_ = st; };
    UpperPadHolder() {};
};

template <class Func, int UP, int LP>
struct PaddedOutput
    : VectorFunction<PaddedOutput<Func, UP, LP>, Func::IRC, SZ_SUM<LP, UP, Func::ORC>::value>,
      UpperPadHolder<UP> {
    using Base =
        VectorFunction<PaddedOutput<Func, UP, LP>, Func::IRC, SZ_SUM<LP, UP, Func::ORC>::value>;
    VF_TYPE_ALIASES(Base)

    Func func_;
    static constexpr bool is_vectorizable = Func::is_vectorizable;

    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static constexpr bool is_linear_function = Func::is_linear_function;

    PaddedOutput(Func f, int upad, int lpad) : UpperPadHolder<UP>(upad), func_(std::move(f)) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows() + upad + lpad);

        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }

    bool is_linear() const { return this->func_.is_linear(); }

    PaddedOutput() {}

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->func_.compute(
            x, fx.template segment<Func::ORC>(this->u_pad_, this->func_.output_rows()));
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        this->func_.compute_jacobian(
            x, fx.template segment<Func::ORC>(this->u_pad_, this->func_.output_rows()),
            jx.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()));
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

        this->func_.compute_jacobian_adjointgradient_adjointhessian(
            x, fx.template segment<Func::ORC>(this->u_pad_, this->func_.output_rows()),
            jx.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()), adjgrad,
            adjhess, adjvars.template segment<Func::ORC>(this->u_pad_, this->func_.output_rows()));
    }

    //////////////////////////////////////////////////////////////////////////////
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_,
                                       CEigRef<Left> left, CEigRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            CMatRef<Right> right_ref(right.derived());
            CDiagRef<Left> left_ref(left.derived());
            Base::right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        } else {
            MatRef<Right> right_ref = right.const_cast_derived();
            MatRef<Left> left_ref = left.const_cast_derived();
            this->func_.right_jacobian_product(
                target_,
                left_ref.template middleCols<Func::ORC>(this->u_pad_, this->func_.output_rows()),
                right_ref.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()),
                assign, aliased);
        }
    }

    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_,
                                    CMatRef<JacType> right_, Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        MatRef<JacType> right = right_.const_cast_derived();

        this->func_.accumulate_jacobian(
            target.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()),
            right.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()), assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_,
                                    CMatRef<JacType> right, Assignment assign) const {
        this->func_.accumulate_gradient(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_,
                                   CMatRef<JacType> right, Assignment assign) const {
        this->func_.accumulate_hessian(target_, right, assign);
    }
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        MatRef<Target> target = target_.const_cast_derived();
        this->func_.scale_jacobian(
            target.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()), s);
    }
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_gradient(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_hessian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_hessian(target_, s);
    }
};

} // namespace tycho::vf
