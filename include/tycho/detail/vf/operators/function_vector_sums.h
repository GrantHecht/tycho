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
#include "tycho/detail/vf/derivatives/detect_diagonal.h"

namespace tycho::vf {

template <class Derived, class Func, bool NegateFunction> struct FunctionVectorSum_Impl;

template <class Func>
struct FunctionPlusVector : FunctionVectorSum_Impl<FunctionPlusVector<Func>, Func, false> {
    using Base = FunctionVectorSum_Impl<FunctionPlusVector<Func>, Func, false>;
    using Base::Base;
};
template <class Func>
struct VectorMinusFunction : FunctionVectorSum_Impl<VectorMinusFunction<Func>, Func, true> {
    using Base = FunctionVectorSum_Impl<VectorMinusFunction<Func>, Func, true>;
    using Base::Base;
};

template <class Derived, class Func, bool NegateFunction>
struct FunctionVectorSum_Impl
    : VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    Func func_;
    Output<double> offset_vector;
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static constexpr bool is_linear_function = Func::is_linear_function;
    static constexpr bool is_vectorizable = Func::is_vectorizable;

    FunctionVectorSum_Impl() {}
    template <class OutType>
    FunctionVectorSum_Impl(Func f, ConstVectorBaseRef<OutType> v) : func_(std::move(f)) {
        this->offset_vector = v;
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }
    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        this->func_.compute(x, fx_);
        fx += this->offset_vector.template cast<Scalar>();
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        // MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        this->func_.compute_jacobian(x, fx_, jx_);
        fx += this->offset_vector.template cast<Scalar>();
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        // MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        // VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        // MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();
        this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                    adjvars);
        fx += this->offset_vector.template cast<Scalar>();
    }

    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left, ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            ConstMatrixBaseRef<Right> right_ref(right.derived());
            ConstDiagonalBaseRef<Left> left_ref(left.derived());
            this->func_.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        } else {
            ConstMatrixBaseRef<Right> right_ref(right.derived());
            ConstMatrixBaseRef<Left> left_ref(left.derived());
            this->func_.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        }
    }
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                          ConstEigenBaseRef<Left> left,
                                          ConstEigenBaseRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func_.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func_.accumulate_jacobian(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func_.accumulate_gradient(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func_.accumulate_hessian(target_, right, assign);
    }
    template <class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func_.scale_jacobian(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func_.scale_gradient(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_hessian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func_.scale_hessian(target_, s);
    }
};
} // namespace tycho::vf
