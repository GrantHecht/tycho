// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines all Vector Functionals that apply linear scalings to
// the outputs of other functions.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

template <class Derived, class Func, class Value> struct StaticScaled_Impl;

template <class Derived, class Func> struct Scaled_Impl;

template <class Derived, class Func> struct RowScaled_Impl;

template <class Derived, class Func, int MRows> struct MatrixScaled_Impl;

template <class Func, class Value>
struct StaticScaled : StaticScaled_Impl<StaticScaled<Func, Value>, Func, Value> {
    using Base = StaticScaled_Impl<StaticScaled<Func, Value>, Func, Value>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
};

template <class Func> struct Scaled : Scaled_Impl<Scaled<Func>, Func> {
    using Base = Scaled_Impl<Scaled<Func>, Func>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
};

template <class Func> struct RowScaled : RowScaled_Impl<RowScaled<Func>, Func> {
    using Base = RowScaled_Impl<RowScaled<Func>, Func>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
};

template <class Func, int MRows>
struct MatrixScaled : MatrixScaled_Impl<MatrixScaled<Func, MRows>, Func, MRows> {
    using Base = MatrixScaled_Impl<MatrixScaled<Func, MRows>, Func, MRows>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
};

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

template <class Derived, class Func, class Value>
struct StaticScaled_Impl
    : VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    Func func;
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    static const bool is_linear_function = Func::is_linear_function;

    StaticScaled_Impl() {}
    StaticScaled_Impl(Func f) : func(std::move(f)) {
        this->set_io_rows(this->func.input_rows(), this->func.output_rows());
    }
    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        // typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        this->func.compute(x, fx_);
        fx *= Value::value;
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        // typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        this->func.compute_jacobian(x, fx_, jx_);

        fx *= Value::value;
        this->func.scale_jacobian(jx, Value::value);
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        // VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        // MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        Output<Scalar> adjv_scaled = adjvars * Value::value;

        this->func.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                   adjv_scaled);
        fx *= Value::value;
        this->func.scale_jacobian(jx, Value::value);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left, ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        this->func.right_jacobian_product(target_, left, right, assign, aliased);
    }
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                          ConstEigenBaseRef<Left> left,
                                          ConstEigenBaseRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_jacobian(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_gradient(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_hessian(target_, right, assign);
    }
    template <class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_jacobian(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_gradient(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_hessian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_hessian(target_, s);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

template <class Derived, class Func>
struct Scaled_Impl : VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    Func func;
    double scale_value_ = 1.0;

    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    static const bool is_linear_function = Func::is_linear_function;
    static const bool is_vectorizable = Func::is_vectorizable;

    Scaled_Impl() {}
    Scaled_Impl(Func f, double s) : func(std::move(f)), scale_value_(s) {
        this->set_io_rows(this->func.input_rows(), this->func.output_rows());
        this->set_input_domain(this->input_rows(), {func.input_domain()});
    }

    bool is_linear() const { return func.is_linear(); }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        this->func.compute(x, fx_);
        fx *= Scalar(this->scale_value_);
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        this->func.compute_jacobian(x, fx_, jx_);
        fx *= Scalar(this->scale_value_);
        this->func.scale_jacobian(jx, Scalar(this->scale_value_));
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &adjv_scaled) {
            adjv_scaled = adjvars * Scalar(this->scale_value_);
            this->func.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                       adjhess_, adjv_scaled);
            fx *= Scalar(this->scale_value_);
            this->func.scale_jacobian(jx_, Scalar(this->scale_value_));
        };

        const int orows = this->func.output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left, ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            ConstMatrixBaseRef<Right> right_ref(right.derived());
            ConstDiagonalBaseRef<Left> left_ref(left.derived());
            this->func.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        } else {
            ConstMatrixBaseRef<Right> right_ref(right.derived());
            ConstMatrixBaseRef<Left> left_ref(left.derived());
            this->func.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        }
    }
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                          ConstEigenBaseRef<Left> left,
                                          ConstEigenBaseRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_jacobian(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_gradient(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_hessian(target_, right, assign);
    }
    template <class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_jacobian(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_gradient(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_hessian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_hessian(target_, s);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

template <class Derived, class Func>
struct RowScaled_Impl
    : VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    Func func;
    Output<double> row_scale_values_;
    static const bool is_linear_function = Func::is_linear_function;
    static const bool is_vectorizable = Func::is_vectorizable;
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    RowScaled_Impl() { this->row_scale_values_.setOnes(); }

    template <class OutType>
    RowScaled_Impl(Func f, ConstVectorBaseRef<OutType> s) : func(std::move(f)) {
        this->row_scale_values_ = s;
        this->set_io_rows(this->func.input_rows(), this->func.output_rows());
        this->set_input_domain(this->input_rows(), {func.input_domain()});
    }
    bool is_linear() const { return func.is_linear(); }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &scales) {
            scales = this->row_scale_values_.template cast<Scalar>();
            this->func.compute(x, fx_);
            fx = fx.cwiseProduct(scales);
        };

        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &scales) {
            scales = this->row_scale_values_.template cast<Scalar>();

            this->func.compute_jacobian(x, fx_, jx_);
            fx = fx.cwiseProduct(scales);
            this->func.right_jacobian_product(jx, scales.asDiagonal(), jx, DirectAssignment(),
                                              std::bool_constant<true>());
        };

        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        auto Impl = [&](auto &scales, auto &adjv_scaled) {
            scales = this->row_scale_values_.template cast<Scalar>();
            adjv_scaled = adjvars.cwiseProduct(scales);

            this->func.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                       adjhess_, adjv_scaled);
            fx = fx.cwiseProduct(scales);
            this->func.right_jacobian_product(jx, scales.asDiagonal(), jx, DirectAssignment(),
                                              std::bool_constant<true>());
        };

        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Output<Scalar>>(orows, 1),
                                           tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left, ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {

        if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            ConstMatrixBaseRef<Right> right_ref(right.derived());
            ConstDiagonalBaseRef<Left> left_ref(left.derived());
            this->func.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        } else {
            ConstMatrixBaseRef<Right> right_ref(right.derived());
            ConstMatrixBaseRef<Left> left_ref(left.derived());
            this->func.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        }
    }
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                          ConstEigenBaseRef<Left> left,
                                          ConstEigenBaseRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_jacobian(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_gradient(target_, right, assign);
    }
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        this->func.accumulate_hessian(target_, right, assign);
    }
    template <class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_jacobian(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_gradient(target_, s);
    }
    template <class Target, class Scalar>
    inline void scale_hessian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        this->func.scale_hessian(target_, s);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

template <class Derived, class Func, int MRows>
struct MatrixScaled_Impl
    : VectorFunction<Derived, Func::IRC, MRows, DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<Derived, Func::IRC, MRows, DenseDerivativeMode::Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func);

    Func func;

    template <class Scalar> using MatType = Eigen::Matrix<Scalar, MRows, Func::ORC>;

    MatType<double> mat;
    bool NoTemp = false;

    static const bool is_linear_function = Func::is_linear_function;
    static const bool is_vectorizable = Func::is_vectorizable;
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    MatrixScaled_Impl() {}

    template <class OutType>
    MatrixScaled_Impl(Func f, ConstVectorBaseRef<OutType> s) : func(std::move(f)) {
        this->mat = s;
        this->set_io_rows(this->func.input_rows(), this->mat.rows());
        this->set_input_domain(this->input_rows(), {this->func.input_domain()});

        if (mat.cols() != func.output_rows()) {
            throw std::invalid_argument(
                "Matrix Must have same number of cols as RHS vector function.");
        }

        if (mat.rows() == mat.cols()) {
            NoTemp = true;
        }
    }
    bool is_linear() const { return this->func.is_linear(); }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        if (NoTemp) {
            auto Impl = [&](auto &mattmp) {
                this->func.compute(x, fx_);
                mattmp = this->mat.template cast<Scalar>();
                fx = (mattmp * fx).eval();
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(), this->func.output_rows()));
        } else {

            auto Impl = [&](auto &mattmp, auto &fxtmp) {
                this->func.compute(x, fxtmp);
                mattmp = this->mat.template cast<Scalar>();
                fx = (mattmp * fxtmp);
            };

            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(), this->func.output_rows()),
                tycho::utils::TempSpec<Func_Output<Scalar>>(this->func.output_rows(), 1));
        }
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        if (NoTemp) {
            auto Impl = [&](auto &mattmp) {
                this->func.compute_jacobian(x, fx_, jx_);
                mattmp = this->mat.template cast<Scalar>();
                fx = (mattmp * fx).eval();
                this->func.right_jacobian_product(jx, mattmp, jx, DirectAssignment(),
                                                  std::bool_constant<true>());
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(), this->func.output_rows()));
        } else {

            auto Impl = [&](auto &mattmp, auto &fxtmp, auto &jxtmp) {
                this->func.compute_jacobian(x, fxtmp, jxtmp);
                mattmp = this->mat.template cast<Scalar>();
                fx = (mattmp * fxtmp);
                this->func.right_jacobian_product(jx, mattmp, jxtmp, DirectAssignment(),
                                                  std::bool_constant<false>());
            };

            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(), this->func.output_rows()),
                tycho::utils::TempSpec<Func_Output<Scalar>>(this->func.output_rows(), 1),
                tycho::utils::TempSpec<Func_jacobian<Scalar>>(this->func.output_rows(), this->func.input_rows()));
        }
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        if (NoTemp) {
            auto Impl = [&](auto &mattmp, auto &adjv_scaled) {
                mattmp = this->mat.template cast<Scalar>();
                adjv_scaled = (adjvars.transpose() * mattmp).transpose();
                this->func.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                           adjhess_, adjv_scaled);

                fx = (mattmp * fx).eval();
                this->func.right_jacobian_product(jx, mattmp, jx, DirectAssignment(),
                                                  std::bool_constant<true>());
            };
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(), this->func.output_rows()),
                tycho::utils::TempSpec<Func_Output<Scalar>>(this->func.output_rows(), 1)

            );
        } else {

            auto Impl = [&](auto &mattmp, auto &fxtmp, auto &jxtmp, auto &adjv_scaled) {
                mattmp = this->mat.template cast<Scalar>();
                adjv_scaled = (adjvars.transpose() * mattmp).transpose();
                this->func.compute_jacobian_adjointgradient_adjointhessian(
                    x, fxtmp, jxtmp, adjgrad_, adjhess_, adjv_scaled);
                fx = (mattmp * fxtmp);
                this->func.right_jacobian_product(jx, mattmp, jxtmp, DirectAssignment(),
                                                  std::bool_constant<false>());
            };

            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<MatType<Scalar>>(this->output_rows(), this->func.output_rows()),
                tycho::utils::TempSpec<Func_Output<Scalar>>(this->func.output_rows(), 1),
                tycho::utils::TempSpec<Func_jacobian<Scalar>>(this->func.output_rows(), this->func.input_rows()),
                tycho::utils::TempSpec<Func_Output<Scalar>>(this->func.output_rows(), 1));
        }
    }
};

} // namespace tycho::vf
