// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Implements the Base class for all dense vector functions in Tycho.
// Forwards the Derived class and compile time input(IR) and output(OR) rows down the CRTP
// inheritance chain. Also inherits from domain holder so dynamic sized vector functions can
// hold an array containing their true input domain that is computed at run-time. Also defines the
// default compile time INPUT_DOMAIN of both constant and dynamic size functions. Composite derived
// classes will override this typedef by design, in order to explicitly participate in the input
// domain tracking system. The default input domain simply says that all inputs are used. This input
// domain info is used to implement the default set of functions for manipulating the
// jacobian,hessian,and gradient of Derived function such as right_jacobian_product. Specialized
// derived classes should perform simple overloads of these methods.
//
// This class also defines most of the vector function indexing (.segment) and unary(.normalize)
// and binary(.dot) mathematical operations that are used when writing expressions.
//
// Additionally, This class also defines the .compute_jacobian etc. methods in terms of the
// compute_jacobian_impl methods implemented in derived classes. These methods are then used to
// implement the dense vector functions interface with psiopt through the constraints_jacobian etc.
// methods.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once
#include "tycho/detail/vf/core/assignment_types.h"
#include "tycho/detail/vf/core/computable.h"
#include "tycho/detail/vf/core/dense_function_operations.h"
#include "tycho/detail/vf/core/expression_fwd_declarations.h"
#include "tycho/detail/vf/core/function_domains.h"
#include "tycho/detail/vf/core/function_type_def_macros.h"
#include "tycho/detail/vf/core/var_tags.h"
#include "tycho/detail/vf/operators/binary_math.h"

namespace tycho::vf {

template <class Derived, int IR, int OR>
struct DenseFunctionBase : Computable<Derived, IR, OR>, DomainHolder<IR> {
    using Base = Computable<Derived, IR, OR>;

    /////////////////////////////////////////////////////////////
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;
    template <class Scalar> using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
    template <class Scalar> using Hessian = Eigen::Matrix<Scalar, IR, IR>;
    /////////////////////////////////////////////////////////////
    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;
    template <class Scalar> using VectorBaseRef = Eigen::MatrixBase<Scalar> &;
    template <class Scalar> using ConstMatrixBaseRef = const Eigen::MatrixBase<Scalar> &;
    template <class Scalar> using MatrixBaseRef = Eigen::MatrixBase<Scalar> &;
    template <class Scalar> using ConstEigenBaseRef = const Eigen::EigenBase<Scalar> &;
    template <class Scalar> using EigenBaseRef = Eigen::EigenBase<Scalar> &;
    /////////////////////////////////////////////////////////////

    using INPUT_DOMAIN = SingleDomain<IR, 0, IR>;

    template <class NewDerived> using AsBaseClass = FunctionHolder<NewDerived, Derived, IR, OR>;

    using NAME = Derived;

    template <class Func> using EVALOP = NestedFunctionSelector<Derived, Func>;
    template <class Func> using FWDOP = NestedFunctionSelector<Func, Derived>;
    template <int SZ, int ST>
    using SEGMENTOP = NestedFunctionSelector<Segment<OR, SZ, ST>, Derived>;

    void set_io_rows(int inputrows, int outputrows) {
        Base::set_io_rows(inputrows, outputrows);
        if constexpr (IR < 0) {
            DomainMatrix dmn(2, 1);
            dmn(0, 0) = 0;
            dmn(1, 0) = inputrows;
            this->set_input_domain(inputrows, {dmn});
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Func, int FuncIRC>
    auto operator()(const DenseFunctionBase<Func, FuncIRC, IR> &f) const {
        return EVALOP<Func>::make_nested(this->derived(), f.derived());
    }

    template <class Func, int FuncIRC, int FuncORC>
        requires Composable<Func, Derived>
    auto eval(const DenseFunctionBase<Func, FuncIRC, FuncORC> &f) const {
        return EVALOP<Func>::make_nested(this->derived(), f.derived());
    }

    template <int SZ, int ST> auto segment() const {
        return SEGMENTOP<SZ, ST>::make_nested(Segment<OR, SZ, ST>(this->output_rows(), SZ, ST),
                                              this->derived());
    }

    template <int SZ, int ST> auto segment(int start, int size) const {
        return SEGMENTOP<SZ, ST>::make_nested(Segment<OR, SZ, ST>(this->output_rows(), size, start),
                                              this->derived());
    }

    auto segment(int start, int size) const {
        return SEGMENTOP<-1, -1>::make_nested(Segment<OR, -1, -1>(this->output_rows(), size, start),
                                              this->derived());
    }

    template <int SZ> auto segment(int start) const {
        return SEGMENTOP<SZ, -1>::make_nested(Segment<OR, SZ, -1>(this->output_rows(), SZ, start),
                                              this->derived());
    }

    template <int SZ> auto head() const {
        return SEGMENTOP<SZ, 0>::make_nested(Segment<OR, SZ, 0>(this->output_rows(), SZ, 0),
                                             this->derived());
    }

    template <int SZ> auto head(int sz) const {
        return SEGMENTOP<SZ, 0>::make_nested(Segment<OR, SZ, 0>(this->output_rows(), sz, 0),
                                             this->derived());
    }

    auto head(int sz) const {
        return SEGMENTOP<-1, -1>::make_nested(Segment<OR, -1, -1>(this->output_rows(), sz, 0),
                                              this->derived());
    }

    template <int SZ> auto tail() const {
        return SEGMENTOP<SZ, tycho::utils::SZ_DIFF<OR, SZ>::value>::make_nested(
            Segment<OR, SZ, tycho::utils::SZ_DIFF<OR, SZ>::value>(this->output_rows(), SZ,
                                                                  this->output_rows() - SZ),
            this->derived());
    }

    template <int SZ> decltype(auto) tail(int sz) const {
        return SEGMENTOP<SZ, tycho::utils::SZ_DIFF<OR, SZ>::value>::make_nested(
            Segment<OR, SZ, tycho::utils::SZ_DIFF<OR, SZ>::value>(this->output_rows(), sz,
                                                                  this->output_rows() - sz),
            this->derived());
    }

    decltype(auto) tail(int sz) const {
        return SEGMENTOP<-1, -1>::make_nested(
            Segment<OR, -1, -1>(this->output_rows(), sz, this->output_rows() - sz),
            this->derived());
    }

    template <int ELE> decltype(auto) coeff() const {
        return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->output_rows(), 1, ELE),
                                              this->derived());
    }

    template <int ELE> decltype(auto) coeff(int ele) const {
        return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->output_rows(), 1, ele),
                                              this->derived());
    }

    auto coeff(int ele) const {
        return SEGMENTOP<1, -1>::make_nested(Segment<OR, 1, -1>(this->output_rows(), 1, ele),
                                             this->derived());
    }

    template <int ELE> decltype(auto) operator[](std::integral_constant<int, ELE> ele) const {
        return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->output_rows(), 1, ELE),
                                              this->derived());
    }

    template <int EL1, int... ELS> decltype(auto) elements() const {
        return FWDOP<Elements<OR, EL1, ELS...>>::make_nested(
            Elements<OR, EL1, ELS...>(this->output_rows()), this->derived());
    }

    decltype(auto) normalized() const {
        return FWDOP<Normalized<OR>>::make_nested(Normalized<OR>(this->output_rows()),
                                                  this->derived());
    }

    template <int PW> decltype(auto) normalized_power() const {
        return FWDOP<NormalizedPower<OR, PW>>::make_nested(
            NormalizedPower<OR, PW>(this->output_rows()), this->derived());
    }

    decltype(auto) norm() const {
        return FWDOP<Norm<OR>>::make_nested(Norm<OR>(this->output_rows()), this->derived());
    }

    decltype(auto) squared_norm() const {
        return FWDOP<SquaredNorm<OR>>::make_nested(SquaredNorm<OR>(this->output_rows()),
                                                   this->derived());
    }

    decltype(auto) inverse_norm() const {
        return FWDOP<InverseNorm<OR>>::make_nested(InverseNorm<OR>(this->output_rows()),
                                                   this->derived());
    }

    decltype(auto) inverse_squared_norm() const {
        return FWDOP<InverseSquaredNorm<OR>>::make_nested(
            InverseSquaredNorm<OR>(this->output_rows()), this->derived());
    }

    template <int PW> decltype(auto) norm_power() const {
        return FWDOP<NormPower<OR, PW>>::make_nested(NormPower<OR, PW>(this->output_rows()),
                                                     this->derived());
    }

    template <int PW> decltype(auto) inverse_norm_power() const {
        return FWDOP<InverseNormPower<OR, PW>>::make_nested(
            InverseNormPower<OR, PW>(this->output_rows()), this->derived());
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int MRows, int MCols, int MMajor = Eigen::ColMajor>
    auto matrix(int rows, int cols) const {
        return MatrixFunctionView<Derived, MRows, MCols, MMajor>(this->derived(), rows, cols);
    }

    auto colmatrix(int rows, int cols) const {
        return MatrixFunctionView<Derived, -1, -1, Eigen::ColMajor>(this->derived(), rows, cols);
    }

    auto rowmatrix(int rows, int cols) const {
        return MatrixFunctionView<Derived, -1, -1, Eigen::RowMajor>(this->derived(), rows, cols);
    }

    template <int MRows, int MCols, int MMajor = Eigen::ColMajor> auto matrix() const {
        return MatrixFunctionView<Derived, MRows, MCols, MMajor>(this->derived(), MRows, MCols);
    }

    auto colvector() const {
        return MatrixFunctionView<Derived, OR, 1, Eigen::ColMajor>(this->derived(),
                                                                   this->output_rows(), 1);
    }

    auto rowvector() const {
        return MatrixFunctionView<Derived, 1, OR, Eigen::ColMajor>(this->derived(), 1,
                                                                   this->output_rows());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto sin() const { return CwiseSin<Derived>(this->derived()); }
    auto cos() const { return CwiseCos<Derived>(this->derived()); }
    auto arc_sin() const { return CwiseArcSin<Derived>(this->derived()); }
    auto arc_cos() const { return CwiseArcCos<Derived>(this->derived()); }
    auto tan() const { return CwiseTan<Derived>(this->derived()); }
    auto square() const { return CwiseSquare<Derived>(this->derived()); }
    auto sqrt() const { return CwiseSqrt<Derived>(this->derived()); }
    auto exp() const { return CwiseExp<Derived>(this->derived()); }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int LP> auto padded_lower() const {
        return PaddedOutput<Derived, 0, LP>(this->derived(), 0, LP);
    }
    template <int UP> auto padded_upper() const {
        return PaddedOutput<Derived, UP, 0>(this->derived(), UP, 0);
    }
    template <int UP, int LP> auto padded() const {
        return PaddedOutput<Derived, UP, LP>(this->derived(), UP, LP);
    }

    auto padded_lower(int LP) const { return PaddedOutput<Derived, 0, -1>(this->derived(), 0, LP); }
    auto padded_upper(int UP) const { return PaddedOutput<Derived, -1, 0>(this->derived(), UP, 0); }
    auto padded(int UP, int LP) const {
        return PaddedOutput<Derived, -1, -1>(this->derived(), UP, LP);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class Func>
    decltype(auto) cwise_product(const DenseFunctionBase<Func, IR, OR> &f) const {
        return CwiseFunctionProduct<Derived, Func>(this->derived(), f.derived());
    }
    decltype(auto) cwise_inverse() const { return CwiseInverse<Derived>(this->derived()); }
    decltype(auto) sum() const {
        if constexpr (OR == 1)
            return Derived(this->derived());
        else
            return CwiseSum<Derived>(this->derived());
    }

    template <class Func> decltype(auto) cross(const DenseFunctionBase<Func, IR, 3> &f) const {
        return FunctionCrossProduct<Derived, Func>(this->derived(), f.derived());
    }

    template <class Func> decltype(auto) dot(const DenseFunctionBase<Func, IR, OR> &f) const {
        return FunctionDotProduct<Derived, Func>(this->derived(), f.derived());
    }
    template <class ScalFunc>
    decltype(auto) scale(const DenseFunctionBase<ScalFunc, IR, 1> &f) const {
        return VectorScalarFunctionProduct<Derived, ScalFunc>(this->derived(), f.derived());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                 ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        if constexpr (!Derived::is_vectorizable) {
            if constexpr (Is_SuperScalar<Scalar>::value) {
                VectorBaseRef<OutType> fx = fx_.const_cast_derived();
                VectorBaseRef<JacType> jx = jx_.const_cast_derived();

                typedef typename Scalar::Scalar RealScalar;

                Input<RealScalar> x_r;
                Output<RealScalar> fx_r;
                Jacobian<RealScalar> jx_r;

                const int IRR = this->input_rows();
                const int ORR = this->output_rows();

                if constexpr (Base::InputIsDynamic)
                    x_r.resize(IRR);
                if constexpr (Base::OutputIsDynamic)
                    fx_r.resize(ORR);
                if constexpr (Base::JacobianIsDynamic)
                    jx_r.resize(ORR, IRR);

                for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
                    for (int j = 0; j < IRR; j++)
                        x_r[j] = x[j][i];
                    this->derived().compute_jacobian_impl(x_r, fx_r, jx_r);
                    for (int j = 0; j < ORR; j++)
                        fx[j][i] = fx_r[j];

                    for (int j = 0; j < IRR; j++)
                        for (int k = 0; k < ORR; k++)
                            jx(k, j)[i] = jx_r(k, j);

                    fx_r.setZero();
                    jx_r.setZero();
                }

            } else {
                this->derived().compute_jacobian_impl(x, fx_, jx_);
            }
        } else {
            this->derived().compute_jacobian_impl(x, fx_, jx_);
        }
    }

    template <class InType, class JacType>
    inline void jacobian(ConstVectorBaseRef<InType> x, ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fx_(this->output_rows());
        fx_.setZero();
        this->derived().compute_jacobian(x, fx_, jx_);
    }

    template <class InType>
    inline Jacobian<typename InType::Scalar> jacobian(ConstVectorBaseRef<InType> x) const {
        typedef typename InType::Scalar Scalar;
        Jacobian<Scalar> jx(this->output_rows(), this->input_rows());
        jx.setZero();
        this->derived().jacobian(x, jx);
        return jx;
    }

    template <class JacType, class AdjGradType, class AdjVarType>
    inline void jacobian_x_adjoint(ConstMatrixBaseRef<JacType> jx_,
                                   ConstVectorBaseRef<AdjGradType> adjgrad_,
                                   ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename JacType::Scalar JScalar;
        typedef typename AdjVarType::Scalar AScalar;
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();

        if constexpr (std::is_same<JScalar, AScalar>::value) {
            adjgrad.noalias() = (adjvars.transpose() * jx_).transpose();
        } else {
            for (int i = 0; i < this->input_rows(); i++) {
                for (int j = 0; j < this->output_rows(); j++) {
                    adjgrad[i] += adjvars[j] * jx_(j, i);
                }
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class InType, class OutType, class AdjGradType, class AdjVarType>
    inline void compute_adjointgradient(ConstVectorBaseRef<InType> x,
                                        ConstVectorBaseRef<OutType> fx_,
                                        ConstVectorBaseRef<AdjGradType> adjgrad_,
                                        ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Jacobian<Scalar> jx(this->output_rows(), this->input_rows());
        jx.setZero();
        this->derived().compute_jacobian(x, fx_, jx);
        this->jacobian_x_adjoint(jx, adjgrad_, adjvars);
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(ConstVectorBaseRef<InType> x,
                                                 ConstVectorBaseRef<OutType> fx_,
                                                 ConstMatrixBaseRef<JacType> jx_,
                                                 ConstVectorBaseRef<AdjGradType> adjgrad_,
                                                 ConstVectorBaseRef<AdjVarType> adjvars) const {
        this->derived().compute_jacobian(x, fx_, jx_);
        this->jacobian_x_adjoint(jx_, adjgrad_, adjvars);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {

        typedef typename InType::Scalar Scalar;
        if constexpr (!Derived::is_vectorizable) {

            if constexpr (Is_SuperScalar<Scalar>::value) {
                VectorBaseRef<OutType> fx = fx_.const_cast_derived();
                MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
                Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
                Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

                typedef typename Scalar::Scalar RealScalar;

                Input<RealScalar> x_r;
                Output<RealScalar> fx_r;
                Jacobian<RealScalar> jx_r;
                Gradient<RealScalar> gx_r;
                Hessian<RealScalar> hx_r;
                Output<RealScalar> l_r;

                const int IRR = this->input_rows();
                const int ORR = this->output_rows();

                if constexpr (Base::InputIsDynamic)
                    x_r.resize(IRR);
                if constexpr (Base::InputIsDynamic)
                    gx_r.resize(IRR);
                if constexpr (Base::InputIsDynamic)
                    hx_r.resize(IRR, IRR);

                if constexpr (Base::OutputIsDynamic)
                    fx_r.resize(ORR);
                if constexpr (Base::OutputIsDynamic)
                    l_r.resize(ORR);

                if constexpr (Base::JacobianIsDynamic)
                    jx_r.resize(ORR, IRR);

                for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
                    for (int j = 0; j < IRR; j++)
                        x_r[j] = x[j][i];
                    for (int j = 0; j < ORR; j++)
                        l_r[j] = adjvars[j][i];

                    this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
                        x_r, fx_r, jx_r, gx_r, hx_r, l_r);
                    for (int j = 0; j < ORR; j++)
                        fx[j][i] = fx_r[j];
                    for (int j = 0; j < IRR; j++)
                        adjgrad[j][i] = gx_r[j];

                    for (int j = 0; j < IRR; j++)
                        for (int k = 0; k < ORR; k++)
                            jx(k, j)[i] = jx_r(k, j);

                    for (int j = 0; j < IRR; j++)
                        for (int k = 0; k < IRR; k++)
                            adjhess(k, j)[i] = hx_r(k, j);

                    fx_r.setZero();
                    jx_r.setZero();
                    gx_r.setZero();
                    hx_r.setZero();
                }

            } else {
                this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
                    x, fx_, jx_, adjgrad_, adjhess_, adjvars);
            }
        } else {
            this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
                x, fx_, jx_, adjgrad_, adjhess_, adjvars);
        }
    }

    template <class InType, class AdjHessType, class AdjVarType>
    inline void adjointhessian(ConstVectorBaseRef<InType> x,
                               ConstMatrixBaseRef<AdjHessType> adjhess_,
                               ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Output<Scalar> fx(this->output_rows());
        Input<Scalar> adjgrad(this->input_rows());
        Jacobian<Scalar> jx(this->output_rows(), this->input_rows());
        fx.setZero();
        jx.setZero();
        adjgrad.setZero();

        this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad,
                                                                        adjhess_, adjvars);
    }

    template <class InType, class AdjVarType>
    inline Hessian<typename InType::Scalar>
    adjointhessian(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        Hessian<Scalar> adjhess(this->input_rows(), this->input_rows());
        adjhess.setZero();
        this->derived().adjointhessian(x, adjhess, adjvars);
        return adjhess;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class GenType> GenType make_generic() const { return GenType(this->derived()); }

    /// Type-erase this expression into a GenericFunction.  Similar in spirit
    /// to Eigen's .eval() in that it breaks expression template chains, but
    /// does not evaluate the function -- it wraps it for deferred evaluation
    /// with reduced compile-time cost.  Preserves compile-time input/output
    /// sizes.  Defined in OperatorOverloads.h (requires complete GenericFunction).
    GenericFunction<IR, OR> pack() const;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*
      Implements the derived objects default implemenation of right_jacobian_product.
      This function mutlplies some arbitarly structured matrix, left, on the right by the matrix,
      right, which is assumed have identical non-zero structure to the jacobian of Derived. The
      output of this product is assigned to the target_ matrix according the the Assigment type. The
      constexpr bool, aliased, indicates whether or not any of the three matrices, target_,left, or
      right share any common memory or are the exact same matrix. Abstracting out the this product
      operation, allows us to avoid multupllying explicit blocks of zeros that may be present in
      right, and can be computed from the input domain of derived. The default implementation here
      only detects non-zero columns as deternmiend from the input domain. The actual implementaion
      of the products are in DenseFunctionOperations.h, and the one called depends on whether the
      input domain of Derived is known at compile time or not. Derived objects can overload this
      function in order to implement a product that takes advantage of the exact non-zero pattern of
      the jacobian.
    */
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left, ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {

        // Run Time Input Domain
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {

            if constexpr (Base::InputIsDynamic) {
                right_jacobian_product_dynamic_impl(this->sub_domains, target_, left, right, assign,
                                                    aliased);
            } else {
                right_jacobian_product_impl(target_, left, right, assign, aliased);
            }

        } else {

            // Compile Time Input Domain
            using DMN = typename Derived::INPUT_DOMAIN;
            right_jacobian_product_constant_impl(DMN(), target_, left, right, assign, aliased);
        }
    }

    /*
      Implements the derived objects default and only implemenation of
      right_jacobian_domain_product. This function mutlplies some arbitarly structured matrix, left,
      on the right by the matrix, right, which is assumed have the input domain structure to the
      jacobian of Derived. The output of this product is assigned to the target_ matrix according
      the the Assigment type. The constexpr bool, aliased, indicates whether or not any of the three
      matrices, target_,left, or right share any common memory or are the exact same matrix.
      Abstracting out the this product operation, allows us to avoid multupllying explicit blocks of
      zeros that may be present in right, and can be computed from the input domain of derived. The
      default implementation here only detects non-zero columns as deternmiend from the input
      domain. The actual implementaion of the products are in DenseFunctionOperations.h, and the one
      called depends on whether the input domain of Derived is known at compile time or not. Derived
      objects should not overload this function.

    */
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_domain_product(ConstMatrixBaseRef<Target> target_,
                                              ConstEigenBaseRef<Left> left,
                                              ConstEigenBaseRef<Right> right, Assignment assign,
                                              std::bool_constant<Aliased> aliased) const {
        this->right_jacobian_product(target_, left, right, assign, aliased);
    }

    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                          ConstEigenBaseRef<Left> left,
                                          ConstEigenBaseRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                symetric_jacobian_product_dynamic_impl(this->sub_domains, target_, left, right,
                                                       assign, aliased);
            } else {
                symetric_jacobian_product_impl(target_, left, right, assign, aliased);
            }

        } else {
            using DMN = typename Derived::INPUT_DOMAIN;
            symetric_jacobian_product_constant_impl(DMN(), target_, left, right, assign, aliased);
        }
    }

    /*
     * Accumulates a matrix with the same non-zero structure as the jacobian of
     * Derived into the target_ matrix according to the speccified Assignment type.
     */
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                accumulate_matrix_dynamic_domain_impl(this->sub_domains, target_, right, assign);

            } else {
                accumulate_impl(target_, right, assign);
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            ConstMatrixBaseRef<JacType> right_ref(right.derived());
            MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    accumulate_impl(target_ref.template middleCols<size>(start, size),
                                    right_ref.template middleCols<size>(start, size), assign);
                });
        }
    }
    /*
     * Accumulates a vector with the same non-zero structure as the jacobian of
     * Derived into the target_ matrix according to the speccified Assignment type.
     */
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                accumulate_vector_dynamic_domain_impl(this->sub_domains, target_, right, assign);
            } else {
                accumulate_impl(target_, right, assign);
            }
        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            ConstMatrixBaseRef<JacType> right_ref(right.derived());
            MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    accumulate_impl(target_ref.template segment<size>(start, size),
                                    right_ref.template segment<size>(start, size), assign);
                });
        }
    }

    /*
     * Accumulates a matrix with the same non-zero structure as the hessian of
     * Derived into the target_ matrix according to the specified Assignment type. Does
     * absolutely nothing if Derived is known to be linear at compile time.
     */
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right, Assignment assign) const {
        if constexpr (Derived::is_linear_function) {
        } else if constexpr (Base::InputIsDynamic ||
                             std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                accumulate_symetric_matrix_dynamic_domain_impl(this->sub_domains, target_, right,
                                                               assign);
            } else {
                accumulate_impl(target_, right, assign);
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            ConstMatrixBaseRef<JacType> right_ref(right.derived());
            MatrixBaseRef<Target> target_ref(target_.const_cast_derived());

            if constexpr (sds == 1) {
                constexpr int start = Derived::INPUT_DOMAIN::sub_domains[0][0];
                constexpr int size = Derived::INPUT_DOMAIN::sub_domains[0][1];

                accumulate_impl(target_ref.template block<size, size>(start, start, size, size),
                                right_ref.template block<size, size>(start, start, size, size),
                                assign);

            } else if constexpr (sds == 2) {
                constexpr int Start1 = Derived::INPUT_DOMAIN::sub_domains[0][0];
                constexpr int Size1 = Derived::INPUT_DOMAIN::sub_domains[0][1];

                constexpr int Start2 = Derived::INPUT_DOMAIN::sub_domains[1][0];
                constexpr int Size2 = Derived::INPUT_DOMAIN::sub_domains[1][1];

                accumulate_impl(
                    target_ref.template block<Size1, Size1>(Start1, Start1, Size1, Size1),
                    right_ref.template block<Size1, Size1>(Start1, Start1, Size1, Size1), assign);

                accumulate_impl(
                    target_ref.template block<Size2, Size2>(Start2, Start2, Size2, Size2),
                    right_ref.template block<Size2, Size2>(Start2, Start2, Size2, Size2), assign);

                accumulate_impl(
                    target_ref.template block<Size1, Size2>(Start1, Start2, Size1, Size2),
                    right_ref.template block<Size1, Size2>(Start1, Start2, Size1, Size2), assign);

                accumulate_impl(
                    target_ref.template block<Size2, Size1>(Start2, Start1, Size2, Size1),
                    right_ref.template block<Size2, Size1>(Start2, Start1, Size2, Size1), assign);

            } else {
                tycho::utils::constexpr_for_loop(
                    std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                    [&](auto i) {
                        constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                        constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                        accumulate_impl(target_ref.template middleCols<size>(start, size),
                                        right_ref.template middleCols<size>(start, size), assign);
                    });
            }
        }
    }

    /*
     * Accumulates a vector with the input domain structure s of
     * Derived into the target_ matrix according to the speccified Assignment type.
     */
    template <class Target, class JacType, class Assignment>
    inline void accumulate_matrix_domain(ConstMatrixBaseRef<Target> target_,
                                         ConstMatrixBaseRef<JacType> right,
                                         Assignment assign) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                accumulate_matrix_dynamic_domain_impl(this->sub_domains, target_, right, assign);
            } else {
                accumulate_impl(target_, right, assign);
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            ConstMatrixBaseRef<JacType> right_ref(right.derived());
            MatrixBaseRef<Target> target_ref(target_.const_cast_derived());

            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    accumulate_impl(target_ref.template middleCols<size>(start, size),
                                    right_ref.template middleCols<size>(start, size), assign);
                });
        }
    }

    template <class Target, class JacType>
    inline void accumulate_product_hessian(ConstMatrixBaseRef<Target> target_,
                                           ConstMatrixBaseRef<JacType> right) const {
        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());

        if constexpr (Derived::InputIsDynamic) {
            const int sds = this->sub_domains.cols();
            if (sds == 0) {
                target_ref += right + right.transpose();
            } else {
                for (int i = 0; i < sds; i++) {
                    int start = this->sub_domains(0, i);
                    int size = this->sub_domains(1, i);
                    target_ref.middleCols(start, size) += right.middleCols(start, size);
                    target_ref.middleRows(start, size) += right.middleCols(start, size).transpose();
                }
            }
        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    target_ref.template middleCols<size>(start, size) +=
                        right.template middleCols<size>(start, size);
                    target_ref.template middleRows<size>(start, size) +=
                        right.template middleCols<size>(start, size).transpose();
                });
        }
    }

    template <class Target, class Scalar>
    inline void scale_impl(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        MatrixBaseRef<Target> target = target_.const_cast_derived();
        target *= s;
    }

    template <class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                scale_matrix_dynamic_domain_impl(this->sub_domains, target_, s);

            } else {
                this->scale_impl(target_, s);
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    this->scale_impl(target_ref.template middleCols<size>(start, size), s);
                });
        }
    }
    template <class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                scale_vector_dynamic_domain_impl(this->sub_domains, target_, s);

            } else {
                this->scale_impl(target_, s);
            }
        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    this->scale_impl(target_ref.template segment<size>(start, size), s);
                });
        }
    }
    template <class Target, class Scalar>
    inline void scale_hessian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
        if constexpr (Derived::is_linear_function) {
        } else if constexpr (Base::InputIsDynamic ||
                             std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                scale_matrix_dynamic_domain_impl(this->sub_domains, target_, s);

            } else {
                this->scale_impl(target_, s);
            }
        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    this->scale_impl(target_ref.template middleCols<size>(start, size), s);
                });
        }
    }

    /*
     * Zeros target_ matrix with the same input domain as derived.
     */
    template <class Target>
    inline void zero_matrix_domain(ConstMatrixBaseRef<Target> target_) const {

        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());

        if constexpr (Base::InputIsDynamic ||
                      std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
            if constexpr (Base::InputIsDynamic) {
                const int sds = this->sub_domains.cols();
                if (sds == 0) {
                    target_ref.setZero();
                } else {
                    for (int i = 0; i < sds; i++) {
                        int start = this->sub_domains(0, i);
                        int size = this->sub_domains(1, i);
                        target_ref.middleCols(start, size).setZero();
                    }
                }
            } else {
                target_ref.setZero();
            }

        } else {
            constexpr int sds = Derived::INPUT_DOMAIN::sub_domains.size();
            tycho::utils::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = Derived::INPUT_DOMAIN::sub_domains[i.value][0];
                    constexpr int size = Derived::INPUT_DOMAIN::sub_domains[i.value][1];
                    target_ref.template middleCols<size>(start, size).setZero();
                });
        }
    }

    /*
     * Counts the number of non-zero elements in the lower triangle of the hessian and
     * the Jacobian of derived and represents the number of elements that will be explicilty
     * scattered into the KKT matrix of an optimization problem on each function call. For dense
     * functions, in almost all cases they are all non-zero, or are assumed non-zero even if they
     * are not. However, this method will account for explicit non-zeros if the
     * hessian_elem_is_nonzero ex. is explicitly overloaded by derived. The sum of the non-zeros
     * here is used when interfacing whith the optimizer throutght the wrapper classes
     * ConstraintFunction and ObjectiveFunction.
     */
    int num_kkt_elements(bool dojac, bool dohess) const {
        int hesselems = 0;
        int jacelems = 0;
        for (int i = 0; i < this->input_rows(); i++) {
            if (dohess) {
                for (int j = i; j < this->input_rows(); j++) {
                    if (this->derived().hessian_elem_is_nonzero(j, i))
                        hesselems++;
                }
            }
            if (dojac) {
                for (int j = 0; j < this->output_rows(); j++) {
                    if (this->derived().jacobian_elem_is_nonzero(j, i))
                        jacelems++;
                }
            }
        }
        return ((hesselems + jacelems));
    }

    /*
     * This function is responsible for determining the target locations of the jacobian and hessian
     * elements of derived inside of the KKT matrix of an optimization problem when it is used as a
     * constraint. Meta data containing the locations of the input variables and constraint row for
     * each call of derived are held in the Solver indexing data struct.
     */

    void get_kkt_space(EigenRef<Eigen::VectorXi> KKTrows, EigenRef<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       tycho::solvers::SolverIndexingData &data) {
        data.inner_kkt_starts_.resize(data.num_appl());

        for (int V = 0; V < data.num_appl(); V++) {
            data.inner_kkt_starts_[V] = freeloc;
            for (int i = 0; i < this->input_rows(); i++) {
                if (dohess) {
                    for (int j = i; j < this->input_rows(); j++) {
                        if (this->derived().hessian_elem_is_nonzero(j, i)) {
                            KKTrows[freeloc] = data.v_loc(j, V);
                            KKTcols[freeloc] = data.v_loc(i, V);
                            freeloc++;
                        }
                    }
                }
                if (dojac) {
                    for (int j = 0; j < this->output_rows(); j++) {
                        if (this->derived().jacobian_elem_is_nonzero(j, i)) {
                            KKTrows[freeloc] = data.c_loc(j, V) + conoffset;
                            KKTcols[freeloc] = data.v_loc(i, V);
                            freeloc++;
                        }
                    }
                }
            }
        }
    }

    ////
    /*
    * This function is the interface allowing Derived to be used as a constraint inside psiopt.
    * It computes both the value of the function and its jacobian and is called during psiopts solve
    * algorithm. Vector X is the total variables vector for the full optimization problem, and FX is
    the total equality or inequality constraints vector for the problem. Sparse Matrix KKTMat is the
    full KKT matrix of the problem. KKTLocations contains the location where every non-zero element
    of each jacobian arising from of a call to derived should be summed into KKTMat. KKTClashes
    contains a flag (-1) indicating if we can sum into the corresponding column without clashing
    with the work of another function on another thread. If the flag is greater than zero, the
    element of KKTClash is the index of the mutex in KKTLocks, which derived should lock when
    summing into the column and then release. This logic is controlled by kkt_fill_all and
    kkt_fill_jac. Do not overload these functions unless you really know what you are doing.
    */

    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const tycho::solvers::SolverIndexingData &data) const {

        auto Impl = [&](auto &x, auto &jx) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());

            auto ScalarImpl = [&](int start, int stop) {
                for (int V = start; V < stop; V++) {
                    this->gather_input(X, x, V, data);

                    new (&fx) Eigen::Map<Output<double>>(
                        FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                    fx.setZero();
                    jx.setZero();
                    this->derived().compute_jacobian(x, fx, jx);

                    this->derived().kkt_fill_jac(V, jx, KKTmat, KKTLocations, KKTClashes, KKTLocks,
                                                 data);
                }
            };
            const int IRR = this->input_rows();
            const int ORR = this->output_rows();
            auto VectorImpl = [&]() {
                using SuperScalar = tycho::DefaultSuperScalar;
                constexpr int vsize = SuperScalar::SizeAtCompileTime;
                int Packs = data.num_appl() / vsize;

                Input<SuperScalar> x_vect(this->input_rows());
                Output<SuperScalar> fx_vect(this->output_rows());
                Jacobian<SuperScalar> jx_vect(this->output_rows(), this->input_rows());

                for (int i = 0; i < Packs; i++) {
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        this->gather_input(X, x, V, data);
                        for (int k = 0; k < IRR; k++) {
                            x_vect[k][j] = x[k];
                        }
                    }
                    fx_vect.setZero();
                    jx_vect.setZero();
                    this->derived().compute_jacobian(x_vect, fx_vect, jx_vect);

                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        for (int k = 0; k < IRR; k++) {
                            for (int l = 0; l < ORR; l++) {
                                jx(l, k) = jx_vect(l, k)[j];
                            }
                        }
                        this->derived().kkt_fill_jac(V, jx, KKTmat, KKTLocations, KKTClashes,
                                                     KKTLocks, data);
                    }
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        new (&fx) Eigen::Map<Output<double>>(
                            FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                        for (int l = 0; l < ORR; l++) {
                            fx[l] = fx_vect[l][j];
                        }
                    }
                }

                ScalarImpl(Packs * vsize, data.num_appl());
            };

            if constexpr (Derived::is_vectorizable) {
                if (this->derived().enable_vectorization_) {
                    VectorImpl();
                } else {
                    ScalarImpl(0, data.num_appl());
                }
            } else {
                ScalarImpl(0, data.num_appl());
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1),
            tycho::utils::TempSpec<Jacobian<double>>(this->output_rows(), this->input_rows()));
    }

    void constraints_jacobian_adjointgradient(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const tycho::solvers::SolverIndexingData &data) const {

        auto Impl = [&](auto &x, auto &l, auto &jx) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());
            Eigen::Map<Input<double>> agx(NULL, this->input_rows());

            for (int V = 0; V < data.num_appl(); V++) {
                this->gather_input(X, x, V, data);
                this->gather_mult(L, l, V, data);

                new (&fx) Eigen::Map<Output<double>>(FX.data() + data.inner_constraint_starts_[V],
                                                     this->output_rows());
                new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.inner_gradient_starts_[V],
                                                     this->input_rows());

                fx.setZero();
                agx.setZero();
                jx.setZero();

                this->derived().compute_jacobian_adjointgradient(x, fx, jx, agx, l);

                this->derived().kkt_fill_jac(V, jx, KKTmat, KKTLocations, KKTClashes, KKTLocks,
                                             data);
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1),
            tycho::utils::TempSpec<Output<double>>(this->output_rows(), 1),
            tycho::utils::TempSpec<Jacobian<double>>(this->output_rows(), this->input_rows()));
    }

    /*
    * This function is the interface allowing Derived to be used as a constraint inside psiopt.
    * It computes both the value of the function and its jacobian gradient and hessian and is called
    during psiopts optimize
    * algorithm. Vector X is the total variables vector for the full optimization problem, and FX is
    the total equality or inequality constraints vector for the problem. Vector L is the vector of
    equality or inequality lagrange multipliers for the full optimization problem and. Sparse Matrix
    KKTMat is the full KKT matrix of the problem. KKTLocations contains the location where every
    non-zero element of each jacobian arising from of call to derived should be summed into KKTMat.
    KKTClashes contains a flag (-1) indicating if we can sum into the corresponding column without
    clashing with the work of another function on another thread. If the flag is greater than zero,
    the element of KKTClash is the index of the mutex in KKTLocks, which derived should lock when
    summing into the column and then release. This logic is controlled by kkt_fill_all and
    kkt_fill_jac. Do not overload these functions unless you really know what you are doing.
    */

    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const tycho::solvers::SolverIndexingData &data) const {

        auto Impl = [&](auto &x, auto &l, auto &jx, auto &hx) {
            Eigen::Map<Output<double>> fx(NULL, this->output_rows());
            Eigen::Map<Input<double>> agx(NULL, this->input_rows());

            auto ScalarImpl = [&](int start, int stop) {
                for (int V = start; V < stop; V++) {
                    this->gather_input(X, x, V, data);
                    this->gather_mult(L, l, V, data);

                    new (&fx) Eigen::Map<Output<double>>(
                        FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                    new (&agx) Eigen::Map<Input<double>>(
                        AGX.data() + data.inner_gradient_starts_[V], this->input_rows());

                    fx.setZero();
                    agx.setZero();
                    jx.setZero();
                    hx.setZero();

                    this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, agx,
                                                                                    hx, l);

                    this->derived().kkt_fill_all(V, jx, hx, KKTmat, KKTLocations, KKTClashes,
                                                 KKTLocks, data);
                }
            };

            const int IRR = this->input_rows();
            const int ORR = this->output_rows();
            auto VectorImpl = [&]() {
                using SuperScalar = tycho::DefaultSuperScalar;
                constexpr int vsize = SuperScalar::SizeAtCompileTime;
                int Packs = data.num_appl() / vsize;

                Input<SuperScalar> x_vect(this->input_rows());
                Output<SuperScalar> fx_vect(this->output_rows());
                Jacobian<SuperScalar> jx_vect(this->output_rows(), this->input_rows());
                Gradient<SuperScalar> agx_vect(this->input_rows());
                Hessian<SuperScalar> hx_vect(this->input_rows(), this->input_rows());
                Output<SuperScalar> l_vect(this->output_rows());

                for (int i = 0; i < Packs; i++) {
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        this->gather_input(X, x, V, data);
                        this->gather_mult(L, l, V, data);

                        for (int k = 0; k < IRR; k++) {
                            x_vect[k][j] = x[k];
                        }
                        for (int k = 0; k < ORR; k++) {
                            l_vect[k][j] = l[k];
                        }
                    }
                    fx_vect.setZero();
                    jx_vect.setZero();
                    hx_vect.setZero();
                    agx_vect.setZero();

                    this->derived().compute_jacobian_adjointgradient_adjointhessian(
                        x_vect, fx_vect, jx_vect, agx_vect, hx_vect, l_vect);

                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        for (int k = 0; k < IRR; k++) {
                            for (int l = 0; l < ORR; l++) {
                                jx(l, k) = jx_vect(l, k)[j];
                            }
                        }
                        for (int k = 0; k < IRR; k++) {
                            for (int l = k; l < IRR; l++) {
                                hx(l, k) = hx_vect(l, k)[j];
                            }
                        }
                        this->derived().kkt_fill_all(V, jx, hx, KKTmat, KKTLocations, KKTClashes,
                                                     KKTLocks, data);
                    }
                    for (int j = 0; j < vsize; j++) {
                        int V = i * vsize + j;
                        new (&fx) Eigen::Map<Output<double>>(
                            FX.data() + data.inner_constraint_starts_[V], this->output_rows());
                        for (int l = 0; l < ORR; l++) {
                            fx[l] = fx_vect[l][j];
                        }
                        new (&agx) Eigen::Map<Input<double>>(
                            AGX.data() + data.inner_gradient_starts_[V], this->input_rows());
                        for (int l = 0; l < IRR; l++) {
                            agx[l] = agx_vect[l][j];
                        }
                    }
                }

                ScalarImpl(Packs * vsize, data.num_appl());
            };

            if constexpr (Derived::is_vectorizable) {
                if (this->derived().enable_vectorization_) {
                    VectorImpl();
                } else {
                    ScalarImpl(0, data.num_appl());
                }
            } else {
                ScalarImpl(0, data.num_appl());
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<double>>(this->input_rows(), 1),
            tycho::utils::TempSpec<Output<double>>(this->output_rows(), 1),
            tycho::utils::TempSpec<Jacobian<double>>(this->output_rows(), this->input_rows()),
            tycho::utils::TempSpec<Hessian<double>>(this->input_rows(), this->input_rows()));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  protected:
    inline constexpr bool jacobian_elem_is_nonzero(int row, int col) const { return true; }
    inline constexpr bool hessian_elem_is_nonzero(int row, int col) const {
        return !Derived::is_linear_function;
    }
    inline void add_hessian_elem(double v, int row, int col, double *mpt, const int *lpt,
                                 int &freeloc) const {
        if constexpr (!Derived::is_linear_function) {
            mpt[lpt[freeloc]] += v;
            freeloc++;
        }
    }
    inline void add_jacobian_elem(double v, int row, int col, double *mpt, const int *lpt,
                                  int &freeloc) const {
        mpt[lpt[freeloc]] += v;
        freeloc++;
    }

    template <class JacType, class HessType>
    void kkt_fill_all(int Apl, const JacType &jx, const HessType &hx,
                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                      EigenRef<Eigen::VectorXi> KKTLocs, EigenRef<Eigen::VectorXi> VarClashes,
                      std::vector<std::mutex> &ClashLocks,
                      const tycho::solvers::SolverIndexingData &data) const {
        int freeloc = data.inner_kkt_starts_[Apl];
        double *mpt = KKTmat.valuePtr();
        const int *lpt = KKTLocs.data();
        int ActiveVar;
        const int IRR = (Base::IRC > 0) ? Base::IRC : this->input_rows();
        const int ORR = (Base::ORC > 0) ? Base::ORC : this->output_rows();

        auto Lock = [&](int var) {
            if (VarClashes[var] == -1) {
                //// uncontested
            } else {
                /// contested lock mutex
                ClashLocks[VarClashes[var]].lock();
            }
        };
        auto UnLock = [&](int var) {
            if (VarClashes[var] == -1) {
                //// uncontested
            } else {
                /// contested unlock mutex
                ClashLocks[VarClashes[var]].unlock();
            }
        };

        const bool uniquecon = data.unique_constraints_;

        // bool uc = data.unique_constraints_;
        for (int i = 0; i < IRR; i++) {
            ActiveVar = data.v_loc(i, Apl);
            Lock(ActiveVar);
            ///// insert hessian column symetrically
            for (int j = i; j < IRR; j++) {
                this->derived().add_hessian_elem(hx(j, i), j, i, mpt, lpt, freeloc);
            }
            if (uniquecon)
                UnLock(ActiveVar);
            ///// insert jacobian column
            for (int j = 0; j < ORR; j++) {
                this->derived().add_jacobian_elem(jx(j, i), j, i, mpt, lpt, freeloc);
            }
            ///////////////////////////////////////////////////////////////////////////////
            if (!uniquecon)
                UnLock(ActiveVar);
        }
    }

    template <class JacType>
    void kkt_fill_jac(int Apl, const JacType &jx,
                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                      Eigen::Ref<Eigen::VectorXi> KKTLocs, Eigen::Ref<Eigen::VectorXi> VarClashes,
                      std::vector<std::mutex> &ClashLocks,
                      const tycho::solvers::SolverIndexingData &data) const {
        int freeloc = data.inner_kkt_starts_[Apl];
        double *mpt = KKTmat.valuePtr();
        const int *lpt = KKTLocs.data();
        int ActiveVar;

        if (data.unique_constraints_) {
            for (int i = 0; i < this->input_rows(); i++) {
                ActiveVar = data.v_loc(i, Apl);
                for (int j = i; j < this->input_rows(); j++) {
                    if (this->derived().hessian_elem_is_nonzero(j, i))
                        freeloc++;
                }
                for (int j = 0; j < this->output_rows(); j++) {
                    this->derived().add_jacobian_elem(jx(j, i), j, i, mpt, lpt, freeloc);
                }
            }
        } else {
            for (int i = 0; i < this->input_rows(); i++) {
                ActiveVar = data.v_loc(i, Apl);
                if (VarClashes[ActiveVar] == -1) {
                    //// uncontested
                } else {
                    /// contested lock mutex
                    ClashLocks[VarClashes[ActiveVar]].lock();
                }

                //////////////////////////Mutex
                /// Locked////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////////////
                ///// insert hessian column symetrically
                for (int j = i; j < this->input_rows(); j++) {
                    if (this->derived().hessian_elem_is_nonzero(j, i))
                        freeloc++;
                }
                ///// insert jacobian column

                for (int j = 0; j < this->output_rows(); j++) {
                    this->derived().add_jacobian_elem(jx(j, i), j, i, mpt, lpt, freeloc);
                }
                ///////////////////////////////////////////////////////////////////////////////
                if (VarClashes[ActiveVar] == -1) {
                    //// uncontested
                } else {
                    /// contested unlock mutex
                    ClashLocks[VarClashes[ActiveVar]].unlock();
                }
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
