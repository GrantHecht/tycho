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

struct CrossProduct : VectorFunction<CrossProduct, 6, 3> {
    using Base = VectorFunction<CrossProduct, 6, 3>;
    using Base::compute;
    using Base::jacobian;

    DENSE_FUNCTION_BASE_TYPES(Base);

    template <class Source, class Target, class Scalar2>
    static void cprodmat(const Eigen::MatrixBase<Source> &x, Eigen::MatrixBase<Target> const &m_,
                         Scalar2 sign) {
        typedef typename Source::Scalar Scalar;
        Eigen::MatrixBase<Target> &m = m_.const_cast_derived();
        m(1, 0) += sign * x[2];
        m(2, 0) += -sign * x[1];
        m(2, 1) += sign * x[0];

        m(0, 1) += -sign * x[2];
        m(0, 2) += sign * x[1];
        m(1, 2) += -sign * x[0];
    }

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        fx = x.template head<3>().cross(x.template tail<3>());
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        fx = x.template head<3>().cross(x.template tail<3>());
        cprodmat(x.template tail<3>(), jx.template leftCols<3>(), -1.0);
        cprodmat(x.template head<3>(), jx.template rightCols<3>(), 1.0);
    }

    template <class InType, class JacType>
    inline void jacobian(const Eigen::MatrixBase<InType> &x,
                         Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        cprodmat(x.template tail<3>(), jx.template leftCols<3>(), -1.0);
        cprodmat(x.template head<3>(), jx.template rightCols<3>(), 1.0);
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

        fx = x.template head<3>().cross(x.template tail<3>());
        cprodmat(x.template tail<3>(), jx.template leftCols<3>(), -1.0);
        cprodmat(x.template head<3>(), jx.template rightCols<3>(), 1.0);

        adjgrad = (adjvars.transpose() * jx).transpose();

        cprodmat(adjvars, adjhess.template topRightCorner<3, 3>(), -1.0);
        cprodmat(adjvars, adjhess.template bottomLeftCorner<3, 3>(), 1.0);
    }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <class Derived, class Func1, class Func2> struct FunctionCrossProduct_Impl;

template <class Func1, class Func2>
struct FunctionCrossProduct
    : FunctionCrossProduct_Impl<FunctionCrossProduct<Func1, Func2>, Func1, Func2> {
    using Base = FunctionCrossProduct_Impl<FunctionCrossProduct<Func1, Func2>, Func1, Func2>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
};

template <class Derived, class Func1, class Func2>
struct FunctionCrossProduct_Impl
    : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, 3> {
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, 3>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);

    Func1 func1;
    Func2 func2;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;

#if defined(_WIN32)
    static const bool is_vectorizable = Func1::is_vectorizable && Func2::is_vectorizable;
#else
    static const bool is_vectorizable = false;
#endif

    FunctionCrossProduct_Impl() {}
    FunctionCrossProduct_Impl(Func1 f1, Func2 f2) : func1(f1), func2(f2) {
        int irtemp = std::max(this->func1.input_rows(), this->func2.input_rows());
        this->set_io_rows(irtemp, 3);

        this->set_input_domain(this->input_rows(),
                               {this->func1.input_domain(), this->func2.input_domain()});

        if (this->func1.output_rows() != 3) {
            throw std::invalid_argument("Function 1 in cross product must have three output rows");
        }
        if (this->func2.output_rows() != 3) {
            throw std::invalid_argument("Function 2 in cross product must have three output rows");
        }
        if (this->func1.input_rows() != this->func1.input_rows()) {
            throw std::invalid_argument(
                "Functions 1,2 in cross product must have same numer of input rows");
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Scalar, class T1, class T2>
    Vector3<Scalar> crossimpl(Scalar sign, ConstVectorBaseRef<T1> x1,
                              ConstVectorBaseRef<T2> x2) const {
        Vector3<Scalar> out;
        out[0] = sign * (x1[1] * x2[2] - x1[2] * x2[1]);
        out[1] = sign * (x2[0] * x1[2] - x2[2] * x1[0]);
        out[2] = sign * (x1[0] * x2[1] - x1[1] * x2[0]);
        return out;
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        Vector3<Scalar> fx1;
        fx1.setZero();
        Vector3<Scalar> fx2;
        fx2.setZero();

        this->func1.compute(x, fx1);
        this->func2.compute(x, fx2);
        // fx = (fx1.cross(fx2)).eval();
        fx = crossimpl(Scalar(1.0), fx1, fx2);
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        Vector3<Scalar> fx1;
        Vector3<Scalar> fx2;
        Eigen::Matrix<Scalar, 3, 3> cpm1;
        Eigen::Matrix<Scalar, 3, 3> cpm2;

        auto Impl = [&](auto &jx1, auto &jx2) {
            this->func1.compute_jacobian(x, fx1, jx1);
            this->func2.compute_jacobian(x, fx2, jx2);
            CrossProduct::cprodmat(fx2, cpm1, -1.0);
            CrossProduct::cprodmat(fx1, cpm2, 1.0);
            fx = crossimpl(Scalar(1.0), fx1, fx2);
            this->func1.right_jacobian_product(jx, cpm1, jx1, DirectAssignment(),
                                               std::bool_constant<false>());
            this->func2.right_jacobian_product(jx, cpm2, jx2, PlusEqualsAssignment(),
                                               std::bool_constant<false>());
        };

        using JType = Eigen::Matrix<Scalar, 3, Base::IRC>;
        const int irows = this->input_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<JType>(3, irows),
                                                  tycho::utils::TempSpec<JType>(3, irows));
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

        Vector3<Scalar> fx1;
        Vector3<Scalar> fx2;
        Eigen::Matrix<Scalar, 3, 3> cpm1;
        Eigen::Matrix<Scalar, 3, 3> cpm2;
        Eigen::Matrix<Scalar, 3, 3> lcpm1;
        Eigen::Matrix<Scalar, 3, 3> lcpm2;

        auto Impl = [&](auto &jx1, auto &jx2, auto &jttemp, auto &gx2, auto &hx2) {
            this->func1.compute(x, fx1);
            this->func2.compute(x, fx2);

            CrossProduct::cprodmat(fx2, cpm1, -1.0);
            CrossProduct::cprodmat(fx1, cpm2, 1.0);

            CrossProduct::cprodmat(adjvars, lcpm1, 1.0);
            CrossProduct::cprodmat(adjvars, lcpm2, -1.0);

            Vector3<Scalar> adjt = adjvars;

            fx = crossimpl(Scalar(1.0), fx1, fx2);
            Vector3<Scalar> adjcross1 = crossimpl(Scalar(1.0), fx2, adjt);
            Vector3<Scalar> adjcross2 = crossimpl(Scalar(-1.0), fx1, adjt);

            fx1.setZero();
            fx2.setZero();

            this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx1, jx1, adjgrad,
                                                                        adjhess, adjcross1);
            this->func2.compute_jacobian_adjointgradient_adjointhessian(x, fx2, jx2, gx2, hx2,
                                                                        adjcross2);

            this->func2.accumulate_gradient(adjgrad, gx2, PlusEqualsAssignment());
            this->func2.accumulate_hessian(adjhess, hx2, PlusEqualsAssignment());

            this->func2.zero_matrix_domain(hx2);

            this->func2.right_jacobian_product(jttemp, lcpm2, jx2, DirectAssignment(),
                                               std::bool_constant<false>());
            this->func1.right_jacobian_product(hx2, jttemp.transpose(), jx1, DirectAssignment(),
                                               std::bool_constant<false>());

            // adjhess += hx2 + hx2^T
            // func1 does this because hx2 now has its domain structure
            this->func1.accumulate_product_hessian(adjhess, hx2);

            this->func1.right_jacobian_product(jx, cpm1, jx1, DirectAssignment(),
                                               std::bool_constant<false>());
            this->func2.right_jacobian_product(jx, cpm2, jx2, PlusEqualsAssignment(),
                                               std::bool_constant<false>());
        };

        using JType = Eigen::Matrix<Scalar, 3, Base::IRC>;
        using GType = Func2_gradient<Scalar>;
        using HType = Func2_hessian<Scalar>;
        const int irows = this->input_rows();

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<JType>(3, irows), tycho::utils::TempSpec<JType>(3, irows),
            tycho::utils::TempSpec<JType>(3, irows), tycho::utils::TempSpec<GType>(irows, 1),
            tycho::utils::TempSpec<HType>(irows, irows));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
