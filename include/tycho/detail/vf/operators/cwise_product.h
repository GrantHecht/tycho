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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

template <class Derived, class Func1, class Func2> struct CwiseFunctionProduct_Impl;

template <class Func1, class Func2>
struct CwiseFunctionProduct
    : CwiseFunctionProduct_Impl<CwiseFunctionProduct<Func1, Func2>, Func1, Func2> {
    using Base = CwiseFunctionProduct_Impl<CwiseFunctionProduct<Func1, Func2>, Func1, Func2>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
};

template <class Derived, class Func1, class Func2>
struct CwiseFunctionProduct_Impl : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                                  SZ_MAX<Func1::ORC, Func2::ORC>::value> {
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                SZ_MAX<Func1::ORC, Func2::ORC>::value>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);
    using Base::compute;

    Func1 func1;
    Func2 func2;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;
    static const bool is_vectorizable = Func1::is_vectorizable && Func2::is_vectorizable;

    CwiseFunctionProduct_Impl() {}
    CwiseFunctionProduct_Impl(Func1 f1, Func2 f2) : func1(std::move(f1)), func2(std::move(f2)) {
        int irtemp = std::max(this->func1.input_rows(), this->func2.input_rows());
        if (this->func1.output_rows() != this->func2.output_rows()) {
            std::cout << "User Input Error in CwiseFunctionProduct:" << this->name() << std::endl;
            std::cout << "	Output Size of Func1 ( " << this->func1.name()
                      << " ) does not match Output Size of Func2 ( " << this->func2.name() << " )"
                      << std::endl;
            throw std::invalid_argument("");
        }
        if (this->func1.input_rows() != this->func2.input_rows()) {
            std::cout << "User Input Error in CwiseFunctionProduct:" << this->name() << std::endl;
            std::cout << "	Input Size of Func1 ( " << this->func1.name()
                      << " ) does not match Output Size of Func2 ( " << this->func2.name() << " )"
                      << std::endl;
            throw std::invalid_argument("");
        }

        this->set_io_rows(irtemp, this->func1.output_rows());

        this->set_input_domain(this->input_rows(),
                               {this->func1.input_domain(), this->func2.input_domain()});
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;

        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &fx2) {
            this->func1.compute(x, fx);
            this->func2.compute(x, fx2);
            fx = fx.cwiseProduct(fx2);
        };

        using FType = Func2_Output<Scalar>;
        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1));
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &fx2, auto &jx2) {
            this->func1.compute_jacobian(x, fx, jx);
            this->func2.compute_jacobian(x, fx2, jx2);
            this->func1.right_jacobian_product(jx, fx2.asDiagonal(), jx, DirectAssignment(),
                                               std::bool_constant<true>());
            this->func2.right_jacobian_product(jx, fx.asDiagonal(), jx2, PlusEqualsAssignment(),
                                               std::bool_constant<false>());
            fx = fx.cwiseProduct(fx2);
        };

        using FType = Func2_Output<Scalar>;
        using JType = Func2_jacobian<Scalar>;
        const int irows = this->input_rows();
        const int orows = this->output_rows();

        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1), tycho::utils::TempSpec<JType>(orows, irows));
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

        auto Impl = [&](auto &fx2, auto &jx2, auto &gx2, auto &hx2, auto &adjprod, auto &jttemp) {
            this->func2.compute(x, fx2);

            adjprod = adjvars.cwiseProduct(fx2);
            this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess,
                                                                        adjprod);

            fx2.setZero();
            adjprod = adjvars.cwiseProduct(fx);

            this->func2.compute_jacobian_adjointgradient_adjointhessian(x, fx2, jx2, gx2, hx2,
                                                                        adjprod);

            this->func2.accumulate_hessian(adjhess, hx2, PlusEqualsAssignment());
            this->func2.accumulate_gradient(adjgrad, gx2, PlusEqualsAssignment());

            this->func2.zero_matrix_domain(hx2);

            this->func2.right_jacobian_product(jttemp, adjvars.asDiagonal(), jx2,
                                               DirectAssignment(), std::bool_constant<false>());
            this->func1.right_jacobian_product(hx2, jttemp.transpose(), jx, PlusEqualsAssignment(),
                                               std::bool_constant<false>());

            this->func1.accumulate_product_hessian(adjhess, hx2);

            ///////////////////////////////////////////////////
            this->func1.right_jacobian_product(jx, fx2.asDiagonal(), jx, DirectAssignment(),
                                               std::bool_constant<true>());
            this->func2.right_jacobian_product(jx, fx.asDiagonal(), jx2, PlusEqualsAssignment(),
                                               std::bool_constant<false>());

            fx = fx.cwiseProduct(fx2);
        };

        const int orows = this->func1.output_rows();
        const int irows = this->func1.input_rows();

        using FType1 = Func1_Output<Scalar>;
        using JType1 = Func1_jacobian<Scalar>;

        using FType2 = Func2_Output<Scalar>;
        using JType2 = Func2_jacobian<Scalar>;
        using GType2 = Func2_gradient<Scalar>;
        using HType2 = Func2_hessian<Scalar>;

        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType2>(orows, 1),
                                    tycho::utils::TempSpec<JType2>(orows, irows), tycho::utils::TempSpec<GType2>(irows, 1),
                                    tycho::utils::TempSpec<HType2>(irows, irows), tycho::utils::TempSpec<FType1>(orows, 1),
                                    tycho::utils::TempSpec<JType2>(orows, irows));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
