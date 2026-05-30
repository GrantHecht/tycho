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

/// @internal
/// @brief Forward declaration of the CwiseFunctionProduct implementation base.
/// @tparam Derived  CRTP-derived product type.
/// @tparam Func1    First operand function.
/// @tparam Func2    Second operand function.
/// @endinternal
template <class Derived, class Func1, class Func2> struct CwiseFunctionProduct_Impl;

/// @brief VectorFunction evaluating the element-wise product @f$ f_1 \odot f_2 @f$.
///
/// Both operands must share the same input and output sizes; the result has the same
/// output size, with element @f$ i @f$ equal to @f$ f_{1,i}(x)\,f_{2,i}(x) @f$. Forwards
/// all compute machinery to CwiseFunctionProduct_Impl.
/// @tparam Func1  First operand function.
/// @tparam Func2  Second operand function.
/// @ingroup vf
template <class Func1, class Func2>
struct CwiseFunctionProduct
    : CwiseFunctionProduct_Impl<CwiseFunctionProduct<Func1, Func2>, Func1, Func2> {
    /// @brief CRTP implementation base providing the compute/derivative methods.
    using Base = CwiseFunctionProduct_Impl<CwiseFunctionProduct<Func1, Func2>, Func1, Func2>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @internal
/// @brief CRTP implementation base for CwiseFunctionProduct.
/// @tparam Derived  CRTP-derived product type (CwiseFunctionProduct<Func1, Func2>).
/// @tparam Func1    First operand function.
/// @tparam Func2    Second operand function.
/// @endinternal
template <class Derived, class Func1, class Func2>
struct CwiseFunctionProduct_Impl : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                                  SZ_MAX<Func1::ORC, Func2::ORC>::value> {
    /// @internal
    /// @brief VectorFunction base sized to the larger of the two operands' I/O rows.
    /// @endinternal
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                SZ_MAX<Func1::ORC, Func2::ORC>::value>;
    VF_TYPE_ALIASES(Base);

    using Base::compute;

    /// @internal
    /// @brief First operand function.
    /// @endinternal
    Func1 func1;
    /// @internal
    /// @brief Second operand function.
    /// @endinternal
    Func2 func2;

    /// @internal
    /// @brief Composite input domain merging both operands' domains.
    /// @endinternal
    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;
    /// @internal
    /// @brief True only when both operands are SIMD-vectorizable.
    /// @endinternal
    static constexpr bool is_vectorizable = Func1::is_vectorizable && Func2::is_vectorizable;

    /// @internal
    /// @brief Default constructor; leaves both operands default-constructed.
    /// @endinternal
    CwiseFunctionProduct_Impl() {}
    /// @internal
    /// @brief Construct from two operands, validating sizes and configuring I/O.
    ///
    /// Throws std::invalid_argument if the operands' input or output sizes disagree.
    /// @param f1  First operand function.
    /// @param f2  Second operand function.
    /// @endinternal
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
    /// @internal
    /// @brief Evaluate the element-wise product of the two operands.
    /// @tparam InType   Eigen type of the input vector.
    /// @tparam OutType  Eigen type of the output.
    /// @param x    Input vector.
    /// @param fx_  Output reference receiving the element-wise product.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;

        VecRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &fx2) {
            this->func1.compute(x, fx);
            this->func2.compute(x, fx2);
            fx = fx.cwiseProduct(fx2);
        };

        using FType = typename Func2::template Output<Scalar>;
        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1));
    }
    /// @internal
    /// @brief Evaluate the element-wise product and its Jacobian via the product rule.
    /// @tparam InType   Eigen type of the input vector.
    /// @tparam OutType  Eigen type of the output.
    /// @tparam JacType  Eigen type of the Jacobian.
    /// @param x    Input vector.
    /// @param fx_  Output reference receiving the element-wise product.
    /// @param jx_  Jacobian reference.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &fx2, auto &jx2) {
            this->func1.compute_jacobian(x, fx, jx);
            this->func2.compute_jacobian(x, fx2, jx2);
            this->func1.right_jacobian_product(jx, fx2.asDiagonal(), jx, DirectAssignment(),
                                               std::bool_constant<true>());
            this->func2.right_jacobian_product(jx, fx.asDiagonal(), jx2, PlusEqualsAssignment(),
                                               std::bool_constant<false>());
            fx = fx.cwiseProduct(fx2);
        };

        using FType = typename Func2::template Output<Scalar>;
        using JType = typename Func2::template Jacobian<Scalar>;
        const int irows = this->input_rows();
        const int orows = this->output_rows();

        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1),
                                                  tycho::utils::TempSpec<JType>(orows, irows));
    }
    /// @internal
    /// @brief Evaluate the product, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen type of the input vector.
    /// @tparam OutType      Eigen type of the output.
    /// @tparam JacType      Eigen type of the Jacobian.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient.
    /// @tparam AdjHessType  Eigen type of the adjoint Hessian.
    /// @tparam AdjVarType   Eigen type of the adjoint (dual) variables.
    /// @param x        Input vector.
    /// @param fx_      Output reference receiving the element-wise product.
    /// @param jx_      Jacobian reference.
    /// @param adjgrad_ Adjoint gradient reference.
    /// @param adjhess_ Adjoint Hessian reference.
    /// @param adjvars  Adjoint (dual) variables seeding the reverse pass.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

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

        using FType1 = typename Func1::template Output<Scalar>;
        using JType1 = typename Func1::template Jacobian<Scalar>;

        using FType2 = typename Func2::template Output<Scalar>;
        using JType2 = typename Func2::template Jacobian<Scalar>;
        using GType2 = typename Func2::template Gradient<Scalar>;
        using HType2 = typename Func2::template Hessian<Scalar>;

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<FType2>(orows, 1),
            tycho::utils::TempSpec<JType2>(orows, irows), tycho::utils::TempSpec<GType2>(irows, 1),
            tycho::utils::TempSpec<HType2>(irows, irows), tycho::utils::TempSpec<FType1>(orows, 1),
            tycho::utils::TempSpec<JType2>(orows, irows));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
