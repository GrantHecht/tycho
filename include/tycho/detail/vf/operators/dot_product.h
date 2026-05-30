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

/// @cond INTERNAL
template <class Derived, class Func1, class Func2> struct FunctionDotProduct_Impl;
/// @endcond

/// @ingroup vf
/// @brief VectorFunction computing the inner product @f$f_1(x)^T f_2(x)@f$.
///
/// Both operands must produce vectors of equal length; the single scalar output
/// is their Euclidean dot product. Provides analytic Jacobian and adjoint Hessian.
/// @tparam Func1  Left operand VectorFunction.
/// @tparam Func2  Right operand VectorFunction.
template <class Func1, class Func2>
struct FunctionDotProduct
    : FunctionDotProduct_Impl<FunctionDotProduct<Func1, Func2>, Func1, Func2> {
    /// @brief Convenience alias for the underlying dot-product implementation.
    using Base = FunctionDotProduct_Impl<FunctionDotProduct<Func1, Func2>, Func1, Func2>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @internal
/// @brief Shared implementation of @ref FunctionDotProduct.
/// @tparam Derived  CRTP host type (@ref FunctionDotProduct).
/// @tparam Func1    Left operand function.
/// @tparam Func2    Right operand function.
/// @endinternal
template <class Derived, class Func1, class Func2>
struct FunctionDotProduct_Impl : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, 1> {
    /// @brief Convenience alias for the VectorFunction CRTP base.
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, 1>;
    VF_TYPE_ALIASES(Base);

    using Base::compute;

    /// @brief True if both operands are plain segments, enabling the direct-segment fast path.
    static constexpr bool IsSegmentOp = Is_Segment<Func1>::value && Is_Segment<Func2>::value;
    /// @brief True if both operands are vectorizable, enabling SuperScalar evaluation.
    static constexpr bool is_vectorizable = Func1::is_vectorizable && Func2::is_vectorizable;

    /// @brief Combined input domain of the two operands.
    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;

    Func1 func1; ///< Left operand function.
    Func2 func2; ///< Right operand function.

    /// @brief Default constructor; leaves operands default-constructed.
    FunctionDotProduct_Impl() {}
    /// @brief Construct from the two operand functions, validating their shapes.
    /// @param f1  Left operand function.
    /// @param f2  Right operand function.
    FunctionDotProduct_Impl(Func1 f1, Func2 f2) : func1(std::move(f1)), func2(std::move(f2)) {
        int irtemp = std::max(this->func1.input_rows(), this->func2.input_rows());
        if (this->func1.output_rows() != this->func2.output_rows()) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Math Error in FunctionDotProduct/.dot method !!!\n"
                       "Output Size of Func1 (ORows = {0:}) does not match Output Size of Func2 "
                       "(ORows = {1:}).\n",
                       this->func1.output_rows(), this->func2.output_rows());

            throw std::invalid_argument("");
        }
        if (this->func1.input_rows() != this->func2.input_rows()) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Math Error in FunctionDotProduct/.dot method !!!\n"
                       "Input Size of Func1 (IRows = {0:}) does not match Input Size of Func2 "
                       "(IRows = {1:}).\n",
                       this->func1.input_rows(), this->func2.input_rows());
            throw std::invalid_argument("");
        }

        this->set_io_rows(irtemp, 1);
        this->set_input_domain(this->input_rows(),
                               {this->func1.input_domain(), this->func2.input_domain()});
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Evaluate the dot product into @p fx_.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input vector.
    /// @param fx_  Output scalar (1-vector) to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;

        VecRef<OutType> fx = fx_.const_cast_derived();

        if constexpr (IsSegmentOp) {
            fx[0] =
                x.template segment<Func1::ORC>(this->func1.seg_start_, this->func1.output_rows())
                    .dot(x.template segment<Func2::ORC>(this->func2.seg_start_,
                                                        this->func2.output_rows()));
        } else {

            /// @cond INTERNAL
            auto Impl = [&](auto &fx1, auto &fx2) {
                this->func1.compute(x, fx1);
                this->func2.compute(x, fx2);
                fx[0] = fx1.dot(fx2);
            };
            /// @endcond

            const int orows = this->func1.output_rows();

            using FType1 = typename Func1::template Output<Scalar>;
            using FType2 = typename Func2::template Output<Scalar>;

            tycho::utils::BumpAllocator::allocate_run(Impl,
                                                      tycho::utils::TempSpec<FType1>(orows, 1),
                                                      tycho::utils::TempSpec<FType2>(orows, 1));
        }
    }

    /// @internal
    /// @brief Evaluate the dot product and its Jacobian.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input vector.
    /// @param fx_  Output scalar (1-vector) to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        if constexpr (IsSegmentOp) {
            fx[0] =
                x.template segment<Func1::ORC>(this->func1.seg_start_, this->func1.output_rows())
                    .dot(x.template segment<Func2::ORC>(this->func2.seg_start_,
                                                        this->func2.output_rows()));

            jx.template block<1, Func1::ORC>(0, this->func1.seg_start_, 1,
                                             this->func1.output_rows()) +=
                x.template segment<Func2::ORC>(this->func2.seg_start_, this->func2.output_rows())
                    .transpose();

            jx.template block<1, Func2::ORC>(0, this->func2.seg_start_, 1,
                                             this->func2.output_rows()) +=
                x.template segment<Func1::ORC>(this->func1.seg_start_, this->func1.output_rows())
                    .transpose();

        } else {

            /// @cond INTERNAL
            auto Impl = [&](auto &fx1, auto &fx2, auto &jx1, auto &jx2) {
                this->func1.compute_jacobian(x, fx1, jx1);
                this->func2.compute_jacobian(x, fx2, jx2);
                fx[0] = fx1.dot(fx2);

                this->func1.right_jacobian_product(jx_, fx2.transpose(), jx1, DirectAssignment(),
                                                   std::bool_constant<false>());
                this->func2.right_jacobian_product(
                    jx_, fx1.transpose(), jx2, PlusEqualsAssignment(), std::bool_constant<false>());
            };
            /// @endcond

            const int orows = this->func1.output_rows();
            const int irows = this->func1.input_rows();

            using FType1 = typename Func1::template Output<Scalar>;
            using JType1 = typename Func2::template Jacobian<Scalar>;

            using FType2 = typename Func2::template Output<Scalar>;
            using JType2 = typename Func2::template Jacobian<Scalar>;

            tycho::utils::BumpAllocator::allocate_run(Impl,
                                                      tycho::utils::TempSpec<FType1>(orows, 1),
                                                      tycho::utils::TempSpec<FType2>(orows, 1),
                                                      tycho::utils::TempSpec<JType1>(orows, irows),
                                                      tycho::utils::TempSpec<JType2>(orows, irows));
        }
    }
    /// @internal
    /// @brief Evaluate the dot product, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input vector.
    /// @param fx_      Output scalar (1-vector) to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to accumulate.
    /// @param adjhess_ Output adjoint Hessian to accumulate.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
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

        if constexpr (IsSegmentOp) {
            fx[0] =
                x.template segment<Func1::ORC>(this->func1.seg_start_, this->func1.output_rows())
                    .dot(x.template segment<Func2::ORC>(this->func2.seg_start_,
                                                        this->func2.output_rows()));

            jx.template block<1, Func1::ORC>(0, this->func1.seg_start_, 1,
                                             this->func1.output_rows()) +=
                x.template segment<Func2::ORC>(this->func2.seg_start_, this->func2.output_rows())
                    .transpose();

            jx.template block<1, Func2::ORC>(0, this->func2.seg_start_, 1,
                                             this->func2.output_rows()) +=
                x.template segment<Func1::ORC>(this->func1.seg_start_, this->func1.output_rows())
                    .transpose();

            adjgrad.template segment<Func1::ORC>(this->func1.seg_start_,
                                                 this->func1.output_rows()) +=
                x.template segment<Func2::ORC>(this->func2.seg_start_, this->func2.output_rows()) *
                adjvars[0];

            adjgrad.template segment<Func2::ORC>(this->func2.seg_start_,
                                                 this->func2.output_rows()) +=
                x.template segment<Func1::ORC>(this->func1.seg_start_, this->func1.output_rows()) *
                adjvars[0];

            for (int i = 0; i < this->func1.output_rows(); i++) {
                adjhess(this->func1.seg_start_ + i, this->func2.seg_start_ + i) += adjvars[0];
                adjhess(this->func2.seg_start_ + i, this->func1.seg_start_ + i) += adjvars[0];
            }

        } else {

            /// @cond INTERNAL
            auto Impl = [&](auto &fx1, auto &fx2, auto &jx1, auto &jx2, auto &gx2, auto &hx2,
                            auto &adjtemp) {
                this->func2.compute(x, adjtemp);

                adjtemp *= adjvars[0];

                this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx1, jx1, adjgrad,
                                                                            adjhess, adjtemp);

                adjtemp = fx1 * adjvars[0];

                this->func2.compute_jacobian_adjointgradient_adjointhessian(x, fx2, jx2, gx2, hx2,
                                                                            adjtemp);

                fx[0] = fx1.dot(fx2);
                if constexpr (!Func2::is_linear_function) {
                    this->func2.accumulate_hessian(adjhess, hx2, PlusEqualsAssignment());
                    this->func2.zero_matrix_domain(hx2);
                }

                this->func2.accumulate_gradient(adjgrad, gx2, PlusEqualsAssignment());

                this->func1.right_jacobian_product(hx2, jx2.transpose(), jx1, DirectAssignment(),
                                                   std::bool_constant<false>());

                if constexpr (Func1::InputIsDynamic) {
                    const int sds = this->func1.sub_domains.cols();
                    if (sds == 0) {
                        adjhess += hx2 + hx2.transpose();
                    } else {
                        for (int i = 0; i < sds; i++) {
                            int start = this->func1.sub_domains(0, i);
                            int size = this->func1.sub_domains(1, i);
                            adjhess.middleCols(start, size) +=
                                hx2.middleCols(start, size) * adjvars[0];
                            adjhess.middleRows(start, size) +=
                                hx2.middleCols(start, size).transpose() * adjvars[0];
                        }
                    }
                } else {
                    constexpr int sds = Func1::INPUT_DOMAIN::sub_domains.size();
                    tycho::utils::constexpr_for_loop(
                        std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                        [&](auto i) {
                            constexpr int start = Func1::INPUT_DOMAIN::sub_domains[i.value][0];
                            constexpr int size = Func1::INPUT_DOMAIN::sub_domains[i.value][1];

                            adjhess.template middleCols<size>(start, size) +=
                                hx2.template middleCols<size>(start, size) * adjvars[0];
                            adjhess.template middleRows<size>(start, size) +=
                                hx2.template middleCols<size>(start, size).transpose() * adjvars[0];
                        });
                }

                this->func1.right_jacobian_product(jx_, fx2.transpose(), jx1, DirectAssignment(),
                                                   std::bool_constant<false>());
                this->func2.right_jacobian_product(
                    jx_, fx1.transpose(), jx2, PlusEqualsAssignment(), std::bool_constant<false>());
            };
            /// @endcond

            const int orows = this->func1.output_rows();
            const int irows = this->func1.input_rows();

            using FType1 = typename Func1::template Output<Scalar>;
            using JType1 = typename Func2::template Jacobian<Scalar>;

            using FType2 = typename Func2::template Output<Scalar>;
            using JType2 = typename Func2::template Jacobian<Scalar>;
            using GType2 = typename Func2::template Gradient<Scalar>;
            using HType2 = typename Func2::template Hessian<Scalar>;

            tycho::utils::BumpAllocator::allocate_run(Impl,
                                                      tycho::utils::TempSpec<FType1>(orows, 1),
                                                      tycho::utils::TempSpec<FType2>(orows, 1),
                                                      tycho::utils::TempSpec<JType1>(orows, irows),
                                                      tycho::utils::TempSpec<JType2>(orows, irows),
                                                      tycho::utils::TempSpec<GType2>(irows, 1),
                                                      tycho::utils::TempSpec<HType2>(irows, irows),
                                                      tycho::utils::TempSpec<FType1>(orows, 1));
        }
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
