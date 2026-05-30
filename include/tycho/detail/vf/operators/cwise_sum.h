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
/// @brief Forward declaration of the CwiseSum implementation base.
/// @tparam Derived  CRTP-derived sum type.
/// @tparam Func     Wrapped operand function whose elements are summed.
/// @endinternal
template <class Derived, class Func> struct CwiseSum_Impl;

/// @brief VectorFunction reducing an operand to the scalar sum of its output elements.
///
/// Wraps an operand function @p Func and evaluates to @f$ \sum_i f_i(x) @f$, a single
/// output row. Forwards all compute machinery to CwiseSum_Impl.
/// @tparam Func  Wrapped operand function whose output elements are summed.
/// @ingroup vf
template <class Func> struct CwiseSum : CwiseSum_Impl<CwiseSum<Func>, Func> {
    /// @brief CRTP implementation base providing the compute/derivative methods.
    using Base = CwiseSum_Impl<CwiseSum<Func>, Func>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

/// @internal
/// @brief CRTP implementation base for CwiseSum providing compute and derivative paths.
/// @tparam Derived  CRTP-derived sum type (CwiseSum<Func>).
/// @tparam Func     Wrapped operand function whose output elements are summed.
/// @endinternal
template <class Derived, class Func> struct CwiseSum_Impl : VectorFunction<Derived, Func::IRC, 1> {
    /// @internal
    /// @brief VectorFunction base specialised to the operand input rows and one output row.
    /// @endinternal
    using Base = VectorFunction<Derived, Func::IRC, 1>;
    using Base::compute;

    /// @internal
    /// @brief Rebind helper yielding the implementation base for a different operand.
    /// @tparam OtherFunc  Substitute operand function type.
    /// @endinternal
    template <class OtherFunc> using BaseTemplate = CwiseSum_Impl<CwiseSum<OtherFunc>, OtherFunc>;
    /// @internal
    /// @brief Rebind helper yielding the derived CwiseSum for different operands.
    /// @tparam OtherFunc  Substitute operand function type pack.
    /// @endinternal
    template <class... OtherFunc> using DerivedTemplate = CwiseSum<OtherFunc...>;

    /// @internal
    /// @brief Input domain inherited from the wrapped operand function.
    /// @endinternal
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    /// @internal
    /// @brief True when the wrapped operand is itself a linear function.
    /// @endinternal
    static constexpr bool is_linear_function = Func::is_linear_function;
    /// @internal
    /// @brief True when the operand is a Segment/Arguments view (enables a fast sum path).
    /// @endinternal
    static constexpr bool IsSegmentOp = Is_Segment<Func>::value || Is_Arguments<Func>::value;

    VF_TYPE_ALIASES(Base);

    /// @internal
    /// @brief The wrapped operand function whose output elements are summed.
    /// @endinternal
    Func func_;
    /// @internal
    /// @brief Default constructor; leaves the operand default-constructed.
    /// @endinternal
    CwiseSum_Impl() {}
    /// @internal
    /// @brief Construct from an operand function and configure I/O rows and input domain.
    /// @param f  Operand function to wrap and sum.
    /// @endinternal
    CwiseSum_Impl(Func f) : func_(std::move(f)) {
        this->set_io_rows(this->func_.input_rows(), 1);
        this->set_input_domain(this->input_rows(), {this->func_.input_domain()});
    }

    /// @internal
    /// @brief Evaluate the scalar sum of the operand outputs.
    /// @tparam InType   Eigen type of the input vector.
    /// @tparam OutType  Eigen type of the (one-row) output.
    /// @param x    Input vector.
    /// @param fx_  Output reference receiving the sum.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        if constexpr (IsSegmentOp) {
            fx[0] = x.template segment<Func::ORC>(this->func_.seg_start_, this->func_.output_rows())
                        .sum();
        } else {
            typename Func::template Output<Scalar> fxv;
            if constexpr (Func::OutputIsDynamic) {
                fxv.resize(this->func_.output_rows());
            }

            this->func_.compute(x, fxv);
            fx[0] = fxv.sum();
        }
    }
    /// @internal
    /// @brief Evaluate the scalar sum and its Jacobian row.
    /// @tparam InType   Eigen type of the input vector.
    /// @tparam OutType  Eigen type of the (one-row) output.
    /// @tparam JacType  Eigen type of the Jacobian.
    /// @param x    Input vector.
    /// @param fx_  Output reference receiving the sum.
    /// @param jx_  Jacobian reference receiving the row of partial derivatives.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        if constexpr (IsSegmentOp) {
            fx[0] = x.template segment<Func::ORC>(this->func_.seg_start_, this->func_.output_rows())
                        .sum();
            jx.template middleCols<Func::ORC>(this->func_.seg_start_, this->func_.output_rows())
                .setOnes();
        } else {

            auto Impl = [&](auto &fxv, auto &jxv) {
                this->func_.compute_jacobian(x, fxv, jxv);
                fx[0] = fxv.sum();

                if constexpr (Func::InputIsDynamic) {
                    if (this->sub_domains.size() == 0) {
                        jx = jxv.colwise().sum();
                    } else {
                        for (int i = 0; i < this->sub_domains.size(); i++) {
                            int start = this->sub_domains(i, 0);
                            int size = this->sub_domains(i, 1);
                            jx.middleCols(start, size) =
                                jxv.middleCols(start, size).colwise().sum().transpose();
                        }
                    }
                } else {

                    constexpr int sds = Func::INPUT_DOMAIN::sub_domains.size();

                    tycho::utils::constexpr_for_loop(
                        std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                        [&](auto i) {
                            constexpr int start = Func::INPUT_DOMAIN::sub_domains[i.value][0];
                            constexpr int size = Func::INPUT_DOMAIN::sub_domains[i.value][1];
                            jx.middleCols(start, size) =
                                jxv.middleCols(start, size).colwise().sum().transpose();
                        });
                }
            };

            tycho::utils::BumpAllocator::allocate_run(
                Impl,
                tycho::utils::TempSpec<typename Func::template Output<Scalar>>(
                    this->func_.output_rows(), 1),
                tycho::utils::TempSpec<typename Func::template Jacobian<Scalar>>(
                    this->func_.output_rows(), this->func_.input_rows()));
        }
    }
    /// @internal
    /// @brief Evaluate the sum, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen type of the input vector.
    /// @tparam OutType      Eigen type of the (one-row) output.
    /// @tparam JacType      Eigen type of the Jacobian.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient.
    /// @tparam AdjHessType  Eigen type of the adjoint Hessian.
    /// @tparam AdjVarType   Eigen type of the adjoint (dual) variables.
    /// @param x        Input vector.
    /// @param fx_      Output reference receiving the sum.
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

        if constexpr (IsSegmentOp) {
            fx[0] = x.template segment<Func::ORC>(this->func_.seg_start_, this->func_.output_rows())
                        .sum();
            jx.template middleCols<Func::ORC>(this->func_.seg_start_, this->func_.output_rows())
                .setOnes();
            adjgrad.template segment<Func::ORC>(this->func_.seg_start_, this->func_.output_rows())
                .setConstant(adjvars[0]);
        } else {

            auto Impl = [&](auto &fxv, auto &jxv, auto &adjv) {
                adjv.setConstant(adjvars[0]);

                this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fxv, jxv, adjgrad,
                                                                            adjhess, adjv);
                fx[0] = fxv.sum();

                if constexpr (Func::InputIsDynamic) {
                    if (this->sub_domains.size() == 0) {
                        jx = jxv.colwise().sum();
                    } else {
                        for (int i = 0; i < this->sub_domains.size(); i++) {
                            int start = this->sub_domains(i, 0);
                            int size = this->sub_domains(i, 1);
                            jx.middleCols(start, size) =
                                jxv.middleCols(start, size).colwise().sum().transpose();
                        }
                    }
                } else {
                    constexpr int sds = Func::INPUT_DOMAIN::sub_domains.size();
                    tycho::utils::constexpr_for_loop(
                        std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                        [&](auto i) {
                            constexpr int start = Func::INPUT_DOMAIN::sub_domains[i.value][0];
                            constexpr int size = Func::INPUT_DOMAIN::sub_domains[i.value][1];
                            jx.middleCols(start, size) =
                                jxv.middleCols(start, size).colwise().sum().transpose();
                        });
                }
            };

            tycho::utils::BumpAllocator::allocate_run(
                Impl,
                tycho::utils::TempSpec<typename Func::template Output<Scalar>>(
                    this->func_.output_rows(), 1),
                tycho::utils::TempSpec<typename Func::template Jacobian<Scalar>>(
                    this->func_.output_rows(), this->func_.input_rows()),
                tycho::utils::TempSpec<typename Func::template Output<Scalar>>(
                    this->func_.output_rows(), 1));
        }
    }
};

} // namespace tycho::vf
