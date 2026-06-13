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
#include "tycho/detail/vf/expressions/segment.h"

namespace tycho::vf {

/// @internal
/// @brief Forward declaration of the function-composition implementation.
/// @tparam Derived    CRTP most-derived type (the user-facing @ref NestedFunction).
/// @tparam OuterFunc  Outer function applied last.
/// @tparam InnerFunc  Inner function applied first.
/// @endinternal
template <class Derived, class OuterFunc, class InnerFunc> struct NestedFunction_Impl;

/// @brief Function composition: `f(x) = OuterFunc(InnerFunc(x))`.
///
/// The return type of `.eval()` / function-of-function composition in the DSL.
/// Evaluates @p InnerFunc on the input, feeds the result to @p OuterFunc, and
/// applies the chain rule for the Jacobian/gradient/Hessian.
///
/// @tparam OuterFunc  Outer function applied last.
/// @tparam InnerFunc  Inner function applied first.
/// @ingroup vf
template <class OuterFunc, class InnerFunc>
struct NestedFunction
    : NestedFunction_Impl<NestedFunction<OuterFunc, InnerFunc>, OuterFunc, InnerFunc> {
    /// @brief Composition implementation base.
    using Base = NestedFunction_Impl<NestedFunction<OuterFunc, InnerFunc>, OuterFunc, InnerFunc>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

//////////////////////////////////////////////////////////////////////
/// @internal
/// @brief CRTP implementation of @ref NestedFunction composition.
///
/// Holds the outer and inner functions, validates that the inner output size
/// matches the outer input size, and implements the chain-rule derivative
/// assembly (using bump-allocated scratch for the intermediate value and
/// Jacobian/gradient/Hessian buffers). A `Segment` outer function is handled by
/// a cheaper slice path.
///
/// @tparam Derived    CRTP most-derived type (the user-facing @ref NestedFunction).
/// @tparam OuterFunc  Outer function applied last.
/// @tparam InnerFunc  Inner function applied first.
/// @endinternal
template <class Derived, class OuterFunc, class InnerFunc>
struct NestedFunction_Impl : VectorFunction<Derived, InnerFunc::IRC, OuterFunc::ORC> {
    /// @internal
    /// @brief VectorFunction base: inner input rows to outer output rows.
    /// @endinternal
    using Base = VectorFunction<Derived, InnerFunc::IRC, OuterFunc::ORC>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    OuterFunc outer_func_; ///< @brief Outer function applied last.
    InnerFunc inner_func_; ///< @brief Inner function applied first.

    /// @internal
    /// @brief Input-domain descriptor inherited from the inner function.
    /// @endinternal
    using INPUT_DOMAIN = typename InnerFunc::INPUT_DOMAIN;
    static constexpr bool is_linear_function =
        OuterFunc::is_linear_function &&
        InnerFunc::is_linear_function; ///< @brief True if both stages are linear.
    static constexpr bool is_vectorizable =
        OuterFunc::is_vectorizable &&
        InnerFunc::is_vectorizable; ///< @brief True if both stages are vectorizable.

    /// @internal
    /// @brief Default-construct an empty composition.
    /// @endinternal
    NestedFunction_Impl() {}
    /// @internal
    /// @brief Construct from outer and inner functions, validating sizes.
    /// @param ofunc  Outer function (applied last).
    /// @param ifunc  Inner function (applied first).
    /// @endinternal
    NestedFunction_Impl(OuterFunc ofunc, InnerFunc ifunc)
        : outer_func_(std::move(ofunc)), inner_func_(std::move(ifunc)) {
        if (this->inner_func_.output_rows() != this->outer_func_.input_rows()) {

            throw std::invalid_argument(
                fmt::format("Math Error in NestedFunction/.eval method: "
                            "Output Size of InnerFunction (ORows = {}) does not match "
                            "Input Size of OuterFunction (IRows = {}).",
                            this->inner_func_.output_rows(), this->outer_func_.input_rows()));
        }

        this->set_io_rows(this->inner_func_.input_rows(), this->outer_func_.output_rows());
        this->set_input_domain(this->input_rows(), {inner_func_.input_domain()});
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Evaluate the composition: `fx = OuterFunc(InnerFunc(x))`.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &fx_inner) {
            if constexpr (Is_Segment<OuterFunc>::value) {
                this->inner_func_.compute(x, fx_inner);
                fx = fx_inner.template segment<OuterFunc::ORC>(this->outer_func_.seg_start_,
                                                               this->output_rows());
            } else {
                this->inner_func_.compute(x, fx_inner);
                this->outer_func_.compute(fx_inner, fx);
            }
        };

        const int orows = this->inner_func_.output_rows();
        using FType = typename InnerFunc::template Output<Scalar>;
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1));
    }
    /// @internal
    /// @brief Evaluate value and chain-rule Jacobian of the composition.
    ///
    /// Computes the inner Jacobian, the outer Jacobian at the inner output, and
    /// folds them via a right-Jacobian product. A `Segment` outer function uses
    /// a cheaper slice path.
    ///
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        // MatRef<JacType> jx = jx_.const_cast_derived();

        if constexpr (Is_Segment<OuterFunc>::value) {

            auto Impl = [&](auto &fx_inner, auto &jx_inner) {
                this->inner_func_.compute_jacobian(x, fx_inner, jx_inner);
                fx = fx_inner.template segment<OuterFunc::ORC>(this->outer_func_.seg_start_,
                                                               this->output_rows());
                this->inner_func_.accumulate_matrix_domain(
                    jx_,
                    jx_inner.template middleRows<OuterFunc::ORC>(this->outer_func_.seg_start_,
                                                                 this->output_rows()),
                    PlusEqualsAssignment());
            };

            const int inner_OR = this->inner_func_.output_rows();
            const int inner_IR = this->inner_func_.input_rows();

            using IFXType = typename InnerFunc::template Output<Scalar>;
            using IJXType = typename InnerFunc::template Jacobian<Scalar>;
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<IFXType>(inner_OR, 1),
                tycho::utils::TempSpec<IJXType>(inner_OR, inner_IR));

        } else {
            auto Impl = [&](auto &fx_inner, auto &jx_inner, auto &jx_outer) {
                this->inner_func_.compute_jacobian(x, fx_inner, jx_inner);
                this->outer_func_.compute_jacobian(fx_inner, fx_, jx_outer);
                this->inner_func_.right_jacobian_product(
                    jx_, jx_outer, jx_inner, DirectAssignment(), std::bool_constant<false>());
            };

            const int inner_OR = this->inner_func_.output_rows();
            const int inner_IR = this->inner_func_.input_rows();
            const int outer_OR = this->outer_func_.output_rows();
            const int outer_IR = this->outer_func_.input_rows();

            using IFXType = typename InnerFunc::template Output<Scalar>;
            using IJXType = typename InnerFunc::template Jacobian<Scalar>;
            using OJXType = typename OuterFunc::template Jacobian<Scalar>;
            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<IFXType>(inner_OR, 1),
                tycho::utils::TempSpec<IJXType>(inner_OR, inner_IR),
                tycho::utils::TempSpec<OJXType>(outer_OR, outer_IR));
        }
    }
    /// @internal
    /// @brief Value, Jacobian, adjoint gradient, and adjoint Hessian of the composition.
    ///
    /// Runs the inner and outer passes and applies the second-order chain rule:
    /// the outer Hessian is pulled back through the inner Jacobian and combined
    /// with the inner Hessian over the inner function's input sub-domains. A
    /// `Segment` outer function uses a cheaper slice path.
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @param  adjgrad_ Adjoint-gradient accumulator (size = `input_rows()`).
    /// @param  adjhess_ Adjoint-Hessian accumulator (`input_rows()` square).
    /// @param  adjvars  Adjoint variables seeding the reverse pass.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        // MatRef<JacType> jx = jx_.const_cast_derived();
        // VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        if constexpr (Is_Segment<OuterFunc>::value) {
            auto Impl = [&](auto &fx_inner, auto &jx_inner, auto &gx_outer) {
                gx_outer.template segment<OuterFunc::ORC>(this->outer_func_.seg_start_,
                                                          this->output_rows()) = adjvars;
                this->inner_func_.compute_jacobian_adjointgradient_adjointhessian(
                    x, fx_inner, jx_inner, adjgrad_, adjhess_, gx_outer);

                fx = fx_inner.template segment<OuterFunc::ORC>(this->outer_func_.seg_start_,
                                                               this->output_rows());
                this->inner_func_.accumulate_matrix_domain(
                    jx_,
                    jx_inner.template middleRows<OuterFunc::ORC>(this->outer_func_.seg_start_,
                                                                 this->output_rows()),
                    PlusEqualsAssignment());
            };

            const int inner_OR = this->inner_func_.output_rows();
            const int inner_IR = this->inner_func_.input_rows();
            const int outer_IR = this->outer_func_.input_rows();

            using IFXType = typename InnerFunc::template Output<Scalar>;
            using IJXType = typename InnerFunc::template Jacobian<Scalar>;
            using OGXType = typename OuterFunc::template Gradient<Scalar>;

            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<IFXType>(inner_OR, 1),
                tycho::utils::TempSpec<IJXType>(inner_OR, inner_IR),
                tycho::utils::TempSpec<OGXType>(outer_IR, 1));
        } else {
            auto Impl = [&](auto &fx_inner, auto &jx_inner, auto &jx_outer, auto &gx_outer,
                            auto &hx_outer, auto &Ht) {
                this->inner_func_.compute(x, fx_inner);
                this->outer_func_.compute_jacobian_adjointgradient_adjointhessian(
                    fx_inner, fx_, jx_outer, gx_outer, hx_outer, adjvars);
                fx_inner.setZero();
                this->inner_func_.compute_jacobian_adjointgradient_adjointhessian(
                    x, fx_inner, jx_inner, adjgrad_, adjhess_, gx_outer);
                this->inner_func_.right_jacobian_product(
                    jx_, jx_outer, jx_inner, DirectAssignment(), std::bool_constant<false>());

                if constexpr (!OuterFunc::is_linear_function) {

                    this->inner_func_.right_jacobian_product(
                        Ht, hx_outer, jx_inner, DirectAssignment(), std::bool_constant<false>());
                    if constexpr (InnerFunc::InputIsDynamic) {
                        int sds = this->inner_func_.sub_domains.cols();
                        if (sds == 0) {
                            this->inner_func_.right_jacobian_product(
                                adjhess, Ht.transpose(), jx_inner, PlusEqualsAssignment(),
                                std::bool_constant<false>());
                        } else {

                            for (int i = 0; i < sds; i++) {
                                int start = this->inner_func_.sub_domains(0, i);
                                int size = this->inner_func_.sub_domains(1, i);
                                this->inner_func_.right_jacobian_product(
                                    adjhess.middleRows(start, size),
                                    Ht.middleCols(start, size).transpose(), jx_inner,
                                    PlusEqualsAssignment(), std::bool_constant<false>());
                            }
                        }

                    } else {
                        Eigen::Matrix<Scalar, InnerFunc::IRC, InnerFunc::ORC> HTT = Ht.transpose();
                        constexpr int sds = InnerFunc::INPUT_DOMAIN::sub_domains.size();
                        tycho::utils::constexpr_for_loop(
                            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                            [&](auto i) {
                                constexpr int start =
                                    InnerFunc::INPUT_DOMAIN::sub_domains[i.value][0];
                                constexpr int size =
                                    InnerFunc::INPUT_DOMAIN::sub_domains[i.value][1];

                                this->inner_func_.right_jacobian_product(
                                    adjhess.template middleRows<size>(start, size),
                                    HTT.template middleRows<size>(start, size), jx_inner,
                                    PlusEqualsAssignment(), std::bool_constant<false>());
                            });
                    }
                }
            };

            const int inner_OR = this->inner_func_.output_rows();
            const int inner_IR = this->inner_func_.input_rows();
            const int outer_OR = this->outer_func_.output_rows();
            const int outer_IR = this->outer_func_.input_rows();

            using IFXType = typename InnerFunc::template Output<Scalar>;
            using IJXType = typename InnerFunc::template Jacobian<Scalar>;
            using OJXType = typename OuterFunc::template Jacobian<Scalar>;
            using OGXType = typename OuterFunc::template Gradient<Scalar>;
            using OHXType = typename OuterFunc::template Hessian<Scalar>;

            tycho::utils::BumpAllocator::allocate_run(
                Impl, tycho::utils::TempSpec<IFXType>(inner_OR, 1),
                tycho::utils::TempSpec<IJXType>(inner_OR, inner_IR),
                tycho::utils::TempSpec<OJXType>(outer_OR, outer_IR),
                tycho::utils::TempSpec<OGXType>(outer_IR, 1),
                tycho::utils::TempSpec<OHXType>(outer_IR, outer_IR),
                tycho::utils::TempSpec<IJXType>(inner_OR, inner_IR));
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

/// @internal
/// @brief Specialization of @ref NestedFunction_Impl for a @ref Segment inner function.
///
/// When the inner function is a plain segment, composition is a direct slice of
/// the input columns/rows rather than a full chain-rule product, so the
/// derivative assembly is specialized for that cheaper path.
///
/// @tparam Derived    CRTP most-derived type.
/// @tparam OuterFunc  Outer function applied to the sliced input.
/// @tparam IR         Parent input-row count.
/// @tparam OR         Segment output-row count.
/// @tparam ST         Segment start index.
/// @endinternal
template <class Derived, class OuterFunc, int IR, int OR, int ST>
struct NestedFunction_Impl<Derived, OuterFunc, Segment<IR, OR, ST>>
    : VectorFunction<Derived, IR, OuterFunc::ORC>, SegStartHolder<ST> {
    /// @internal
    /// @brief VectorFunction base: parent input rows to outer output rows.
    /// @endinternal
    using Base = VectorFunction<Derived, IR, OuterFunc::ORC>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    /// @internal
    /// @brief The segment inner-function type.
    /// @endinternal
    using InnerFunc = Segment<IR, OR, ST>;

    OuterFunc outer_func_; ///< @brief Outer function applied to the sliced input.

    /// @internal
    /// @brief Input-domain descriptor inherited from the segment.
    /// @endinternal
    using INPUT_DOMAIN = typename Segment<IR, OR, ST>::INPUT_DOMAIN;
    static constexpr bool is_linear_function =
        OuterFunc::is_linear_function &&
        InnerFunc::is_linear_function; ///< @brief True iff both outer and inner functions are linear.
    static constexpr bool is_vectorizable =
        OuterFunc::is_vectorizable &&
        InnerFunc::is_vectorizable; ///< @brief True iff both outer and inner functions are vectorizable.

    /// @internal
    /// @brief Default-construct an empty segment-composition.
    /// @endinternal
    NestedFunction_Impl() {}
    /// @internal
    /// @brief Construct from an outer function and a segment, validating sizes.
    /// @param ofunc  Outer function (applied to the sliced input).
    /// @param ifunc  The segment selecting the outer function's inputs.
    /// @endinternal
    NestedFunction_Impl(OuterFunc ofunc, Segment<IR, OR, ST> ifunc) {
        this->outer_func_ = ofunc;
        this->set_seg_start(ifunc.seg_start_);

        if (ifunc.output_rows() != this->outer_func_.input_rows()) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Math Error in NestedFunction/.eval method !!!\n"
                       "Output Size of InnerFunction (ORows = {0:}) does not match Input Size of "
                       "OuterFunction "
                       "(IRows = {1:}).\n",
                       ifunc.output_rows(), this->outer_func_.input_rows());
            throw std::invalid_argument("");
        }

        this->set_io_rows(ifunc.input_rows(), this->outer_func_.output_rows());
        this->set_input_domain(this->input_rows(), {ifunc.input_domain()});
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Evaluate the outer function on the sliced input.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Parent input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        const int size = this->outer_func_.input_rows();

        this->outer_func_.compute(x.template segment<InnerFunc::ORC>(this->seg_start_, size), fx);
    }
    /// @internal
    /// @brief Evaluate value and Jacobian, writing into the sliced columns.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Parent input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer; the sliced columns are written.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {

        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        const int size = this->outer_func_.input_rows();

        this->outer_func_.compute_jacobian(
            x.template segment<InnerFunc::ORC>(this->seg_start_, size), fx,
            jx.template middleCols<InnerFunc::ORC>(this->seg_start_, size));
    }
    /// @internal
    /// @brief Value, Jacobian, adjoint gradient, and adjoint Hessian via the sliced input.
    ///
    /// Forwards to the outer function on the selected columns/rows, scattering
    /// the derivatives into the sliced sub-blocks.
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Parent input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer; sliced columns written.
    /// @param  adjgrad_ Adjoint-gradient accumulator; sliced rows written.
    /// @param  adjhess_ Adjoint-Hessian accumulator; sliced block written.
    /// @param  adjvars  Adjoint variables seeding the reverse pass.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {

        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        const int size = this->outer_func_.input_rows();

        this->outer_func_.compute_jacobian_adjointgradient_adjointhessian(
            x.template segment<InnerFunc::ORC>(this->seg_start_, size), fx,
            jx.template middleCols<InnerFunc::ORC>(this->seg_start_, size),
            adjgrad.template segment<InnerFunc::ORC>(this->seg_start_, size),
            adjhess.template block<InnerFunc::ORC, InnerFunc::ORC>(
                this->seg_start_, this->seg_start_, this->outer_func_.input_rows(), size),
            adjvars);
    }

    /// @internal
    /// @brief Accumulate the outer Jacobian into the sliced columns of a target.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-Jacobian type.
    /// @tparam Assignment  Assignment policy.
    /// @param  target_  Destination block.
    /// @param  right    Source Jacobian.
    /// @param  assign   Assignment-policy tag instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        MatRef<JacType> right_ref = right.const_cast_derived();
        // typedef typename Target::Scalar Scalar;
        if constexpr (OR > 0 && IR > 0 && OuterFunc::IRC == -1) {
            Base::accumulate_jacobian(target_, right, assign);
        } else {
            const int size = this->outer_func_.input_rows();
            this->outer_func_.accumulate_jacobian(
                target.template middleCols<InnerFunc::ORC>(this->seg_start_, size),
                right_ref.template middleCols<InnerFunc::ORC>(this->seg_start_, size), assign);
        }
    }

    /// @internal
    /// @brief Accumulate the outer Hessian into the sliced block of a target.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-Hessian type.
    /// @tparam Assignment  Assignment policy.
    /// @param  target_  Destination block.
    /// @param  right    Source Hessian.
    /// @param  assign   Assignment-policy tag instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        MatRef<JacType> right_ref = right.const_cast_derived();
        // typedef typename Target::Scalar Scalar;

        if constexpr (OR > 0 && IR > 0 && OuterFunc::IRC == -1) {
            Base::accumulate_hessian(target_, right, assign);
        } else {
            const int size = this->outer_func_.input_rows();
            this->outer_func_.accumulate_hessian(
                target.template block<InnerFunc::ORC, InnerFunc::ORC>(this->seg_start_,
                                                                      this->seg_start_, size, size),
                right_ref.template block<InnerFunc::ORC, InnerFunc::ORC>(
                    this->seg_start_, this->seg_start_, size, size),
                assign);
        }
    }
};

////////////////////////////////////////////////////////////////////////
} // namespace tycho::vf
