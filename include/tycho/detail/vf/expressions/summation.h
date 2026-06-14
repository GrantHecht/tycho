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
/// @brief Forward declaration of the two-function sum/difference implementation.
/// @tparam Derived       CRTP most-derived type.
/// @tparam Func1         First addend.
/// @tparam Func2         Second addend (subtrahend if @p DoDifference).
/// @tparam DoDifference  When true, compute `f1 - f2` instead of `f1 + f2`.
/// @endinternal
template <class Derived, class Func1, class Func2, bool DoDifference> struct TwoFunctionSum_Impl;

/// @internal
/// @brief Forward declaration of the variadic multi-function sum implementation.
/// @tparam Derived  CRTP most-derived type.
/// @tparam Func1    First addend.
/// @tparam Func2    Second addend.
/// @tparam Funcs    Remaining addends.
/// @endinternal
template <class Derived, class Func1, class Func2, class... Funcs> struct MultiFunctionSum_Impl;

/// @brief Elementwise sum of two functions: `f(x) = f1(x) + f2(x)`.
/// @tparam Func1  First addend.
/// @tparam Func2  Second addend.
/// @ingroup vf
template <class Func1, class Func2>
struct TwoFunctionSum : TwoFunctionSum_Impl<TwoFunctionSum<Func1, Func2>, Func1, Func2, false> {
    /// @brief Sum/difference implementation base (difference disabled).
    using Base = TwoFunctionSum_Impl<TwoFunctionSum<Func1, Func2>, Func1, Func2, false>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);
};

/// @brief Elementwise difference of two functions: `f(x) = f1(x) - f2(x)`.
/// @tparam Func1  Minuend.
/// @tparam Func2  Subtrahend.
/// @ingroup vf
template <class Func1, class Func2>
struct FunctionDifference
    : TwoFunctionSum_Impl<FunctionDifference<Func1, Func2>, Func1, Func2, true> {
    /// @brief Sum/difference implementation base (difference enabled).
    using Base = TwoFunctionSum_Impl<FunctionDifference<Func1, Func2>, Func1, Func2, true>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);
};

/// @brief Elementwise sum of three or more functions: `f(x) = f1(x) + f2(x) + ...`.
/// @tparam Func1  First addend.
/// @tparam Func2  Second addend.
/// @tparam Funcs  Remaining addends.
/// @ingroup vf
template <class Func1, class Func2, class... Funcs>
struct MultiFunctionSum
    : MultiFunctionSum_Impl<MultiFunctionSum<Func1, Func2, Funcs...>, Func1, Func2, Funcs...> {
    /// @brief Multi-function sum implementation base.
    using Base =
        MultiFunctionSum_Impl<MultiFunctionSum<Func1, Func2, Funcs...>, Func1, Func2, Funcs...>;
    using Base::Base;
    VF_TYPE_ALIASES(Base);
};

/// @internal
/// @brief Trait: detects whether `F` is a @ref TwoFunctionSum or @ref FunctionDifference.
/// @tparam F  Candidate type.
/// @endinternal
template <class F> struct Is_SumorDiff : std::false_type {};

/// @internal
/// @brief Specialization recognizing @ref TwoFunctionSum as a sum/difference.
/// @tparam F1  First addend.
/// @tparam F2  Second addend.
/// @endinternal
template <class F1, class F2> struct Is_SumorDiff<TwoFunctionSum<F1, F2>> : std::true_type {};

/// @internal
/// @brief Specialization recognizing @ref FunctionDifference as a sum/difference.
/// @tparam F1  Minuend.
/// @tparam F2  Subtrahend.
/// @endinternal
template <class F1, class F2> struct Is_SumorDiff<FunctionDifference<F1, F2>> : std::true_type {};

/// @brief Build a @ref MultiFunctionSum from three or more functions.
/// @tparam Func1  First addend.
/// @tparam Func2  Second addend.
/// @tparam Funcs  Remaining addends.
/// @param  f1     First addend.
/// @param  f2     Second addend.
/// @param  fs     Remaining addends.
/// @return The summed VectorFunction.
/// @ingroup vf
template <class Func1, class Func2, class... Funcs> auto make_sum(Func1 f1, Func2 f2, Funcs... fs) {
    return MultiFunctionSum<Func1, Func2, Funcs...>(f1, f2, fs...);
}

/// @brief Build a @ref TwoFunctionSum from two functions.
/// @tparam Func1  First addend.
/// @tparam Func2  Second addend.
/// @param  f1     First addend.
/// @param  f2     Second addend.
/// @return The summed VectorFunction.
/// @ingroup vf
template <class Func1, class Func2> auto make_sum(Func1 f1, Func2 f2) {
    return TwoFunctionSum<Func1, Func2>(f1, f2);
}

/// @brief Build a sum from a tuple of functions.
///
/// A one-element tuple returns the function itself; otherwise a
/// @ref MultiFunctionSum over all tuple elements is returned.
///
/// @tparam Funcs  Function types in the tuple.
/// @param  fs     Tuple of functions to sum.
/// @return The single function or a @ref MultiFunctionSum over them.
/// @ingroup vf
template <class... Funcs> auto make_sum_tuple(std::tuple<Funcs...> fs) {
    constexpr int sz = sizeof...(Funcs);
    if constexpr (sz == 1) {
        return std::get<0>(fs);
    } else {
        return MultiFunctionSum<Funcs...>(fs);
    }
}

/// @brief Sum two functions (alias for the two-argument @ref make_sum).
/// @tparam Func1  First addend.
/// @tparam Func2  Second addend.
/// @param  f1     First addend.
/// @param  f2     Second addend.
/// @return A @ref TwoFunctionSum.
/// @ingroup vf
template <class Func1, class Func2> auto sum(Func1 f1, Func2 f2) {
    return TwoFunctionSum<Func1, Func2>(f1, f2);
}

/// @brief Sum three or more functions.
/// @tparam Func1  First addend.
/// @tparam Func2  Second addend.
/// @tparam Funcs  Remaining addends.
/// @param  f1     First addend.
/// @param  f2     Second addend.
/// @param  fs     Remaining addends.
/// @return A @ref MultiFunctionSum.
/// @ingroup vf
template <class Func1, class Func2, class... Funcs> auto sum(Func1 f1, Func2 f2, Funcs... fs) {
    return MultiFunctionSum<Func1, Func2, Funcs...>(f1, f2, std::tuple{fs...});
}

/// @brief Build a runtime-sized sum from a homogeneous vector of functions.
///
/// Groups the functions five at a time into nested sums, returning the common
/// @p RetType. Throws if @p funcs is empty.
///
/// @tparam RetType   Common return/storage type the sum collapses to.
/// @tparam FuncType  Element type of the input vector.
/// @param  funcs     Functions to sum.
/// @return The assembled summed function as @p RetType.
/// @throws std::invalid_argument if @p funcs is empty.
/// @ingroup vf
template <class RetType, class FuncType>
RetType make_dynamic_sum(const std::vector<FuncType> &funcs) {
    int size = funcs.size();
    RetType summed;
    if (size == 0) {
        throw std::invalid_argument(
            "make_dynamic_sum: cannot construct a sum from an empty vector of functions");
    } else if (size == 1) {
        summed = funcs[0];
    } else if (size == 2) {
        summed = make_sum(funcs[0], funcs[1]);
    } else if (size == 3) {
        summed = make_sum(funcs[0], funcs[1], funcs[2]);
    } else if (size == 4) {
        summed = make_sum(funcs[0], funcs[1], funcs[2], funcs[3]);
    } else if (size == 5) {
        summed = make_sum(funcs[0], funcs[1], funcs[2], funcs[3], funcs[4]);
    } else {
        RetType summed_t = make_sum(funcs[0], funcs[1], funcs[2], funcs[3], funcs[4]);
        std::vector<FuncType> nfuncs;
        for (int i = 5; i < funcs.size(); i++) {
            nfuncs.push_back(funcs[i]);
        }
        RetType rest = make_dynamic_sum<RetType, FuncType>(nfuncs);
        summed = make_sum(summed_t, rest);
    }
    return summed;
}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

/// @internal
/// @brief CRTP implementation of two-function sum/difference.
///
/// Computes `f1(x) ± f2(x)` and assembles derivatives by evaluating `f1` into
/// the output and accumulating `f2` (with the sign chosen by @p DoDifference)
/// using bump-allocated scratch. Several `constexpr` traits enable fast paths
/// when the addends are segments or nested sums.
///
/// @tparam Derived       CRTP most-derived type.
/// @tparam Func1         First addend.
/// @tparam Func2         Second addend (subtrahend if @p DoDifference).
/// @tparam DoDifference  When true, compute `f1 - f2` instead of `f1 + f2`.
/// @endinternal
template <class Derived, class Func1, class Func2, bool DoDifference>
struct TwoFunctionSum_Impl
    : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value,
                     SZ_MAX<Func1::ORC, Func2::ORC>::value, DenseDerivativeMode::Analytic> {
    /// @internal
    /// @brief VectorFunction base (analytic derivatives) for the sum.
    /// @endinternal
    using Base =
        VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value,
                       SZ_MAX<Func1::ORC, Func2::ORC>::value, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    Func1 func1; ///< @brief First addend.
    Func2 func2; ///< @brief Second addend (subtrahend if @p DoDifference).

    static constexpr bool func1_is_segment =
        Is_Segment<Func1>::value ||
        Is_ScaledSegment<Func1>::value; ///< @brief True if @p Func1 is a (scaled) segment.
    static constexpr bool func2_is_segment =
        Is_Segment<Func2>::value ||
        Is_ScaledSegment<Func2>::value; ///< @brief True if @p Func2 is a (scaled) segment.
    static constexpr bool is_sum_of_segments =
        func1_is_segment && func2_is_segment; ///< @brief True if both addends are segments.

    static constexpr bool func1_is_sumordiff =
        Is_SumorDiff<Func1>::value; ///< @brief True if @p Func1 is itself a sum/difference.
    static constexpr bool func2_is_sumordiff =
        Is_SumorDiff<Func2>::value; ///< @brief True if @p Func2 is itself a sum/difference.

    static constexpr bool is_sum_of_sums =
        func1_is_sumordiff ||
        func2_is_sumordiff; ///< @brief True if either addend is a sum/difference.
    static constexpr bool IsSegmentOp =
        Is_Segment<Func1>::value &&
        Is_Segment<Func2>::value; ///< @brief True if both addends are plain segments.

    static constexpr bool is_linear_function =
        Func1::is_linear_function &&
        Func2::is_linear_function; ///< @brief True if both addends are linear.
    static constexpr bool is_vectorizable =
        Func1::is_vectorizable &&
        Func2::is_vectorizable; ///< @brief True if both addends are vectorizable.

    /// @internal
    /// @brief Composite input-domain over both addends' domains.
    /// @endinternal
    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;

    /// @internal
    /// @brief Default-construct an empty sum.
    /// @endinternal
    TwoFunctionSum_Impl() {}
    /// @internal
    /// @brief Construct from two addends, validating matching I/O sizes.
    /// @param f1  First addend.
    /// @param f2  Second addend.
    /// @endinternal
    TwoFunctionSum_Impl(Func1 f1, Func2 f2) : func1(std::move(f1)), func2(std::move(f2)) {
        int irtemp = std::max(this->func1.input_rows(), this->func2.input_rows());

        if (this->func1.output_rows() != this->func2.output_rows()) {
            throw std::invalid_argument(
                fmt::format("TwoFunctionSum: output size mismatch (Func1 ORows={}, Func2 ORows={})",
                            this->func1.output_rows(), this->func2.output_rows()));
        }
        if (this->func1.input_rows() != this->func2.input_rows()) {
            throw std::invalid_argument(
                fmt::format("TwoFunctionSum: input size mismatch (Func1 IRows={}, Func2 IRows={})",
                            this->func1.input_rows(), this->func2.input_rows()));
        }

        this->set_io_rows(irtemp, this->func1.output_rows());
        this->set_input_domain(this->input_rows(), {func1.input_domain(), func2.input_domain()});
    }

    /// @brief Whether both addends are (runtime) linear.
    /// @return True if `func1` and `func2` are both linear.
    bool is_linear() const { return func1.is_linear() && func2.is_linear(); }

    /// @internal
    /// @brief Evaluate `fx = f1(x) ± f2(x)`.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &func2_fx) {
            this->func1.compute(x, fx);
            this->func2.compute(x, func2_fx);
            if constexpr (DoDifference) {
                fx -= func2_fx;
            } else {
                fx += func2_fx;
            }
        };

        const int orows = this->func2.output_rows();
        using FType = typename Func2::template Output<Scalar>;
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1));
    }
    /// @internal
    /// @brief Evaluate value and Jacobian, accumulating `f2` with the sum's sign.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &func2_fx, auto &func2_jx) {
            this->func1.compute_jacobian(x, fx, jx);
            this->func2.compute_jacobian(x, func2_fx, func2_jx);
            if constexpr (DoDifference) {
                fx -= func2_fx;
                this->func2.accumulate_jacobian(jx, func2_jx, MinusEqualsAssignment());
            } else {
                fx += func2_fx;
                this->func2.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
            }
        };

        const int orows = this->func2.output_rows();
        const int irows = this->func2.input_rows();
        using FType = typename Func2::template Output<Scalar>;
        using JType = typename Func2::template Jacobian<Scalar>;
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1),
                                                  tycho::utils::TempSpec<JType>(orows, irows));
    }
    /// @internal
    /// @brief Value, Jacobian, adjoint gradient, and adjoint Hessian of `f1 ± f2`.
    ///
    /// `f1` writes the primary outputs; `f2`'s Jacobian/gradient/Hessian are
    /// accumulated with the sum's sign (negated for a difference) using bump-
    /// allocated scratch.
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @param  adjgrad_ Adjoint-gradient accumulator.
    /// @param  adjhess_ Adjoint-Hessian accumulator.
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
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        //////////////////////////////////////////////

        auto Impl = [&](auto &func2_fx, auto &func2_jx, auto &func2_adjgrad, auto &func2_adjhess) {
            this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                        adjhess_, adjvars);

            this->func2.compute_jacobian_adjointgradient_adjointhessian(
                x, func2_fx, func2_jx, func2_adjgrad, func2_adjhess, adjvars);

            if constexpr (DoDifference) {
                fx -= func2_fx;
                this->func2.accumulate_jacobian(jx, func2_jx, MinusEqualsAssignment());
                this->func2.accumulate_gradient(adjgrad, func2_adjgrad, MinusEqualsAssignment());
                this->func2.accumulate_hessian(adjhess, func2_adjhess, MinusEqualsAssignment());

            } else {
                fx += func2_fx;
                this->func2.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
                this->func2.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());

                this->func2.accumulate_hessian(adjhess, func2_adjhess, PlusEqualsAssignment());
            }
        };

        const int orows = this->func2.output_rows();
        const int irows = this->func2.input_rows();

        using FType = typename Func2::template Output<Scalar>;
        using JType = typename Func2::template Jacobian<Scalar>;
        using GType = typename Func2::template Gradient<Scalar>;
        using HType = typename Func2::template Hessian<Scalar>;
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1),
                                                  tycho::utils::TempSpec<JType>(orows, irows),
                                                  tycho::utils::TempSpec<GType>(irows, 1),
                                                  tycho::utils::TempSpec<HType>(irows, irows));
    }

    ///////////////////////////////////////////////////////////////////////////////////////

    /// @internal
    /// @brief Chain-rule product, with a fast path when both addends are segments.
    ///
    /// When both addends are segments (or a sum-of-segments plus a segment) each
    /// writes its own disjoint columns, so the products are applied directly;
    /// otherwise it falls back to the base implementation.
    ///
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam Left        Concrete Eigen left-operand block type.
    /// @tparam Right       Concrete Eigen right-operand block type.
    /// @tparam Assignment  Assignment policy.
    /// @tparam Aliased     Whether target and operands may alias.
    /// @param  target_  Destination block.
    /// @param  left     Upstream Jacobian factor.
    /// @param  right    The sum's local Jacobian factor.
    /// @param  assign   Assignment-policy tag instance.
    /// @param  aliased  Aliasing-flag tag instance.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        if constexpr (is_sum_of_segments) {
            this->func1.right_jacobian_product(target_, left, right, assign, aliased);
            if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
                if constexpr (Aliased) {
                    // target and right alias: target columns hold the original
                    // right values, not zero.  Each segment writes to its own
                    // non-overlapping columns, so DirectAssignment is correct.
                    this->func2.right_jacobian_product(target_, left, right, DirectAssignment(),
                                                       aliased);
                } else {
                    this->func2.right_jacobian_product(target_, left, right, PlusEqualsAssignment(),
                                                       aliased);
                }
            } else {
                this->func2.right_jacobian_product(target_, left, right, assign, aliased);
            }
        } else if constexpr (func1_is_sumordiff && func2_is_segment) {
            if constexpr (Func1::is_sum_of_segments) {
                this->func1.right_jacobian_product(target_, left, right, assign, aliased);
                if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
                    if constexpr (Aliased) {
                        this->func2.right_jacobian_product(target_, left, right, DirectAssignment(),
                                                           aliased);
                    } else {
                        this->func2.right_jacobian_product(target_, left, right,
                                                           PlusEqualsAssignment(), aliased);
                    }
                } else {
                    this->func2.right_jacobian_product(target_, left, right, assign, aliased);
                }
            } else {
                Base::right_jacobian_product(target_, left, right, assign, aliased);
            }
        } else {
            Base::right_jacobian_product(target_, left, right, assign, aliased);
        }
    }
    /// @internal
    /// @brief Accumulate the sum's Jacobian, with a segment fast path.
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
        if constexpr (is_sum_of_segments) {
            this->func1.accumulate_jacobian(target_, right, assign);
            if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
                this->func2.accumulate_jacobian(target_, right, PlusEqualsAssignment());
            } else {
                this->func2.accumulate_jacobian(target_, right, assign);
            }
        } else {
            Base::accumulate_jacobian(target_, right, assign);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////
};

/// @internal
/// @brief CRTP implementation of a variadic multi-function sum.
///
/// Computes `f1(x) + f2(x) + ... ` and accumulates each addend's Jacobian,
/// gradient, and Hessian into the shared output buffers, with a segment fast
/// path for the right-Jacobian product. Uses bump-allocated scratch per addend.
///
/// @tparam Derived  CRTP most-derived type.
/// @tparam Func1    First addend.
/// @tparam Func2    Second addend.
/// @tparam Funcs    Remaining addends.
/// @endinternal
template <class Derived, class Func1, class Func2, class... Funcs>
struct MultiFunctionSum_Impl
    : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC, Funcs::IRC...>::value,
                     SZ_MAX<Func1::ORC, Func2::ORC, Funcs::ORC...>::value> {
    /// @internal
    /// @brief VectorFunction base sized from the max I/O of all addends.
    /// @endinternal
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC, Funcs::IRC...>::value,
                                SZ_MAX<Func1::ORC, Func2::ORC, Funcs::ORC...>::value>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    Func1 func1;                ///< @brief First addend.
    Func2 func2;                ///< @brief Second addend.
    std::tuple<Funcs...> funcs; ///< @brief Remaining addends.

    static constexpr bool is_linear_function =
        SZ_PROD<int(Func1::is_linear_function), int(Func2::is_linear_function),
                int(Funcs::is_linear_function)...>::value ==
        1; ///< @brief True if every addend is linear.

    static constexpr bool is_vectorizable =
        SZ_PROD<int(Func1::is_vectorizable), int(Func2::is_vectorizable),
                int(Funcs::is_vectorizable)...>::value ==
        1; ///< @brief True if every addend is vectorizable.

    static constexpr bool IsSumofSegments =
        SZ_PROD<int(Is_Segment<Func1>::value || Is_ScaledSegment<Func1>::value),
                int(Is_Segment<Func2>::value || Is_ScaledSegment<Func2>::value),
                int(Is_Segment<Funcs>::value || Is_ScaledSegment<Funcs>::value)...>::value ==
        1; ///< @brief True if every addend is a (scaled) segment.

    /// @internal
    /// @brief Composite input-domain over all addends' domains.
    /// @endinternal
    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN,
                        typename Funcs::INPUT_DOMAIN...>;

    /// @internal
    /// @brief Default-construct an empty multi-sum.
    /// @endinternal
    MultiFunctionSum_Impl() {}
    /// @internal
    /// @brief Construct from the addends as a parameter pack.
    /// @param f1  First addend.
    /// @param f2  Second addend.
    /// @param fs  Remaining addends.
    /// @endinternal
    MultiFunctionSum_Impl(Func1 f1, Func2 f2, Funcs... fs)
        : func1(std::move(f1)), func2(std::move(f2)), funcs(fs...) {
        int irtemp = std::max(this->func1.input_rows(), this->func2.input_rows());

        this->set_io_rows(irtemp, this->func1.output_rows());
        setdmn();
    }
    /// @internal
    /// @brief Construct from the first two addends and the rest as a tuple.
    /// @param f1  First addend.
    /// @param f2  Second addend.
    /// @param fs  Tuple of remaining addends.
    /// @endinternal
    MultiFunctionSum_Impl(Func1 f1, Func2 f2, std::tuple<Funcs...> fs)
        : func1(std::move(f1)), func2(std::move(f2)), funcs(fs) {
        int irtemp = std::max(this->func1.input_rows(), this->func2.input_rows());
        this->set_io_rows(irtemp, this->func1.output_rows());
        setdmn();
    }

    /// @internal
    /// @brief Construct from a tuple containing all addends.
    /// @param fs  Tuple of all addends (first, second, then the rest).
    /// @endinternal
    MultiFunctionSum_Impl(std::tuple<Func1, Func2, Funcs...> fs) {
        this->func1 = std::get<0>(fs);
        this->func2 = std::get<1>(fs);
        tycho::utils::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sizeof...(Funcs)>(),
            [&](auto i) { std::get<i.value>(this->funcs) = std::get<i.value + 2>(fs); });

        int irtemp = std::max(this->func1.input_rows(), this->func2.input_rows());
        this->set_io_rows(irtemp, this->func1.output_rows());
        setdmn();
    }

    /// @internal
    /// @brief Build the composite input domain and validate all addend I/O sizes.
    ///
    /// Pushes each addend's input domain into the runtime domain list and throws
    /// if any addend's input or output size disagrees with the first.
    /// @endinternal
    void setdmn() {
        std::vector<DomainMatrix> tmp;
        tmp.push_back(func1.input_domain());
        tmp.push_back(func2.input_domain());

        if (this->func1.output_rows() != this->func2.output_rows()) {
            throw std::invalid_argument(fmt::format(
                "MultiFunctionSum: output size mismatch (Func1 ORows={}, Func2 ORows={})",
                this->func1.output_rows(), this->func2.output_rows()));
        }
        if (this->func1.input_rows() != this->func2.input_rows()) {
            throw std::invalid_argument(fmt::format(
                "MultiFunctionSum: input size mismatch (Func1 IRows={}, Func2 IRows={})",
                this->func1.input_rows(), this->func2.input_rows()));
        }

        tycho::utils::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sizeof...(Funcs)>(),
            [&](auto i) {
                tmp.push_back(std::get<i.value>(this->funcs).input_domain());
                if (this->func1.output_rows() != std::get<i.value>(this->funcs).output_rows()) {
                    throw std::invalid_argument(
                        fmt::format("MultiFunctionSum: output size mismatch "
                                    "(Func1 ORows={}, Func{} ORows={})",
                                    this->func1.output_rows(), i.value + 3,
                                    std::get<i.value>(this->funcs).output_rows()));
                }
                if (this->func1.input_rows() != std::get<i.value>(this->funcs).input_rows()) {
                    throw std::invalid_argument(
                        fmt::format("MultiFunctionSum: input size mismatch "
                                    "(Func1 IRows={}, Func{} IRows={})",
                                    this->func1.input_rows(), i.value + 3,
                                    std::get<i.value>(this->funcs).input_rows()));
                }
            });
        this->set_input_domain(this->input_rows(), tmp);
    }

    /// @internal
    /// @brief Evaluate the sum of all addends: `fx = f1(x) + f2(x) + ...`.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &func2_fx) {
            this->func1.compute(x, fx);
            func2_fx.setZero();
            this->func2.compute(x, func2_fx);
            fx += func2_fx;

            tycho::utils::tuple_for_each(this->funcs, [&](const auto &funci) {
                func2_fx.setZero();
                funci.compute(x, func2_fx);
                fx += func2_fx;
            });
        };

        const int orows = this->func2.output_rows();
        using FType = typename Func2::template Output<Scalar>;
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1));
    }
    /// @internal
    /// @brief Evaluate value and Jacobian of the sum of all addends.
    ///
    /// Each addend's Jacobian is accumulated into @p jx_, zeroing only the
    /// sub-domains it touches between addends to avoid redundant clears.
    ///
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &func2_fx, auto &func2_jx) {
            this->func1.compute_jacobian(x, fx, jx);
            this->func2.compute_jacobian(x, func2_fx, func2_jx);
            fx += func2_fx;
            this->func2.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());

            tycho::utils::tuple_for_each(this->funcs, [&](const auto &funci) {
                func2_fx.setZero();

                typedef typename std::remove_reference<decltype(funci)>::type FunciType;
                if constexpr (FunciType::InputIsDynamic) {
                    func2_jx.setZero();
                } else {
                    constexpr int sds = FunciType::INPUT_DOMAIN::sub_domains.size();
                    tycho::utils::constexpr_for_loop(
                        std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                        [&](auto i) {
                            constexpr int Start1 = FunciType::INPUT_DOMAIN::sub_domains[i.value][0];
                            constexpr int Size1 = FunciType::INPUT_DOMAIN::sub_domains[i.value][1];
                            func2_jx.template middleCols<Size1>(Start1, Size1).setZero();
                        });
                }

                funci.compute_jacobian(x, func2_fx, func2_jx);
                fx += func2_fx;
                funci.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
            });
        };

        const int orows = this->func2.output_rows();
        const int irows = this->func2.input_rows();
        using FType = typename Func2::template Output<Scalar>;
        using JType = typename Func2::template Jacobian<Scalar>;
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1),
                                                  tycho::utils::TempSpec<JType>(orows, irows));
    }
    /// @internal
    /// @brief Value, Jacobian, adjoint gradient, and adjoint Hessian of the multi-sum.
    ///
    /// Accumulates every addend's contribution into the shared adjoint buffers,
    /// skipping the Hessian for linear addends and zeroing only the touched
    /// sub-domains between addends. Uses bump-allocated scratch.
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @param  adjgrad_ Adjoint-gradient accumulator.
    /// @param  adjhess_ Adjoint-Hessian accumulator.
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
        MatRef<JacType> jx = jx_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        //
        auto Impl = [&](auto &func2_fx, auto &func2_jx, auto &func2_adjgrad, auto &func2_adjhess) {
            this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess,
                                                                        adjvars);

            this->func2.compute_jacobian_adjointgradient_adjointhessian(
                x, func2_fx, func2_jx, func2_adjgrad, func2_adjhess, adjvars);

            fx += func2_fx;

            this->func2.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
            this->func2.accumulate_gradient(adjgrad_, func2_adjgrad, PlusEqualsAssignment());
            this->func2.accumulate_hessian(adjhess_, func2_adjhess, PlusEqualsAssignment());

            tycho::utils::tuple_for_each(this->funcs, [&](const auto &funci) {
                func2_fx.setZero();
                func2_adjgrad.setZero();

                typedef typename std::remove_reference<decltype(funci)>::type FunciType;

                // func2_jx.setZero();
                funci.zero_matrix_domain(func2_jx);
                if constexpr (!FunciType::is_linear_function) {
                    // func2_adjhess.setZero();
                    funci.zero_matrix_domain(func2_adjhess);
                }

                funci.compute_jacobian_adjointgradient_adjointhessian(
                    x, func2_fx, func2_jx, func2_adjgrad, func2_adjhess, adjvars);
                fx += func2_fx;
                funci.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
                funci.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());
                if constexpr (!FunciType::is_linear_function)
                    funci.accumulate_hessian(adjhess, func2_adjhess, PlusEqualsAssignment());
            });
        };

        const int orows = this->func2.output_rows();
        const int irows = this->func2.input_rows();

        using FType = typename Func2::template Output<Scalar>;
        using JType = typename Func2::template Jacobian<Scalar>;
        using GType = typename Func2::template Gradient<Scalar>;
        using HType = typename Func2::template Hessian<Scalar>;
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1),
                                                  tycho::utils::TempSpec<JType>(orows, irows),
                                                  tycho::utils::TempSpec<GType>(irows, 1),
                                                  tycho::utils::TempSpec<HType>(irows, irows));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Chain-rule product for the multi-sum, with an all-segments fast path.
    ///
    /// When every addend is a segment each writes its own disjoint columns, so
    /// the products are applied directly across `func1`, `func2`, and the tuple;
    /// otherwise it falls back to the base implementation.
    ///
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam Left        Concrete Eigen left-operand block type.
    /// @tparam Right       Concrete Eigen right-operand block type.
    /// @tparam Assignment  Assignment policy.
    /// @tparam Aliased     Whether target and operands may alias.
    /// @param  target_  Destination block.
    /// @param  left     Upstream Jacobian factor.
    /// @param  right    The sum's local Jacobian factor.
    /// @param  assign   Assignment-policy tag instance.
    /// @param  aliased  Aliasing-flag tag instance.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        if constexpr (IsSumofSegments) {
            this->func1.right_jacobian_product(target_, left, right, assign, aliased);
            if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
                if constexpr (Aliased) {
                    // target and right alias: target columns hold the original
                    // right values, not zero.  Each segment writes to its own
                    // non-overlapping columns, so DirectAssignment is correct.
                    this->func2.right_jacobian_product(target_, left, right, DirectAssignment(),
                                                       aliased);
                    tycho::utils::tuple_for_each(this->funcs, [&](const auto &func) {
                        func.right_jacobian_product(target_, left, right, DirectAssignment(),
                                                    aliased);
                    });
                } else {
                    this->func2.right_jacobian_product(target_, left, right, PlusEqualsAssignment(),
                                                       aliased);
                    tycho::utils::tuple_for_each(this->funcs, [&](const auto &func) {
                        func.right_jacobian_product(target_, left, right, PlusEqualsAssignment(),
                                                    aliased);
                    });
                }
            } else {
                this->func2.right_jacobian_product(target_, left, right, assign, aliased);
                tycho::utils::tuple_for_each(this->funcs, [&](const auto &func) {
                    func.right_jacobian_product(target_, left, right, assign, aliased);
                });
            }
        } else {
            Base::right_jacobian_product(target_, left, right, assign, aliased);
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
