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
#include "tycho/detail/vf/core/vector_function_concepts.h"

namespace tycho::vf {

/// @internal
/// @brief Forward declaration of the two-output stacking implementation.
/// @tparam Derived  CRTP most-derived type.
/// @tparam Func1    First (top) stacked function.
/// @tparam Func2    Second (bottom) stacked function.
/// @endinternal
template <class Derived, class Func1, class Func2>
    requires Stackable<Func1, Func2>
struct StackTwoOutputs_Impl;

/// @brief Vertically stacks several functions' outputs: `f(x) = [f1(x); f2(x); ...]`.
///
/// All stacked functions share the same input @p x; their outputs are
/// concatenated into one output vector. The variadic form recursively pairs the
/// head two functions (`StackedOutputs<Func1, Func2>`) with the stacked tail
/// (`StackedOutputs<Funcs...>`). The return type of `vf.stack(...)`.
///
/// @tparam Func1  First (top) stacked function.
/// @tparam Func2  Second stacked function.
/// @tparam Funcs  Remaining stacked functions.
/// @ingroup vf
template <class Func1, class Func2, class... Funcs>
    requires MutuallyStackable<Func1, Func2, Funcs...>
struct StackedOutputs
    : StackTwoOutputs_Impl<StackedOutputs<Func1, Func2, Funcs...>, StackedOutputs<Func1, Func2>,
                           StackedOutputs<Funcs...>> {
    /// @brief Two-output stacking base pairing the head with the tail stack.
    using Base = StackTwoOutputs_Impl<StackedOutputs<Func1, Func2, Funcs...>,
                                      StackedOutputs<Func1, Func2>, StackedOutputs<Funcs...>>;
    using Base::Base;
    /// @brief Default-construct an empty stack.
    StackedOutputs() {};
    /// @brief Construct from the stacked functions.
    /// @param f1  First function.
    /// @param f2  Second function.
    /// @param fs  Remaining functions.
    StackedOutputs(Func1 f1, Func2 f2, Funcs... fs)
        : Base(StackedOutputs<Func1, Func2>(f1, f2), StackedOutputs<Funcs...>(fs...)) {};
};

/// @brief Three-function stacking specialization: `[f1(x); f2(x); f3(x)]`.
/// @tparam Func1  First stacked function.
/// @tparam Func2  Second stacked function.
/// @tparam Func3  Third stacked function.
/// @ingroup vf
template <class Func1, class Func2, class Func3>
    requires MutuallyStackable<Func1, Func2, Func3>
struct StackedOutputs<Func1, Func2, Func3>
    : StackTwoOutputs_Impl<StackedOutputs<Func1, Func2, Func3>, StackedOutputs<Func1, Func2>,
                           Func3> {
    /// @brief Two-output stacking base pairing `[f1;f2]` with `f3`.
    using Base = StackTwoOutputs_Impl<StackedOutputs<Func1, Func2, Func3>,
                                      StackedOutputs<Func1, Func2>, Func3>;
    using Base::Base;
    /// @brief Default-construct an empty stack.
    StackedOutputs() {};
    /// @brief Construct from the three stacked functions.
    /// @param f1  First function.
    /// @param f2  Second function.
    /// @param f3  Third function.
    StackedOutputs(Func1 f1, Func2 f2, Func3 f3)
        : Base(StackedOutputs<Func1, Func2>(f1, f2), f3) {};
};

/// @brief Two-function stacking specialization: `[f1(x); f2(x)]`.
/// @tparam Func1  First (top) stacked function.
/// @tparam Func2  Second (bottom) stacked function.
/// @ingroup vf
template <class Func1, class Func2>
    requires Stackable<Func1, Func2>
struct StackedOutputs<Func1, Func2>
    : StackTwoOutputs_Impl<StackedOutputs<Func1, Func2>, Func1, Func2> {
    /// @brief Two-output stacking base over @p Func1 and @p Func2.
    using Base = StackTwoOutputs_Impl<StackedOutputs<Func1, Func2>, Func1, Func2>;
    using Base::Base;
};

/// @brief Stack several functions' outputs into one VectorFunction.
/// @tparam Func1  First stacked function.
/// @tparam Func2  Second stacked function.
/// @tparam Funcs  Remaining stacked functions.
/// @param  f1     First function.
/// @param  f2     Second function.
/// @param  fs     Remaining functions.
/// @return A @ref StackedOutputs concatenating all of their outputs.
/// @ingroup vf
template <class Func1, class Func2, class... Funcs>
    requires MutuallyStackable<Func1, Func2, Funcs...>
StackedOutputs<Func1, Func2, Funcs...> stack(Func1 f1, Func2 f2, Funcs... fs) {
    return StackedOutputs<Func1, Func2, Funcs...>(f1, f2, fs...);
}

/// @brief Build a runtime-sized stack from a homogeneous vector of functions.
///
/// Recursively groups the functions five at a time into nested
/// @ref StackedOutputs, returning the common @p RetType (typically a
/// type-erased VectorFunction).
///
/// @tparam RetType   Common return/storage type the stack collapses to.
/// @tparam FuncType  Element type of the input vector.
/// @param  funcs     Functions to stack, in output order.
/// @return The assembled stacked function as @p RetType.
/// @ingroup vf
template <class RetType, class FuncType>
RetType make_dynamic_stack(const std::vector<FuncType> &funcs) {
    int size = funcs.size();
    RetType stacked;
    if (size == 0) {
    } else if (size == 1) {
        stacked = funcs[0];
    } else if (size == 2) {
        stacked = StackedOutputs{funcs[0], funcs[1]};
    } else if (size == 3) {
        stacked = StackedOutputs{funcs[0], funcs[1], funcs[2]};
    } else if (size == 4) {
        stacked = StackedOutputs{funcs[0], funcs[1], funcs[2], funcs[3]};
    } else if (size == 5) {
        stacked = StackedOutputs{funcs[0], funcs[1], funcs[2], funcs[3], funcs[4]};
    } else {
        RetType stacked_t = StackedOutputs{funcs[0], funcs[1], funcs[2], funcs[3], funcs[4]};
        std::vector<FuncType> nfuncs;
        for (int i = 5; i < funcs.size(); i++) {
            nfuncs.push_back(funcs[i]);
        }
        RetType rest = make_dynamic_stack<RetType, FuncType>(nfuncs);
        stacked = StackedOutputs{stacked_t, rest};
    }
    return stacked;
}

/// @internal
/// @brief CRTP implementation stacking exactly two functions' outputs.
///
/// Output rows are the sum of the two functions' output rows; both read the
/// same input. Value, Jacobian, and adjoint-gradient/Hessian are assembled by
/// writing each function into its own contiguous row block.
///
/// @tparam Derived  CRTP most-derived type (a @ref StackedOutputs).
/// @tparam Func1    First (top) function.
/// @tparam Func2    Second (bottom) function.
/// @endinternal
template <class Derived, class Func1, class Func2>
    requires Stackable<Func1, Func2>
struct StackTwoOutputs_Impl : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                             SZ_SUM<Func1::ORC, Func2::ORC>::value> {
    /// @internal
    /// @brief VectorFunction base: shared input rows, summed output rows.
    /// @endinternal
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                SZ_SUM<Func1::ORC, Func2::ORC>::value>;
    using Base::compute;

    VF_TYPE_ALIASES(Base);

    static constexpr bool is_linear_function =
        Func1::is_linear_function &&
        Func2::is_linear_function; ///< @brief True if both stacked functions are linear.
    static constexpr bool is_vectorizable =
        Func1::is_vectorizable &&
        Func2::is_vectorizable; ///< @brief True if both stacked functions are vectorizable.

    Func1 func1; ///< @brief First (top) stacked function.
    Func2 func2; ///< @brief Second (bottom) stacked function.

    /// @internal
    /// @brief Composite input-domain over both stacked functions' domains.
    /// @endinternal
    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;

    /// @internal
    /// @brief Default-construct an empty two-output stack.
    /// @endinternal
    StackTwoOutputs_Impl() {}
    /// @internal
    /// @brief Construct from two functions, validating matching input sizes.
    /// @param f1  First (top) function.
    /// @param f2  Second (bottom) function.
    /// @endinternal
    StackTwoOutputs_Impl(Func1 f1, Func2 f2) : func1(std::move(f1)), func2(std::move(f2)) {
        int irtemp = std::max(this->func1.input_rows(), this->func2.input_rows());
        this->set_io_rows(irtemp, this->func1.output_rows() + this->func2.output_rows());

        if (this->func1.input_rows() != this->func2.input_rows()) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Math Error in StackOutputs/vf.stack method !!!\n"
                       "Input Size of Func1 (IRows = {0:}) does not match Input Size of Func2 "
                       "(IRows = {1:}).\n",
                       this->func1.input_rows(), this->func2.input_rows());
            throw std::invalid_argument("");
        }

        this->set_input_domain(this->input_rows(), {func1.input_domain(), func2.input_domain()});
    }

    /// @brief Whether both stacked functions are (runtime) linear.
    /// @return True if `func1` and `func2` are both linear.
    bool is_linear() const { return func1.is_linear() && func2.is_linear(); }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Evaluate both functions into their stacked output blocks.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Shared input vector.
    /// @param  fx_      Stacked output buffer (size = `output_rows()`).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        this->func1.compute(x, fx.template segment<Func1::ORC>(0, this->func1.output_rows()));
        this->func2.compute(x, fx.template segment<Func2::ORC>(this->func1.output_rows(),
                                                               this->func2.output_rows()));
    }
    /// @internal
    /// @brief Evaluate both functions' values and Jacobian row blocks.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Shared input vector.
    /// @param  fx_      Stacked output buffer.
    /// @param  jx_      Stacked Jacobian buffer (top/bottom row blocks).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        this->func1.compute_jacobian(x,
                                     fx.template segment<Func1::ORC>(0, this->func1.output_rows()),
                                     jx.template topRows<Func1::ORC>(this->func1.output_rows()));

        this->func2.compute_jacobian(
            x,
            fx.template segment<Func2::ORC>(this->func1.output_rows(), this->func2.output_rows()),
            jx.template bottomRows<Func2::ORC>(this->func2.output_rows()));
    }
    /// @internal
    /// @brief Value, Jacobian, adjoint gradient, and adjoint Hessian of the stack.
    ///
    /// Each function writes its own output/Jacobian row block; the adjoint
    /// gradient and Hessian are accumulated from both, with a fast path when one
    /// function is linear (its Hessian contribution is skipped) and bump-
    /// allocated scratch for the second function's adjoint buffers.
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Shared input vector.
    /// @param  fx_      Stacked output buffer.
    /// @param  jx_      Stacked Jacobian buffer.
    /// @param  adjgrad_ Adjoint-gradient accumulator.
    /// @param  adjhess_ Adjoint-Hessian accumulator.
    /// @param  adjvars  Adjoint variables, sliced per stacked function.
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

        auto Impl = [&](auto &func2_adjgrad, auto &func2_adjhess) {
            if constexpr (Func1::is_generic_function && Func2::is_generic_function) {
                if (this->func1.is_linear()) {
                    this->func1.compute_jacobian_adjointgradient_adjointhessian(
                        x, fx.template segment<Func1::ORC>(0, this->func1.output_rows()),
                        jx.template topRows<Func1::ORC>(this->func1.output_rows()), func2_adjgrad,
                        func2_adjhess,
                        adjvars.template segment<Func1::ORC>(0, this->func1.output_rows()));

                    this->func2.compute_jacobian_adjointgradient_adjointhessian(
                        x,
                        fx.template segment<Func2::ORC>(this->func1.output_rows(),
                                                        this->func2.output_rows()),
                        jx.template bottomRows<Func2::ORC>(this->func2.output_rows()), adjgrad,
                        adjhess,
                        adjvars.template segment<Func2::ORC>(this->func1.output_rows(),
                                                             this->func2.output_rows()));

                    this->func1.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());

                } else {
                    this->func1.compute_jacobian_adjointgradient_adjointhessian(
                        x, fx.template segment<Func1::ORC>(0, this->func1.output_rows()),
                        jx.template topRows<Func1::ORC>(this->func1.output_rows()), adjgrad,
                        adjhess,
                        adjvars.template segment<Func1::ORC>(0, this->func1.output_rows()));

                    this->func2.compute_jacobian_adjointgradient_adjointhessian(
                        x,
                        fx.template segment<Func2::ORC>(this->func1.output_rows(),
                                                        this->func2.output_rows()),
                        jx.template bottomRows<Func2::ORC>(this->func2.output_rows()),
                        func2_adjgrad, func2_adjhess,
                        adjvars.template segment<Func2::ORC>(this->func1.output_rows(),
                                                             this->func2.output_rows()));

                    // if (!this->func2.is_linear())
                    this->func2.accumulate_hessian(adjhess, func2_adjhess, PlusEqualsAssignment());
                    this->func2.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());
                }

            } else {
                this->func1.compute_jacobian_adjointgradient_adjointhessian(
                    x, fx.template segment<Func1::ORC>(0, this->func1.output_rows()),
                    jx.template topRows<Func1::ORC>(this->func1.output_rows()), adjgrad, adjhess,
                    adjvars.template segment<Func1::ORC>(0, this->func1.output_rows()));

                this->func2.compute_jacobian_adjointgradient_adjointhessian(
                    x,
                    fx.template segment<Func2::ORC>(this->func1.output_rows(),
                                                    this->func2.output_rows()),
                    jx.template bottomRows<Func2::ORC>(this->func2.output_rows()), func2_adjgrad,
                    func2_adjhess,
                    adjvars.template segment<Func2::ORC>(this->func1.output_rows(),
                                                         this->func2.output_rows()));

                this->func2.accumulate_hessian(adjhess, func2_adjhess, PlusEqualsAssignment());
                this->func2.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());
            }
        };

        const int irows = this->func2.input_rows();
        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<typename Func2::template Gradient<Scalar>>(irows, 1),
            tycho::utils::TempSpec<typename Func2::template Hessian<Scalar>>(irows, irows));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Chain-rule product split across the two stacked functions' row blocks.
    ///
    /// Each function applies its right-Jacobian product to the matching
    /// column/row block of @p left and @p right; a diagonal @p left falls back
    /// to the base implementation.
    ///
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam Left        Concrete Eigen left-operand block type.
    /// @tparam Right       Concrete Eigen right-operand block type.
    /// @tparam Assignment  Assignment policy.
    /// @tparam Aliased     Whether target and operands may alias.
    /// @param  target_  Destination block.
    /// @param  left     Upstream Jacobian factor.
    /// @param  right    Block holding the stacked Jacobian rows.
    /// @param  assign   Assignment-policy tag instance.
    /// @param  aliased  Aliasing-flag tag instance.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            Base::right_jacobian_product(target_, left, right, assign, aliased);

        } else {
            MatRef<Right> right_ref = right.const_cast_derived();
            MatRef<Left> left_ref = left.const_cast_derived();

            this->func1.right_jacobian_product(
                target_, left_ref.template leftCols<Func1::ORC>(this->func1.output_rows()),
                right_ref.template topRows<Func1::ORC>(this->func1.output_rows()), assign, aliased);
            if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
                this->func2.right_jacobian_product(
                    target_, left_ref.template rightCols<Func2::ORC>(this->func2.output_rows()),
                    right_ref.template bottomRows<Func2::ORC>(this->func2.output_rows()),
                    PlusEqualsAssignment(), aliased);
            } else {
                this->func2.right_jacobian_product(
                    target_, left_ref.template rightCols<Func2::ORC>(this->func2.output_rows()),
                    right_ref.template bottomRows<Func2::ORC>(this->func2.output_rows()), assign,
                    aliased);
            }
        }
    }

    /// @internal
    /// @brief Scale each function's Jacobian row block in place.
    /// @tparam Target  Concrete Eigen target/output block type.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param  target_ Stacked Jacobian to scale.
    /// @param  s       Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        MatRef<Target> target = target_.const_cast_derived();
        this->func1.scale_jacobian(target.template topRows<Func1::ORC>(this->func1.output_rows()),
                                   s);
        this->func2.scale_jacobian(
            target.template bottomRows<Func2::ORC>(this->func2.output_rows()), s);
    }

    /// @internal
    /// @brief Accumulate each function's Jacobian into its target row block.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-Jacobian type.
    /// @tparam Assignment  Assignment policy.
    /// @param  target_  Destination stacked Jacobian.
    /// @param  right    Source stacked Jacobian.
    /// @param  assign   Assignment-policy tag instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        MatRef<JacType> right_ref = right.const_cast_derived();

        // typedef typename Target::Scalar Scalar;

        this->func1.accumulate_jacobian(
            target.template topRows<Func1::ORC>(this->func1.output_rows()),
            right_ref.template topRows<Func1::ORC>(this->func1.output_rows()), assign);
        this->func2.accumulate_jacobian(
            target.template bottomRows<Func2::ORC>(this->func2.output_rows()),
            right_ref.template bottomRows<Func2::ORC>(this->func2.output_rows()), assign);
    }

    /// @internal
    /// @brief Accumulate the stack's adjoint Hessian.
    ///
    /// Stacking does not couple the two functions in the Hessian (they share the
    /// input but write disjoint outputs), so a linear function contributes
    /// nothing and the work is delegated to whichever function is non-linear.
    ///
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-Hessian type.
    /// @tparam Assignment  Assignment policy.
    /// @param  target_  Destination Hessian.
    /// @param  right    Source Hessian.
    /// @param  assign   Assignment-policy tag instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        if constexpr (Func1::is_linear_function) {
            this->func2.accumulate_hessian(target_, right, assign);
        } else if constexpr (Func2::is_linear_function) {
            this->func1.accumulate_hessian(target_, right, assign);
        } else {
            Base::accumulate_hessian(target_, right, assign);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

/// @brief Runtime-sized vertical stack of a homogeneous list of functions.
///
/// Like @ref StackedOutputs but the number of stacked functions is a runtime
/// `std::vector` rather than a variadic template, so the output size is dynamic.
/// All functions must share the same input size; outputs are concatenated.
///
/// @tparam Func  Common type of every stacked function.
/// @ingroup vf
template <class Func>
struct DynamicStackedOutputs : VectorFunction<DynamicStackedOutputs<Func>, Func::IRC, -1> {
    /// @brief VectorFunction base with dynamic (`-1`) output rows.
    using Base = VectorFunction<DynamicStackedOutputs<Func>, Func::IRC, -1>;
    using Base::compute;

    VF_TYPE_ALIASES(Base);

    static constexpr bool is_linear_function =
        Func::is_linear_function; ///< @brief True if the element function type is linear.
    static constexpr bool is_vectorizable =
        Func::is_vectorizable; ///< @brief True if the element function type is vectorizable.

    std::vector<Func> funcs; ///< @brief The stacked functions, in output order.

    /// @brief Input-domain descriptor inherited from the element function type.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    /// @brief Default-construct an empty dynamic stack.
    DynamicStackedOutputs() {}
    /// @brief Construct from a vector of functions, validating matching input sizes.
    /// @param funcs  Functions to stack, in output order.
    DynamicStackedOutputs(const std::vector<Func> &funcs) : funcs(funcs) {

        if (this->funcs.size() == 0) {
            throw std::invalid_argument("Empty List passed to Dynamic Stack");
        }
        int ortemp = 0;
        int irtemp = this->funcs[0].input_rows();

        std::vector<DomainMatrix> dmn;

        for (auto &func : this->funcs) {
            ortemp += func.output_rows();
            dmn.push_back(func.input_domain());

            if (!func.is_linear())
                this->_linear = false;

            if (func.input_rows() != irtemp) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Math Error in StackOutputs/vf.stack method !!!\n"
                           "Input Size of all Functions must match.\n");
                throw std::invalid_argument("");
            }
        }

        this->set_io_rows(irtemp, ortemp);
        this->set_input_domain(this->input_rows(), dmn);
    }

    /// @brief Whether every stacked function is (runtime) linear.
    /// @return True if the stack is linear.
    bool is_linear() const { return this->_linear; }
    bool _linear = false; ///< @brief Cached linearity flag for the whole stack.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Evaluate each function into its consecutive output block.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Shared input vector.
    /// @param  fx_      Stacked output buffer.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        int start = 0;
        for (auto &func : this->funcs) {
            int orows = func.output_rows();
            func.compute(x, fx.template segment<Func::ORC>(start, orows));
            start += orows;
        }
    }
    /// @internal
    /// @brief Evaluate each function's value and Jacobian row block.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Shared input vector.
    /// @param  fx_      Stacked output buffer.
    /// @param  jx_      Stacked Jacobian buffer (consecutive row blocks).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        int start = 0;
        for (auto &func : this->funcs) {
            int orows = func.output_rows();
            func.compute_jacobian(x, fx.template segment<Func::ORC>(start, orows),
                                  jx.template middleRows<Func::ORC>(start, orows));
            start += orows;
        }
    }
    /// @internal
    /// @brief Value, Jacobian, adjoint gradient, and adjoint Hessian of the dynamic stack.
    ///
    /// Loops over the functions, writing each into its output/Jacobian block and
    /// accumulating its adjoint gradient/Hessian (using bump-allocated scratch
    /// for all but the first function).
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Shared input vector.
    /// @param  fx_      Stacked output buffer.
    /// @param  jx_      Stacked Jacobian buffer.
    /// @param  adjgrad_ Adjoint-gradient accumulator.
    /// @param  adjhess_ Adjoint-Hessian accumulator.
    /// @param  adjvars  Adjoint variables, sliced per stacked function.
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

        auto Impl = [&](auto &func_adjgrad, auto &func_adjhess) {
            int start = 0;
            for (auto &func : this->funcs) {
                int orows = func.output_rows();

                if (start == 0) {
                    func.compute_jacobian_adjointgradient_adjointhessian(
                        x, fx.template segment<Func::ORC>(start, orows),
                        jx.template middleRows<Func::ORC>(start, orows), adjgrad, adjhess,
                        adjvars.template segment<Func::ORC>(start, orows));
                } else {
                    func.compute_jacobian_adjointgradient_adjointhessian(
                        x, fx.template segment<Func::ORC>(start, orows),
                        jx.template middleRows<Func::ORC>(start, orows), func_adjgrad, func_adjhess,
                        adjvars.template segment<Func::ORC>(start, orows));

                    func.accumulate_hessian(adjhess, func_adjhess, PlusEqualsAssignment());
                    func.accumulate_gradient(adjgrad, func_adjgrad, PlusEqualsAssignment());

                    func.zero_matrix_domain(func_adjhess);
                    func_adjgrad.setZero();
                }
                start += orows;
            }
        };

        const int irows = this->input_rows();
        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<typename Func::template Gradient<Scalar>>(irows, 1),
            tycho::utils::TempSpec<typename Func::template Hessian<Scalar>>(irows, irows));
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
