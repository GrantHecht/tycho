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

/// @brief Comparison and logical operators usable in a ConditionalStatement.
/// @ingroup vf
enum class ConditionalFlags {
    LessThanFlag,           ///< @brief Less-than comparison (`<`).
    GreaterThanFlag,        ///< @brief Greater-than comparison (`>`).
    LessThanEqualToFlag,    ///< @brief Less-than-or-equal comparison (`<=`).
    GreaterThanEqualToFlag, ///< @brief Greater-than-or-equal comparison (`>=`).
    EqualToFlag,            ///< @brief Equality comparison (`==`).
    ANDFlag,                ///< @brief Logical AND of two sub-conditions.
    ORFlag,                 ///< @brief Logical OR of two sub-conditions.
};

/// @brief Boolean predicate comparing two scalar functions or combining two conditions.
///
/// Evaluates a comparison (`LHS <op> RHS`) when @p LHS and @p RHS are scalar
/// VectorFunctions, or a logical combination (AND/OR) when both sides are
/// themselves conditionals (a "meta conditional"). This is the building block
/// from which GenericConditional predicates and IfElseFunction tests are formed.
/// @tparam LHS  Left-hand operand: a scalar function, or a conditional (meta case).
/// @tparam RHS  Right-hand operand: a scalar function, or a conditional (meta case).
/// @ingroup vf
template <class LHS, class RHS> struct ConditionalStatement {

    /// @brief Compile-time input-row count (max of the two operands').
    static constexpr int IRC = SZ_MAX<LHS::IRC, RHS::IRC>::value;
    /// @brief Trait flag: this is a conditional.
    static constexpr bool is_conditional = true;
    /// @brief True when both operands are themselves conditionals (logical combination).
    static constexpr bool meta_conditional = LHS::is_conditional && RHS::is_conditional;

    /// @brief Fixed-size input vector type for a given scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Input = Eigen::Matrix<Scalar, IRC, 1>;

    /// @brief Construct an empty (default) conditional statement.
    ConditionalStatement() {}
    /// @brief Construct from a left operand, an operator, and a right operand.
    /// @param lhss   Left-hand operand.
    /// @param flagss Comparison or logical operator to apply.
    /// @param rhss   Right-hand operand.
    ConditionalStatement(LHS lhss, ConditionalFlags flagss, RHS rhss)
        : lhs_(std::move(lhss)), flag_(flagss), rhs_(std::move(rhss)) {
        this->input_rows_val_ = lhs_.input_rows();
        if (lhs_.input_rows() != rhs_.input_rows()) {
            throw std::invalid_argument(
                "LHS and RHS of conditional statement must have same input rows");
        }
        if constexpr (!meta_conditional) {
            if (lhs_.output_rows() > 1 || rhs_.output_rows() > 1) {
                throw std::invalid_argument(
                    "LHS and RHS of conditional statement must be scalar functions");
            }
            if (flag_ == ConditionalFlags::ANDFlag || flag_ == ConditionalFlags::ORFlag) {
                throw std::invalid_argument("AND OR not defined for scalar conditionals");
            }
        } else {
            if (flag_ != ConditionalFlags::ANDFlag && flag_ != ConditionalFlags::ORFlag) {
                throw std::invalid_argument("Comparisons not defined for meta conditionals");
            }
        }
    }

    /// @brief Evaluate the predicate at @p x.
    /// @tparam InType  Eigen expression type of the input vector.
    /// @param x  Input vector.
    /// @return Boolean result of the comparison or logical combination.
    template <class InType> inline bool compute(CVecRef<InType> x) const {
        typedef typename InType::Scalar Scalar;
        if constexpr (meta_conditional) {
            bool left = this->lhs_.compute(x);
            bool right = this->rhs_.compute(x);
            bool result = false;
            if (this->flag_ == ConditionalFlags::ANDFlag) {
                result = left && right;
            } else if (this->flag_ == ConditionalFlags::ORFlag) {
                result = left || right;
            } else {
            }
            return result;
        } else {
            Vector1<Scalar> left;
            Vector1<Scalar> right;
            this->lhs_.compute(x, left);
            this->rhs_.compute(x, right);
            bool result = false;
            switch (this->flag_) {
            case ConditionalFlags::LessThanFlag: {
                result = left[0] < right[0];
                break;
            }
            case ConditionalFlags::GreaterThanFlag: {
                result = left[0] > right[0];
                break;
            }
            case ConditionalFlags::LessThanEqualToFlag: {
                result = left[0] <= right[0];
                break;
            }
            case ConditionalFlags::GreaterThanEqualToFlag: {
                result = left[0] >= right[0];
                break;
            }
            case ConditionalFlags::EqualToFlag: {
                result = left[0] == right[0];
                break;
            }
            default: {
            }
            }

            return result;
        }
    }

    /// @brief Number of input rows the statement accepts.
    /// @return Input-row count.
    int input_rows() const { return this->input_rows_val_; }

  protected:
    ConditionalFlags flag_;  ///< @brief Comparison/logical operator applied.
    LHS lhs_;                ///< @brief Left-hand operand.
    RHS rhs_;                ///< @brief Right-hand operand.
    int input_rows_val_ = 0; ///< @brief Cached input-row count.
};

/// @brief Conditional predicate that always returns a fixed boolean value.
///
/// Used as a trivial/leaf predicate (e.g. an always-true or always-false branch
/// selector) where a conditional is required but no comparison is needed.
/// @ingroup vf
struct ConstantConditional {
    /// @brief Dynamic-size input vector type for a given scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Input = Eigen::Matrix<Scalar, -1, 1>;

    /// @brief Construct a default constant conditional.
    ConstantConditional() {}
    /// @brief Construct with an explicit input-row count and value.
    /// @param irows  Number of input rows to report.
    /// @param val    Constant boolean result.
    ConstantConditional(int irows, bool val) : input_rows_val_(irows), value_(val) {}
    /// @brief Construct from a value (input rows default to 0).
    /// @param val  Constant boolean result.
    ConstantConditional(bool val) : value_(val) {}
    /// @brief Return the fixed value, ignoring the input.
    /// @tparam InType  Eigen expression type of the input vector.
    /// @param x  Input vector (unused).
    /// @return The stored constant boolean.
    template <class InType> inline bool compute(CVecRef<InType> x) const { return this->value_; }

    /// @brief Number of input rows the predicate reports.
    /// @return Input-row count.
    int input_rows() const { return this->input_rows_val_; }

  protected:
    bool value_;             ///< @brief Constant boolean result.
    int input_rows_val_ = 0; ///< @brief Reported input-row count.
};

/// @brief VectorFunction that selects between two branches based on a predicate.
///
/// At evaluation time @p TestFunc is checked: if true, @p TrueFunc is evaluated;
/// otherwise @p FalseFunc. Both branches must share the same input and output
/// dimensions. Derivatives are taken from whichever branch is selected, so the
/// function is only piecewise-differentiable across the switching boundary.
/// @tparam TestFunc   Predicate (conditional) selecting the branch.
/// @tparam TrueFunc   Function evaluated when the predicate is true.
/// @tparam FalseFunc  Function evaluated when the predicate is false.
/// @ingroup vf
template <class TestFunc, class TrueFunc, class FalseFunc>
struct IfElseFunction : VectorFunction<IfElseFunction<TestFunc, TrueFunc, FalseFunc>,
                                       SZ_MAX<TrueFunc::IRC, FalseFunc::IRC>::value,
                                       SZ_MAX<TrueFunc::ORC, FalseFunc::ORC>::value> {
    /// @brief CRTP VectorFunction base type.
    using Base = VectorFunction<IfElseFunction<TestFunc, TrueFunc, FalseFunc>,
                                SZ_MAX<TrueFunc::IRC, FalseFunc::IRC>::value,
                                SZ_MAX<TrueFunc::ORC, FalseFunc::ORC>::value>;
    /// @brief Composite input domain (union of the two branches' domains).
    using INPUT_DOMAIN = CompositeDomain<Base::IRC, typename TrueFunc::INPUT_DOMAIN,
                                         typename FalseFunc::INPUT_DOMAIN>;

    VF_TYPE_ALIASES(Base);
    static constexpr bool is_vectorizable =
        false; ///< @brief Branch selection precludes SIMD lanes.

    TestFunc test_func_;   ///< @brief Predicate selecting the active branch.
    TrueFunc true_func_;   ///< @brief Branch evaluated when the predicate is true.
    FalseFunc false_func_; ///< @brief Branch evaluated when the predicate is false.
    /// @brief Construct an empty (default) if/else function.
    IfElseFunction() {}

    /// @brief Construct from a test predicate and the two branch functions.
    /// @param test   Predicate selecting the branch.
    /// @param _true  Function for the true branch.
    /// @param _false Function for the false branch.
    IfElseFunction(TestFunc test, TrueFunc _true, FalseFunc _false)
        : test_func_(std::move(test)), true_func_(std::move(_true)),
          false_func_(std::move(_false)) {
        this->set_io_rows(this->true_func_.input_rows(), this->true_func_.output_rows());

        this->set_input_domain(this->input_rows(),
                               {this->true_func_.input_domain(), this->false_func_.input_domain()});
        if (this->true_func_.output_rows() != this->false_func_.output_rows()) {
            throw std::invalid_argument("True and false functions in conditional statement must "
                                        "have same number of outputrows.");
        }
        if (this->true_func_.input_rows() != this->false_func_.input_rows()) {
            throw std::invalid_argument("True and false functions in conditional statement must "
                                        "have same number of inputrows.");
        }
        if (this->test_func_.input_rows() != this->false_func_.input_rows()) {

            throw std::invalid_argument("Test,True,and False functions in conditional statement "
                                        "must have same number of inputrows.");
        }
    }

    /// @internal
    /// @brief Evaluate the selected branch into @p fx_ (CRTP compute hook).
    /// @tparam InType   Input vector expression type.
    /// @tparam OutType  Output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to fill.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {

        VecRef<OutType> fx = fx_.const_cast_derived();

        if (this->test_func_.compute(x)) {
            this->true_func_.compute(x, fx);
        } else {
            this->false_func_.compute(x, fx);
        }
    }
    /// @internal
    /// @brief Evaluate value and Jacobian of the selected branch (CRTP hook).
    /// @tparam InType   Input vector expression type.
    /// @tparam OutType  Output vector expression type.
    /// @tparam JacType  Jacobian matrix expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to fill.
    /// @param jx_  Jacobian matrix to fill.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        if (this->test_func_.compute(x)) {
            this->true_func_.compute_jacobian(x, fx_, jx_);
        } else {
            this->false_func_.compute_jacobian(x, fx_, jx_);
        }
    }
    /// @internal
    /// @brief Evaluate value, Jacobian, adjoint gradient and Hessian of the selected branch.
    /// @tparam InType       Input vector expression type.
    /// @tparam OutType      Output vector expression type.
    /// @tparam JacType      Jacobian matrix expression type.
    /// @tparam AdjGradType  Adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Adjoint (Lagrange-multiplier) vector expression type.
    /// @param x        Input vector.
    /// @param fx_      Output vector to fill.
    /// @param jx_      Jacobian matrix to fill.
    /// @param adjgrad_ Adjoint gradient to fill.
    /// @param adjhess_ Adjoint Hessian to fill.
    /// @param adjvars  Adjoint variables (input).
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        if (this->test_func_.compute(x)) {
            this->true_func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                             adjhess_, adjvars);
        } else {
            this->false_func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                              adjhess_, adjvars);
        }
    }
};

} // namespace tycho::vf
