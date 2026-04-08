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

enum class ConditionalFlags {
    LessThanFlag,
    GreaterThanFlag,
    LessThanEqualToFlag,
    GreaterThanEqualToFlag,
    EqualToFlag,
    ANDFlag,
    ORFlag,
};

template <class LHS, class RHS> struct ConditionalStatement {

    static constexpr int IRC = SZ_MAX<LHS::IRC, RHS::IRC>::value;
    static constexpr bool is_conditional = true;
    static constexpr bool meta_conditional = LHS::is_conditional && RHS::is_conditional;

    template <class Scalar> using Input = Eigen::Matrix<Scalar, IRC, 1>;
    template <class Scalar> using CVecRef = const Eigen::MatrixBase<Scalar> &;

    ConditionalStatement() {}
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

    int input_rows() const { return this->input_rows_val_; }

  protected:
    ConditionalFlags flag_;
    LHS lhs_;
    RHS rhs_;
    int input_rows_val_ = 0;
};

struct ConstantConditional {
    template <class Scalar> using Input = Eigen::Matrix<Scalar, -1, 1>;
    template <class Scalar> using CVecRef = const Eigen::MatrixBase<Scalar> &;

    ConstantConditional() {}
    ConstantConditional(int irows, bool val) : input_rows_val_(irows), value_(val) {}
    ConstantConditional(bool val) : value_(val) {}
    template <class InType> inline bool compute(CVecRef<InType> x) const {
        return this->value_;
    }

    int input_rows() const { return this->input_rows_val_; }

  protected:
    bool value_;
    int input_rows_val_ = 0;
};

template <class TestFunc, class TrueFunc, class FalseFunc>
struct IfElseFunction : VectorFunction<IfElseFunction<TestFunc, TrueFunc, FalseFunc>,
                                       SZ_MAX<TrueFunc::IRC, FalseFunc::IRC>::value,
                                       SZ_MAX<TrueFunc::ORC, FalseFunc::ORC>::value> {
    using Base = VectorFunction<IfElseFunction<TestFunc, TrueFunc, FalseFunc>,
                                SZ_MAX<TrueFunc::IRC, FalseFunc::IRC>::value,
                                SZ_MAX<TrueFunc::ORC, FalseFunc::ORC>::value>;
    using INPUT_DOMAIN = CompositeDomain<Base::IRC, typename TrueFunc::INPUT_DOMAIN,
                                         typename FalseFunc::INPUT_DOMAIN>;

    VF_TYPE_ALIASES(Base);
    static constexpr bool is_vectorizable = false;

    TestFunc test_func_;
    TrueFunc true_func_;
    FalseFunc false_func_;
    IfElseFunction() {}

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

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {

        VecRef<OutType> fx = fx_.const_cast_derived();

        if (this->test_func_.compute(x)) {
            this->true_func_.compute(x, fx);
        } else {
            this->false_func_.compute(x, fx);
        }
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        if (this->test_func_.compute(x)) {
            this->true_func_.compute_jacobian(x, fx_, jx_);
        } else {
            this->false_func_.compute_jacobian(x, fx_, jx_);
        }
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
        CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars) const {
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
