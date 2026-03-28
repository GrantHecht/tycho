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

    static const int IRC = SZ_MAX<LHS::IRC, RHS::IRC>::value;
    static const bool is_conditional = true;
    static const bool meta_conditional = LHS::is_conditional && RHS::is_conditional;

    template <class Scalar> using Input = Eigen::Matrix<Scalar, IRC, 1>;
    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;

    ConditionalStatement() {}
    ConditionalStatement(LHS lhss, ConditionalFlags flagss, RHS rhss)
        : lhs(std::move(lhss)), flag(flagss), rhs(std::move(rhss)) {
        this->input_rows_val = lhs.input_rows();
        if (lhs.input_rows() != rhs.input_rows()) {
            throw std::invalid_argument(
                "LHS and RHS of conditional statement must have same input rows");
        }
        if constexpr (!meta_conditional) {
            if (lhs.output_rows() > 1 || rhs.output_rows() > 1) {
                throw std::invalid_argument(
                    "LHS and RHS of conditional statement must be scalar functions");
            }
            if (flag == ConditionalFlags::ANDFlag || flag == ConditionalFlags::ORFlag) {
                throw std::invalid_argument("AND OR not defined for scalar conditionals");
            }
        } else {
            if (flag != ConditionalFlags::ANDFlag && flag != ConditionalFlags::ORFlag) {
                throw std::invalid_argument("Comparisons not defined for meta conditionals");
            }
        }
    }

    template <class InType> inline bool compute(ConstVectorBaseRef<InType> x) const {
        typedef typename InType::Scalar Scalar;
        if constexpr (meta_conditional) {
            bool left = this->lhs.compute(x);
            bool right = this->rhs.compute(x);
            bool result = false;
            if (this->flag == ConditionalFlags::ANDFlag) {
                result = left && right;
            } else if (this->flag == ConditionalFlags::ORFlag) {
                result = left || right;
            } else {
            }
            return result;
        } else {
            Vector1<Scalar> left;
            Vector1<Scalar> right;
            this->lhs.compute(x, left);
            this->rhs.compute(x, right);
            bool result = false;
            switch (this->flag) {
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

    int input_rows() const { return this->input_rows_val; }

  protected:
    ConditionalFlags flag;
    LHS lhs;
    RHS rhs;
    int input_rows_val = 0;
};

struct ConstantConditional {
    template <class Scalar> using Input = Eigen::Matrix<Scalar, -1, 1>;
    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;

    ConstantConditional() {}
    ConstantConditional(int irows, bool value) : input_rows_val(irows), value(value) {}
    ConstantConditional(bool value) : value(value) {}
    template <class InType> inline bool compute(ConstVectorBaseRef<InType> x) const {
        return this->value;
    }

    int input_rows() const { return this->input_rows_val; }

  protected:
    bool value;
    int input_rows_val = 0;
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

    DENSE_FUNCTION_BASE_TYPES(Base);
    static const bool is_vectorizable = false;

    TestFunc test_func;
    TrueFunc true_func;
    FalseFunc false_func;
    IfElseFunction() {}

    IfElseFunction(TestFunc test, TrueFunc _true, FalseFunc _false)
        : test_func(std::move(test)), true_func(std::move(_true)), false_func(std::move(_false)) {
        this->set_io_rows(this->true_func.input_rows(), this->true_func.output_rows());

        this->set_input_domain(this->input_rows(),
                               {this->true_func.input_domain(), this->false_func.input_domain()});
        if (this->true_func.output_rows() != this->false_func.output_rows()) {
            throw std::invalid_argument("True and false functions in conditional statement must "
                                        "have same number of outputrows.");
        }
        if (this->true_func.input_rows() != this->false_func.input_rows()) {
            throw std::invalid_argument("True and false functions in conditional statement must "
                                        "have same number of inputrows.");
        }
        if (this->test_func.input_rows() != this->false_func.input_rows()) {

            throw std::invalid_argument("Test,True,and False functions in conditional statement "
                                        "must have same number of inputrows.");
        }
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {

        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        if (this->test_func.compute(x)) {
            this->true_func.compute(x, fx);
        } else {
            this->false_func.compute(x, fx);
        }
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        if (this->test_func.compute(x)) {
            this->true_func.compute_jacobian(x, fx_, jx_);
        } else {
            this->false_func.compute_jacobian(x, fx_, jx_);
        }
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        if (this->test_func.compute(x)) {
            this->true_func.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                            adjhess_, adjvars);
        } else {
            this->false_func.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_,
                                                                             adjhess_, adjvars);
        }
    }
};

} // namespace tycho::vf
