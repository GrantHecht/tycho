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
//   - PR 8b: Restructured as a flat N-ary VectorFunction (VF §2.1). The prior
//     implementation composed ComparativeFunction from a recursive chain of
//     IfElseFunction<ConditionalStatement<...>, ...> nodes; each recursion
//     level constructed its tail operand TWICE (once inside the predicate,
//     once as the IfElse false branch), giving O(2^N) stored sub-expressions
//     and evaluations for N operands, and the IfElseFunction JGH hook
//     re-evaluated the selected branch's value a second time on top of the
//     predicate's own evaluation. The new implementation stores each operand
//     exactly once (a std::tuple), evaluates every operand's plain value
//     exactly once to select the winner, and computes derivatives only for
//     the selected operand: N value evals + 1 derivative eval, independent of
//     N. The tie-break rule is unchanged (see class docstring) — this is a
//     performance restructure only, not a semantics change.
// =============================================================================

#pragma once

#include "tycho/detail/vf/type_erasure/conditional.h"

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace tycho::vf {

/// @brief Selects whether a ComparativeFunction computes a minimum or maximum.
/// @ingroup vf
enum class ComparativeFlags {
    MinFlag, ///< @brief Take the scalar minimum across the operand functions.
    MaxFlag, ///< @brief Take the scalar maximum across the operand functions.
};

////////////////////////////////////////////////////////////////////////////////
// Class Definition
/// @brief VectorFunction returning the min or max of two or more scalar functions.
///
/// Each operand is stored exactly once (a `std::tuple<First, Rest...>`).
/// Evaluation proceeds in two phases: first every operand's scalar *value* is
/// computed exactly once (`N` value evals total, regardless of `N`); the
/// winning operand is then selected by a linear scan, and only that operand's
/// derivatives (Jacobian / adjoint gradient / adjoint Hessian) are computed —
/// one additional "derivative eval" that necessarily recomputes the winner's
/// value as part of producing its Jacobian. This replaces a prior
/// IfElse-over-ConditionalStatement composition that stored and evaluated
/// operands ~2^N times (see the file-header PR 8b note).
///
/// **Tie-break rule (preserved bit-for-bit from the prior implementation):**
/// on an exact tie, the LATER operand (higher index in argument order) wins.
/// Concretely: scanning operands from last to first, an earlier operand only
/// displaces the current champion when it is STRICTLY better (`<` for min,
/// `>` for max) than the champion found so far; on an exact tie the
/// incumbent (later-indexed) operand keeps the win. This is exactly the
/// generalization of the prior pairwise rule
/// (`ConditionalFlags::LessThanFlag`/`GreaterThanFlag`, "false branch wins
/// ties") applied right-associatively down the old recursion
/// (`First` vs `ComparativeFunction<Rest...>`, tail wins ties) — verified by
/// hand-tracing the old `BaseCond`/`IfElseFunction` selection for N=2..4
/// before this restructure. The result is only piecewise-differentiable, so
/// which operand's derivatives are returned at a tie is behaviorally
/// significant (a kink); this restructure does not change that choice.
/// @tparam First  First operand function.
/// @tparam Rest   Remaining operand functions (one or more).
/// @ingroup vf
template <class First, class... Rest>
struct ComparativeFunction<First, Rest...>
    : VectorFunction<ComparativeFunction<First, Rest...>,
                     tycho::utils::SZ_MAX<First::IRC, Rest::IRC...>::value, 1> {
    /// @brief Number of operands being compared.
    static constexpr std::size_t NumOperands = 1 + sizeof...(Rest);
    /// @brief Index sequence over all operands (tuple-fold dispatch helper).
    using OperandIndices = std::make_index_sequence<NumOperands>;

    /// @brief CRTP VectorFunction base type.
    using Base = VectorFunction<ComparativeFunction<First, Rest...>,
                                tycho::utils::SZ_MAX<First::IRC, Rest::IRC...>::value, 1>;
    VF_TYPE_ALIASES(Base)
    static constexpr bool is_vectorizable =
        false; ///< @brief Branch selection precludes SIMD lanes (matches IfElseFunction).

    ComparativeFlags type_ = ComparativeFlags::MinFlag; ///< @brief Minimum or maximum selection.
    std::tuple<First, Rest...> ops_; ///< @brief Operands, stored exactly once each.

    // ---------------------------------------------------------------------------
    // Constructors
    /// @brief Construct an empty (default) comparative function.
    ComparativeFunction() {}
    /// @brief Construct a min/max over @p first and @p rest.
    /// @param type   Whether to compute the minimum or maximum.
    /// @param first  First operand function.
    /// @param rest   Remaining operand functions.
    ComparativeFunction(ComparativeFlags type, First first, Rest... rest)
        : type_(type), ops_(std::move(first), std::move(rest)...) {
        this->init_from_operands();
    }

  private:
    // ---------------------------------------------------------------------------
    // Construction-time validation / domain setup
    /// @internal
    /// @brief Validate every operand and set io-rows/input-domain from the tuple.
    /// @endinternal
    void init_from_operands() {
        const int irows = std::get<0>(this->ops_).input_rows();
        std::vector<DomainMatrix> doms;
        doms.reserve(NumOperands);
        this->validate_and_collect(irows, doms, OperandIndices{});
        this->set_io_rows(irows, 1);
        this->set_input_domain(irows, doms);
    }

    template <std::size_t... Is>
    void validate_and_collect(int irows, std::vector<DomainMatrix> &doms,
                              std::index_sequence<Is...>) const {
        (this->validate_and_collect_one(std::get<Is>(this->ops_), irows, doms), ...);
    }

    template <class Op>
    void validate_and_collect_one(const Op &op, int irows, std::vector<DomainMatrix> &doms) const {
        if (op.output_rows() != 1) {
            throw std::invalid_argument(
                "All operands of a ComparativeFunction (min/max) must be scalar functions");
        }
        if (op.input_rows() != irows) {
            throw std::invalid_argument("All operands of a ComparativeFunction (min/max) must "
                                        "have the same number of input rows");
        }
        doms.push_back(op.input_domain());
    }

    // ---------------------------------------------------------------------------
    // Value pass + selection (N value evals, independent of derivative mode)
    /// @internal
    /// @brief Evaluate every operand's scalar value exactly once into @p vals.
    /// @endinternal
    template <class InType, class Scalar, std::size_t... Is>
    inline void eval_all(CVecRef<InType> x, std::array<Scalar, NumOperands> &vals,
                         std::index_sequence<Is...>) const {
        Vector1<Scalar> tmp;
        // Operands may partial-write and rely on a zeroed output (same contract as
        // computable_base.h), so tmp must be re-zeroed before each operand's compute().
        ((tmp[0] = Scalar(0), std::get<Is>(this->ops_).compute(x, tmp), vals[Is] = tmp[0]), ...);
    }

    /// @internal
    /// @brief Right-to-left champion scan (see class docstring for the tie-break rule).
    /// @endinternal
    template <class Scalar>
    inline std::size_t champion_index(const std::array<Scalar, NumOperands> &vals) const {
        std::size_t champ = NumOperands - 1;
        Scalar champ_val = vals[NumOperands - 1];
        for (std::size_t k = NumOperands - 1; k-- > 0;) {
            const bool better = (this->type_ == ComparativeFlags::MinFlag) ? (vals[k] < champ_val)
                                                                           : (vals[k] > champ_val);
            if (better) {
                champ = k;
                champ_val = vals[k];
            }
        }
        return champ;
    }

    // ---------------------------------------------------------------------------
    // Runtime-index -> compile-time tuple-element dispatch (derivative eval,
    // called exactly once, only for the selected operand)
    template <class InType, class OutType, class JacType, std::size_t... Is>
    inline void dispatch_jacobian(std::size_t champ, CVecRef<InType> x, CVecRef<OutType> fx_,
                                  CMatRef<JacType> jx_, std::index_sequence<Is...>) const {
        ((Is == champ ? (void)(std::get<Is>(this->ops_).compute_jacobian(x, fx_, jx_)) : (void)0),
         ...);
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType, std::size_t... Is>
    inline void dispatch_jgh(std::size_t champ, CVecRef<InType> x, CVecRef<OutType> fx_,
                             CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
                             CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars,
                             std::index_sequence<Is...>) const {
        ((Is == champ ? (void)(std::get<Is>(this->ops_)
                                   .compute_jacobian_adjointgradient_adjointhessian(
                                       x, fx_, jx_, adjgrad_, adjhess_, adjvars))
                      : (void)0),
         ...);
    }

  public:
    /// @internal
    /// @brief Evaluate the selected (min/max) operand's value into @p fx_ (CRTP compute hook).
    /// @tparam InType   Input vector expression type.
    /// @tparam OutType  Output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to fill.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        std::array<Scalar, NumOperands> vals;
        this->eval_all(x, vals, OperandIndices{});
        // The champion's value was already computed above -- no re-evaluation.
        fx[0] = vals[this->champion_index(vals)];
    }

    /// @internal
    /// @brief Evaluate value and Jacobian of the selected operand (CRTP hook).
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
        typedef typename InType::Scalar Scalar;
        std::array<Scalar, NumOperands> vals;
        this->eval_all(x, vals, OperandIndices{});
        const std::size_t champ = this->champion_index(vals);
        // Exactly one derivative eval, on the selected operand only.
        this->dispatch_jacobian(champ, x, fx_, jx_, OperandIndices{});
    }

    /// @internal
    /// @brief Evaluate value, Jacobian, adjoint gradient and Hessian of the selected operand.
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
        typedef typename InType::Scalar Scalar;
        std::array<Scalar, NumOperands> vals;
        this->eval_all(x, vals, OperandIndices{});
        const std::size_t champ = this->champion_index(vals);
        // Exactly one derivative eval, on the selected operand only.
        this->dispatch_jgh(champ, x, fx_, jx_, adjgrad_, adjhess_, adjvars, OperandIndices{});
    }
};

} // namespace tycho::vf
