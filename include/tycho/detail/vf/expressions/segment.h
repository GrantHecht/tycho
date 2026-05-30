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

#include "tycho/detail/vf/common/constant.h"
#include "tycho/detail/vf/core/vector_function.h"
#include "tycho/detail/vf/derivatives/detect_diagonal.h"
#include "tycho/detail/vf/type_erasure/conditional.h"

namespace tycho::vf {

/// @internal
/// @brief Forward declaration of the segment/indexing-view implementation.
/// @tparam Derived  CRTP most-derived type.
/// @tparam IR       Input-row count of the parent vector.
/// @tparam OR       Output-row count (length of the selected sub-range).
/// @tparam ST       Compile-time start index, or -1 for a runtime start.
/// @endinternal
template <class Derived, int IR, int OR, int ST> struct Segment_Impl;

/// @brief Identity VectorFunction returning all of its input: `f(x) = x`.
///
/// `Arguments` is the entry point of the DSL: it represents the full input
/// vector and is sliced via @ref Segment / `.head()` / `.tail()` / `.segment()`
/// to build expressions over named subsets of the decision variables.
///
/// @tparam IR_OR  Input/output row count (input and output are identical).
/// @ingroup vf
template <int IR_OR> struct Arguments : Segment_Impl<Arguments<IR_OR>, IR_OR, IR_OR, 0> {
    /// @brief Segment implementation base (identity range over all rows).
    using Base = Segment_Impl<Arguments<IR_OR>, IR_OR, IR_OR, 0>;
    using Base::Base;
    /// @brief Construct an identity over a runtime-sized input.
    /// @param iror  Number of input/output rows.
    Arguments(int iror) : Base(iror, iror, 0) {}
};
/// @brief Contiguous sub-range view of the input: `f(x) = x[ST : ST+OR]`.
///
/// The return type of `.head()`, `.tail()`, `.segment()`, and element indexing
/// on a VectorFunction. Selects @p OR contiguous rows of the input starting at
/// @p ST; its Jacobian is the corresponding identity sub-block.
///
/// @tparam IR  Input-row count of the parent vector.
/// @tparam OR  Output-row count (length of the selected range).
/// @tparam ST  Compile-time start index, or -1 for a runtime start.
/// @ingroup vf
template <int IR, int OR, int ST> struct Segment : Segment_Impl<Segment<IR, OR, ST>, IR, OR, ST> {
    /// @brief Segment implementation base for this view.
    using Base = Segment_Impl<Segment<IR, OR, ST>, IR, OR, ST>;
    using Base::Base;
};

/// @internal
/// @brief Trait: detects whether `T` is a @ref Segment specialization.
/// @tparam T  Candidate type.
/// @endinternal
template <class T> struct Is_Segment : std::false_type {};
/// @internal
/// @brief Specialization recognizing @ref Segment types as true.
/// @tparam IR  Input-row count.
/// @tparam OR  Output-row count.
/// @tparam ST  Start index.
/// @endinternal
template <int IR, int OR, int ST> struct Is_Segment<Segment<IR, OR, ST>> : std::true_type {};

/// @internal
/// @brief Trait: detects whether `T` is an @ref Arguments specialization.
/// @tparam T  Candidate type.
/// @endinternal
template <class T> struct Is_Arguments : std::false_type {};
/// @internal
/// @brief Specialization recognizing @ref Arguments types as true.
/// @tparam IR  Input/output row count.
/// @endinternal
template <int IR> struct Is_Arguments<Arguments<IR>> : std::true_type {};

/// @internal
/// @brief Trait: detects whether `T` is a (statically or dynamically) scaled segment.
/// @tparam T  Candidate type.
/// @endinternal
template <class T> struct Is_ScaledSegment : std::false_type {};
/// @internal
/// @brief Specialization for a runtime-`Scaled` @ref Segment.
/// @tparam IR  Input-row count.
/// @tparam OR  Output-row count.
/// @tparam ST  Start index.
/// @endinternal
template <int IR, int OR, int ST>
struct Is_ScaledSegment<Scaled<Segment<IR, OR, ST>>> : std::true_type {};
/// @internal
/// @brief Specialization for a compile-time `StaticScaled` @ref Segment.
/// @tparam IR     Input-row count.
/// @tparam OR     Output-row count.
/// @tparam ST     Start index.
/// @tparam VALUE  Compile-time scale-factor wrapper type.
/// @endinternal
template <int IR, int OR, int ST, class VALUE>
struct Is_ScaledSegment<StaticScaled<Segment<IR, OR, ST>, VALUE>> : std::true_type {};

/// @internal
/// @brief Holder for a segment start index, compile-time variant.
///
/// When `ST != -1` the start index is a `static constexpr` and
/// `set_seg_start` is a no-op; the -1 specialization stores it at runtime.
///
/// @tparam ST  Compile-time start index.
/// @endinternal
template <int ST> struct SegStartHolder {
    static constexpr int seg_start_ = ST; ///< @brief Compile-time segment start index.
    /// @brief No-op: the start index is fixed at compile time.
    /// @param st  Ignored.
    void set_seg_start(int st) {};
};
/// @internal
/// @brief Runtime-start specialization of @ref SegStartHolder (`ST == -1`).
/// @endinternal
template <> struct SegStartHolder<-1> {
    int seg_start_ = -1; ///< @brief Runtime segment start index.
    /// @brief Set the runtime segment start index.
    /// @param st  Start row of the segment within the parent input.
    void set_seg_start(int st) { this->seg_start_ = st; };
};

/// @internal
/// @brief CRTP implementation of @ref Segment and @ref Arguments index views.
///
/// Provides the value (`fx = x[start : start+orows]`), the identity-sub-block
/// Jacobian, and the diagonal-structured derivative-accumulation overloads
/// that make slicing cheap. Inherits the start index from @ref SegStartHolder.
///
/// @tparam Derived  CRTP most-derived type (@ref Segment or @ref Arguments).
/// @tparam IR       Input-row count of the parent vector.
/// @tparam OR       Output-row count (length of the selected range).
/// @tparam ST       Compile-time start index, or -1 for a runtime start.
/// @endinternal
template <class Derived, int IR, int OR, int ST>
struct Segment_Impl : VectorFunction<Derived, IR, OR>, SegStartHolder<ST> {
    /// @internal
    /// @brief Input-domain descriptor: a single contiguous sub-range.
    /// @endinternal
    using INPUT_DOMAIN = SingleDomain<IR, ST, OR>;
    /// @internal
    /// @brief VectorFunction CRTP base for this index view.
    /// @endinternal
    using Base = VectorFunction<Derived, IR, OR>;
    // using SegStartHolder<ST>::seg_start_;
    VF_TYPE_ALIASES(Base);

    /// @internal
    /// @brief Default-construct an empty segment view.
    /// @endinternal
    Segment_Impl() {}
    /// @internal
    /// @brief Construct a segment view, validating bounds.
    /// @param irows  Parent input-row count.
    /// @param orows  Number of rows selected.
    /// @param start  Start row of the selected range.
    /// @endinternal
    Segment_Impl(int irows, int orows, int start) {
        this->set_io_rows(irows, orows);
        this->set_seg_start(start);
        DomainMatrix dmn(2, 1);
        dmn(0, 0) = start;
        dmn(1, 0) = orows;

        this->set_input_domain(irows, {dmn});

        if (start + orows > this->input_rows() || start < 0) {
            throw std::invalid_argument("Segment/Element Index Out of Bounds");
        }
    }

    static constexpr bool is_linear_function = true; ///< @brief A segment view is always linear.
    static constexpr bool is_vectorizable =
        true; ///< @brief Segment views support SuperScalar packs.

    /// @internal
    /// @brief Copy the selected sub-range: `fx = x[seg_start_ : seg_start_+OR]`.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Parent input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = x.template segment<OR>(this->seg_start_, this->output_rows());
    }
    /// @internal
    /// @brief Copy the sub-range and write its identity-block Jacobian.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Parent input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer; the selected diagonal is set to 1.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        const int OROWS = this->output_rows();
        Scalar ONE = Scalar(1.0);
        fx = x.template segment<OR>(this->seg_start_, OROWS);
        jx.template middleCols<OR>(this->seg_start_, OROWS).diagonal().setConstant(ONE);
    }

    /// @internal
    /// @brief Value, identity Jacobian, and adjoint gradient for a segment.
    ///
    /// The adjoint Hessian is zero (a segment is linear), so only the selected
    /// rows of @p adjgrad_ receive the incoming adjoint variables.
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Parent input vector.
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer; selected diagonal set to 1.
    /// @param  adjgrad_ Adjoint-gradient accumulator.
    /// @param  adjhess_ Adjoint-Hessian accumulator (unused; segment is linear).
    /// @param  adjvars  Adjoint variables scattered into the selected rows.
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
        //  MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        const int OROWS = this->output_rows();
        Scalar ONE = Scalar(1.0);

        fx = x.template segment<OR>(this->seg_start_, OROWS);
        jx.template middleCols<OR>(this->seg_start_, OROWS).diagonal().setConstant(ONE);
        adjgrad.template segment<OR>(this->seg_start_, OROWS) = adjvars;
    }

    /// @internal
    /// @brief Chain-rule product specialized for a diagonal segment Jacobian.
    ///
    /// Because a segment's Jacobian is a (sub-)identity, the product reduces to
    /// scaling @p left by the diagonal of the selected columns of @p right and
    /// writing it back into the matching columns of @p target_, dispatched on
    /// the @p Assignment policy.
    ///
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam Left        Concrete Eigen left-operand block type.
    /// @tparam Right       Concrete Eigen right-operand block type.
    /// @tparam Assignment  Assignment policy (direct, plus/minus, scaled).
    /// @tparam Aliased     Whether target and operands may alias.
    /// @param  target_  Destination block.
    /// @param  left     Upstream Jacobian factor.
    /// @param  right    Block whose selected diagonal is the segment Jacobian.
    /// @param  assign   Assignment-policy tag instance.
    /// @param  aliased  Aliasing-flag tag instance.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        MatRef<Target> target = target_.const_cast_derived();
        typedef typename Target::Scalar Scalar;

        auto Impl = [&](auto &diag) {
            diag = right.derived()
                       .template middleCols<OR>(this->seg_start_, this->output_rows())
                       .diagonal();

            if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
                if constexpr (Is_EigenDiagonalMatrix<typename std::remove_const_reference<
                                  decltype(left.derived())>::type>::value) {
                    target.template middleCols<OR>(this->seg_start_, this->output_rows())
                        .diagonal() = left.derived().diagonal().cwiseProduct(diag);
                } else {
                    if constexpr (Aliased)
                        target.template middleCols<OR>(this->seg_start_, this->output_rows()) =
                            left.derived() * diag.asDiagonal();
                    else
                        target.template middleCols<OR>(this->seg_start_, this->output_rows())
                            .noalias() = left.derived() * diag.asDiagonal();
                }

            } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
                if constexpr (Is_EigenDiagonalMatrix<typename std::remove_const_reference<
                                  decltype(left.derived())>::type>::value) {
                    target.template middleCols<OR>(this->seg_start_, this->output_rows())
                        .diagonal() += left.derived().diagonal().cwiseProduct(diag);
                } else {
                    if constexpr (Aliased)
                        target.template middleCols<OR>(this->seg_start_, this->output_rows()) +=
                            left.derived() * diag.asDiagonal();
                    else
                        target.template middleCols<OR>(this->seg_start_, this->output_rows())
                            .noalias() += left.derived() * diag.asDiagonal();
                }

            } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {

                if constexpr (Is_EigenDiagonalMatrix<typename std::remove_const_reference<
                                  decltype(left.derived())>::type>::value) {
                    target.template middleCols<OR>(this->seg_start_, this->output_rows())
                        .diagonal() -= left.derived().diagonal().cwiseProduct(diag);
                } else {
                    if constexpr (Aliased)
                        target.template middleCols<OR>(this->seg_start_, this->output_rows()) -=
                            left.derived() * diag.asDiagonal();
                    else
                        target.template middleCols<OR>(this->seg_start_, this->output_rows())
                            .noalias() -= left.derived() * diag.asDiagonal();
                }

            } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
                target.template middleCols<OR>(this->seg_start_, this->output_rows()).noalias() =
                    assign.value * left.derived() * diag.asDiagonal();
            } else if constexpr (std::is_same<Assignment,
                                              ScaledPlusEqualsAssignment<Scalar>>::value) {
                target.template middleCols<OR>(this->seg_start_, this->output_rows()).noalias() +=
                    assign.value * left.derived() * diag.asDiagonal();
            } else {
                std::cout << "right_jacobian_product has not been implemented for: " << this->name
                          << std::endl
                          << std::endl;
            }
        };

        const int orows = this->output_rows();
        tycho::utils::BumpAllocator::allocate_run(Impl,
                                                  tycho::utils::TempSpec<Output<Scalar>>(orows, 1));
    }

    /// @internal
    /// @brief Symmetric (Hessian-style) product specialized for a segment.
    ///
    /// Writes `left * D^2` (where `D` is the segment's diagonal Jacobian) into
    /// the selected square block of @p target_, dispatched on @p Assignment.
    ///
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam Left        Concrete Eigen left-operand block type.
    /// @tparam Right       Concrete Eigen right-operand block type.
    /// @tparam Assignment  Assignment policy (direct, plus/minus, scaled).
    /// @tparam Aliased     Whether target and operands may alias.
    /// @param  target_  Destination square block.
    /// @param  left     Left operand of the symmetric product.
    /// @param  right    Block whose selected diagonal is the segment Jacobian.
    /// @param  assign   Assignment-policy tag instance.
    /// @param  aliased  Aliasing-flag tag instance.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                          CEigRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        MatRef<Target> target = target_.const_cast_derived();
        typedef typename Target::Scalar Scalar;

        Eigen::DiagonalMatrix<Scalar, OR> diag;
        diag.diagonal() = right.derived()
                              .template middleCols<OR>(this->seg_start_, this->output_rows())
                              .diagonal();
        diag.diagonal() = diag.diagonal().cwiseProduct(diag.diagonal());

        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
            target
                .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                        this->output_rows())
                .noalias() = left.derived() * diag;
        } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
            if constexpr (Is_EigenDiagonalMatrix<typename std::remove_const_reference<
                              decltype(left.derived())>::type>::value) {
                target
                    .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                            this->output_rows())
                    .diagonal() += left.derived().diagonal().cwiseProduct(diag.diagonal());
            } else {
                target
                    .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                            this->output_rows())
                    .noalias() += left.derived() * diag;
            }

        } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
            target
                .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                        this->output_rows())
                .noalias() -= left.derived() * diag;
        } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
            target
                .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                        this->output_rows())
                .noalias() = assign.value * left.derived() * diag;
        } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
            target
                .template block<OR, OR>(this->seg_start_, this->seg_start_, this->output_rows(),
                                        this->output_rows())
                .noalias() += assign.value * left.derived() * diag;
        } else {
            std::cout << "symetric_jacobian_product has not been implemented for: " << this->name
                      << std::endl
                      << std::endl;
        }
    }

    /// @internal
    /// @brief Accumulate the segment's diagonal Jacobian into a target block.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-Jacobian type.
    /// @tparam Assignment  Assignment policy (direct or plus/minus-equals).
    /// @param  target_  Destination block.
    /// @param  right    Source whose selected diagonal is accumulated.
    /// @param  assign   Assignment-policy tag instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CEigRef<JacType> right,
                                    Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
            target.template middleCols<OR>(this->seg_start_, this->output_rows()).diagonal() =
                right.derived()
                    .template middleCols<OR>(this->seg_start_, this->output_rows())
                    .diagonal();
        } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
            target.template middleCols<OR>(this->seg_start_, this->output_rows()).diagonal() +=
                right.derived()
                    .template middleCols<OR>(this->seg_start_, this->output_rows())
                    .diagonal();
        } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
            target.template middleCols<OR>(this->seg_start_, this->output_rows()).diagonal() -=
                right.derived()
                    .template middleCols<OR>(this->seg_start_, this->output_rows())
                    .diagonal();
        } else {
        }
    }
    /// @internal
    /// @brief Accumulate the selected rows of a gradient into a target vector.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-gradient type.
    /// @tparam Assignment  Assignment policy (direct or plus/minus-equals).
    /// @param  target_  Destination vector.
    /// @param  right    Source whose selected rows are accumulated.
    /// @param  assign   Assignment-policy tag instance.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_, CEigRef<JacType> right,
                                    Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
            target.template segment<OR>(this->seg_start_, this->output_rows()) =
                right.derived().template segment<OR>(this->seg_start_, this->output_rows());
        } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
            target.template segment<OR>(this->seg_start_, this->output_rows()) +=
                right.derived().template segment<OR>(this->seg_start_, this->output_rows());
        } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
            target.template segment<OR>(this->seg_start_, this->output_rows()) -=
                right.derived().template segment<OR>(this->seg_start_, this->output_rows());
        } else {
        }
    }
    /// @internal
    /// @brief Scale the segment's diagonal Jacobian entries in place.
    /// @tparam Target  Concrete Eigen target/output block type.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param  target_ Jacobian block to scale.
    /// @param  s       Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        MatRef<Target> target = target_.const_cast_derived();
        target.template middleCols<OR>(this->seg_start_, this->output_rows()).diagonal() *= s;
    }
    /// @internal
    /// @brief Scale the segment's selected gradient rows in place.
    /// @tparam Target  Concrete Eigen target/output block type.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param  target_ Gradient block to scale.
    /// @param  s       Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        MatRef<Target> target = target_.const_cast_derived();
        target.template segment<OR>(this->seg_start_, this->output_rows()) *= s;
    }
    /// @internal
    /// @brief Re-argue a function through this segment, producing a nested expression.
    ///
    /// Composes @p f so its inputs are taken from this segment's selected rows.
    ///
    /// @tparam Func     Inner function type.
    /// @tparam FuncIRC  Inner function input-row count.
    /// @param  f        The function to re-argue onto this segment.
    /// @return The nested VectorFunction expression.
    /// @endinternal
    template <class Func, int FuncIRC>
    decltype(auto) rearged(const DenseFunctionBase<Func, FuncIRC, IR> &f) const {
        return Base::template EVALOP<Func>::make_nested(this->derived(), f.derived());
    }
};

} // namespace tycho::vf
