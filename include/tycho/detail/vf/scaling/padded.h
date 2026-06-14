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

/// @file
/// @brief VectorFunction wrapper that zero-pads a function's output vector.
#pragma once

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @internal
/// @brief Holds the leading (upper) pad count for @ref PaddedOutput.
///
/// The general template stores a compile-time pad count as a constant; the
/// `-1` specialization stores it as a runtime member instead.
/// @tparam St  Compile-time leading pad count, or `-1` for a runtime count.
/// @endinternal
template <int St> struct UpperPadHolder {
    /// @internal
    /// @brief Compile-time leading pad count.
    /// @endinternal
    static constexpr int u_pad_ = St;
    /// @internal
    /// @brief Constructs from a runtime pad value (ignored in this template).
    /// @endinternal
    UpperPadHolder(int st) {};
    /// @internal
    /// @brief Default-constructs the holder.
    /// @endinternal
    UpperPadHolder() {};
};

/// @internal
/// @brief Runtime specialization of @ref UpperPadHolder storing a mutable count.
/// @endinternal
template <> struct UpperPadHolder<-1> {
    /// @internal
    /// @brief Runtime leading pad count.
    /// @endinternal
    int u_pad_ = 0;
    /// @internal
    /// @brief Constructs the holder, recording the runtime pad value.
    /// @endinternal
    UpperPadHolder(int st) { u_pad_ = st; };
    /// @internal
    /// @brief Default-constructs the holder with a zero pad count.
    /// @endinternal
    UpperPadHolder() {};
};

/// @ingroup vf
/// @brief Pads the output of a wrapped function with leading and/or trailing zeros.
///
/// The wrapped function's output is written into a contiguous middle segment of
/// a larger output vector; the leading and trailing rows are left zero. The
/// Jacobian and adjoint quantities are placed in the matching middle rows so the
/// padded rows contribute no derivatives.
/// @tparam Func  Wrapped VectorFunction type.
/// @tparam UP    Number of leading (upper) zero rows, or `-1` for a runtime count.
/// @tparam LP    Number of trailing (lower) zero rows.
template <class Func, int UP, int LP>
struct PaddedOutput
    : VectorFunction<PaddedOutput<Func, UP, LP>, Func::IRC, SZ_SUM<LP, UP, Func::ORC>::value>,
      UpperPadHolder<UP> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base =
        VectorFunction<PaddedOutput<Func, UP, LP>, Func::IRC, SZ_SUM<LP, UP, Func::ORC>::value>;
    VF_TYPE_ALIASES(Base)

    /// @brief The wrapped function whose output is padded.
    Func func_;
    /// @brief Vectorizability inherited from the wrapped function.
    static constexpr bool is_vectorizable = Func::is_vectorizable;

    /// @brief Input-domain sparsity inherited from the wrapped function.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    /// @brief Linearity inherited from the wrapped function (padding is linear).
    static constexpr bool is_linear_function = Func::is_linear_function;

    /// @brief Constructs a padded-output wrapper around a function.
    /// @param f     Function to wrap.
    /// @param upad  Number of leading zero rows to prepend.
    /// @param lpad  Number of trailing zero rows to append.
    PaddedOutput(Func f, int upad, int lpad) : UpperPadHolder<UP>(upad), func_(std::move(f)) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows() + upad + lpad);

        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }

    /// @brief Returns whether the wrapped function is currently linear.
    /// @return True if the wrapped function reports itself linear.
    bool is_linear() const { return this->func_.is_linear(); }

    /// @brief Constructs an empty padded-output wrapper (configured later).
    PaddedOutput() {}

    /// @internal
    /// @brief Evaluates the wrapped function into the unpadded middle segment.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Padded output vector; only the middle segment is written.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        this->func_.compute(
            x, fx.template segment<Func::ORC>(this->u_pad_, this->func_.output_rows()));
    }

    /// @internal
    /// @brief Evaluates value and Jacobian into the unpadded middle rows.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param x    Input vector.
    /// @param fx_  Padded output vector; only the middle segment is written.
    /// @param jx_  Padded Jacobian; only the middle rows are written.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        this->func_.compute_jacobian(
            x, fx.template segment<Func::ORC>(this->u_pad_, this->func_.output_rows()),
            jx.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()));
    }

    /// @internal
    /// @brief Evaluates value, Jacobian, adjoint gradient, and adjoint Hessian.
    ///
    /// The adjoint variables for the padded rows are ignored; only the middle
    /// segment is forwarded to the wrapped function.
    /// @tparam InType       Eigen input vector expression type.
    /// @tparam OutType      Eigen output vector expression type.
    /// @tparam JacType      Eigen Jacobian matrix expression type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector expression type.
    /// @param x        Input vector.
    /// @param fx_      Padded output vector; only the middle segment is written.
    /// @param jx_      Padded Jacobian; only the middle rows are written.
    /// @param adjgrad_ Adjoint-gradient buffer (full input width).
    /// @param adjhess_ Adjoint-Hessian buffer (full input width).
    /// @param adjvars  Adjoint seed vector; only the middle segment is used.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        // typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

        this->func_.compute_jacobian_adjointgradient_adjointhessian(
            x, fx.template segment<Func::ORC>(this->u_pad_, this->func_.output_rows()),
            jx.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()), adjgrad,
            adjhess, adjvars.template segment<Func::ORC>(this->u_pad_, this->func_.output_rows()));
    }

    //////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Forwards a right-Jacobian product into the wrapped function's rows.
    /// @tparam Target      Eigen target-matrix expression type.
    /// @tparam Left        Eigen left-operand expression type.
    /// @tparam Right       Eigen right-operand expression type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @tparam Aliased     Whether the operands may alias the target.
    /// @param target_  Destination matrix for the product.
    /// @param left     Left operand of the product.
    /// @param right    Right operand of the product.
    /// @param assign   Assignment policy (assign vs. accumulate).
    /// @param aliased  Aliasing flag forwarded to the wrapped function.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            CMatRef<Right> right_ref(right.derived());
            CDiagRef<Left> left_ref(left.derived());
            Base::right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
        } else {
            MatRef<Right> right_ref = right.const_cast_derived();
            MatRef<Left> left_ref = left.const_cast_derived();
            this->func_.right_jacobian_product(
                target_,
                left_ref.template middleCols<Func::ORC>(this->u_pad_, this->func_.output_rows()),
                right_ref.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()),
                assign, aliased);
        }
    }

    /// @internal
    /// @brief Accumulates a Jacobian block into the unpadded middle rows.
    /// @tparam Target      Eigen target-matrix expression type.
    /// @tparam JacType     Eigen source-Jacobian expression type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @param target_  Destination Jacobian (padded).
    /// @param right_   Source Jacobian block.
    /// @param assign   Assignment policy.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right_,
                                    Assignment assign) const {
        MatRef<Target> target = target_.const_cast_derived();
        MatRef<JacType> right = right_.const_cast_derived();

        this->func_.accumulate_jacobian(
            target.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()),
            right.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()), assign);
    }
    /// @internal
    /// @brief Forwards a gradient accumulation to the wrapped function.
    /// @tparam Target      Eigen target expression type.
    /// @tparam JacType     Eigen source expression type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @param target_  Destination gradient.
    /// @param right    Source gradient contribution.
    /// @param assign   Assignment policy.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_gradient(target_, right, assign);
    }
    /// @internal
    /// @brief Forwards a Hessian accumulation to the wrapped function.
    /// @tparam Target      Eigen target expression type.
    /// @tparam JacType     Eigen source expression type.
    /// @tparam Assignment  Assignment-policy tag type.
    /// @param target_  Destination Hessian.
    /// @param right    Source Hessian contribution.
    /// @param assign   Assignment policy.
    /// @endinternal
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        this->func_.accumulate_hessian(target_, right, assign);
    }
    /// @internal
    /// @brief Scales the wrapped function's Jacobian rows within the middle block.
    /// @tparam Target  Eigen target-matrix expression type.
    /// @tparam Scalar  Scalar factor type.
    /// @param target_  Jacobian to scale (padded).
    /// @param s        Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        MatRef<Target> target = target_.const_cast_derived();
        this->func_.scale_jacobian(
            target.template middleRows<Func::ORC>(this->u_pad_, this->func_.output_rows()), s);
    }
    /// @internal
    /// @brief Forwards a gradient scaling to the wrapped function.
    /// @tparam Target  Eigen target expression type.
    /// @tparam Scalar  Scalar factor type.
    /// @param target_  Gradient to scale.
    /// @param s        Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_gradient(target_, s);
    }
    /// @internal
    /// @brief Forwards a Hessian scaling to the wrapped function.
    /// @tparam Target  Eigen target expression type.
    /// @tparam Scalar  Scalar factor type.
    /// @param target_  Hessian to scale.
    /// @param s        Scalar factor.
    /// @endinternal
    template <class Target, class Scalar>
    inline void scale_hessian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_hessian(target_, s);
    }
};

} // namespace tycho::vf
