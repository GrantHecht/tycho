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

/// @brief Adapts a function to read its inputs from selected positions of a larger vector.
///
/// Wraps a function whose inputs are a subset of the caller's input vector,
/// gathering the required entries (named by the index map) before evaluating
/// the wrapped function and scattering its derivatives back to the
/// corresponding outer columns and rows. Detects the contiguous-range case for
/// a fast no-copy path.
///
/// @tparam Func  Wrapped VectorFunction whose inputs are gathered.
/// @tparam IRC   Input row count of the outer (gathered-from) vector.
/// @tparam ORC   Output row count (inherited from the wrapped function).
/// @ingroup vf
template <class Func, int IRC, int ORC>
struct ParsedInput
    : VectorFunction<ParsedInput<Func, IRC, ORC>, IRC, ORC, DenseDerivativeMode::Analytic> {
    /// @brief VectorFunction base carrying the wrapped function's IO sizes.
    using Base =
        VectorFunction<ParsedInput<Func, IRC, ORC>, IRC, ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    /// @brief The wrapped function whose inputs are gathered from the outer vector.
    Func func_;
    /// @brief Index of each wrapped input within the outer input vector.
    typename Func::template Input<int> varlocs_;
    /// @brief True when the gathered indices form a contiguous range.
    bool contiguous_ = false;

    /// @brief Default-construct an unconfigured wrapper.
    ParsedInput() {}
    /// @brief Construct from a wrapped function and its input-location map.
    /// @param f      Function to wrap.
    /// @param vlocs  Index of each wrapped input within the outer input vector.
    /// @param irr    Number of input rows of the outer (gathered-from) vector.
    ParsedInput(Func f, const typename Func::template Input<int> &vlocs, int irr)
        : func_(std::move(f)), varlocs_(vlocs) {
        this->set_io_rows(irr, this->func_.output_rows());

        this->contiguous_ = true;
        for (int i = 0; i < varlocs_.size() - 1; i++) {
            int delta = varlocs_[i + 1] - varlocs_[i];
            if (delta != 1)
                this->contiguous_ = false;
        }
        // this->set_input_domain(irr, func_.input_domain());
    }

    /// @brief Gather inner inputs from @p x and evaluate the inner function.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Outer input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;

        if (this->contiguous_) {
            this->func_.compute(x.segment(varlocs_[0], this->func_.input_rows()), fx_);

        } else {
            typename Func::template Input<Scalar> xin(this->func_.input_rows());
            for (int i = 0; i < this->func_.input_rows(); i++) {
                xin[i] = x[this->varlocs_[i]];
            }
            this->func_.compute(xin, fx_);
        }
    }
    /// @brief Evaluate value and Jacobian, scattering columns to outer indices.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Outer input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();

        using Scalar = typename InType::Scalar;

        if (this->contiguous_) {
            this->func_.compute_jacobian(x.segment(varlocs_[0], this->func_.input_rows()), fx_,
                                         jx.middleCols(varlocs_[0], this->func_.input_rows()));
        } else {
            typename Func::template Input<Scalar> xin(this->func_.input_rows());
            typename Func::template Jacobian<Scalar> jxin(this->func_.output_rows(),
                                                          this->func_.input_rows());
            jxin.setZero();
            for (int i = 0; i < this->func_.input_rows(); i++) {
                xin[i] = x[this->varlocs_[i]];
            }
            this->func_.compute_jacobian(xin, fx_, jxin);
            for (int i = 0; i < this->func_.input_rows(); i++) {
                jx.col(this->varlocs_[i]) = jxin.col(i);
            }
        }
    }
    /// @brief Evaluate value, Jacobian, adjoint gradient, and adjoint Hessian.
    ///
    /// Inner derivatives are computed on the gathered inputs, then scattered
    /// back to the outer columns/rows named by `varlocs_`.
    ///
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Outer input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @param  adjgrad_ Adjoint-gradient accumulator (size = `input_rows()`).
    /// @param  adjhess_ Adjoint-Hessian accumulator (`input_rows()` square).
    /// @param  adjvars  Adjoint variables seeding the reverse pass.
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

        using Scalar = typename InType::Scalar;

        if (this->contiguous_) {
            this->func_.compute_jacobian_adjointgradient_adjointhessian(
                x.segment(varlocs_[0], this->func_.input_rows()), fx_,
                jx.middleCols(varlocs_[0], this->func_.input_rows()),
                adjgrad.segment(varlocs_[0], this->func_.input_rows()),
                adjhess.block(varlocs_[0], varlocs_[0], this->func_.input_rows(),
                              this->func_.input_rows()),
                adjvars);

        } else {
            typename Func::template Input<Scalar> xin(this->func_.input_rows());
            typename Func::template Jacobian<Scalar> jxin(this->func_.output_rows(),
                                                          this->func_.input_rows());
            typename Func::template Gradient<Scalar> gxin(this->func_.input_rows());
            typename Func::template Hessian<Scalar> hxin(this->func_.input_rows(),
                                                         this->func_.input_rows());
            jxin.setZero();
            hxin.setZero();
            gxin.setZero();

            for (int i = 0; i < this->func_.input_rows(); i++) {
                xin[i] = x[this->varlocs_[i]];
            }

            this->func_.compute_jacobian_adjointgradient_adjointhessian(xin, fx_, jxin, gxin, hxin,
                                                                        adjvars);

            for (int i = 0; i < this->func_.input_rows(); i++) {
                jx.col(this->varlocs_[i]) = jxin.col(i);
                adjgrad[this->varlocs_[i]] = gxin[i];
            }
            for (int i = 0; i < this->func_.input_rows(); i++) {
                for (int j = 0; j < this->func_.input_rows(); j++) {
                    adjhess(this->varlocs_[j], this->varlocs_[i]) = hxin(j, i);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
