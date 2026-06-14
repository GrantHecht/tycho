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

/// @brief Wraps a function so its derivatives are taken by finite differences.
///
/// ADFun forwards the primal value of @p Func but overrides its derivative modes
/// to finite differencing (central-difference array Jacobian, forward-difference
/// Hessian). It is primarily a debugging/validation aid: the test() helpers
/// compare the wrapped function's own Jacobian and Hessian (computed with
/// whatever derivative mode @p Func was configured with) against the
/// finite-difference reference. The "AD" name is historical; this path is
/// finite-difference, not algorithmic differentiation.
/// @tparam Func  The wrapped VectorFunction whose value is evaluated.
/// @ingroup vf
template <class Func>
struct ADFun : VectorFunction<ADFun<Func>, Func::IRC, Func::ORC,
                              DenseDerivativeMode::FDiffCentArray, DenseDerivativeMode::FDiffFwd> {
    /// @brief CRTP VectorFunction base with finite-difference derivative modes.
    using Base = VectorFunction<ADFun<Func>, Func::IRC, Func::ORC,
                                DenseDerivativeMode::FDiffCentArray, DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base)

    Func func_; ///< @brief The wrapped function whose value is evaluated.
    /// @brief Construct by taking ownership of the wrapped function.
    /// @param f  Function to wrap.
    ADFun(Func f) : func_(std::move(f)) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
    }
    /// @internal
    /// @brief Forward the primal value to the wrapped function (CRTP compute hook).
    /// @tparam InType   Input vector expression type.
    /// @tparam OutType  Output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to fill.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        this->func_.compute(x, fx_);
    }

    /// @brief Run the derivative-comparison check at a random input.
    void test() {
        Input<double> x;
        x.setRandom();
        return test_derivs(x);
    };
    /// @brief Run the derivative-comparison check at a given input.
    /// @param x  Input point to evaluate at.
    void test(Input<double> x) { return test_derivs(x); };
    /// @brief Print the Jacobian and Hessian error between finite-diff and the wrapped function.
    /// @param x  Input point to evaluate at.
    void test_derivs(Input<double> x) {
        Output<double> l;
        l.setOnes();

        std::cout << "Jacobian Error" << std::endl;
        std::cout << this->jacobian(x) - this->func_.jacobian(x) << std::endl << std::endl;
        ;
        std::cout << "Hessian Error" << std::endl;
        std::cout << (this->adjointhessian(x, l) - this->func_.adjointhessian(x, l)) << std::endl
                  << std::endl;
        ;
    }
};

} // namespace tycho::vf
