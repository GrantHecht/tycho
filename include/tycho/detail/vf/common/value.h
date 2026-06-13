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
/// @brief Literal-vector VectorFunction primitive.
#pragma once

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @ingroup vf
/// @brief A VectorFunction that always returns a fixed stored vector.
///
/// This is the canonical literal-vector primitive in the VectorFunction DSL.
/// The stored value is returned regardless of the input and the derivatives are
/// zero.
/// @tparam OSZ  Compile-time input/output row count (Eigen::Dynamic for runtime size).
template <int OSZ> struct Value : VectorFunction<Value<OSZ>, OSZ, OSZ> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base = VectorFunction<Value<OSZ>, OSZ, OSZ>;
    VF_TYPE_ALIASES(Base)

    /// @brief Re-argument alias: substituting any function yields the same Value.
    /// @tparam Func  Candidate function being substituted (ignored).
    template <class Func> using REARGUMENT = Value<OSZ>;

    /// @brief The fixed output vector returned for every input.
    typename Base::template Output<double> value_;

    /// @brief Constructs a Value initialized to all ones.
    Value() { this->value_.setOnes(); }

    /// @internal
    /// @brief Writes the fixed stored value into the output.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param x    Input vector (unused).
    /// @param fx_  Output vector to fill with the stored value.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value_;
    }
    /// @internal
    /// @brief Computes value and Jacobian; the Jacobian is left zero.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param x    Input vector (unused).
    /// @param fx_  Output vector to fill with the stored value.
    /// @param jx_  Jacobian buffer; not written (a constant's Jacobian is zero).
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        fx = this->value_;
    }

    /// @brief Re-argues this Value against another function, returning a fresh Value.
    ///
    /// A constant Value does not depend on its argument, so re-arguing simply
    /// yields a default-constructed Value of the same size.
    /// @tparam Func     Wrapped function type of the new argument.
    /// @tparam FuncIRC  Compile-time input row count of the new argument.
    /// @tparam ii       Compile-time output row count of the new argument.
    /// @param f  The function being substituted as the new argument (ignored).
    /// @return A new Value with the same output size.
    template <class Func, int FuncIRC, int ii>
    decltype(auto) rearged(const DenseFunctionBase<Func, FuncIRC, ii> &f) const {
        return Value<OSZ>();
    }
};

} // namespace tycho::vf
