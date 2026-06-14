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
/// @brief Compile-time indexing VectorFunction selecting input components.
#pragma once

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @ingroup vf
/// @brief Selects a fixed set of input components by compile-time index.
///
/// Each output row copies one input component named by a template index, so the
/// output dimension equals the number of indices. The mapping is linear: the
/// Jacobian holds one unit entry per output row and the adjoint Hessian is zero.
/// @tparam IR   Compile-time input row count (Eigen::Dynamic for runtime size).
/// @tparam EL1  Index of the first selected input component.
/// @tparam ELS  Indices of any further selected input components.
template <int IR, int EL1, int... ELS>
struct Elements : VectorFunction<Elements<IR, EL1, ELS...>, IR, 1 + sizeof...(ELS)> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base = VectorFunction<Elements<IR, EL1, ELS...>, IR, 1 + sizeof...(ELS)>;
    VF_TYPE_ALIASES(Base);
    using Base::compute;
    /// @brief Tuple of the selected input indices as integral constants.
    static const std::tuple<std::integral_constant<int, EL1>, std::integral_constant<int, ELS>...>
        elements;
    /// @brief Number of selected elements (the output dimension).
    static constexpr int num_elements = 1 + sizeof...(ELS);

    /// @brief Constructs an empty element selector (input size set later).
    Elements() {}
    /// @brief Constructs an element selector for a given input size.
    /// @param irows  Number of input rows.
    Elements(int irows) { this->set_io_rows(irows, num_elements); }

    /// @brief Marks the selection as a linear function.
    static constexpr bool is_linear_function = true;
    /// @internal
    /// @brief Copies each selected input component into the output.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector receiving the selected components.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        tycho::utils::tuple_for_loop(elements,
                                     [&](const auto &ele, auto i) { fx[i] = x[ele.value]; });
    }
    /// @internal
    /// @brief Copies the selected components and sets the unit Jacobian entries.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector receiving the selected components.
    /// @param jx_  Jacobian buffer; one unit entry is set per selected column.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        tycho::utils::tuple_for_loop(elements, [&](const auto &ele, int i) {
            fx[i] = x[ele.value];
            jx(i, ele.value) = 1.0;
        });
    }
    /// @internal
    /// @brief Computes value, Jacobian, adjoint gradient, and adjoint Hessian.
    ///
    /// The selection is linear, so the adjoint Hessian is zero; the adjoint
    /// gradient scatters each adjoint variable back to its selected input slot.
    /// @tparam InType       Eigen input vector expression type.
    /// @tparam OutType      Eigen output vector expression type.
    /// @tparam JacType      Eigen Jacobian matrix expression type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector expression type.
    /// @param x        Input vector.
    /// @param fx_      Output vector receiving the selected components.
    /// @param jx_      Jacobian buffer; one unit entry is set per selected column.
    /// @param adjgrad_ Adjoint-gradient buffer; selected slots are scattered into.
    /// @param adjhess_ Adjoint-Hessian buffer; left unchanged (zero).
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
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

        tycho::utils::tuple_for_loop(elements, [&](const auto &ele, int i) {
            fx[i] = x[ele.value];
            jx(i, ele.value) = 1.0;
            adjgrad[ele.value] = adjvars[i];
        });
    }
};

} // namespace tycho::vf
