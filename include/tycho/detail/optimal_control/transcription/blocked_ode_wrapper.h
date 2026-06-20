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
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

/// @internal
/// @brief Re-presents an ODE's controls as parameters for block-constant transcription.
///
/// Wraps an ODE so that, from the defect machinery's perspective, the controls
/// become part of the parameter block (zero controls, controls+params
/// parameters). Used when controls are held constant over each transcription
/// block.
/// @tparam DODE  The wrapped ODE type.
/// @endinternal
template <class DODE> struct Blocked_ODE_Wrapper : DODE {
    static constexpr int UV = 0; ///< No controls in the wrapped view.
    static constexpr int PV = SZ_SUM<DODE::PV, DODE::UV>::value; ///< Controls folded into params.
    static constexpr int XtUV = DODE::XtV;                       ///< State + time dimension.
    using Base = DODE; ///< @brief Convenience alias for the wrapped ODE base class.

    inline int xtu_vars() const { return Base::xt_vars(); } ///< @brief State + time variable count.
    inline int u_vars() const { return 0; }                 ///< @brief Control count (always zero).
    /// @brief Parameter count (original controls plus parameters).
    /// @return The combined control + parameter count of the wrapped ODE.
    inline int p_vars() const { return Base::u_vars() + Base::p_vars(); }

    /// @brief Default constructor.
    Blocked_ODE_Wrapper() {};
    /// @brief Wrap an existing ODE.
    /// @param ode  The ODE to wrap.
    Blocked_ODE_Wrapper(const DODE &ode) : Base(ode) {}
};

} // namespace tycho::oc
