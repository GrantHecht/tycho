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

#include "tycho/detail/utils/sizing_helpers.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_NEG;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

/// @internal
/// @brief Compile-time input/output sizes of a collocation defect function.
///
/// Derives the packed-input and output row counts of a defect constraint from
/// the ODE dimensions and the number of cardinal states per interval.
/// @tparam _CS    Number of cardinal states per defect interval.
/// @tparam ODEXV  ODE state-vector dimension.
/// @tparam ODEUV  ODE control-vector dimension.
/// @tparam ODEPV  ODE parameter-vector dimension.
/// @endinternal
template <int _CS, int ODEXV, int ODEUV, int ODEPV> struct DefectConstSizes {
    static constexpr int CS = _CS;                   ///< Cardinal states per interval.
    static constexpr int IS = SZ_NEG<CS - 1>::value; ///< Number of interior states.
    static constexpr int XTU =
        SZ_SUM<ODEXV, ODEUV, 1>::value; ///< State + time + control dimension.
    static constexpr int DefIRC =
        SZ_SUM<SZ_PROD<CS, XTU>::value, ODEPV>::value;       ///< Defect input row count.
    static constexpr int DefORC = SZ_PROD<IS, ODEXV>::value; ///< Defect output row count.
};

} // namespace tycho::oc
