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

namespace tycho::vf {

/// @brief Stores a VectorFunction's input and output row counts.
/// @ingroup vf
///
/// The primary template holds both sizes as compile-time constants. Partial
/// specializations promote a dynamic size (`-1`) to a runtime data member.
/// @tparam IR  Compile-time input row count, or `-1` if dynamic.
/// @tparam OR  Compile-time output row count, or `-1` if dynamic.
template <int IR, int OR> struct InputOutputSize {
    static constexpr int input_rows_val = IR;  ///< @brief Compile-time input row count.
    static constexpr int output_rows_val = OR; ///< @brief Compile-time output row count.
};

/// @brief Both input and output sizes dynamic.
/// @ingroup vf
template <> struct InputOutputSize<-1, -1> {
    int input_rows_val = 0;  ///< @brief Runtime input row count.
    int output_rows_val = 0; ///< @brief Runtime output row count.
};

/// @brief Dynamic input size, fixed output size.
/// @ingroup vf
/// @tparam OR  Compile-time output row count.
template <int OR> struct InputOutputSize<-1, OR> {
    int input_rows_val = 0;                    ///< @brief Runtime input row count.
    static constexpr int output_rows_val = OR; ///< @brief Compile-time output row count.
};

/// @brief Fixed input size, dynamic output size.
/// @ingroup vf
/// @tparam IR  Compile-time input row count.
template <int IR> struct InputOutputSize<IR, -1> {
    static constexpr int input_rows_val = IR; ///< @brief Compile-time input row count.
    int output_rows_val = 0;                  ///< @brief Runtime output row count.
};

} // namespace tycho::vf
