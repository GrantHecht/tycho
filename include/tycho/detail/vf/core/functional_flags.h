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

/// @brief Classifies how a parsed input/output slot maps into a VF's storage.
/// @ingroup vf
enum class ParsedIOFlags {
    HiddenInput = -2,  ///< @brief Slot is an input not exposed to the caller.
    IngoreOutput = -1, ///< @brief Slot is an output that should be ignored.
    NotContiguous,     ///< @brief Slot maps to a non-contiguous range.
    Contiguous,        ///< @brief Slot maps to a contiguous range.
};

/// @brief Selects the thread assignment / dispatch strategy for a VF evaluation.
/// @ingroup vf
///
/// Negative values are special dispatch modes; non-negative values name an
/// explicit worker-thread index.
enum class ThreadingFlags : int {
    RoundRobin = -4,    ///< @brief Distribute work round-robin across the pool.
    MainThread = -3,    ///< @brief Run on the calling (main) thread.
    NeedsPool = -2,     ///< @brief Requires a thread pool to execute.
    ByApplication = -1, ///< @brief Threading decided by the application.
    Thread0 = 0,        ///< @brief Run on worker thread 0.
    Thread1 = 1,        ///< @brief Run on worker thread 1.
    Thread2 = 2,        ///< @brief Run on worker thread 2.
    Thread3 = 3,        ///< @brief Run on worker thread 3.
    Thread4 = 4,        ///< @brief Run on worker thread 4.
    Thread5 = 5,        ///< @brief Run on worker thread 5.
    Thread6 = 6,        ///< @brief Run on worker thread 6.
    Thread7 = 7,        ///< @brief Run on worker thread 7.
    Thread8 = 8,        ///< @brief Run on worker thread 8.
    Thread9 = 9,        ///< @brief Run on worker thread 9.
    Thread10 = 10,      ///< @brief Run on worker thread 10.
    Thread11 = 11,      ///< @brief Run on worker thread 11.
    Thread12 = 12,      ///< @brief Run on worker thread 12.
    Thread13 = 13,      ///< @brief Run on worker thread 13.
    Thread14 = 14,      ///< @brief Run on worker thread 14.
    Thread15 = 15,      ///< @brief Run on worker thread 15.
    Thread16 = 16,      ///< @brief Run on worker thread 16.
    Thread17 = 17,      ///< @brief Run on worker thread 17.
    Thread18 = 18,      ///< @brief Run on worker thread 18.
    Thread19 = 19,      ///< @brief Run on worker thread 19.
    Thread20 = 20,      ///< @brief Run on worker thread 20.
    Thread21 = 21,      ///< @brief Run on worker thread 21.
    Thread22 = 22,      ///< @brief Run on worker thread 22.
    Thread23 = 23,      ///< @brief Run on worker thread 23.
    Thread24 = 24,      ///< @brief Run on worker thread 24.
    Thread25 = 25,      ///< @brief Run on worker thread 25.
    Thread26 = 26,      ///< @brief Run on worker thread 26.
    Thread27 = 27,      ///< @brief Run on worker thread 27.
    Thread28 = 28,      ///< @brief Run on worker thread 28.
};

/// @brief Classifies how a variable enters a function (used by the NLP layer).
/// @ingroup vf
enum class VarTypes {
    NonLinear = 0, ///< @brief Variable enters nonlinearly.
    Linear = 1,    ///< @brief Variable enters linearly.
    Quadratic = 2, ///< @brief Variable enters quadratically.
    Inactive = 3,  ///< @brief Variable does not affect the function.
    BiLinear = 4,  ///< @brief Variable enters bilinearly with another variable.
};
} // namespace tycho::vf
