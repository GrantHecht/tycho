// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// The thread assignment policy a solver function carries. Owned by the solver
// layer: SolverFunctionBase stores one per function and NonLinearProgram's
// partitioner is the only thing that reads it. It used to live with the
// VectorFunction flags, which made the solver headers include a VectorFunction
// header for an enum the VectorFunction layer only forwards; the enum now lives
// here and vf/core/functional_flags.h re-exports it, so every existing
// tycho::vf::ThreadingFlags reference keeps working unchanged.
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

namespace tycho::solvers {

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

} // namespace tycho::solvers
