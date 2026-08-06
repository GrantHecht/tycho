// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include <Accelerate/Accelerate.h>
#include <cstdlib>
#include <string>

// BLASSetThreading (macOS 15+) provides per-thread dynamic control via
// thread-local storage. It supports a binary toggle: single-threaded vs
// multi-threaded. "Multi" returns to the thread count cached from
// VECLIB_MAXIMUM_THREADS at the first BLAS call.
//
// VECLIB_MAXIMUM_THREADS is read exactly once — at the first BLAS call.
// setenv() after that point is a no-op. BLASSetThreading is the only
// working dynamic control (macOS 15+).
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
#define TYCHO_HAS_BLAS_SET_THREADING 1
#endif

// MT-METIS (multi-threaded METIS) is available starting in macOS 26.
#if defined(__APPLE__) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&                               \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
#define TYCHO_HAS_MTMETIS 1
#endif

// Maps a requested ordering onto one this HOST supports.
//
// TYCHO_HAS_MTMETIS is an SDK-version macro: it says the enum constant exists
// at compile time, not that the running OS implements it. Passing
// SparseOrderMTMetis on macOS < 26 yields SparseParameterError and a dead
// solver, so downgrade at runtime instead.
inline SparseOrder_t accelerate_supported_order(SparseOrder_t order) {
#ifdef TYCHO_HAS_MTMETIS
    // __builtin_available must be the sole condition of its `if` -- combining
    // it with `&&`/`!` in one expression (as a first draft of this function
    // did) trips -Wunsupported-availability-guard, because the compiler can
    // no longer identify the guarded scope. Two single-condition ifs instead.
    if (order != SparseOrderMTMetis)
        return order;
    if (__builtin_available(macOS 26.0, *))
        return order;
    return SparseOrderMetis;
#else
    return order;
#endif
}

// Warm up the Accelerate sparse solver subsystem by performing a trivial
// LDLT factorization. On macOS 26+, this triggers MT-METIS thread pool
// initialization so the first real solve doesn't pay the cold-start penalty.
inline void warmup_sparse_solver() {
    // Tiny 2x2 SPD matrix (upper triangle of [[2,-1],[-1,2]])
    long columnStarts[] = {0, 1, 3};
    int rowIndices[] = {0, 0, 1};
    double values[] = {2.0, -1.0, 2.0};

    SparseMatrixStructure structure{};
    structure.rowCount = 2;
    structure.columnCount = 2;
    structure.columnStarts = columnStarts;
    structure.rowIndices = rowIndices;
    structure.attributes.kind = SparseSymmetric;
    structure.attributes.triangle = SparseUpperTriangle;
    structure.blockSize = 1;

    SparseMatrix_Double A{};
    A.structure = structure;
    A.data = values;

    SparseSymbolicFactorOptions fopts{};
    fopts.control = SparseDefaultControl;
    fopts.orderMethod = accelerate_supported_order(
#ifdef TYCHO_HAS_MTMETIS
        SparseOrderMTMetis
#else
        SparseOrderMetis
#endif
    );
    fopts.malloc = malloc;
    fopts.free = free;
    // Without a reportError callback, an Accelerate parameter-check failure
    // takes its null-callback branch: os_log_error followed by _SparseTrap(),
    // i.e. an abort on the startup path.
    fopts.reportError = [](const char *) {};

    auto sym = SparseFactor(SparseFactorizationLDLTTPP, structure, fopts);
    if (sym.status == SparseStatusOK) {
        auto num = SparseFactor(sym, A);
        // Apple requires SparseCleanup even for a FAILED factorization.
        SparseCleanup(num);
    }
    SparseCleanup(sym);
}

// Called once at startup (before any BLAS call) to set the Accelerate thread
// pool size and warm up the runtime (thread pool, CPU dispatch, memory
// allocators). VECLIB_MAXIMUM_THREADS must be set before the first BLAS call;
// it is cached at that point and setenv() is a no-op afterwards.
inline void ensure_accelerate_initialized(int num_threads) {
    setenv("VECLIB_MAXIMUM_THREADS", std::to_string(num_threads).c_str(), 1);
    // Trigger BLAS/vDSP runtime initialization.
    double x = 1.0, result = 0.0;
    vDSP_dotprD(&x, 1, &x, 1, &result, 1);
    // Warm up the sparse solver subsystem (including MT-METIS thread pool).
    warmup_sparse_solver();
}

// Dynamic thread control for Accelerate BLAS/LAPACK operations.
// On macOS 15+, uses BLASSetThreading (per-thread, thread-local).
// On older systems, falls back to VECLIB_MAXIMUM_THREADS env var
// (global, only effective before first BLAS call).
inline void accelerate_set_num_threads(int num_threads) {
#ifdef TYCHO_HAS_BLAS_SET_THREADING
    // The #ifdef only guarantees the DECLARATION exists in this SDK.
    // BLASSetThreading is API_AVAILABLE(macos(15.0)) and therefore weak-linked,
    // so a binary built against a newer SDK with an older deployment target
    // would null-call it. The runtime check is the actual guard.
    if (__builtin_available(macOS 15.0, *)) {
        if (num_threads <= 1)
            BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        else
            BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
        return;
    }
#endif
    setenv("VECLIB_MAXIMUM_THREADS", std::to_string(num_threads).c_str(), 1);
}
