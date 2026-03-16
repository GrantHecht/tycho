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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

#include <Accelerate/Accelerate.h>
#include <cstdlib>
#include <string>

// BLASSetThreading (macOS 15+) provides per-thread control via thread-local
// storage. On older systems, fall back to the global VECLIB_MAXIMUM_THREADS
// environment variable.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
#define TYCHO_HAS_BLAS_SET_THREADING 1
#endif

inline void accelerate_set_num_threads(int num_threads) {
#ifdef TYCHO_HAS_BLAS_SET_THREADING
    if (num_threads <= 1)
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    else
        BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
#else
    setenv("VECLIB_MAXIMUM_THREADS", std::to_string(num_threads).c_str(), 1);
#endif
}
