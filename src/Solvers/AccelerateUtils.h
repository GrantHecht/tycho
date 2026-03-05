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

#include <cstdlib>
#include <string>

// Sets the number of threads for Accelerate via the environment variable VECLIB_MAXIMUM_THREADS.
// Currently, the VECLIB_MAXIMUM_THREADS is the only way to specify the number of threads
// Accelerate will use. This should be swapped to an Accelerate API call if Apple ever adds one.
inline void accelerate_set_num_threads(int num_threads) {
    setenv("VECLIB_MAXIMUM_THREADS", std::to_string(num_threads).c_str(), 1);
}
