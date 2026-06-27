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
namespace tycho::utils {

/// @brief Return the number of physical CPU cores on the current machine.
///
/// Uses platform-specific APIs (e.g. `sysconf`, `GetSystemInfo`) to query the
/// physical core count. Falls back to `std::thread::hardware_concurrency()`
/// when the platform query is unavailable or fails.
///
/// @return The physical core count, or 1 if detection fails.
int get_core_count();

} // namespace tycho::utils
