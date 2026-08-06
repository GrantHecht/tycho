// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// How one parsed input/output slot maps into storage. SolverIndexingData tags
// every column of its variable and constraint index maps with one of these, and
// the gather routines read the tag to choose between a segment copy and an
// element-by-element gather. It used to live with the VectorFunction flags,
// which made the solver's own indexing header include a VectorFunction header
// for an enum the two layers merely share; the enum now lives here and
// vf/core/functional_flags.h re-exports it, so every existing
// tycho::vf::ParsedIOFlags reference keeps working unchanged.
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

namespace tycho::solvers {

/// @brief Classifies how a parsed input/output slot maps into a VF's storage.
/// @ingroup vf
enum class ParsedIOFlags {
    HiddenInput = -2,  ///< @brief Slot is an input not exposed to the caller.
    IngoreOutput = -1, ///< @brief Slot is an output that should be ignored.
    NotContiguous,     ///< @brief Slot maps to a non-contiguous range.
    Contiguous,        ///< @brief Slot maps to a contiguous range.
};

} // namespace tycho::solvers
