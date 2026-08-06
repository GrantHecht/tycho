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

#include "tycho/detail/solvers/parsed_io_flags.h"
#include "tycho/detail/solvers/threading_flags.h"

namespace tycho::vf {

// Two of the three flag enums this header used to define now live in the solver
// layer, which owns the state they describe: SolverIndexingData tags its index
// maps with ParsedIOFlags, and SolverFunctionBase stores a ThreadingFlags per
// function. Keeping the definitions here made the solver headers reach into the
// VectorFunction tree for them. They are re-exported under their original names
// so that tycho::vf::ParsedIOFlags and tycho::vf::ThreadingFlags -- and every
// unqualified use inside this namespace -- name exactly the same types they
// always did.
using tycho::solvers::ParsedIOFlags;
using tycho::solvers::ThreadingFlags;

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
