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
//   - pybind11 Build() methods moved to bindings layer (PR 2)
//   - Migrated pybind11 -> nanobind (PR 3)
// =============================================================================

#include "pch.h"

#include "Tycho_VectorFunctions.h"

namespace Tycho {

void ExtensionsBuild(FunctionRegistry &reg, nb::module_ &extmod);

}