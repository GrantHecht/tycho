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
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Namespace: Tycho
// =============================================================================

#include "tycho_optimal_control.h"

namespace Tycho {

void GenericODESBuildPart6(FunctionRegistry &reg, nb::module_ &m) {

    Bind::BuildGenODEModule<GenericFunction<-1, -1>, 6, 0, 0>("ode_6", m, reg);

    Bind::BuildGenODEModule<GenericFunction<-1, -1>, 4, 0, 0>("ode_4", m, reg);
}

} // namespace Tycho