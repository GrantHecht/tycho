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

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

void GenericODESBuildPart3(FunctionRegistry &reg, nb::module_ &m) {

    bind::BuildGenODEModule<GenericFunction<-1, -1>, 6, 3, 0>("ode_6_3", m, reg);
}

} // namespace tycho