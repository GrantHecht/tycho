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

#include "tycho_vector_functions.h"

namespace Tycho {
void ArgsSegBuildPart4(FunctionRegistry &reg, nb::module_ &m) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using ELEM = Segment<-1, 1, -1>;

    reg.Build_Register<SEG2>(m, "Segment2");
}

} // namespace Tycho