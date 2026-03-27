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

#include "tycho/detail/integrators/rk_coeffs.h"

namespace Tycho {

void RKFlagsBuild(nb::module_ &m) {
    nb::enum_<RKOptions>(m, "RKOptions")
        .value("RK4", RKOptions::RK4Classic)
        .value("DOPRI54", RKOptions::DOPRI54)
        .value("DOPRI87", RKOptions::DOPRI87);
}

} // namespace Tycho
