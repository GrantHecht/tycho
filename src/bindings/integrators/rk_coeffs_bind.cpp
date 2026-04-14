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
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#include "tycho/detail/integrators/rk_coeffs.h"

namespace tycho {
using namespace tycho::vf;
using namespace tycho::integrators;

void RKFlagsBuild(nb::module_ &m) {
    nb::enum_<IVPAlg>(m, "IVPAlg")
        .value("DOPRI54", IVPAlg::DOPRI54)
        .value("DOPRI87", IVPAlg::DOPRI87);
}

} // namespace tycho
