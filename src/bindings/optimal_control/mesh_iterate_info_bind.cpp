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

#include "mesh_iterate_info_bind.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

void TychoBind<MeshIterateInfo>::Build(nb::module_ &m) {
    auto obj = nb::class_<MeshIterateInfo>(m, "MeshIterateInfo");

    obj.def_ro("times", &MeshIterateInfo::times);
    obj.def_ro("error", &MeshIterateInfo::error);
    obj.def_ro("distribution", &MeshIterateInfo::distribution);
    obj.def_ro("distintegral", &MeshIterateInfo::distintegral);
    obj.def_ro("avg_error", &MeshIterateInfo::avg_error);
    obj.def_ro("max_error", &MeshIterateInfo::max_error);
    obj.def_ro("numsegs", &MeshIterateInfo::numsegs);
    obj.def_ro("converged", &MeshIterateInfo::converged);
}
