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

#include "BumpAllocatorBind.h"
#include "Utils/tycho_utils.h"

namespace Tycho {

void UtilsBuild(nb::module_ &m) {
    auto um = m.def_submodule("Utils", "Contains miscilanaeous utilities");
    um.def("get_core_count", &Tycho::get_core_count);
    um.def("set_num_threads", &Tycho::set_num_threads, nb::arg("n"),
           "Set the number of threads for parallel work. "
           "n=1 for single-threaded mode, n>1 to use n threads.");
    um.def("get_num_threads", &Tycho::get_num_threads,
           "Get the current thread count setting. 1 = single-threaded mode.");
    TychoBind<BumpAllocator>::Build(um);
}

} // namespace Tycho
