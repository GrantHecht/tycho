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

#include "bump_allocator_bind.h"
#include "tycho/detail/utils/memory_management.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

void TychoBind<BumpAllocator>::Build(nb::module_ &m) {
    auto obj = nb::class_<BumpAllocator>(m, "BumpAllocator");
    obj.def_static("resize", nb::overload_cast<int>(&BumpAllocator::resize));
    obj.def_static("resize", nb::overload_cast<int, int>(&BumpAllocator::resize));
    obj.def_static("size_scalar", &BumpAllocator::size_scalar);
    obj.def_static("size_super_scalar", &BumpAllocator::size_super_scalar);
}
