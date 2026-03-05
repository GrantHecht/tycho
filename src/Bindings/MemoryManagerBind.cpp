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

#include "MemoryManagerBind.h"
#include "MemoryManagement.h"

using namespace Tycho;

void TychoBind<MemoryManager>::Build(nb::module_ &m) {
    auto obj = nb::class_<MemoryManager>(m, "MemoryManager");
    obj.def_static("enable_arena_memory", []() { MemoryManager::enable_arena_memory(); });
    obj.def_static("disable_arena_memory", []() { MemoryManager::disable_arena_memory(); });
}
