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

void TychoBind<BumpAllocator>::build(nb::module_ &m) {
    auto obj = nb::class_<BumpAllocator>(m, "BumpAllocator",
                                         R"doc(Per-thread bump allocator for ODE evaluation temporaries.

Tycho's ODE and VectorFunction evaluation stack uses thread-local
SIMD-aligned arena buffers to avoid heap allocation in inner loops.
This class exposes the management interface for those arenas.

There are two per-thread arenas:

* **Scalar arena** — holds ``double`` temporaries.
* **SuperScalar arena** — holds ``DefaultSuperScalar`` (SIMD-width
  ``double`` array) temporaries.

The default arena sizes are chosen automatically; use :meth:`resize` only
when profiling reveals arena overflow (fallback heap allocation) in tight
loops.
)doc");
    obj.def_static("resize", nb::overload_cast<int>(&BumpAllocator::resize),
                   R"doc(Resize both per-thread arenas to the same element count.

Must be called with no active allocations (i.e., outside any ODE or
VectorFunction evaluation call).

Parameters
----------
size : int
    New capacity in elements for both the scalar and super-scalar arenas.
)doc");
    obj.def_static("resize", nb::overload_cast<int, int>(&BumpAllocator::resize),
                   R"doc(Resize the scalar and super-scalar arenas independently.

Must be called with no active allocations (i.e., outside any ODE or
VectorFunction evaluation call).

Parameters
----------
size_scalar : int
    New capacity in ``double`` elements for the scalar arena.
size_super_scalar : int
    New capacity in ``DefaultSuperScalar`` elements for the
    super-scalar arena.
)doc");
    obj.def_static("size_scalar", &BumpAllocator::size_scalar,
                   R"doc(Return the current capacity of the per-thread scalar arena.

Returns
-------
int
    Capacity in ``double`` elements.
)doc");
    obj.def_static("size_super_scalar", &BumpAllocator::size_super_scalar,
                   R"doc(Return the current capacity of the per-thread super-scalar arena.

Returns
-------
int
    Capacity in ``DefaultSuperScalar`` elements.
)doc");
}
