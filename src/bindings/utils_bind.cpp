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
#include "utils/tycho_utils.h"

namespace tycho {
using namespace tycho::utils;

void utils_build(nb::module_ &m) {
    auto um = m.def_submodule("utils", "Contains miscilanaeous utilities");
    um.def("get_core_count", &tycho::utils::get_core_count,
           R"doc(Return the number of physical CPU cores on the current machine.

Uses platform-specific APIs (e.g. ``sysconf`` on Linux/macOS,
``GetSystemInfo`` on Windows) to query the physical core count.  Falls
back to ``std::thread::hardware_concurrency()`` when the platform query
is unavailable or fails.

Returns
-------
int
    Physical core count, or ``1`` if detection fails.
)doc");
    um.def("set_num_threads", &tycho::utils::set_num_threads, nb::arg("n"),
           R"doc(Set the number of threads used for parallel work.

This is the process-global execution budget and is intended to be called
once at startup, before any parallel computation begins.  Do not call it
from inside a parallel region or from a worker thread.

Parameters
----------
n : int
    Thread count.  ``n <= 1`` selects single-threaded mode (all work
    runs inline on the calling thread; the thread pool is not used).
    ``n > 1`` sizes the pool to ``n`` worker threads.
)doc");
    um.def("get_num_threads", &tycho::utils::get_num_threads,
           R"doc(Return the current process-global thread count.

Returns
-------
int
    Number of threads.  ``1`` means single-threaded mode.
)doc");
    TychoBind<BumpAllocator>::build(um);
}

} // namespace tycho
