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
    auto um = m.def_submodule("utils", "Contains miscellaneous utilities");
    um.def("get_core_count", &tycho::utils::get_core_count,
           R"doc(Return the number of physical CPU cores on the current machine.

Uses platform-specific APIs to query the physical core count:
``/proc/cpuinfo`` on Linux, ``sysctlbyname("hw.physicalcpu")`` on macOS,
and ``GetLogicalProcessorInformation`` on Windows.  The result is capped
at ``std::thread::hardware_concurrency()``.  Falls back to
``std::thread::hardware_concurrency()`` when the platform query is
unavailable or fails.

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
    Thread count.  ``n <= 1`` (including 0/negative) selects
    single-threaded mode: work runs inline on the calling thread; the
    pool stays alive but is bypassed.  ``n > 1`` resizes the pool to
    ``n`` worker threads.
)doc");
    um.def("get_num_threads", &tycho::utils::get_num_threads,
           R"doc(Return the current process-global thread count.

The default before any :func:`set_num_threads` call is
``std::thread::hardware_concurrency()``.

Returns
-------
int
    Number of threads.  ``1`` means single-threaded mode.
)doc");
    TychoBind<BumpAllocator>::build(um);
}

} // namespace tycho
