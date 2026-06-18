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

#include "mesh_iterate_info_bind.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

void TychoBind<MeshIterateInfo>::build(nb::module_ &m) {
    auto obj = nb::class_<MeshIterateInfo>(
        m, "MeshIterateInfo",
        R"doc(Per-iteration diagnostics of an adaptive-mesh refinement pass.

One :class:`MeshIterateInfo` is recorded per mesh-refinement iteration of a
phase; the full history is returned by ``phase.get_mesh_iters()``. The arrays
describe the node times, the per-segment error estimate, and the resulting
error distribution that drives the next round of node redistribution.
)doc");

    obj.def_ro("times", &MeshIterateInfo::times_,
               "numpy.ndarray, shape (numsegs + 1,): node times of the mesh for this iteration, "
               "normalized to ``[0, 1]``.");
    obj.def_ro("error", &MeshIterateInfo::error_,
               "numpy.ndarray, shape (numsegs,): per-segment estimated discretization error.");
    obj.def_ro("distribution", &MeshIterateInfo::distribution_,
               "numpy.ndarray: error-density distribution over the mesh, used to redistribute "
               "nodes for the next iteration.");
    obj.def_ro("distintegral", &MeshIterateInfo::distintegral_,
               "numpy.ndarray: normalized cumulative integral of :attr:`distribution`.");
    obj.def_ro("avg_error", &MeshIterateInfo::avg_error_,
               "float: time-weighted average per-segment error.");
    obj.def_ro("max_error", &MeshIterateInfo::max_error_,
               "float: maximum per-segment error over the mesh.");
    obj.def_ro("numsegs", &MeshIterateInfo::numsegs_,
               "int: number of segments in this iteration's mesh.");
    obj.def_ro("converged", &MeshIterateInfo::converged_,
               "bool: whether the mesh met the error tolerance on this iteration.");
}
