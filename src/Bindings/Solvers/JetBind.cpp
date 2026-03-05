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

#include "JetBind.h"
#include "Solvers/Jet.h"

using namespace Tycho;

void TychoBind<Jet>::Build(nb::module_ &m) {
    auto obj = nb::class_<Jet>(m, "Jet");

    obj.def_static(
        "map",
        [](const std::vector<std::shared_ptr<OptimizationProblemBase>> &optprobs, int nt) {
            return Jet::map(optprobs, nt, true);
        },
        nb::call_guard<nb::gil_scoped_release>());

    obj.def_static(
        "map",
        [](std::function<std::shared_ptr<OptimizationProblemBase>(nb::detail::args_proxy)> genfun,
           const std::vector<nb::args> &args, int nt) { return Jet::map(genfun, args, nt, true); },
        nb::call_guard<nb::gil_scoped_release>());

    obj.def_static(
        "map",
        [](const std::vector<std::shared_ptr<OptimizationProblemBase>> &optprobs, int nt, bool v) {
            return Jet::map(optprobs, nt, v);
        },
        nb::call_guard<nb::gil_scoped_release>());

    obj.def_static(
        "map",
        [](std::function<std::shared_ptr<OptimizationProblemBase>(nb::detail::args_proxy)> genfun,
           const std::vector<nb::args> &args, int nt,
           bool v) { return Jet::map(genfun, args, nt, v); },
        nb::call_guard<nb::gil_scoped_release>());
}
