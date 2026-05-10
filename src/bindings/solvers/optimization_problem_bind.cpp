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

#include "optimization_problem_bind.h"
#include "tycho/detail/solvers/optimization_problem.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

void TychoBind<OptimizationProblem>::build(nb::module_ &m) {
    using VectorFunctionalX = OptimizationProblem::VectorFunctionalX;
    using ScalarFunctionalX = OptimizationProblem::ScalarFunctionalX;
    using VectorXi = OptimizationProblem::VectorXi;

    auto obj = nb::class_<OptimizationProblem, OptimizationProblemBase>(m, "OptimizationProblem");

    obj.def(nb::init<>());

    obj.def("set_vars", &OptimizationProblem::set_vars);
    obj.def("return_vars", &OptimizationProblem::return_vars);

    obj.def("add_equal_con", nb::overload_cast<VectorFunctionalX, const std::vector<VectorXi> &>(
                                 &OptimizationProblem::add_equal_con));

    obj.def("add_equal_con",
            nb::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::add_equal_con));

    obj.def("add_inequal_con", nb::overload_cast<VectorFunctionalX, const std::vector<VectorXi> &>(
                                   &OptimizationProblem::add_inequal_con));

    obj.def("add_inequal_con",
            nb::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::add_inequal_con));

    obj.def("add_objective", nb::overload_cast<ScalarFunctionalX, const std::vector<VectorXi> &>(
                                 &OptimizationProblem::add_objective));

    obj.def("add_objective",
            nb::overload_cast<ScalarFunctionalX, VectorXi>(&OptimizationProblem::add_objective));
}
