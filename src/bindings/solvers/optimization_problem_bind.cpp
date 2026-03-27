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

#include "OptimizationProblemBind.h"
#include "tycho/detail/solvers/optimization_problem.h"

using namespace Tycho;

void TychoBind<OptimizationProblem>::Build(nb::module_ &m) {
    using VectorFunctionalX = OptimizationProblem::VectorFunctionalX;
    using ScalarFunctionalX = OptimizationProblem::ScalarFunctionalX;
    using VectorXi = OptimizationProblem::VectorXi;

    auto obj = nb::class_<OptimizationProblem, OptimizationProblemBase>(m, "OptimizationProblem");

    obj.def(nb::init<>());

    obj.def("setVars", &OptimizationProblem::setVars);
    obj.def("returnVars", &OptimizationProblem::returnVars);

    obj.def("addEqualCon", nb::overload_cast<VectorFunctionalX, const std::vector<VectorXi> &>(
                               &OptimizationProblem::addEqualCon));

    obj.def("addEqualCon",
            nb::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::addEqualCon));

    obj.def("addInequalCon", nb::overload_cast<VectorFunctionalX, const std::vector<VectorXi> &>(
                                 &OptimizationProblem::addInequalCon));

    obj.def("addInequalCon",
            nb::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::addInequalCon));

    obj.def("addObjective", nb::overload_cast<ScalarFunctionalX, const std::vector<VectorXi> &>(
                                &OptimizationProblem::addObjective));

    obj.def("addObjective",
            nb::overload_cast<ScalarFunctionalX, VectorXi>(&OptimizationProblem::addObjective));
}
