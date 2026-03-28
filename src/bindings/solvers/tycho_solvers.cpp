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

#include "solvers/tycho_solvers.h"
#include "jet_bind.h"
#include "optimization_problem_bind.h"
#include "psiopt_bind.h"

namespace tycho {

using namespace tycho::solvers;

void SolversBuild(FunctionRegistry &reg, nb::module_ &m) {
    auto &sol = reg.getSolversModule();
    ensure_solver_initialized();
    TychoBind<PSIOPT>::Build(sol);
    TychoBind<OptimizationProblemBase>::Build(sol);
    TychoBind<Jet>::Build(sol);
    TychoBind<OptimizationProblem>::Build(sol);
}

} // namespace tycho
