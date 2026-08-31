// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#include "solvers/tycho_solvers.h"
#include "engines_bind.h"
#include "interior_point_solver_bind.h"
#include "jet_bind.h"
#include "optimization_problem_bind.h"
#include "solve_types_bind.h"

namespace tycho {

using namespace tycho::solvers;

void solvers_build(FunctionRegistry &reg, nb::module_ &m) {
    auto &sol = reg.getSolversModule();
    ensure_solver_initialized();
    TychoBind<InteriorPointSolver>::build(sol);
    TychoBind<SolveResult>::build(sol);
    TychoBind<SqpSolver>::build(sol);
    TychoBind<BackendProblemBase>::build(sol);
    TychoBind<Jet>::build(sol);
    TychoBind<OptimizationProblem>::build(sol);
}

} // namespace tycho
