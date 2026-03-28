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

#include "tycho/detail/solvers/optimization_problem_base.h"
#include "optimization_problem_bind.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

void TychoBind<OptimizationProblemBase>::Build(nb::module_ &m) {
    using JetJobModes = OptimizationProblemBase::JetJobModes;
    auto obj = nb::class_<OptimizationProblemBase>(m, "OptimizationProblemBase");
    obj.def_rw("jet_job_mode", &OptimizationProblemBase::JetJobMode);
    obj.def_rw("num_partitions", &OptimizationProblemBase::NumPartitions);
    obj.def_ro("optimizer", &OptimizationProblemBase::optimizer);

    obj.def("set_num_partitions",
            nb::overload_cast<int, int>(&OptimizationProblemBase::setNumPartitions),
            nb::arg("NumPartitions"), nb::arg("QPThreads"));

    obj.def("set_num_partitions",
            nb::overload_cast<int>(&OptimizationProblemBase::setNumPartitions));

    obj.def("set_jet_job_mode",
            nb::overload_cast<JetJobModes>(&OptimizationProblemBase::setJetJobMode));
    obj.def("set_jet_job_mode",
            nb::overload_cast<const std::string &>(&OptimizationProblemBase::setJetJobMode));

    obj.def("solve", &OptimizationProblemBase::solve, nb::call_guard<nb::gil_scoped_release>());
    obj.def("optimize", &OptimizationProblemBase::optimize,
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("solve_optimize", &OptimizationProblemBase::solve_optimize,
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("solve_optimize_solve", &OptimizationProblemBase::solve_optimize_solve,
            nb::call_guard<nb::gil_scoped_release>());
    obj.def("optimize_solve", &OptimizationProblemBase::optimize_solve,
            nb::call_guard<nb::gil_scoped_release>());

    /// <summary>
    /// Probably need to move these enums somewhere else
    /// </summary>
    /// <param name="m"></param>
    nb::enum_<JetJobModes>(m, "JetJobModes")
        .value("DoNothing", JetJobModes::DoNothing)
        .value("NotSet", JetJobModes::NotSet)
        .value("Solve", JetJobModes::Solve)
        .value("Optimize", JetJobModes::Optimize)
        .value("SolveOptimize", JetJobModes::SolveOptimize)
        .value("SolveOptimizeSolve", JetJobModes::SolveOptimizeSolve)
        .value("OptimizeSolve", JetJobModes::OptimizeSolve);
}