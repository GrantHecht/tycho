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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#include "tycho/detail/solvers/optimization_problem.h"

void tycho::solvers::OptimizationProblem::transcribe() {
    this->nlp = std::make_shared<NonLinearProgram>(this->NumPartitions);

    int numVars = this->ActiveVariables.size();

    if (numVars == 0) {
        fmt::print(fmt::fg(fmt::color::red), "Transcription Error!!!\n"
                                             "No variables provided to OptimizationProblem");
        throw std::invalid_argument("");
    }

    int numEqCons = 0;
    int numIqCons = 0;

    for (auto &func : this->userEqualities) {
        int irows = func.func.IRows();
        int orows = func.func.ORows();
        int numappl = func.indices.size();

        MatrixXi vindex(irows, numappl);
        MatrixXi cindex(orows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices[i].maxCoeff() > numVars) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Variable indices out of bounds in equality constraint");
                throw std::invalid_argument("");
            }
            vindex.col(i) = func.indices[i];
            for (int j = 0; j < orows; j++) {
                cindex(j, i) = numEqCons;
                numEqCons++;
            }
        }

        this->nlp->EqualityConstraints.emplace_back(ConstraintFunction(func.func, vindex, cindex));

        ThreadingFlags ThreadMode =
            func.func.thread_safe()
                ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                : ThreadingFlags::MainThread;

        this->nlp->EqualityConstraints.back().ThreadMode = ThreadMode;
    }

    for (auto &func : this->userInequalities) {
        int irows = func.func.IRows();
        int orows = func.func.ORows();
        int numappl = func.indices.size();

        MatrixXi vindex(irows, numappl);
        MatrixXi cindex(orows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices[i].maxCoeff() > numVars) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Variable indices out of bounds in inequality constraint");
                throw std::invalid_argument("");
            }
            vindex.col(i) = func.indices[i];
            for (int j = 0; j < orows; j++) {
                cindex(j, i) = numIqCons;
                numIqCons++;
            }
        }

        this->nlp->InequalityConstraints.emplace_back(
            ConstraintFunction(func.func, vindex, cindex));

        ThreadingFlags ThreadMode =
            func.func.thread_safe()
                ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                : ThreadingFlags::MainThread;

        this->nlp->InequalityConstraints.back().ThreadMode = ThreadMode;
    }

    for (auto &func : this->userObjectives) {
        int irows = func.func.IRows();
        int numappl = func.indices.size();

        MatrixXi vindex(irows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices[i].maxCoeff() > numVars) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Variable indices out of bounds in inequality constraint");
                throw std::invalid_argument("");
            }
            vindex.col(i) = func.indices[i];
        }

        this->nlp->Objectives.emplace_back(ObjectiveFunction(func.func, vindex));

        ThreadingFlags ThreadMode =
            func.func.thread_safe()
                ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                : ThreadingFlags::MainThread;

        this->nlp->Objectives.back().ThreadMode = ThreadMode;
    }

    this->nlp->make_NLP(numVars, numEqCons, numIqCons);
    this->optimizer->setNLP(this->nlp);

    //////DO NOT GET RID OF THIS!!!!!!//
    this->doTranscription = false;
    ////////////////////////////////////
}
