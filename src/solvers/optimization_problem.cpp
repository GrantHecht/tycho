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
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#include "tycho/detail/solvers/optimization_problem.h"

using tycho::solvers::ConstraintFunction;
using tycho::solvers::ObjectiveFunction;

void tycho::solvers::OptimizationProblem::transcribe() {
    this->nlp_ = std::make_shared<NonLinearProgram>(this->num_partitions_);

    int numVars = this->active_variables_.size();

    if (numVars == 0) {
        fmt::print(fmt::fg(fmt::color::red), "Transcription Error!!!\n"
                                             "No variables provided to OptimizationProblem");
        throw std::invalid_argument("");
    }

    int numEqCons = 0;
    int numIqCons = 0;

    for (auto &func : this->user_equalities_) {
        int irows = func.func_.input_rows();
        int orows = func.func_.output_rows();
        int numappl = func.indices_.size();

        MatrixXi vindex(irows, numappl);
        MatrixXi cindex(orows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices_[i].maxCoeff() > numVars) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Variable indices out of bounds in equality constraint");
                throw std::invalid_argument("");
            }
            vindex.col(i) = func.indices_[i];
            for (int j = 0; j < orows; j++) {
                cindex(j, i) = numEqCons;
                numEqCons++;
            }
        }

        this->nlp_->equality_constraints_.emplace_back(
            ConstraintFunction(func.func_, vindex, cindex));

        ThreadingFlags ThreadMode =
            func.func_.thread_safe()
                ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                : ThreadingFlags::MainThread;

        this->nlp_->equality_constraints_.back().thread_mode_ = ThreadMode;
    }

    for (auto &func : this->user_inequalities_) {
        int irows = func.func_.input_rows();
        int orows = func.func_.output_rows();
        int numappl = func.indices_.size();

        MatrixXi vindex(irows, numappl);
        MatrixXi cindex(orows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices_[i].maxCoeff() > numVars) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Variable indices out of bounds in inequality constraint");
                throw std::invalid_argument("");
            }
            vindex.col(i) = func.indices_[i];
            for (int j = 0; j < orows; j++) {
                cindex(j, i) = numIqCons;
                numIqCons++;
            }
        }

        this->nlp_->inequality_constraints_.emplace_back(
            ConstraintFunction(func.func_, vindex, cindex));

        ThreadingFlags ThreadMode =
            func.func_.thread_safe()
                ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                : ThreadingFlags::MainThread;

        this->nlp_->inequality_constraints_.back().thread_mode_ = ThreadMode;
    }

    for (auto &func : this->user_objectives_) {
        int irows = func.func_.input_rows();
        int numappl = func.indices_.size();

        MatrixXi vindex(irows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices_[i].maxCoeff() > numVars) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "Variable indices out of bounds in inequality constraint");
                throw std::invalid_argument("");
            }
            vindex.col(i) = func.indices_[i];
        }

        this->nlp_->objectives_.emplace_back(ObjectiveFunction(func.func_, vindex));

        ThreadingFlags ThreadMode =
            func.func_.thread_safe()
                ? (numappl > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin)
                : ThreadingFlags::MainThread;

        this->nlp_->objectives_.back().thread_mode_ = ThreadMode;
    }

    this->nlp_->make_nlp(numVars, numEqCons, numIqCons);
    this->optimizer_->set_nlp(this->nlp_);

    //////DO NOT GET RID OF THIS!!!!!!//
    this->do_transcription_ = false;
    ////////////////////////////////////
}
