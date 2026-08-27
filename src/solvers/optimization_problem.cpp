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
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#include "tycho/detail/solvers_vf/optimization_problem.h"

#include "tycho/detail/solvers_vf/transcribed_aggregate.h"
#include "tycho/detail/solvers_vf/transcription_declaration.h"

using tycho::solvers::ConstraintFunction;
using tycho::solvers::ObjectiveFunction;
using tycho::solvers::TranscriptionDeclaration;

namespace {

/// The thread assignment policy a user-supplied function is partitioned under.
/// A function that is not thread safe stays on the calling thread; one that is
/// goes by application where it has several, and round-robin where it has one.
tycho::solvers::ThreadingFlags transcription_thread_mode(bool thread_safe, int applications) {
    using tycho::solvers::ThreadingFlags;
    if (!thread_safe) {
        return ThreadingFlags::MainThread;
    }
    return applications > 1 ? ThreadingFlags::ByApplication : ThreadingFlags::RoundRobin;
}

} // namespace

void tycho::solvers::OptimizationProblem::transcribe() {
    // Built into locals and committed together at the end. A transcription that
    // refuses part-way -- a bad index, a bound history that intersects to
    // nothing -- then leaves the problem on the program it already had, with its
    // published provider still describing that same program, rather than on a
    // half-laid new one.
    auto nlp = std::make_shared<NonLinearProgram>(this->num_partitions_);

    int numVars = this->active_variables_.size();

    if (numVars == 0) {
        throw std::invalid_argument("No variables provided to OptimizationProblem");
    }

    TranscriptionDeclaration declaration(this->num_partitions_);

    int numEqCons = 0;
    int numIqCons = 0;

    for (auto &func : this->user_equalities_) {
        int irows = func.func_.input_rows();
        int orows = func.func_.output_rows();
        int numappl = int(func.indices_.size());

        MatrixXi vindex(irows, numappl);
        MatrixXi cindex(orows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices_[i].minCoeff() < 0 || func.indices_[i].maxCoeff() >= numVars) {
                throw std::invalid_argument(
                    "Variable indices out of bounds in equality constraint");
            }
            vindex.col(i) = func.indices_[i];
            for (int j = 0; j < orows; j++) {
                cindex(j, i) = numEqCons;
                numEqCons++;
            }
        }

        declaration.add_equality(ConstraintFunction(func.func_, vindex, cindex),
                                 transcription_thread_mode(func.func_.thread_safe(), numappl));
    }

    for (auto &func : this->user_inequalities_) {
        int irows = func.func_.input_rows();
        int orows = func.func_.output_rows();
        int numappl = int(func.indices_.size());

        MatrixXi vindex(irows, numappl);
        MatrixXi cindex(orows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices_[i].minCoeff() < 0 || func.indices_[i].maxCoeff() >= numVars) {
                throw std::invalid_argument(
                    "Variable indices out of bounds in inequality constraint");
            }
            vindex.col(i) = func.indices_[i];
            for (int j = 0; j < orows; j++) {
                cindex(j, i) = numIqCons;
                numIqCons++;
            }
        }

        declaration.add_inequality(ConstraintFunction(func.func_, vindex, cindex),
                                   transcription_thread_mode(func.func_.thread_safe(), numappl));
    }

    for (auto &func : this->user_objectives_) {
        int irows = func.func_.input_rows();
        int numappl = int(func.indices_.size());

        MatrixXi vindex(irows, numappl);

        for (int i = 0; i < numappl; i++) {
            if (func.indices_[i].minCoeff() < 0 || func.indices_[i].maxCoeff() >= numVars) {
                throw std::invalid_argument("Variable indices out of bounds in objective");
            }
            vindex.col(i) = func.indices_[i];
        }

        declaration.add_objective(ObjectiveFunction(func.func_, vindex),
                                  transcription_thread_mode(func.func_.thread_safe(), numappl));
    }

    for (const auto &bound : this->user_var_bounds_) {
        declaration.set_variable_bound(bound.index_, bound.lower_, bound.upper_);
    }

    declaration.lay(*nlp, numVars, numEqCons, numIqCons);
    auto provider = std::make_shared<TranscribedAggregate>(nlp);

    this->nlp_ = std::move(nlp);
    this->provider_ = std::move(provider);

    //////DO NOT GET RID OF THIS!!!!!!//
    this->do_transcription_ = false;
    ////////////////////////////////////
}
