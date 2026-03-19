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
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once
#include "Solvers/NonLinearProgram.h"
#include "Solvers/PSIOPT.h"
#include "pch.h"

namespace Tycho {

struct OptimizationProblemBase {

    enum class JetJobModes {
        NotSet,
        DoNothing,
        Solve,
        Optimize,
        SolveOptimize,
        SolveOptimizeSolve,
        OptimizeSolve
    };

    int NumPartitions = TYCHO_DEFAULT_NUM_PARTITIONS;
    JetJobModes JetJobMode = JetJobModes::NotSet;

    std::shared_ptr<NonLinearProgram> nlp;
    std::shared_ptr<PSIOPT> optimizer;

    virtual ~OptimizationProblemBase() = default;

    OptimizationProblemBase() {
        this->optimizer = std::make_shared<PSIOPT>();
        this->initPartitions();
    }

    virtual PSIOPT::ConvergenceFlags solve() = 0;
    virtual PSIOPT::ConvergenceFlags optimize() = 0;
    virtual PSIOPT::ConvergenceFlags solve_optimize() = 0;
    virtual PSIOPT::ConvergenceFlags solve_optimize_solve() = 0;
    virtual PSIOPT::ConvergenceFlags optimize_solve() = 0;

    virtual void initPartitions() {
        this->NumPartitions =
            std::min(TYCHO_DEFAULT_NUM_PARTITIONS, int(std::thread::hardware_concurrency()));
        this->optimizer->QPThreads = std::min(TYCHO_DEFAULT_QP_THREADS, get_core_count());
    }

    virtual void setNumPartitions(int num_partitions, int qp_threads) {
        if (num_partitions < 1 || qp_threads < 1) {
            throw std::invalid_argument("Number of partitions/threads must be positive");
        }
        this->NumPartitions = num_partitions;
        this->optimizer->QPThreads = qp_threads;
    }
    virtual void setNumPartitions(int num_partitions) {
        if (num_partitions < 1) {
            throw std::invalid_argument("Number of partitions must be positive");
        }
        this->NumPartitions = num_partitions;
    }

    virtual void jet_initialize() = 0;
    virtual void jet_release() = 0;

    // IMPORTANT: jet_run() is called from Jet::map() which dispatches jobs to
    // the global thread pool. To prevent deadlock (pool workers submitting sub-
    // tasks to the same pool and blocking), jet_initialize() MUST set this
    // problem to single-threaded mode (NumPartitions=1). This ensures NLP eval
    // methods run inline rather than dispatching to the pool.
    virtual PSIOPT::ConvergenceFlags jet_run() {
        this->jet_initialize();

        PSIOPT::ConvergenceFlags flag;

        switch (this->JetJobMode) {
        case JetJobModes::Solve: {
            flag = this->solve();
            break;
        }
        case JetJobModes::Optimize: {
            flag = this->optimize();
            break;
        }
        case JetJobModes::SolveOptimize: {
            flag = this->solve_optimize();
            break;
        }
        case JetJobModes::SolveOptimizeSolve: {
            flag = this->solve_optimize_solve();
            break;
        }
        case JetJobModes::OptimizeSolve: {
            flag = this->optimize_solve();
            break;
        }
        case JetJobModes::NotSet: {
            throw ::std::invalid_argument("JetJobMode not set");
        }
        default:
            throw std::invalid_argument("Unrecognized JetJobMode");
        }

        this->jet_release();
        return flag;
    }

    static JetJobModes strto_JetJobMode(const std::string &str) {

        if (str == "solve" || str == "Solve")
            return JetJobModes::Solve;
        else if (str == "optimize" || str == "Optimize")
            return JetJobModes::Optimize;
        else if (str == "solve_optimize" || str == "SolveOptimize" || str == "Solve_Optimize")
            return JetJobModes::SolveOptimize;
        else if (str == "solve_optimize_solve" || str == "SolveOptimizeSolve" ||
                 str == "Solve_Optimize_Solve")
            return JetJobModes::SolveOptimizeSolve;
        else if (str == "optimize_solve" || str == "OptimizeSolve" || str == "Optimize_Solve")
            return JetJobModes::OptimizeSolve;
        else if (str == "DoNothing" || str == "do_nothing" || str == "Do_Nothing")
            return JetJobModes::DoNothing;
        else {
            auto msg = fmt::format("Unrecognized JetJobMode: {0}\n", str);
            throw std::invalid_argument(msg);
        }
    }

    void setJetJobMode(JetJobModes m) { this->JetJobMode = m; }
    void setJetJobMode(const std::string &str) { this->setJetJobMode(strto_JetJobMode(str)); }
};

} // namespace Tycho
