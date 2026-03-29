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

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/psiopt.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/thread_pool.h"

namespace tycho::solvers {

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

    int NumPartitions = 1;
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

    /// Compute default partition count from the global thread budget.
    /// Over-partitions by 4x so the work-stealing pool can smooth out
    /// unequal partition costs. make_NLP() further caps this via
    /// MIN_NNZ_PER_PARTITION on small problems.
    static int default_num_partitions() {
        int nt = tycho::utils::get_num_threads();
        if (nt <= 1)
            return 1;
        return nt * 4;
    }

    virtual void initPartitions() {
        this->NumPartitions = default_num_partitions();
        this->optimizer->QPThreads = std::min(TYCHO_DEFAULT_QP_THREADS, utils::get_core_count());
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

    // IMPORTANT: jet_run() is called from Jet::map() on pool worker threads.
    // If jet_initialize() did NOT set NumPartitions=1, the NLP eval methods
    // would call parallel_sequence/parallel_task from a pool worker, triggering
    // the nested-dispatch guard (std::logic_error). jet_initialize() MUST set
    // NumPartitions=1 so NLP eval methods run inline.
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

} // namespace tycho::solvers
