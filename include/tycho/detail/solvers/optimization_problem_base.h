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
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include "tycho/detail/solvers/ipopt_backend.h"
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

    int num_partitions_ = 1;
    JetJobModes jet_job_mode_ = JetJobModes::NotSet;

    std::shared_ptr<NonLinearProgram> nlp_;
    std::shared_ptr<PSIOPT> optimizer_;

    /// NLP solver backend for the solve/optimize entry points. psiopt (default)
    /// is the built-in path, byte-identical to previous behavior. ipopt runs
    /// the identical transcribed NLP through a linked Ipopt installation
    /// (requires a build with Ipopt support; throws std::runtime_error
    /// otherwise).
    NLPSolvers nlp_solver_ = NLPSolvers::psiopt;

    /// String key/value options forwarded verbatim to Ipopt (e.g.
    /// {"linear_solver", "pardisomkl"}). Applied after the matched-tolerance
    /// baseline, so user entries win. Ignored by the psiopt backend.
    std::map<std::string, std::string> ipopt_options_;

    /// Diagnostics of the most recent ipopt-backend run on this problem.
    IpoptRunInfo last_ipopt_result_;

    virtual ~OptimizationProblemBase() = default;

    OptimizationProblemBase() {
        this->optimizer_ = std::make_shared<PSIOPT>();
        this->init_partitions();
    }

    virtual PSIOPT::ConvergenceFlags solve() = 0;
    virtual PSIOPT::ConvergenceFlags optimize() = 0;
    virtual PSIOPT::ConvergenceFlags solve_optimize() = 0;
    virtual PSIOPT::ConvergenceFlags solve_optimize_solve() = 0;
    virtual PSIOPT::ConvergenceFlags optimize_solve() = 0;

    /// Compute default partition count from the global thread budget.
    /// Over-partitions by 4x so the work-stealing pool can smooth out
    /// unequal partition costs. make_nlp() further caps this via
    /// MIN_NNZ_PER_PARTITION on small problems.
    static int default_num_partitions() {
        int nt = tycho::utils::get_num_threads();
        if (nt <= 1)
            return 1;
        return nt * 4;
    }

    virtual void init_partitions() {
        this->num_partitions_ = default_num_partitions();
        this->optimizer_->set_qp_threads(
            std::min(TYCHO_DEFAULT_QP_THREADS, utils::get_core_count()));
    }

    virtual void set_num_partitions(int num_partitions, int qp_threads) {
        if (num_partitions < 1) {
            throw std::invalid_argument("Number of partitions must be positive");
        }
        this->optimizer_->set_qp_threads(qp_threads); // may throw — do before mutating
        this->num_partitions_ = num_partitions;
    }
    virtual void set_num_partitions(int num_partitions) {
        if (num_partitions < 1) {
            throw std::invalid_argument("Number of partitions must be positive");
        }
        this->num_partitions_ = num_partitions;
    }

    virtual void jet_initialize() = 0;
    virtual void jet_release() = 0;

    // IMPORTANT: jet_run() is called from Jet::map() on pool worker threads.
    // If jet_initialize() did NOT set num_partitions_=1, the NLP eval methods
    // would call parallel_sequence/parallel_task from a pool worker, triggering
    // the nested-dispatch guard (std::logic_error). jet_initialize() MUST set
    // num_partitions_=1 so NLP eval methods run inline.
    virtual PSIOPT::ConvergenceFlags jet_run() {
        this->jet_initialize();

        PSIOPT::ConvergenceFlags flag;

        switch (this->jet_job_mode_) {
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
            throw ::std::invalid_argument("jet_job_mode_ not set");
        }
        default:
            throw std::invalid_argument("Unrecognized jet_job_mode");
        }

        this->jet_release();
        return flag;
    }

    /// Uniform output of one backend solve: the updated variable vector, the
    /// constraint multipliers, and the convergence flag, regardless of backend.
    struct NlpSolveOutput {
        Eigen::VectorXd variables_;
        Eigen::VectorXd eq_lmults_;
        Eigen::VectorXd iq_lmults_;
        ConvergenceFlags flag_ = ConvergenceFlags::NOTCONVERGED;
    };

    /// Single backend dispatch point for the five solve modes. The psiopt
    /// branch reproduces the historic call and result reads exactly; the
    /// ipopt branch always runs a single NLP solve from the given input (the
    /// feasibility-then-optimize staging has no Ipopt analog).
    ///
    /// Defined out of line below, after the ipopt_backend::solve declaration
    /// it calls (a member function body defined in the class cannot name a
    /// namespace-scope function declared later in the file).
    NlpSolveOutput run_nlp_solver(JetJobModes mode, const Eigen::VectorXd &input);

    static JetJobModes strto_jet_job_mode(const std::string &str) {

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
            auto msg = fmt::format("Unrecognized jet_job_mode: {0}\n", str);
            throw std::invalid_argument(msg);
        }
    }

    void set_jet_job_mode(JetJobModes m) { this->jet_job_mode_ = m; }
    void set_jet_job_mode(const std::string &str) {
        this->set_jet_job_mode(strto_jet_job_mode(str));
    }
};

namespace ipopt_backend {
/// Run Ipopt on the problem's transcribed NLP. A real implementation is linked
/// only in builds configured with Ipopt support; the stub throws
/// std::runtime_error.
OptimizationProblemBase::NlpSolveOutput solve(OptimizationProblemBase &prob,
                                              OptimizationProblemBase::JetJobModes mode,
                                              const Eigen::VectorXd &input);
} // namespace ipopt_backend

inline OptimizationProblemBase::NlpSolveOutput
OptimizationProblemBase::run_nlp_solver(JetJobModes mode, const Eigen::VectorXd &input) {
    if (this->nlp_solver_ == NLPSolvers::ipopt) {
        return ipopt_backend::solve(*this, mode, input);
    }
    NlpSolveOutput out;
    switch (mode) {
    case JetJobModes::Solve:
        out.variables_ = this->optimizer_->solve(input);
        break;
    case JetJobModes::Optimize:
        out.variables_ = this->optimizer_->optimize(input);
        break;
    case JetJobModes::SolveOptimize:
        out.variables_ = this->optimizer_->solve_optimize(input);
        break;
    case JetJobModes::SolveOptimizeSolve:
        out.variables_ = this->optimizer_->solve_optimize_solve(input);
        break;
    case JetJobModes::OptimizeSolve:
        out.variables_ = this->optimizer_->optimize_solve(input);
        break;
    default:
        throw std::invalid_argument("Unrecognized NLP solve mode");
    }
    out.eq_lmults_ = this->optimizer_->result().eq_lmults_;
    out.iq_lmults_ = this->optimizer_->result().iq_lmults_;
    out.flag_ = this->optimizer_->result().converge_flag_;
    return out;
}

} // namespace tycho::solvers
