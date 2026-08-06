// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <memory>

#include <Eigen/Core>

#include "tycho/detail/solvers/nlp_adapter.h"
#include "tycho/detail/solvers/optimization_problem_base.h"
#include "tycho/solvers/nlp_problem.h"

namespace tycho::solvers {

/// Solves an NLPProblem with PSIOPT. Owns the optimizer and its settings
/// (via OptimizationProblemBase); transcription happens lazily on the first
/// solve call. The problem is evaluated single-partition on the calling
/// thread, which is what a subclass implemented in Python requires anyway.
struct NLPSolver : OptimizationProblemBase {
    std::shared_ptr<NLPProblem> problem_;
    std::shared_ptr<NLPAdapterCore> core_;

    Eigen::VectorXd active_variables_;
    Eigen::VectorXd active_eq_lmults_;
    Eigen::VectorXd active_iq_lmults_;
    bool do_transcription_ = true;

    explicit NLPSolver(std::shared_ptr<NLPProblem> problem);

    void transcribe();

    tycho::ConvergenceFlags solve(ConstEigenRef<Eigen::VectorXd> x0);
    tycho::ConvergenceFlags optimize(ConstEigenRef<Eigen::VectorXd> x0);
    tycho::ConvergenceFlags solve_optimize(ConstEigenRef<Eigen::VectorXd> x0);
    tycho::ConvergenceFlags optimize_solve(ConstEigenRef<Eigen::VectorXd> x0);
    tycho::ConvergenceFlags solve_optimize_solve(ConstEigenRef<Eigen::VectorXd> x0);

    /// Solution primal vector, in the problem's own variable space.
    Eigen::VectorXd return_x() const { return this->active_variables_; }

    /// Solution constraint multipliers in the problem's own row space, Ipopt
    /// sign convention (L = obj_factor*f + lambda^T g). Zero for rows the
    /// classification dropped.
    Eigen::VectorXd return_multipliers() const;

    // --- OptimizationProblemBase's Jet-batch surface. NLPSolver mirrors
    // OptimizationProblem's implementation of these (same pattern: the
    // no-arg overloads reuse whatever is currently in active_variables_,
    // e.g. from a prior solve or a batch driver's own setup). The x0-taking
    // overloads above remain the primary entry points for direct use. ---
    tycho::ConvergenceFlags solve() override;
    tycho::ConvergenceFlags optimize() override;
    tycho::ConvergenceFlags solve_optimize() override;
    tycho::ConvergenceFlags solve_optimize_solve() override;
    tycho::ConvergenceFlags optimize_solve() override;
    void jet_initialize() override;
    void jet_release() override;

  private:
    tycho::ConvergenceFlags run(JetJobModes mode, ConstEigenRef<Eigen::VectorXd> x0);
    void apply_starting_multipliers();
};

} // namespace tycho::solvers
