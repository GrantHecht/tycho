// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/solvers/nlp_solver.h"

#include <cmath>

#include <fmt/format.h>

namespace tycho::solvers {

NLPSolver::NLPSolver(std::shared_ptr<NLPProblem> problem) : problem_(std::move(problem)) {
    if (!this->problem_) {
        throw std::invalid_argument("NLPSolver: the problem pointer is null");
    }
}

void NLPSolver::transcribe() {
    this->core_ = std::make_shared<NLPAdapterCore>(this->problem_);
    this->nlp_ = make_nlp_program(this->core_);
    this->optimizer_->set_nlp(this->nlp_);
    this->do_transcription_ = false;
}

tycho::ConvergenceFlags NLPSolver::run(JetJobModes mode, ConstEigenRef<Eigen::VectorXd> x0) {
    if (this->do_transcription_) {
        this->transcribe();
    }
    if (x0.size() != this->core_->n_) {
        throw std::invalid_argument(
            fmt::format("{}: the initial guess has {} elements but the problem has {} variables",
                        this->problem_->name(), x0.size(), this->core_->n_));
    }
    this->apply_starting_multipliers();
    auto out = this->run_nlp_solver(mode, Eigen::VectorXd(x0));
    this->active_variables_ = out.variables_;
    this->active_eq_lmults_ = out.eq_lmults_;
    this->active_iq_lmults_ = out.iq_lmults_;
    return out.flag_;
}

tycho::ConvergenceFlags NLPSolver::solve(ConstEigenRef<Eigen::VectorXd> x0) {
    return this->run(JetJobModes::Solve, x0);
}
tycho::ConvergenceFlags NLPSolver::optimize(ConstEigenRef<Eigen::VectorXd> x0) {
    return this->run(JetJobModes::Optimize, x0);
}
tycho::ConvergenceFlags NLPSolver::solve_optimize(ConstEigenRef<Eigen::VectorXd> x0) {
    return this->run(JetJobModes::SolveOptimize, x0);
}
tycho::ConvergenceFlags NLPSolver::optimize_solve(ConstEigenRef<Eigen::VectorXd> x0) {
    return this->run(JetJobModes::OptimizeSolve, x0);
}
tycho::ConvergenceFlags NLPSolver::solve_optimize_solve(ConstEigenRef<Eigen::VectorXd> x0) {
    return this->run(JetJobModes::SolveOptimizeSolve, x0);
}

// OptimizationProblemBase's no-arg entry points, mirroring OptimizationProblem:
// each reuses whatever is currently in active_variables_ as the input iterate.
tycho::ConvergenceFlags NLPSolver::solve() {
    return this->run(JetJobModes::Solve, this->active_variables_);
}
tycho::ConvergenceFlags NLPSolver::optimize() {
    return this->run(JetJobModes::Optimize, this->active_variables_);
}
tycho::ConvergenceFlags NLPSolver::solve_optimize() {
    return this->run(JetJobModes::SolveOptimize, this->active_variables_);
}
tycho::ConvergenceFlags NLPSolver::solve_optimize_solve() {
    return this->run(JetJobModes::SolveOptimizeSolve, this->active_variables_);
}
tycho::ConvergenceFlags NLPSolver::optimize_solve() {
    return this->run(JetJobModes::OptimizeSolve, this->active_variables_);
}

void NLPSolver::jet_initialize() {
    this->set_num_partitions(1, 1);
    this->optimizer_->set_print_level(10);
    this->transcribe();
}

void NLPSolver::jet_release() {
    this->optimizer_->release();
    this->set_num_partitions(1, 1);
    this->optimizer_->set_print_level(0);
    this->nlp_ = std::shared_ptr<NonLinearProgram>();
    this->do_transcription_ = true;
}

Eigen::VectorXd NLPSolver::return_multipliers() const {
    if (!this->core_) {
        throw std::runtime_error("NLPSolver::return_multipliers: nothing has been solved yet");
    }
    // compose_user_lambda tolerates empty vectors (all-zero result), so this is
    // safe even if a solve never ran or the problem has no constraints.
    const_cast<NLPAdapterCore &>(*this->core_)
        .compose_user_lambda(this->active_eq_lmults_, this->active_iq_lmults_);
    return this->core_->lambda_user_;
}

void NLPSolver::apply_starting_multipliers() {
    Eigen::VectorXd lam = Eigen::VectorXd::Zero(this->core_->m_);
    if (!this->problem_->starting_multipliers(lam)) {
        return;
    }
    if (!lam.allFinite()) {
        throw std::invalid_argument(fmt::format(
            "{}: starting_multipliers returned a non-finite value", this->problem_->name()));
    }
    const auto &rc = this->core_->rows_;
    Eigen::VectorXd eqm = Eigen::VectorXd::Zero(rc.num_eq_);
    Eigen::VectorXd iqm = Eigen::VectorXd::Zero(rc.num_iq_);
    for (int r = 0; r < this->core_->m_; r++) {
        switch (rc.kinds_[r]) {
        case NLPRowKind::Equality:
            eqm[rc.eq_row_[r]] = lam[r];
            break;
        case NLPRowKind::UpperBounded:
            iqm[rc.iq_upper_row_[r]] = lam[r];
            break;
        case NLPRowKind::LowerBounded:
            iqm[rc.iq_lower_row_[r]] = -lam[r];
            break;
        case NLPRowKind::Range:
            iqm[rc.iq_upper_row_[r]] = std::max(lam[r], 0.0);
            iqm[rc.iq_lower_row_[r]] = std::max(-lam[r], 0.0);
            break;
        case NLPRowKind::Free:
            break;
        }
    }
    // NOTE: the mapping above is fully computed but not yet applied. PSIOPT
    // does not expose a seeding entry point for the reduced multiplier space
    // yet (set_initial_multipliers arrives in the next change in this
    // series); until then, a problem that asks to seed starting multipliers
    // gets a clear failure instead of a silently ignored request.
    throw std::invalid_argument(fmt::format(
        "{}: starting_multipliers seeding requires the PSIOPT seeding entry (next change in "
        "this series)",
        this->problem_->name()));
}

} // namespace tycho::solvers
