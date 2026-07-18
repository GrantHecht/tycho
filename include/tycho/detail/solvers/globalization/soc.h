// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// SocRecovery — the first live RecoveryChain link: an opt-in second-order
// correction (SOC) applied after the line search rejects a step on its very
// first trial. The mechanics follow Wächter & Biegler, "On the implementation
// of an interior-point filter line-search algorithm for large-scale nonlinear
// programming", Math. Program. 106(1):25-57, 2006, §2.4 (second-order
// corrections):
//
//   * Trigger only when the first trial step was the one rejected AND that
//     trial did not reduce the constraint violation relative to the current
//     iterate. When the merit variant records no infeasibility reading (LANG
//     never does), SOC conservatively does not trigger.
//   * A correction re-solves the SAME KKT system on the LIVE factorization (no
//     refactor — one back-substitution) with the constraint block of the
//     right-hand side replaced by an accumulated corrected value; the objective
//     block is unchanged.
//   * The corrected direction is fraction-to-boundary scaled and the full
//     acceptance backtrack is re-run on it. If it is accepted, the corrected
//     step replaces the rejected one (RecoveryChain::Action::kRetry).
//   * Otherwise corrections repeat while the corrected trial's violation keeps
//     dropping by at least the kSocViolationDecrease factor, up to
//     Settings::max_soc_ corrections; once the violation stagnates or the cap
//     is hit, SOC gives up and the originally-rejected step is taken
//     (RecoveryChain::Action::kAcceptAsIs — bit-identical to the no-SOC path).
//
// Default is off (max_soc_ == 0): SocRecovery is not even constructed then
// (set_nlp installs a NoopRecovery), so the solver is bit-identical to its
// pre-SOC behavior. The correctness-critical numeric path is exercised
// end-to-end by the solver corpus; the unit tests here truth-table the trigger,
// the termination policy, and the correction counter with scripted outcomes.
//
// Ownership: stateless, per RecoveryChain's ownership rule — the correction
// loop's transient buffers are local to on_step_rejected, and reset() has
// nothing to clear.

#pragma once

#include <vector>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"

namespace tycho::solvers {

// κ_soc — the required per-correction reduction factor in the constraint
// violation (Wächter & Biegler 2006, §2.4). A correction is worth continuing
// only while it keeps cutting the violation by at least this factor.
inline constexpr double kSocViolationDecrease = 0.99;

// Recommended value to enable SOC with (the Ipopt default: up to four
// corrections per rejected step, Wächter & Biegler 2006, §2.4). Settings'
// own default stays 0 (off); this names the value to opt in with.
inline constexpr int kSocRecommendedMaxCorrections = 4;

// -----------------------------------------------------------------------------
// Trigger predicate (Wächter & Biegler 2006, §2.4).
//
// Fire a second-order correction only when the FIRST trial step (backtracking
// index 0) was the one rejected AND that trial did not reduce the constraint
// violation relative to the current iterate. A negative theta means "no
// infeasibility reading available" — the LANG merit variant records none, and
// -1 is IterateInfo's unset sentinel — in which case the reduction test cannot
// be made and SOC conservatively does not trigger.
//
// `current_infeasibility` must be the SAME quantity theta_at_first_rejection_
// records: the squared L2 norm of the full constraint block (all_cons) under
// the merit line search's eval-plus-slack-reset convention. alg_impl derives it
// from RHS.all_cons() at the hook, whose inequality block carries the identical
// slack reset (see the RHS assembly in psiopt.cpp).
//
// Known asymmetry: the augmented-Lagrangian merit variant zeroes constraint
// entries within tolerance when it records theta_at_first_rejection_, while
// current_infeasibility is not tolerance-zeroed. The zeroing can only SHRINK
// the trial-side theta, making the `theta >= current` trigger HARDER to
// satisfy — i.e. the asymmetry suppresses SOC near feasibility (where it has
// nothing to gain) and can never spuriously fire it.
// -----------------------------------------------------------------------------
inline bool soc_should_trigger(const IterateInfo &citer, double current_infeasibility) {
    if (citer.first_rejection_iter_ != 0)
        return false;
    const double trial_infeasibility = citer.theta_at_first_rejection_;
    if (trial_infeasibility < 0.0)
        return false;
    return trial_infeasibility >= current_infeasibility;
}

// -----------------------------------------------------------------------------
// Termination predicate (Wächter & Biegler 2006, §2.4).
//
// After a rejected correction, keep correcting only while (a) the correction
// cap has not been reached and (b) the corrected trial's violation is still
// dropping by at least the kSocViolationDecrease factor versus the previous
// violation. Once the drop stagnates (slower than the factor) or the cap is
// hit, SOC gives up.
// -----------------------------------------------------------------------------
inline bool soc_should_continue(double trial_violation, double prev_violation,
                                int corrections_done, int max_soc) {
    if (corrections_done >= max_soc)
        return false;
    return trial_violation <= kSocViolationDecrease * prev_violation;
}

// Outcome of a single correction attempt, returned by the correction primitive
// to the loop driver.
struct SocCorrectionOutcome {
    // The corrected step was accepted by the re-run acceptance test; the
    // corrected direction/length have been committed in place.
    bool accepted;
    // The corrected first-trial constraint violation (squared L2). Only
    // meaningful when !accepted; drives the next accumulation and the
    // termination test.
    double trial_violation;
};

// -----------------------------------------------------------------------------
// Correction loop driver — pure policy, no KKT/nlp machinery. Runs the
// correction primitive `do_correction(correction_index, prev_violation)` until
// a correction is accepted (returns kRetry) or the termination policy stops it
// (returns kAcceptAsIs). Increments `soc_steps` once per attempted correction
// (each is one back-substitution). Templated on the primitive so the unit tests
// drive it with a scripted lambda in place of the real solve.
//
// The first correction is always attempted once the trigger has fired (matching
// the paper); subsequent corrections are gated by soc_should_continue.
// -----------------------------------------------------------------------------
template <class CorrectionFn>
RecoveryChain::Action soc_run_loop(double first_trial_violation, int max_soc, int &soc_steps,
                                   CorrectionFn &&do_correction) {
    double prev_violation = first_trial_violation;
    int corrections = 0;
    while (true) {
        const SocCorrectionOutcome outcome = do_correction(corrections, prev_violation);
        ++corrections;
        ++soc_steps;
        if (outcome.accepted)
            return RecoveryChain::Action::kRetry;
        if (!soc_should_continue(outcome.trial_violation, prev_violation, corrections, max_soc))
            return RecoveryChain::Action::kAcceptAsIs;
        prev_violation = outcome.trial_violation;
    }
}

// =============================================================================
// SocRecovery — opt-in second-order correction recovery link.
// =============================================================================
class SocRecovery : public RecoveryChain {
  public:
    SocRecovery() = default;

    Action on_step_rejected(IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                            SolverContext &ctx, AcceptanceStrategy &acceptance,
                            GlobalizationMechanism &mechanism, PSIOPT::LineSearchModes lsmode,
                            double obj_scale, double mu, double prim_obj, double barr_obj,
                            Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                            Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2, double &alpha,
                            double &alphap, double &alphad, int &soc_steps, int &resolved_depth,
                            int &watchdog_activations) override;

    // Stateless (per RecoveryChain's ownership rule): nothing to clear.
    void reset() override {}

  private:
    // Evaluate the constraint block c(x_k + alpha*dir) into `cons_out` (size
    // equal_cons_ + inequal_cons_, layout [eq | iq]) using the same eval_occ +
    // slack-reset convention the merit line search and alg_impl's RHS assembly
    // use, so the value is directly comparable to RHS.all_cons(). Trial
    // primals/slacks are written into `xsl2_scratch`.
    void eval_trial_constraints(SolverContext &ctx, double obj_scale, const Eigen::VectorXd &XSL,
                                const Eigen::VectorXd &dir, double alpha,
                                Eigen::VectorXd &xsl2_scratch, Eigen::VectorXd &cons_out);
};

} // namespace tycho::solvers
