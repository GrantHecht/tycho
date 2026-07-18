// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Second batch of live RecoveryChain links: extended backtracking, the
// watchdog, and the ChainedRecovery composition that ties them to SOC
// (soc.h). All three are opt-in and default OFF; with both
// Settings::ls_extended_iters_ == 0 and Settings::watchdog_ == false (and
// Settings::max_soc_ == 0), PSIOPT::rebuild_globalization_components()
// installs a plain NoopRecovery and the solver is bit-identical to its
// pre-recovery-chain behavior — see the wiring comment in psiopt.cpp's
// rebuild_globalization_components().
//
// -----------------------------------------------------------------------------
// Extended backtracking (ExtendedBacktrackRecovery)
// -----------------------------------------------------------------------------
// When the classic capped backtrack rejects a step (and SOC, if enabled, is
// either not triggered or exhausted), extended backtracking continues the
// SAME ladder — same direction (DXSL, untouched), same alpha_red_ divisor,
// same merit acceptance test — for up to Settings::ls_extended_iters_ further
// external trials.
//
// No new math: each external trial is realized as a single call to the SAME
// acceptance.classic_line_search() entry point the classic backtrack itself
// uses, on a SCALED COPY of DXSL. This works out exactly because of how the
// classic ls_lang/ls_l1/ls_auglang loop is structured (psiopt_globalization.
// cpp): each starts its OWN internal alpha at 1.0 and, on exhaustion (every
// one of ctx.settings_.max_ls_iters_ internal trials rejected), returns the
// alpha value ONE MORE division past the last-tested point — i.e. the
// returned alpha is always the next untested rung of the ladder, never a
// value already tried. So:
//   - Seed `scale` with the alpha value live at hook time (compute_step's
//     return value, forwarded here as the `alpha` parameter) — this is
//     exactly that untested next rung. Do NOT restart at 1.0.
//   - Scale a local copy: dxsl_ext = scale * DXSL. Calling
//     classic_line_search on dxsl_ext makes ITS internal alpha=1.0 trial
//     land exactly on `scale` (relative to the original DXSL), and every
//     further internal division it performs lands on the next rung down —
//     zero redundant re-testing, same alpha_red_ divisor, same test.
//   - If that call's internal loop exhausts without accepting, its returned
//     alpha (relative to dxsl_ext) times the `scale` used for that call is
//     the next untested rung — carry it forward as the next external trial's
//     `scale`.
// Each external trial therefore may itself run more than one internal ls
// sub-trial when ctx.settings_.max_ls_iters_ > 1 (the SAME call-counts-as-
// one-attempt convention SocRecovery already uses for its correction calls —
// see soc.h's do_correction, which also re-runs classic_line_search's full
// internal loop per correction). Settings::ls_extended_iters_ bounds the
// number of EXTERNAL calls, not the raw count of alpha divisions tested.
//
// -----------------------------------------------------------------------------
// Watchdog (WatchdogRecovery / WatchdogState)
// -----------------------------------------------------------------------------
// The Chamberlain, Powell, Lemaréchal & Pedersen (1982) watchdog technique
// ("The watchdog technique for forcing convergence in algorithms for
// constrained optimization", Mathematical Programming Study 16, 1-17), with
// the arming/trial-window constants taken from the reference interior-point
// implementation in Wächter & Biegler, "On the implementation of an
// interior-point filter line-search algorithm for large-scale nonlinear
// programming", Math. Program. 106(1):25-57, 2006: arm after
// kWatchdogShortenedIterTrigger consecutive shortened iterations, then allow
// up to kWatchdogTrialIterMax trial iterations under relaxed acceptance
// before reverting.
//
// Architectural scope note (read before touching arming semantics): the ONLY
// REJECTION-observing hook this recovery-chain layer has into the solve loop
// is on_step_rejected, invoked exactly when the classic backtrack FULLY
// rejects a step (every one of max_ls_iters_ trials failed the merit test) —
// see should_dispatch_recovery in recovery_chain.h. An iteration that
// backtracks to some alpha < 1 but still finds an acceptable trial
// (Citer.accepted_ == true at some j > 0) never reaches this hook at all.
// Consequently "shortened iteration" for WatchdogState's purposes means "a
// full rejection was dispatched here", not the broader "any alpha < 1
// iteration" a fully integrated watchdog (observing every solve iteration)
// would use. This is a CONSERVATIVE narrowing: it can only arm later (or not
// at all) relative to full paper semantics, counting only the most severe
// form of shortening. A genuinely ACCEPTED iteration in between two
// rejections is observed through the separate notify_step_accepted() hook
// (RecoveryChain), which WatchdogRecovery uses to reset
// consecutive_shortened_ on real progress — without that hook the counter
// would keep summing non-consecutive rejections straight across an accepted
// iteration and mis-arm. Known observation gap: an iteration with a
// non-finite step (the divergence path) reaches NEITHER hook — it is
// invisible to the counter, which neither accumulates nor resets there. A
// future change that gives alg_impl a per-iteration (not just per-rejection)
// hook could widen the rejection side of this further and close that gap.
//
// WatchdogState is a pure, machinery-free state machine (mirrors soc.h's
// soc_should_trigger/soc_should_continue/soc_run_loop split of policy from
// plumbing) so its arm/trial/revert/reset transitions are unit-testable with
// scripted (mu, merit) sequences, independent of Eigen/solver types.
// WatchdogRecovery is the RecoveryChain decorator that drives it against the
// real working set: it snapshots XSL (the pre-watchdog iterate) at the
// moment WatchdogState arms, and uses `merit` = prim_obj + barr_obj — the
// same leading term every classic merit variant's own LangInit/ptest+btest
// value is built from (see ls_lang/ls_l1/ls_auglang in psiopt_globalization.
// cpp) — as an always-available proxy for "did the point improve", since
// AcceptanceStrategy exposes no generic current-merit accessor on the
// classic path (is_iterate_acceptable throws there — see
// acceptance_strategy.h).
//
// Semantics, precisely:
//   - Not armed: each rejected iteration increments a consecutive-shortened
//     counter (reset to 0 if the barrier parameter mu changed since the
//     previous call — a mu change invalidates the count, since it belongs to
//     a different barrier subproblem — and ALSO reset to 0 by
//     notify_step_accepted() on any genuinely accepted iteration observed in
//     between, since that iteration broke the "consecutive" run). On reaching
//     kWatchdogShortenedIterTrigger, WatchdogState arms: it records the
//     CURRENT mu and merit as the snapshot reference (WatchdogRecovery also
//     copies XSL), and this SAME call becomes trial #1 of the window — it is
//     granted relaxed acceptance (accept regardless of merit), exactly like
//     every subsequent trial in the window.
//   - Armed, within the window: each further rejected iteration is trial
//     #2, #3, ... . If mu has changed since the snapshot, the WHOLE state
//     resets (unarmed, counters cleared) — a reset, not a revert (see the
//     class doc on WatchdogState::record_rejected_iteration) — and this call
//     is re-processed as the first iteration of a fresh count. Otherwise: if
//     this iterate's merit beats (is strictly less than) the snapshot's
//     merit reference, disarm and hand the rejection back to the wrapped
//     chain for NORMAL treatment (the emergency is over). If the trial
//     count reaches kWatchdogTrialIterMax without that improvement, REVERT
//     XSL to the snapshot (DXSL zeroed, alpha = 0, so alg_impl's `XSL +=
//     alpha*DXSL` commit is a no-op) and disarm.
//
// Ownership: WatchdogRecovery holds real per-solve state (WatchdogState plus
// the XSL snapshot vector) — the one RecoveryChain link that isn't stateless
// — behind reset(), per the ownership rule in recovery_chain.h. reset() also
// propagates to the wrapped inner chain.

#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"

namespace tycho::solvers {

// Chamberlain, Powell, Lemaréchal & Pedersen (1982); constants per the
// reference implementation in Wächter & Biegler (2006) — see the file
// docstring above.
inline constexpr int kWatchdogShortenedIterTrigger = 10;
inline constexpr int kWatchdogTrialIterMax = 3;

// =============================================================================
// ExtendedBacktrackRecovery — opt-in continuation of the classic backtracking
// ladder past its capped rejection. See the file docstring's "Extended
// backtracking" section for the exact mechanics.
//
// Stateless (per RecoveryChain's ownership rule): the scaled-direction
// scratch is local to on_step_rejected.
// =============================================================================
class ExtendedBacktrackRecovery : public RecoveryChain {
  public:
    ExtendedBacktrackRecovery() = default;

    Action on_step_rejected(IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                            SolverContext &ctx, AcceptanceStrategy &acceptance,
                            GlobalizationMechanism &mechanism, PSIOPT::LineSearchModes lsmode,
                            double obj_scale, double mu, double prim_obj, double barr_obj,
                            Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                            Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2, double &alpha,
                            double &alphap, double &alphad, int &soc_steps, int &resolved_depth,
                            int &watchdog_activations) override;

    // Stateless: nothing to clear.
    void reset() override {}
};

// =============================================================================
// WatchdogState — pure arm/trial/revert/reset state machine (Chamberlain,
// Powell, Lemaréchal & Pedersen 1982; constants per Wächter & Biegler 2006).
// See the file docstring's "Watchdog" section for full semantics and the
// architectural scope note on what "shortened iteration" means here.
// =============================================================================
class WatchdogState {
  public:
    enum class Outcome {
        kAccumulate,    // not armed: no override — delegate to the wrapped chain.
        kArmed,         // just reached the arm threshold THIS call (== trial #1):
                        // override with a relaxed accept; caller should snapshot now.
        kTrialRelax,    // armed, still within the window, no progress yet:
                        // override with a relaxed accept.
        kTrialProgress, // armed, this iterate beat the snapshot's merit reference:
                        // disarm — delegate to the wrapped chain.
        kTrialRevert,   // armed, window exhausted with no progress: revert to
                        // the snapshot and disarm.
    };

    void reset() {
        consecutive_shortened_ = 0;
        armed_ = false;
        trial_count_ = 0;
        have_last_mu_ = false;
        last_mu_ = 0.0;
        snapshot_mu_ = 0.0;
        snapshot_merit_ = 0.0;
    }

    bool armed() const { return armed_; }
    int consecutive_shortened() const { return consecutive_shortened_; }
    int trial_count() const { return trial_count_; }

    // Feed one more fully-rejected iteration into the state machine. `mu` is
    // the current barrier parameter; `merit` is a scalar quality measure for
    // the CURRENT iterate (WatchdogRecovery passes prim_obj + barr_obj — see
    // the file docstring for why that is a sound always-available proxy).
    Outcome record_rejected_iteration(double mu, double merit) {
        if (armed_) {
            if (mu != snapshot_mu_) {
                // Barrier parameter moved mid-watchdog: the snapshot's merit
                // reference is no longer comparable to `merit`, so the whole
                // state resets (NOT a revert — see the class doc above) and
                // this call is re-processed as the start of a fresh count.
                reset();
                return accumulate(mu, merit);
            }
            ++trial_count_;
            if (merit < snapshot_merit_) {
                reset();
                return Outcome::kTrialProgress;
            }
            if (trial_count_ >= kWatchdogTrialIterMax) {
                reset();
                return Outcome::kTrialRevert;
            }
            return Outcome::kTrialRelax;
        }
        return accumulate(mu, merit);
    }

    // Reset the consecutive-shortened counter on a genuinely accepted
    // iteration (see notify_step_accepted() on RecoveryChain, which
    // WatchdogRecovery threads into this call): a real accept breaks the
    // "consecutive" run that arming depends on, so counting toward the NEXT
    // arm attempt must start over from this point. Does not touch
    // armed_/trial_count_/the snapshot -- if a trial window is already open,
    // that window's own rejected-iteration bookkeeping
    // (record_rejected_iteration) is unaffected by this call.
    void record_accepted_iteration() { consecutive_shortened_ = 0; }

  private:
    Outcome accumulate(double mu, double merit) {
        if (have_last_mu_ && mu != last_mu_)
            consecutive_shortened_ = 0;
        last_mu_ = mu;
        have_last_mu_ = true;
        ++consecutive_shortened_;
        if (consecutive_shortened_ >= kWatchdogShortenedIterTrigger) {
            armed_ = true;
            snapshot_mu_ = mu;
            snapshot_merit_ = merit;
            trial_count_ = 1; // this call is trial #1 of the window.
            return Outcome::kArmed;
        }
        return Outcome::kAccumulate;
    }

    int consecutive_shortened_ = 0;
    bool armed_ = false;
    int trial_count_ = 0;
    bool have_last_mu_ = false;
    double last_mu_ = 0.0;
    double snapshot_mu_ = 0.0;
    double snapshot_merit_ = 0.0;
};

// =============================================================================
// WatchdogRecovery — RecoveryChain decorator driving WatchdogState against
// the real working set. Wraps whatever chain is configured underneath it
// (ChainedRecovery, a single link, or NoopRecovery) as an OUTER decorator:
// while WatchdogState reports kAccumulate/kTrialProgress, the rejection is
// forwarded unchanged to the wrapped chain; while armed and within the trial
// window (kArmed/kTrialRelax), it is instead overridden with a relaxed
// accept; on kTrialRevert, XSL is restored to the pre-watchdog snapshot.
//
// `inner` must be non-null -- every construction site
// (PSIOPT::rebuild_globalization_components()) always supplies a real chain
// (NoopRecovery at minimum), so the constructor enforces
// this invariant once, up front, letting on_step_rejected/notify_step_accepted/
// reset dereference inner_ unconditionally rather than guarding a state that
// can never actually occur.
//
// Ownership: holds real per-solve state (see the file docstring's Ownership
// note) behind reset().
// =============================================================================
class WatchdogRecovery : public RecoveryChain {
  public:
    explicit WatchdogRecovery(std::unique_ptr<RecoveryChain> inner) : inner_(std::move(inner)) {
        if (!inner_)
            throw std::invalid_argument("WatchdogRecovery: inner recovery chain must not be null");
    }

    Action on_step_rejected(IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                            SolverContext &ctx, AcceptanceStrategy &acceptance,
                            GlobalizationMechanism &mechanism, PSIOPT::LineSearchModes lsmode,
                            double obj_scale, double mu, double prim_obj, double barr_obj,
                            Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                            Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2, double &alpha,
                            double &alphap, double &alphad, int &soc_steps, int &resolved_depth,
                            int &watchdog_activations) override;

    // Resets consecutive_shortened_ on real progress (see the file
    // docstring's Architectural scope note), then threads the notification
    // through to inner_ so a nested link with its own counter (none exist
    // yet) would see it too -- the same threading on_step_rejected uses.
    void notify_step_accepted() override {
        state_.record_accepted_iteration();
        inner_->notify_step_accepted();
    }

    void reset() override {
        state_.reset();
        snapshot_xsl_.resize(0);
        inner_->reset();
    }

  private:
    std::unique_ptr<RecoveryChain> inner_;
    WatchdogState state_;
    Eigen::VectorXd snapshot_xsl_;
};

// =============================================================================
// ChainedRecovery — composes SOC and extended backtracking in a fixed order:
// SOC is tried before extended backtracking. Rationale: SOC re-solves the
// rejected direction on the live factorization (one extra back-substitution,
// informed by the actual KKT system), while extended backtracking is a pure
// re-test of the ALREADY-rejected direction at smaller alpha (no solve at
// all) — the more info-bearing recovery gets first refusal, and only if it
// declines (not triggered, or exhausted) does the cheaper ladder-continuation
// get a turn. Either pointer may be null (that link disabled by Settings);
// on_step_rejected skips null links.
//
// Ownership: stateless itself (holds only the two link pointers); reset()
// propagates to whichever links are present.
// =============================================================================
class ChainedRecovery : public RecoveryChain {
  public:
    ChainedRecovery(std::unique_ptr<RecoveryChain> soc, std::unique_ptr<RecoveryChain> extended)
        : soc_(std::move(soc)), extended_(std::move(extended)) {}

    Action on_step_rejected(IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                            SolverContext &ctx, AcceptanceStrategy &acceptance,
                            GlobalizationMechanism &mechanism, PSIOPT::LineSearchModes lsmode,
                            double obj_scale, double mu, double prim_obj, double barr_obj,
                            Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                            Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2, double &alpha,
                            double &alphap, double &alphad, int &soc_steps, int &resolved_depth,
                            int &watchdog_activations) override;

    // Propagates to whichever link(s) are present -- same null-guard pattern
    // as reset() below, since soc_/extended_ (unlike WatchdogRecovery::inner_)
    // are legitimately null when that link is disabled by Settings.
    void notify_step_accepted() override {
        if (soc_)
            soc_->notify_step_accepted();
        if (extended_)
            extended_->notify_step_accepted();
    }

    void reset() override {
        if (soc_)
            soc_->reset();
        if (extended_)
            extended_->reset();
    }

  private:
    std::unique_ptr<RecoveryChain> soc_;
    std::unique_ptr<RecoveryChain> extended_;
};

} // namespace tycho::solvers
