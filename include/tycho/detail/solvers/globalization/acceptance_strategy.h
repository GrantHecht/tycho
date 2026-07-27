// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: this is the acceptance-
// strategy component (line search & merit), whose interface shape is adapted
// from Uno (source-verified).
//
// This file: the interface plus a handful of small defaulted hooks —
// notify_switch_to_feasibility()/notify_switch_to_optimality()/
// append_diagnostics() (no-op defaults) and classic_line_search() (a
// throwing default, not real behavior). The generic
// is_iterate_acceptable()/is_infeasibility_sufficiently_reduced() surface is
// implemented for real by ModernMeritAcceptance, FunnelAcceptance, and
// FilterAcceptance; ClassicMeritAcceptance (a separate merit_acceptance.h)
// stubs those two methods with "unused on classic path" bodies and
// implements classic_line_search verbatim from the former
// ls_impl/ls_lang/ls_l1/ls_auglang.
//
// Ownership rule: an AcceptanceStrategy instance holds no SOLVER-owned state
// (no XSL/DXSL/mu/iterate history members — those are always passed as
// explicit per-call parameters, or reached through a SolverContext reference
// passed to the call: settings_, dims, nlp_). Strategy-internal state is a
// different matter and is explicitly permitted: ModernMeritAcceptance holds
// per-solve penalty state (nu_/pi_l_/pi_u_/smallest_known_infeasibility_,
// modern_merit.h), and FilterAcceptance holds its filter plus reset-heuristic
// counters (filter_acceptance.h) — both cleared in reset(), never cached
// beyond it. reset() is the μ-event/phase-change hook: called whenever
// PSIOPT starts a new phase (run_phase_sequence) or the barrier parameter is
// reset, so a stateful strategy (e.g. the filter clearing its (θ,f) pairs)
// has a defined place to do it. The ClassicMeritAcceptance implementation of
// reset() is a no-op (the classic merit test carries no persistent state
// across iterations).

#pragma once

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/iterate_info.h"
// AcceptanceStrategy::classic_line_search takes PSIOPT::LineSearchModes by
// value, which requires the complete PSIOPT class (a nested enum cannot be
// forward-declared independently of its enclosing class). psiopt.h does NOT
// include this directory back (see solver_context.h's include-discipline
// note) so this is a plain, one-directional, non-circular include — exactly
// like include/tycho/detail/solvers/optimization_problem_base.h already does.
#include "tycho/detail/solvers/psiopt.h"

namespace tycho::solvers {

// =============================================================================
// Restoration-exit constant shared by the two acceptance strategies whose exit
// test uses the Ipopt-style relative θ-reduction floor: ClassicMeritAcceptance
// (merit_acceptance.h) and FilterAcceptance (filter_acceptance.h). It lives on
// the shared base header because both strategies compile into the same
// translation unit and a single definition is required; ModernMeritAcceptance
// and FunnelAcceptance use their own source-specific exit criteria and never
// read it.
// =============================================================================

// Required infeasibility reduction to leave feasibility restoration: a trial's
// constraint violation must fall to kKappaResto·θ_ref before the point is
// eligible to exit (subject to the constraint-tolerance floor the two consumers
// apply on top). Ipopt option "required_infeasibility_reduction", shipped
// default 0.9 (coin-or/Ipopt 72a29c9, src/Algorithm/IpRestoConvCheck.cpp,
// kappa_resto_).
inline constexpr double kKappaResto = 0.9;

// =============================================================================
// AcceptanceStrategy — decides whether a trial step is accepted.
// =============================================================================
class AcceptanceStrategy {
  public:
    virtual ~AcceptanceStrategy() = default;

    // --- Generic interface (future strategies implement these for real) ---
    // Filter/funnel/WMNO acceptance tests: is the trial point's (θ, f)
    // pair acceptable relative to the current iterate, given what the step
    // model predicted? ClassicMeritAcceptance stubs this with a documented
    // "unused on classic path" body (today's classic acceptance is entirely
    // inside classic_line_search's fused loop+test); ModernMeritAcceptance,
    // FunnelAcceptance, and FilterAcceptance all implement it for real and are
    // selectable via Settings::acceptance_strategy_. The generic surface is
    // driven whenever any non-classic_merit strategy is selected.
    //
    // step_length is the trial's alpha, in (0, 1] — the live ladder value
    // from the generic backtracking loop that produced `trial`.
    // predicted_reduction stays alpha-scaled (as documented on
    // ProgressMeasures/modern_merit.h); a strategy that needs the raw,
    // unscaled directional derivative recovers it as
    // predicted_reduction.objective / step_length.
    virtual bool is_iterate_acceptable(const ProgressMeasures &current,
                                        const ProgressMeasures &trial,
                                        const ProgressMeasures &predicted_reduction,
                                        double objective_multiplier, double step_length) = 0;

    // Restoration-exit test: has infeasibility been reduced enough (relative
    // to `reference`, the point restoration was entered from) to leave
    // restoration mode? Driven by alg_impl once a feasibility-restoration
    // strategy is active (the near-feasible and κ_resto-ratchet exit
    // branches, psiopt.cpp).
    virtual bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &reference,
                                                        const ProgressMeasures &trial) const = 0;

    // μ-event / phase-change reset hook — see the ownership-rule note above.
    virtual void reset() = 0;

    // Selects which driving path GlobalizationMechanism::compute_step uses.
    //   true  — the FUSED classic path: compute_step forwards straight to
    //           classic_line_search (below), whose own backtracking loop hosts
    //           the merit test. This is the bit-identical classic behavior;
    //           only ClassicMeritAcceptance returns it.
    //   false — the GENERIC path: compute_step runs the loop itself (trial
    //           eval -> ProgressMeasures -> is_iterate_acceptable -> backtrack)
    //           and never calls classic_line_search. ModernMeritAcceptance
    //           overrides this to false — it is the loop-in-mechanism /
    //           judgment-in-strategy split the generic surface was designed for.
    // Pure virtual on purpose: every strategy must declare its driving path at
    // compile time. A defaulted answer here would let a new strategy silently
    // inherit the classic path and hit classic_line_search's throwing default
    // at solve time instead of failing to compile.
    virtual bool drives_classic_path() const = 0;

    // Mode-switch notifications (restoration handoff), called by the solver
    // seam when it enters / leaves feasibility restoration. `current` is the
    // ProgressMeasures at the switch point (the entry point on the way in, the
    // exit point on the way out). Default no-op: the classic and modern merit
    // strategies keep it (their acceptance state is not invalidated by the
    // objective swap — matching Uno's MeritFunction, which defines no switch
    // hooks). FunnelAcceptance overrides only the optimality (exit) hook to
    // re-base its width; FilterAcceptance overrides both to augment/stash/restore
    // its filter (see funnel_acceptance.h / filter_acceptance.h).
    virtual void notify_switch_to_feasibility(const ProgressMeasures &) {}
    virtual void notify_switch_to_optimality(const ProgressMeasures &) {}

    // Solver-level observability hook: writes this strategy's diagnostic
    // state (if any) into `result`. Called by run_phase_sequence() once per
    // phase, right after that phase's alg_impl() returns and before the NEXT
    // phase's reset() (see the call site's comment in psiopt.cpp) — so a
    // multi-phase solve (e.g. solve_optimize()) ends up with the LAST
    // phase's values, overwritten in phase order like every other SolveResult
    // field. WRITE-ONLY on purpose: this hook never reads `result` or any
    // other solver state, so it cannot influence control flow — a strategy
    // that changed behavior based on prior diagnostics would need a real
    // feedback path, not this one. The default body is a no-op, which is
    // exactly right for ClassicMeritAcceptance and ModernMeritAcceptance
    // (neither has funnel/filter-style state to report): the classic path
    // stays bit-identical because this hook never touches `result` unless a
    // strategy overrides it. FunnelAcceptance and FilterAcceptance override
    // this to report their width/size (+ reset count for the filter) — see
    // funnel_acceptance.h / filter_acceptance.h and the corresponding
    // SolveResult fields in psiopt.h.
    virtual void append_diagnostics(PSIOPT::SolveResult &result) const { (void)result; }

    // --- Classic fused entry point ---
    // Signature mirrors the former private PSIOPT::ls_impl dispatcher exactly
    // (the symbol no longer exists — its body was extracted into
    // ClassicMeritAcceptance; see the note beside PSIOPT::alg_impl in
    // psiopt.h) — NOT the private per-variant
    // ls_lang/ls_l1/ls_auglang signatures, which take PSIOPT::KKTVector
    // views. KKTVector is a private nested class of PSIOPT (psiopt.h)
    // and is not name-accessible from a non-member, non-friend type such as
    // this one; ls_impl's own public-facing signature already operates on
    // the raw Eigen::VectorXd blocks for exactly this reason, so mirroring
    // IT (rather than the KKTVector-typed private helpers) is what lets this
    // interface host the existing calls without adapting any FP-relevant
    // argument. A future implementation reconstructs KKTVector-equivalent
    // segment views internally from SolverContext's dims if/when it needs
    // the named-segment accessors ls_lang/ls_l1/ls_auglang use today.
    //
    // Loop + merit test fused together (not split into separate "step" and
    // "accept" calls) because today's ls_lang/ls_l1/ls_auglang each run their
    // own backtracking loop with the merit test as the loop's own exit
    // condition — splitting them would require re-deriving the per-variant
    // trial-point evaluation (eval_rhs for LANG vs. eval_trial_point_occ for
    // L1/AUGLANG) at a new seam, which risks reordering the FP operations the
    // CBWR gate depends on. Returns the accepted step-length alpha.
    //
    // NOT pure: only the ClassicMeritAcceptance implementation (defined in
    // merit_acceptance.h) overrides this. Generic future acceptance
    // strategies are driven purely through is_iterate_acceptable() and never
    // call this entry point, so the default body is a T6-style logic error,
    // not a silent fallback.
    virtual double classic_line_search(PSIOPT::LineSearchModes lsmode, double obj_scale, double mu,
                                        double prim_obj, double barr_obj, Eigen::VectorXd &XSL,
                                        Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                                        Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2,
                                        IterateInfo &Citer, const std::vector<IterateInfo> &iters) {
        (void)lsmode;
        (void)obj_scale;
        (void)mu;
        (void)prim_obj;
        (void)barr_obj;
        (void)XSL;
        (void)DXSL;
        (void)XSL2;
        (void)RHS;
        (void)RHS2;
        (void)Citer;
        (void)iters;
        throw std::logic_error(
            "classic_line_search is only implemented by ClassicMeritAcceptance");
    }
};

} // namespace tycho::solvers
