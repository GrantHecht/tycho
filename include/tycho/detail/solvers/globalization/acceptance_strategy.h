// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: this is the acceptance-
// strategy component (line search & merit), whose interface shape is adapted
// from Uno (source-verified).
//
// This file: pure interface declaration, no implementation anywhere in this
// header. `classic_line_search` is the one exception with a body (see
// below) — a throwing default, not real behavior. The generic
// is_iterate_acceptable()/is_infeasibility_sufficiently_reduced() surface
// exists for future filter/funnel/WMNO strategies; ClassicMeritAcceptance (a
// separate merit_acceptance.h) stubs those two methods with "unused on
// classic path" bodies and implements classic_line_search verbatim from
// today's ls_impl/ls_lang/ls_l1/ls_auglang.
//
// Ownership rule: an AcceptanceStrategy instance holds NO solver state of its
// own (no XSL/DXSL/mu/iterate history members). Every quantity it needs is
// either passed as an explicit per-call parameter (the per-iteration
// transients: obj_scale, mu, prim_obj, barr_obj, the working vectors) or
// reached through a SolverContext reference passed to the call (settings_,
// dims, nlp_) — never cached across calls. reset() is the μ-event/phase-
// change hook: called whenever PSIOPT starts a new phase (run_phase_sequence)
// or the barrier parameter is reset, so a stateful future acceptance strategy
// (e.g. a filter that must clear its (θ,f) pairs) has a defined place to do
// it. The ClassicMeritAcceptance implementation of reset() is a no-op (the
// classic merit test carries no persistent state across iterations today).

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
    // inside classic_line_search's fused loop+test); it is not driven from
    // anywhere until a filter/funnel/WMNO strategy is selected.
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
    // restoration mode? Unused until a feasibility-restoration strategy lands
    // that calls it.
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

    // Mode-switch notifications (restoration handoff); default no-op so the
    // classic path and any strategy that doesn't care about the switch need
    // not override them.
    virtual void notify_switch_to_feasibility(const ProgressMeasures &) {}
    virtual void notify_switch_to_optimality(const ProgressMeasures &) {}

    // --- Classic fused entry point ---
    // Signature mirrors today's private PSIOPT::ls_impl dispatcher exactly
    // (psiopt.h:530-533) — NOT the private per-variant
    // ls_lang/ls_l1/ls_auglang signatures, which take PSIOPT::KKTVector
    // views. KKTVector is a private nested class of PSIOPT (psiopt.h:448)
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
