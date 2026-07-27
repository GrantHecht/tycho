// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: this is the barrier-update
// component.
//
// ClassicAdaptiveGovernor implements BarrierGovernor::update_barrier by hosting
// today's PROBE/LOQO barrier-parameter block (psiopt.cpp barmode switch: the
// PROBE Mehrotra predictor-corrector branch, the LOQO branch, and the common
// clamp + barrier_objective + corrector barrier_gradient tail) plus the loqo_mu
// / mpc_mu oracles, moved VERBATIM from src/solvers/psiopt.cpp (statement order
// and operand order preserved exactly — the merge gate is a bit-identical CBWR
// iteration-count comparison). The only edits are context-plumbing renames:
// former PSIOPT member reads (kkt_sol_ -> ctx.kkt_solver_, settings_/dims ->
// ctx.*) and the mechanism_ base pointer -> the mechanism reference parameter.
// Definitions live in src/solvers/psiopt_globalization.cpp.
//
// PROBE-impurity design note:
//   PROBE's mu update is NOT a pure function of (mu_in, avgcomp, mincomp). It
//   runs a predictor KKT solve into DXSL (DXSL = -ctx.kkt_solver_.solve(RHS)),
//   scales that predictor DXSL to the fraction-to-boundary via the
//   GlobalizationMechanism (mechanism.max_primal_dual_step — the SAME
//   entry point the main-path step uses; see globalization_mechanism.h), forms
//   the predictor point Temp = XSL + DXSL, and computes
//   mpc_mu(Temp.slacks(), Temp.iq_lmults(), ...). All of that predictor state
//   is consumed inside this method: DXSL is overwritten by the REAL step solve
//   that alg_impl runs immediately AFTER update_barrier returns (that real
//   solve does NOT move — only the predictor solve moves here), and Temp is
//   solver scratch the line search re-initialises. The only quantities that
//   escape are the returned (clamped) mu, the barr_obj out-param, and the
//   corrector dual gradient written into RHS's dual_grad() block by the common
//   tail (which the real solve then consumes).
//
//   Divergence-path note (mirrors backtracking_line_search.h): the
//   predictor max_primal_dual_step writes its alphap/alphad into locals that
//   are discarded (the BarrierGovernor interface surfaces no alpha out-params).
//   Pre-extraction the predictor wrote alg_impl's shared alphap/alphad; on the
//   classic GoodStep path the main-path compute_step overwrites those before
//   they are recorded, so this is bit-identical there. On a non-finite (!GoodStep)
//   PROBE step, the old code could record predictor-derived alphap/alphad on the
//   terminal DIVERGING iterate where the new code records the 1.0 loop-init
//   values. Print-only exposure on diverging runs (Citer.alpha_p_/alpha_d_ are
//   IterateInfo display fields that converge_check never reads; DXSL is
//   discarded before the state commit on that path). Gate-bit-identical:
//   iterates, mu, and iteration counts are unaffected.
//
// Byte-identity design note (references-only channel):
//   Like ClassicMeritAcceptance (merit_acceptance.h) and BacktrackingLineSearch
//   (backtracking_line_search.h), this component reaches PSIOPT state ONLY
//   through SolverContext, and builds a KKTVector view over the raw
//   XSL/RHS/Temp blocks from the context's dimensions. The view type is the
//   shared tycho::solvers::KKTVector (detail/solvers/kkt_vector.h) — this
//   header used to carry a verbatim copy of it, because the type was private to
//   PSIOPT and not name-accessible from this non-member, non-friend type. Of the
//   barrier oracles this governor needs, barrier_objective and the four-argument
//   barrier_gradient are now one-line forwarders into
//   detail/solvers/barrier_math.h; the two-argument (PROBE corrector)
//   barrier_gradient has no second copy and stays here. complementarity is
//   deliberately NOT shared: it is a TOKEN-IDENTICAL copy of
//   PSIOPT::complementarity INCLUDING its ULP warning, because its .sum()
//   reduction feeds mu (via mpc_mu) and the reduction order must NOT change;
//   unifying it needs its own evidence.
//
// Ownership rule: ClassicAdaptiveGovernor holds NO solver state (matches the
// BarrierGovernor ownership rule — mu is passed in and returned, never cached).
// Every quantity it needs is either an explicit per-call parameter or reached
// through the SolverContext reference passed to the call. reset() is a no-op
// (this component has no monotone-mode bookkeeping to clear; MonitoredBarrierGovernor,
// the shipped free<->monotone governor, has a real reset() body for exactly
// that state — see monitored_governor.h).

#pragma once

#include <cassert>
#include <utility>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/barrier_governor.h"
#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/psiopt.h"

namespace tycho::solvers {

// =============================================================================
// ClassicAdaptiveGovernor — the classic PROBE/LOQO barrier-parameter update.
//
// Stateless (holds NO solver state, per BarrierGovernor's ownership rule).
// Constructed by PSIOPT::rebuild_globalization_components() at the start of
// every solve invocation; every call receives the live SolverContext view of
// the solver and the GlobalizationMechanism as explicit parameters. Always
// reports in_monotone_mode() == false (free-mode only).
// =============================================================================
class ClassicAdaptiveGovernor : public BarrierGovernor {
  public:
    ClassicAdaptiveGovernor() = default;

    // Verbatim today's psiopt.cpp barmode switch + common clamp/objective/
    // gradient tail — see the PROBE-impurity design note above. Called under
    // alg_impl's `if (inequal_cons_ > 0)` guard (the guard stays at the call
    // site, exactly as the block was guarded before extraction).
    //
    // Free-mode only: `current` is ignored (the PROBE/LOQO oracles read no
    // iteration state) and `mu_event` is never written (this governor has no
    // monotone mode), so the caller's mu-event reset branch stays dead on the
    // classic path.
    double update_barrier(PSIOPT::BarrierModes barmode, double mu_in, double avgcomp,
                          double mincomp, Eigen::VectorXd &XSL, Eigen::VectorXd &RHS,
                          Eigen::VectorXd &DXSL, Eigen::VectorXd &Temp,
                          GlobalizationMechanism &mechanism, SolverContext &ctx, double &barr_obj,
                          const IterateInfo &current, bool &mu_event) override;

    // μ-event / phase-change hook — no-op: the classic PROBE/LOQO oracles carry
    // no persistent state across iterations (free mode; see barrier_governor.h).
    void reset() override {}

  private:
    // The compound-KKT segment view (tycho::solvers::KKTVector,
    // detail/solvers/kkt_vector.h) is shared with PSIOPT and the sibling
    // components, so the moved barrier block keeps its named-segment accessors
    // (v_xsl.slacks()/v_rhs.dual_grad()/v_temp.iq_lmults()/…) unchanged. Only
    // the factory below is per-component: the dimensions come from the
    // SolverContext.

    /// Create a KKTVector view over a VectorXd using the context's dimensions.
    static KKTVector kkt_view(Eigen::VectorXd &v, const SolverContext &ctx) {
        return KKTVector(v, ctx.primal_vars_, ctx.slack_vars_, ctx.equal_cons_, ctx.inequal_cons_);
    }

    // --- Barrier oracles: moved VERBATIM from PSIOPT (psiopt.cpp). ---
    // loqo_mu / mpc_mu move here as the BarrierGovernor's own members; the
    // barrier_* helpers are verbatim copies (PSIOPT keeps its own for now).

    // TOKEN-IDENTICAL copy of PSIOPT::complementarity INCLUDING the ULP warning:
    // the .sum() reduction order feeds mu (via mpc_mu) and must not be reordered.
    // Uses ctx.stli_scratch_ (the same PSIOPT-owned buffer) instead of a PSIOPT
    // member.
    void complementarity(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI,
                         double &avgcomp, double &mincomp, double &maxcomp,
                         const SolverContext &ctx) const;
    double barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu,
                             const SolverContext &ctx) const;
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu,
                          Eigen::Ref<Eigen::VectorXd> AGS) const;
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> LI, Eigen::Ref<Eigen::VectorXd> AGS) const;

    // The LOQO oracle is a pure function of the complementarity aggregates; it
    // took the slack/multiplier blocks too, but never read them.
    double loqo_mu(double avgcomp, double mincomp) const;
    // mpc_mu re-runs complementarity on the predictor point (ctx for the shared
    // stli_scratch_); the reduction feeds mu, hence the ULP note above.
    double mpc_mu(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                  double mincomp, const SolverContext &ctx) const;
};

} // namespace tycho::solvers
