// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: PSIOPT's step-acceptance,
// step-length, and barrier-parameter logic is being pulled out of the
// monolithic solver into standalone components under this directory.
//
// This file: pure-data value type, no behavior. Read and written by the
// classic acceptance/globalization/barrier components (none of them owns an
// instance across calls) and consumed as-is by the shipped FunnelAcceptance
// and FilterAcceptance strategies (funnel_acceptance.h, filter_acceptance.h).

#pragma once

namespace tycho::solvers {

// =============================================================================
// ProgressMeasures — the (θ, f, aux) triple used by every AcceptanceStrategy.
//
// This is a plain value type: no ownership, no solver-state references. It is
// constructed fresh from the current iterate/trial point on every call and
// passed by const reference into AcceptanceStrategy/GlobalizationMechanism
// methods (see acceptance_strategy.h, globalization_mechanism.h). The SAME
// struct shape is reused for three distinct roles in the AcceptanceStrategy
// interface: the current iterate's measures, a trial point's measures, and
// the *predicted* reduction model (i.e. what a linear/quadratic model of the
// step predicts (θ, f, aux) will become) — see is_iterate_acceptable() in
// acceptance_strategy.h. A conceptually separate "predicted-reduction struct"
// would also be a reasonable design; this implementation instead types
// predicted_reduction as `const ProgressMeasures&` rather than a distinct
// type — introducing a separate type is deferred until a concrete future
// acceptance strategy actually needs fields ProgressMeasures does not carry.
//
// infeasibility (θ) and objective (f, σ-scaled) are the classic two-term pair
// consumed by merit/filter/funnel acceptance tests. auxiliary carries
// barrier/proximal terms (e.g. the current μ-barrier objective contribution)
// OUTSIDE the (θ, f) pair on purpose: folding a barrier term into the merit
// objective would contaminate the filter/funnel machinery — this is the hook
// that makes filter/funnel work inside an interior-point method, and it is
// already exercised by the shipped FunnelAcceptance/FilterAcceptance
// strategies (psiopt_globalization.cpp). The classic merit acceptance
// strategy does not need this separation (it sums prim_obj_ + barr_obj_
// directly — see ls_lang/ls_l1/ls_auglang in merit_acceptance.h); the slot
// exists so this data shape did not need to change when filter/funnel were
// added.
//
// θ is the KKT constraint block and nothing else — ‖c‖ over the equality and
// inequality rows, as built by the trial-point evaluators. Native primal
// variable bounds are not rows (they are condensed into the primal diagonal;
// see barrier_math.h), so a bound never enters θ no matter how tightly an
// iterate is pressed against one; their barrier contribution arrives in
// auxiliary alongside the slack barrier. That is a structural exclusion, not a
// guarded one, and it is what lets the filter, the funnel, the
// feasibility-stall detector and the restoration-entry gate read the same
// purified constraint-violation signal on a bounded problem as on an
// unbounded one.
// =============================================================================
struct ProgressMeasures {
    double infeasibility = 0.0; ///< θ — constraint violation measure.
    double objective = 0.0;     ///< σ-scaled objective measure (f, not the raw objective).
    double auxiliary = 0.0;     ///< Barrier/proximal terms; never folded into (f, θ).
};

} // namespace tycho::solvers
