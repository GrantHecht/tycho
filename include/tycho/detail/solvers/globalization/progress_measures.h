// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the E2 G1 globalization extraction. Spec:
// docs/superpowers/specs/2026-07-16-e2-psiopt-globalization-design.md §3
// ("Component architecture (G1)"). Ground-truth recon for what these types
// eventually host: docs/superpowers/plans/2026-07-16-e2-g1-dossier.md.
//
// G1 (this file): pure-data value type, no behavior. Filled in by later G1
// tasks (the classic acceptance/globalization/barrier components read and
// write ProgressMeasures instances but never own one across calls) and
// consumed as-is by G2+ (filter/funnel acceptance strategies, §3 of the spec).

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
// acceptance_strategy.h. The design spec (§3) describes a conceptually
// separate "predicted-reduction struct"; Task 1 follows the brief's own code
// skeleton, which types predicted_reduction as `const ProgressMeasures&`
// rather than a distinct type — introducing a separate type is deferred until
// a concrete G2+ strategy actually needs fields ProgressMeasures does not
// carry.
//
// infeasibility (θ) and objective (f, σ-scaled) are the classic two-term pair
// consumed by merit/filter/funnel acceptance tests. auxiliary carries
// barrier/proximal terms (e.g. the current μ-barrier objective contribution)
// OUTSIDE the (θ, f) pair on purpose: folding a barrier term into the merit
// objective would contaminate the filter/funnel machinery G3 introduces —
// see spec §3, "the hook that makes filter/funnel work inside an IPM (§7.1)".
// G1's classic merit acceptance does not need this separation (it sums
// prim_obj_ + barr_obj_ directly, see dossier §2 ls_lang/ls_l1/ls_auglang);
// the slot exists so G1's data shape does not need to change again in G3.
// =============================================================================
struct ProgressMeasures {
    double infeasibility = 0.0; ///< θ — constraint violation measure.
    double objective = 0.0;     ///< σ-scaled objective measure (f, not the raw objective).
    double auxiliary = 0.0;     ///< Barrier/proximal terms; never folded into (f, θ).
};

} // namespace tycho::solvers
