// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the E2 G1 globalization extraction (Task 5). Spec:
// docs/superpowers/specs/2026-07-16-e2-psiopt-globalization-design.md §3
// ("recovery_chain.h — Ordered dispatch on step rejection ... G1: empty chain
// (today's give-up behavior)") and §4 ("G2 — ... Recovery chain"). Ground-truth
// recon: docs/superpowers/plans/2026-07-16-e2-g1-dossier.md §8 ("RecoveryChain"
// verdict).
//
// NoopRecovery implements the G1 RecoveryChain::on_step_rejected() hook as a
// pure no-op: it unconditionally returns Action::kAcceptAsIs and never
// inspects/mutates Citer, iters, or ctx. This is the CLASSIC behavior today's
// alg_impl already exhibits without any recovery chain at all — whatever alpha
// survived BacktrackingLineSearch's capped backtrack (compute_step) is simply
// taken as-is; there is no SOC retry, no extended backtrack, no watchdog
// revert, and no feasibility-switch handoff. Wiring this hook in G1 is
// therefore purely structural: the call site in alg_impl (src/solvers/
// psiopt.cpp) is provably inert (see the comment block at that call site) —
// NoopRecovery cannot produce kRetry / kSwitchToFeasibility / kGiveUp, so no
// dispatch branch downstream of the call is reachable, and the CBWR gate
// (bit-identical iteration counts on the 31-problem corpus) is expected to
// hold trivially.
//
// G2 replaces this class (not this call site) with the real SOC ->
// extended-backtrack -> watchdog -> feasibility-switch dispatcher (spec §4).
// The inertia/perturbation ladder (factor_impl's Zfac cycling + 8x/(1/3)
// escalation, dossier §4) is a SEPARATE mechanism that stays inside
// PSIOPT::factor_impl and is NOT part of this chain until G6 (inertia_mode).
//
// Ownership: stateless, per RecoveryChain's ownership rule — reset() has
// nothing to clear.

#pragma once

#include "tycho/detail/solvers/globalization/recovery_chain.h"

namespace tycho::solvers {

// =============================================================================
// NoopRecovery — G1's empty recovery chain: always accepts the step as-is.
// =============================================================================
class NoopRecovery : public RecoveryChain {
  public:
    NoopRecovery() = default;

    Action on_step_rejected(IterateInfo & /*Citer*/, const std::vector<IterateInfo> & /*iters*/,
                            SolverContext & /*ctx*/) override {
        return Action::kAcceptAsIs;
    }

    void reset() override {
        // No-op: NoopRecovery holds no state to reset.
    }
};

} // namespace tycho::solvers
