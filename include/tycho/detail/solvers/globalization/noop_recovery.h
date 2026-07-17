// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: RecoveryChain provides an
// ordered dispatch on step rejection (second-order correction -> extended
// backtracking -> watchdog revert -> feasibility switch); this file supplies
// the empty-chain implementation, matching today's give-up behavior.
//
// NoopRecovery implements the RecoveryChain::on_step_rejected() hook as a
// pure no-op: it unconditionally returns Action::kAcceptAsIs and never
// inspects/mutates Citer, iters, or ctx. This is the CLASSIC behavior today's
// alg_impl already exhibits without any recovery chain at all — whatever alpha
// survived BacktrackingLineSearch's capped backtrack (compute_step) is simply
// taken as-is; there is no SOC retry, no extended backtrack, no watchdog
// revert, and no feasibility-switch handoff. Wiring this hook is therefore
// purely structural: the call site in alg_impl (src/solvers/psiopt.cpp) is
// provably inert (see the comment block at that call site) — NoopRecovery
// cannot produce kRetry / kSwitchToFeasibility / kGiveUp, so no dispatch
// branch downstream of the call is reachable, and the CBWR gate (bit-identical
// iteration counts on the 31-problem corpus) is expected to hold trivially.
//
// A future change replaces this class (not this call site) with a real SOC ->
// extended-backtrack -> watchdog -> feasibility-switch dispatcher. The
// inertia/perturbation ladder (factor_impl's Zfac cycling + 8x/(1/3)
// escalation) is a SEPARATE mechanism that stays inside PSIOPT::factor_impl
// and is NOT part of this chain until the proximal-regularization inertia
// mode is implemented.
//
// Ownership: stateless, per RecoveryChain's ownership rule — reset() has
// nothing to clear.

#pragma once

#include "tycho/detail/solvers/globalization/recovery_chain.h"

namespace tycho::solvers {

// =============================================================================
// NoopRecovery — the empty recovery chain: always accepts the step as-is.
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
