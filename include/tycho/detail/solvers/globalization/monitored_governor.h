// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// MonitoredBarrierGovernor — a free<->monotone barrier-parameter governor. In
// FREE mode it delegates the barrier update to a composed BarrierGovernor (by
// default a ClassicAdaptiveGovernor running the PROBE/LOQO oracles unchanged).
// A KKT-error monitor watches the recent iteration history; when the free mode
// stops making sufficient progress the governor hands off to a MONOTONE
// (Fiacco–McCormick) mode that holds the barrier parameter fixed until the
// barrier subproblem converges, then decreases it. When monotone progress
// brings the KKT error back into the reference band the governor re-enters free
// mode. Opt-in: nothing constructs this type on the default solve path.
//
// =============================================================================
// FORMULATION — derived from the Ipopt sources (fetched + read, not from
// memory). Line numbers are pinned to Ipopt releases/3.14.19
// (2695946fa79d2e84f3034e065e788933a81466eb) — every AMU:/MMU:/QF: shorthand
// citation below, and every corresponding citation in
// MonitoredBarrierGovernor's implementation (psiopt_globalization.cpp), is
// against that tagged commit, not a moving `master`:
//   [AMU]  src/Algorithm/IpAdaptiveMuUpdate.cpp — the adaptive (free<->fixed)
//          mu strategy with the "kkt-error" globalization. This governor
//          reproduces its KKT_ERROR path.
//   [MMU]  src/Algorithm/IpMonotoneMuUpdate.cpp — the standalone Fiacco–McCormick
//          update; [AMU]'s fixed-mode branch shares the same arithmetic.
//   [QF]   src/Algorithm/IpQualityFunctionMuOracle.cpp — the option defaults for
//          the quality-function norm/centrality/balancing terms.
//
// (0) Constants (each cites the Ipopt option name + shipped default):
//       kAdaptiveMuKktErrorRedIters  = 4     [AMU] "adaptive_mu_kkterror_red_iters"
//                                            RegisterOptions default (AMU:89-96).
//       kAdaptiveMuKktErrorRedFact   = 0.9999 [AMU] "adaptive_mu_kkterror_red_fact"
//                                            RegisterOptions default (AMU:98-105).
//       kAdaptiveMuMonotoneInitFactor= 0.8   [AMU] "adaptive_mu_monotone_init_factor"
//                                            RegisterOptions default (AMU:133-140).
//       kBarrierKappaMu              = 0.2   [MMU] "mu_linear_decrease_factor"
//                                            RegisterOptions default (MMU:58-67).
//       kBarrierThetaMu              = 1.5   [MMU] "mu_superlinear_decrease_power"
//                                            RegisterOptions default (MMU:68-77).
//       kBarrierTolFactor            = 10.0  [MMU] "barrier_tol_factor"
//                                            RegisterOptions default (MMU:49-57).
//
//   NOTE — kAdaptiveMuKktErrorRedFact is 0.9999, the value shipped in the
//   Ipopt source (AMU:103). An earlier planning memo cited 0.9998; the source
//   is authoritative, so 0.9999 is used here.
//
// (1) Monitor error measure (the reference-list quantity), [AMU]
//     quality_function_pd_system() (AMU:629-745) under the default
//     adaptive_mu_kkt_norm_type = "2-norm-squared" (AMU:141-151):
//
//       kkt_error = ||dual_inf||₂²/n_dual + ||primal_inf||₂²/n_pri
//                 + ||compl||₂²/n_comp   (+ centrality + balancing)
//
//     centrality and balancing are zero under their defaults ("none",
//     [QF] quality_function_centrality / quality_function_balancing_term,
//     QF:81-99), so the default error is the sum of the three squared,
//     per-dimension-averaged part norms.
//
//     Tycho mapping (monitor_error()): the IterateInfo residual scalars that
//     converge_check() consumes are ∞-norm (max) reductions, not 2-norms, and
//     Tycho decomposes the constraint block into equality/inequality parts.
//     The closest faithful composition preserving the "sum of squared parts"
//     structure is
//
//       monitor_error = kkt_inf_² + econ_inf_² + icon_inf_² + barr_inf_².
//
//     Documented deviations from [AMU] (chosen because these are the only
//     residual scalars carried on IterateInfo, and using them keeps the monitor
//     consistent with Tycho's own convergence test):
//       - each part is an ∞-norm scalar (kkt_inf_ = ‖prim_grad‖∞, econ_inf_ =
//         ‖eq_cons‖∞, icon_inf_ = ‖iq_cons‖∞), not the ‖·‖₂ Ipopt squares;
//       - no per-dimension division (n_dual/n_pri/n_comp): these are THREE
//         separate per-term weights, not one global scalar, so omitting them
//         re-weights the dual/primal/complementarity blocks against each
//         other and CAN shift exactly when the free->monotone switch
//         triggers relative to [AMU]'s formula. This cannot affect
//         correctness: monitor_error() and every value it is compared
//         against (the reference window) are computed with the identical
//         formula here, and the monotone-mode fallback this monitor gates is
//         safe regardless of exactly when it engages;
//       - the complementarity part uses barr_inf_ = max(sᵢ·λᵢ) (the same
//         quantity the barrier convergence test reads) in place of ‖compl‖₂².
//
// (2) Sufficient-progress test (nonmonotone), [AMU] CheckSufficientProgress()
//     KKT_ERROR case (AMU:452-469): with fewer than kAdaptiveMuKktErrorRedIters
//     reference values, progress is trivially sufficient; otherwise progress is
//     sufficient iff the current error is a sufficient decrease relative to AT
//     LEAST ONE reference value:
//
//       curr_error ≤ kAdaptiveMuKktErrorRedFact · ref   for some ref in refs.
//
// (3) Reference-list maintenance (FIFO of at most kAdaptiveMuKktErrorRedIters),
//     [AMU] RememberCurrentPointAsAccepted() KKT_ERROR case (AMU:496-505): on an
//     accepted point, pop the front if the list is full, then push the current
//     error. The list is updated ONLY while progress is being made (staying
//     free, or re-entering free) — never while remaining in monotone mode — so
//     during monotone mode the reference band is frozen at the last free-mode
//     errors, and re-entry (5) is measured against that frozen band.
//
// (4) Handoff free -> monotone, [AMU] free-mode else-branch (AMU:358-388) +
//     NewFixedMu() default oracle (AMU:616-624): on monitor failure, set
//
//       μ ← clamp( kAdaptiveMuMonotoneInitFactor · avgcomp, min_mu, max_mu ),
//
//     with no iterate restore (Ipopt default adaptive_mu_restore_previous_iterate
//     = no, AMU:126-132). The safeguard and max_ref caps in NewFixedMu are inert
//     under their defaults (adaptive_mu_safeguard_factor = 0 -> lower_mu_safeguard
//     returns 0, AMU:751-753; max_ref = 1e20, AMU:589) and are omitted; the
//     min_mu/max_mu clamp matches ClassicAdaptiveGovernor's common-tail clamp.
//
// (5) Re-entry monotone -> free, [AMU] fixed-mode branch (AMU:299-311):
//     CONFIRMED PRESENT IN SOURCE — while in fixed mode, CheckSufficientProgress()
//     is re-evaluated each iteration against the frozen reference band; if it
//     passes (and no tiny step) Ipopt calls SetFreeMuMode(true) and re-enters
//     free mode. This resolves the earlier planning note ("Ipopt returns to free
//     mode once sufficient progress is re-established") in favour of the memo:
//     the shipped source does re-enter.
//
// (6) Monotone Fiacco–McCormick update, [AMU] fixed-mode decrease (AMU:325-336),
//     identical arithmetic to [MMU] CalcNewMuAndTau (MMU:202-219). Advance μ only
//     when the barrier subproblem has converged, i.e.
//
//       sub_problem_error ≤ kBarrierTolFactor · μ            (gate, AMU:320)
//
//     and then
//
//       μ⁺ = max( floor, min( kBarrierKappaMu·μ, μ^kBarrierThetaMu ) ),   (AMU:327-329)
//       floor = min(bar_tol, kkt_tol) / (kBarrierTolFactor + 1).
//
//     [AMU]'s floor is Min(compl_inf_tol, tol)/(barrier_tol_factor+1); Tycho has
//     no single overall "tol" and instead carries per-part tolerances, so the
//     complementarity tolerance maps to bar_tol_ (≈ Ipopt compl_inf_tol) and the
//     overall optimality tolerance to kkt_tol_ (the stationarity gate). The
//     result is additionally clamped to [min_mu, max_mu] for consistency with
//     ClassicAdaptiveGovernor's common-tail clamp (Ipopt clamps to [mu_min,
//     mu_max] in NewFixedMu / the free oracle, AMU:623-624).
//
//     sub_problem_error (barrier_subproblem_error()) is the ∞-norm barrier
//     optimality error, the analog of [AMU]'s IpCq().curr_barrier_error():
//
//       sub_problem_error = max(kkt_inf_, econ_inf_, icon_inf_, barr_inf_),
//
//     where barr_inf_ = max(sᵢ·λᵢ) stands in for the max-norm perturbed
//     complementarity ‖s·λ − μ·e‖∞ (at a barrier solution s·λ ≈ μ, so the gate
//     becomes "complementarity is within kBarrierTolFactor of μ").
//
// =============================================================================
// μ-EVENT semantics (the per-barrier-subproblem acceptance-reset trigger).
// update_barrier sets its `mu_event` out-param true exactly when a new monotone
// barrier subproblem is initialized with a fresh barrier parameter — the two
// points where [AMU] calls linesearch_->Reset() to reinitialize the filter line
// search for a new subproblem:
//   - the free -> monotone handoff (AMU:386), and
//   - each monotone Fiacco–McCormick μ-advance (AMU:339, guarded by the
//     subproblem-convergence gate and only when μ strictly decreases).
// mu_event does NOT fire on the monotone -> free re-entry: Ipopt's reset there
// (AMU:431) is the free-mode oracle's per-iteration line-search reset, fired on
// every free-mode iteration, which Tycho's classic free-mode path (delegated to
// ClassicAdaptiveGovernor) does not reproduce and which is out of scope for the
// barrier-subproblem event. Firing on the handoff as well as the advances (both
// begin a new barrier subproblem) is the source-faithful reading of the earlier
// planning note, which named only the advances: the fetched source resets the
// filter at both AMU:386 and AMU:339.
// =============================================================================

#pragma once

#include <cstddef>
#include <deque>
#include <memory>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/barrier_governor.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"

namespace tycho::solvers {

class GlobalizationMechanism;

// --- Ipopt option defaults; each cites the source location (see (0) above). ---
inline constexpr int kAdaptiveMuKktErrorRedIters = 4;       // adaptive_mu_kkterror_red_iters
inline constexpr double kAdaptiveMuKktErrorRedFact = 0.9999; // adaptive_mu_kkterror_red_fact
inline constexpr double kAdaptiveMuMonotoneInitFactor = 0.8; // adaptive_mu_monotone_init_factor
inline constexpr double kBarrierKappaMu = 0.2;              // mu_linear_decrease_factor
inline constexpr double kBarrierThetaMu = 1.5;              // mu_superlinear_decrease_power
inline constexpr double kBarrierTolFactor = 10.0;           // barrier_tol_factor

// =============================================================================
// MonitoredBarrierGovernor — free<->monotone monitored barrier governor.
//
// Holds monotone-mode bookkeeping (the KKT-error reference window, the current
// mode, and the current monotone barrier parameter) — the state reset() exists
// to clear at phase boundaries. Everything else it needs is a per-call parameter
// or reached through SolverContext.
// =============================================================================
class MonitoredBarrierGovernor : public BarrierGovernor {
  public:
    // Default: composes a ClassicAdaptiveGovernor as the free-mode delegate.
    MonitoredBarrierGovernor();
    // Injects the free-mode delegate (tests pass a recording fake).
    explicit MonitoredBarrierGovernor(std::unique_ptr<BarrierGovernor> free_delegate);
    ~MonitoredBarrierGovernor() override;

    double update_barrier(PSIOPT::BarrierModes barmode, double mu_in, double avgcomp,
                          double mincomp, Eigen::VectorXd &XSL, Eigen::VectorXd &RHS,
                          Eigen::VectorXd &DXSL, Eigen::VectorXd &Temp,
                          GlobalizationMechanism &mechanism, SolverContext &ctx, double &barr_obj,
                          const IterateInfo &current, bool &mu_event) override;

    bool in_monotone_mode() const override { return monotone_mode_; }

    // This governor supplies its own safeguarded barrier schedule during a nested
    // restoration phase (its free<->monotone monitor forces the Fiacco-McCormick
    // decrease), so the alg_impl seam does NOT overlay update_barrier_monotone on
    // it — see BarrierGovernor::provides_restoration_barrier_safeguard().
    //
    // Scope of the safeguard, per oracle: under the LOQO oracle the free-mode
    // update reads the elastic-augmented complementarity aggregates directly, so
    // the barrier parameter tracks the restoration problem's own barrier state.
    // Under the PROBE oracle the predictor recomputes complementarity from the
    // original slack/multiplier pairs only (see mpc_mu), so the free window after
    // the phase-entry reset (up to the reference-fill length plus one monitor
    // reaction) runs without the augmented signal; containment then rests on the
    // monitor's error measure (which does fold the elastic complementarity) and
    // its monotone handoff re-anchoring at the augmented average. Validated
    // empirically on the restoration-exercising corpus cases: under PROBE the
    // phase enters, iterates, and exits normally with no barrier collapse or
    // frozen-phase behavior, and outcomes match the no-restoration PROBE
    // baseline within a few iterations.
    bool provides_restoration_barrier_safeguard() const override { return true; }

    // Clears the reference window, mode, monotone-mu bookkeeping, and the
    // write-only diagnostics; also resets the free-mode delegate. Phase
    // boundaries start in free mode.
    void reset() override;

    // Reports last_monotone_switches_/last_monotone_iters_ into the
    // corresponding SolveResult fields (psiopt.h) — see BarrierGovernor::
    // append_diagnostics() for the call-site contract this overrides.
    void append_diagnostics(PSIOPT::SolveResult &result) const override;

    // ------------------------------------------------------------------------
    // Testable state machine. `decide` advances the monitor/mode state from
    // `current` (the in-progress iterate's residuals) and returns the barrier
    // decision WITHOUT touching any KKT vectors (the tolerances/bounds it
    // needs are passed as scalars). update_barrier calls it, then applies the
    // barrier tail in monotone mode or delegates in free mode. Exposed so the
    // monitor, handoff, Fiacco–McCormick, re-entry, and mu-event logic are
    // drivable in unit tests without a real KKT solve.
    // ------------------------------------------------------------------------
    struct BarrierDecision {
        double mu = 0.0;      // barrier parameter to use (meaningful iff monotone).
        bool mu_event = false; // a new monotone barrier subproblem began.
        bool monotone = false; // resulting mode after this decision.
    };
    BarrierDecision decide(const IterateInfo &current, double mu_in, double avgcomp,
                           double bar_tol, double kkt_tol, double min_mu, double max_mu);

    // Pure quantities (see (1), (6), (4)).
    static double monitor_error(const IterateInfo &it);
    static double barrier_subproblem_error(const IterateInfo &it);
    static double fiacco_mccormick_mu(double mu, double bar_tol, double kkt_tol, double min_mu,
                                      double max_mu);
    static double handoff_mu(double avgcomp, double min_mu, double max_mu);

    // Reference-window operations (see (2), (3)).
    bool check_sufficient_progress(double curr_error) const;
    void remember_accepted(double curr_error);

    // Test/diagnostic observers.
    const std::deque<double> &reference_values() const { return refs_vals_; }
    double monotone_mu() const { return monotone_mu_; }
    int last_monotone_switches() const { return last_monotone_switches_; }
    int last_monotone_iters() const { return last_monotone_iters_; }

  private:
    // Barrier tail helpers — verbatim copies of the identically-named
    // PSIOPT/ClassicAdaptiveGovernor methods (reading through ctx), applied in
    // monotone mode to write the barrier objective and dual gradient at the
    // fixed monotone μ (free mode delegates this to the composed governor).
    double barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu,
                             const SolverContext &ctx) const;
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu,
                          Eigen::Ref<Eigen::VectorXd> AGS) const;

    std::unique_ptr<BarrierGovernor> free_delegate_;
    // KKT-error reference window (FIFO, at most kAdaptiveMuKktErrorRedIters entries).
    std::deque<double> refs_vals_;
    bool monotone_mode_ = false;
    double monotone_mu_ = 0.0; // current monotone barrier parameter (meaningful iff monotone_mode_)

    // Write-only SolveResult diagnostics, bound to the result via
    // append_diagnostics() (see PSIOPT::SolveResult::last_monotone_switches_/
    // last_monotone_iters_ in psiopt.h).
    int last_monotone_switches_ = 0; // free -> monotone handoffs this phase.
    int last_monotone_iters_ = 0;    // iterations spent in monotone mode this phase.
};

} // namespace tycho::solvers
