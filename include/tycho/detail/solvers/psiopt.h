// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
//   - Configuration fields grouped into Settings struct
//   - Converted from struct to class with public/private access sections
//   - KKTVector extracted to detail/solvers/kkt_vector.h; EvalErrorLog extracted
//     to detail/solvers/eval_error_log.h (both shared with the globalization
//     components)
// =============================================================================

#pragma once
#include <array>
#include <cassert>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <fmt/color.h>
#include <fmt/core.h>

#include "tycho/detail/solvers/bound_set.h"
#include "tycho/detail/solvers/eval_error_log.h"
#include "tycho/detail/solvers/iterate_info.h"
#include "tycho/detail/solvers/kkt_vector.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/psiopt_fwd.h"
#include "tycho/detail/typedefs/eigen_types.h"

#ifdef USE_ACCELERATE_SPARSE
#include "tycho/detail/solvers/linear/accelerate_interface.h"
#include <limits>
#else
#include "tycho/detail/solvers/linear/pardiso_interface.h"
#endif

// Forward declarations of gtest-generated test-fixture classes (global
// namespace, per gtest's TEST() expansion) that are befriended below. See the
// "Test access" comment in the PSIOPT class body for why this exists.
class RecoveryDispatchGate_FunnelSelectionConstructsFunnelAcceptance_Test;
class RecoveryDispatchGate_FilterSelectionConstructsFilterAcceptance_Test;
class RecoveryDispatchGate_MonitoredSelectionConstructsMonitoredGovernor_Test;
class RecoveryDispatchGate_MeritPenaltyRuleSelectionReachesTheStrategy_Test;
class FeasibilitySwitch_ProximalSwitchConstructsRestorationAndWrapsRecovery_Test;
class FeasibilitySwitch_OffModeConstructsNoRestoration_Test;
class FeasibilitySwitch_FilterSeedsRestorationConstraintTol_Test;
// Test harness for the nested feasibility-restoration eval/step seam: reaches
// private eval_nlp / alg_impl / restoration_ / dims to drive the seam directly.
class NestedSeamHarness;
// Inequality-row variant of the seam harness: drives the eval seam on a problem
// with an inequality constraint so the slack-completed inequality condensation is
// verified through the assembled KKT.
class NestedSeamIneqHarness;
// Test harness for the nested feasibility-restoration LIFECYCLE (entry
// orchestration, exit ratchet, multiplier re-entry): reaches the private
// enter_/exit_feasibility_restoration helpers, the stashed-μ / ratchet state,
// restoration_, and alg_impl to drive the whole phase end-to-end.
class NestedLifecycleHarness;
// Test harness for the persistence-based divergence classification in
// converge_check(): reaches the private converge_check() and settings_ so the
// trailing-window logic can be exercised directly on synthetic iterate
// histories.
class DivergencePersistenceHarness;
// Test harness for the SOC / extended-backtracking recovery links under the
// generic-path acceptance strategies: reaches the private nlp_ / kkt_sol_ /
// dims / scratch / restoration_ / acceptance_ / recovery_ so it can build a
// live SolverContext and drive the mechanism's acceptance-backtrack seam with a
// generic acceptance strategy (see test_soc_generic_acceptance.cpp).
class SocGenericHarness;
// Test harness for the native variable-bound machinery: reaches the private
// interior push, the bound-multiplier direction/update helpers and the
// bound_duals_/bounds_ state so each can be checked against a hand calculation
// without a full bounded solve (there is no fraction-to-boundary leg yet).
class NativeBoundsHarness;

namespace tycho::solvers {

// Pull root-namespace Eigen type aliases into tycho::solvers so that PSIOPT
// member declarations (EigenRef<VectorXd>, ConstEigenRef<VectorXd>, …) resolve
// without full qualification inside this namespace.
using tycho::ConstEigenRef;
using tycho::EigenRef;

// Number of consecutive trailing iterates that must ALL exceed a divergence
// threshold before converge_check() declares DIVERGING on a finite (but large)
// residual. A single iterate breaching a threshold no longer aborts the solve;
// the breach must persist across this many iterations in a row.
//
// Rationale. Non-finite residuals (NaN/Inf) remain an immediate hard abort — no
// iterate recovers from a corrupted state — so this window governs only the
// finite-overshoot case, where a single blown-up iterate can be a recoverable
// transient rather than true divergence. The classic Maratos-effect example
// (min 2(x1²+x2²−1)−x1 s.t. x1²+x2²−1=0, started on the constraint manifold)
// makes the case concrete: under every solver configuration it takes one step
// whose equality residual momentarily explodes to ~5e15, then converges in
// roughly forty iterations to the textbook optimum (obj −1) with no recovery
// machinery engaged. A per-iterate abort mistakes that single-iteration
// excursion for divergence and kills an otherwise convergent solve.
//
// Three is the smallest window that survives the observed one- and
// two-iteration recoverable excursions (Maratos-class overshoots,
// restoration-entry transients) while still failing fast — within three
// iterations of the onset — on genuine divergence. It is a Tycho policy choice
// with no external reference: Ipopt ships no divergence abort at all. The
// supporting evidence is the corpus differential — the same literature problem
// diverges at iteration two with the per-iterate abort and converges to the
// optimum without it.
inline constexpr int kDivergencePersistIters = 3;

// Part of the globalization component extraction: PSIOPT owns its
// step-acceptance strategy through a std::unique_ptr<AcceptanceStrategy>.
// Only the forward declaration is needed here (the complete type lives in
// detail/solvers/globalization/acceptance_strategy.h, which includes THIS
// header — so psiopt.h must not include it back). Because the member is a
// unique_ptr to this incomplete type, PSIOPT's constructors and destructor are
// declared here and defined out-of-line in psiopt.cpp, where the concrete
// ClassicMeritAcceptance is complete.
//
// PSIOPT likewise owns its step-length globalization mechanism
// through a std::unique_ptr<GlobalizationMechanism> (concrete type
// BacktrackingLineSearch, complete only in psiopt.cpp). Same forward-declare +
// out-of-line ctor/dtor discipline as AcceptanceStrategy above.
//
// PSIOPT owns its barrier-parameter governor through a
// std::unique_ptr<BarrierGovernor> (concrete type ClassicAdaptiveGovernor,
// complete only in psiopt.cpp). Same forward-declare + out-of-line ctor/dtor
// discipline.
//
// PSIOPT owns its post-rejection recovery chain through a
// std::unique_ptr<RecoveryChain> (concrete type depends on settings — see
// the recovery_ field comment below; complete only in psiopt.cpp). Same
// forward-declare + out-of-line ctor/dtor discipline. NoopRecovery is
// installed only on the all-default path (max_soc_ == 0, ls_extended_iters_
// == 0, watchdog_ == false, restoration_mode_ == off); live links exist for
// every opt-in (SocRecovery, ExtendedBacktrackRecovery, WatchdogRecovery,
// FeasibilitySwitchRecovery — see rebuild_globalization_components()).
// PSIOPT also owns an optional feasibility-restoration mode-switch through a
// std::unique_ptr<RestorationStrategy> (concrete type ProximalSwitchRestoration
// or NestedL1Restoration depending on restoration_mode_, complete only in
// psiopt.cpp). Unlike the four components above it is NOT always constructed:
// rebuild_globalization_components() leaves it null unless restoration_mode_
// != off, so on the default path every restoration branch guards on
// `restoration_ != nullptr` and is provably dead. Same forward-declare +
// out-of-line ctor/dtor discipline as the others.
class AcceptanceStrategy;
class GlobalizationMechanism;
class BarrierGovernor;
class RecoveryChain;
class RestorationStrategy;
struct ProgressMeasures;
struct FeasibilityStallDetector;

class PSIOPT {
  public:
    enum class BarrierModes { PROBE, LOQO };
    enum class LineSearchModes { AUGLANG, LANG, L1, NOLS };
    enum class AlgorithmModes { OPT, OPTNO, SOE, INIT };

    enum class QPAlgModes {
        Classic = 0,
        TwoLevel = 1,
    };

    enum class QPOrderingModes { MINDEG = 0, METIS = 2, PARMETIS = 3 };
    enum class BestCriteriaModes { ECONS, ICONS, KKT, OBJ };

    enum class QPPivotModes {
        OneByOne = 0,
        TwoByTwo = 1,
        E4 = 4,
        E6 = 6,
        E8 = 8,
        E13 = 13,
    };
    enum class PDStepStrategies { PrimSlackEq_Iq, AllMinimum, PrimSlack_EqIq, MaxEq };

    // --- Static string-to-enum converters (defined in psiopt.cpp) ---
    static QPOrderingModes strto_OrderingMode(const std::string &str);
    static LineSearchModes strto_LineSearchMode(const std::string &str);
    static BarrierModes strto_BarrierMode(const std::string &str);
    static BestCriteriaModes strto_BestCriteriaMode(const std::string &str);

    // =========================================================================
    // Settings — all user-configurable parameters grouped in one place
    // =========================================================================
    struct Settings {
        // --- Iteration limits ---
        int max_iters_ = 500;
        int max_ls_iters_ = 2;
        int max_acc_iters_ = 50;
        int max_refac_ = 15;
        // Maximum second-order corrections attempted after a first-trial
        // rejection (Wächter & Biegler 2006, §2.4). 0 = off (default): the
        // solver behaves exactly as it did before SOC existed. Set > 0 to opt
        // in; the recommended enable value is 4 (kSocRecommendedMaxCorrections
        // in globalization/soc.h).
        int max_soc_ = 0;

        // Extended backtracking: further trials continuing the SAME classic
        // ladder (same direction, same alpha_red_ divisor, same merit test)
        // once the classic capped backtrack rejects and SOC (if enabled) is
        // exhausted or not triggered. 0 = off (default): the solver behaves
        // exactly as it did before extended backtracking existed. This cap
        // extends the classic cap (max_ls_iters_) ONLY when the recovery
        // dispatch is active on a rejected step — max_ls_iters_ itself is
        // unaffected. See ExtendedBacktrackRecovery, globalization/watchdog.h.
        int ls_extended_iters_ = 0;

        // Watchdog (Chamberlain, Powell, Lemaréchal & Pedersen 1982;
        // constants per Wächter & Biegler 2006's implementation — see
        // globalization/watchdog.h): arms after kWatchdogShortenedIterTrigger
        // consecutive fully-rejected iterations, then accepts up to
        // kWatchdogTrialIterMax trial iterations under relaxed acceptance
        // before reverting to the pre-watchdog snapshot. false = off
        // (default): the solver behaves exactly as it did before the
        // watchdog existed.
        bool watchdog_ = false;

        // Per-phase feasibility-restoration entry budget: the maximum number of
        // times restoration mode may be entered within a single phase. Read by
        // ProximalSwitchRestoration::entry_permitted() (globalization/
        // proximal_restoration.h) or NestedL1Restoration::entry_permitted()
        // (globalization/l1_restoration.h), whichever restoration_mode_
        // selects — 0 refuses restoration entirely (budget exhausted before
        // the first entry). Ignored when restoration_mode_ == off. validate()
        // requires it >= 0.
        int max_feas_rest_ = 2;

        // --- Convergence tolerances ---
        double kkt_tol_ = 1.0e-6;
        double econ_tol_ = 1.0e-6;
        double icon_tol_ = 1.0e-6;
        double bar_tol_ = 1.0e-6;

        // --- Acceptable tolerances ---
        double acc_kkt_tol_ = 1.0e-2;
        double acc_econ_tol_ = 1.0e-3;
        double acc_icon_tol_ = 1.0e-3;
        double acc_bar_tol_ = 1.0e-3;

        // --- Divergence tolerances ---
        double div_kkt_tol_ = 1.0e15;
        double div_econ_tol_ = 1.0e15;
        double div_icon_tol_ = 1.0e15;
        double div_bar_tol_ = 1.0e15;

        // --- Algorithm modes ---
        AlgorithmModes soe_mode_ = AlgorithmModes::SOE;
        BarrierModes opt_bar_mode_ = BarrierModes::LOQO;
        BarrierModes soe_bar_mode_ = BarrierModes::LOQO;
        LineSearchModes opt_ls_mode_ = LineSearchModes::AUGLANG;
        LineSearchModes soe_ls_mode_ = LineSearchModes::NOLS;
        PDStepStrategies pd_step_strategy_ = PDStepStrategies::PrimSlackEq_Iq;

        // --- Step-acceptance strategy (opt-in modernized merit) ---
        // classic_merit (default) reproduces today's fused backtracking merit
        // line search bit-identically. merit selects the modernized merit
        // family driven through the GENERIC AcceptanceStrategy path, with the
        // penalty rule chosen by merit_penalty_rule_ (only read when
        // acceptance_strategy_ == merit). Both enums live in psiopt_fwd.h.
        AcceptanceStrategies acceptance_strategy_ = AcceptanceStrategies::classic_merit;
        MeritPenaltyRules merit_penalty_rule_ = MeritPenaltyRules::wmno;

        // --- Barrier-parameter governor (opt-in monitored free<->monotone) ---
        // classic_adaptive (default) reproduces today's PROBE/LOQO free-mode
        // barrier update bit-identically. monitored selects the free<->monotone
        // MonitoredBarrierGovernor, which composes a ClassicAdaptiveGovernor as
        // its free-mode delegate — so it may pair with any acceptance_strategy_.
        // The funnel/filter acceptance strategies are designed to operate above
        // a monotone barrier safeguard; validate() rejects them combined with
        // classic_adaptive unless never_monotone_ is explicitly set (see
        // validate()'s guard below). Enum lives in psiopt_fwd.h.
        BarrierGovernors barrier_governor_ = BarrierGovernors::classic_adaptive;

        // Expert escape hatch mirroring Ipopt's never-monotone-mode: explicitly
        // accepts running funnel/filter above the classic_adaptive (free-only)
        // barrier governor without its monotone safeguard. false (default).
        // Contradictory when combined with barrier_governor_ == monitored (the
        // monitored governor already provides the monotone fallback) — validate()
        // rejects that combination.
        bool never_monotone_ = false;

        // --- Feasibility restoration (opt-in proximal mode-switch / nested l1) ---
        // off (default) reproduces today's behavior bit-identically: no
        // RestorationStrategy is constructed and every restoration branch in
        // the solver is provably dead. proximal_switch selects the proximal
        // feasibility mode-switch (ProximalSwitchRestoration), which — on a
        // ladder-exhausted step rejection at a not-near-feasible point — swaps
        // the true objective for a proximal term until infeasibility is
        // sufficiently reduced, then resumes optimality mode. l1_nested
        // selects the nested l1 elastic feasibility restoration
        // (NestedL1Restoration, globalization/l1_restoration.h) instead: the
        // same trigger, but the l1 elastic reformulation runs as a condensed
        // in-place phase reusing the outer barrier algorithm's KKT system
        // rather than swapping the outer objective. Both modes compose with
        // every acceptance_strategy_ and barrier_governor_ (no matrix
        // restrictions — every shipped acceptance strategy implements the
        // restoration exit test the modes rely on). Enum lives in
        // psiopt_fwd.h; the per-phase entry budget is max_feas_rest_ above,
        // shared by both modes.
        RestorationModes restoration_mode_ = RestorationModes::off;

        // --- Barrier parameters ---
        double init_mu_ = 0.001;
        double max_mu_ = 100.0;
        double min_mu_ = 1.0e-12;

        // --- Step parameters ---
        double bound_fraction_ = 0.99;
        double bound_push_ = 1.0e-3;
        // Absolute component of the interior push applied to a bounded primal
        // variable at solve entry: the push away from a bound is at least
        // bound_push_ * max(1, |bound|) (Ipopt's bound_push). It also caps the
        // initial slack the INIT multiplier pass hands an inequality row.
        //
        // Relative component of that same push, applied only to a TWO-SIDED
        // variable: the push is additionally capped at
        // bound_interval_push_ * (upper - lower), so a narrow interval is never
        // pushed past its own midpoint (Ipopt's bound_frac, same default).
        // Read only when the problem declares native variable bounds.
        double bound_interval_push_ = 1.0e-2;
        double neg_slack_reset_ = 1.0e-12;
        double alpha_red_ = 2.0;

        // --- Fixed-variable treatment ---
        // How a primal variable whose declared lower and upper bounds are equal
        // is handed to the solver. make_parameter (the default) eliminates it,
        // so the factorized system is one row and column narrower per fixed
        // variable and the variable's value in the returned solution is exact.
        // make_constraint keeps it and adds one internal equality row per fixed
        // variable, appended after every row the transcription declared, so the
        // system is one row and column WIDER instead. relax_bounds keeps it as a
        // two-sided bounded variable whose bounds have been pushed apart by
        // bound_relax_factor_, holding it near its value through the barrier.
        // All three reach the same solution on a well-posed problem. Closed-set
        // enum; it lives in non_linear_program.h alongside the classification
        // that reads it.
        FixedVariableTreatments fixed_variable_treatment_ = FixedVariableTreatments::MakeParameter;

        // Widening applied to every finite variable bound before the classifier
        // records it: the bound is moved outward by this factor times
        // max(1, |bound|), so the box the barrier terms divide by is never
        // exactly the declared one (Ipopt's bound_relax_factor, same default).
        // It is also what separates the bounds of a fixed variable under the
        // relax_bounds treatment, which therefore requires it to be positive.
        // Zero records every declared bound verbatim.
        //
        // Both of these are read once per solve, at the NLP's classification
        // pass; changing either between two solves on one solver re-classifies.
        double bound_relax_factor_ = kDefaultBoundRelaxFactor;

        // --- Hessian perturbation ---
        double delta_h_ = 1.0e-5;
        double incr_h_ = 8.0;
        double decr_h_ = 0.333333;

        // KKT inertia-correction / regularization mode. classic (default) runs
        // the on-demand inertia ladder exactly as before — bit-identical.
        // proximal_regularization bakes a persistent, decaying primal base shift
        // ρ_k and an always-on barrier-scaled dual shift −δ_c into the base
        // matrix each iteration (the same ladder still escalates on top when the
        // base attempt has wrong inertia or is singular). ρ_k starts at
        // kProxRegFloor and decays by decr_h_ toward that floor; δ_c uses the
        // δ_c-ladder constants in globalization/inertia_regularization.h and is
        // suppressed while a nested l1 restoration phase is active. Closed-set
        // enum, so validate() needs no range check; no other setting is
        // required. Enum lives in psiopt_fwd.h.
        InertiaModes inertia_mode_ = InertiaModes::classic;

        // --- QP solver ---
        int qp_threads_ = TYCHO_DEFAULT_QP_THREADS;
        QPAlgModes qp_alg_ = QPAlgModes::Classic;
        QPOrderingModes qp_ord_ = QPOrderingModes::METIS;
        QPPivotModes qp_pivot_strategy_ = QPPivotModes::TwoByTwo;
        // MKL Pardiso weighted matching (iparm[12]) / MPS scaling (iparm[10]), 0/1 flags.
        // Matching stays ON by default; scaling stays OFF by default. Enabling scaling
        // (qp_scaling=1) measured -16% wall on PolarLT-class collocation problems and drops
        // perturbed pivots 95/120 -> ~0, but on the full example suite it deterministically
        // degraded convergence elsewhere (Delta3Launch CONVERGED->ACCEPTABLE, TopputtoLowThrust
        // 5.4x iterations, intermittent MultiSpacecraft divergence) — see
        // docs/dev/analysis/2026-07-pr9-pardiso-options.md. Opt in via the qp_scaling knob.
        int qp_matching_ = 1;
        int qp_scaling_ = 0;
        int qp_pivot_perturb_ = 8;
        int qp_ref_steps_ = 0;
        int qp_par_solve_ = 0;
        bool qp_print_ = false;
#ifdef USE_ACCELERATE_SPARSE
        double accel_pivot_tolerance_ = 0.01;
        double accel_zero_tolerance_ = 1e-4 * std::numeric_limits<double>::epsilon();
#endif

        // --- Objective ---
        double obj_scale_ = 1.0;

        // --- Output/behavior ---
        // Output verbosity:
        //   0 — full output (stats + iteration table + exit + timing)
        //   1 — no iteration table (phase banners + timing summary)
        //   2 — exit status and warnings only
        //   3+ — fully silent
        int print_level_ = 0;
        bool wide_console_ = false;
        bool cnr_mode_ = false;
        bool fast_factor_alg_ = true;
        bool force_qp_analysis_ = false;
        bool return_best_ = false;
        BestCriteriaModes best_criteria_ = BestCriteriaModes::ECONS;

        /// Validate all settings, throwing std::invalid_argument on the first
        /// violation. Checks per-field conditions (matching the individual
        /// set_*() methods) plus cross-field invariants (min_mu <= init_mu <=
        /// max_mu, convergence tols <= their respective acceptable tols <=
        /// their respective divergence tols).
        void validate() const;
    };

    // =========================================================================
    // SolveResult — accumulated outputs from the most recent solve/optimize call
    // =========================================================================
    struct SolveResult {
        // --- Solve outcome ---
        int iter_num_ = 0;
        double obj_val_ = 0;
        ConvergenceFlags converge_flag_ = ConvergenceFlags::NOTCONVERGED;

        // --- Solution ---
        Eigen::VectorXd primals_;

        // --- Multipliers and constraints ---
        Eigen::VectorXd eq_lmults_;
        Eigen::VectorXd iq_lmults_;
        Eigen::VectorXd eq_cons_;
        Eigen::VectorXd iq_cons_;

        // --- Timing (seconds) ---
        double total_time_ = 0;
        double pre_time_ = 0;
        double func_time_ = 0;
        double kkt_time_ = 0;
        double print_time_ = 0;
        double solver_init_time_ = 0;

        // Derived timing — total wall-clock minus all categorized components.
        // Excludes solver_init_time_ (measured before the main timer starts).
        // Captures: callback time, step application, convergence checks, etc.
        double misc_time() const {
            return total_time_ - pre_time_ - kkt_time_ - func_time_ - print_time_;
        }

        // --- Factorization stats ---
        int factor_mem_ = 0;
        int factor_flops_ = 0;

        // Number of second-order correction back-substitutions performed across
        // the whole solve (one per correction attempt; each costs a single
        // constraint evaluation + one back-substitution on the live
        // factorization). Always 0 when SOC is off (max_soc_ == 0). Reset per
        // solve alongside the other accumulators.
        int soc_steps_taken_ = 0;

        // Number of times the watchdog armed across the whole solve
        // (Chamberlain, Powell, Lemaréchal & Pedersen 1982; constants per
        // Wächter & Biegler 2006's implementation, globalization/watchdog.h).
        // Always 0 when the watchdog is off (watchdog_ == false). Reset per
        // solve alongside the other accumulators.
        int watchdog_activations_ = 0;

        // Per-rejection recovery-chain outcome depth, indexed by the
        // kRecoveryDepth* constants in globalization/recovery_chain.h:
        // recovery_depth_histogram_[0] SOC, [1] extended backtracking,
        // [2] watchdog, [3] unresolved (today's classic give-up: the
        // originally-rejected step was simply taken; this is the ONLY bucket
        // that increments when SOC/extended/watchdog are all off),
        // [4] restoration (a feasibility-restoration mode-switch was taken —
        // only increments when restoration_mode_ != off). Counts
        // rejections, i.e. every should_dispatch_recovery-gated call, not
        // just ones where a recovery link actually intervened. Reset per
        // solve alongside the other accumulators.
        std::array<int, 5> recovery_depth_histogram_{};

        // Final funnel width (τ) reported by FunnelAcceptance::
        // append_diagnostics() (globalization/funnel_acceptance.h) at the end
        // of the most recent solve's LAST PHASE. Sentinel -1.0 when the
        // selected acceptance strategy does not report this field (every
        // strategy except funnel — the default AcceptanceStrategy::
        // append_diagnostics() no-op leaves this untouched). A multi-phase
        // call (e.g. solve_optimize()) reports only the LAST phase's value,
        // not a running total across phases — see the collection point in
        // run_phase_sequence(). Reset per solve alongside the other
        // accumulators; NOT touched by AcceptanceStrategy::reset() (the
        // per-phase hook), only by reset_accumulators() (the per-solve hook).
        // Sentinel -1.0 reports when no acceptance test ran in the selected
        // phase (e.g. the phase converged at its initial iterate).
        double last_funnel_width_ = -1.0;

        // Final filter size (number of stored (θ, φ) pairs, Filter::size())
        // reported by FilterAcceptance::append_diagnostics()
        // (globalization/filter_acceptance.h) at the end of the most recent
        // solve's LAST PHASE. Sentinel -1 when the selected acceptance
        // strategy is not filter. Same last-phase-only semantics as
        // last_funnel_width_ above.
        int last_filter_size_ = -1;

        // Total number of filter-reset-heuristic clears
        // (FilterAcceptance::filter_resets(), Ipopt n_filter_resets_ — see
        // filter_acceptance.h rule (4)) reported at the end of the most
        // recent solve's LAST PHASE. Sentinel -1 when the selected acceptance
        // strategy is not filter. PER-PHASE semantics: the counter is
        // cleared by FilterAcceptance::reset_bounds() at every phase
        // boundary (via AcceptanceStrategy::reset(), called at the top of
        // each run_phase_sequence() loop iteration), and append_diagnostics()
        // is collected once per phase right before that reset runs for the
        // NEXT phase — so a multi-phase call (e.g. solve_optimize()) reports
        // only the LAST phase's total resets, not a running total across
        // phases within the same solve() call. Under barrier_governor_ ==
        // monitored, each mu-event ALSO clears the counter (the acceptance
        // strategy is reset per barrier subproblem), so this reports resets
        // since the last mu-event of the last phase — the Ipopt-faithful
        // per-subproblem scope, not a whole-phase total.
        int last_filter_resets_ = -1;

        // Number of free -> monotone handoffs during the most recent solve's
        // LAST PHASE, reported by MonitoredBarrierGovernor::append_diagnostics()
        // (globalization/monitored_governor.h). Sentinel -1 when the selected
        // barrier_governor_ is not monitored. PER-PHASE semantics matching
        // last_filter_resets_ above: MonitoredBarrierGovernor::reset() clears
        // its own last_monotone_switches_/last_monotone_iters_ counters at
        // every phase boundary (via BarrierGovernor::reset(), called at the top
        // of each run_phase_sequence() loop iteration), and
        // append_diagnostics() is collected once per phase right before that
        // reset runs for the NEXT phase — so a multi-phase call reports only
        // the LAST phase's totals, not a running total across phases.
        int last_monotone_switches_ = -1;

        // Number of iterations spent in monotone mode during the most recent
        // solve's LAST PHASE, reported by MonitoredBarrierGovernor::
        // append_diagnostics(). Sentinel -1 when the selected barrier_governor_
        // is not monitored. Same per-phase semantics as last_monotone_switches_.
        int last_monotone_iters_ = -1;

        // Number of times feasibility restoration was entered during the most
        // recent solve's LAST PHASE, reported by RestorationStrategy::
        // append_diagnostics() (globalization/restoration.h;
        // ProximalSwitchRestoration and NestedL1Restoration are today's
        // concrete reporters — globalization/proximal_restoration.h,
        // globalization/l1_restoration.h). WRITE-ONLY diagnostics field: no
        // algorithm code reads it back. Sentinel -1 when no restoration
        // strategy is constructed, i.e. restoration_mode_ == off. Same
        // last-phase-wins semantics as last_monotone_switches_. Counting is
        // identical across both modes: entries_ increments once per
        // enter_restoration()/enter_nested() call, and iterations_in_mode_
        // once per note_iteration() call while active — the nested mode has
        // no separate inner/outer iteration split (its phase shares the
        // outer loop's own iteration counter; see l1_restoration.h disclosure
        // (a)), so this field means the same thing under both modes.
        int last_feas_rest_entries_ = -1;

        // Number of iterations spent in restoration mode during the most
        // recent solve's LAST PHASE, reported by RestorationStrategy::
        // append_diagnostics(). WRITE-ONLY diagnostics field. Sentinel -1
        // when no restoration strategy is constructed. Same per-phase
        // semantics as last_feas_rest_entries_ (including the nested-mode
        // counting note above).
        int last_feas_rest_iters_ = -1;

        // Proximal primal-dual regularization shifts applied at the LAST
        // FACTORIZED ITERATION of the most recent solve's LAST PHASE, written
        // by alg_impl() at phase close from mode-local state (there is no
        // dedicated component object with its own append_diagnostics() hook
        // here — unlike the acceptance/governor/restoration fields above; and
        // the trailing iterate-history entry is the wrong source because a
        // converged exit appends a non-factorized convergence probe).
        // last_prox_reg_primal_ is the persistent primal base shift ρ_k added
        // to the Hessian diagonal at that iteration; last_prox_reg_dual_ is
        // the barrier-scaled dual shift δ_c subtracted from the
        // constraint-row diagonals (0.0 when suppressed inside a nested l1
        // restoration phase). Sentinel -1.0 for BOTH fields when
        // inertia_mode_ != proximal_regularization — the classic path never
        // writes them — and when a mode-on phase converged before its first
        // factorization. Same last-phase-wins semantics as the fields above.
        double last_prox_reg_primal_ = -1.0;
        double last_prox_reg_dual_ = -1.0;

        // Message of the most recent trial-evaluation exception absorbed by
        // the acceptance machinery during the most recent solve call (all
        // phases). Empty when every evaluation succeeded. A populated value
        // means the solver rejected un-evaluable trial steps and continued —
        // to full recovery, to a graceful ACCEPTABLE-level exit at an
        // already-acceptable iterate, or into feasibility restoration. When
        // none of those paths existed, the solve threw the latched message
        // wrapped in solver context instead.
        std::string last_eval_exception_;

        // T6 (dead-status fix): the last non-Success status observed from
        // kkt_sol_.info() by factor_impl() within the CURRENT phase (alg_impl
        // resets it on entry, so print_exit_stats reports per-phase status).
        // kkt_sol_.info() was previously computed by every
        // Compute()/Refactor() call and never read anywhere; this field is purely
        // observational (surfaced by print_exit_stats()) and does not feed back into
        // any control-flow decision in factor_impl -- see the comment there.
        Eigen::ComputationInfo last_kkt_info_ = Eigen::Success;

        // Only resets accumulated timing/iteration counters and the convergence flag.
        // primals_ and obj_val_ are overwritten unconditionally by alg_impl each
        // phase. eq_lmults_ and eq_cons_ are overwritten when equal_cons_ > 0;
        // iq_lmults_ and iq_cons_ are overwritten when inequal_cons_ > 0.
        // factor_mem_ and factor_flops_ reflect the last factorization's stats
        // (set by init_impl) and are not accumulated across phases.
        void reset_accumulators() {
            converge_flag_ = ConvergenceFlags::NOTCONVERGED;
            total_time_ = 0;
            pre_time_ = 0;
            func_time_ = 0;
            kkt_time_ = 0;
            print_time_ = 0;
            solver_init_time_ = 0;
            iter_num_ = 0;
            last_kkt_info_ = Eigen::Success;
            soc_steps_taken_ = 0;
            watchdog_activations_ = 0;
            recovery_depth_histogram_.fill(0);
            last_funnel_width_ = -1.0;
            last_filter_size_ = -1;
            last_filter_resets_ = -1;
            last_monotone_switches_ = -1;
            last_monotone_iters_ = -1;
            last_feas_rest_entries_ = -1;
            last_feas_rest_iters_ = -1;
            last_prox_reg_primal_ = -1.0;
            last_prox_reg_dual_ = -1.0;
            last_eval_exception_.clear();
        }
    };

    using VectorXd = Eigen::VectorXd;

    // CALLBACK VARIABLE SPACE. Both callbacks are handed the solver's own
    // iterate, right-hand side and (for the early one) KKT matrix. On a problem
    // with bound-fixed variables that space is the REDUCED one: variables whose
    // bounds fix them are eliminated, so the primal block is narrower than the
    // initial guess the caller passed to optimize()/solve(), and every segment
    // offset inside these vectors follows the narrowed width. That is the only
    // internally consistent choice -- the early callback receives the KKT matrix
    // itself, so a full-space iterate beside it would have every block boundary
    // in the wrong place. A callback that needs the caller's own numbering maps
    // through NonLinearProgram::reduced_to_full(), and can rebuild a full-space
    // primal vector with scatter_full_x(). The returned solution, by contrast,
    // is always in the caller's space. print_stats() likewise reports the
    // solver's primal count, i.e. the width of the system being factorized.
    using EarlyCallBackType =
        std::function<int(int, double, EigenRef<VectorXd>, double, EigenRef<VectorXd>,
                          EigenRef<VectorXd>, Eigen::SparseMatrix<double, Eigen::RowMajor> &)>;

    using LateCallBackType =
        std::function<int(const IterateInfo &, ConstEigenRef<VectorXd>, ConstEigenRef<VectorXd>)>;

    // --- Constructors / destructor ---
    // Defined out-of-line in psiopt.cpp: the unique_ptr<AcceptanceStrategy>
    // member forces even the constructors' exception-cleanup paths (and the
    // destructor) to see the complete AcceptanceStrategy type, which is only
    // available in the .cpp. Bodies are otherwise unchanged.
    PSIOPT();
    PSIOPT(std::shared_ptr<NonLinearProgram> np);
    ~PSIOPT();

    // Neither copyable nor movable: the out-of-line destructor above (needed
    // for the incomplete-type unique_ptr members) already silently suppresses
    // the implicit move members, and PSIOPT's raw kkt_sol_ solver handle plus
    // its unique_ptr<...> globalization components have no defined transfer
    // semantics today. Explicit rather than relying on that suppression, so
    // the constraint is visible at the declaration instead of discovered at a
    // failed call site.
    PSIOPT(const PSIOPT &) = delete;
    PSIOPT &operator=(const PSIOPT &) = delete;
    PSIOPT(PSIOPT &&) = delete;
    PSIOPT &operator=(PSIOPT &&) = delete;

    // --- Accessors ---
    /// Returns a mutable reference to the settings struct. Direct writes bypass
    /// per-field validation in the set_*() methods. All settings are re-validated
    /// at run_phase_sequence() entry via Settings::validate().
    Settings &settings() { return settings_; }
    const Settings &settings() const { return settings_; }
    const SolveResult &result() const { return result_; }
    const EvalErrorLog &eval_error_log() const { return eval_error_log_; }

    // --- NLP management ---
    void set_nlp(std::shared_ptr<NonLinearProgram> np);
    void release();

    // --- Entry points ---
    Eigen::VectorXd optimize(const Eigen::VectorXd &x);
    Eigen::VectorXd solve(const Eigen::VectorXd &x);
    Eigen::VectorXd solve_optimize(const Eigen::VectorXd &x);
    Eigen::VectorXd optimize_solve(const Eigen::VectorXd &x);
    Eigen::VectorXd solve_optimize_solve(const Eigen::VectorXd &x);

    // --- Validated setter methods (defined in psiopt.cpp) ---
    void set_max_iters(int max_iters);
    void set_max_acc_iters(int max_acc_iters);
    void set_max_ls_iters(int max_ls_iters);
    void set_all_max_iters(int m1, int m2);
    void set_max_soc(int max_soc);
    void set_ls_extended_iters(int ls_extended_iters);
    void set_max_feas_rest(int max_feas_rest);

    void set_kkt_tol(double kkt_tol);
    void set_bar_tol(double bar_tol);
    void set_econ_tol(double econ_tol);
    void set_icon_tol(double icon_tol);
    void set_tols(double kkt_tol, double econ_tol, double icon_tol, double bar_tol);

    void set_acc_kkt_tol(double acc_kkt_tol);
    void set_acc_bar_tol(double acc_bar_tol);
    void set_acc_econ_tol(double acc_econ_tol);
    void set_acc_icon_tol(double acc_icon_tol);
    void set_acc_tols(double acc_kkt_tol, double acc_econ_tol, double acc_icon_tol,
                      double acc_bar_tol);

    void set_div_kkt_tol(double div_kkt_tol);
    void set_div_bar_tol(double div_bar_tol);
    void set_div_econ_tol(double div_econ_tol);
    void set_div_icon_tol(double div_icon_tol);
    void set_div_tols(double div_kkt_tol, double div_econ_tol, double div_icon_tol,
                      double div_bar_tol);

    void set_bound_fraction(double bound_fraction);
    void set_bound_push(double bound_push);
    void set_bound_interval_push(double bound_interval_push);
    void set_bound_relax_factor(double bound_relax_factor);
    void set_fixed_variable_treatment(FixedVariableTreatments treatment);
    void set_alpha_red(double ared);

    void set_delta_h(double delta_h);
    void set_incr_h(double incr_h);
    void set_decr_h(double decr_h);
    void set_hpert_params(double delta_h, double incr_h, double decr_h);

    void set_print_level(int plevel);

    void set_init_mu(double mu);
    void set_min_mu(double mu);
    void set_max_mu(double mu);
    void set_neg_slack_reset(double val);
    void set_qp_threads(int n);
    void set_qp_pivot_perturb(int v);
    void set_qp_matching(int v);
    void set_qp_scaling(int v);
    void set_qp_ref_steps(int v);
    void set_qp_par_solve(int v);
    void set_obj_scale(double scale);

    void set_qp_ordering_mode(QPOrderingModes mode);
    void set_qp_ordering_mode(const std::string &str);

    void set_opt_bar_mode(BarrierModes mode);
    void set_opt_bar_mode(const std::string &str);
    void set_soe_bar_mode(BarrierModes mode);
    void set_soe_bar_mode(const std::string &str);

    void set_opt_ls_mode(LineSearchModes mode);
    void set_opt_ls_mode(const std::string &str);
    void set_soe_ls_mode(LineSearchModes mode);
    void set_soe_ls_mode(const std::string &str);

    void set_best_criteria(BestCriteriaModes mode);
    void set_best_criteria(const std::string &str);

#ifdef USE_ACCELERATE_SPARSE
    void set_accel_pivot_tolerance(double tol);
    void set_accel_zero_tolerance(double tol);
#endif

    // --- Named configuration presets ---
    // Assigns exactly the nine globalization fields (acceptance_strategy_,
    // merit_penalty_rule_, barrier_governor_, never_monotone_,
    // restoration_mode_, inertia_mode_, max_soc_, ls_extended_iters_,
    // watchdog_) per the named preset; every other Settings field (tolerances,
    // iteration caps, QP parameters, ...) is left untouched. Throws
    // std::invalid_argument (listing every valid name) for an unrecognized
    // name. The preset table -- field values, evidence-of-record citations,
    // and the name list this error message dispatches against -- lives in
    // detail/solvers/psiopt_presets.h. The Python binding's docstring repeats
    // the preset names by hand; a Python test pins it against this table.
    // Defined in psiopt_settings.cpp alongside the other Settings-only logic.
    void apply_preset(std::string_view name);

    // --- Callback methods ---
    /// Installs the per-iteration early callback. The vectors and matrix it
    /// receives are in the SOLVER's variable space, which is narrower than the
    /// caller's on a problem with bound-fixed variables -- see the space note on
    /// EarlyCallBackType above.
    void set_early_callback(const EarlyCallBackType &f) {
        this->early_callback_enabled_ = true;
        this->early_callback_ = f;
    }
    void disable_early_callback() { this->early_callback_enabled_ = false; }
    /// Installs the per-iteration late callback. Same variable-space caveat as
    /// set_early_callback -- see the note on LateCallBackType above.
    void set_late_callback(const LateCallBackType &f) {
        this->late_callback_enabled_ = true;
        this->late_callback_ = f;
    }
    void disable_late_callback() { this->late_callback_enabled_ = false; }

    // --- Printing ---
    static void print_header() { fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65); }

  private:
    // Test access: these unit tests verify which concrete acceptance strategy
    // the settings dispatch constructs; befriended narrowly instead of
    // exposing a public rebuild hook.
    friend class ::RecoveryDispatchGate_FunnelSelectionConstructsFunnelAcceptance_Test;
    friend class ::RecoveryDispatchGate_FilterSelectionConstructsFilterAcceptance_Test;
    friend class ::RecoveryDispatchGate_MonitoredSelectionConstructsMonitoredGovernor_Test;
    friend class ::RecoveryDispatchGate_MeritPenaltyRuleSelectionReachesTheStrategy_Test;
    friend class ::FeasibilitySwitch_ProximalSwitchConstructsRestorationAndWrapsRecovery_Test;
    friend class ::FeasibilitySwitch_OffModeConstructsNoRestoration_Test;
    friend class ::FeasibilitySwitch_FilterSeedsRestorationConstraintTol_Test;
    friend class ::NestedSeamHarness;
    friend class ::NestedSeamIneqHarness;
    friend class ::NestedLifecycleHarness;
    friend class ::DivergencePersistenceHarness;
    friend class ::SocGenericHarness;
    friend class ::NativeBoundsHarness;

    Settings settings_;
    SolveResult result_;
    EvalErrorLog eval_error_log_;
    std::shared_ptr<NonLinearProgram> nlp_;

    // Classic merit line-search acceptance, extracted from the former
    // ls_impl/ls_lang/ls_l1/ls_auglang bodies (now ClassicMeritAcceptance). Held
    // through the AcceptanceStrategy interface (forward-declared above); rebuilt
    // by rebuild_globalization_components() wired to a SolverContext view of
    // this solver. Never null once run_phase_sequence has run it once, which
    // every solve entry point guarantees before any iteration.
    std::unique_ptr<AcceptanceStrategy> acceptance_;

    // Step-length globalization mechanism, extracted from the
    // former max_primal_dual_step/max_step_to_boundary bodies (now
    // BacktrackingLineSearch). Held through the GlobalizationMechanism interface
    // (forward-declared above); rebuilt by rebuild_globalization_components()
    // alongside acceptance_. Never null once run_phase_sequence has run it
    // once, which every solve entry point guarantees before any iteration.
    std::unique_ptr<GlobalizationMechanism> mechanism_;

    // Barrier-parameter governor, extracted from the former
    // PROBE/LOQO barmode switch + loqo_mu/mpc_mu bodies (now
    // ClassicAdaptiveGovernor). Held through the BarrierGovernor interface
    // (forward-declared above); rebuilt by rebuild_globalization_components()
    // alongside acceptance_/mechanism_. Never null once run_phase_sequence has
    // run it once, which every solve entry point guarantees before any
    // iteration.
    std::unique_ptr<BarrierGovernor> governor_;

    // Post-rejection recovery chain, extracted-as-a-hook (no
    // prior code existed for this — this hook point is wired with a no-op
    // implementation). Held through the RecoveryChain interface
    // (forward-declared above); rebuilt by rebuild_globalization_components()
    // alongside acceptance_/mechanism_/governor_. Never null once
    // run_phase_sequence has run it once, which every solve entry point
    // guarantees before any iteration. With max_soc_ == 0, ls_extended_iters_
    // == 0, and watchdog_ == false (all defaults), rebuild_globalization_
    // components() installs plain NoopRecovery, which always returns
    // kAcceptAsIs and is stateless — bit-identical to pre-recovery-chain
    // behavior. Opt in to any subset of SocRecovery/ExtendedBacktrackRecovery
    // (composed in that order by ChainedRecovery) and WatchdogRecovery (an
    // outer decorator over whatever chain results) via the corresponding
    // Settings fields — see globalization/soc.h and globalization/watchdog.h.
    std::unique_ptr<RecoveryChain> recovery_;

    // Optional feasibility-restoration mode-switch. Held through the
    // RestorationStrategy interface (forward-declared above). Unlike
    // acceptance_/mechanism_/governor_/recovery_ this is NOT always constructed:
    // rebuild_globalization_components() leaves it null unless restoration_mode_
    // != off, in which case it holds a ProximalSwitchRestoration
    // (restoration_mode_ == proximal_switch) or a NestedL1Restoration
    // (restoration_mode_ == l1_nested), and FeasibilitySwitchRecovery is
    // wrapped as the outermost recovery link either way. On the default path
    // (off) it stays null and every restoration branch in eval_nlp / the
    // classic+generic trial-eval seams / alg_impl guards on
    // `restoration_ != nullptr` (or `ctx.restoration_ != nullptr`) and is
    // provably dead. run_phase_sequence() resets it (when present) at each phase
    // boundary alongside the other components, and collects its diagnostics into
    // SolveResult::last_feas_rest_entries_/last_feas_rest_iters_.
    std::unique_ptr<RestorationStrategy> restoration_;

    // (Re)builds acceptance_/mechanism_/governor_/recovery_ from the current
    // Settings. Called once per run_phase_sequence(), right after the
    // variable-treatment configuration and before the first phase (i.e.
    // once per solve invocation — optimize()/solve()/etc. all route through
    // it), NOT from set_nlp(): construction-time knobs (acceptance_strategy,
    // max_soc, ls_extended_iters, watchdog, merit_penalty_rule) must take
    // effect on the very next solve even without a re-transcription in
    // between, matching every other Settings field's live-at-next-solve
    // semantics. See psiopt.cpp's definition for the neutrality argument on
    // the default (all-off) path.
    void rebuild_globalization_components();

    // QP parameter setup — called automatically by set_nlp()
    void set_qp_params();

    // Re-reads the problem dimensions from the NLP into the members below.
    // Called by set_nlp, and again at solve entry whenever the fixed-variable
    // configuration changed the size of the problem the solver factorizes.
    void refresh_nlp_dimensions();

    // --- Problem dimensions ---
    // primal_vars_ is the SOLVER's primal width: the NLP's variable count minus
    // the variables the fixed-variable treatment eliminated. Every vector this
    // solver sizes, every KKTVector segment and the KKT matrix are in that
    // space. full_primal_vars_ is the problem's own count -- the width of an
    // initial guess and of the returned solution, and the only one a caller
    // ever sees. They are equal unless a variable is bound-fixed.
    int full_primal_vars_ = 0;
    int primal_vars_ = 0;
    int slack_vars_ = 0;
    int equal_cons_ = 0;
    int inequal_cons_ = 0;
    int kkt_dim_ = 0;

    // --- Reusable per-iteration scratch buffers (avoid per-call heap allocation) ---
    // complementarity()/barrier_hessian() are only ever invoked serially from
    // alg_impl's single-threaded control loop for this PSIOPT instance (no
    // partition-level concurrency at this level -- that only happens inside
    // NLP eval calls). Sized to inequal_cons_/slack_vars_ (resize-in-place;
    // a no-op once the size matches, which it does for the lifetime of a solve).
    mutable Eigen::VectorXd stli_scratch_; ///< @internal complementarity() S*LI buffer.
    Eigen::VectorXd hp_scratch_; ///< @internal barrier_hessian() LI.cwiseQuotient(S) buffer.

    // alg_impl's return_best_ path (off by default, settings_.return_best_) copies
    // the full XSL/RHS iterate on every improving iteration. Hoisted so repeated
    // alg_impl calls (one per phase in run_phase_sequence) reuse the same backing
    // store instead of starting from an empty vector each time; resize-on-assign
    // is then a no-op once kkt_dim_ is stable across a solve.
    Eigen::VectorXd best_xsl_scratch_; ///< @internal alg_impl() return_best_ XSL snapshot.
    Eigen::VectorXd best_rhs_scratch_; ///< @internal alg_impl() return_best_ RHS snapshot.

    // Nested feasibility-restoration eval-seam scratch (all dead unless a nested
    // restoration strategy is active). The seam runs in the per-iteration hot
    // path, so these back the condensed-elastic outputs without per-call heap
    // allocation, following the *_scratch_ discipline above: resize-on-assign is
    // a no-op once dims are stable across a solve. resto_pdiag_scratch_ holds the
    // proximal Hessian diagonal η(μ)·D_R² (primal_vars_); resto_epiv_/ipiv_scratch_
    // hold the NEGATED constraint-row pivots scattered into the KKT (y,y) blocks
    // (equal_cons_/inequal_cons_); resto_ec_/ic_scratch_ copy the raw constraint
    // residuals out before the condensed r̃ overwrites the RHS segments in place.
    Eigen::VectorXd resto_pdiag_scratch_;
    Eigen::VectorXd resto_epiv_scratch_;
    Eigen::VectorXd resto_ipiv_scratch_;
    Eigen::VectorXd resto_ec_scratch_;
    Eigen::VectorXd resto_ic_scratch_;

    // Nested feasibility-restoration lifecycle state (all dead unless a nested
    // restoration strategy is active). stashed_mu_ holds the outer barrier
    // parameter captured at entry; the governor drives a fresh in-phase schedule
    // in between, and the multiplier re-entry restores it on exit. resto_first_
    // iter_ guards the first phase iteration (take at least one step before any
    // exit test fires). resto_theta_orig_prev_ carries the previous phase
    // iteration's original-problem infeasibility for the per-iteration κ_resto
    // ratchet (seeded at entry with the entry-point value, ratcheted each
    // iteration — NOT frozen at entry). resto_dz_scratch_ backs the re-entry
    // slack-multiplier Newton step, following the *_scratch_ no-per-call-alloc
    // discipline. This state obeys the same reset invariant as the acceptance
    // stash: a μ-event reset() mid-phase does NOT touch it (only the phase-
    // boundary reset in run_phase_sequence() clears it), so the stashed outer μ
    // survives a barrier subproblem restart inside the phase.
    double stashed_mu_ = 0.0;
    bool resto_first_iter_ = false;
    double resto_theta_orig_prev_ = 0.0;
    Eigen::VectorXd resto_dz_scratch_;
    // The bound families' re-centring steps at that same return, backing the two
    // sides separately because they index different lists. Deliberately NOT
    // bound_duals_.dz_*: that pair is the ITERATE's Newton direction, consumed by
    // the commit and by the fraction-to-boundary rule, and the restoration
    // return is a different event that applies no dz — see the two-event note on
    // bound_duals_ above. Empty unless a nested restoration phase returns on a
    // problem with variable bounds.
    Eigen::VectorXd resto_bound_dz_lower_scratch_;
    Eigen::VectorXd resto_bound_dz_upper_scratch_;

    // --- Native primal variable bounds (all inert on a problem without any) ---
    //
    // bounds_ points at the NonLinearProgram's classification of the finite
    // variable bounds this solve must keep barrier terms for, in the solver's
    // REDUCED index space. It is set ONLY on the configuration success path in
    // run_phase_sequence() and ONLY when the set is non-empty: a configuration
    // that threw leaves a rejected classification behind on the NLP, so the
    // pointer is cleared before the configuration attempt and re-read after it,
    // and set_nlp()/release() clear it too. Null therefore means "this solve has
    // no variable-bound barrier terms", which every bound branch in this class
    // and in the globalization components tests -- and which is what makes the
    // KKT assembly on such a problem byte-identical to the pre-bounds solver.
    const BoundSet *bounds_ = nullptr;
    // The bound multipliers and their Newton step, index-aligned to bounds_'s
    // two lists. Iterate state, so solver-owned rather than NLP-owned; sized by
    // the interior push at solve entry and empty whenever bounds_ is null.
    //
    // TWO EVENTS write z, and only two. It MOVES ALONG dz at exactly one site,
    // the iterate commit (apply_bound_dual_step), once per committed iterate and
    // with the κ_Σ clip against the new x. It is RE-ANCHORED at exactly one
    // other, the nested restoration return (exit_feasibility_restoration_nested),
    // which applies no dz and moves no x — it re-centres z on the stashed outer
    // barrier parameter because the phase it is returning from ran on a
    // different one. The two are different event classes: different formula,
    // different damping, different trigger. The slack multipliers have always
    // lived under exactly this discipline (the same restoration return rewrites
    // them); the bound family matches it rather than being the exception.
    BoundDualState bound_duals_;

    // Bound scratch, following the *_scratch_ no-per-call-allocation discipline
    // (resize-on-assign is a no-op once dims are stable across a solve); all
    // three stay empty on a problem without variable bounds.
    //
    // bound_sigma_scratch_ backs the primal-diagonal base vector that carries
    // the condensed bound curvature into the KKT assembly.
    // bound_resid_scratch_ backs the z-form dual-infeasibility block the
    // convergence account reads (mutable: the residual accessor is const).
    // bound_grad_scratch_ holds the primal RHS block as it stands BEFORE the
    // mu-form Newton terms are installed, so the restore after the step is an
    // exact copy back rather than an add-then-subtract round trip.
    Eigen::VectorXd bound_sigma_scratch_;
    mutable Eigen::VectorXd bound_resid_scratch_;
    Eigen::VectorXd bound_grad_scratch_;

    // One-shot guard for the second-level elastic re-centering fallback (nested l1
    // restoration only, disclosure (f) in l1_restoration.h). Set true when an
    // in-phase ladder-exhausted rejection re-centers the elastic pairs instead of
    // taking the failed step; a second consecutive ladder exhaustion while set
    // falls through to accept-as-is (no re-center loop). Cleared on any accepted
    // step and re-armed at each phase entry / phase-boundary reset. Dead unless a
    // nested restoration phase is active.
    bool resto_recentered_ = false;

    // --- KKT solver ---
#ifdef USE_ACCELERATE_SPARSE
    Eigen::AccelerateLDLTTPP<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper> kkt_sol_;
#else
    Eigen::PardisoLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper> kkt_sol_;
#endif
    bool qp_analyzed_ = false;

    // --- Callbacks ---
    EarlyCallBackType early_callback_;
    bool early_callback_enabled_ = false;
    LateCallBackType late_callback_;
    bool late_callback_enabled_ = false;

    // KKTVector — the compound-KKT segment view — now lives in
    // detail/solvers/kkt_vector.h as tycho::solvers::KKTVector, shared with the
    // globalization components (each of which used to carry a verbatim copy,
    // since the type was private here and they are non-member, non-friend).

    /// Create a KKTVector view over a VectorXd using this solver's dimensions.
    KKTVector kkt_view(Eigen::VectorXd &v) {
        return KKTVector(v, primal_vars_, slack_vars_, equal_cons_, inequal_cons_);
    }

    // --- Phase sequence ---
    // Describes one phase in a multi-phase solve strategy. run_phase_sequence
    // executes steps in order, skipping conditional steps when an earlier phase
    // already converged, and re-initializing the KKT system between phases.
    struct PhaseStep {
        AlgorithmModes alg_mode_;
        BarrierModes bar_mode_;
        LineSearchModes ls_mode_;
        const char *label_;
        bool conditional_ = false; // skip if converge_flag_ == CONVERGED
                                   // (still runs on ACCEPTABLE / NOTCONVERGED;
                                   // DIVERGING breaks the loop before reaching this)
    };

    Eigen::VectorXd run_phase_sequence(const Eigen::VectorXd &x,
                                       std::initializer_list<PhaseStep> steps);

    // --- Core algorithm (defined in psiopt.cpp) ---
    Eigen::VectorXd alg_impl(AlgorithmModes algmode, BarrierModes barmode, LineSearchModes lsmode,
                             double obj_scale, double MuI, Eigen::Ref<Eigen::VectorXd> xsl);

    Eigen::VectorXd init_impl(const Eigen::VectorXd &x, double Mu, bool docompute);

    // --- Line search ---
    // The classic merit line search (former ls_impl/ls_lang/ls_l1/ls_auglang and
    // their eval_trial_point_occ/compute_penalties/secondary_accept helpers) was
    // extracted verbatim into ClassicMeritAcceptance; the
    // fraction-to-boundary step-length (former max_primal_dual_step/
    // max_step_to_boundary) was extracted verbatim into BacktrackingLineSearch.
    // alg_impl now drives both through mechanism_->compute_step
    // (which fuses the step scaling and acceptance backtrack).

    // --- KKT factorization (defined in psiopt.cpp) ---
    // `finalpert` is the last perturbation DELTA applied via Perturb() -- this is
    // the exact value alg_impl's Hpert0 warm-start consumes today and must keep
    // consuming byte-identically (see the comment at its call site). `cumpert` is
    // a separate, display-only accumulator: the running SUM of every Perturb()
    // delta applied during this call (i.e. the actual total added to the KKT
    // diagonal), used only for the HPert iteration-table column. Neither
    // `finalpert` nor any control-flow decision in factor_impl reads `cumpert`.
    // `base_prox` and `dual_shift` are the proximal-regularization base shifts
    // (ρ_k on the Hessian diagonal, δ_c on the constraint-row diagonals); both
    // are read only when inertia_mode_ == proximal_regularization and default to
    // 0.0, so the classic path is byte-identical regardless of their values.
    int factor_impl(bool docompute, bool ZFac, double ipurt, double incpurt0, double incpurt,
                    double &finalpert, double &cumpert, double base_prox = 0.0,
                    double dual_shift = 0.0);

    bool claim_kkt_analysis();

    void ensure_solver_initialized();

    // --- Barrier math helpers (defined in psiopt.cpp) ---
    void apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> FXI) const;
    // max_step_to_boundary was extracted verbatim into BacktrackingLineSearch;
    // it is now a private helper of that mechanism.
    // The complementarity account mu is driven by and barr_inf_ reports: the
    // inequality slack/multiplier pairs, plus -- when the problem declares
    // variable bounds -- the bound pairs (x-l)*z_L and (u-x)*z_U, which is why
    // it takes the primal block `X` alongside the slack and multiplier blocks.
    void complementarity(Eigen::Ref<Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> S,
                         Eigen::Ref<Eigen::VectorXd> LI, double &avgcomp, double &mincomp,
                         double &maxcomp) const;

    // How many pairs complementarity() reduced into its aggregates, given the
    // slack block length it was handed: the slack/multiplier pairs plus one per
    // finite variable bound. This is the weight the union average carries, and
    // therefore the base_count any FURTHER fold-in (augment_complementarity_
    // nested) has to re-weight against -- that helper reconstructs the base sum
    // as avgcomp*base_count, so a count that omitted the bound pairs would
    // reconstruct the wrong sum. Returns the slack count unchanged off the bound
    // path.
    int complementarity_pair_count(int slack_count) const;
    // Folds an active nested restoration phase's elastic complementarity pairs
    // into complementarity()'s aggregates. base_count is the number of original
    // slack/multiplier pairs already reduced into avgcomp (so their sum can be
    // reconstructed as avgcomp*base_count and re-averaged over the union). A pure
    // no-op unless a nested restoration is active — the aggregates are returned
    // untouched off that path, so the default/proximal barrier machinery is
    // byte-identical. Only ever combines separately-computed aggregates (min of
    // mins, max of maxes, count-weighted average); it never re-reduces the
    // original pairs, so complementarity()'s reduction ordering is preserved.
    void augment_complementarity_nested(double &avgcomp, double &mincomp, double &maxcomp,
                                        int base_count) const;
    void barrier_hessian(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                         Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu);
    // --- Native variable-bound helpers (defined in psiopt.cpp) ---
    // Every one of these is a no-op when bounds_ is null.

    // Projects `x` into the strict interior of the recorded bounds and seeds the
    // bound multipliers there. Per bounded variable the push away from a bound is
    // p = bound_push_ * max(1, |bound|), additionally capped at
    // bound_interval_push_ * (upper - lower) when the variable is two-sided
    // (Ipopt's kappa1/kappa2 rule); the lower push is applied before the upper,
    // and with bound_interval_push_ below one half the two can never cross. A
    // guess at or outside a bound is projected, never rejected. The multipliers
    // are then seeded at min(kBoundMultInitCap, mu0 / distance) -- after the
    // push, so the distance is safely interior. Runs once per solve, at entry,
    // after the reduced gather and before any evaluation.
    void push_initial_point_interior(EigenRef<Eigen::VectorXd> x, double mu0);

    // Bound-multiplier Newton step from the UNSCALED primal step `dx`:
    // dz_L = mu*(X-L)^-1 e - z_L - Sigma_L*dx and, with the sign the upper-bound
    // row carries, dz_U = mu*(U-X)^-1 e - z_U + Sigma_U*dx. Must be called on the
    // raw KKT solution, before the fraction-to-boundary rule scales it.
    void compute_bound_dual_direction(ConstEigenRef<Eigen::VectorXd> x,
                                      ConstEigenRef<Eigen::VectorXd> dx, double mu);

    // Commits the bound multipliers for an accepted iterate: z += alphad*dz,
    // then the kappa_sigma safeguard clamps each into
    // [mu_clip/(kKappaSigma*d), kKappaSigma*mu_clip/d] for the distance d
    // measured at the NEW x. Exactly one call per committed iterate; `xsl_new`
    // is the already-committed iterate.
    //
    // `monotone_mu` selects which barrier parameter the clamp is taken at,
    // transcribing Ipopt's correct_bound_multiplier: under a MONOTONE schedule
    // the clamp uses the barrier parameter itself (`mu`), and under a FREE-mu
    // schedule it uses the average complementarity at the new point, capped at
    // kFreeModeClipMuCap. The two differ because a free-mode barrier parameter
    // is an oracle's proposal for the NEXT step rather than a description of
    // where the iterate currently sits, and it is the latter the safeguard
    // needs. PSIOPT's classic_adaptive governor is the free-mu case, the
    // monitored governor reports its live mode, and a phase with no inequality
    // constraints runs no governor at all and so holds mu fixed.
    void apply_bound_dual_step(double alphad, KKTVector &xsl_new, double mu, bool monotone_mu);

    // Accumulates the condensed bound curvature onto a primal-diagonal base
    // vector. The two callers that already have a base vector in hand call this
    // directly; install_primal_diags_with_sigma() is the scalar-base form, which
    // collapses to a plain set_primal_diags(base) when there are no bounds.
    void add_bound_sigma(ConstEigenRef<Eigen::VectorXd> x, EigenRef<Eigen::VectorXd> diag) const;
    void install_primal_diags_with_sigma(ConstEigenRef<Eigen::VectorXd> x, double base);

    // The dual infeasibility whose infinity norm is the solver's kkt_inf_. Off
    // the bound path this is exactly the base block's norm, as it always was;
    // with bounds it is that block plus the z-FORM terms (-z_L + z_U), built in
    // scratch. It is deliberately NOT accumulated into the RHS itself: the same
    // primal block is the condensed Newton right-hand side, which carries the
    // mu-form instead -- see the staging comments in alg_impl().
    //
    // `prim_base` is the BASE-form primal stationarity block (grad f + J'lambda)
    // for the point being measured, passed explicitly rather than read off a
    // KKTVector because the live RHS's own block is staged in the mu-form for
    // part of each iteration; a caller inside that bracket passes the snapshot.
    double dual_infeasibility_inf(ConstEigenRef<Eigen::VectorXd> prim_base) const;

    // loqo_mu / mpc_mu were extracted verbatim into ClassicAdaptiveGovernor
    // (src/solvers/psiopt_globalization.cpp); the barrier-
    // parameter update now runs through governor_->update_barrier(). The
    // barrier_objective()/barrier_gradient() helpers formerly declared here were
    // dead after that extraction (PSIOPT no longer called them) and have been
    // removed; ClassicMeritAcceptance and ClassicAdaptiveGovernor each carry
    // their own copies. complementarity() STAYS — it is still called from the
    // evaluate stage (its maxcomp output feeds converge_check's barr_inf_).

    // --- NLP eval dispatch methods (defined in psiopt.cpp) ---
    // The four wrappers below differ only in which NonLinearProgram entry point
    // they call; the segment expressions that slice XSL/GX/AGXS_FX into the
    // compound [primals | slacks | eq | iq] layout are written once, here. `fn` is
    // a pointer to the NonLinearProgram member to invoke. Defined in psiopt.cpp,
    // its only translation unit.
    template <class Fn>
    void eval_dispatch(Fn fn, double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                       EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                       Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    void eval_kkt(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void eval_kkt_no(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                     EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                     Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void eval_aug(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);
    void eval_soe(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val, EigenRef<VectorXd> GX,
                  EigenRef<VectorXd> AGXS_FX, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat);

    // `mu` is the live phase barrier parameter. It is consulted only by the
    // nested feasibility-restoration branch (which recomputes its proximity
    // weight, pivots, and condensed residuals from the live μ every evaluation);
    // every other mode ignores it, so the default and proximal-switch paths are
    // unaffected by its value.
    void eval_nlp(AlgorithmModes algmode, double obj_scale, ConstEigenRef<VectorXd> XSL,
                  double &val, EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat, double mu);

    // --- Feasibility-restoration exit measures (defined in psiopt.cpp) ---
    // Shared by every restoration exit/teardown site (the two continuing-exit
    // arms, the in-loop locally-infeasible break, and the post-loop teardown).
    // While restoration is active, the loop's own prim_obj_ is φ_prox (the
    // proximal objective substituted by the eval seam) — never valid outside
    // restoration, since the OPTIMALITY filter/funnel's accumulated pairs are
    // all true-objective-scale (see the cross-phase pair-incomparability
    // disclosure in globalization/filter_acceptance.h). This helper re-evaluates the TRUE
    // objective once at the live primals so every exit site hands
    // notify_switch_to_optimality (and, ultimately, obj_val_) a measures
    // triple in the same scale as the filter/funnel it is augmenting into.
    ProgressMeasures build_restoration_exit_measures(double obj_scale, double infeasibility,
                                                     ConstEigenRef<VectorXd> primals,
                                                     double barr_obj);

    // --- Feasibility-restoration lifecycle (defined in psiopt.cpp) ---
    // Shared entry orchestration for the kSwitchToFeasibility case. Builds the
    // (θ,f) entry measures from the current RHS/primals, then dispatches on the
    // strategy family: the proximal switch takes enter_restoration; the nested
    // l1 phase takes enter_nested (with the current equality/inequality residual
    // vectors) and additionally stashes the outer μ, sets μ ← entry_mu(), resets
    // the governor for a fresh in-phase barrier schedule, and applies the
    // verified entry multiplier init (equality constraint multipliers ← 0; the
    // slack/bound multipliers clamped to min(ρ, current)). Both families then
    // notify the acceptance strategy of the switch and reset the recovery chain.
    // Passed the raw XSL/RHS blocks (KKTVector views are rebuilt inside) so it is
    // directly drivable from a friend test harness. `mu` is updated in place.
    void enter_feasibility_restoration(Eigen::VectorXd &XSL, Eigen::VectorXd &RHS, double prim_obj,
                                       double barr_obj, double &mu);

    // The restoration-entry dispatch, in the ONE order every entry site uses:
    // record `theta` as the stall detector's handback yardstick, enter
    // restoration, then re-arm the stall window. Ordering matters because
    // enter_feasibility_restoration takes XSL/RHS by non-const reference; it does
    // not write RHS today, but nothing in its signature says so, and the four
    // former open-coded sites did not agree on whether note_dispatch ran before
    // or after it. `theta` stays a parameter: each site already has the
    // constraint violation it needs in hand, and computing it here instead would
    // add a reduction at two of them.
    void dispatch_restoration_entry(Eigen::VectorXd &XSL, Eigen::VectorXd &RHS, double prim_obj,
                                    double barr_obj, double &mu, double theta,
                                    FeasibilityStallDetector &feas_stall);

    // The restoration EXIT protocol, in the one order every exit site must use:
    // (optionally restore the stashed outer μ and reset the governor, which only
    // a nested phase ever needs), exit_restoration(), notify the acceptance
    // strategy of the switch back to optimality, reset the recovery chain.
    //
    // The order is load-bearing. exit_restoration() flips is_active() false, so
    // any μ/governor work that belongs to the phase must precede it.
    // notify_switch_to_optimality augments `measures` into the restored OPTIMALITY
    // filter/funnel, whose accumulated pairs are all true-objective-scale — so
    // callers build `measures` through build_restoration_exit_measures() rather
    // than passing the loop's own prim_obj (which is φ_prox/φ_l1 while active).
    // The recovery-chain reset runs last and exactly once per transition: the
    // watchdog's objective-scale-bound snapshot and counters must not survive back
    // into the optimality phase.
    void leave_restoration(const ProgressMeasures &measures, bool restore_stashed_mu, double &mu);

    // The nested phase's multiplier re-entry sequence — shared byte-for-byte by
    // the κ_resto ratchet exit and the near-feasible stall exit (Ipopt
    // MinC_1NrmRestorationPhase::PerformRestoration, strict order): (1) keep the
    // phase's final x/s; (2) slack-multiplier Newton complementarity step under
    // the STASHED outer μ, damped by the dual fraction-to-boundary rule; (3) if
    // max|z| over ALL inequality multipliers exceeds kBoundMultResetThreshold,
    // reset every inequality multiplier to 1; (4) equality constraint
    // multipliers ← 0; (5) restore the stashed outer μ, reset the governor,
    // exit_restoration, notify the acceptance strategy of the switch back to
    // optimality (with true-objective exit measures), reset the recovery chain.
    // `theta_orig` is the current original-problem infeasibility (∞-norm),
    // carried into the exit measures. `mu` is restored in place.
    //
    // SCOPE: steps (2) and (3) reach EVERY bound-multiplier family the solver
    // carries — the inequality (slack) multipliers and, when the problem
    // declares variable bounds, both sides of those — matching Ipopt's
    // PerformRestoration, which applies its ComputeBoundMultiplierStep to all
    // four of its families under ONE shared dual fraction-to-boundary damping
    // and takes its reset-threshold max over all four. The shared damping is
    // the detail worth naming: a per-family fraction is the plausible wrong
    // implementation, and it is what the exit's unit pin exists to catch.
    //
    // This is the second of the two events that write the bound multipliers,
    // and the only one that applies no dz and moves no x — see the two-event
    // note on bound_duals_ above.
    void exit_feasibility_restoration_nested(Eigen::VectorXd &XSL, double obj_scale,
                                             double theta_orig, double barr_obj, double &mu);

    // Per-iteration κ_resto ratchet test for the nested phase: the current
    // original-problem infeasibility must fall to at most max(kKappaResto ·
    // previous-iteration infeasibility, econ_tol_) (Ipopt RestoConvCheck's
    // orig_inf_pr_max, single-tolerance floor). Reads resto_theta_orig_prev_
    // (seeded at entry, ratcheted each phase iteration). Defined in psiopt.cpp so
    // the kKappaResto constant (globalization/acceptance_strategy.h) stays out of
    // this header's include set.
    bool resto_ratchet_passes(double theta_orig) const;

    // ‖c‖₁ over a KKT vector's constraint block — the L1 constraint violation the
    // restoration entry guards, the proximal exit test and the stall detector all
    // measure. One home for the reduction, which was written at seven sites in two
    // spellings of the same expression (v.all_cons() is exactly the
    // tail(equal_cons_ + inequal_cons_) the other spelling wrote out).
    double constraint_violation_l1(KKTVector &v) const;

    // Original-problem infeasibility (∞-norm) for an active NESTED restoration
    // phase, taken from the raw equality/inequality residuals the eval seam saves
    // each active iteration (the RHS constraint rows carry the condensed r̃ by
    // then, so they are not a valid source). Two separate Eigen reductions,
    // deliberately not fused. Meaningless off the nested path — every caller is
    // inside a nested-active branch.
    double original_infeasibility_inf() const;

    // Second-level elastic re-centering fallback for the nested l1 phase
    // (disclosure (f) in l1_restoration.h). Invoked by alg_impl's kAcceptAsIs case
    // when an in-phase line search exhausts the recovery ladder (a nested phase is
    // active and no recovery link resolved the rejection). Re-centers the elastic
    // pairs in closed form at the current phase μ from the raw residuals held in
    // resto_ec_/ic_scratch_ (this iteration's eval seam), INSTEAD of taking the
    // failed step. One-shot per consecutive-failure run: returns true and consumes
    // the resto_recentered_ budget on the first call; returns false (fall through
    // to accept-as-is) while the flag is still set. The flag re-arms on any
    // accepted step and at each phase entry. Reachable only with restoration_
    // non-null, active, and nested (the call site gates on nested_active).
    bool try_recenter_elastics(double mu);

    // Primal-dual system error at barrier parameter `mu`: the ∞-norm of the full
    // KKT residual — primal stationarity (rhs.prim_grad, the Lagrangian gradient
    // as assembled for the current iterate), primal infeasibility (equality and
    // slack-completed inequality residuals), and the complementarity deviation
    // max|s·z − μ| — as one scalar. Maps Ipopt's primal_dual_system_error(μ)
    // (coin-or/Ipopt 72a29c9, src/Algorithm/IpBacktrackingLineSearch.cpp
    // TrySoftRestoStep) onto this solver's single unscaled max-norm KKT measure.
    // Read-only; the caller passes vectors already populated the same way the
    // main loop populates the current iterate's RHS (stationarity including the
    // objective/barrier gradient contribution, inequality residual slack-
    // completed). Used only by the nested soft feasibility pre-stage.
    //
    // The stationarity term is the z-FORM dual infeasibility, matching Ipopt's
    // error, which norms the undamped Lagrangian gradient at whichever point it
    // measures. `prim_base` carries that point's BASE primal block for the same
    // reason dual_infeasibility_inf takes one: the comparison this feeds comes
    // from two points, and both must be measured in the same form or the
    // reduction test acquires a direction.
    double primal_dual_error(KKTVector &xsl, KKTVector &rhs,
                             ConstEigenRef<Eigen::VectorXd> prim_base, double mu) const;

    // Nested soft feasibility pre-stage trial (defined in psiopt.cpp). Forms the
    // full fraction-to-boundary trial point XSL + DXSL (DXSL already carries the
    // fraction-to-boundary scaling from compute_step), evaluates the original
    // problem there (into the caller-supplied XSL2/RHS2/GX scratch), and returns
    // whether its primal-dual error is at most kSoftRestoPdErrorReductionFactor
    // times the current point's. A true return means the soft step is accepted
    // (alg_impl takes the full step and stays in the pre-stage); a false return
    // means alg_impl escalates to the full restoration switch. Dead on the
    // default path (only reached with a nested restoration strategy configured,
    // via the kSoftFeasibilityStep recovery action).
    bool try_soft_feasibility_step(AlgorithmModes algmode, double obj_scale, double mu,
                                   Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL,
                                   Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
                                   Eigen::VectorXd &RHS2, Eigen::VectorXd &GX);

    // --- Convergence and stepping ---
    // The residual formulas shared by the pre-factorization early
    // convergence check and the post-line-search fill_iter_info() call live here
    // ONCE, so neither call site can drift out of sync. fill_residual_info() sets
    // every IterateInfo field derivable from rhs/xsl alone (valid immediately after
    // eval + the barrier/complementarity block, before any factorization). It
    // deliberately does NOT set barr_obj_/mu_ (only settled once the barrier-
    // parameter update runs, later this iteration) or p_pivots_ (kkt_sol_.ppivs(),
    // which only reflects a real value once this iteration's factorization has
    // actually run) -- see the definition in psiopt.cpp for the full rationale.
    void fill_residual_info(KKTVector &xsl, KKTVector &rhs, double pobj, IterateInfo &iter) const;
    void fill_iter_info(KKTVector &xsl, KKTVector &rhs, double pobj, double bobj, double mu,
                        IterateInfo &iter) const;
    ConvergenceFlags converge_check(std::vector<IterateInfo> &iters);

    // Best-iterate bookkeeping for the return_best_ path (off by default). Scores
    // `iter` under best_criteria_ and, when it ties or beats the incumbent (or is
    // the phase's first iterate), snapshots XSL/RHS into best_xsl_scratch_/
    // best_rhs_scratch_ and records the criterion value and iteration index. The
    // return_best_ / restoration-active guard stays at the call sites, which
    // differ in why they are reached; only the scoring and snapshot live here.
    void track_best_iterate(const IterateInfo &iter, int i, const VectorXd &XSL,
                            const VectorXd &RHS, double &BestCriteriaVal, int &BestIter);
    // max_primal_dual_step was extracted verbatim into BacktrackingLineSearch;
    // alg_impl now drives it through mechanism_ (fused into
    // compute_step on the main path, and via the public method at the PROBE
    // predictor call site).

    // --- Printing methods ---
    static void print_psiopt();
    void print_settings();
    void print_stats();
    void print_last_iterate(const std::vector<IterateInfo> &iters);
    void print_beginning(std::string_view msg) const;
    void print_finished(std::string_view msg) const;
    void print_exit_stats(ConvergenceFlags ExitCode, const IterateInfo &last, int iternum,
                          double tottime, double nlptime, double qptime, double printtime);
    void print_timing_summary();
    static fmt::text_style calculate_color(double val, double targ, double acc);
};

} // namespace tycho::solvers
