// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
//   - Configuration fields grouped into Settings struct
//   - Converted from struct to class with public/private access sections
// =============================================================================

#pragma once
#include <algorithm>
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

#include "tycho/detail/solvers/iterate_info.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/psiopt_fwd.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/get_core_count.h"

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
class FeasibilitySwitch_ProximalSwitchConstructsRestorationAndWrapsRecovery_Test;
class FeasibilitySwitch_OffModeConstructsNoRestoration_Test;
class FeasibilitySwitch_FilterSeedsRestorationConstraintTol_Test;
// Test harness for the nested feasibility-restoration eval/step seam: reaches
// private eval_nlp / alg_impl / restoration_ / dims to drive the seam directly.
class NestedSeamHarness;
// Test harness for the nested feasibility-restoration LIFECYCLE (entry
// orchestration, exit ratchet, multiplier re-entry): reaches the private
// enter_/exit_feasibility_restoration helpers, the stashed-μ / ratchet state,
// restoration_, and alg_impl to drive the whole phase end-to-end.
class NestedLifecycleHarness;

namespace tycho::solvers {

// Pull root-namespace Eigen type aliases into tycho::solvers so that PSIOPT
// member declarations (EigenRef<VectorXd>, ConstEigenRef<VectorXd>, …) resolve
// without full qualification inside this namespace.
using tycho::ConstEigenRef;
using tycho::EigenRef;

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
// std::unique_ptr<RecoveryChain> (concrete type NoopRecovery, complete only in
// psiopt.cpp). Same forward-declare + out-of-line ctor/dtor discipline. The
// NoopRecovery implementation shipped today always returns kAcceptAsIs; the
// alg_impl call site is provably inert (see noop_recovery.h and the call-site
// comment in alg_impl) — no live recovery dispatcher exists yet.
// PSIOPT also owns an optional feasibility-restoration mode-switch through a
// std::unique_ptr<RestorationStrategy> (concrete type ProximalSwitchRestoration,
// complete only in psiopt.cpp). Unlike the four components above it is NOT
// always constructed: rebuild_globalization_components() leaves it null unless
// restoration_mode_ == proximal_switch, so on the default path every
// restoration branch guards on `restoration_ != nullptr` and is provably dead.
// Same forward-declare + out-of-line ctor/dtor discipline as the others.
class AcceptanceStrategy;
class GlobalizationMechanism;
class BarrierGovernor;
class RecoveryChain;
class RestorationStrategy;
struct ProgressMeasures;

class PSIOPT {
  public:
    enum class BarrierModes { PROBE, LOQO };
    enum class LineSearchModes { AUGLANG, LANG, L1, NOLS };
    enum class AlgorithmModes { OPT, OPTNO, SOE, INIT };

    /// Alias so existing callers using PSIOPT::ConvergenceFlags continue to work.
    using ConvergenceFlags = tycho::ConvergenceFlags;

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
        // proximal_restoration.h) — 0 refuses restoration entirely (budget
        // exhausted before the first entry). Ignored when restoration_mode_ ==
        // off. validate() requires it >= 0.
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

        // --- Unacceptable tolerances (reserved — not currently read by any algorithm code) ---
        double unacc_kkt_tol_ = 10;
        double unacc_econ_tol_ = 2;
        double unacc_icon_tol_ = 2;
        double unacc_bar_tol_ = 2;

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

        // --- Feasibility restoration (opt-in proximal mode-switch) ---
        // off (default) reproduces today's behavior bit-identically: no
        // RestorationStrategy is constructed and every restoration branch in
        // the solver is provably dead. proximal_switch selects the proximal
        // feasibility mode-switch (ProximalSwitchRestoration), which — on a
        // ladder-exhausted step rejection at a not-near-feasible point — swaps
        // the true objective for a proximal term until infeasibility is
        // sufficiently reduced, then resumes optimality mode. Composes with
        // every acceptance_strategy_ and barrier_governor_ (no matrix
        // restrictions). Enum lives in psiopt_fwd.h; the per-phase entry budget
        // is max_feas_rest_ above.
        RestorationModes restoration_mode_ = RestorationModes::off;

        // --- Barrier parameters ---
        double init_mu_ = 0.001;
        double max_mu_ = 100.0;
        double min_mu_ = 1.0e-12;

        // --- Step parameters ---
        double bound_fraction_ = 0.99;
        double bound_push_ = 1.0e-3;
        double neg_slack_reset_ = 1.0e-12;
        double soe_bound_relax_ = 1.0e-8; // reserved — not currently read by algorithm code
        double alpha_red_ = 2.0;

        // --- Hessian perturbation ---
        double delta_h_ = 1.0e-5;
        double incr_h_ = 8.0;
        double decr_h_ = 0.333333;

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
        // only increments when restoration_mode_ == proximal_switch). Counts
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
        // ProximalSwitchRestoration is the concrete reporter today —
        // globalization/proximal_restoration.h). WRITE-ONLY diagnostics field:
        // no algorithm code reads it back. Sentinel -1 when no restoration
        // strategy is constructed on the active solve path (today's only
        // path — feasibility restoration is not yet wired into the solver).
        // Same last-phase-wins semantics as last_monotone_switches_ once a
        // restoration strategy is wired and this is collected per phase.
        int last_feas_rest_entries_ = -1;

        // Number of iterations spent in restoration mode during the most
        // recent solve's LAST PHASE, reported by RestorationStrategy::
        // append_diagnostics(). WRITE-ONLY diagnostics field. Sentinel -1
        // when no restoration strategy is constructed on the active solve
        // path. Same per-phase semantics as last_feas_rest_entries_.
        int last_feas_rest_iters_ = -1;

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
        }
    };

    using VectorXd = Eigen::VectorXd;

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

    // --- Accessors ---
    /// Returns a mutable reference to the settings struct. Direct writes bypass
    /// per-field validation in the set_*() methods. All settings are re-validated
    /// at run_phase_sequence() entry via Settings::validate().
    Settings &settings() { return settings_; }
    const Settings &settings() const { return settings_; }
    const SolveResult &result() const { return result_; }

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

    void set_unacc_tols(double kktol, double etol, double itol, double bartol);

    void set_div_kkt_tol(double div_kkt_tol);
    void set_div_bar_tol(double div_bar_tol);
    void set_div_econ_tol(double div_econ_tol);
    void set_div_icon_tol(double div_icon_tol);
    void set_div_tols(double div_kkt_tol, double div_econ_tol, double div_icon_tol,
                      double div_bar_tol);

    void set_bound_fraction(double bound_fraction);
    void set_bound_push(double bound_push);
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

    // --- Callback methods ---
    void set_early_callback(const EarlyCallBackType &f) {
        this->early_callback_enabled_ = true;
        this->early_callback_ = f;
    }
    void disable_early_callback() { this->early_callback_enabled_ = false; }
    void set_late_callback(const LateCallBackType &f) {
        this->late_callback_enabled_ = true;
        this->late_callback_ = f;
    }
    void disable_late_callback() { this->late_callback_enabled_ = false; }

    // --- Query methods ---
    ConvergenceFlags get_convergence_flag() const { return result_.converge_flag_; }

    // --- Printing ---
    static void print_header() { fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65); }

  private:
    // Test access: these unit tests verify which concrete acceptance strategy
    // the settings dispatch constructs; befriended narrowly instead of
    // exposing a public rebuild hook.
    friend class ::RecoveryDispatchGate_FunnelSelectionConstructsFunnelAcceptance_Test;
    friend class ::RecoveryDispatchGate_FilterSelectionConstructsFilterAcceptance_Test;
    friend class ::RecoveryDispatchGate_MonitoredSelectionConstructsMonitoredGovernor_Test;
    friend class ::FeasibilitySwitch_ProximalSwitchConstructsRestorationAndWrapsRecovery_Test;
    friend class ::FeasibilitySwitch_OffModeConstructsNoRestoration_Test;
    friend class ::FeasibilitySwitch_FilterSeedsRestorationConstraintTol_Test;
    friend class ::NestedSeamHarness;
    friend class ::NestedLifecycleHarness;

    Settings settings_;
    SolveResult result_;
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
    // == proximal_switch, in which case it holds a ProximalSwitchRestoration and
    // FeasibilitySwitchRecovery is wrapped as the outermost recovery link. On
    // the default path (off) it stays null and every restoration branch in
    // eval_nlp / the classic+generic trial-eval seams / alg_impl guards on
    // `restoration_ != nullptr` (or `ctx.restoration_ != nullptr`) and is
    // provably dead. run_phase_sequence() resets it (when present) at each phase
    // boundary alongside the other components, and collects its diagnostics into
    // SolveResult::last_feas_rest_entries_/last_feas_rest_iters_.
    std::unique_ptr<RestorationStrategy> restoration_;

    // (Re)builds acceptance_/mechanism_/governor_/recovery_ from the current
    // Settings. Called once at the top of every run_phase_sequence() (i.e.
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

    // --- Problem dimensions ---
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

    // =========================================================================
    // KKTVector — lightweight non-owning view over compound KKT layout
    //   [primals | slacks | eq_lmults | iq_lmults]
    // Used for both the iterate vector (x, s, lambda_e, lambda_i) and the
    // RHS/gradient vector (grad_x, grad_s, c_eq, c_iq). The two accessor
    // groups provide semantic names for each interpretation.
    //
    // const-correctness: const overloads use std::as_const(data_) to force
    // Eigen's .head()/.segment()/.tail() to return immutable segment
    // expressions. Without this, calling .head() on the non-const VectorXd&
    // member would return a mutable expression even from a const method.
    // Lifetime: must not outlive the referenced VectorXd.
    // =========================================================================
    class KKTVector {
      public:
        KKTVector(Eigen::VectorXd &data, int pv, int sv, int ec, int ic)
            : data_(data), pv_(pv), sv_(sv), ec_(ec), ic_(ic) {
            assert(pv >= 0 && sv >= 0 && ec >= 0 && ic >= 0);
            assert(data.size() >= pv + sv + ec + ic);
        }

        // --- Primal/slack segments ---
        auto primals() { return data_.head(pv_); }
        auto primals() const { return std::as_const(data_).head(pv_); }
        auto slacks() { return data_.segment(pv_, sv_); }
        auto slacks() const { return std::as_const(data_).segment(pv_, sv_); }
        auto primals_slacks() { return data_.head(pv_ + sv_); }
        auto primals_slacks() const { return std::as_const(data_).head(pv_ + sv_); }

        // --- Multiplier segments ---
        auto eq_lmults() { return data_.segment(pv_ + sv_, ec_); }
        auto eq_lmults() const { return std::as_const(data_).segment(pv_ + sv_, ec_); }
        auto iq_lmults() { return data_.tail(ic_); }
        auto iq_lmults() const { return std::as_const(data_).tail(ic_); }
        auto lmults() { return data_.tail(ec_ + ic_); }
        auto lmults() const { return std::as_const(data_).tail(ec_ + ic_); }

        // --- Gradient/constraint segments (intentional aliases) ---
        // Same memory layout as the primal/multiplier accessors above, but with
        // names matching the RHS/gradient interpretation: the primal block holds
        // the objective gradient, the slack block holds the dual gradient, and
        // the multiplier blocks hold constraint values.
        // These are intentional aliases: prim_grad() == primals(),
        // dual_grad() == slacks(), eq_cons() == eq_lmults(), iq_cons() == iq_lmults().
        auto prim_grad() { return data_.head(pv_); }
        auto prim_grad() const { return std::as_const(data_).head(pv_); }
        auto dual_grad() { return data_.segment(pv_, sv_); }
        auto dual_grad() const { return std::as_const(data_).segment(pv_, sv_); }
        auto prim_dual_grad() { return data_.head(pv_ + sv_); }
        auto prim_dual_grad() const { return std::as_const(data_).head(pv_ + sv_); }
        auto eq_cons() { return data_.segment(pv_ + sv_, ec_); }
        auto eq_cons() const { return std::as_const(data_).segment(pv_ + sv_, ec_); }
        auto iq_cons() { return data_.tail(ic_); }
        auto iq_cons() const { return std::as_const(data_).tail(ic_); }
        auto all_cons() { return data_.tail(ec_ + ic_); }
        auto all_cons() const { return std::as_const(data_).tail(ec_ + ic_); }

        // --- Full vector access ---
        Eigen::VectorXd &data() { return data_; }
        const Eigen::VectorXd &data() const { return data_; }

      private:
        Eigen::VectorXd &data_;
        int pv_, sv_, ec_, ic_;
    };

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
    int factor_impl(bool docompute, bool ZFac, double ipurt, double incpurt0, double incpurt,
                    double &finalpert, double &cumpert);

    bool analyze_kkt_matrix();

    void ensure_solver_initialized();

    // --- Barrier math helpers (defined in psiopt.cpp) ---
    void apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> FXI) const;
    // max_step_to_boundary was extracted verbatim into BacktrackingLineSearch;
    // it is now a private helper of that mechanism.
    void complementarity(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI,
                         double &avgcomp, double &mincomp, double &maxcomp) const;
    void barrier_hessian(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                         Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> LI, double mu);
    // loqo_mu / mpc_mu were extracted verbatim into ClassicAdaptiveGovernor
    // (src/solvers/psiopt_globalization.cpp); the barrier-
    // parameter update now runs through governor_->update_barrier(). The
    // barrier_objective()/barrier_gradient() helpers formerly declared here were
    // dead after that extraction (PSIOPT no longer called them) and have been
    // removed; ClassicMeritAcceptance and ClassicAdaptiveGovernor each carry
    // their own copies. complementarity() STAYS — it is still called from the
    // evaluate stage (its maxcomp output feeds converge_check's barr_inf_).

    // --- NLP eval dispatch methods (defined in psiopt.cpp) ---
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
