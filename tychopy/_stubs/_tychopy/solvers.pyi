"""SubModule Containing InteriorPointSolver,NLP, and Solver Flags"""

from collections.abc import Callable, Mapping, Sequence
import enum
from typing import Callable, overload

import numpy

import _tychopy.vector_functions


class InteriorPointSolver:
    @overload
    def __init__(self, arg: "hven::solvers::NonLinearProgram", /) -> None: ...

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, **kwargs) -> None:
        """
        Construct an InteriorPointSolver, optionally overriding settings by name.

        Every settings property below is also accepted as a keyword argument
        here (e.g. ``InteriorPointSolver(max_iters=500, kkt_tol=1e-9)``).
        ``preset`` (a name accepted by :meth:`apply_preset`) is applied FIRST,
        before any other keyword argument override, so
        ``InteriorPointSolver(preset="soc_recovery_l1", max_soc=6)`` starts from
        the preset and then raises max_soc past what the preset itself sets.
        Validated properties keep their validation (raise ValueError exactly as
        the corresponding ``set_*`` method / property assignment would).

        Raises
        ------
        TypeError
            If an unrecognized keyword argument is given, naming it.
        """

    def optimize(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    def solve_optimize(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    def solve(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @property
    def max_iters(self) -> int: ...

    @max_iters.setter
    def max_iters(self, arg: int, /) -> None: ...

    @property
    def max_acc_iters(self) -> int: ...

    @max_acc_iters.setter
    def max_acc_iters(self, arg: int, /) -> None: ...

    @property
    def max_ls_iters(self) -> int: ...

    @max_ls_iters.setter
    def max_ls_iters(self, arg: int, /) -> None: ...

    def set_max_iters(self, arg: int, /) -> None: ...

    def set_max_acc_iters(self, arg: int, /) -> None: ...

    def set_max_ls_iters(self, arg: int, /) -> None: ...

    @property
    def max_soc(self) -> int:
        """
        Maximum number of second-order correction steps attempted after a rejected trial step. 0 (default) disables second-order correction entirely, so the solver behaves exactly as it did before this feature existed; the recommended enable value is 4 (Wachter & Biegler 2006).
        """

    @max_soc.setter
    def max_soc(self, arg: int, /) -> None: ...

    @property
    def ls_extended_iters(self) -> int:
        """
        Extra backtracking trials allowed on the classic line-search ladder once the normal cap and second-order correction (if enabled) are exhausted. 0 (default) disables extended backtracking entirely.
        """

    @ls_extended_iters.setter
    def ls_extended_iters(self, arg: int, /) -> None: ...

    @property
    def alpha_red(self) -> float: ...

    @alpha_red.setter
    def alpha_red(self, arg: float, /) -> None: ...

    def set_alpha_red(self, arg: float, /) -> None: ...

    @property
    def wide_console(self) -> bool: ...

    @wide_console.setter
    def wide_console(self, arg: bool, /) -> None: ...

    @property
    def fast_factor_alg(self) -> bool: ...

    @fast_factor_alg.setter
    def fast_factor_alg(self, arg: bool, /) -> None: ...

    @property
    def last_total_time(self) -> float: ...

    @property
    def last_pre_time(self) -> float: ...

    @property
    def last_func_time(self) -> float: ...

    @property
    def last_kkt_time(self) -> float: ...

    @property
    def last_misc_time(self) -> float: ...

    @property
    def last_print_time(self) -> float: ...

    @property
    def last_solver_init_time(self) -> float: ...

    @property
    def last_iter_num(self) -> int: ...

    @property
    def last_obj_val(self) -> float: ...

    @property
    def last_primals(self) -> numpy.ndarray: ...

    @property
    def last_soc_steps(self) -> int:
        """
        Number of second-order correction back-substitutions performed during the most recent solve. Always 0 unless max_soc is set > 0.
        """

    @property
    def last_watchdog_activations(self) -> int:
        """
        Number of times the watchdog recovery heuristic armed during the most recent solve. Always 0 unless watchdog is enabled.
        """

    @property
    def last_recovery_depth_histogram(self) -> list[int]:
        """
        Counts of how each rejected step's recovery was resolved during the most recent solve, as a 5-element list: [second-order correction, extended backtracking, watchdog, unresolved, restoration]. The final bucket only increments when restoration_mode is proximal_switch or l1_nested.
        """

    @property
    def last_funnel_width(self) -> float:
        """
        Final funnel width (tau) at the end of the most recent solve's last phase. -1.0 unless acceptance_strategy is funnel, or if no acceptance test ran.
        """

    @property
    def last_filter_size(self) -> int:
        """
        Final number of stored filter (theta, phi) pairs at the end of the most recent solve's last phase. -1 unless acceptance_strategy is filter.
        """

    @property
    def last_filter_resets(self) -> int:
        """
        Number of filter-reset-heuristic clears during the most recent solve's last phase. -1 unless acceptance_strategy is filter.
        """

    @property
    def last_monotone_switches(self) -> int:
        """
        Number of free -> monotone handoffs during the most recent solve's last phase. -1 unless barrier_governor is monitored.
        """

    @property
    def last_monotone_iters(self) -> int:
        """
        Number of iterations spent in monotone mode during the most recent solve's last phase. -1 unless barrier_governor is monitored.
        """

    @property
    def last_feas_rest_entries(self) -> int:
        """
        Number of times feasibility restoration was entered during the most recent solve's last phase. -1 unless restoration_mode is proximal_switch or l1_nested (no restoration strategy is constructed when restoration_mode is off). Counts identically under both modes -- l1_nested has no separate inner/outer iteration split, so this and last_feas_rest_iters mean the same thing regardless of which mode is selected.
        """

    @property
    def last_feas_rest_iters(self) -> int:
        """
        Number of iterations spent in the feasibility-restoration phase during the most recent solve's last phase. -1 unless restoration_mode is proximal_switch or l1_nested.
        """

    @property
    def last_prox_reg_primal(self) -> float:
        """
        Persistent primal base shift (rho_k) applied to the Hessian diagonal at the last factorized iteration of the most recent solve's last phase. -1.0 unless inertia_mode is proximal_regularization (or if that phase converged before its first factorization, so no shift was ever applied).
        """

    @property
    def last_prox_reg_dual(self) -> float:
        """
        Barrier-scaled dual shift (delta_c) subtracted from the constraint-row diagonals at the last factorized iteration of the most recent solve's last phase. -1.0 unless inertia_mode is proximal_regularization (or if that phase converged before its first factorization); 0.0 if that iteration fell inside a nested l1 restoration phase, where the shift is suppressed.
        """

    @property
    def last_eval_exception(self) -> str:
        """
        Message of the most recent trial-point evaluation exception absorbed during the most recent solve call, or the empty string when every evaluation succeeded. A populated value means the acceptance machinery rejected one or more un-evaluable trial steps (for example an iterate that stepped outside an interpolation table's domain) and the solve continued -- to full recovery, to a graceful ACCEPTABLE-level exit at an already-acceptable iterate, or into feasibility restoration; a solve with none of those paths available raises RuntimeError instead. In a multi-phase solve, an earlier phase's message persists on this property even when a later phase aborts, since the diagnostic is written at each phase's close.
        """

    @property
    def obj_scale(self) -> float: ...

    @obj_scale.setter
    def obj_scale(self, arg: float, /) -> None: ...

    @property
    def print_level(self) -> int: ...

    @print_level.setter
    def print_level(self, arg: int, /) -> None: ...

    def set_print_level(self, arg: int, /) -> None: ...

    @property
    def converge_flag(self) -> ConvergenceFlags: ...

    @property
    def kkt_tol(self) -> float: ...

    @kkt_tol.setter
    def kkt_tol(self, arg: float, /) -> None: ...

    @property
    def bar_tol(self) -> float: ...

    @bar_tol.setter
    def bar_tol(self, arg: float, /) -> None: ...

    @property
    def eq_con_tol(self) -> float: ...

    @eq_con_tol.setter
    def eq_con_tol(self, arg: float, /) -> None: ...

    @property
    def ineq_con_tol(self) -> float: ...

    @ineq_con_tol.setter
    def ineq_con_tol(self, arg: float, /) -> None: ...

    def set_kkt_tol(self, arg: float, /) -> None: ...

    def set_bar_tol(self, arg: float, /) -> None: ...

    def set_eq_con_tol(self, arg: float, /) -> None: ...

    def set_ineq_con_tol(self, arg: float, /) -> None: ...

    def set_tols(self, kkt_tol: float = 1e-06, eq_con_tol: float = 1e-06, ineq_con_tol: float = 1e-06, bar_tol: float = 1e-06) -> None: ...

    @property
    def acc_kkt_tol(self) -> float: ...

    @acc_kkt_tol.setter
    def acc_kkt_tol(self, arg: float, /) -> None: ...

    @property
    def acc_bar_tol(self) -> float: ...

    @acc_bar_tol.setter
    def acc_bar_tol(self, arg: float, /) -> None: ...

    @property
    def acc_eq_con_tol(self) -> float: ...

    @acc_eq_con_tol.setter
    def acc_eq_con_tol(self, arg: float, /) -> None: ...

    @property
    def acc_ineq_con_tol(self) -> float: ...

    @acc_ineq_con_tol.setter
    def acc_ineq_con_tol(self, arg: float, /) -> None: ...

    def set_acc_kkt_tol(self, arg: float, /) -> None: ...

    def set_acc_bar_tol(self, arg: float, /) -> None: ...

    def set_acc_eq_con_tol(self, arg: float, /) -> None: ...

    def set_acc_ineq_con_tol(self, arg: float, /) -> None: ...

    def set_acc_tols(self, acc_kkt_tol: float = 0.01, acc_eq_con_tol: float = 0.001, acc_ineq_con_tol: float = 0.001, acc_bar_tol: float = 0.001) -> None: ...

    @property
    def div_kkt_tol(self) -> float: ...

    @div_kkt_tol.setter
    def div_kkt_tol(self, arg: float, /) -> None: ...

    @property
    def div_bar_tol(self) -> float: ...

    @div_bar_tol.setter
    def div_bar_tol(self, arg: float, /) -> None: ...

    @property
    def div_eq_con_tol(self) -> float: ...

    @div_eq_con_tol.setter
    def div_eq_con_tol(self, arg: float, /) -> None: ...

    @property
    def div_ineq_con_tol(self) -> float: ...

    @div_ineq_con_tol.setter
    def div_ineq_con_tol(self, arg: float, /) -> None: ...

    def set_div_kkt_tol(self, arg: float, /) -> None: ...

    def set_div_bar_tol(self, arg: float, /) -> None: ...

    def set_div_eq_con_tol(self, arg: float, /) -> None: ...

    def set_div_ineq_con_tol(self, arg: float, /) -> None: ...

    @property
    def neg_slack_reset(self) -> float: ...

    @neg_slack_reset.setter
    def neg_slack_reset(self, arg: float, /) -> None: ...

    @property
    def bound_fraction(self) -> float: ...

    @bound_fraction.setter
    def bound_fraction(self, arg: float, /) -> None: ...

    def set_bound_fraction(self, arg: float, /) -> None: ...

    @property
    def bound_push(self) -> float: ...

    @bound_push.setter
    def bound_push(self, arg: float, /) -> None: ...

    @property
    def bound_interval_push(self) -> float:
        """
        Fraction of a two-sided bounded variable's declared interval (upper - lower) that additionally caps its interior push at solve entry, on top of bound_push's absolute push, so a narrow interval is never pushed past its own midpoint (Ipopt's bound_frac, same default). Read only when the problem declares native variable bounds. 1e-2 (default); must lie in (0, 0.5).
        """

    @bound_interval_push.setter
    def bound_interval_push(self, arg: float, /) -> None: ...

    @property
    def bound_relax_factor(self) -> float:
        """
        Widening applied to every finite declared variable bound before it is recorded, as this factor times max(1, |bound|), so the box every barrier term divides by is never exactly the declared one (Ipopt's bound_relax_factor, same default). 1e-8 (default); must lie in [0, 1e-2]. Zero records every declared bound verbatim. Also separates the bounds of a fixed variable under fixed_variable_treatment=RelaxBounds. Read only when native variable bounds are declared.
        """

    @bound_relax_factor.setter
    def bound_relax_factor(self, arg: float, /) -> None: ...

    @property
    def fixed_variable_treatment(self) -> FixedVariableTreatments:
        """
        How a primal variable whose declared lower and upper bounds are equal is handed to the solver, corresponding to Ipopt's fixed_variable_treatment option. MakeParameter (default) eliminates the variable from the factorized system entirely -- one row and column narrower per fixed variable, with an exact value in the returned solution. MakeConstraint keeps the variable free and adds one internal equality row per fixed variable instead -- one row and column wider. RelaxBounds keeps the variable as a two-sided bounded variable with its bounds pushed apart by bound_relax_factor, held near its value by the barrier. All three reach the same solution on a well-posed problem. See FixedVariableTreatments for the full mechanism.
        """

    @fixed_variable_treatment.setter
    def fixed_variable_treatment(self, arg: FixedVariableTreatments, /) -> None: ...

    @property
    def delta_h(self) -> float: ...

    @delta_h.setter
    def delta_h(self, arg: float, /) -> None: ...

    @property
    def incr_h(self) -> float: ...

    @incr_h.setter
    def incr_h(self, arg: float, /) -> None: ...

    @property
    def decr_h(self) -> float: ...

    @decr_h.setter
    def decr_h(self, arg: float, /) -> None: ...

    def set_delta_h(self, arg: float, /) -> None: ...

    def set_incr_h(self, arg: float, /) -> None: ...

    def set_decr_h(self, arg: float, /) -> None: ...

    def set_hpert_params(self, delta_h: float, incr_h: float, decr_h: float) -> None: ...

    @property
    def inertia_mode(self) -> InertiaModes:
        """
        KKT inertia-correction / regularization mode: classic (default) runs the on-demand inertia ladder -- each iteration first attempts an unperturbed factorization and shifts the Hessian diagonal (by increasing amounts) when the factorization's inertia is not exactly (kkt_dim - m, m, 0); on a singularity signal (rank deficiency, or neigs < m by Gould's inertia theorem) it additionally engages the barrier-scaled dual shift on the constraint-row diagonals, at most once per phase (later iterations pre-apply it), and an exhausted ladder fails the step -- SINGULAR_KKT when nothing resolves it. proximal_regularization bakes two shifts into the base matrix every iteration instead of the classic zero-perturbation first attempt: a small persistent, decaying primal base shift on the Hessian diagonal, and an always-on barrier-scaled dual shift on the constraint-row diagonals (suppressed while a nested l1 restoration phase is active, since the elastic pivots already regularize those rows). The same incr_h/decr_h escalation ladder still fires on top when the base attempt has wrong inertia or is singular. See InertiaModes for the full mechanism and literature citations, and last_prox_reg_primal/last_prox_reg_dual for the resulting diagnostics.
        """

    @inertia_mode.setter
    def inertia_mode(self, arg: InertiaModes, /) -> None: ...

    @property
    def init_mu(self) -> float: ...

    @init_mu.setter
    def init_mu(self, arg: float, /) -> None: ...

    @property
    def min_mu(self) -> float: ...

    @min_mu.setter
    def min_mu(self, arg: float, /) -> None: ...

    @property
    def max_mu(self) -> float: ...

    @max_mu.setter
    def max_mu(self, arg: float, /) -> None: ...

    @property
    def pd_step_strategy(self) -> PDStepStrategies: ...

    @pd_step_strategy.setter
    def pd_step_strategy(self, arg: PDStepStrategies, /) -> None: ...

    @property
    def qp_par_solve(self) -> int: ...

    @qp_par_solve.setter
    def qp_par_solve(self, arg: int, /) -> None: ...

    @property
    def soe_mode(self) -> AlgorithmModes: ...

    @soe_mode.setter
    def soe_mode(self, arg: AlgorithmModes, /) -> None: ...

    @property
    def opt_bar_mode(self) -> BarrierModes: ...

    @opt_bar_mode.setter
    def opt_bar_mode(self, arg: BarrierModes, /) -> None: ...

    @property
    def soe_bar_mode(self) -> BarrierModes: ...

    @soe_bar_mode.setter
    def soe_bar_mode(self, arg: BarrierModes, /) -> None: ...

    @overload
    def set_opt_bar_mode(self, arg: BarrierModes, /) -> None: ...

    @overload
    def set_opt_bar_mode(self, arg: str, /) -> None: ...

    @overload
    def set_soe_bar_mode(self, arg: BarrierModes, /) -> None: ...

    @overload
    def set_soe_bar_mode(self, arg: str, /) -> None: ...

    @property
    def opt_ls_mode(self) -> LineSearchModes: ...

    @opt_ls_mode.setter
    def opt_ls_mode(self, arg: LineSearchModes, /) -> None: ...

    @property
    def soe_ls_mode(self) -> LineSearchModes: ...

    @soe_ls_mode.setter
    def soe_ls_mode(self, arg: LineSearchModes, /) -> None: ...

    @overload
    def set_opt_ls_mode(self, arg: LineSearchModes, /) -> None: ...

    @overload
    def set_opt_ls_mode(self, arg: str, /) -> None: ...

    @overload
    def set_soe_ls_mode(self, arg: LineSearchModes, /) -> None: ...

    @overload
    def set_soe_ls_mode(self, arg: str, /) -> None: ...

    @property
    def acceptance_strategy(self) -> AcceptanceStrategies:
        """
        Step-acceptance strategy: classic_merit (default) reproduces the original fused backtracking merit line search bit-for-bit; merit switches to the modernized penalty-based acceptance test selected by merit_penalty_rule; funnel switches to a single-scalar bound on constraint violation, tightened while accepted iterates stay within it; filter switches to a (violation, objective) Wachter-Biegler-style filter. funnel and filter are designed to operate above a monotone barrier safeguard, so each requires barrier_governor=monitored, or never_monotone=True to run without the monotone-barrier safeguard (the two are mutually exclusive; ValueError at validate() time otherwise); watchdog is compatible with all four strategies. These are heuristically-motivated acceptance alternatives, not one another's strict improvement -- compare against classic_merit on your own problem before adopting one.
        """

    @acceptance_strategy.setter
    def acceptance_strategy(self, arg: AcceptanceStrategies, /) -> None: ...

    @property
    def merit_penalty_rule(self) -> MeritPenaltyRules:
        """
        Penalty-parameter update rule for the modernized merit strategy; only read when acceptance_strategy is merit. wmno (default) updates a single penalty value from the directional-derivative condition; flexible tracks a penalty interval and accepts a step that improves the merit for at least one value in that interval.
        """

    @merit_penalty_rule.setter
    def merit_penalty_rule(self, arg: MeritPenaltyRules, /) -> None: ...

    @property
    def watchdog(self) -> bool:
        """
        Enables the watchdog recovery heuristic, which tolerates a temporarily worse step after repeated rejections instead of immediately shrinking the step further. false (default) preserves the original behavior.
        """

    @watchdog.setter
    def watchdog(self, arg: bool, /) -> None: ...

    @property
    def barrier_governor(self) -> BarrierGovernors:
        """
        Barrier-parameter governor: classic_adaptive (default) reproduces the original PROBE/LOQO free-mode barrier update bit-for-bit; monitored composes a classic_adaptive delegate with a KKT-error monitor that watches a sliding reference window of recent iterations and, when free-mode progress is no longer a sufficient decrease relative to that window, hands off to a monotone (Fiacco-McCormick) mode with the barrier parameter initialized to 0.8 times the average complementarity and held fixed until the barrier subproblem converges, then decreased; the monitor re-enters free mode once progress against the (frozen) reference window resumes. Each free->monotone handoff and each monotone barrier-parameter decrease resets the acceptance strategy's per-barrier-subproblem state — the filter set is cleared and the violation thresholds (and funnel width) are RE-DERIVED from the current iterate's violation on the next acceptance test, exactly as a new barrier subproblem re-bases them in Ipopt. The funnel/filter acceptance strategies are designed to operate above a monotone barrier safeguard, which classic_adaptive does not provide; validate() raises ValueError if they are combined with classic_adaptive unless never_monotone is set. Any acceptance_strategy may pair with monitored.
        """

    @barrier_governor.setter
    def barrier_governor(self, arg: BarrierGovernors, /) -> None: ...

    @property
    def never_monotone(self) -> bool:
        """
        Expert escape hatch, mirroring Ipopt's never-monotone-mode: explicitly accepts running funnel/filter above barrier_governor=classic_adaptive without its monotone safeguard, forfeiting that guard rather than switching to barrier_governor=monitored. false (default). Contradictory with barrier_governor=monitored (which already supplies the monotone fallback this knob opts out of) -- validate() raises ValueError on that combination.
        """

    @never_monotone.setter
    def never_monotone(self, arg: bool, /) -> None: ...

    @property
    def restoration_mode(self) -> RestorationModes:
        """
        Feasibility-restoration mode selector: off (default) reproduces today's behavior bit-identically -- no restoration strategy is constructed, so every restoration branch in the solver is provably dead. proximal_switch enables the proximal feasibility mode-switch: when the recovery chain exhausts on a rejected step, the solver switches to a pure feasibility phase -- the objective is replaced by a proximal term centered on the switch point (coefficient sqrt(mu) at entry) while all constraints and barrier machinery keep running -- and returns to the true objective once the acceptance strategy's infeasibility-reduction test passes (per-strategy: classic_merit uses a relative infeasibility-reduction test against the entry point, Ipopt restoration-convergence style; merit reduces against the smallest-known infeasibility held from the optimality phase -- frozen at restoration entry and unchanged by feasibility-phase iterates; funnel/filter use their own reference-solver tests). l1_nested enables the nested l1 elastic feasibility restoration instead: the same trigger and the same acceptance-strategy exit test, but the elastic reformulation runs as a condensed in-place phase reusing the outer barrier algorithm's own KKT system, rather than swapping the outer objective for a proximal term -- see RestorationModes for the mechanism and Ipopt-lineage citations. Unlike proximal_switch, l1_nested first tries a soft feasibility pre-stage (full fraction-to-boundary steps tested under a primal-dual-error reduction rule) and only escalates to the full elastic switch after several soft steps in a row fail to recover; proximal_switch has no pre-stage and switches directly. Both modes refuse entry at a near-feasible point or once the per-phase budget max_feas_rest is exhausted. Composes with every acceptance_strategy and barrier_governor (no matrix restrictions -- every shipped acceptance strategy implements the exit test either mode relies on). Mode-switch lineage: Knitro's bar_switchobj=scalarprox for proximal_switch, with entry/exit semantics derived from Ipopt's restoration phase and Uno's phase switching for both modes.
        """

    @restoration_mode.setter
    def restoration_mode(self, arg: RestorationModes, /) -> None: ...

    @property
    def max_feas_rest(self) -> int:
        """
        Per-phase cap on the number of times feasibility restoration may be entered. 0 disables restoration entry entirely (the budget is exhausted before the first entry); 2 (default). Ignored when restoration_mode is off. Negative values raise ValueError immediately on assignment; validate() re-checks non-negativity as a backstop.
        """

    @max_feas_rest.setter
    def max_feas_rest(self, arg: int, /) -> None: ...

    @property
    def force_qp_analysis(self) -> bool: ...

    @force_qp_analysis.setter
    def force_qp_analysis(self, arg: bool, /) -> None: ...

    @property
    def qp_ref_steps(self) -> int: ...

    @qp_ref_steps.setter
    def qp_ref_steps(self, arg: int, /) -> None: ...

    @property
    def qp_pivot_perturb(self) -> int: ...

    @qp_pivot_perturb.setter
    def qp_pivot_perturb(self, arg: int, /) -> None: ...

    @property
    def qp_matching(self) -> int: ...

    @qp_matching.setter
    def qp_matching(self, arg: int, /) -> None: ...

    @property
    def qp_scaling(self) -> int: ...

    @qp_scaling.setter
    def qp_scaling(self, arg: int, /) -> None: ...

    @property
    def qp_threads(self) -> int: ...

    @qp_threads.setter
    def qp_threads(self, arg: int, /) -> None: ...

    @property
    def qp_pivot_strategy(self) -> QPPivotModes: ...

    @qp_pivot_strategy.setter
    def qp_pivot_strategy(self, arg: QPPivotModes, /) -> None: ...

    @property
    def qp_ordering_mode(self) -> QPOrderingModes: ...

    @qp_ordering_mode.setter
    def qp_ordering_mode(self, arg: QPOrderingModes, /) -> None: ...

    @overload
    def set_qp_ordering_mode(self, arg: QPOrderingModes, /) -> None: ...

    @overload
    def set_qp_ordering_mode(self, arg: str, /) -> None: ...

    @property
    def qp_print(self) -> bool: ...

    @qp_print.setter
    def qp_print(self, arg: bool, /) -> None: ...

    @property
    def return_best(self) -> bool: ...

    @return_best.setter
    def return_best(self, arg: bool, /) -> None: ...

    @property
    def best_criteria(self) -> BestCriteriaModes: ...

    @best_criteria.setter
    def best_criteria(self, arg: object, /) -> None: ...

    @overload
    def set_best_criteria(self, arg: BestCriteriaModes, /) -> None: ...

    @overload
    def set_best_criteria(self, arg: str, /) -> None: ...

    @property
    def cnr_mode(self) -> bool: ...

    @cnr_mode.setter
    def cnr_mode(self, arg: bool, /) -> None: ...

    def apply_preset(self, name: str) -> None:
        """
        Apply a named globalization-mechanism configuration.

        Assigns exactly nine Settings fields -- acceptance_strategy,
        merit_penalty_rule, barrier_governor, never_monotone, restoration_mode,
        inertia_mode, max_soc, ls_extended_iters, and watchdog. No other
        Settings field (tolerances, iteration caps, QP/threading parameters,
        ...) is read or written.

        Valid names
        -----------
        classic
            Restores the stock configuration: classic_merit acceptance, the
            classic_adaptive barrier governor, restoration off, classic inertia
            mode, and SOC/extended-backtracking/watchdog all disabled -- the
            bit-identical Settings{} default.
        filter_l1
            Filter acceptance with a monitored barrier governor and nested-l1
            restoration.
        soc_recovery_l1
            Classic-merit acceptance with a monitored barrier governor,
            proximal-regularization inertia, second-order correction
            (max_soc=4), extended backtracking (ls_extended_iters=2), the
            watchdog enabled, and nested-l1 restoration.
        soc_proximal
            Classic-merit acceptance with a monitored barrier governor,
            proximal-regularization inertia, second-order correction
            (max_soc=4), and proximal-switch restoration.
        merit_l1
            Merit acceptance with the classic_adaptive barrier governor and
            nested-l1 restoration.

        See the solver configuration comparison in the reference documentation
        for the evidence behind each non-classic preset.

        Parameters
        ----------
        name : str
            One of the five names above.

        Raises
        ------
        ValueError
            If ``name`` is not one of the five presets above.
        """

class BarrierModes(enum.Enum):
    PROBE = 0

    LOQO = 1

class LineSearchModes(enum.Enum):
    AUGLANG = 0

    LANG = 1

    L1 = 2

    NOLS = 3

class QPPivotModes(enum.Enum):
    OneByOne = 0

    TwoByTwo = 1

class FixedVariableTreatments(enum.Enum):
    """
    Fixed-variable handling selector for InteriorPointSolver.fixed_variable_treatment, corresponding to Ipopt's fixed_variable_treatment option.
    """

    MakeParameter = 0
    """
    Eliminates a fixed variable (equal declared lower and upper bound) from the factorized system entirely -- one row and column narrower per fixed variable, with an exact value in the returned solution. Default.
    """

    MakeConstraint = 1
    """
    Keeps a fixed variable free and adds one internal equality row per fixed variable instead -- one row and column wider than MakeParameter.
    """

    RelaxBounds = 2
    """
    Keeps a fixed variable as an ordinary two-sided bounded variable with its bounds pushed apart by bound_relax_factor, held near its value by the barrier.
    """

class AcceptanceStrategies(enum.Enum):
    classic_merit = 0

    merit = 1

    funnel = 2
    """
    Single-scalar upper bound on constraint violation (the funnel width), tightened while accepted iterates remain within it (Kiessling, Leyffer & Vanaret funnel formulation, implemented after Uno's funnel). Requires a monotone barrier safeguard, so it rejects combination with the default barrier_governor=classic_adaptive unless never_monotone=True (ValueError at validate() time); composes with watchdog. Heuristically motivated -- no convergence guarantee is implied; compare against classic_merit and filter on your own problem.
    """

    filter = 3
    """
    (Constraint violation, objective) pair filter with margined dominance (Wachter-Biegler filter line search, Ipopt lineage). Requires a monotone barrier safeguard, so it rejects combination with the default barrier_governor=classic_adaptive unless never_monotone=True (ValueError at validate() time); composes with watchdog. Heuristically motivated -- no convergence guarantee is implied; compare against classic_merit and funnel on your own problem.
    """

class MeritPenaltyRules(enum.Enum):
    wmno = 0

    flexible = 1

class BarrierGovernors(enum.Enum):
    classic_adaptive = 0
    """
    The classic PROBE/LOQO free-mode barrier update, unchanged -- the bit-identical default.
    """

    monitored = 1
    """
    Free<->monotone monitored barrier governor: a KKT-error monitor hands off to a Fiacco-McCormick monotone mode when free-mode progress stalls, then re-enters free mode once progress resumes -- see the barrier_governor property docstring for the full mechanism.
    """

class RestorationModes(enum.Enum):
    off = 0
    """
    No feasibility restoration -- the bit-identical default. No restoration strategy is constructed, so every restoration branch in the solver is provably dead.
    """

    proximal_switch = 1
    """
    Proximal feasibility mode-switch: on a ladder-exhausted step rejection, keep the same barrier algorithm running but swap the true objective for a proximal term pulling the primals back toward the switch point, until the acceptance strategy's infeasibility-reduction test passes -- see the restoration_mode property docstring for the full mechanism. Composes with every acceptance_strategy and barrier_governor.
    """

    l1_nested = 2
    """
    Nested l1 elastic feasibility restoration: on a ladder-exhausted step rejection, solve the l1 elastic reformulation of the current KKT system as a condensed in-place phase -- each row gets a pair of nonnegative elastic slacks (n, p) absorbing the residual, penalized at rho=1e3 plus a proximity term pulling the primals back toward the switch point with weight sqrt(mu) -- rather than switching the outer objective the way proximal_switch does; the phase reuses the outer barrier algorithm's own KKT system instead of spinning up a separate nested solver. Constants (the penalty rho, the proximity weight factor, the entry/re-entry rules) are pinned at Ipopt's restoration-phase literature defaults (coin-or/Ipopt's IpRestoIpoptNLP / IpRestoIterateInitializer / IpRestoMinC_1Nrm). Before committing to the full elastic switch, a soft feasibility pre-stage first tries ordinary fraction-to-boundary steps under a primal-dual-error reduction rule for a bounded number of consecutive iterations (adapted from Ipopt's soft restoration phase) and only escalates once that budget is exhausted; proximal_switch has no such pre-stage. Prefer l1_nested over proximal_switch when a stall is a genuinely constraint-infeasible point the elastic reformulation can relax productively (the pre-stage also gives it a cheaper recovery attempt before the full switch); prefer proximal_switch for a simpler, cheaper mode-switch with no elastic-slack bookkeeping. Returns to the true objective on the same acceptance-strategy infeasibility-reduction test proximal_switch uses -- see the restoration_mode property docstring. Composes with every acceptance_strategy and barrier_governor; the diagnostics last_feas_rest_entries/last_feas_rest_iters count identically for both modes.
    """

class InertiaModes(enum.Enum):
    classic = 0
    """
    The on-demand inertia ladder inline in factor_impl -- the default. Each call attempts an unperturbed factorization first, unless the sticky per-phase degeneracy latch is already set from an earlier iteration, in which case the constraint-block dual shift (delta_c) is pre-applied before that base attempt. The full Ipopt inertia-correction condition (Wachter & Biegler 2006, Algorithm IC) accepts only exact inertia (kkt_dim - m, m, 0); a singularity signal -- rank deficiency, or neigs < m by Gould's inertia theorem -- engages the on-demand constraint-block dual shift once per call and sets the latch, and if inertia is still wrong the classic Hessian-diagonal shift ladder fires on top (by increasing amounts). Ladder exhaustion, under either mode, force-rejects the step through the recovery chain and -- if the rejection goes unresolved -- aborts the phase as ConvergenceFlags.SINGULAR_KKT (see max_refac).
    """

    proximal_regularization = 1
    """
    Proximal primal-dual regularization: a small persistent, decaying primal base shift (rho_k, floored at 1e-10, the Cipolla-Gondzio floor) on the Hessian diagonal, plus an always-on barrier-scaled dual shift (delta_c = 1e-8 * mu^0.25, Ipopt's jacobian_regularization_value/exponent constants, matching its perturb_always_cd semantics) on the constraint-row diagonals, are baked into the base matrix every iteration in place of the classic zero-perturbation first attempt -- the same escalation ladder still fires on top when the base attempt has wrong inertia or is singular (a singular base attempt is itself treated as wrong inertia under this mode, matching classic). Ladder exhaustion, under either mode, force-rejects the step through the recovery chain and -- if the rejection goes unresolved -- aborts the phase as ConvergenceFlags.SINGULAR_KKT (see max_refac). rho_k decays toward its floor by decr_h each iteration the base attempt sufficed, or persists at the decayed total shift (rho_k plus the ladder's last delta) when the ladder fired. The dual shift is suppressed while a nested l1 restoration phase is active -- the elastic pivots already regularize those constraint rows at a magnitude the dual shift would be negligible against, and stacking it would make the elastic step-recovery algebra inconsistent with the solved system; the proximal mode-switch restoration touches only the primal diagonal, so the dual shift stays on under it. No new tunable constants -- rho_k's floor and the dual shift's scale/exponent are fixed. See last_prox_reg_primal/last_prox_reg_dual for the per-solve diagnostics this mode reports.
    """

def ipopt_available() -> bool:
    """True when this build was configured with ENABLE_IPOPT."""

class PDStepStrategies(enum.Enum):
    PrimSlackEq_Iq = 0

    AllMinimum = 1

    PrimSlack_EqIq = 2

    MaxEq = 3

class ConvergenceFlags(enum.IntEnum):
    CONVERGED = 0

    ACCEPTABLE = 1

    NOTCONVERGED = 2

    DIVERGING = 3

    SINGULAR_KKT = 4

class AlgorithmModes(enum.Enum):
    OPT = 0

    OPTNO = 1

    SOE = 2

    INIT = 3

class QPOrderingModes(enum.Enum):
    MINDEG = 0

    METIS = 2

    PARMETIS = 3

class BestCriteriaModes(enum.Enum):
    ECONS = 0

    ICONS = 1

    KKT = 2

    OBJ = 3

class Mode(enum.Enum):
    """
    Which objective a solve() call pursued: drive to optimality, or only to feasibility.
    """

    Optimal = 0

    Feasible = 1

class DeclarationKey:
    """
    The declared problem's identity stamp a WarmStartData was taken under. Engine-independent and treatment-independent by construction -- see hven's structure_identity.h for the exact coverage.
    """

    def __init__(self) -> None: ...

    @property
    def declaration_digest(self) -> int: ...

    @declaration_digest.setter
    def declaration_digest(self, arg: int, /) -> None: ...

    @property
    def bound_digest(self) -> int: ...

    @bound_digest.setter
    def bound_digest(self, arg: int, /) -> None: ...

    def digest(self) -> int:
        """
        The two conjuncts folded into one value, for diagnostics. Comparing folded digests is weaker than comparing keys -- prefer ==.
        """

    def __eq__(self, arg: DeclarationKey, /) -> bool: ...

    def __hash__(self) -> int: ...

    def __getstate__(self) -> tuple[int, int]: ...

    def __setstate__(self, arg: tuple[int, int], /) -> None: ...

class WarmExtension:
    """
    One opaque engine extension carried by a WarmStartData: a tag naming the producer and meaning, and payload bytes only that producer interprets.
    """

    def __init__(self, tag: str, payload: bytes) -> None: ...

    @property
    def tag(self) -> str: ...

    @tag.setter
    def tag(self, arg: str, /) -> None: ...

    @property
    def payload(self) -> bytes: ...

    @payload.setter
    def payload(self, arg: bytes, /) -> None: ...

    def __eq__(self, arg: WarmExtension, /) -> bool: ...

    def __hash__(self) -> int: ...

    def __getstate__(self) -> tuple[str, bytes]: ...

    def __setstate__(self, arg: tuple[str, bytes], /) -> None: ...

class WarmStartData:
    """
    The engine-neutral warm-start currency: a declared-space primal/dual core, the declaration-identity stamp it was taken under, and opaque engine extensions. Value-semantic and comparable; nothing here interprets an extension's bytes.
    """

    def __init__(self) -> None: ...

    @property
    def primal(self) -> numpy.ndarray: ...

    @primal.setter
    def primal(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def eq_lmults(self) -> numpy.ndarray: ...

    @eq_lmults.setter
    def eq_lmults(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def iq_lmults(self) -> numpy.ndarray: ...

    @iq_lmults.setter
    def iq_lmults(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def bound_lmults(self) -> numpy.ndarray: ...

    @bound_lmults.setter
    def bound_lmults(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def structure_key(self) -> DeclarationKey: ...

    @structure_key.setter
    def structure_key(self, arg: DeclarationKey, /) -> None: ...

    @property
    def extensions(self) -> list[WarmExtension]: ...

    @extensions.setter
    def extensions(self, arg: Sequence[WarmExtension], /) -> None: ...

    def __eq__(self, arg: WarmStartData, /) -> bool: ...

    def __hash__(self) -> int:
        """
        Hashes a cheap, stable subset consistent with == -- the declaration-identity stamp's digest plus the four block sizes -- not the full primal/dual/extension content. Two equal WarmStartData values always hash equal; two unequal values sharing that subset (e.g. differing only in payload values) hash equal too, which is a legal (if collision-prone) hash under Python's contract.
        """

    def __getstate__(self) -> bytes: ...

    def __setstate__(self, arg: bytes, /) -> None: ...

class StageResult:
    """
    One solver stage's outcome: which stage it was, which engine ran it, and the numbers that describe how it finished.
    """

    def __init__(self) -> None: ...

    @property
    def role(self) -> str:
        """"presolve" | "main" | "polish"."""

    @property
    def engine_name(self) -> str:
        """Engine class name, e.g. "InteriorPointSolver"."""

    @property
    def flag(self) -> ConvergenceFlags: ...

    @property
    def iterations(self) -> int: ...

    @property
    def objective(self) -> float:
        """Caller's scale."""

    @property
    def kkt_residual(self) -> float: ...

    @property
    def eq_violation(self) -> float:
        """Max-norm."""

    @property
    def iq_violation(self) -> float:
        """Max-norm."""

    @property
    def wall_time_s(self) -> float: ...

    @property
    def engine_details(self) -> dict[str, float]:
        """Engine-specific numeric annex, as a plain dict."""

    @property
    def engine_notes(self) -> dict[str, str]:
        """Engine-specific string annex, as a plain dict."""

    def __getstate__(self) -> tuple[str, str, ConvergenceFlags, int, float, float, float, float, float, dict[str, float], dict[str, str]]: ...

    def __setstate__(self, arg: tuple[str, str, ConvergenceFlags, int, float, float, float, float, float, Mapping[str, float], Mapping[str, str]], /) -> None: ...

class PhaseResult:
    """
    One OCP phase's slice of a solve, keyed the same way as the OCP itself (index == 0 for a single Phase, no OCP). Every field is a snapshot taken at solve time.
    """

    def __init__(self) -> None: ...

    @property
    def index(self) -> int: ...

    @property
    def var_start(self) -> int: ...

    @property
    def var_count(self) -> int: ...

    @property
    def eq_start(self) -> int: ...

    @property
    def eq_count(self) -> int: ...

    @property
    def iq_start(self) -> int: ...

    @property
    def iq_count(self) -> int: ...

    @property
    def eq_lmults(self) -> numpy.ndarray: ...

    @property
    def iq_lmults(self) -> numpy.ndarray: ...

    @property
    def bound_lmults(self) -> numpy.ndarray:
        """Declared-space signed z = zL - zU slice."""

    def __getstate__(self) -> tuple[int, int, int, int, int, int, int, numpy.ndarray, numpy.ndarray, numpy.ndarray]: ...

    def __setstate__(self, arg: tuple[int, int, int, int, int, int, int, numpy.ndarray, numpy.ndarray, numpy.ndarray], /) -> None: ...

class SolveResult:
    """
    What a solve() call hands back: the deciding convergence flag, every stage that ran, every OCP phase's slice (empty for a bare VF problem), and the declared-space warm-start currency taken from the final deciding stage.
    """

    def __init__(self) -> None: ...

    @property
    def flag(self) -> ConvergenceFlags: ...

    @property
    def stages(self) -> list[StageResult]:
        """Run order: presolve?, main, polish?. A fresh copy on every access."""

    @property
    def phases(self) -> list[PhaseResult]:
        """
        Index-keyed like the OCP; empty for a bare VF problem. A fresh copy on every access.
        """

    @property
    def warm(self) -> WarmStartData:
        """
        Declared-space warm-start payload from the final deciding stage. A fresh copy on every access.
        """

    @property
    def structure_key(self) -> DeclarationKey:
        """The declaration-identity stamp warm was taken under."""

    def converged(self) -> bool:
        """
        True for CONVERGED or ACCEPTABLE -- ACCEPTABLE is convergence to the acceptable tolerance ladder, still a caller-usable answer.
        """

    def __bool__(self) -> bool: ...

    def objective(self) -> float: ...

    def iterations(self) -> int: ...

    def __getstate__(self) -> tuple[ConvergenceFlags, list[StageResult], list[PhaseResult], WarmStartData, DeclarationKey]: ...

    def __setstate__(self, arg: tuple[ConvergenceFlags, Sequence[StageResult], Sequence[PhaseResult], WarmStartData, DeclarationKey], /) -> None: ...

class SqpSolver:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, **kwargs) -> None:
        """
        Construct an SqpSolver, optionally overriding SqpOptions fields by name.

        Every plain-value SqpOptions field is accepted as a keyword argument
        (kkt_tol, feas_tol, max_iter, tr_init, tr_max, tr_min, enable_soc,
        adaptive_mu, start_level, warm_full_step, budget_mode,
        elastic_ladder_early_exit, crash_basis, qp_mode, ssn_prox_carry,
        ssn_certify_from_face, ssn_sigma_rule, ssn_hint_rule,
        ssn_infeasibility_rule). The QP sub-options (``qp``) and the
        globalization-strategy factory (``make_strategy``) are not kwargs-
        exposed; use the property/attribute defaults for those.

        Raises
        ------
        TypeError
            If an unrecognized keyword argument is given, naming it.
        """

    @property
    def kkt_tol(self) -> float: ...

    @kkt_tol.setter
    def kkt_tol(self, arg: float, /) -> None: ...

    @property
    def feas_tol(self) -> float: ...

    @feas_tol.setter
    def feas_tol(self, arg: float, /) -> None: ...

    @property
    def max_iter(self) -> int: ...

    @max_iter.setter
    def max_iter(self, arg: int, /) -> None: ...

    @property
    def tr_init(self) -> float: ...

    @tr_init.setter
    def tr_init(self, arg: float, /) -> None: ...

    @property
    def tr_max(self) -> float: ...

    @tr_max.setter
    def tr_max(self, arg: float, /) -> None: ...

    @property
    def tr_min(self) -> float: ...

    @tr_min.setter
    def tr_min(self, arg: float, /) -> None: ...

    @property
    def enable_soc(self) -> bool: ...

    @enable_soc.setter
    def enable_soc(self, arg: bool, /) -> None: ...

    @property
    def adaptive_mu(self) -> bool: ...

    @adaptive_mu.setter
    def adaptive_mu(self, arg: bool, /) -> None: ...

    @property
    def start_level(self) -> StartLevel: ...

    @start_level.setter
    def start_level(self, arg: StartLevel, /) -> None: ...

    @property
    def warm_full_step(self) -> bool: ...

    @warm_full_step.setter
    def warm_full_step(self, arg: bool, /) -> None: ...

    @property
    def budget_mode(self) -> bool: ...

    @budget_mode.setter
    def budget_mode(self, arg: bool, /) -> None: ...

    @property
    def elastic_ladder_early_exit(self) -> bool: ...

    @elastic_ladder_early_exit.setter
    def elastic_ladder_early_exit(self, arg: bool, /) -> None: ...

    @property
    def crash_basis(self) -> bool: ...

    @crash_basis.setter
    def crash_basis(self, arg: bool, /) -> None: ...

    @property
    def qp_mode(self) -> QpMode: ...

    @qp_mode.setter
    def qp_mode(self, arg: QpMode, /) -> None: ...

    @property
    def ssn_prox_carry(self) -> bool: ...

    @ssn_prox_carry.setter
    def ssn_prox_carry(self, arg: bool, /) -> None: ...

    @property
    def ssn_certify_from_face(self) -> bool: ...

    @ssn_certify_from_face.setter
    def ssn_certify_from_face(self, arg: bool, /) -> None: ...

    @property
    def ssn_sigma_rule(self) -> SsnSigmaRule: ...

    @ssn_sigma_rule.setter
    def ssn_sigma_rule(self, arg: SsnSigmaRule, /) -> None: ...

    @property
    def ssn_hint_rule(self) -> SsnHintRule: ...

    @ssn_hint_rule.setter
    def ssn_hint_rule(self, arg: SsnHintRule, /) -> None: ...

    @property
    def ssn_infeasibility_rule(self) -> SsnInfeasibilityRule: ...

    @ssn_infeasibility_rule.setter
    def ssn_infeasibility_rule(self, arg: SsnInfeasibilityRule, /) -> None: ...

class StartLevel(enum.Enum):
    """
    How much of a previous solve's state a caller intends to feed into the next one -- kCold ignores it, kSeeded trusts values but not provenance, kWarm additionally trusts the globalization state, kHot additionally reuses a factorization.
    """

    kCold = 0

    kSeeded = 1

    kWarm = 2

    kHot = 3

class QpMode(enum.Enum):
    """Which QP subproblem solver the SQP driver dispatches to."""

    kWalk = 0

    kSsn = 1

class SsnSigmaRule(enum.Enum):
    """How the SSN proximal/Levenberg-Marquardt shift sigma is sized."""

    kLadder = 0

    kResidualArmed = 1

    kResidualAlways = 2

class SsnHintRule(enum.Enum):
    """What protects the SSN hinted first step."""

    kIterationZeroFree = 0

    kWatchdog = 1

class SsnInfeasibilityRule(enum.Enum):
    """What turns an SSN infeasibility suspicion into an exit."""

    kSymptoms = 0

    kFarkasGated = 1

class IpoptSolver:
    """
    Ipopt as a peer engine handle. Constructible only when the backend is compiled in (ENABLE_IPOPT); otherwise raises RuntimeError.
    """

    def __init__(self) -> None: ...

    @property
    def options(self) -> dict[str, str]:
        """
        String key/value options forwarded verbatim to Ipopt. Reading this attribute returns a copy; assign a whole dict to change it.
        """

    @options.setter
    def options(self, arg: Mapping[str, str], /) -> None: ...

class OptimizationProblemBase:
    @property
    def num_partitions(self) -> int:
        """
        Number of NLP matrix partitions.

        Assignment routes through :meth:`set_num_partitions` and raises
        ``ValueError`` for values < 1. The QP thread count is a separate setting,
        set on whichever engine is passed to :meth:`solve`.
        """

    @num_partitions.setter
    def num_partitions(self, arg: int, /) -> None: ...

    def set_num_partitions(self, num_partitions: int) -> None:
        """
        Set the number of NLP matrix partitions (must be >= 1).

        The QP thread count is a separate setting, set on whichever engine is
        passed to :meth:`solve`.
        """

    def solve(self, engine: object, mode: object = 'optimal', presolve: object | None = False, polish: object | None = None, warm: object | None = None) -> SolveResult:
        """
        Run the engine-driven staged solve: an optional Feasible presolve
        stage, the main stage (``mode``), and an optional Optimal polish stage,
        in that order.

        Parameters
        ----------
        engine : InteriorPointSolver | SqpSolver | IpoptSolver
            The main-stage engine.
        mode : Mode | str, optional
            ``Mode.Optimal``/``"optimal"`` (default) or ``Mode.Feasible``/
            ``"feasible"``.
        presolve : bool | InteriorPointSolver | SqpSolver | IpoptSolver | None, optional
            ``False`` (default) or ``None``: no presolve stage (``None`` is the
            same as ``False``). ``True``: run a Feasible presolve stage on
            ``engine`` itself. An engine instance: run the presolve stage on
            that engine instead (implies presolve).
        polish : InteriorPointSolver | SqpSolver | IpoptSolver | None, optional
            When given, run an Optimal polish stage on this engine after the
            main stage.
        warm : SolveResult | WarmStartData | None, optional
            Declared-space warm-start currency seeding the first stage that
            runs. A usable payload's declaration-identity stamp must match the
            current transcription's, or the call raises ValueError naming both.
            A payload that is empty, or that carries a non-finite value in any
            block, is not an error: that stage simply runs cold and records why
            in its ``engine_notes["warm"]``.

        Returns
        -------
        SolveResult
            The deciding convergence flag, every stage that ran, every OCP
            phase's slice, and the warm-start currency from the final deciding
            stage.

        Raises
        ------
        ValueError
            Per the refusal matrix (mode/presolve/polish combinations, a stale
            warm stamp, an engine already solving), if ``engine``/``presolve``/
            ``polish``/``mode``/``warm`` is not one of the types listed above,
            or whatever the dispatched engine itself raises for a malformed
            problem.
        """

    def set_jet_job(self, prototype: object, mode: object = 'optimal', presolve: object | None = False, polish: object | None = None, warm: object | None = None) -> None:
        """
        Stage a batched (Jet) solve on this problem.

        jet_run() -- which ``Jet.map`` calls on each pool-worker job -- clones
        ``prototype``, and any staged ``presolve``/``polish`` engine, and runs
        the staged options against the clones; none of the staged engines is
        ever run itself, so several problems in one ``Jet.map`` batch can
        safely share any of them.

        Every ``InteriorPointSolver`` clone jet_run() makes is pinned to
        ``qp_threads=1`` (the shared thread pool already parallelizes across
        jobs) and forced to a silent ``print_level`` UNLESS the staged engine
        had already moved it off the class default of 0 -- so a batch is quiet
        by default, and you opt into per-job console output by setting
        ``print_level`` explicitly on whichever engine you stage.

        Parameters
        ----------
        prototype : InteriorPointSolver | SqpSolver | IpoptSolver
            Non-owning; kept alive for as long as this problem is (so a caller
            need not hold its own reference past this call).
        mode : Mode | str, optional
            ``Mode.Optimal``/``"optimal"`` (default) or ``Mode.Feasible``/
            ``"feasible"``.
        presolve : bool | InteriorPointSolver | SqpSolver | IpoptSolver | None, optional
            ``False`` (default) or ``None``: no presolve stage. ``True``: run a
            Feasible presolve stage on a clone of ``prototype``. An engine
            instance: run the presolve stage on a clone of that engine instead
            (implies presolve); kept alive the same way as ``prototype``.
        polish : InteriorPointSolver | SqpSolver | IpoptSolver | None, optional
            When given, run an Optimal polish stage (on a clone) after the main
            stage. Kept alive the same way as ``presolve``.
        warm : SolveResult | WarmStartData | None, optional
            Declared-space warm-start currency seeding the first stage that
            runs on every jet_run() call. The payload is copied into this
            problem's own storage at staging time, so (unlike ``prototype``/
            ``presolve``/``polish``) the source object need not be kept alive
            at all past this call.

        Raises
        ------
        ValueError
            If ``mode=Feasible`` is combined with ``presolve``/``polish``
            (checked eagerly, at this call), or if ``prototype``/``presolve``/
            ``polish``/``mode``/``warm`` is not one of the types listed above.
        """

class Jet:
    @overload
    @staticmethod
    def map(arg: Sequence[OptimizationProblemBase], /) -> list[OptimizationProblemBase]: ...

    @overload
    @staticmethod
    def map(arg0: Callable[[object], OptimizationProblemBase], arg1: Sequence[tuple], /) -> list[OptimizationProblemBase]: ...

    @overload
    @staticmethod
    def map(arg0: Sequence[OptimizationProblemBase], arg1: bool, /) -> list[OptimizationProblemBase]: ...

    @overload
    @staticmethod
    def map(arg0: Callable[[object], OptimizationProblemBase], arg1: Sequence[tuple], arg2: bool, /) -> list[OptimizationProblemBase]: ...

class OptimizationProblem(OptimizationProblemBase):
    def __init__(self) -> None: ...

    def set_vars(self, arg: numpy.ndarray, /) -> None: ...

    def return_vars(self) -> numpy.ndarray: ...

    @overload
    def add_equal_con(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: Sequence[numpy.ndarray], /) -> int: ...

    @overload
    def add_equal_con(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: numpy.ndarray, /) -> int: ...

    @overload
    def add_inequal_con(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: Sequence[numpy.ndarray], /) -> int: ...

    @overload
    def add_inequal_con(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: numpy.ndarray, /) -> int: ...

    @overload
    def add_objective(self, arg0: _tychopy.vector_functions.ScalarFunction, arg1: Sequence[numpy.ndarray], /) -> int: ...

    @overload
    def add_objective(self, arg0: _tychopy.vector_functions.ScalarFunction, arg1: numpy.ndarray, /) -> int: ...
