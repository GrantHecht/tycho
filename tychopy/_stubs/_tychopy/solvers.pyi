"""SubModule Containing PSIOPT,NLP, and Solver Flags"""

from collections.abc import Callable, Mapping, Sequence
import enum
from typing import Callable, overload

import numpy

import _tychopy.vector_functions


class PSIOPT:
    @overload
    def __init__(self, arg: "tycho::solvers::NonLinearProgram", /) -> None: ...

    @overload
    def __init__(self) -> None: ...

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
        Message of the most recent trial-point evaluation exception absorbed during the most recent solve call, or the empty string when every evaluation succeeded. A populated value means the acceptance machinery rejected one or more un-evaluable trial steps (for example an iterate that stepped outside an interpolation table's domain) and the solve recovered; a solve that could not recover raises RuntimeError instead. In a multi-phase solve, an earlier phase's message persists on this property even when a later phase aborts, since the diagnostic is written at each phase's close.
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

    def get_convergence_flag(self) -> ConvergenceFlags: ...

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
        KKT inertia-correction / regularization mode: classic (default) reproduces the on-demand inertia ladder bit-for-bit -- each iteration first attempts an unperturbed factorization and only shifts the Hessian diagonal (by increasing amounts) when the factorization reports wrong inertia, with no constraint-block shift. proximal_regularization bakes two shifts into the base matrix every iteration instead of the classic zero-perturbation first attempt: a small persistent, decaying primal base shift on the Hessian diagonal, and an always-on barrier-scaled dual shift on the constraint-row diagonals (suppressed while a nested l1 restoration phase is active, since the elastic pivots already regularize those rows). The same incr_h/decr_h escalation ladder still fires on top when the base attempt has wrong inertia or is singular; a singular base attempt is treated as wrong inertia under this mode (classic only warns and proceeds). See InertiaModes for the full mechanism and literature citations, and last_prox_reg_primal/last_prox_reg_dual for the resulting diagnostics.
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
    def soe_bound_relax(self) -> float: ...

    @soe_bound_relax.setter
    def soe_bound_relax(self, arg: float, /) -> None: ...

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
        Step-acceptance strategy: classic_merit (default) reproduces the original fused backtracking merit line search bit-for-bit; merit switches to the modernized penalty-based acceptance test selected by merit_penalty_rule; funnel switches to a single-scalar bound on constraint violation, tightened while accepted iterates stay within it; filter switches to a (violation, objective) Wachter-Biegler-style filter. merit, funnel, and filter each require max_soc == 0 and ls_extended_iters == 0 (ValueError raised otherwise); watchdog is compatible with all four strategies. These are heuristically-motivated acceptance alternatives, not one another's strict improvement -- compare against classic_merit on your own problem before adopting one.
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

class AcceptanceStrategies(enum.Enum):
    classic_merit = 0

    merit = 1

    funnel = 2
    """
    Single-scalar upper bound on constraint violation (the funnel width), tightened while accepted iterates remain within it (Kiessling, Leyffer & Vanaret funnel formulation, implemented after Uno's funnel). Rejects combination with max_soc > 0 or ls_extended_iters > 0 (ValueError at validate() time); composes with watchdog. Heuristically motivated -- no convergence guarantee is implied; compare against classic_merit and filter on your own problem.
    """

    filter = 3
    """
    (Constraint violation, objective) pair filter with margined dominance (Wachter-Biegler filter line search, Ipopt lineage). Rejects combination with max_soc > 0 or ls_extended_iters > 0 (ValueError at validate() time); composes with watchdog. Heuristically motivated -- no convergence guarantee is implied; compare against classic_merit and funnel on your own problem.
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
    The on-demand inertia ladder inline in factor_impl -- the bit-identical default. Each iteration first attempts an unperturbed factorization and only shifts the Hessian diagonal (by increasing amounts) when the factorization reports wrong inertia. No constraint-block shift.
    """

    proximal_regularization = 1
    """
    Proximal primal-dual regularization: a small persistent, decaying primal base shift (rho_k, floored at 1e-10, the Cipolla-Gondzio floor) on the Hessian diagonal, plus an always-on barrier-scaled dual shift (delta_c = 1e-8 * mu^0.25, Ipopt's jacobian_regularization_value/exponent constants, matching its perturb_always_cd semantics) on the constraint-row diagonals, are baked into the base matrix every iteration in place of the classic zero-perturbation first attempt -- the same escalation ladder still fires on top when the base attempt has wrong inertia or is singular (a singular base attempt is itself treated as wrong inertia under this mode, unlike classic's warn-and-proceed). rho_k decays toward its floor by decr_h each iteration the base attempt sufficed, or persists at the decayed total shift (rho_k plus the ladder's last delta) when the ladder fired. The dual shift is suppressed while a nested l1 restoration phase is active -- the elastic pivots already regularize those constraint rows at a magnitude the dual shift would be negligible against, and stacking it would make the elastic step-recovery algebra inconsistent with the solved system; the proximal mode-switch restoration touches only the primal diagonal, so the dual shift stays on under it. No new tunable constants -- rho_k's floor and the dual shift's scale/exponent are fixed. See last_prox_reg_primal/last_prox_reg_dual for the per-solve diagnostics this mode reports.
    """

class IpoptRunInfo:
    @property
    def ran(self) -> bool: ...

    @property
    def status(self) -> str: ...

    @property
    def normalized(self) -> str: ...

    @property
    def converge_flag(self) -> ConvergenceFlags: ...

    @property
    def iterations(self) -> int: ...

    @property
    def objective(self) -> float: ...

    @property
    def constraint_violation(self) -> float: ...

    @property
    def wall_time_s(self) -> float: ...

class NLPSolvers(enum.Enum):
    """NLP solver backend selector for the solve/optimize entry points."""

    psiopt = 0
    """Built-in interior-point solver (default)."""

    ipopt = 1
    """
    Linked Ipopt on the identical transcribed NLP (requires ENABLE_IPOPT build).
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

class OptimizationProblemBase:
    @property
    def jet_job_mode(self) -> JetJobModes: ...

    @jet_job_mode.setter
    def jet_job_mode(self, arg: JetJobModes, /) -> None: ...

    @property
    def num_partitions(self) -> int:
        """
        Number of NLP matrix partitions.

        Assignment routes through :meth:`set_num_partitions` and raises
        ``ValueError`` for values < 1. Use ``set_num_partitions(n, qp_threads)`` to
        also set the QP thread count.
        """

    @num_partitions.setter
    def num_partitions(self, arg: int, /) -> None: ...

    @property
    def optimizer(self) -> PSIOPT: ...

    @property
    def nlp_solver(self) -> NLPSolvers:
        """
        NLP solver backend for the solve/optimize entry points.

        NLPSolvers.psiopt (default) is the built-in solver, byte-identical to
        previous behavior. NLPSolvers.ipopt runs the identical transcribed NLP
        through a linked Ipopt installation; requires a build configured with
        ENABLE_IPOPT (raises RuntimeError otherwise). The ipopt backend always
        performs a single NLP solve: the feasibility-then-optimize staging modes
        have no Ipopt analog.

        The built-in solver's own diagnostics (``optimizer.last_obj_val``,
        ``optimizer.last_iter_num``, and every other result()-backed property on
        ``optimizer``) reflect only the most recent PSIOPT run and are left
        untouched by an ipopt-backend run -- use ``last_ipopt_result`` as the
        source of truth for diagnostics of the most recent ipopt-backend
        solve.
        """

    @nlp_solver.setter
    def nlp_solver(self, arg: NLPSolvers, /) -> None: ...

    @property
    def ipopt_options(self) -> dict[str, str]:
        """
        String key/value options forwarded verbatim to Ipopt (e.g.
        {"linear_solver": "pardisomkl"}). Applied after the matched-tolerance
        baseline, so entries here win. Ignored by the psiopt backend.
        """

    @ipopt_options.setter
    def ipopt_options(self, arg: Mapping[str, str], /) -> None: ...

    @property
    def last_ipopt_result(self) -> IpoptRunInfo:
        """
        Diagnostics of the most recent ipopt-backend run on this problem
        (sentinel values with ran == False before any such run).
        """

    @overload
    def set_num_partitions(self, num_partitions: int, qp_threads: int) -> None: ...

    @overload
    def set_num_partitions(self, arg: int, /) -> None: ...

    @overload
    def set_jet_job_mode(self, arg: JetJobModes, /) -> None: ...

    @overload
    def set_jet_job_mode(self, arg: str, /) -> None: ...

    def solve(self) -> ConvergenceFlags: ...

    def optimize(self) -> ConvergenceFlags: ...

    def solve_optimize(self) -> ConvergenceFlags: ...

    def solve_optimize_solve(self) -> ConvergenceFlags: ...

    def optimize_solve(self) -> ConvergenceFlags: ...

class JetJobModes(enum.Enum):
    DoNothing = 1

    NotSet = 0

    Solve = 2

    Optimize = 3

    SolveOptimize = 4

    SolveOptimizeSolve = 5

    OptimizeSolve = 6

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
