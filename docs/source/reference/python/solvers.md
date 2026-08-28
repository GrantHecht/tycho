(solvers-python)=
# Solvers

The Python API for Tycho's solver subsystem, exposed through the
`tychopy.solvers` module: the three solve **engines** — `InteriorPointSolver`
(alias `IPM`), the built-in primal-dual interior-point optimizer; `SqpSolver`
(alias `SQP`), the sequential-quadratic-programming engine; and `IpoptSolver`,
a peer engine wrapping a linked Ipopt installation — the `solve()` call
surface every problem container exposes; the `SolveResult`/`StageResult`/
`PhaseResult`/`WarmStartData` value types a solve hands back or accepts; and
the enumerations that configure all of the above. Every symbol below is a
thin re-export of a nanobind-bound C++ type.

```{eval-rst}
.. currentmodule:: tychopy.solvers
```

## The solver object

Every {py:class}`~tychopy.optimal_control.PhaseInterface` and
{py:class}`~tychopy.optimal_control.OptimalControlProblem` derives from
`OptimizationProblemBase`, which owns no engine of its own. You construct an
engine directly and hand it to the container's `solve()` method:

```python
ipm = tycho.solvers.IPM(max_iters=300, print_level=1)
result = phase.solve(ipm)         # -> SolveResult
```

`InteriorPointSolver()`/`IPM()` and `SqpSolver()`/`SQP()` are default-constructible and
carry no reference to any problem — the same instance can be reused across
`solve()` calls on the same or different problems (settings persist between
calls; each call re-validates them). `IpoptSolver()` raises `RuntimeError`
naming `ENABLE_IPOPT` when the running build was not configured with Ipopt
support (see [Ipopt backend](#ipopt-backend-build-optional) below). An engine
that is already mid-solve refuses a second, concurrent `solve()` call — see
the refusal notes on [`solve()`](#problem-containers) below.

Both engine classes accept their settings as constructor keyword arguments,
in addition to the usual read-write properties:

```python
ipm = tycho.solvers.IPM(preset="soc_recovery_l1", max_soc=6, print_level=2)
sqp = tycho.solvers.SQP(kkt_tol=1e-8, max_iter=42, enable_soc=False)
```

An unrecognized keyword, or one whose value cannot convert to the field's
type, raises `TypeError` naming the offending keyword. For `IPM`/
`InteriorPointSolver`, a `preset=` keyword (see
[Configuration presets](#configuration-presets)) is applied **first**, so
later keywords in the same call override individual fields the preset set —
`IPM(preset="soc_recovery_l1", max_soc=6)` starts from the preset and then
raises `max_soc` past what the preset itself sets. `SqpSolver`'s kwargs
constructor accepts every plain-value field of its options (`kkt_tol`,
`feas_tol`, `max_iter`, `tr_init`, `tr_max`, `tr_min`, `enable_soc`,
`adaptive_mu`, `start_level`, `warm_full_step`, `budget_mode`,
`elastic_ladder_early_exit`, `crash_basis`, `qp_mode`, `ssn_prox_carry`,
`ssn_certify_from_face`, `ssn_sigma_rule`, `ssn_hint_rule`,
`ssn_infeasibility_rule`); its QP sub-options and globalization-strategy
factory are not kwargs-exposed.

The `InteriorPointSolver` settings below (termination, barrier, globalization,
regularization, threading, output) are plain properties: assign at any point
before a solve, and the new value takes effect on the next call. Many
properties also have a matching `set_<name>()` method with identical
validation — the twelve tolerances, `max_iters`, `max_acc_iters`,
`max_ls_iters`, `alpha_red`, `bound_fraction`, `print_level`, the
`delta_h`/`incr_h`/`decr_h` ladder, and the mode selectors `opt_bar_mode`,
`soe_bar_mode`, `opt_ls_mode`, `soe_ls_mode`, `qp_ordering_mode`, and
`best_criteria` — plus three grouped setters, `set_tols()`, `set_acc_tols()`,
and `set_hpert_params()`. The rest are assignment-only, including all nine
globalization fields a preset assigns and every `qp_*` parameter except
`qp_ordering_mode`. Assignments are range-checked immediately (`ValueError`
on a bad value); the full settings block, including its cross-field
invariants, is re-validated on every solve entry, and the globalization
components are rebuilt from it there, so a setting changed between two
solves on the same engine takes effect on the second.

### The `solve()` call

Every problem container's `solve()` runs an optional Feasible presolve
stage, a main stage, and an optional Optimal polish stage, in that order,
and returns a {py:class}`~tychopy.solvers.SolveResult`:

```python
result = phase.solve(
    ipm,                    # engine : InteriorPointSolver | SqpSolver | IpoptSolver
    mode="optimal",         # Mode.Optimal/"optimal" (default) or Mode.Feasible/"feasible"
    presolve=False,         # False/None: no presolve; True: presolve on `ipm` itself;
                             # an engine instance: presolve on that engine instead
    polish=None,            # None: no polish stage; an engine instance: Optimal polish on it
    warm=None,               # SolveResult | WarmStartData | None: seeds the first stage that runs
)
```

A "stage" here is one full engine run — for `InteriorPointSolver` that is one full run of the
barrier algorithm with its own barrier mode, line-search mode, and per-phase
diagnostic counters. `result.stages` lists every stage that ran, in order
(`role` is `"presolve"`, `"main"`, or `"polish"`); `result.flag` mirrors the
*final* stage's convergence flag, since that stage's result is what the
caller actually gets. The per-phase diagnostics described under
[Diagnostics](#diagnostics-of-the-last-solve) (`InteriorPointSolver.last_*`) always report
the engine's own most recent run, i.e. the last `InteriorPointSolver` stage that ran across
any `solve()` call on that engine instance.

`SqpSolver` refuses `mode="feasible"` (`ValueError`): the SQP engine has no
feasibility-only mode, and neither does the Ipopt backend. For the same
reason, neither can run the presolve stage — `presolve=sqp`, and
`presolve=True` on a `SqpSolver`/`IpoptSolver` main engine, are refused by
name before anything else happens. `presolve=`/`polish=` combined with
`mode="feasible"` on the main engine are refused the same way — feasibility
mode runs exactly one stage. A `warm=` value whose declaration-identity stamp
does not match the current transcription raises `ValueError` naming both
stamps; see [Warm-starting](#warm-starting-solveresult-and-warmstartdata)
below.

Stages may mix engines: `solve(ipm, polish=sqp)` runs the interior-point
engine and then refines with SQP, and `solve(ipm)` followed by
`solve(sqp, warm=r)` does the same across two calls. Whenever the engine
changes from one stage to the next, the problem is transcribed again first,
so that a layout one engine left behind (the interior-point engine's default
fixed-variable treatment leaves the program on a reduced variable space)
never reaches the next engine, which applies variable bounds directly. The
cost is one extra transcription per crossover.

After a `polish=` stage, the problem holds the *polish* stage's iterate, and
`result.flag` is the polish stage's flag — a polish stage runs to completion
and reports whatever it found, even when that is worse than the main stage's
result. Both stages are on the record in `result.stages`, so gate on
`result.stages[0]` when it is the main stage's outcome you want.

Old call shape → new call shape:

| Old | New |
| --- | --- |
| `phase.optimizer.set_X(v)` / `phase.optimizer.X = v` | `ipm = tycho.solvers.IPM(X=v, ...)`, or set the property on that instance before `solve()` |
| `prob.optimize()` | `prob.solve(ipm)` |
| `prob.solve()` | `prob.solve(ipm, mode="feasible")` |
| `prob.solve_optimize()` | `prob.solve(ipm, presolve=True)` |
| `prob.optimize_solve()` | `r = prob.solve(ipm)`; `if not r: prob.solve(ipm, mode="feasible", warm=r)` |
| `prob.solve_optimize_solve()` | same two-call chain, `presolve=True` on the first call |
| a `ConvergenceFlags` return | `result.flag`, or `bool(result)` for the converged/not-converged question |

## Settings: termination

The four monitored residuals — KKT (dual) error, equality-constraint
infeasibility, inequality-constraint infeasibility, and barrier
complementarity — are compared against three tiers of tolerance.
`CONVERGED` is declared when all four are inside the convergence tolerances.

| Property | Meaning | Default |
| --- | --- | --- |
| `kkt_tol` | Convergence threshold on the KKT (dual) error. | `1e-6` |
| `eq_con_tol` | Convergence threshold on equality-constraint infeasibility. | `1e-6` |
| `ineq_con_tol` | Convergence threshold on inequality-constraint infeasibility. | `1e-6` |
| `bar_tol` | Convergence threshold on barrier complementarity. | `1e-6` |
| `acc_kkt_tol` | Acceptable-tier threshold on the KKT error. | `1e-2` |
| `acc_eq_con_tol` | Acceptable-tier threshold on equality-constraint infeasibility. | `1e-3` |
| `acc_ineq_con_tol` | Acceptable-tier threshold on inequality-constraint infeasibility. | `1e-3` |
| `acc_bar_tol` | Acceptable-tier threshold on barrier complementarity. | `1e-3` |
| `div_kkt_tol` | Divergence threshold on the KKT error. | `1e15` |
| `div_eq_con_tol` | Divergence threshold on equality-constraint infeasibility. | `1e15` |
| `div_ineq_con_tol` | Divergence threshold on inequality-constraint infeasibility. | `1e15` |
| `div_bar_tol` | Divergence threshold on barrier complementarity. | `1e15` |
| `max_iters` | Maximum iterations per stage. | `500` |
| `max_acc_iters` | Length of the trailing window of consecutive acceptable iterates required to declare `ACCEPTABLE`. | `50` |

`ACCEPTABLE` is not a single-iterate verdict: it is declared only when *every*
one of the last `max_acc_iters` iterates had all four residuals inside the
acceptable tolerances. The same per-iterate acceptable test also decides
whether a solve that can no longer evaluate a trial step may exit gracefully
at the acceptable level instead of raising.

`DIVERGING` splits into two verdicts. A non-finite residual aborts the stage
immediately. A finite residual merely past a divergence threshold is treated
as a possibly-recoverable transient and only aborts once three consecutive
iterates are all past threshold — a Tycho policy choice (the persistence
window is an internal constant, not a setting) that keeps single-iteration
Maratos-class excursions from killing an otherwise convergent solve.

`validate()` requires each convergence tolerance to be no larger than its
acceptable tolerance, which in turn must be no larger than its divergence
tolerance.

## Settings: barrier parameters

| Property | Meaning | Default |
| --- | --- | --- |
| `init_mu` | Initial barrier parameter. | `1e-3` |
| `min_mu` | Lower clamp applied after every barrier update. | `1e-12` |
| `max_mu` | Upper clamp applied after every barrier update. | `100.0` |
| `opt_bar_mode` | Barrier-update heuristic used by the optimization stage (see `BarrierModes`). | `LOQO` |
| `soe_bar_mode` | Barrier-update heuristic used by the feasibility stage. | `LOQO` |
| `bound_fraction` | Fraction-to-the-boundary factor limiting how far a step may drive slacks and bound multipliers toward zero. | `0.99` |
| `bound_push` | Floor for the initial slack values chosen at solver initialization. | `1e-3` |
| `neg_slack_reset` | Floor a slack is reset to when an evaluation would leave it non-positive. | `1e-12` |

The barrier *governor* — which decides whether the update above runs in free
mode or hands off to a monotone mode — is a globalization setting, listed
below.

## Settings: globalization

These settings select the step-acceptance test, the barrier governor, the
post-rejection recovery chain, and the feasibility-restoration mode. The nine
fields a [configuration preset](#configuration-presets) assigns are drawn from
this group and from `inertia_mode` in the next one.

| Property | Meaning | Default |
| --- | --- | --- |
| `acceptance_strategy` | Which step-acceptance test runs: the fused classic backtracking merit search, the modernized penalty-based merit test, a funnel bound on constraint violation, or a (violation, objective) filter. | `classic_merit` |
| `merit_penalty_rule` | Penalty-parameter update rule for the modernized merit test. Read only when `acceptance_strategy` is `merit`. | `wmno` |
| `barrier_governor` | Whether the barrier parameter is updated purely in free mode or by the monitored governor that hands off to a monotone mode when free-mode progress stalls. | `classic_adaptive` |
| `never_monotone` | Expert escape hatch: run `funnel`/`filter` above `classic_adaptive` without a monotone safeguard. Contradictory with `barrier_governor=monitored`. | `False` |
| `restoration_mode` | Feasibility-restoration mode entered when the recovery chain exhausts on a rejected step: none, a proximal objective mode-switch, or a nested l1 elastic phase. | `off` |
| `max_feas_rest` | Per-stage cap on restoration entries. `0` refuses restoration entirely; ignored when `restoration_mode` is `off`. | `2` |
| `max_soc` | Maximum second-order corrections attempted after a rejected trial step. `0` disables second-order correction; the recommended enable value is `4`. | `0` |
| `ls_extended_iters` | Extra backtracking trials on the same classic ladder once the normal cap and second-order correction are exhausted. `0` disables extended backtracking. | `0` |
| `watchdog` | Enables the watchdog heuristic, which tolerates a temporarily worse step after repeated rejections instead of shrinking further. | `False` |
| `max_ls_iters` | Backtracking trials per iteration on the classic ladder, before any recovery link. | `2` |
| `alpha_red` | Divisor applied to the trial step length on each backtracking rejection. | `2.0` |
| `opt_ls_mode` | Merit function used by the optimization stage's line search (see `LineSearchModes`). | `AUGLANG` |
| `soe_ls_mode` | Merit function used by the feasibility stage's line search. | `NOLS` |
| `pd_step_strategy` | How the primal and dual maximum step lengths are combined across the primal, slack, equality-multiplier, and inequality-multiplier blocks. | `PrimSlackEq_Iq` |
| `soe_mode` | Which algorithm mode the feasibility stage runs (see `AlgorithmModes`). | `SOE` |

Two cross-field rules are enforced at validation time and raise `ValueError`:

- `funnel` and `filter` acceptance are designed to sit above a monotone
  barrier safeguard, so each requires `barrier_governor=monitored` *or*
  `never_monotone=True`.
- `never_monotone=True` combined with `barrier_governor=monitored` is
  contradictory — the monitored governor already supplies the monotone
  fallback that flag opts out of.

Every acceptance strategy composes with the watchdog, with either restoration
mode, and with either barrier governor otherwise. These are heuristically
motivated alternatives, not one another's strict improvement; measure on your
own problem before adopting one.

## Settings: regularization and inertia correction

| Property | Meaning | Default |
| --- | --- | --- |
| `inertia_mode` | `classic` attempts an unperturbed factorization each iteration, unless the sticky per-phase degeneracy latch pre-applies the constraint-block dual shift; it accepts only exact inertia (`kkt_dim - m, m, 0`) and, on a singularity signal (rank deficiency or `neigs < m`), engages the on-demand constraint-block dual shift and — if inertia is still wrong — escalates through the Hessian-diagonal shift ladder, whose exhaustion aborts the phase as `SINGULAR_KKT` via the recovery chain. `proximal_regularization` instead bakes a persistent decaying primal shift and an always-on barrier-scaled dual shift into the base matrix every iteration, and treats a singular base attempt as wrong inertia. | `classic` |
| `delta_h` | First Hessian-diagonal shift tried when the inertia ladder fires. | `1e-5` |
| `incr_h` | Multiplier that grows the shift on each further wrong-inertia factorization. | `8.0` |
| `decr_h` | Multiplier that shrinks the retained shift after a successful factorization. | `0.333333` |
| `fast_factor_alg` | Cycling heuristic: skip the unperturbed first attempt when the last four consecutive iterations all required perturbation, re-probing periodically for recovered inertia. | `True` |

`inertia_mode` is one of the nine fields a configuration preset assigns; the
other four settings here control the escalation ladder, which is shared by
both modes — under `proximal_regularization` it still fires on top of the base
shifts whenever the base attempt has wrong inertia or is singular.

## Settings: threading and the sparse linear solver

The KKT system is factorized by Intel MKL Pardiso on Linux and Windows builds
and by Apple Accelerate on macOS builds. The settings below are the Pardiso
control parameters; the Accelerate path reads only `qp_ordering_mode`,
`qp_threads`, and `qp_ref_steps`, and ignores the rest.

| Property | Meaning | Default |
| --- | --- | --- |
| `qp_threads` | Thread count handed to the sparse factorization. | `8` |
| `qp_ordering_mode` | Fill-reducing ordering algorithm (see `QPOrderingModes`). | `METIS` |
| `qp_pivot_strategy` | Pivoting scheme for the symmetric indefinite factorization (see `QPPivotModes`). | `TwoByTwo` |
| `qp_pivot_perturb` | Pivot-perturbation exponent: pivots below `1e-<value>` are perturbed rather than failing the factorization. | `8` |
| `qp_ref_steps` | Maximum iterative-refinement steps per solve. `0` disables refinement. | `0` |
| `qp_matching` | Enables Pardiso's maximum weighted matching (`0`/`1`). | `1` |
| `qp_scaling` | Enables Pardiso's nonsymmetric permutation and MPS scaling (`0`/`1`). | `0` |
| `qp_par_solve` | Enables the parallel forward/backward substitution (`0`/`1`). | `0` |
| `cnr_mode` | Pins Pardiso's conditional-numerical-reproducibility thread count to `qp_threads`, making factorizations reproducible run to run. | `False` |
| `qp_print` | Enables the sparse solver's own message output. | `False` |
| `force_qp_analysis` | Re-runs the symbolic analysis phase even when the sparsity pattern has already been analysed. | `False` |

A default-constructed `InteriorPointSolver`/`IPM` reports `qp_threads = 8` — the engine you
construct and pass to `solve()` is not implicitly resized to the machine's
core count, so set `qp_threads` explicitly if `8` is not what you want.
`qp_threads` is distinct from the NLP partition count and from the
process-global thread pool — see {doc}`How to control parallelism and
threading </how_to/threading_model>`.

`qp_scaling` is off by default deliberately: enabling it measured a large
wall-clock win on PolarLT-class problems with many perturbed pivots, but
altering Pardiso's pivoting moved example-suite iteration counts in both
directions — including `TopputtoLowThrust` 203 → 1102 — so it ships as a
per-problem opt-in rather than a default
(see `docs/dev/analysis/2026-07-pr9-pardiso-options.md`).

## Settings: output and best-iterate selection

| Property | Meaning | Default |
| --- | --- | --- |
| `print_level` | `0` full output (problem statistics, iteration table, exit, timing); `1` drops the problem-statistics banner and the iteration table; `2` exit status and warnings only; `3` and above fully silent. | `0` |
| `wide_console` | Prints the wide iteration table, which adds per-iteration multiplier, step-length, and merit columns. | `False` |
| `return_best` | Returns the best iterate seen rather than the final one. | `False` |
| `best_criteria` | Residual that scores "best" under `return_best` (see `BestCriteriaModes`). | `ECONS` |
| `obj_scale` | Scale factor applied to the objective throughout the KKT evaluation. | `1.0` |

(diagnostics-of-the-last-solve)=
## Diagnostics of the last solve

Every `last_*` property is read-only and reads the result record accumulated
by the most recent stage this `InteriorPointSolver` instance ran, whether that stage came
from a `solve(mode=...)` call, a presolve stage, or a polish stage. Counters
are reset at the start of each stage, so the values always describe one
stage, never a running total across stages or calls.

**Outcome and timing.**

| Property | Meaning |
| --- | --- |
| `converge_flag` | Outcome of the most recent solve. |
| `last_iter_num` | Iteration count of the most recent solve. |
| `last_obj_val` | Objective value at the returned iterate. |
| `last_primals` | Primal vector at the returned iterate. |
| `last_total_time` | Total wall-clock time, in seconds. |
| `last_pre_time` | Time spent in pre-processing. |
| `last_func_time` | Time spent in user-function evaluation. |
| `last_kkt_time` | Time spent in KKT factorization and solves. |
| `last_print_time` | Time spent printing. |
| `last_solver_init_time` | One-time sparse-solver initialization, measured before the main timer starts. |
| `last_misc_time` | Derived: total minus the categorized components above, excluding `last_solver_init_time`. Covers callback time, step application, and convergence checks. |

**Recovery counters.** These are ordinary counters, so `0` means "the
mechanism was enabled and never fired" *or* "the mechanism was off".

| Property | Meaning |
| --- | --- |
| `last_soc_steps` | Second-order correction back-substitutions performed. Always `0` unless `max_soc > 0`. |
| `last_watchdog_activations` | Times the watchdog armed. Always `0` unless `watchdog` is enabled. |
| `last_recovery_depth_histogram` | Five-element list counting how each rejected step's recovery resolved: second-order correction, extended backtracking, watchdog, unresolved, restoration. The final bucket only increments when `restoration_mode` is not `off`; the "unresolved" bucket is the only one that increments when every recovery link is disabled. |

**Mechanism diagnostics and the sentinel convention.** The remaining fields
are reported by an optional component, so they carry an explicit
not-applicable sentinel: `-1` for integer fields and `-1.0` for floating-point
fields. A sentinel means either *the mechanism that reports this field was not
selected*, or *it was selected but never ran* — for example, a stage that
converged at its initial iterate ran no acceptance test. Zero is therefore a
real measurement, never a stand-in for "not applicable".

These fields are also **last-stage-only**: they are collected once per stage,
so a multi-stage `solve()` call (a presolve and/or polish stage alongside the
main stage) reports the last stage this engine ran, not a total across
stages — read them from the specific `StageResult` in `result.stages` you
care about instead when a `solve()` call ran more than one stage.

| Property | Sentinel when | Meaning |
| --- | --- | --- |
| `last_funnel_width` | `acceptance_strategy` is not `funnel`, or no acceptance test ran | Final funnel width at the end of the last stage. |
| `last_filter_size` | `acceptance_strategy` is not `filter` | Number of stored (violation, objective) filter pairs at the end of the last stage. |
| `last_filter_resets` | `acceptance_strategy` is not `filter` | Filter-reset-heuristic clears. Scoped per barrier subproblem: under `barrier_governor=monitored` each barrier event also clears the counter, so this reports resets since the last barrier event of the last stage. |
| `last_monotone_switches` | `barrier_governor` is not `monitored` | Free-to-monotone handoffs in the last stage. |
| `last_monotone_iters` | `barrier_governor` is not `monitored` | Iterations spent in monotone mode in the last stage. |
| `last_feas_rest_entries` | `restoration_mode` is `off` | Times feasibility restoration was entered. Counts identically under both restoration modes. |
| `last_feas_rest_iters` | `restoration_mode` is `off` | Iterations spent in the restoration phase. The nested l1 mode has no separate inner/outer iteration split, so this means the same thing under both modes. |
| `last_prox_reg_primal` | `inertia_mode` is not `proximal_regularization`, or that stage converged before its first factorization | Persistent primal base shift applied to the Hessian diagonal at the **last factorized iteration** of the last stage. |
| `last_prox_reg_dual` | same as above | Barrier-scaled dual shift subtracted from the constraint-row diagonals at the last factorized iteration. Reports `0.0` — not the sentinel — when that iteration fell inside a nested l1 restoration phase, where the shift is suppressed. |

The two proximal-regularization fields are sampled at the last *factorized*
iteration rather than at the trailing iterate, because a converged exit
appends a convergence probe that was never factorized; reading the trailing
history entry would report a shift that was never applied.

**Absorbed evaluation errors.**

`last_eval_exception` holds the message of the most recent trial-point
evaluation exception the acceptance machinery absorbed during the last solve,
or the empty string when every evaluation succeeded. A populated value means
the solver rejected one or more un-evaluable trial steps — an iterate that
stepped outside an interpolation table's domain, say — and continued: to full
recovery, to a graceful `ACCEPTABLE`-level exit at an already-acceptable
iterate, or into feasibility restoration. When none of those paths is
available the solve raises `RuntimeError` instead. In a multi-stage solve an
earlier stage's message persists even when a later stage aborts, since the
diagnostic is written at each stage's close.

## Enumerations

### Solve outcome

```{eval-rst}
.. autoclass:: ConvergenceFlags
   :members:
   :undoc-members:
```

### Globalization selectors

The four enums a configuration preset draws from, plus the merit penalty rule.

```{eval-rst}
.. autoclass:: AcceptanceStrategies
   :members:
   :undoc-members:

.. autoclass:: MeritPenaltyRules
   :members:
   :undoc-members:

.. autoclass:: BarrierGovernors
   :members:
   :undoc-members:

.. autoclass:: RestorationModes
   :members:
   :undoc-members:

.. autoclass:: InertiaModes
   :members:
   :undoc-members:
```

### Algorithm and line-search modes

`AlgorithmModes` selects what a stage's KKT system contains: `OPT` assembles
the full objective plus constraints and is what every optimization stage runs;
`OPTNO` assembles the same system with the objective dropped; `SOE` also drops
the objective and shifts the primal block, making the step a Newton step on
the constraints; `INIT` is the augmented evaluation used by the solver's
initialization pass. Only the feasibility stage is configurable, through
`soe_mode`.

`LineSearchModes` selects the merit function the classic backtracking ladder
tests against: the augmented Lagrangian (`AUGLANG`), the Lagrangian (`LANG`),
an l1 penalty (`L1`), or none at all (`NOLS`, which accepts the full
fraction-to-the-boundary step without a backtrack).

`BarrierModes` selects the barrier-parameter update: `PROBE` takes a predictor
step and derives the new parameter from the resulting complementarity, while
`LOQO` derives it directly from the average and minimum complementarity of the
current iterate.

`PDStepStrategies` selects how the primal and dual maximum step lengths are
distributed across the primal, slack, equality-multiplier, and
inequality-multiplier blocks — from applying each block its own limit
(`PrimSlackEq_Iq`) to applying the single minimum everywhere (`AllMinimum`).

```{eval-rst}
.. autoclass:: AlgorithmModes
   :members:
   :undoc-members:

.. autoclass:: LineSearchModes
   :members:
   :undoc-members:

.. autoclass:: BarrierModes
   :members:
   :undoc-members:

.. autoclass:: PDStepStrategies
   :members:
   :undoc-members:
```

### Sparse-solver and result-selection modes

`QPOrderingModes` selects the fill-reducing ordering. `QPPivotModes` selects
the pivoting scheme for the symmetric indefinite factorization: `OneByOne` is
1x1 diagonal pivoting, `TwoByTwo` is Bunch-Kaufman pivoting, which admits 2x2
pivots. `BestCriteriaModes` names the residual that scores the best iterate
under `return_best` — equality-constraint infeasibility,
inequality-constraint infeasibility, KKT error, or the primal objective.

```{eval-rst}
.. autoclass:: QPOrderingModes
   :members:
   :undoc-members:

.. autoclass:: QPPivotModes
   :members:
   :undoc-members:

.. autoclass:: BestCriteriaModes
   :members:
   :undoc-members:
```

### Solve mode

`Mode` selects which objective a `solve()`/`set_jet_job()` call pursues:
`Mode.Optimal` (also accepted as the string `"optimal"`, the default) drives
to optimality; `Mode.Feasible` (`"feasible"`) drives to feasibility only.

```{eval-rst}
.. autoclass:: Mode
   :members:
   :undoc-members:
```

(configuration-presets)=
## Configuration presets

`apply_preset(name)` applies a named globalization-mechanism configuration. It
assigns exactly nine `Settings` fields — `acceptance_strategy`,
`merit_penalty_rule`, `barrier_governor`, `never_monotone`,
`restoration_mode`, `inertia_mode`, `max_soc`, `ls_extended_iters`, and
`watchdog`. No other setting is read or written: tolerances, iteration caps,
QP and threading parameters, and output settings all survive a preset
unchanged. An unrecognized name raises `ValueError` listing the five valid
names.

```python
ipm = tycho.solvers.IPM()
ipm.apply_preset("soc_proximal")
# or, equivalently, at construction time:
ipm = tycho.solvers.IPM(preset="soc_proximal")
```

The five names, each described by its mechanism:

`classic`
: Classic-merit acceptance, the classic adaptive barrier governor,
  restoration off, classic inertia mode, and second-order correction,
  extended backtracking, and the watchdog all disabled. This is the stock
  default configuration — applying it restores the shipped defaults for those
  nine fields, and it is what every other preset is measured against.

`filter_l1`
: Filter acceptance with a monitored barrier governor and nested l1 elastic
  restoration.

`soc_recovery_l1`
: Classic-merit acceptance with a monitored barrier governor,
  proximal-regularization inertia, second-order correction (`max_soc=4`),
  extended backtracking (`ls_extended_iters=2`), the watchdog enabled, and
  nested l1 restoration.

`soc_proximal`
: Classic-merit acceptance with a monitored barrier governor,
  proximal-regularization inertia, second-order correction (`max_soc=4`), and
  proximal-switch restoration.

`merit_l1`
: Modernized merit acceptance with the classic adaptive barrier governor and
  nested l1 restoration.

### Fields assigned

| Preset | `acceptance_strategy` | `barrier_governor` | `restoration_mode` | `inertia_mode` | `max_soc` | `ls_extended_iters` | `watchdog` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `classic` | `classic_merit` | `classic_adaptive` | `off` | `classic` | `0` | `0` | `False` |
| `filter_l1` | `filter` | `monitored` | `l1_nested` | `classic` | `0` | `0` | `False` |
| `soc_recovery_l1` | `classic_merit` | `monitored` | `l1_nested` | `proximal_regularization` | `4` | `2` | `True` |
| `soc_proximal` | `classic_merit` | `monitored` | `proximal_switch` | `proximal_regularization` | `4` | `0` | `False` |
| `merit_l1` | `merit` | `classic_adaptive` | `l1_nested` | `classic` | `0` | `0` | `False` |

The two fields omitted from the table are constant across all five presets:
`merit_penalty_rule` is `wmno` (read only under `merit` acceptance, so it is
inert in four of the five) and `never_monotone` is `False`.

### Measured behaviour

The corpus column below is re-measured after variable bounds became native
solver bounds: bounds no longer lower to inequality rows, which changes what
the barrier subproblem sees on every bound-bearing problem, so the 17-problem
solver corpus was re-run per preset. `classic` is the bit-identical
`Settings{}` default, so its own corpus run *is* its scorecard; the corpus
column is scored as converged plus acceptable. The iteration columns predate
this change and have not been recaptured under native bounds: they come from
the globalization campaign's post-fixes evidence refresh,
`docs/dev/analysis/2026-07-e2-fixes-evidence-refresh.md` (2026-07-26), a
temporary default-flip capture of the 34-example suite with MKL's conditional
bitwise reproducibility pinned, compared against that campaign's own stock
baseline.

| Preset | Corpus (of 17) | Example-suite iterations vs stock | Worst example tails |
| --- | --- | --- | --- |
| `classic` | 10 (8 converged, 2 acceptable) — the corpus baseline | baseline | baseline |
| `filter_l1` | 12 (9 converged, 3 acceptable) | +31% aggregate, median at parity | DionysusLowThrust +609%, MinimumTimeToClimb +395%, MultiPhaseCannon +387% |
| `soc_recovery_l1` | 11 (9 converged, 2 acceptable) | +42% aggregate, median at parity | OptimalDocking +100%, Zermelo +64%, BettsLowThrust +56% |
| `soc_proximal` | 11 (9 converged, 2 acceptable) | +27% aggregate, median at parity | MinimumTimeToClimb +219%, OptimalDocking +100%, MultiPhaseCannon +67% |
| `merit_l1` | 11 (9 converged, 2 acceptable) | not captured | not captured |

All three example-suite arms passed 33 of 34 examples with the same single
failure, BettsLowThrustCentralShooting — a committed-point integrator failure
inside the feasibility stage, present in all three and out of scope for the
fixes those arms measured. Three examples that the reproducibility gates
document as machine-unstable (MultiSpacecraftOptimization, SimpleLowThrust,
ParallelParking) are excluded from the ratio statistics. `merit_l1` was not
captured as an example arm, so it has no iteration-ratio row.

The `merit_l1` corpus score above uses each problem's own call shape, the way
`apply_preset` is normally exercised. Under native bounds it no longer takes
two different scores to describe: the corpus problems' own call shapes and a
single `optimize()` per problem land on the same combined total, though not
on the same per-problem outcomes — call shape still moves individual
problems, including the zermelo wrong-basin case, in both directions; see
{doc}`How to troubleshoot a failing solve </how_to/solver_troubleshooting>`
for the current worked example and the full treatment of the call-shape
lever.

## Problem containers

`OptimizationProblemBase` is the shared base of every solvable object: the
optimal-control {py:class}`~tychopy.optimal_control.PhaseInterface` and
{py:class}`~tychopy.optimal_control.OptimalControlProblem`, and the bare
`OptimizationProblem` container for a hand-assembled NLP. It owns no engine —
every solve names its engine explicitly (see [The `solve()`
call](#the-solve-call) above) — but it does own the NLP partition count and
the batched-solve staging surface.

| Property or method | Meaning |
| --- | --- |
| `num_partitions` | Number of NLP matrix partitions. Assignment raises `ValueError` below 1. |
| `set_num_partitions(n)` | Sets the partition count. The linear solver's thread count is a separate, per-engine setting (`qp_threads` on the `InteriorPointSolver`/`IPM` passed to `solve()`). |
| `solve(engine, mode="optimal", presolve=False, polish=None, warm=None)` | Runs a staged solve and returns a `SolveResult`. See above. |
| `set_jet_job(prototype, mode="optimal", presolve=False, polish=None, warm=None)` | Stages a batched solve for `Jet.map` to run later, with the same argument shapes as `solve()`. |

An engine instance already inside a `solve()` call refuses a second,
concurrent `solve()` call on itself with `ValueError` ("this engine instance
is already inside a solve; engines serve solves sequentially") — engines
serve one solve at a time. `set_jet_job`/`Jet.map` sidestep this: each queued
job clones its own copy of the prototype engine (and any staged presolve/
polish engine) before running, so concurrent pool workers never contend for
one shared engine object.

`OptimizationProblem` adds the bare-NLP construction surface — `set_vars`,
`return_vars`, `add_equal_con`, `add_inequal_con`, and `add_objective` — for
problems written directly against VectorFunctions rather than transcribed from
a phase.

`Jet.map` runs a batch of problems concurrently, either from a prepared list
or from a factory callable plus an argument sequence; each problem runs
whichever solve `set_jet_job` staged on it. Example (adapted from
`examples/python_examples/HangingChain.py`):

```python
def Job(target_length):
    phase = build_phase(target_length)
    ipm = tycho.solvers.IPM()
    # A presolve stage runs the feasibility algorithm first, then the
    # Optimal main stage. set_jet_job() keeps `ipm` alive for the deferred
    # jet_run() call, even after this function returns.
    phase.set_jet_job(ipm, presolve=True)
    return phase

results = tycho.solvers.Jet.map(Job, job_args, True)
```

## Ipopt backend (build-optional)

Builds configured with `ENABLE_IPOPT` can hand the identical transcribed NLP
to a linked Ipopt installation instead of `InteriorPointSolver`/`SqpSolver`, which makes
cross-solver comparison possible without re-modelling the problem. Construct
an `IpoptSolver` and pass it to `solve()`/`set_jet_job()` like any other
engine:

```python
ipopt_available = tycho.solvers.ipopt_available()   # True iff this build has Ipopt support
ipopt = tycho.solvers.IpoptSolver()                  # raises RuntimeError, naming ENABLE_IPOPT,
                                                      # if the running build lacks it
ipopt.options = {"tol": "1e-8"}                      # dict[str, str], forwarded verbatim to Ipopt
result = phase.solve(ipopt)
```

The backend always performs a single NLP solve of the full objective-bearing
problem, because InteriorPointSolver's feasibility-then-optimize staging has no Ipopt
analog — in particular `mode="feasible"` against an `IpoptSolver` engine
minimizes the objective exactly like `mode="optimal"` does. Options set on
`ipopt.options` are applied after a matched-tolerance baseline, so they win;
reading the property returns a copy, so in-place mutation has no effect and
you must assign a whole dict. An `IpoptSolver` reports its outcome the same
way every other engine does, through the returned `SolveResult`'s `stages`
(`engine_name == "Ipopt"`); it carries no `InteriorPointSolver`-shaped `last_*`
diagnostics, since those describe only `InteriorPointSolver`'s own tolerance surface —
Ipopt-specific diagnostics, where present, land in that stage's
`engine_details`/`engine_notes` annex instead.

```{eval-rst}
.. autofunction:: ipopt_available

.. autoclass:: IpoptSolver
   :members:
```

## Solve results and warm-starting

`solve()`/`set_jet_job()`+`Jet.map` hand back a `SolveResult` — the
engine-neutral record of what a solve did:

| Property or method | Meaning |
| --- | --- |
| `flag` | The final stage's `ConvergenceFlags`. |
| `stages` | `list[StageResult]`, run order: presolve?, main, polish?. |
| `phases` | `list[PhaseResult]`, index-keyed like the OCP; empty for a bare VF problem. |
| `warm` | The `WarmStartData` taken from the final deciding stage. |
| `structure_key` | The `DeclarationKey` that `warm` was taken under. |
| `converged()` / `bool(result)` | `True` for `CONVERGED` or `ACCEPTABLE`. |
| `objective()` | The final stage's objective value. |
| `iterations()` | The final stage's iteration count. |

`StageResult` reports one stage's outcome: `role` (`"presolve"` /
`"main"` / `"polish"`), `engine_name`, `flag`, `iterations`, `objective`,
`kkt_residual`, `eq_violation`, `iq_violation` (both max-norm),
`wall_time_s`, and two annex dicts for engine-specific diagnostics that
don't warrant a named field — `engine_details` (`dict[str, float]`) and
`engine_notes` (`dict[str, str]`).

`PhaseResult` reports one OCP phase's slice of the solve: `index`,
`var_start`/`var_count`, `eq_start`/`eq_count`, `iq_start`/`iq_count`, and
the declared-space multiplier slices `eq_lmults`, `iq_lmults`, and
`bound_lmults` (signed `z = zL - zU`).

### Warm-starting

`WarmStartData` is the engine-neutral warm-start currency: a declared-space
primal/dual core (`primal`, `eq_lmults`, `iq_lmults`, `bound_lmults`), the
`DeclarationKey` identity stamp it was taken under (`structure_key`), and a
list of opaque per-engine `extensions` (each a `tag` + `payload` bytes pair
that only its producing engine interprets). Pass a previous `SolveResult` or
`WarmStartData` directly as `warm=` to `solve()`/`set_jet_job()`:

```python
r1 = phase.solve(ipm)                       # cold solve
r2 = phase.solve(ipm, warm=r1)              # warm-started off r1's SolveResult
r3 = phase.solve(ipm, warm=r1.warm)         # equivalently, off the WarmStartData directly
```

The `warm=` value is read, not consumed — the same `SolveResult`/
`WarmStartData` can seed any number of subsequent calls. Its
`structure_key`/`DeclarationKey` must match the current transcription's, or
the call raises `ValueError` naming both keys; this catches, for example,
warm-starting a phase against a `SolveResult` taken before a mesh
refinement changed the transcription's shape.

A payload that is *empty*, or that carries a non-finite value in any block,
is not an error: the stage it would have seeded simply runs cold, from the
problem's own current point, and records why in that stage's
`engine_notes["warm"]`. This is what makes the retry idiom in the migration
table above — `r = prob.solve(ipm)`, then on a non-convergent `r`,
`prob.solve(ipm, mode="feasible", warm=r)` — work in the case it exists for:
a stage that diverged is exactly the one whose export may be non-finite.

Both `SolveResult` and
`WarmStartData` (along with `StageResult`, `PhaseResult`, and
`DeclarationKey`) are picklable, so a warm-start payload can be saved to
disk and reloaded in a later process.

```{eval-rst}
.. autoclass:: SolveResult
   :members:

.. autoclass:: StageResult
   :members:

.. autoclass:: PhaseResult
   :members:

.. autoclass:: WarmStartData
   :members:

.. autoclass:: WarmExtension
   :members:

.. autoclass:: DeclarationKey
   :members:
```
