(solvers-python)=
# Solvers

The Python API for Tycho's solver subsystem, exposed through the
`tychopy.solvers` module: **PSIOPT**, the built-in primal-dual interior-point
optimizer; the problem containers that own an optimizer instance; and the
enumerations that configure both. Every symbol below is a thin re-export of a
nanobind-bound C++ type.

```{eval-rst}
.. currentmodule:: tychopy.solvers
```

## The solver object

You rarely construct a solver yourself. Every
{py:class}`~tychopy.optimal_control.PhaseInterface` and
{py:class}`~tychopy.optimal_control.OptimalControlProblem` derives from
`OptimizationProblemBase`, which owns a `PSIOPT` instance and exposes it
through the read-only `optimizer` property. Configuring a solve means setting
properties on that instance before calling the container's solve entry point:

```python
phase.optimizer.max_iters = 300
phase.optimizer.print_level = 1
flag = phase.optimize()          # -> ConvergenceFlags
```

`PSIOPT()` is default-constructible, but a solver only becomes runnable once a
problem hands it a transcribed NLP — so in practice you always configure
`problem.optimizer` rather than building your own. Calling a solve entry point
on a solver with no NLP raises `RuntimeError`.

Settings are plain properties: assign at any point before a solve, and the new
value takes effect on the next call. Many properties also have a matching
`set_<name>()` method with identical validation — the twelve tolerances,
`max_iters`, `max_acc_iters`, `max_ls_iters`, `alpha_red`, `bound_fraction`,
`print_level`, the `delta_h`/`incr_h`/`decr_h` ladder, and the mode selectors
`opt_bar_mode`, `soe_bar_mode`, `opt_ls_mode`, `soe_ls_mode`,
`qp_ordering_mode`, and `best_criteria` — plus three grouped setters,
`set_tols()`, `set_acc_tols()`, and `set_hpert_params()`. The rest are
assignment-only, including all nine globalization fields a preset assigns and
every `qp_*` parameter except `qp_ordering_mode`. Assignments are
range-checked immediately (`ValueError` on a bad value); the full settings
block, including its cross-field invariants, is re-validated on every solve
entry, and the globalization components are rebuilt from it there, so a
setting changed between two solves on the same optimizer takes effect on the
second.

### Entry points

The `PSIOPT` methods take an initial primal vector and return the primal
vector at the returned iterate; the container methods take no argument and
return a {py:class}`~tychopy.solvers.ConvergenceFlags`.

| Container method | `PSIOPT` method | Stages run |
| --- | --- | --- |
| `solve()` | `solve(x)` | Feasibility stage only (the `soe_mode` algorithm) |
| `optimize()` | `optimize(x)` | Optimization stage only |
| `solve_optimize()` | `solve_optimize(x)` | Feasibility, then optimization |
| `optimize_solve()` | — | Optimization, then a conditional feasibility stage |
| `solve_optimize_solve()` | — | Feasibility, optimization, conditional feasibility |

A "stage" here is one full run of the barrier algorithm with its own barrier
mode, line-search mode, and per-phase diagnostic counters. The per-phase
diagnostics described under [Diagnostics](#diagnostics-of-the-last-solve)
report the *last* stage of a multi-stage call.

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

A default-constructed `PSIOPT` reports `qp_threads = 8`; a problem-owned
optimizer is initialized to `min(8, core count)` instead. `qp_threads` is
distinct from the NLP partition count and from the process-global thread pool
— see {doc}`How to control parallelism and threading </how_to/threading_model>`.

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
by the most recent `solve`/`optimize`/`solve_optimize` call on that optimizer.
Counters are reset at the start of each call, so the values always describe
one solve, never a running total across solves.

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
so a multi-stage call such as `solve_optimize()` reports the last stage's
values rather than a total across stages.

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

### Backend and batch modes

```{eval-rst}
.. autoclass:: NLPSolvers
   :members:
   :undoc-members:

.. autoclass:: JetJobModes
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
phase.optimizer.apply_preset("soc_proximal")
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

The numbers below are transcribed from the globalization campaign's
post-fixes evidence refresh,
`docs/dev/analysis/2026-07-e2-fixes-evidence-refresh.md` (2026-07-26). The
corpus column is the 17-problem solver corpus, scored as converged plus
acceptable. The iteration columns come from temporary default-flip captures of
the 34-example suite with MKL's conditional bitwise reproducibility pinned,
compared against the stock baseline.

| Preset | Corpus (of 17) | Example-suite iterations vs stock | Worst example tails |
| --- | --- | --- | --- |
| `classic` | stock baseline the corpus was measured against | baseline | baseline |
| `filter_l1` | 12 (8 converged, 4 acceptable) | +31% aggregate, median at parity | DionysusLowThrust +609%, MinimumTimeToClimb +395%, MultiPhaseCannon +387% |
| `soc_recovery_l1` | 12 (8 converged, 4 acceptable) | +42% aggregate, median at parity | OptimalDocking +100%, Zermelo +64%, BettsLowThrust +56% |
| `soc_proximal` | 12 (8 converged, 4 acceptable) | +27% aggregate, median at parity | MinimumTimeToClimb +219%, OptimalDocking +100%, MultiPhaseCannon +67% |
| `merit_l1` | 7 converged + 2 acceptable under the corpus modules' own call shapes; 8 + 2 under a single `optimize()` per problem | not captured | not captured |

All three example-suite arms passed 33 of 34 examples with the same single
failure, BettsLowThrustCentralShooting — a committed-point integrator failure
inside the feasibility stage, present in all three and out of scope for the
fixes those arms measured. Three examples that the reproducibility gates
document as machine-unstable (MultiSpacecraftOptimization, SimpleLowThrust,
ParallelParking) are excluded from the ratio statistics. `merit_l1` was not
captured as an example arm, so it has no iteration-ratio row.

`merit_l1`'s two corpus scores differ only in how each problem was called. The
mover is the zermelo problem from a wrong-basin initial guess: it diverges
under that problem's own call shape and converges when the whole problem is
handed to a single `optimize()` call, at iteration 40 to objective
1.7009270229362865 — the value the Ipopt backend agrees on. Call shape rather
than the acceptance mechanism decides that outcome, so a single `optimize()`
is worth trying against a staged solve independently of which preset is
selected; see
{doc}`How to troubleshoot a failing solve </how_to/solver_troubleshooting>`
for the full treatment.

## Problem containers

`OptimizationProblemBase` is the shared base of every solvable object: the
optimal-control {py:class}`~tychopy.optimal_control.PhaseInterface` and
{py:class}`~tychopy.optimal_control.OptimalControlProblem`, and the bare
`OptimizationProblem` container for a hand-assembled NLP. It owns the
`optimizer`, the NLP partition count, the batch-run mode, and the NLP backend
selection.

| Property or method | Meaning |
| --- | --- |
| `optimizer` | The owned `PSIOPT` instance (read-only). |
| `num_partitions` | Number of NLP matrix partitions. Assignment raises `ValueError` below 1. |
| `set_num_partitions(n)` / `set_num_partitions(n, qp_threads)` | Sets the partition count, optionally also the linear solver's thread count. |
| `jet_job_mode` | Which solve entry point `Jet.map` runs for this problem (see `JetJobModes`). |
| `nlp_solver` | NLP backend for the solve entry points (see below). |
| `ipopt_options` | String key/value options (a `dict[str, str]`) forwarded verbatim to Ipopt. |
| `last_ipopt_result` | `IpoptRunInfo` for the most recent Ipopt-backend run. |

`OptimizationProblem` adds the bare-NLP construction surface — `set_vars`,
`return_vars`, `add_equal_con`, `add_inequal_con`, and `add_objective` — for
problems written directly against VectorFunctions rather than transcribed from
a phase.

`Jet.map` runs a batch of problems concurrently, either from a prepared list
or from a factory callable plus an argument sequence; each problem's
`jet_job_mode` selects which solve it runs.

## Ipopt backend (build-optional)

Builds configured with `ENABLE_IPOPT` can hand the identical transcribed NLP
to a linked Ipopt installation instead of PSIOPT, which makes cross-solver
comparison possible without re-modelling the problem. `ipopt_available()`
reports whether the running build has that support; selecting
`nlp_solver = NLPSolvers.ipopt` without it raises `RuntimeError` when the
solve runs.
The backend always performs a single NLP solve of the full objective-bearing
problem, because the feasibility-then-optimize staging modes have no Ipopt
analog — in particular `solve()`, which runs the feasibility-only stage under
PSIOPT, minimizes the objective exactly like `optimize()` under this backend.
Options in `ipopt_options` are applied after a matched-tolerance baseline, so
they win; reading the property returns a copy, so in-place mutation has no
effect and you must assign a whole dict. Batch runs reject this backend
outright: Ipopt is not reliably re-entrant, so a `Jet` job whose problem
selects it raises `ValueError` before that job's solve begins — run the Ipopt
backend one solve at a time. Finally, an Ipopt-backend run leaves every
`optimizer.last_*` property untouched, since those reflect only the most
recent PSIOPT run; `last_ipopt_result` is the source of truth for the most
recent Ipopt solve.

```{eval-rst}
.. autofunction:: ipopt_available

.. autoclass:: IpoptRunInfo
   :members:
```
