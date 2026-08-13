# How to troubleshoot a failing solve

InteriorPointSolver does not fail in one way. A solve that does not return `CONVERGED`
has usually failed for a reason that is visible in the iteration table —
residuals climbing, step lengths collapsing, a perturbation column that
never returns to zero — and each of those signatures points at a different
part of the algorithm. This page is organized by what you see, not by what
the code does.

**Before you change anything, try a preset.** The globalization mechanisms
are not independent: filter acceptance requires a monotone barrier
safeguard, restoration composes with the recovery chain, and the inertia
mode changes what the factorization reports back to the ladder. A preset
applies a combination that has been measured on a 17-problem solver corpus,
and for three of them on the 34-example suite. Hand-tuning individual fields is
the *second* move, once a preset has told you which family of mechanism
helps:

```python
phase.optimizer.apply_preset("filter_l1")
flag = phase.solve_optimize()
```

`apply_preset` assigns exactly nine fields and touches nothing else — your
tolerances, iteration caps, threading, and print settings survive it. The
five names, and the exact fields each assigns, are in the
{doc}`solver reference </reference/python/solvers>`. A one-line summary:

| Preset | Reach for it when |
| --- | --- |
| `classic` | You want the shipped defaults back after experimenting. |
| `filter_l1` | Steps are rejected that look like they should have been accepted; or you need a feasibility-restoration certificate. |
| `soc_recovery_l1` | Repeated rejection *and* an ill-conditioned KKT system — this is the everything-on recovery arm. |
| `soc_proximal` | Rank deficiency or heavy Hessian perturbation, without the l1 elastic machinery. |
| `merit_l1` | You want the modernized penalty-based merit test rather than filter or funnel bookkeeping. For a wrong-basin guess, try [call shape](#call-shape-staged-solve-then-single-optimize) as an independent lever before reaching for a different preset. |

`funnel` acceptance appears in none of the five presets; select it by hand
if you want it.

## Read the iteration table first

With `print_level = 0` every iteration prints a row. The columns you will
actually use when diagnosing:

| Column | Field | What a bad value looks like |
| --- | --- | --- |
| `mu Val` | Barrier parameter | Stops decreasing while residuals are still large. |
| `KKT Inf` | Dual (KKT) error | Climbing over several iterations, or frozen. |
| `ECons Inf` / `ICons Inf` | Constraint infeasibility | Plateaus orders of magnitude above `eq_con_tol`. |
| `AlphaP` / `AlphaD` | Primal and dual step lengths | Collapsing toward `1e-6` and below. |
| `LS` | Backtracking trials used this iteration | Pinned at `max_ls_iters` (default `2`). |
| `PPS` | Perturbed pivots in the factorization | Persistently nonzero. |
| `HF` | Factorization attempts this iteration | Repeatedly greater than one. |
| `HPert` | Cumulative Hessian-diagonal perturbation | Grows and never returns to zero. |

`wide_console = True` adds the largest multipliers, the raw step length
(`AlphaT`), and the merit value.
The table is the per-iteration view; the *outcome* view — which mechanism
fired how often, where the time went, whether an evaluation threw — is the
read-only `optimizer.last_*` properties, which is what you want in a
script. The full list, including the `-1` / `-1.0` not-applicable sentinel
convention, is under
{ref}`Diagnostics of the last solve <diagnostics-of-the-last-solve>`. Each
rung below names the ones worth reading for that signature.

## The failure-signature ladder

### The residuals diverge

**What you see.** `KKT Inf` or `ECons Inf` climbing by orders of magnitude
across consecutive iterations, then a `DIVERGING` exit. A non-finite
residual aborts immediately; a merely large one has to persist for three
consecutive iterates before the stage gives up, so a single bad excursion
is not what you are looking at.

**What it usually means.** The Newton step is being taken in a region where
the linearization is not informative. Overwhelmingly the cause is
conditioning rather than globalization: a state in metres sitting next to
one in radians makes every step length a compromise.

**First move.** Fix the scaling before touching the solver — see
{doc}`How to scale variables and constraints </how_to/scaling>`. Supplying
units and enabling auto-scaling resolves more divergences than any preset
will. Only once the problem is scaled is `apply_preset("filter_l1")` or
`apply_preset("soc_recovery_l1")` worth trying: both replace the free-mode
barrier update with the monitored governor, which is what stops a runaway
barrier decrease.

**Second move.** `barrier_governor = BarrierGovernors.monitored` on its
own; a larger `max_ls_iters` so the backtrack has more rungs to fall back
to before the step is taken as-is.

**Diagnostics.** `converge_flag`, `last_iter_num`, and the `AlphaP` /
`AlphaD` columns. Step lengths that stay near `1.0` while residuals climb
mean nothing is restraining the step — the feasibility stage runs
`soe_ls_mode = NOLS` by default, which accepts the full
fraction-to-the-boundary step with no backtrack at all — set it to
`AUGLANG` or `L1` if the stage needs restraining.

### The solve stops at a locally infeasible point

**What you see.** A yellow line reading `Feasibility restoration converged
to a locally infeasible point (infeasibility ... > ...); stopping (not
converged).`, or `Feasibility phase stalled with its restoration budget
exhausted and no relative improvement over the violation at its last
restoration entry`. `ECons Inf` sits on a plateau far above `eq_con_tol`
and stays there.

**What it usually means.** The solver reached a stationary point of the
constraint violation that is not feasible. That is frequently a modelling
result rather than a solver result: boundary conditions that over-specify
the problem, a path bound the dynamics cannot respect, or a mesh too coarse
for any collocated trajectory to satisfy the defects — for the last of
these see
{doc}`How to refine the mesh adaptively </how_to/mesh_refinement>`.

**First move.** Note that *neither message can appear with the default
settings*: both are emitted from the restoration path, and
`restoration_mode` defaults to `off`. Turning restoration on is therefore
partly a diagnostic move — it is what gets the solver to tell you the
point is locally infeasible instead of silently returning `NOTCONVERGED`.
`apply_preset("filter_l1")` (nested l1 elastic restoration) or
`apply_preset("soc_proximal")` (proximal mode-switch restoration) both do
this.

**Second move.** Swap the two restoration modes against each other —
`l1_nested` relaxes each row with a pair of elastic slacks and has a
cheaper soft pre-stage before it commits, while `proximal_switch` simply
swaps the objective and carries no elastic bookkeeping. Raise
`max_feas_rest` above its default of `2` if the budget is being exhausted.
Then go back to the model and relax the constraint the elastic slacks are
absorbing.

**Diagnostics.** `last_feas_rest_entries` and `last_feas_rest_iters` (both
`-1` when `restoration_mode` is `off`), and the fifth bucket of
`last_recovery_depth_histogram`, which counts rejections that escalated all
the way to restoration.

### Steps are repeatedly rejected

**What you see.** `AlphaP` collapsing toward `1e-8`, `LS` pinned at
`max_ls_iters` every iteration, the objective and the violation trading
places without either improving, and the iteration budget burning down.

**What it usually means.** The acceptance test is rejecting steps that are
in fact good. A full Newton step that reduces constraint violation while
raising the objective (or the reverse) fails a scalar merit test even
though it makes real progress — the Maratos effect. Backtracking makes this
worse rather than better, because the shortened step inherits the same
defect.

**First move.** `apply_preset("soc_recovery_l1")`, which enables the whole
recovery chain: second-order correction (`max_soc = 4`), extended
backtracking (`ls_extended_iters = 2`), and the watchdog. Alternatively
`apply_preset("filter_l1")` — a filter is the direct structural answer,
since it stops requiring a single scalar to improve at all.

**Second move.** Enable the links individually, in dispatch order:
`max_soc = 4` first (it is the cheapest — one back-substitution on the
already-computed factorization), then `ls_extended_iters = 2`, then
`watchdog = True`. Raising `max_ls_iters` and lowering `alpha_red` widen
the plain backtrack but do not address the cause.

**Diagnostics.** `last_soc_steps` and `last_watchdog_activations` tell you
whether the mechanism you enabled actually fired — both are ordinary
counters, so `0` means "enabled and never fired" *or* "off".
`last_recovery_depth_histogram` is the one to read: its buckets are
`[second-order correction, extended backtracking, watchdog, unresolved,
restoration]`, and a large "unresolved" bucket means the chain is being
dispatched and losing. Under filter acceptance, `last_filter_size` and
`last_filter_resets` show whether the filter is accumulating pairs or being
cleared repeatedly.

### Rank-deficiency or heavy-perturbation warnings

**What you see.** `Warning: Potential Rank Deficiency Detected` on the
console; `HPert` growing and never returning to zero; `HF` above one on
most iterations; `PPS` persistently nonzero; and, in the exit block,
`KKT Factor Status : NumericalIssue`.

**What it usually means.** The KKT matrix does not have the inertia the
algorithm expects. The constraint Jacobian may be rank-deficient — a
duplicated or linearly dependent constraint, an over-specified boundary
condition, a control that does not actually appear in the dynamics — or the
Hessian block is strongly indefinite at that iterate.

**First move.** `apply_preset("soc_proximal")` or
`apply_preset("soc_recovery_l1")`. Both select
`inertia_mode = proximal_regularization`, which bakes a small persistent
primal shift and an always-on barrier-scaled dual shift into the base
matrix instead of discovering every iteration that the unperturbed
factorization has the wrong inertia. The dual shift on the constraint-row
diagonals is what makes a rank-deficient Jacobian factorizable.

**Second move.** `inertia_mode = InertiaModes.proximal_regularization`
alone. Raise `delta_h` (the first shift the ladder tries) if the ladder is
starting too small and escalating repeatedly. `qp_pivot_perturb` is the
exponent of the threshold `1e-<value>` below which the sparse solver
perturbs a pivot rather than failing the factorization, so a *lower*
exponent perturbs more aggressively.

**Diagnostics.** `last_prox_reg_primal` and `last_prox_reg_dual`, sampled
at the last factorized iteration; the `HPert`, `HF`, and `PPS` columns for
the per-iteration history.

Regularization makes a singular system factorizable. It does not make a
genuinely redundant constraint set well-posed, and it will not tell you
which constraint is redundant. If the warning fires from the very first
iteration, the shifts are treating a symptom — go find the duplicated
constraint.

### A slow crawl near the solution

**What you see.** Residuals hovering one or two decades above tolerance for
hundreds of iterations, `mu Val` no longer decreasing, and an exit at
`max_iters` with `ACCEPTABLE` or `NOTCONVERGED`.

**What it usually means.** Either the barrier parameter is being driven
down faster than the iterate can follow — the free-mode update is
optimistic by design — or the constraint residual has hit a floor set by
the discretization rather than by the optimizer.

**First move.** `apply_preset("filter_l1")` or `apply_preset("soc_proximal")`;
both switch `barrier_governor` to `monitored`, which hands off to a fixed
barrier parameter when free-mode progress stalls and re-enters free mode
once progress resumes.

**Second move.** `barrier_governor = BarrierGovernors.monitored` on its
own. If the constraint residual is the one stuck, tighten the mesh before
tightening the solver — a residual floor at the discretization error will
not move for any solver setting (see
{doc}`How to refine the mesh adaptively </how_to/mesh_refinement>`). If the
answer is good enough where it is, `return_best = True` with an appropriate
`best_criteria` returns the best iterate rather than the last one, and the
`acc_*` tolerances decide when `ACCEPTABLE` is declared.

**Diagnostics.** `last_monotone_switches` and `last_monotone_iters` (both
`-1` unless the governor is `monitored`): many switches with many monotone
iterations means the free mode is not the problem, the subproblems are.
`last_iter_num` against `max_iters` confirms you ran out of budget rather
than stalling.

### The solve raises instead of returning

**What you see.** A `RuntimeError` out of `optimize()` or `solve()`, or a
solve that exits `ACCEPTABLE` earlier than expected, with
`optimizer.last_eval_exception` holding a non-empty message.

**What it usually means.** A trial point could not be evaluated at all — an
iterate that stepped outside an interpolation table's domain, a negative
argument to a square root in the dynamics. The acceptance machinery absorbs
these and treats the trial as rejected, so the solve continues when it can:
to full recovery, to a graceful exit at an already-acceptable iterate, or
into restoration. It raises only when none of those paths is available.

**First move.** Read `last_eval_exception` — it names the failure. Then
give the solver somewhere to go: any preset with a restoration mode
(`filter_l1`, `soc_recovery_l1`, `soc_proximal`) converts a dead end into a
restoration entry.

**Second move.** Bound the variable that leaves the valid domain, so the
fraction-to-the-boundary rule keeps trial points inside it, rather than
relying on the solver to reject its way out.

## Which mechanism to reach for

One row per mechanism, independent of the presets that bundle them. All of
these compose: any acceptance strategy pairs with either barrier governor,
either restoration mode, and the watchdog. The values named below are
members of the enums re-exported from `tychopy.solvers`:

```python
import tychopy.solvers as slv

phase.optimizer.acceptance_strategy = slv.AcceptanceStrategies.filter
phase.optimizer.barrier_governor = slv.BarrierGovernors.monitored
```

| Mechanism | What it does | When it helps | Cost |
| --- | --- | --- | --- |
| `acceptance_strategy = classic_merit` | Fused backtracking search on a scalar merit function. The default. | Baseline. Cheapest per iteration; no extra state. | None — this is the reference behaviour. |
| `acceptance_strategy = merit` | Modernized penalty-based merit test with an explicitly updated penalty parameter. | The classic merit test is rejecting good steps but you do not want filter/funnel bookkeeping. | One penalty parameter of per-solve state; needs a matching `merit_penalty_rule`. |
| `merit_penalty_rule = wmno` / `flexible` | Single penalty value updated from the directional-derivative condition, versus a penalty *interval* where a step is accepted if it improves the merit for any value in the interval. | `flexible` when a single penalty value keeps overshooting between iterations. Read only under `merit`. | `flexible` tracks two parameters instead of one. |
| `acceptance_strategy = funnel` | Collapses acceptance history into one scalar upper bound on constraint violation, tightened as accepted iterates stay inside it. | Repeated rejection where a filter's memory is more than you need. In no preset — hand-select. | Requires `barrier_governor = monitored` (or `never_monotone`). |
| `acceptance_strategy = filter` | Keeps a set of non-dominated (violation, objective) pairs; a trial is acceptable if no stored pair dominates it. | The structural answer to repeated rejection of good steps. | Requires `barrier_governor = monitored`; stores a growing pair set; worst single-example iteration tails of any preset arm. |
| `barrier_governor = monitored` | Watches the KKT error and hands off to a fixed (monotone) barrier parameter when free-mode progress stalls, re-entering free mode when it resumes. | Barrier parameter outrunning the iterate: divergence early, or a stall near the solution. | Monotone phases are slower per unit of progress; resets acceptance state at every barrier event. |
| `restoration_mode = proximal_switch` | On a ladder-exhausted rejection, keeps the same barrier algorithm running but swaps the objective for a proximal term pulling primals back to the entry point. | Feasibility-hostile problems, when you want the simpler of the two modes. | Extra iterations spent ignoring the objective; refuses entry near feasibility or once `max_feas_rest` is spent. |
| `restoration_mode = l1_nested` | Solves the l1 elastic reformulation as a condensed in-place phase reusing the outer KKT system, after a cheaper soft pre-stage. | A stall at a genuinely infeasible point the elastic reformulation can relax productively. | Elastic-slack bookkeeping on every row; the most machinery of any restoration option. |
| `inertia_mode = proximal_regularization` | Bakes a decaying primal shift and an always-on barrier-scaled dual shift into the base matrix each iteration, instead of only shifting on a wrong-inertia report. | Rank deficiency, singular factorizations, persistent perturbation. | Every step is slightly more conservative than the true Newton step. |
| `max_soc = 4` | After a *first-trial* rejection that did not reduce the violation, re-solves the same KKT system on the live factorization with a corrected constraint right-hand side. | Maratos-effect rejection. The cheapest recovery link. | One back-substitution per correction, no refactorization. |
| `ls_extended_iters = 2` | Extra external calls to the acceptance backtrack once the normal cap and second-order correction are exhausted. | Rejection that second-order correction alone does not fix. | Bounded extra trial-point evaluations per rejected step. |
| `watchdog = True` | After repeated shortened iterations, allows a bounded window of trial iterations under relaxed acceptance before reverting to the armed point. | Oscillation where every individual step looks worse but the sequence is going somewhere. | Holds a snapshot of the armed iterate; can spend a window and revert with nothing gained. |

## What these mechanisms are, briefly

Citations below are transcribed from the source headers under
`hven/detail/globalization/` in the solver library, which carry the full
rule-by-rule derivations.

**Filter acceptance and the switching skeleton.** The filter stores
non-dominated (θ, φ) pairs — constraint violation against barrier objective
— and accepts a trial that no stored pair dominates within a margin. It
shares a switching condition and an F-type Armijo test with the funnel
strategy. Both follow Wächter & Biegler, "On the implementation of an
interior-point filter line-search algorithm for large-scale nonlinear
programming", Math. Program. 106(1):25-57 (2006); the acceptability margins
are that paper's Eqs. (18a)/(18b) and augmentation its Eq. (17). The
practical details the paper leaves open — the barrier-objective ceiling
test, the filter-reset heuristic, the exact dominance comparison — follow
the COIN-OR Ipopt reference implementation.

**Funnel acceptance.** One scalar width τ bounds the constraint violation;
every accepted iterate lies inside the funnel, and each accepted
feasibility-improving step tightens it. From Kiessling, Leyffer & Vanaret,
"A Unified Funnel Restoration SQP Algorithm", arXiv:2409.09208, with
constants taken from Vanaret's Uno solver.

**The modernized merit test.** Two penalty rules. WMNO is Waltz, Morales,
Nocedal & Orban, "An interior algorithm for nonlinear optimization that
combines line search and trust region steps", Math. Program. 107:391-408
(2006), §3.1 — a single penalty ν updated from the directional-derivative
condition. Flexible is Curtis & Nocedal, "Flexible penalty functions for
nonlinear constrained optimization", IMA J. Numer. Anal. 28(4):749-769
(2008) — a penalty *interval* [π_l, π_u], where a step is accepted if it
reduces the merit for at least one π in the interval.

**Second-order correction.** Wächter & Biegler 2006, §2.4. A correction
re-solves the same KKT system on the live factorization with the constraint
block of the right-hand side replaced by an accumulated corrected value,
then re-runs the full acceptance backtrack on the corrected direction, so
the corrected trial faces the same acceptance criteria the ordinary step
faced.

**The watchdog.** The Chamberlain, Powell, Lemaréchal & Pedersen (1982)
watchdog technique ("The watchdog technique for forcing convergence in
algorithms for constrained optimization", Mathematical Programming Study
16, 1-17), with arming and trial-window constants from the reference
interior-point implementation in Wächter & Biegler (2006).

**The monitored barrier governor.** Reproduces the KKT-error path of
Ipopt's adaptive (free ↔ fixed) barrier strategy, with the fixed mode being
the standalone Fiacco-McCormick monotone update; constants are Ipopt's
shipped option defaults, pinned to a tagged release in the header.

**Restoration.** The nested l1 mode's elastic reformulation, penalty
parameter, proximity weight, reference scaling, and closed-form elastic
slack initialization are transcribed from a pinned coin-or/Ipopt commit
(`IpRestoIpoptNLP`, `IpRestoIterateInitializer`, `IpRestoMinC_1Nrm`). The
proximal mode-switch concept — same barrier algorithm, objective swapped
for a scalar-weighted proximal term rather than a nested solve — follows
Knitro's documented `bar_switchobj=scalarprox` behaviour, with entry and
exit mechanics derived from Ipopt's restoration phase and Uno's phase
switching.

**Proximal primal-dual regularization.** The dual shift matches Ipopt's
`perturb_always_cd` semantics with its shipped
`jacobian_regularization_value` / `_exponent` constants (Wächter & Biegler,
Math. Program. 106(1):25-57, 2006). The persistent primal shift is in the
Cipolla-Gondzio / Friedlander-Orban lineage of proximal-stabilized
interior-point methods (S. Cipolla & J. Gondzio, arXiv:2205.01775, 2022 /
JOTA 197 (2023) 1061-1103; M. P. Friedlander & D. Orban, Math. Prog. Comp.
4 (2012) 71-107); its per-iteration decay rule is a Tycho composition and
carries no direct literature reference.

(call-shape-staged-solve-then-single-optimize)=
## Call shape: staged solve, then single optimize

The staged shape is the default and should stay the default:

```python
flag = phase.solve_optimize()     # feasibility stage, then optimization
```

The feasibility stage runs the `SOE` algorithm mode: the objective is
dropped from the KKT system and the primal block is shifted, making the
step a Newton step on the constraints alone. From a rough initial guess
this is the right thing to do — it produces an iterate where the linearized
constraints are meaningful before the objective is allowed to pull on
anything, and it keeps the optimizer from chasing objective reductions
through regions the model does not describe.

Its weakness follows directly from its strength. The feasibility stage is
objective-blind: it lands on whichever feasible point is nearest the guess,
and the optimization stage that follows is a local method that cannot leave
that basin. When the initial guess is in the wrong basin, staging *commits*
you to the wrong basin.

For those problems, hand the whole problem to a single call instead:

```python
flag = phase.optimize()           # objective present from iteration 0
```

The measured case, re-measured after variable bounds became native solver
bounds: under the `merit_l1` configuration — merit acceptance with the
classic adaptive governor and nested l1 restoration — the corpus scores
9 converged + 2 acceptable under the corpus problems' own call shapes. Call
shape still changes individual outcomes: run the same corpus through a
single `optimize()` per problem instead and three problems move.
`hard_mountaincar_badguess` moves from converged at iteration 164 to
acceptable at iteration 195; `hard_hypersens_stiff` moves from acceptable at
iteration 123 to acceptable at iteration 72; and the zermelo wrong-basin
problem moves from **diverging** to **failing** — worse, not better, under
native bounds (this problem's own module docstring already flags its exact
iteration count and objective near this failure as order-sensitive from run
to run; the status change is the stable part of the measurement). The
combined corpus total is unchanged either way (11 of 17
converged-or-acceptable); which individual problems land where is not.

So: keep `solve_optimize()` as the default, and do not assume a single
`optimize()` call is a strict improvement — call shape is a lever
independent of the acceptance mechanism, and it can move an individual
problem in either direction. When a solve converges to something you believe
is the wrong answer, or the feasibility stage itself is where the trouble
lives, trying the other call shape is still worth a measurement; just
measure it rather than assume it will help.

## Honest limits

Presets are not silver bullets, and the measurements say so plainly.

Across the 34-example suite, the three example-measured presets
(`soc_proximal` +27%, `filter_l1` +31%, `soc_recovery_l1` +42%) sit at
**median iteration parity** with the stock defaults while raising the
aggregate iteration count and carrying severe single-example tails —
`filter_l1` costs DionysusLowThrust +609% and MinimumTimeToClimb +395%.
What a preset changes is the tail of the distribution, not its centre: it
buys you problems that previously failed, at the price of iterations on
problems that were already fine. That is why none of them is the default.

The corpus also retains problems that nothing in the matrix solves. The
committed-point class is the named example:
BettsLowThrustCentralShooting fails identically under all three preset
arms — a committed-point integrator failure inside the feasibility stage,
recorded as that stage's next frontier rather than as something a
globalization setting can reach. No combination of acceptance strategy,
governor, restoration mode, or inertia mode moves it.

If you have worked the ladder above and the problem still does not solve,
the next move is usually the model: the scaling, the mesh, the constraint
set, or the initial guess — not another solver setting.

## See also

- {doc}`Configuring the solver </tutorials/basics/solver_configuration>` — a
  runnable walkthrough of this ladder on one problem: a failing solve, its
  diagnostics, the preset that fixes it, and the hand-set fields that
  reproduce the preset exactly.
- {doc}`Solvers reference </reference/python/solvers>` — every setting,
  every `last_*` diagnostic and its sentinel convention, the exact fields
  each preset assigns, and the measured preset behaviour table.
- {doc}`How to scale variables and constraints </how_to/scaling>` — the
  first thing to fix when residuals diverge or step lengths collapse.
- {doc}`How to refine the mesh adaptively </how_to/mesh_refinement>` — when
  the constraint residual has hit a discretization floor the solver cannot
  cross.
- {doc}`Direct collocation in Tycho </explanation/direct_collocation>` —
  how the NLP the solver sees is assembled.
