(tutorial-solver-configuration)=
# Configuring the solver

InteriorPointSolver's stock settings solve most problems, and when they do you should leave
them alone. This tutorial is about the other case: a problem where the defaults
stall, and what you do about it. You will take one small, famous NLP from a
failed solve to a converged one — first by applying a named **configuration
preset**, then by setting the same fields by hand — reading the solver's
`last_*` diagnostics at every step to see *why* it failed, and finally ablating
the configuration to find out which mechanism actually rescued it.

The lesson generalizes past this problem. A preset is not magic: it is nine
field assignments on the `InteriorPointSolver` engine you pass to `solve()`,
chosen because those nine mechanisms were measured together. Once you have
seen a preset and its hand-written equivalent produce bit-identical results,
the reference table of presets stops being a menu of incantations and becomes
a list of starting points.

We use a bare NLP rather than a phase to keep the problem three variables wide
and the solve instantaneous, but nothing here is NLP-specific. A
{doc}`phase </tutorials/basics/your_first_phase>` and an
{py:class}`~tychopy.optimal_control.OptimalControlProblem` expose the same
`solve(engine, ...)` call, the same presets, and the same diagnostics on
whichever `InteriorPointSolver` engine you pass in.

Every `{doctest}` block below (the ones showing `>>>` prompts) is executed as
part of Tycho's test suite, so the results shown are real. Each step builds on
the previous one; run them in order.

## Setup

Two modules: `solvers` for the problem container, the presets, and the setting
enumerations, and `vector_functions` (`vf`) to write the objective and
constraints.

```{doctest}
>>> import tychopy.solvers as slv
>>> from tychopy import vector_functions as vf
```

## 1. A problem built to defeat a line search

The Wächter–Biegler counterexample is a three-variable NLP constructed
specifically so that a class of line-search interior-point methods *jams*: the
iterates converge to a point that is neither optimal nor even feasible, and no
amount of extra iterations moves them. It comes from A. Wächter and
L. T. Biegler, "Failure of global convergence for a class of interior point
methods for nonlinear programming", Math. Program. 88(3):565–574 (2000), which
is the paper the filter line-search work answers — so it is the ideal specimen
for a tutorial about acceptance strategies. The constants below are the
instance Benson, Shanno & Vanderbei reproduce in ORFE-00-02 §4.

$$
\min_{x} \; x_1 \quad \text{subject to} \quad
x_1^2 - x_2 - 1 = 0, \quad
x_1 - x_3 - 1 = 0, \quad
x_2 \ge 0, \; x_3 \ge 0.
$$

The two equalities force $x_2 = x_1^2 - 1$ and $x_3 = x_1 - 1$, so the
non-negativity bounds collapse the feasible set to the ray $x_1 \ge 1$ and the
answer is $x^\star = (1, 0, 0)$. The trap is the starting point
$(-2, 3, 1)$: it satisfies $x_2 > 0$ and $x_3 > 0$ but sits on the wrong side
of $x_1 = 1$, so the solver has to travel through infeasible territory to
reach the optimum, and every individual step that reduces the objective makes
the constraint violation worse.

`OptimizationProblem` is the container for an NLP written directly against
VectorFunctions. `set_vars` supplies the initial primal vector;
`add_objective`, `add_equal_con`, and `add_inequal_con` each take a
VectorFunction plus the list of problem-variable indices that feed it — so
`[0, 1]` below means "call this two-input function on $x_1$ and $x_2$".
Inequalities use InteriorPointSolver's $g(x) \le 0$ convention, which is why $x_2 \ge 0$ is
written $-x_2 \le 0$.

```{doctest}
>>> def wachter_biegler():
...     one, two = vf.Arguments(1), vf.Arguments(2)
...     prob = slv.OptimizationProblem()
...     prob.set_vars([-2.0, 3.0, 1.0])
...     prob.add_objective(one[0], [0])
...     prob.add_equal_con(two[0] ** 2 - two[1] - 1.0, [0, 1])
...     prob.add_equal_con(two[0] - two[1] - 1.0, [0, 2])
...     prob.add_inequal_con(-one[0], [1])
...     prob.add_inequal_con(-one[0], [2])
...     return prob
```

Wrapping the construction in a function is deliberate: we build the identical
problem six times below, once per configuration, so that each solve starts
from the same guess with a fresh engine and its own diagnostic record.

```{doctest}
>>> prob = wachter_biegler()
```

## 2. Pin the reproducibility knobs first

Before any configuration experiment, take the parallelism out of the solve. Two
settings introduce it, and they are easy to confuse because both look like
"threads":

- `problem.num_partitions` is how many partitions the NLP's function and
  derivative evaluation is split into, each partition running on its own
  thread. It defaults to a value derived from the machine's core count, so it
  is a property of *where* you ran as much as of *what* you ran.
- the engine's `qp_threads` is the thread count handed to the sparse
  factorization of the KKT system.

Both distribute floating-point work whose accumulation order then depends on
how it was split, and neither is guaranteed bitwise identical from one run —
or one machine — to the next. Pinning both to `1` removes that variable. It
costs wall-clock time on a large problem, so it is a debugging posture rather
than a production one, but while you are comparing two configurations a
difference you cannot attribute to your change is worse than a slow solve. (If
you need reproducibility *and* the threads, the engine's `cnr_mode = True`
pins the sparse solver's conditional-numerical-reproducibility mode instead.)

Be honest about the scale at which this matters: the three-variable problem
below is far too small for either knob to bite, and it produces identical
results pinned or not. On a collocated phase with thousands of variables it is
a different story, which is why the habit is worth forming on a problem where
you can see everything.

We also silence the iteration table with `print_level = 3`; drop that line to
watch the failure happen live.

```{doctest}
>>> stock = slv.IPM()
>>> prob.set_num_partitions(1)
>>> stock.qp_threads = 1
>>> stock.print_level = 3
>>> prob.num_partitions, stock.qp_threads
(1, 1)
```

## 3. Solve with the stock settings

`solve(engine)` runs the engine-driven solve and returns a
{py:class}`~tychopy.solvers.SolveResult`; `result.flag` is the
{py:class}`~tychopy.solvers.ConvergenceFlags` for the final stage that ran.

```{doctest}
>>> result = prob.solve(stock)
>>> str(result.flag)
'ConvergenceFlags.NOTCONVERGED'
```

`NOTCONVERGED` is the flag; it is not yet a diagnosis. The diagnosis lives on
the engine, whose read-only `last_*` properties describe the solve that just
finished. Start with the coarsest question — did the solver run out of budget,
or did it stop for a reason?

```{doctest}
>>> stock.last_iter_num == stock.max_iters
True
```

It exhausted `max_iters`. So where did it get to? The objective *is* $x_1$, so
the objective value and the first primal are the same number, and both are
nowhere near the optimum of `1.0`:

```{doctest}
>>> round(float(stock.last_primals[0]), 3)
-0.968
```

That is the jam. The solver spent its entire budget shuffling around
$x_1 \approx -0.968$ — well outside the feasible ray $x_1 \ge 1$ — because
every trial step it computed was rejected. `last_recovery_depth_histogram`
counts how each rejected step was eventually resolved, in five buckets:
second-order correction, extended backtracking, watchdog, unresolved,
restoration.

```{doctest}
>>> hist = list(stock.last_recovery_depth_histogram)
>>> [i for i, n in enumerate(hist) if n > 0]
[3]
```

Only bucket 3, "unresolved", ever incremented — hundreds of times. That is the
signature of the default configuration: with second-order correction, extended
backtracking, the watchdog, and restoration all disabled, a rejected step has
nowhere to go, so it is shortened and the iteration is spent. The mechanisms
that *could* have resolved it were never in the picture, which the
mechanism-specific diagnostics report with their not-applicable sentinel rather
than with a zero:

```{doctest}
>>> stock.last_feas_rest_entries, stock.last_filter_size
(-1, -1)
```

`-1` means "the mechanism that reports this field was not selected" —
restoration is `off` and acceptance is not `filter`. A `0` in these fields
would have been a real measurement. The counters in the histogram follow the
opposite convention: they are plain counters, so a `0` there means "enabled and
never fired" *or* "off".

## 4. Apply a preset

The failure signature — steps rejected over and over, with nothing to escalate
to — is the one the {doc}`troubleshooting guide </how_to/solver_troubleshooting>`
answers with a filter. A filter stops demanding that a single scalar merit
improve on every step, which is exactly the demand this problem was built to
make impossible.

`apply_preset` applies a named combination. `filter_l1` selects filter
acceptance, the monitored barrier governor, and nested l1 elastic restoration:

```{doctest}
>>> opt = slv.IPM()
>>> tuned = wachter_biegler()
>>> tuned.set_num_partitions(1)
>>> opt.qp_threads = 1
>>> opt.print_level = 3
>>> opt.apply_preset("filter_l1")
>>> opt.acceptance_strategy
AcceptanceStrategies.filter
>>> opt.barrier_governor
BarrierGovernors.monitored
>>> opt.restoration_mode
RestorationModes.l1_nested
```

A preset assigns exactly nine fields and reads nothing else, so the
reproducibility and output settings we chose above survive it untouched:

```{doctest}
>>> opt.qp_threads, opt.print_level, opt.max_iters
(1, 3, 500)
```

That is worth internalizing before you experiment: a preset will never quietly
undo your tolerances, your iteration cap, or your threading. Now solve the
identical problem again.

```{doctest}
>>> result = tuned.solve(opt)
>>> str(result.flag)
'ConvergenceFlags.CONVERGED'
>>> [round(float(v), 4) for v in opt.last_primals]
[1.0, 0.0, 0.0]
>>> abs(float(opt.last_obj_val) - 1.0) < 1e-4
True
```

$x^\star = (1, 0, 0)$, and with a fraction of the budget the stock run burned
through:

```{doctest}
>>> opt.last_iter_num < stock.last_iter_num
True
```

An unrecognized preset name raises `ValueError` and lists the five valid names,
so a typo fails loudly rather than silently leaving the defaults in place:

```{doctest}
>>> try:
...     slv.InteriorPointSolver().apply_preset("robust")
... except ValueError as err:
...     print(type(err).__name__)
ValueError
```

## 5. The same configuration, one field at a time

Nothing in `apply_preset` is privileged. Every field it sets is a public
property with the same validation, so the preset above has an exact
hand-written equivalent. The enumerations come from `tychopy.solvers`:

```{doctest}
>>> byhand_engine = slv.IPM()
>>> byhand = wachter_biegler()
>>> byhand.set_num_partitions(1)
>>> byhand_engine.qp_threads = 1
>>> byhand_engine.print_level = 3
>>> byhand_engine.acceptance_strategy = slv.AcceptanceStrategies.filter
>>> byhand_engine.barrier_governor = slv.BarrierGovernors.monitored
>>> byhand_engine.restoration_mode = slv.RestorationModes.l1_nested
>>> result = byhand.solve(byhand_engine)
>>> str(result.flag)
'ConvergenceFlags.CONVERGED'
```

The three assignments are the only ones that matter here, because `filter_l1`'s
other six fields already hold their default values. And the two solves agree
exactly — same iteration count, same answer:

```{doctest}
>>> byhand_engine.last_iter_num == opt.last_iter_num
True
>>> float(byhand_engine.last_obj_val) == float(opt.last_obj_val)
True
```

So why reach for the preset at all? Because the fields are not independent, and
the presets encode combinations that were measured together. Filter acceptance
in particular is designed to sit above a monotone barrier safeguard, and asking
for it without one is rejected — the whole settings block is re-validated when a
solve starts, not when you assign:

```{doctest}
>>> solo_engine = slv.IPM()
>>> solo = wachter_biegler()
>>> solo.set_num_partitions(1)
>>> solo_engine.qp_threads = 1
>>> solo_engine.print_level = 3
>>> solo_engine.acceptance_strategy = slv.AcceptanceStrategies.filter
>>> try:
...     _ = solo.solve(solo_engine)
... except ValueError as err:
...     print(type(err).__name__)
ValueError
```

The message names the offending pair and both ways out — set
`barrier_governor = monitored`, or set `never_monotone = True` to accept
adaptive-only operation deliberately. Hand-setting fields means you own those
interactions; a preset means someone else already did.

## 6. Read the diagnostics of the successful solve

The converged run has its own record, and it says which mechanisms ran. Start
with the recovery histogram again — the same five buckets, now with the
restoration bucket live:

```{doctest}
>>> hist = list(opt.last_recovery_depth_histogram)
>>> hist[4] > 0
True
```

Rejected steps escalated all the way to feasibility restoration, and the
restoration diagnostics confirm it. These read `-1` under the stock
configuration; here they are real measurements:

```{doctest}
>>> opt.last_feas_rest_entries > 0
True
>>> opt.last_feas_rest_iters > 0
True
```

The filter was working too — it accumulated the non-dominated (violation,
objective) pairs that let it accept steps a scalar merit test would have thrown
away:

```{doctest}
>>> opt.last_filter_size > 0
True
```

So both mechanisms fired. Which one actually broke the jam? **The diagnostics
cannot tell you that.** They report what ran, not what was decisive, and it is
an easy and expensive mistake to read a nonzero counter as an explanation. The
only way to find out is to ablate — turn one mechanism off and re-solve. Drop
restoration, keep filter acceptance (with the monitored governor it requires):

```{doctest}
>>> ablated_engine = slv.IPM()
>>> ablated = wachter_biegler()
>>> ablated.set_num_partitions(1)
>>> ablated_engine.qp_threads = 1
>>> ablated_engine.print_level = 3
>>> ablated_engine.acceptance_strategy = slv.AcceptanceStrategies.filter
>>> ablated_engine.barrier_governor = slv.BarrierGovernors.monitored
>>> str(ablated.solve(ablated_engine).flag)
'ConvergenceFlags.CONVERGED'
>>> abs(float(ablated_engine.last_obj_val) - 1.0) < 1e-4
True
>>> ablated_engine.last_feas_rest_entries
-1
```

The sentinel proves restoration was never in play, and the problem still
solves. Now the other direction — keep nested l1 restoration, revert acceptance
to the default classic merit test:

```{doctest}
>>> other_engine = slv.IPM()
>>> other = wachter_biegler()
>>> other.set_num_partitions(1)
>>> other_engine.qp_threads = 1
>>> other_engine.print_level = 3
>>> other_engine.barrier_governor = slv.BarrierGovernors.monitored
>>> other_engine.restoration_mode = slv.RestorationModes.l1_nested
>>> str(other.solve(other_engine).flag)
'ConvergenceFlags.NOTCONVERGED'
>>> other_engine.last_feas_rest_entries > 0
True
```

Restoration entered, and the solve still failed. **Filter acceptance is the
active ingredient here**; restoration is a safety net that fired and, on this
problem, shortened the run. That is exactly what you would predict from the
history — this is the counterexample the filter line search was designed to
answer — but predicting it and measuring it are different things, and the
measurement cost two re-solves.

Just as usefully, the diagnostics say what did *not* contribute. `filter_l1`
leaves second-order correction and the watchdog disabled, so their counters are
structurally zero, and it keeps the classic inertia mode, so the
proximal-regularization fields report their floating-point sentinel:

```{doctest}
>>> opt.last_soc_steps, opt.last_watchdog_activations
(0, 0)
>>> opt.last_prox_reg_primal, opt.last_prox_reg_dual
(-1.0, -1.0)
```

Reading those two lines together is the habit worth forming. `(0, 0)` are
counters for mechanisms this preset never enabled; `(-1.0, -1.0)` are sentinels
for a mechanism that reports nothing because it was not selected. Neither says
"tried and achieved nothing" — and if you had enabled second-order correction
and *still* saw `last_soc_steps == 0`, that would be the genuinely interesting
result.

Finally, every diagnostic is scoped to one engine and one solve. The stock
problem's record is untouched by everything we did afterwards, which is what
makes side-by-side comparison of two configurations possible at all:

```{doctest}
>>> str(stock.converge_flag), round(float(stock.last_primals[0]), 3)
('ConvergenceFlags.NOTCONVERGED', -0.968)
```

:::{note}
`filter_l1` rescued *this* problem. It is not a better default — across Tycho's
34-example suite it raises the aggregate iteration count by about 31% and
costs some individual examples several times their stock iterations. Presets
change the tail of the distribution, not its centre: they buy you problems that
previously failed, at the price of iterations on problems that were already
fine. That is why the stock configuration is still the stock configuration.
:::

## What you learned

- Pin `problem.set_num_partitions(1)` and the engine's `qp_threads = 1` before
  comparing configurations, so that a difference between two runs is
  attributable to the settings you changed rather than to how the work was
  split across threads.
- A `NOTCONVERGED` flag is the start of the diagnosis, not the end. The
  `last_*` properties say whether the budget ran out, where the iterates
  stalled, and which recovery mechanisms fired.
- `last_recovery_depth_histogram` buckets rejected steps by how they resolved;
  an "unresolved" bucket that dominates means the recovery chain is disabled or
  losing.
- Mechanism diagnostics use `-1` / `-1.0` to mean *not selected*. A `0` from a
  plain counter is ambiguous between "off" and "never fired"; a sentinel is not.
- `apply_preset(name)` assigns nine globalization fields and touches nothing
  else — tolerances, caps, threading, and print settings all survive it.
- Those nine fields are ordinary properties. Setting them by hand reproduces a
  preset exactly, and makes you responsible for the cross-field rules the
  solver validates at solve time.
- Diagnostics tell you which mechanisms *ran*, never which one was decisive.
  Ablate — disable one and re-solve — before you credit a mechanism with a fix,
  and before you carry it into your next problem.

## Next steps

- {doc}`How to troubleshoot a failing solve </how_to/solver_troubleshooting>` —
  the full failure-signature ladder: what each symptom looks like in the
  iteration table, which mechanism answers it, and which diagnostics confirm it
  fired.
- {doc}`Solvers reference </reference/python/solvers>` — every InteriorPointSolver setting
  and its default, the exact nine fields each of the five presets assigns, the
  measured preset behaviour on the corpus and example suite, and the complete
  `last_*` catalog.
- {doc}`How to scale variables and constraints </how_to/scaling>` — the first
  thing to check when residuals diverge, and more often the real fix than any
  solver setting.
- {doc}`Setting up a phase </tutorials/basics/your_first_phase>` — the same
  engine-driven `solve()` call, presets, and diagnostics on an optimal-control
  problem.
