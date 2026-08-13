# Interior-point solver robustness corpus

A corpus of optimal-control / NLP problems where today's interior-point solver is expected to
struggle (poor scaling, redundant/conflicting constraints, cold-started or
degraded initial guesses, classic literature counterexamples for interior-
point / SQP methods, ...), plus a scoring harness
(`scripts/run_corpus.py`) that runs every problem and records how the interior-point solver's
defaults behave on it.

**The corpus never gates anything.** Problems here are *expected* to fail,
diverge, or time out on current defaults — that is the point. Every
globalization/robustness change that follows uses this corpus + harness as
its evidence engine for whether the change actually helps, by diffing two
scorecards (`--diff`). The only thing that *does* gate is the smoke test
(`tychopy/test/test_corpus_smoke.py`), which only checks that every problem
module honors the contract below and that the harness runs end-to-end — it
never asserts that a corpus problem converges.

## Running

```bash
conda activate tycho
# Or: conda run -n tycho python scripts/run_corpus.py ...

python scripts/run_corpus.py                                   # run everything registered
python scripts/run_corpus.py --filter deg                      # only modules matching a substring
python scripts/run_corpus.py --out results.jsonl                # custom output path
python scripts/run_corpus.py --config max_iters=200 kkt_tol=1e-8  # tweak the optimizer
python scripts/run_corpus.py --cbwr --repeat 2                 # determinism check
python scripts/run_corpus.py --diff a.jsonl b.jsonl             # compare two runs
python scripts/run_corpus.py --backend ipopt --filter lit_      # drive the literature tier through Ipopt
python scripts/run_corpus.py --backend ipopt --config linear_solver=pardisomkl  # Ipopt options (verbatim strings)
```

`--backend ipopt` requires a Tycho build configured with `-DENABLE_IPOPT=ON`;
the harness checks `tychopy.solvers.ipopt_available()` up front and fails
fast (before spawning any child) if that build support is missing.

Run from the repo root, in the `tycho` conda environment. **Stale user-site
trap:** if you have ever `pip install --user`-ed a tychopy build, Python's
user-site directory can shadow the conda env's `_tychopy` extension module.
Before trusting any run, verify with:

```bash
conda run -n tycho python -c "import tychopy, _tychopy; print(tychopy.__file__); print(_tychopy.__file__)"
```

`_tychopy.__file__` should point into `<conda env>/lib/python*/site-packages/`,
not `~/.local/lib/python*/site-packages/`.

## The problem-module contract

Every module under `tests/corpus/problems/` defines exactly:

```python
TIER = "degenerate" | "hard" | "literature"
TIMEOUT = <int seconds>          # subprocess kill ceiling
SOLVE_MODE = "solve" | "optimize" | "solve_optimize" | "solve_optimize_solve" | "optimize_solve"

def build():
    """Construct the problem (fully, but unsolved) and return it -- a
    Phase or an OptimizationProblem, both of which expose `.optimizer`,
    `.optimize()`/`.solve()`/etc, and `.nlp_solver`/`.ipopt_options`/
    `.last_ipopt_result` for backend selection. Must not call the harness's
    `configure` callback and must not call any solve entry point (optimizer
    knobs the *problem itself* owns do belong here -- see the `build()`
    bullet below), must not plot, must not write files, must not read wall
    clock."""
```

and, optionally, at most one of:

```python
NOTES = "<static string>"          # most modules omit this (implicit "")

def POST_SOLVE(prob) -> str:
    """Called only on the psiopt backend, only after the SOLVE_MODE entry
    point has run, to compute notes that depend on post-solve state (e.g.
    hard_hypersens_stiff's `phase.mesh_converged` check -- the one module
    in the corpus that needs this)."""
```

- `TIER` groups problems by why they're hard: `"degenerate"` (structurally
  ill-posed: redundant/conflicting constraints, zero objectives, near-
  infeasibility), `"hard"` (realistic in-tree examples perturbed into a
  strained regime), `"literature"` (small classic NLP counterexamples for
  interior-point/SQP methods, verified against their cited source).
- `TIMEOUT` is a plain `int` (seconds); the harness passes it straight to
  `subprocess.run(..., timeout=TIMEOUT)` for the child process running this
  problem.
- `SOLVE_MODE` names the entry point the pre-split `build_and_solve` used to
  call (`getattr(prob, SOLVE_MODE)()` on the psiopt backend -- see
  `tests/corpus/driver.py`). Only `"optimize"`, `"solve_optimize"`, and
  `"optimize_solve"` are used by any module today; `"solve"` and
  `"solve_optimize_solve"` are valid contract values with no current user.
- `build()` does everything the old `build_and_solve` did up to (excluding)
  the `configure(...)` call -- construct the ODE/dynamics, the phase or
  `OptimizationProblem`, boundary values/bounds/objective, and any
  optimizer knobs the problem itself sets (tolerances, `opt_ls_mode`,
  `num_partitions`, ...) -- and returns the unsolved problem object.
- The shared driver (`tests/corpus/driver.py::run`) owns everything the old
  `build_and_solve` tail used to do: calling `configure(prob.optimizer)`
  *immediately before* the solve (this is how `--config KEY=VALUE` reaches
  the interior-point solver -- the harness's `configure` does `setattr(optimizer, key, value)`
  for each pair), dispatching `SOLVE_MODE`, and normalizing the result dict.
  A problem module has no way to opt out of `configure` being called on the
  psiopt backend.
- The result dict's `"flag"` is the *name* of the convergence flag
  (`flag.name`, e.g. `"CONVERGED"`), not the enum member itself, so it
  round-trips through JSON without a custom encoder. On the psiopt backend
  this comes from the module's own `SOLVE_MODE` call; on the ipopt backend,
  from `prob.optimize()` (which still returns a `ConvergenceFlags` member
  under that backend).
- `"objective"` / `"iterations"` come from `optimizer.last_obj_val` /
  `optimizer.last_iter_num` on the psiopt backend (`None` if unreachable,
  e.g. the solve raised before those properties were ever populated -- each
  guarded independently by its own `try/except AttributeError`), or from
  `prob.last_ipopt_result.objective` / `.iterations` on the ipopt backend.
- `build()` (and `POST_SOLVE`, if defined) must be silent w.r.t. side
  effects that would make corpus runs non-reproducible or slow: no
  plotting, no file writes, no wall-clock reads. (the interior-point solver's own console
  printing is fine and in fact required on the psiopt backend — see
  "Iteration counting" below.)
- See "Backend selection" below for what changes (and what a module cannot
  control) when `--backend ipopt` is used instead.

`registry.py` exposes `ALL_PROBLEMS: list[str]` — the tier-grouped list of
module names (no package prefix) that both the harness and the smoke test
import as the single source of truth for "what's in the corpus."

## The convergence-flag -> status mapping

Discovered by introspecting `_tychopy.solvers.ConvergenceFlags` in the tycho
conda env (`tychopy/_stubs/_tychopy/solvers.pyi` confirms the same 5 members
statically):

```
>>> import _tychopy
>>> list(_tychopy.solvers.ConvergenceFlags.__members__.items())
[('CONVERGED', <ConvergenceFlags.CONVERGED: 0>),
 ('ACCEPTABLE', <ConvergenceFlags.ACCEPTABLE: 1>),
 ('NOTCONVERGED', <ConvergenceFlags.NOTCONVERGED: 2>),
 ('DIVERGING', <ConvergenceFlags.DIVERGING: 3>),
 ('SINGULAR_KKT', <ConvergenceFlags.SINGULAR_KKT: 4>)]
```

`ConvergenceFlags` is an `enum.IntEnum` with **exactly these five members** —
it is what `phase.optimize()` / `.solve()` / `.solve_optimize()` /
`.optimize_solve()` / `.solve_optimize_solve()` return, and also what
`optimizer.converge_flag` reports after the fact. `phase.optimizer` (a
`_tychopy.solvers.InteriorPointSolver` instance) also exposes `last_iter_num: int` and
`last_obj_val: float` as read-only properties — this is deliberately the
entire per-solve surface the interior-point solver exposes to Python today (no
feasibility/KKT-residual/factorization data); richer diagnostics may arrive
with future interior-point solver diagnostics counters, and this schema may grow then.

The harness maps flag name to JSONL `status` exhaustively:

| `ConvergenceFlags` member | harness `status` |
| --- | --- |
| `CONVERGED` | `converged` |
| `ACCEPTABLE` | `acceptable` |
| `NOTCONVERGED` | `failed` |
| `DIVERGING` | `diverged` |
| `SINGULAR_KKT` | `singular_kkt` |
| *(anything else — should never happen)* | `error` (with the unrecognized name recorded in `notes`) |

Two more `status` values are synthesized entirely by the harness, not from
any flag: `timeout` (the child was killed after exceeding `TIMEOUT`) and
`error` (nonzero child exit code, an uncaught exception, or a malformed/
missing result — see "Child isolation" below).

## The harness CLI (`scripts/run_corpus.py`)

This file is *also* its own subprocess child entry point — there is no
separate child script. Per-problem execution:

1. The parent imports `problems.<name>` just far enough to read its `TIER`
   and `TIMEOUT` (module-level constants only; `build()` is not called in
   the parent process).
2. The parent spawns
   `python scripts/run_corpus.py --_child <name> --_config <json> --_result-file <path> --_backend <psiopt|ipopt>`
   and waits up to `TIMEOUT` seconds.
3. The child imports the module and `tests/corpus/driver.py`, calls
   `driver.run(module, configure, backend=..., backend_options=...)`
   (`--_config`'s JSON doubles as both the psiopt-backend `configure`
   source and the ipopt-backend `backend_options` — the driver only
   consults whichever one the backend actually uses), and writes the
   returned dict as JSON to `--_result-file`.
4. The parent reads that file (if the child exited 0 and it exists), maps
   the flag to a status, and (psiopt backend only) also greps the child's
   captured stdout for the interior-point solver's own iteration-count line (see below) to
   populate `iterations`; on the ipopt backend, `iterations` instead comes
   straight from the child's own result dict (`IpoptRunInfo.iterations` via
   the driver — see "Backend selection" below for why the stdout instrument
   doesn't apply there).

### Why the result is a file, not a printed line

An earlier revision had the child `print()` its JSON result to stdout
behind a sentinel prefix, for the parent to `grep` out of the captured
output. This **does not work**: the interior-point solver's C++ console output goes through
its own buffered stdio stream, and when the whole subprocess's stdout is
captured through a pipe (not a tty), that buffer flushes on a schedule
independent of Python's own `sys.stdout` buffer. The two streams interleave
at the *byte* level, not the *line* level — observed in practice as the
Python-printed JSON landing in the middle of a still-buffered interior-point solver
output line, corrupting both. Writing the result to a dedicated file next
to (not through) the child's stdout sidesteps the race entirely: the parent
only reads that file after the child process has fully exited (so it has
definitely been closed/flushed), and the stdout capture is still used for
regex iteration counting, which does not care about byte-level line
integrity — only about how many times the pattern occurs.

### Iteration counting

On the **psiopt backend**, `iterations` in the JSONL record is **not** taken
from the child's returned dict — it is
`sum(re.findall(r"Iterations : *([0-9]+)", ansi_stripped_stdout))` over the
child's full captured stdout (ANSI SGR sequences stripped first with
`\x1b\[[0-9;]*m`). This is the same instrument proven out in earlier
bitwise-reproducibility (CBWR) work: the interior-point solver's console printer emits a line
of the form `" Iterations : N"` once per solve, whenever `print_level < 2`
(the library default, so problem modules should not raise their print
level above that unless they want to lose this signal). Summing over all
matches
means a problem that calls `optimize()` more than once (e.g. a two-stage
solve) gets its iteration counts combined. `-1` means no match was found
(e.g. the child crashed before ever calling into the interior-point solver, or print_level was
raised too high).

On the **ipopt backend**, the interior-point solver's console printer never runs (Ipopt solves
the identical transcribed NLP directly), so this stdout instrument has
nothing to match. The parent instead trusts the child's own reported
`iterations` (`driver.py`'s ipopt branch reads it straight from
`prob.last_ipopt_result.iterations`).

### `--backend {psiopt,ipopt}`

Selects the NLP solver backend every selected problem is driven through
(default: `psiopt`). `ipopt` requires a Tycho build configured with
`-DENABLE_IPOPT=ON`; the parent process checks
`tychopy.solvers.ipopt_available()` before spawning any child and exits
immediately with a clear message if it's `False`, rather than letting all
17 children fail one at a time. See "Backend selection" below for what a
problem module does and does not control on this path.

### `--config KEY=VALUE ...`

On the **psiopt backend**, applied via `setattr(optimizer, key, value)`
inside the child, immediately before the solve (see the problem-module
contract above). Each `VALUE` is parsed as `int`, then `float`, then left
as a plain `str` (first cast that doesn't raise wins) — e.g.
`--config max_iters=200 kkt_tol=1e-8 opt_ls_mode=L1` sets an int, a float,
and a string respectively. `KEY` must name a *settable* interior-point solver property
(see `tychopy/_stubs/_tychopy/solvers.pyi` for the full list — `max_iters`,
`kkt_tol`, `print_level`, `opt_ls_mode`, ...); `setattr` on a read-only
property (e.g. `last_iter_num`) raises inside the child, which the parent
then records as `status: "error"`.

On the **ipopt backend**, each `VALUE` stays a plain string (no int/float
casting — Ipopt's own option parser does its own type coercion) and the
whole `{KEY: VALUE}` mapping populates `problem.ipopt_options` verbatim
(e.g. `--config linear_solver=pardisomkl tol=1e-8`), applied after the
Ipopt adapter's matched-tolerance baseline so these entries win.

## Backend selection

Every problem module's `build()` is identical regardless of backend — only
the driver's dispatch after `build()` returns changes:

- **psiopt** (default): `configure(prob.optimizer)` then the module's
  `SOLVE_MODE` entry point, exactly reproducing the pre-split behavior.
- **ipopt**: sets `prob.nlp_solver = NLPSolvers.ipopt` and forwards
  `--config` (as verbatim strings) into `prob.ipopt_options`, then always
  calls `prob.optimize()` — a single NLP solve, regardless of the module's
  `SOLVE_MODE` (the staging modes `solve_optimize`/`optimize_solve`/
  `solve_optimize_solve` have no Ipopt analog). `SOLVE_MODE` is recorded in
  `notes` on this path instead of being honored. `NOTES` (when the module
  defines it) is still included in the ipopt-backend notes string, but
  `POST_SOLVE` is skipped on the ipopt backend (it only ever inspects
  psiopt-specific post-solve state, e.g. `phase.mesh_converged`, which is
  orthogonal to what an ipopt-backend solve leaves behind).
  `"objective"`/`"iterations"` come from
  `prob.last_ipopt_result`, not from `optimizer.last_obj_val`/
  `last_iter_num` (those reflect only the most recent interior-point solver run and are
  left untouched by an ipopt-backend solve).

A problem module cannot select or refuse a backend — that is exclusively
the harness's `--backend` flag.

### `--preset NAME`

Reserved for a future named-configuration table (e.g. "aggressive
line-search", "conservative barrier schedule"). No presets are defined yet;
passing `--preset` always raises a clear `SystemExit` pointing at
`--config` instead.

### `--filter SUBSTRING`

Only run problems whose module name (as it appears in `registry.ALL_PROBLEMS`)
contains `SUBSTRING`. Empty string (default) runs everything registered.

### `--cbwr`

Sets `MKL_CBWR=AUTO,STRICT` in every child's environment, for
bitwise-reproducible MKL reductions across repeats/machines.

### `--repeat N`

Runs every selected problem `N` times (default 1), appending one JSONL
record per run — used to check repeat-to-repeat determinism before trusting
a scorecard as a baseline.

### `--diff A.jsonl B.jsonl`

Prints a per-problem table of status and iteration-count changes between
two previously recorded JSONL files, plus summary counts (`unchanged`,
`improved`, `regressed` — ranked `converged < acceptable < failed < diverged
< singular_kkt < timeout < error`, so a lower rank is "better" — plus `only-in-a` /
`only-in-b` for problems present in only one file). If a file has multiple
records for the same problem (e.g. from `--repeat`), the diff compares each
file's *last* recorded run per problem. Does not run anything; exits after
printing.

## JSONL schema

One JSON object per line, one line per (problem, repeat):

```json
{"problem": "...", "tier": "...", "status": "converged|acceptable|failed|diverged|singular_kkt|timeout|error",
 "flag": "..." | null, "iterations": <int, -1 if unmatched>, "wall_s": <float>,
 "objective": <float | null>, "returncode": <int | null>, "notes": "...",
 "backend": "psiopt" | "ipopt"}
```

- `flag` is `null` for the harness-synthesized statuses (`timeout`, and the
  `error` cases that never got as far as a child result at all); otherwise
  it is the raw `ConvergenceFlags` member name the child reported.
- `returncode` is `null` only for `timeout` (the process was killed, it
  never produced an exit code the harness waited on).
- `wall_s` is measured by the parent around the child subprocess call
  (`time.monotonic()` before/after `subprocess.run`) — this is the one
  wall-clock read in the whole system, and it lives in the parent, never in
  a problem module's `build()`.
- `backend` is whichever backend `--backend` selected for this run
  (`"psiopt"` by default). `--diff` default-fills `"psiopt"` for records
  from before this column existed, so old scorecards still compare cleanly.

## Literature tier

The literature tier's problems (`lit_*.py`) are small, static (dynamics-free)
NLPs from the optimization literature, each chosen because a specific class
of NLP algorithm is documented to struggle on it (jamming, the Maratos
effect, a failed constraint qualification, bad scaling, merit-function
cycling). Unlike the degenerate/hard tiers, these are NOT built as an
`ODEBase` + `phase()` with trivial dynamics -- tychopy exposes a first-class
static-NLP container, `tychopy.solvers.OptimizationProblem` (mirrored 1:1
from `_tychopy.solvers.OptimizationProblem`; see
`tychopy/_stubs/_tychopy/solvers.pyi` and its exerciser at
`tychopy/test/test_FullProblems/test_RosenBrock.py`), so every `lit_*`
module uses that directly instead of a workaround phase:

```python
prob = tychopy.solvers.OptimizationProblem()
prob.set_vars([...])                          # initial guess, plain list/ndarray
prob.add_objective(scalar_func, [indices])     # optional -- omit entirely for
                                                # a pure-feasibility problem
prob.add_equal_con(vector_func, [indices])     # g(x) == 0
prob.add_inequal_con(vector_func, [indices])   # g(x) <= 0  (see below)
flag = prob.optimize()
prob.return_vars()
```

`add_objective`/`add_equal_con`/`add_inequal_con` each take a
`tychopy.vector_functions` expression (built from `Arguments(n)`, exactly
as in phase-based constraints) and a list of variable indices into the flat
vector passed to `set_vars`; the expression's input width must equal
`len(indices)`. There is no separate variable-bounds API on
`OptimizationProblem` (unlike `Phase`'s `add_lu_var_bound` etc.) -- simple
bounds like `x_i >= 0` are encoded as ordinary inequality constraints
(`-x_i <= 0`).

**Inequality sign convention**: the interior-point solver's `add_inequal_con` (both on `Phase`
and on `OptimizationProblem`) requires the constraint function to be
`<= 0` at a feasible point -- confirmed in
`doc-legacy/tutorials/PhaseGuide.rst`, "Inequality Constraints" section:
"we assume our function is in the feasible region whenever its value is
negative." A literature-problem constraint stated as `h(x) >= 0` in its
source is therefore encoded here as `-h(x) <= 0`.

Every `lit_*` module's docstring records: the full citation for its source,
the exact verified formulation (quoted from the source where the source
was directly readable, or from a secondary source that itself quotes the
primary verbatim, when the primary was paywalled/unreachable), where it was
verified (URL and/or edition), and any correction made to the originally
assumed starting formula once checked against the verified source.
`lit_cycling.py` (the Chamberlain, Powell, Lemarechal, Pedersen 1982
watchdog-paper cycling example) is **not** present: the paper (Mathematical
Programming Study 16, 1982) is paywalled on SpringerLink with no accessible
preprint, and no secondary source found reproduces its actual motivating
example (as opposed to merely citing the paper's existence/topic) --
literature-tier problems are verified against an accessible source rather
than implemented from memory, so this one was skipped instead. The
corpus's target range is 15-25 problems; 17 without it is within range.

## Smoke-test end-to-end case

The smoke test's harness end-to-end check
(`test_harness_end_to_end_fast_problems` in
`tychopy/test/test_corpus_smoke.py`) originally ran against two throwaway
stub problems (`stub_converges.py` / `stub_fails.py`) that existed solely
to exercise the harness before any real corpus problem landed. Those stubs
were later deleted (removed from `registry.py` and from
`tests/corpus/problems/`) once the real degenerate/hard/literature tiers
existed, and the smoke test was repointed at the two fastest real problems
instead:
`deg_dup_equality` (converges in 3 iterations, ~1 s) and `hard_vanderpol`
(diverges in 1 iteration, ~1 s — the fastest genuine failure in the corpus;
the other candidate failures in the degenerate tier grind through the full
500-iteration `max_iters` cap before reporting `NOTCONVERGED`). See those
two modules' docstrings for their own problem descriptions.

## Campaign sweeps (`scripts/run_campaign.py`)

`scripts/run_campaign.py` sweeps the globalization configuration space over
this corpus: it enumerates the six settings axes, pre-probes each cell against
`Settings::validate()`, runs valid cells through `run_corpus.py` under
`MKL_CBWR` with per-cell, config-hash-keyed scorecards (crash-resumable), and
provides `aggregate` (one table over all complete cells, plus fixed context
rows from named scorecards) and `shortlist` (top solve-rate band, repeat-
stable, capped, intersection-tie-broken) subcommands. The committed evidence
from the 2026-07 sweep lives in `tests/corpus/campaign/` (aggregate CSV/JSON
plus the example-arm capture CSVs); regenerate with `sweep --store <dir>
--repeats 2` followed by `aggregate`/`shortlist` against the same store. See
`scripts/capture_example_iters.py` for the per-example iteration instrument
used by the example arm.
