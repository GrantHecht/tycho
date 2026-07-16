# PSIOPT robustness corpus (E2 G0)

A corpus of optimal-control / NLP problems where today's PSIOPT is expected to
struggle (poor scaling, redundant/conflicting constraints, cold-started or
degraded initial guesses, classic literature counterexamples for interior-
point / SQP methods, ...), plus a scoring harness
(`scripts/run_corpus.py`) that runs every problem and records how PSIOPT's
defaults behave on it.

**The corpus never gates anything.** Problems here are *expected* to fail,
diverge, or time out on current defaults — that is the point. Every
subsequent E2 PR (G1-G8) uses this corpus + harness as its evidence engine
for whether a globalization/robustness change actually helps, by diffing two
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
```

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

def build_and_solve(configure) -> dict:
    """Construct the problem, call configure(optimizer) immediately before
    optimize/solve_optimize, run it, and return:
      {"flag": <str name of the returned convergence flag>,
       "objective": <float or None>,
       "iterations": <int or None>,     # optimizer.last_iter_num if reachable, else None
       "notes": <str>}                  # anything odd, "" normally
    Must not plot, must not write files, must not read wall clock."""
```

- `TIER` groups problems by why they're hard: `"degenerate"` (structurally
  ill-posed: redundant/conflicting constraints, zero objectives, near-
  infeasibility), `"hard"` (realistic in-tree examples perturbed into a
  strained regime), `"literature"` (small classic NLP counterexamples for
  interior-point/SQP methods, verified against their cited source).
- `TIMEOUT` is a plain `int` (seconds); the harness passes it straight to
  `subprocess.run(..., timeout=TIMEOUT)` for the child process running this
  problem.
- `build_and_solve(configure)` receives a single callable, `configure`,
  which it must call with the live PSIOPT optimizer handle (e.g.
  `phase.optimizer`) *immediately before* the call that actually runs the
  solve (`.optimize()`, `.solve_optimize()`, etc). This is how
  `--config KEY=VALUE` reaches the solver: the harness's `configure` does
  `setattr(optimizer, key, value)` for each pair. A problem module that
  never calls `configure` breaks `--config` for that problem.
- The returned dict's `"flag"` is the *name* of the convergence flag
  (`flag.name`, e.g. `"CONVERGED"`), not the enum member itself, so it
  round-trips through JSON without a custom encoder.
- `"objective"` / `"iterations"` should come from `optimizer.last_obj_val`
  / `optimizer.last_iter_num` when reachable; `None` if not (e.g. the solve
  raised before those properties were ever populated).
- The function must be silent w.r.t. side effects that would make corpus
  runs non-reproducible or slow: no plotting, no file writes, no wall-clock
  reads. (PSIOPT's own console printing is fine and in fact required — see
  "Iteration counting" below.)

`registry.py` exposes `ALL_PROBLEMS: list[str]` — the tier-grouped list of
module names (no package prefix) that both the harness and the smoke test
import as the single source of truth for "what's in the corpus."

## The convergence-flag -> status mapping

Discovered by introspecting `_tychopy.solvers.ConvergenceFlags` in the tycho
conda env (`tychopy/_stubs/_tychopy/solvers.pyi` confirms the same 4 members
statically):

```
>>> import _tychopy
>>> list(_tychopy.solvers.ConvergenceFlags.__members__.items())
[('CONVERGED', <ConvergenceFlags.CONVERGED: 0>),
 ('ACCEPTABLE', <ConvergenceFlags.ACCEPTABLE: 1>),
 ('NOTCONVERGED', <ConvergenceFlags.NOTCONVERGED: 2>),
 ('DIVERGING', <ConvergenceFlags.DIVERGING: 3>)]
```

`ConvergenceFlags` is an `enum.IntEnum` with **exactly these four members** —
it is what `phase.optimize()` / `.solve()` / `.solve_optimize()` /
`.optimize_solve()` / `.solve_optimize_solve()` return, and also what
`optimizer.converge_flag` / `optimizer.get_convergence_flag()` report after
the fact. `phase.optimizer` (a `_tychopy.solvers.PSIOPT` instance) also
exposes `last_iter_num: int` and `last_obj_val: float` as read-only
properties — this is deliberately the entire per-solve surface PSIOPT
exposes to Python today (no feasibility/KKT-residual/factorization data);
richer diagnostics arrive with the E2 diagnostics counters and this schema
may grow then.

The harness maps flag name to JSONL `status` exhaustively:

| `ConvergenceFlags` member | harness `status` |
| --- | --- |
| `CONVERGED` | `converged` |
| `ACCEPTABLE` | `acceptable` |
| `NOTCONVERGED` | `failed` |
| `DIVERGING` | `diverged` |
| *(anything else — should never happen)* | `error` (with the unrecognized name recorded in `notes`) |

Two more `status` values are synthesized entirely by the harness, not from
any flag: `timeout` (the child was killed after exceeding `TIMEOUT`) and
`error` (nonzero child exit code, an uncaught exception, or a malformed/
missing result — see "Child isolation" below).

## The harness CLI (`scripts/run_corpus.py`)

This file is *also* its own subprocess child entry point — there is no
separate child script. Per-problem execution:

1. The parent imports `problems.<name>` just far enough to read its `TIER`
   and `TIMEOUT` (module-level constants only; `build_and_solve` is not
   called in the parent process).
2. The parent spawns
   `python scripts/run_corpus.py --_child <name> --_config <json> --_result-file <path>`
   and waits up to `TIMEOUT` seconds.
3. The child imports the module, calls `build_and_solve(configure)`, and
   writes the returned dict as JSON to `--_result-file`.
4. The parent reads that file (if the child exited 0 and it exists), maps
   the flag to a status, and also greps the child's captured stdout for
   PSIOPT's own iteration-count line (see below) to populate `iterations`.

### Why the result is a file, not a printed line

An earlier revision had the child `print()` its JSON result to stdout
behind a sentinel prefix, for the parent to `grep` out of the captured
output. This **does not work**: PSIOPT's C++ console output goes through
its own buffered stdio stream, and when the whole subprocess's stdout is
captured through a pipe (not a tty), that buffer flushes on a schedule
independent of Python's own `sys.stdout` buffer. The two streams interleave
at the *byte* level, not the *line* level — observed in practice as the
Python-printed JSON landing in the middle of a still-buffered PSIOPT
output line, corrupting both. Writing the result to a dedicated file next
to (not through) the child's stdout sidesteps the race entirely: the parent
only reads that file after the child process has fully exited (so it has
definitely been closed/flushed), and the stdout capture is still used for
regex iteration counting, which does not care about byte-level line
integrity — only about how many times the pattern occurs.

### Iteration counting

`iterations` in the JSONL record is **not** taken from the child's returned
dict — it is `sum(re.findall(r"Iterations : *([0-9]+)", ansi_stripped_stdout))`
over the child's full captured stdout (ANSI SGR sequences stripped first
with `\x1b\[[0-9;]*m`). This is the same instrument proven in the earlier
CBWR work (PR 9): PSIOPT's console printer emits a line of the form
`" Iterations : N"` once per solve, whenever `print_level < 2` (the
library default, so problem modules should not raise their print level
above that unless they want to lose this signal). Summing over all matches
means a problem that calls `optimize()` more than once (e.g. a two-stage
solve) gets its iteration counts combined. `-1` means no match was found
(e.g. the child crashed before ever calling into PSIOPT, or print_level was
raised too high).

### `--config KEY=VALUE ...`

Applied via `setattr(optimizer, key, value)` inside the child, immediately
before the solve (see the problem-module contract above). Each `VALUE` is
parsed as `int`, then `float`, then left as a plain `str` (first cast that
doesn't raise wins) — e.g. `--config max_iters=200 kkt_tol=1e-8 opt_ls_mode=L1`
sets an int, a float, and a string respectively. `KEY` must name a
*settable* PSIOPT property (see `tychopy/_stubs/_tychopy/solvers.pyi` for
the full list — `max_iters`, `kkt_tol`, `print_level`, `opt_ls_mode`, ...);
`setattr` on a read-only property (e.g. `last_iter_num`) raises inside the
child, which the parent then records as `status: "error"`.

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
a scorecard as a baseline (see Task 5).

### `--diff A.jsonl B.jsonl`

Prints a per-problem table of status and iteration-count changes between
two previously recorded JSONL files, plus summary counts (`unchanged`,
`improved`, `regressed` — ranked `converged < acceptable < failed < diverged
< timeout < error`, so a lower rank is "better" — plus `only-in-a` /
`only-in-b` for problems present in only one file). If a file has multiple
records for the same problem (e.g. from `--repeat`), the diff compares each
file's *last* recorded run per problem. Does not run anything; exits after
printing.

## JSONL schema

One JSON object per line, one line per (problem, repeat):

```json
{"problem": "...", "tier": "...", "status": "converged|acceptable|failed|diverged|timeout|error",
 "flag": "..." | null, "iterations": <int, -1 if unmatched>, "wall_s": <float>,
 "objective": <float | null>, "returncode": <int | null>, "notes": "..."}
```

- `flag` is `null` for the harness-synthesized statuses (`timeout`, and the
  `error` cases that never got as far as a child result at all); otherwise
  it is the raw `ConvergenceFlags` member name the child reported.
- `returncode` is `null` only for `timeout` (the process was killed, it
  never produced an exit code the harness waited on).
- `wall_s` is measured by the parent around the child subprocess call
  (`time.monotonic()` before/after `subprocess.run`) — this is the one
  wall-clock read in the whole system, and it lives in the parent, never in
  a problem module's `build_and_solve`.

## Literature tier (Task 4)

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

**Inequality sign convention**: PSIOPT's `add_inequal_con` (both on `Phase`
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
verified (URL and/or edition), and any correction made to the Task 4
brief's starting formula. `lit_cycling.py` (the Chamberlain, Powell,
Lemarechal, Pedersen 1982 watchdog-paper cycling example) is **not**
present: the paper (Mathematical Programming Study 16, 1982) is paywalled
on SpringerLink with no accessible preprint, and no secondary source found
reproduces its actual motivating example (as opposed to merely citing the
paper's existence/topic) -- per the binding rule for this task, the
problem was skipped rather than implemented from memory. The corpus's
target range is 15-25 problems; 17 without it is within range.

## Smoke-test end-to-end case (Task 5)

The smoke test's harness end-to-end check
(`test_harness_end_to_end_fast_problems` in
`tychopy/test/test_corpus_smoke.py`) originally ran against two Task 1
throwaway stub problems (`stub_converges.py` / `stub_fails.py`) that existed
solely to exercise the harness before any real corpus problem landed. Task 5
deleted both stubs (removed from `registry.py` and from
`tests/corpus/problems/`) now that the real degenerate/hard/literature tiers
exist, and repointed the smoke test at the two fastest real problems instead:
`deg_dup_equality` (converges in 3 iterations, ~1 s) and `hard_vanderpol`
(diverges in 1 iteration, ~1 s — the fastest genuine failure in the corpus;
the other candidate failures in the degenerate tier grind through the full
500-iteration `max_iters` cap before reporting `NOTCONVERGED`). See those
two modules' docstrings for their own problem descriptions.
