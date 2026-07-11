# Handoff: macOS verification of PR 7 Accelerate fixes

> **Audience:** a fresh Claude (Fable) session on Grant's MacBook (Apple Silicon).
> **Goal:** build + runtime-verify four fixes to the Apple Accelerate sparse-solver
> interface that were authored and code-reviewed on Linux, where this header cannot
> compile (`USE_ACCELERATE_SPARSE` is Apple-only). You are the first real compiler
> and the first runtime these changes ever see.
> **Deliverable:** a results file (template at the bottom), committed and pushed to
> the same branch.

## Context

Branch: `chore/review-sweeps` (review-series PR 7). The Accelerate commit is
`fa02be79` — `fix(solvers): Accelerate interface — factorize canonicalization,
solve status, size_t widths, zero-init dims (CODEBASE 1.3; macOS-only, needs
macOS verification)`. It touches ONLY
`include/tycho/detail/solvers/linear/accelerate_interface.h`, four mechanical hunks:

- **P1** — `factorize(a)` previously did `matrix_ = a;`, bypassing the triangle
  canonicalization that `analyze_pattern` applies via the `get_matrix` enable_if
  pair. It now calls `get_matrix(a)`, so the numeric matrix matches the
  single-triangle symbolic structure in the analyze→factorize reuse pattern.
  (Deliberately NOT `set_matrix`, which would destroy the reused symbolic
  factorization.)
- **P2** — `_solve_impl` previously reported a hardcoded `SparseStatusOK`; it now
  reads the real `m_numericFactorization->status` after `SparseSolve`. All
  `SparseSolve` overloads used here are void-returning; the in-code comment
  documents that solve-time-specific failures remain unobservable through this
  API. Key reviewer-identified unknown: whether `SparseSolve` can CLOBBER
  `->status` on a successful solve (would make the fix unsound — see check 2c).
- **P3** — `doFactorization()`'s `factorSize`/`workspaceSize` locals widened
  `int` → `size_t`, matching the Accelerate fields and
  `resizeForAccelerateAlignment`'s parameter.
- **P4** — `Index n_rows_ = 0, n_cols_ = 0;` in-class initializers (previously
  uninitialized; `rows()`/`cols()`/the `factorize` assert could read indeterminate
  values on a default-constructed instance).

The Linux-side review (full-file inspection, Opus) concluded both risky hunks are
strict improvements with no regression path — but flagged the three ⚠️ items that
checks 0, 2c, and the gate below exist to close.

## Ground rules (CLAUDE.md, restated)

- `conda activate tycho` before configuring; run all Python from the repo root.
- ONE build at a time. `cmake --preset macos-llvm-release`; `ninja -j6 all` from
  `build/` (`-j2`/`-j3` if you also enable benchmarks). NEVER start a second build
  while one runs.
- Do not run global `ninja clang-format` (version churn); don't reformat files.
- Commit messages end with the `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>` trailer.

## Steps

### 0. Sync + build (covers ⚠️-2: first-ever compile of these hunks)

```bash
cd <tycho clone>   # Grant's local clone
git fetch origin && git switch chore/review-sweeps && git pull --ff-only
git log --oneline | grep fa02be79   # must be an ancestor; if missing, STOP and report
conda activate tycho
cmake --preset macos-llvm-release -DBUILD_CPP_TESTS=ON
cd build && ninja -j6 all
```

PASS = clean build with zero errors/warnings sourced from
`accelerate_interface.h`. Specifically watch that `get_matrix(a)` resolves
unambiguously inside `factorize`, `m_numericFactorization->status` is accessible
in const `_solve_impl`, and the `size_t` locals feed
`resizeForAccelerateAlignment` without narrowing warnings.

### 1. P1 probe — both-triangle analyze→factorize reuse

Write a standalone probe (suggested: `/tmp/accel_probe/p1.cpp`, compiled against
the repo's include tree + `-framework Accelerate`; copy compile flags from a
`ninja -v` line for any solvers TU, or use a small `add_executable` in a scratch
CMake dir — your choice, just don't commit build-system changes):

- Instantiate the symmetric Accelerate solver type PSIOPT uses (look at how
  `psiopt.h`/`psiopt.cpp` instantiate `AccelerateImpl` under
  `USE_ACCELERATE_SPARSE` and mirror it).
- Build a symmetric sparse matrix `A1` with entries populated in BOTH triangles;
  `analyze_pattern(A1)`; then `factorize(A2)` where `A2` has the same pattern,
  different values, also both-triangle-populated; solve `A2 x = b`.
- PASS = `||A2*x - b||_inf` at solver tolerance (compare against
  `Eigen::SimplicialLDLT` or a dense solve of the true symmetric `A2`).
  Pre-fix this was a symbolic/numeric structure mismatch.
- Regression guard: repeat with single-triangle-only inputs — results must be
  unchanged-good.

### 2. P2 probe — solve status wiring (⚠️-1, the key unknown)

- **2a Nominal:** well-conditioned solve → `info() == Eigen::Success`.
- **2b Failure surfacing:** `analyze_pattern` + `factorize` on a good matrix,
  then `factorize` again with a singular/near-singular same-pattern matrix (or
  whatever most directly drives `doRefactorization` to a failed status), then
  `solve` → `info()` must now be `NumericalIssue`/`InvalidInput`, not `Success`
  (pre-fix it lied `Success`).
- **2c Clobber check:** read `m_numericFactorization->status` immediately before
  and after a KNOWN-GOOD `SparseSolve` (the member is protected — subclass the
  solver in the probe to expose it, or temporarily add a local uncommitted debug
  accessor). PASS = status stays `SparseStatusOK` across the solve. **If it
  changes on a successful solve, the P2 fix is unsound — record FAIL and stop;
  do not patch around it.**

### 3. Gate (also covers P3 behaviorally and P4)

```bash
cd build && ctest --output-on-failure
cd .. && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
cmake --preset macos-llvm-release -DBUILD_CPP_EXAMPLES=ON && cd build && ninja -j6 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp   # obj ≈ 1.8013
```

All ctest must pass; all examples exit 0; brach converges. (P3 is a pure widening
below INT_MAX — behavioral parity here is the check. P4 optional extra:
default-construct an `AccelerateImpl` in the probe, `rows()`/`cols()` before
`compute()` → both `0`.)

Note: this branch also carries the rest of PR 7 (abs sweep, dead-code deletions,
thread-pool startup fix, platform guards, new `-Wabsolute-value` flag). A test or
example failure is not automatically an Accelerate problem — bisect the failure
to the responsible change and report it either way.

### 4. Report + push

Fill in `docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification-RESULTS.md`:

```markdown
# RESULTS: PR 7 Accelerate macOS verification
- Date / machine / macOS + Xcode/clang versions:
- HEAD verified (git rev-parse HEAD):
- 0 Build: PASS/FAIL (+ any accelerate_interface.h diagnostics verbatim)
- 1 P1 both-triangle reuse: PASS/FAIL (residual norms, both variants)
- 2a P2 nominal: PASS/FAIL
- 2b P2 failure surfacing: PASS/FAIL (what info() returned)
- 2c P2 clobber check: PASS/FAIL (status before/after values)
- 3 ctest: N passed / N failed (name failures)
- 3 examples: N/N
- 3 brachistochrone: objective value
- Probe sources: (paste or path)
- Verdict: fixes hold on macOS? Anything Grant must look at before merge?
```

Commit the results file (and nothing else — probes stay out of the repo) to
`chore/review-sweeps` with prefix `docs(handoff):` and push.
