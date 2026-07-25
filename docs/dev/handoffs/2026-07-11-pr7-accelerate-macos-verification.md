# Handoff: macOS verification of the Apple Accelerate interface overhaul (I1-I6)

> **UPDATE (2026-07-25, Task 9 of the overhaul plan):** this handoff originally
> scoped four mechanical fixes (P1-P4) from a single commit. The branch has
> since grown into a six-invariant overhaul (I1-I6, 20+ commits) with a
> permanent test suite (`tests/cpp/solvers/test_accelerate_interface.cpp`).
> The original checks below are kept for historical record but are now
> superseded as noted inline; see
> `docs/dev/handoffs/2026-07-11-pr7-accelerate-macos-verification-RESULTS.md`
> for the current, authoritative verification run.
>
> **Audience:** a fresh Claude (Fable) session on Grant's MacBook (Apple Silicon).
> **Goal:** build + runtime-verify four fixes to the Apple Accelerate sparse-solver
> interface that were authored and code-reviewed on Linux, where this header cannot
> compile (`USE_ACCELERATE_SPARSE` is Apple-only). You are the first real compiler
> and the first runtime these changes ever see.
> **Deliverable:** a results file (template at the bottom), committed and pushed to
> the same branch.

## Context

Branch: `fix/review-accelerate` (split out of review-series PR 7 so the rest of
PR 7 can merge without waiting on macOS). The Accelerate commit is
`5b581aef` — `fix(solvers): Accelerate interface — factorize canonicalization,
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
git fetch origin && git switch fix/review-accelerate && git pull --ff-only
git log --oneline | grep 5b581aef   # must be an ancestor; if missing, STOP and report
git submodule update --init dep/pocketfft   # new submodule introduced on main; blocks configure without it
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

> **SUPERSEDED (Task 9):** this probe is now permanent coverage —
> `AccelerateInterface.BothTrianglePatternSurvivesAnalyzeThenFactorize` and
> `AccelerateInterface.SingleTrianglePatternSurvivesAnalyzeThenFactorize` in
> `tests/cpp/solvers/test_accelerate_interface.cpp`. Run
> `ctest -R AccelerateInterface` instead of writing a throwaway probe.

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

> **SUPERSEDED (Task 9), checks 2a and 2b:** now permanent coverage in
> `tests/cpp/solvers/test_accelerate_interface.cpp` —
> `AccelerateInterface.SolvesAnSpdSystem` (2a) and
> `AccelerateInterface.FailedFactorizationIsReported` /
> `AccelerateInterface.SolveAfterFailedFactorizePreservesClassification` (2b).
> Run `ctest -R AccelerateInterface` instead of writing a throwaway probe.

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

  > **RESOLVED — PASS (Task 9):** answered by reading the Apple SDK's inline
  > source directly rather than instrumenting a probe. Every `SparseSolve`
  > overload (`SolveImplementationTyped.h:754-761` in the macOS SDK) takes its
  > `Factored` parameter **by value** and passes `&Factored` (the address of
  > that local copy) to `_SparseSolveOpaque`. The caller's stored
  > `m_numericFactorization->status` is therefore a different object than the
  > one `_SparseSolveOpaque` can write through — a solve provably cannot
  > mutate the caller's status. No rework of P2 is needed. The
  > `updateInfoStatus(m_numericFactorization->status)` call after
  > `SparseSolve` in `_solve_impl` (`accelerate_interface.h`) is consequently
  > a documented no-op (kept for defensiveness/documentation, not effect) —
  > see the comment directly above that call in the header.

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

Note: this branch carries ONLY the Accelerate changes over `main` (the rest of
PR 7 lives on `chore/review-sweeps` / PR #87 and may or may not be merged yet) —
so a failure here that bisects to this branch's single code commit IS an
Accelerate problem. If `main` itself fails the gate on your machine, report
that separately as a pre-existing macOS issue.

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
`fix/review-accelerate` with prefix `docs(handoff):` and push.
