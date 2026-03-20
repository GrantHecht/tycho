# Plan: Address PR #23 Code Review Findings

## Context

PR #23 introduced a global thread pool singleton replacing per-instance `BS::thread_pool`
instances across Integrator, NLP, and Jet. A comprehensive code review (`PR23_REVIEW.md`)
identified 15 issues: 4 must-fix, 4 recommended, 7 cleanup. This plan addresses all 15
items in a single pass, organized into 5 commits for clean git history.

**Item #3 (breaking Python API) — No action.** The `Threads` -> `NumPartitions` and
`Jet.map(nt)` removal are intentional breaking changes. No deprecation shims or
backward-compatibility wrappers will be added. The stale doc references (Item #2) are
the only user-facing fix needed.

---

## Commit 1: Thread pool safety hardening

**Items addressed:** #1, #4, #5, #6

**File:** `src/Utils/ThreadPool.h`

### 1a. Add nested-dispatch guard to `parallel_task` (Item #1)

At line 480, after `if (nparts > 1 && use_thread_pool()) {`, add the same guard
that `parallel_blocks` (line 405) and `parallel_sequence` (line 445) have, but as a
runtime check (which also addresses Item #4):

```cpp
if (nparts > 1 && use_thread_pool()) {
    if (detail::g_is_pool_worker)
        throw std::logic_error("Tycho::parallel_task: nested dispatch from pool worker");
```

### 1b. Upgrade all three dispatch asserts to runtime checks (Item #4)

Replace the `assert` on line 405 (`parallel_blocks`) and line 445 (`parallel_sequence`)
with the same `if + throw` pattern:

- Line 405: replace `assert(!detail::g_is_pool_worker && "nested dispatch from pool worker");`
  with `if (detail::g_is_pool_worker) throw std::logic_error("Tycho::parallel_blocks: nested dispatch from pool worker");`
- Line 445: replace `assert(!detail::g_is_pool_worker && "nested dispatch from pool worker");`
  with `if (detail::g_is_pool_worker) throw std::logic_error("Tycho::parallel_sequence: nested dispatch from pool worker");`

### 1c. Upgrade `enqueue_work` memory ordering (Item #5)

Line 316: change `std::memory_order_relaxed` to `std::memory_order_release`:
```cpp
m_tasks_pending.fetch_add(1, std::memory_order_release);
```

### 1d. Document `reset()` precondition (Item #6)

Replace the existing doc comment at line 345 with:
```cpp
/// Shut down all workers and restart with n threads.
/// Precondition: no concurrent enqueue_work() or submit_task() calls.
/// This is a configuration API, not a hot-path operation.
```

---

## Commit 2: Fix stale documentation (Items #2, #9)

### 2a. Fix stale `phase.Threads` in doc examples (Item #2)

Six replacements across three files. Replace `.Threads` with `.NumPartitions`
and drop the now-misleading `# Equal to number of physical cores` comments
since NumPartitions != thread count:

| File | Line | Old | New |
|------|------|-----|-----|
| `doc/examples/zermelo.rst` | 171 | `phase.Threads = 10  # Equal to number of physical cores` | `phase.NumPartitions = 10` |
| `doc/examples/zermelo.rst` | 391 | `phase.Threads = 10  # Equal to number of physical cores` | `phase.NumPartitions = 10` |
| `doc/examples/zermelolink.rst` | 58 | `phase.Threads = 8  # Equal to number of physical cores` | `phase.NumPartitions = 8` |
| `doc/examples/zermelolink.rst` | 252 | `phase.Threads = 8  # Equal to number of physical cores` | `phase.NumPartitions = 8` |
| `doc/examples/halo.rst` | 113 | `odePhase.Threads = 8  # Equal to number of physical cores` | `odePhase.NumPartitions = 8` |
| `doc/examples/halo.rst` | 375 | `odePhase.Threads = 8  # Equal to number of physical cores` | `odePhase.NumPartitions = 8` |

### 2b. Mark plan document as superseded (Item #9)

In `docs/plans/2026-03-17-global-thread-pool-design.md`, change line 4 from:
```
**Status:** Approved
```
to:
```
**Status:** Superseded — see actual implementation in `src/Utils/ThreadPool.h`
```

---

## Commit 3: NLP dead code cleanup (Items #7, #8, #13)

**File:** `src/Solvers/NonLinearProgram.cpp`

### 3a. Clean up `evalKKTNO` dead `Vals` and add `parallel_task` (Item #7)

Lines 457-484. Remove the dead `Vals` vector, dead sum loop, and upgrade
sequential `fillSolverCoeffs`/`fillRHS` to `parallel_task`:

```cpp
void Tycho::NonLinearProgram::evalKKTNO(double ObjScale, ConstEigenRef<VectorXd> X,
                                        ConstEigenRef<VectorXd> LE, ConstEigenRef<VectorXd> LI,
                                        double &val, EigenRef<VectorXd> PGX, EigenRef<VectorXd> AGX,
                                        EigenRef<VectorXd> FXE, EigenRef<VectorXd> FXI,
                                        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {

    this->setRHSCoeffsZero();

    auto KKTevalOP = [&](int thrnum) {
        for (auto &Con : this->ThrEq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LE, this->EConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
        for (auto &Con : this->ThrIq[thrnum])
            Con.constraints_jacobian_adjointgradient_adjointhessian(
                X, LI, this->IConCoeffs(), this->AGXCoeffs(), KKTmat, this->KKTLocations,
                this->KKTClashes, this->KKTLocks);
    };

    Tycho::parallel_sequence(this->NumPartitions, KKTevalOP);

    // NOTE: fillSolverCoeffs internally calls parallel_blocks, creating a nested
    // dispatch from the inline arm. Safe because the calling thread is the main
    // thread (not a pool worker), so the pool absorbs all tasks without deadlock.
    Tycho::parallel_task(
        this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
        [&] { this->fillSolverCoeffs(KKTmat); });
}
```

Changes vs current code:
- Remove `std::vector<double> Vals(this->NumPartitions, 0.0);` (line 463)
- Remove `for (int i = 0; i < this->NumPartitions; i++) val += Vals[i];` (lines 478-479)
- Replace sequential `fillSolverCoeffs` + `fillRHS` (lines 481-483) with `parallel_task`
- Add the nested-dispatch safety comment (Item #13)

### 3b. Clean up `evalSOE` dead `Vals` (Item #7)

Lines 485-508. Remove the dead `Vals` allocation (line 491). `evalSOE` has no
objectives so its lambda never writes to `Vals`, and there is no sum loop after
`parallel_sequence` — just a dead `std::vector<double>` allocation on every call.

- Remove `std::vector<double> Vals(this->NumPartitions, 0.0);` (line 491)

### 3c. Add nested-dispatch comment to all `parallel_task` call sites (Item #13)

Four methods use `parallel_task` for concurrent `fillRHS` / `fillSolverCoeffs`.
Since `fillSolverCoeffs` internally calls `parallel_blocks` (creating a nested
dispatch from the inline arm), add this comment before each `parallel_task` call:

```cpp
    // NOTE: fillSolverCoeffs internally calls parallel_blocks, creating a nested
    // dispatch from the inline arm. Safe because the calling thread is the main
    // thread (not a pool worker), so the pool absorbs all tasks without deadlock.
```

Locations:
- `evalKKT` line 452
- `evalKKTNO` (new, added in step 3a above)
- `evalSOE` line 505
- `evalAUG` line 537

### 3d. Document `MIN_NNZ_PER_PARTITION` (Item #8)

Replace the comment block at lines 26-29 with:
```cpp
    // Cap partitions so each has enough work to offset dispatch overhead.
    // numUserKKTElems counts Jacobian + Hessian NNZ across all functions —
    // proportional to per-partition compute in evalKKT/evalAUG. Below ~1000
    // NNZ per partition, dispatch overhead dominates actual work.
    // Threshold empirically chosen based on solver benchmarks (bench_all);
    // re-evaluate with bench/bench_track.sh if dispatch overhead changes.
```

---

## Commit 4: Test quality improvements (Item #10)

**File:** `tests/cpp/utils/test_thread_pool.cpp`

Add an RAII helper near the top of the file (after the includes, before the first test):

```cpp
// RAII guard to restore global thread count on scope exit.
// Ensures test failures don't leave stale thread counts for subsequent tests.
struct ScopedThreadCount {
    int prev;
    explicit ScopedThreadCount(int n) : prev(Tycho::get_num_threads()) {
        Tycho::set_num_threads(n);
    }
    ~ScopedThreadCount() { Tycho::set_num_threads(prev); }
    ScopedThreadCount(const ScopedThreadCount &) = delete;
    ScopedThreadCount &operator=(const ScopedThreadCount &) = delete;
};
```

Then refactor all 10 tests that use `set_num_threads` + manual restore. Each test
changes from:

```cpp
TEST(GlobalThreadPool, SetAndGet) {
    Tycho::set_num_threads(4);
    // ... test body ...
    // Restore default
    Tycho::set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));
}
```

to:

```cpp
TEST(GlobalThreadPool, SetAndGet) {
    ScopedThreadCount guard(4);
    // ... test body (remove manual restore) ...
}
```

For tests that change thread count multiple times within the body (e.g., `SetAndGet`
which sets 4 then 1 then restores), keep `set_num_threads` calls in the body but
remove the final "Restore default" — the `ScopedThreadCount` destructor handles it.

**Tests to refactor (10 total):**
- `GlobalThreadPool::SetAndGet` (line 84)
- `GlobalThreadPool::SetZeroMeansSingleThreaded` (line 97)
- `GlobalThreadPool::PoolDispatch` (line 106)
- `GlobalThreadPool::SingleThreadedBypass` (line 121)
- `DispatchHelpers::ExceptionPropagation_ParallelBlocks` (line 141)
- `DispatchHelpers::ExceptionPropagation_ParallelSequence` (line 155)
- `DispatchHelpers::ExceptionPropagation_ParallelTask` (line 167)
- `DispatchHelpers::InlineException_ParallelBlocks` (line 176)
- `DispatchHelpers::WorkerDetection` (line 192)
- `DispatchHelpers::StressTest_ParallelSequence` (line 219)

---

## Commit 5: Documentation comments (Items #11, #12, #14, #15)

### 5a. Document `parallel_task` fallback order (Item #11)

**File:** `src/Utils/ThreadPool.h`, line 500. Add comment before the else block:

```cpp
    // Fallback: execute sequentially (pool_work first, then inline_work).
    // In the parallel path above, execution order is unspecified.
    } else {
        pool_work();
        inline_work();
    }
```

### 5b. Document `task` SBO buffer headroom (Item #12)

**File:** `src/Utils/ThreadPool.h`, lines 65-70. Expand the section comment:

```
// task — SBO callable wrapper (move-only, 64-byte inline buffer)
//
// Replaces std::function<void()> as the task unit. Zero heap allocation for
// all Tycho dispatch closures (verified <= 64 bytes). The static_assert
// fires at compile time if a closure exceeds the buffer.
//
// Current largest closures (approximate, 64-bit):
//   parallel_sequence wrapper: ~20 bytes (&func, &ctx, int i)
//   parallel_blocks wrapper:   ~28 bytes (&func, &ctx, int start, int end)
//   parallel_task wrapper:     ~24 bytes (captured lambda, &ctx)
// Headroom: ~36 bytes
```

### 5c. Verify `ocp.NumPartitions = 8` semantic intent (Item #14)

**File:** `tycho/test/test_FullProblems/test_Delta3Launch.py`, line 261.

This is semantically correct — it's a test that explicitly sets 8 partitions. The
test passes regardless of global thread count because `parallel_sequence` will
either dispatch to the pool or run inline. No code change needed — just verify
during the test run that it passes.

### 5d. Add `track` lambda sequential-access comment (Item #15)

**File:** `src/Solvers/Jet.h`, before line 181. Add:

```cpp
            // NOTE: track() mutates local counters and MUST be called sequentially
            // on the main thread. Do not parallelize this loop.
            for (int i = 0; i < NumJobs; i++)
                track(results[i].get(), i);
```

---

## Files Modified (summary)

| File | Commits | Items |
|------|---------|-------|
| `src/Utils/ThreadPool.h` | 1, 5 | #1, #4, #5, #6, #11, #12 |
| `src/Solvers/NonLinearProgram.cpp` | 3 | #7, #8, #13 |
| `src/Solvers/Jet.h` | 5 | #15 |
| `doc/examples/zermelo.rst` | 2 | #2 |
| `doc/examples/zermelolink.rst` | 2 | #2 |
| `doc/examples/halo.rst` | 2 | #2 |
| `docs/plans/2026-03-17-global-thread-pool-design.md` | 2 | #9 |
| `tests/cpp/utils/test_thread_pool.cpp` | 4 | #10 |

---

## Verification

### Build
```bash
cd build && ninja -j2 all
```
Must compile cleanly — particularly the `static_assert` in `task` should still
pass, and the new `throw` in dispatch helpers should not break any existing code
paths.

### C++ unit tests
```bash
cd build && ninja -j2 tycho_tests && ctest --output-on-failure
```
All tests must pass. The `ScopedThreadCount` refactor must not change test behavior.
Pay particular attention to the `DispatchHelpers::ExceptionPropagation_*` tests —
they now exercise `throw std::logic_error` in the same code path as the runtime
nested-dispatch check (though they won't trigger it since they run on the main thread).

### Python examples (integration tests)
```bash
python scripts/run_examples.py
```
All 38 examples must pass.

### C++ brachistochrone example
```bash
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```
Must converge with "Optimal Solution Found".

---

## Commit order and messages

1. `fix: upgrade nested-dispatch guards to runtime checks and harden thread pool API`
2. `docs: fix stale phase.Threads references and mark design doc as superseded`
3. `fix: remove dead Vals code in evalKKTNO/evalSOE, add parallel_task to evalKKTNO`
4. `test: use RAII ScopedThreadCount for thread pool test cleanup`
5. `docs: add safety and headroom comments to ThreadPool, NLP, and Jet`
