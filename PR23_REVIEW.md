# PR #23 Code Review: Global Thread Pool Singleton

**Reviewer:** (Senior dev, code-level review)
**Scope:** 19 commits (`71c83ae..5d0efff`), replacing per-instance `BS::thread_pool` with a process-global work-stealing pool singleton.
**Files changed:** 76 files, +1268 / -878 lines

---

## Executive Summary

The architecture is sound. Consolidating three independent thread pool instances
(Integrator, NLP, Jet) into a single Meyers singleton with centralized dispatch
helpers is the correct design. The SBO `task` wrapper, per-dispatch `std::latch`
synchronization, and `DispatchContext` exception propagation are all well-built.

The review below identifies 15 items: 4 must-fix (one latent deadlock, broken
docs, breaking API, debug-only safety checks), 4 strong recommendations, and 7
cleanup items. Each claim has been verified against the source.

---

## Verification Notes

Several initial concerns were investigated and **found to be non-issues**:

- **`DispatchContext::first_exception` is NOT racy.** The `has_exception.exchange(true)`
  guarantees a single writer. The happens-before chain is:
  worker writes `first_exception` (sequenced-before) `count_down()` (synchronizes-with)
  `wait()` (sequenced-before) `rethrow_if_exception()` reads `first_exception`.
  Correct per C++20 [thread.latch].

- **`wait()` does NOT get stuck.** `atomic::wait(val)` returns immediately when the
  current value differs from `val`, regardless of whether `notify` was called.
  Intermediate decrements (5->4) cause `wait(5)` to return because the value no
  longer matches. The 1->0 `notify_all` handles the case where the value hasn't
  changed between `load` and `wait`.

- **NLP RHS coefficients are NOT racy.** `getRHSSpace()` allocates coefficient
  buffer slots sequentially via a monotonic `freeloc` counter, iterating
  partitions in order. Each partition's functions get exclusive, non-overlapping
  regions. The potentially-conflicting scatter into final PGX/AGX/FXE/FXI
  vectors (where multiple partitions may target the same primal variable index)
  is deferred to a post-barrier single-threaded `fillRHS` step.

- **Meyers singleton destruction is safe.** No C++ static-lifetime objects
  dispatch work during destruction. No `atexit` handlers or nanobind finalizers
  exist. Python objects are GC'd during `Py_Finalize`, well before shared-library
  static destructors run.

- **`ctpl-apache2.txt` deletion is license-compliant.** The deleted file was a
  misplaced duplicate at `tycho/ctpl-apache2.txt`. The authoritative
  `notices/ctpl-apache2.txt` remains. `notices/bs-thread-pool-mit.txt` exists.
  `.gitmodules` and `dep/` are clean.

- **`enqueue_work` relaxed ordering is not a bug in current usage.** The dispatch
  helpers synchronize via `std::latch`, never calling `wait()`. `reset()` is a
  single-threaded startup-only call. See item 5 for the defensive hardening
  recommendation.

- **Accelerate thread-count not restored after `Jet::map` is pre-existing.**
  The old code had the identical `accelerate_set_num_threads(1)` at the top of
  `Jet::map` with no restore. Furthermore, `jet_release()` -> `resetTranscription()`
  ensures the next `solve()`/`optimize()` re-transcribes, which calls
  `PSIOPT::setNLP()` -> `accelerate_set_num_threads(QPThreads)`. Not a regression.

---

## Must-Fix (4 items)

### 1. `parallel_task` Missing Nested-Dispatch Guard

**File:** `src/Utils/ThreadPool.h:479`
**Severity:** Latent deadlock risk

`parallel_blocks` (line 405) and `parallel_sequence` (line 445) both have:
```cpp
assert(!detail::g_is_pool_worker && "nested dispatch from pool worker");
```

`parallel_task` does **not**. If `parallel_task` is ever called from a pool worker
context, it enqueues `pool_work` to the pool and blocks on the latch -- a pool
worker blocking while occupying a pool slot, waiting for another pool task. With
a saturated pool, this is a deadlock.

Currently safe (only called from NLP eval methods on the main thread), but the
inconsistency is a trap for future refactors. All three dispatch helpers should
have the same guard.

**Fix:**
```cpp
void parallel_task(int nparts, FTask &&pool_work, FInline &&inline_work) {
    if (nparts > 1 && use_thread_pool()) {
        assert(!detail::g_is_pool_worker && "nested dispatch from pool worker");
        // ...
```

---

### 2. Documentation Still References Dead `phase.Threads` API

**Files:** `doc/examples/zermelo.rst`, `doc/examples/zermelolink.rst`, `doc/examples/halo.rst`
**Severity:** User-facing broken docs (6 occurrences)

| File | Line(s) | Current (broken) | Should be |
|------|---------|-------------------|-----------|
| `doc/examples/zermelo.rst` | 171, 391 | `phase.Threads = 10` | `phase.NumPartitions = 10` |
| `doc/examples/zermelolink.rst` | 58, 252 | `phase.Threads = 8` | `phase.NumPartitions = 8` |
| `doc/examples/halo.rst` | 113, 375 | `odePhase.Threads = 8` | `odePhase.NumPartitions = 8` |

These will cause `AttributeError` at runtime if a user copies them. The PSIOPT
tutorial (`doc/tutorials/PSIOPT.rst`) was correctly updated, so these were simply
missed.

---

### 3. Breaking Python API with No Deprecation Path

**Severity:** User-facing silent breakage

Three breaking changes with no shims, warnings, or migration guide:

**a) `Jet.map()` signature changed:**
```python
# Old (worked):
ocps = Jet.map(ocps, 8)           # (problems, nt)
ocps = Jet.map(gen, args, 8)      # (generator, args, nt)

# New (breaks with TypeError):
ocps = Jet.map(ocps)              # nt removed
ocps = Jet.map(gen, args)         # nt removed
```

**b) `OptimizationProblemBase.Threads` renamed:**
```python
# Old:
phase.Threads = 8                 # AttributeError now

# New:
phase.NumPartitions = 8
```

**c) `setThreads()` renamed:**
```python
# Old:
phase.setThreads(8, 8)            # AttributeError now

# New:
phase.setNumPartitions(8, 8)
```

**Recommended fix:** Add Python-side deprecation shims. The simplest approach is
to add them in the nanobind binding (`OptimizationProblemBase.cpp`):
```cpp
// Deprecation shims
obj.def_prop_rw("Threads",
    [](OptimizationProblemBase &self) {
        PyErr_WarnEx(PyExc_DeprecationWarning,
            "Threads is deprecated, use NumPartitions", 1);
        return self.NumPartitions;
    },
    [](OptimizationProblemBase &self, int v) {
        PyErr_WarnEx(PyExc_DeprecationWarning,
            "Threads is deprecated, use NumPartitions", 1);
        self.NumPartitions = v;
    });
```

Or, at minimum, document the migration in the PSIOPT tutorial's "Parallelism"
section and in a changelog entry.

---

### 4. Nested-Dispatch Asserts Vanish in Release Builds

**File:** `src/Utils/ThreadPool.h:405, 445`
**Severity:** Silent deadlock in production

```cpp
assert(!detail::g_is_pool_worker && "nested dispatch from pool worker");
```

These `assert` calls compile to nothing under `NDEBUG` (release builds). A nested
dispatch from a pool worker would silently deadlock instead of failing fast.
Since `is_pool_worker()` reads a `thread_local bool` (zero cost), this should be
a runtime check:

```cpp
if (detail::g_is_pool_worker)
    throw std::logic_error("nested dispatch from pool worker — deadlock avoided");
```

The cost is one branch per dispatch entry -- completely negligible compared to
the work being dispatched.

---

## Strong Recommendations (4 items)

### 5. `enqueue_work` Should Use `memory_order_release` on Increment

**File:** `src/Utils/ThreadPool.h:268`

```cpp
m_tasks_pending.fetch_add(1, std::memory_order_relaxed);
```

Not a bug in current usage (dispatch helpers use latches, not `wait()`; `reset()`
is single-threaded). However, `wait()` is public API. If someone calls
`enqueue_work()` from thread A and `wait()` from thread B concurrently, the
relaxed increment could theoretically be invisible to the acquire load in `wait()`
on ARM64.

Upgrading to `memory_order_release` closes this for all possible callers at
negligible cost (ARM64 RMW ops already use exclusive cache-line access).

---

### 6. Document `reset()` Single-Caller Precondition

**File:** `src/Utils/ThreadPool.h:346`

`reset()` does `wait()` + destroy queues + recreate with no mutex. A concurrent
`enqueue_work()` would access destroyed queues (use-after-free). This is by
design -- `set_num_threads()` is documented as startup-only -- but the
precondition is implicit on `reset()` itself.

**Fix:** Add a doc comment:
```cpp
/// Shut down all workers and restart with n threads.
/// Precondition: no concurrent enqueue_work() or submit_task() calls.
/// This is a configuration API, not a hot-path operation.
void reset(unsigned n) {
```

---

### 7. Dead `Vals` Code in `evalKKTNO` and `evalSOE`

**File:** `src/Solvers/NonLinearProgram.cpp:463,491`

Both methods allocate `std::vector<double> Vals(NumPartitions, 0.0)` and sum them
afterward, but their lambdas **never write to `Vals`** (they have no objective
functions). The allocation and sum loop are dead code that allocates heap memory
on every call for no purpose.

This is pre-existing (the old code had the same pattern with `Threads`), but this
refactor is the right time to clean it up.

Additionally, `evalKKTNO` runs `fillSolverCoeffs` and `fillRHS` sequentially
(lines 481-483), while the structurally identical `evalKKT`/`evalSOE`/`evalAUG`
use `parallel_task` to overlap them. This is an inconsistency worth addressing.

---

### 8. `MIN_NNZ_PER_PARTITION` Threshold Undocumented

**File:** `src/Solvers/NonLinearProgram.cpp:30`

```cpp
static constexpr int MIN_NNZ_PER_PARTITION = 1000;
```

The comment explains *what* it does but not *why 1000*. This threshold determines
whether small problems get capped to fewer partitions -- a significant behavioral
decision. Was it benchmarked? On what hardware? What problem?

**Fix:** Add a comment like:
```cpp
// Empirically determined: below ~1000 NNZ per partition, dispatch overhead
// dominates eval cost. Benchmarked on [problem] ([hardware], [date]).
// See bench/bench_track.sh compare output from commit [hash].
```

---

## Cleanup (7 items)

### 9. Plan Document Is Stale

**File:** `docs/plans/2026-03-17-global-thread-pool-design.md`

The "Implementation" section describes wrapping `BS::thread_pool` with a Meyers
singleton. The actual implementation is a from-scratch `Tycho::ThreadPool` with
work-stealing queues, SBO task wrapper, and per-dispatch latch synchronization --
a fundamentally different architecture. The "Consumer Changes" sections describe
`detach_sequence`/`detach_blocks` APIs that don't exist.

Either update the document to reflect the actual design or add a header:
```
**Status:** Superseded — see actual implementation in src/Utils/ThreadPool.h
```

---

### 10. Tests Mutate Global State Without RAII Cleanup

**File:** `tests/cpp/utils/test_thread_pool.cpp`

Every test that calls `set_num_threads(4)` manually restores with
`set_num_threads(hardware_concurrency())` at the end. If a test fails or throws
before the restore line, subsequent tests run with incorrect thread counts.

**Fix:** Use a test fixture or RAII wrapper:
```cpp
struct ScopedThreadCount {
    int prev;
    ScopedThreadCount(int n) : prev(Tycho::get_num_threads()) { Tycho::set_num_threads(n); }
    ~ScopedThreadCount() { Tycho::set_num_threads(prev); }
};
```

---

### 11. `parallel_task` Single-Threaded Fallback Execution Order

**File:** `src/Utils/ThreadPool.h:500-503`

```cpp
} else {
    pool_work();
    inline_work();
}
```

In the parallel path, `pool_work` and `inline_work` are concurrent (unordered).
In the fallback, `pool_work` runs first, then `inline_work`. This is a semantic
difference. In current NLP usage (`fillRHS` vs `fillSolverCoeffs`), they write to
different memory so it's safe. But the ordering difference should be documented:

```cpp
// Fallback: execute sequentially (pool_work first, then inline_work).
// In the parallel path above, execution order is unspecified.
```

---

### 12. `task` SBO Buffer Headroom Undocumented

**File:** `src/Utils/ThreadPool.h:68`

```cpp
static constexpr std::size_t BUF_SIZE = 64;
```

The `static_assert` fires at compile time if a closure exceeds 64 bytes --
excellent. But when it fires, the developer has no guidance on what the current
closures look like. Add a comment documenting actual sizes:

```
// Current largest closures (approximate, 64-bit):
//   parallel_sequence wrapper: ~20 bytes (&func, &ctx, int i)
//   parallel_blocks wrapper:   ~28 bytes (&func, &ctx, int start, int end)
//   parallel_task wrapper:     ~24 bytes (captured lambda, &ctx)
// Headroom: ~36 bytes
```

---

### 13. Nested `parallel_blocks` Inside `parallel_task` Needs Comment

**File:** `src/Solvers/NonLinearProgram.cpp:452-454`

```cpp
Tycho::parallel_task(
    this->NumPartitions, [&] { this->fillRHS(PGX, AGX, FXE, FXI); },
    [&] { this->fillSolverCoeffs(KKTmat); });
```

`fillSolverCoeffs` (line 246) itself calls `parallel_blocks(numSolverKKTElems,
FillOp, NumPartitions)`. So the inline arm of `parallel_task` is a nested
parallel dispatch. This works because:

1. The calling thread is the main thread (not a pool worker), so
   `parallel_blocks`'s assert passes.
2. `parallel_blocks` dispatches N-1 tasks to the pool. Combined with the
   `fillRHS` task from `parallel_task`, the total pool demand is N tasks.
   Workers drain them sequentially -- no deadlock because no pool worker
   blocks on another pool task.

This is safe but non-obvious. A comment at the call site would prevent future
developers from panicking:

```cpp
// NOTE: fillSolverCoeffs internally calls parallel_blocks, creating a
// nested dispatch. This is safe because the inline arm runs on the main
// thread (not a pool worker), so the pool absorbs all tasks without
// circular blocking.
```

---

### 14. Test `ocp.NumPartitions = 8` Semantic Change

**File:** `tycho/test/test_FullProblems/test_Delta3Launch.py:261`

```python
ocp.NumPartitions = 8
```

In the old code, `Threads = 8` created a pool with 8 threads. Now
`NumPartitions = 8` splits work into 8 partitions, but actual parallelism
depends on the global `set_num_threads()` setting. If the test environment
has `get_num_threads() == 1`, this is 8 inline-executed partitions (slightly
more overhead than 1). The test likely works either way, but verify this is
the intended behavior and not a leftover from mechanical renaming.

---

### 15. `Jet::map` `track` Lambda Sequential-Access Invariant

**File:** `src/Solvers/Jet.h:155-172`

The `track` lambda mutates stack-local counters (`NumConv`, `NumAcc`,
`NumNoConv`, `NumDiv`). It is safe only because `results[i].get()` blocks
sequentially -- `track` is always called on the main thread. If someone
refactored the loop to process results out-of-order (e.g., for better progress
reporting), they would introduce a data race.

**Fix:** Add a comment:
```cpp
// NOTE: track() mutates local counters and MUST be called sequentially
// on the main thread. Do not parallelize this loop.
for (int i = 0; i < NumJobs; i++)
    track(results[i].get(), i);
```

---

## Summary Table

| Priority | # | Issue | Location |
|----------|---|-------|----------|
| **Must-fix** | 1 | `parallel_task` missing `g_is_pool_worker` assert | `ThreadPool.h:479` |
| **Must-fix** | 2 | Stale `phase.Threads` in doc examples (6 occurrences) | `doc/examples/{zermelo,zermelolink,halo}.rst` |
| **Must-fix** | 3 | Breaking Python API with no deprecation shims | `JetBind.cpp`, `OptimizationProblemBase.cpp` |
| **Must-fix** | 4 | Nested-dispatch asserts should be runtime checks | `ThreadPool.h:405,445` |
| **Recommended** | 5 | `enqueue_work` fetch_add -> `memory_order_release` | `ThreadPool.h:268` |
| **Recommended** | 6 | Document `reset()` single-caller precondition | `ThreadPool.h:346` |
| **Recommended** | 7 | Dead `Vals` code in `evalKKTNO`/`evalSOE`; missing `parallel_task` in `evalKKTNO` | `NonLinearProgram.cpp:463,491` |
| **Recommended** | 8 | Document `MIN_NNZ_PER_PARTITION` benchmark origin | `NonLinearProgram.cpp:30` |
| **Cleanup** | 9 | Plan document stale | `docs/plans/2026-03-17-...` |
| **Cleanup** | 10 | Tests need RAII thread-count restore | `test_thread_pool.cpp` |
| **Cleanup** | 11 | `parallel_task` fallback order documentation | `ThreadPool.h:500` |
| **Cleanup** | 12 | `task` SBO buffer headroom documentation | `ThreadPool.h:68` |
| **Cleanup** | 13 | Nested `parallel_blocks` inside `parallel_task` comment | `NonLinearProgram.cpp:452` |
| **Cleanup** | 14 | `ocp.NumPartitions = 8` semantic change in test | `test_Delta3Launch.py:261` |
| **Cleanup** | 15 | `Jet::map` `track` lambda sequential-access invariant | `Jet.h:155` |
