# Global Thread Pool Singleton

**Date:** 2026-03-17
**Status:** Superseded — see implementation in `src/Utils/ThreadPool.h` (branch `feat/alternative-thread-pool`)
**Branch:** `feat/alternative-thread-pool`

## Problem

Tycho creates multiple independent `BS::thread_pool` instances across its subsystems:

- `Integrator<DODE>` owns a `shared_ptr<BS::thread_pool<>>` per instance
- `NonLinearProgram` owns a `BS::thread_pool<>` member, constructed each `transcribe()` call
- `Jet::map()` creates a local `BS::thread_pool<>` per invocation

This causes two performance problems:

1. **Pool construction overhead.** Spawning OS threads is expensive (~20 us per thread). The integrator benchmarks create a new `Integrator` each iteration (including a pool), adding 10-17x overhead. Phase benchmarks create a new NLP (18-thread pool) each `transcribe()`, adding ~2x overhead.

2. **Oversubscription risk.** Multiple independent pools can collectively spawn more threads than hardware cores, causing contention.

## Prior Art

Every major C++ parallel computing library uses a process-global pool:

- **OpenMP**: process-global thread pool, created at first parallel region, lives for program lifetime. Resizable via `omp_set_num_threads()`.
- **Intel oneTBB**: global "market" thread pool shared across all `task_arena` instances. Controllable via `tbb::global_control`.
- **Eigen**: uses OpenMP's global pool or its own internal pool.
- **Taskflow**: recommends a single `tf::Executor` per program.

## Design

### API

New files: `src/Utils/ThreadPool.h` and `src/Utils/ThreadPool.cpp`.

```cpp
namespace Tycho {

/// Returns the process-global thread pool.
/// Created lazily on first call, sized to hardware_concurrency().
BS::thread_pool<>& thread_pool();

/// Set the number of threads used for parallel work.
/// n=1: single-threaded mode (all work runs inline, pool is never touched).
/// n>1: resize the pool to n threads.
/// Intended as a configuration call (e.g., at program startup), not per-solve.
void set_num_threads(int n);

/// Returns the current thread count setting. 1 = single-threaded mode.
int get_num_threads();

/// Fast check: returns true if the global pool should be used (num_threads > 1).
bool use_thread_pool();

} // namespace Tycho
```

### Implementation

- `thread_pool()` uses a Meyers singleton (function-local static). Thread-safe by the C++ standard, zero overhead after first call.
- An internal `std::atomic<int>` tracks the thread count setting.
- `set_num_threads(1)` sets the atomic to 1 without touching the pool. If the pool was never created, it stays uncreated.
- `set_num_threads(N)` where N > 1 sets the atomic to N and calls `thread_pool().reset(N)`.
- `use_thread_pool()` returns `get_num_threads() > 1`. Dispatch sites check this before submitting work.

### Consumer Changes

#### Integrator (`src/Integrators/Integrator.h`)

- Remove `std::shared_ptr<BS::thread_pool<>> pool` member.
- Remove `ensurePool()` helper.
- Parallel dispatch sites check `Tycho::use_thread_pool()`:
  - If true: `Tycho::thread_pool().submit_blocks(...)`.
  - If false: call the job lambda directly inline.
- Single-trajectory `integrate()` never touches the pool (no construction cost).

#### NonLinearProgram (`src/Solvers/NonLinearProgram.h/.cpp`)

- Remove `BS::thread_pool<> TP` member.
- Remove pool construction from constructors.
- Dispatch sites check `use_thread_pool()`:
  - If true: `Tycho::thread_pool().detach_sequence(...)` / `detach_blocks(...)` + `wait()`.
  - If false: loop over all thread partitions inline on the calling thread.
- `int Threads` member stays: it controls work *partitioning*, not pool size.

#### Jet (`src/Solvers/Jet.h`)

- Remove local `BS::thread_pool<>(nt)` construction.
- Use `Tycho::thread_pool().submit_sequence()` for outer parallelism.
- Inner `jet_initialize()` already forces `setNumPartitions(1, 1)`, so NLP never dispatches to the pool during Jet runs.

#### Python Bindings

- Expose `tycho.set_num_threads(n)` and `tycho.get_num_threads()` at module level.

### Files Touched

| File | Change |
|---|---|
| `src/Utils/ThreadPool.h` | New: singleton API |
| `src/Utils/ThreadPool.cpp` | New: implementation |
| `src/Utils/Tycho_Utils.h` | Add `#include "ThreadPool.h"` |
| `src/Integrators/Integrator.h` | Remove pool member, use global |
| `src/Solvers/NonLinearProgram.h` | Remove `TP` member |
| `src/Solvers/NonLinearProgram.cpp` | Use global pool |
| `src/Solvers/Jet.h` | Use global pool |
| `CMakeLists.txt` | Add `ThreadPool.cpp` to utils sources |

### What Stays the Same

- `OptimizationProblemBase::NumPartitions`: controls NLP work partitioning (renamed from Threads).
- `PSIOPT::QPThreads`: controls Pardiso/Accelerate solver threads (unrelated).
- `setNumPartitions(num_partitions, qp_threads)`: sets partition count per-problem (renamed from setThreads).
- Benchmark/test files that create their own local `BS::thread_pool<>` for measuring raw dispatch overhead (unchanged).

### Edge Cases

- **`set_num_threads(0)`**: treated as `set_num_threads(1)` (single-threaded mode).
- **Destruction order at exit**: Meyers singleton destroyed during `atexit`. `BS::thread_pool` destructor calls `wait()` then joins threads. No Tycho static-lifetime objects dispatch work during their destructors.
- **Python lifecycle**: singleton created on first use within the Python process, lives until process exit. Multiple `import tycho` calls don't create multiple pools.
- **NLP `Threads + 2` sizing**: the global pool at `hardware_concurrency()` naturally exceeds this. No special handling needed.
