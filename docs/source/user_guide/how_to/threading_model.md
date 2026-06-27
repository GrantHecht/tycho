# How to control parallelism and threading

Tycho maintains a process-global thread pool that is shared by all parallel
work: batch integrations, Jacobian evaluations, and the optimizer's NLP
partition passes. The default pool size is the machine's hardware concurrency,
which is usually the right choice — but you may need to pin it lower when
running Tycho inside a larger multi-process job, or raise it selectively for a
batch integration sweep. This recipe shows how to read and set the thread count,
how to exploit parallelism across many independent integrations, and how to
configure the optimizer's NLP partition count without conflating it with the
linear-solver thread count.

This recipe assumes you have already constructed an integrator — see
{doc}`Choosing an integrator </user_guide/how_to/choosing_an_integrator>`.

The C++ tabs show the equivalent calls — illustrative fragments that assume
`integ`, `ode`, and `phase` already in scope plus the headers and
`using namespace` lines shown in the first tab, not standalone programs.

## How it works

A few facts about the threading model explain the behavior of the calls below:

- **One process-global pool.** Tycho owns a single work-stealing thread pool,
  created lazily the first time parallel work is requested and sized to the
  machine's hardware concurrency. Every parallel operation — batch integration,
  the optimizer's Jacobian partitions, internal dispatch — draws from this same
  pool. Resizing it with `set_num_threads(n)` changes the budget for *all*
  parallel work at once, which is why it is meant to be called once at startup.
- **Parallelism pays off in batches, not single runs.** Distributing work to the
  pool has overhead. A single short integration or a single small solve will not
  recover that cost — the win comes when you have many independent units of work
  (a grid of initial conditions, a Monte-Carlo sweep, a manifold fan-out), where
  each worker stays busy on its own trajectory with no shared mutable state.
- **Threads and SIMD are different axes.** The thread pool spreads *separate*
  trajectories across cores. Orthogonally, Tycho's SuperScalar (SIMD)
  vectorization packs several trajectories' arithmetic into one wide instruction
  inside a single worker — the batched state-transition-matrix paths use this.
  The two compose, and SIMD is automatic where the dynamics support it; there is
  no knob to set.

## Query and set the thread count

The utilities submodule exposes three functions for thread management:

::::{tab-set}
:::{tab-item} Python
```python
from tychopy import utils

cores = utils.get_core_count()     # physical CPU cores
n     = utils.get_num_threads()    # current pool size (default: hardware_concurrency)

utils.set_num_threads(4)           # resize to 4 worker threads
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::oc;

int cores = utils::get_core_count();    // physical CPU cores
int n     = utils::get_num_threads();   // current pool size (default: hardware_concurrency)

utils::set_num_threads(4);              // resize to 4 worker threads
```
:::
::::

`set_num_threads(n)` is a process-global call intended to be made once at
startup, before any parallel work begins. Do not call it from inside a parallel
region or from a worker thread.

Passing `n <= 1` (including zero or negative values) selects single-threaded
mode: the pool stays alive but all work runs inline on the calling thread. This
is useful for reproducible profiling and for checking whether a result changes
under serial execution:

::::{tab-set}
:::{tab-item} Python
```python
utils.set_num_threads(1)   # single-threaded — pool bypassed, work runs inline
```
:::
:::{tab-item} C++
```cpp
utils::set_num_threads(1);   // single-threaded — pool bypassed, work runs inline
```
:::
::::

Tycho does not clamp `n` to the core count. If you set it higher than the
number of physical cores, the pool will still work correctly — the operating
system scheduler handles over-subscription. Whether that is beneficial depends
on your workload.

## Parallelize a batch of independent integrations

A single short integration does not benefit from multi-threading — the overhead
of distributing work to the pool dominates. Parallelism pays when you have many
independent initial conditions to propagate, such as a grid of departure
conditions, a Monte Carlo uncertainty sweep, or a manifold computation that
fans out from a periodic orbit.

Prepare a list of initial-condition vectors and a matching array of final times,
then call `integrate_parallel`:

::::{tab-set}
:::{tab-item} Python
```python
import numpy as np
from tychopy import utils

x0s = [x0_base + np.random.randn(7) * 1e-4 for _ in range(500)]
tfs  = np.full(500, 10.0)

results = integ.integrate_parallel(x0s, tfs, threads=utils.get_num_threads())
# results[i] is the final state-and-time vector for x0s[i]
```
:::
:::{tab-item} C++
```cpp
// x0s: std::vector<Eigen::VectorXd> of initial states; tfs: Eigen::VectorXd.
// The third argument is the partition count (the C++ analogue of threads=).
auto results = integ.integrate_parallel(x0s, tfs, utils::get_num_threads());
// results[i] is the final state-and-time vector for x0s[i]
```
:::
::::

For trajectory output use `integrate_dense_parallel`:

::::{tab-set}
:::{tab-item} Python
```python
trajs = integ.integrate_dense_parallel(x0s, tfs, threads=utils.get_num_threads())
# trajs[i] is the dense trajectory for x0s[i]
```
:::
:::{tab-item} C++
```cpp
auto trajs = integ.integrate_dense_parallel(x0s, tfs, utils::get_num_threads());
// trajs[i] is the dense trajectory for x0s[i]
```
:::
::::

Both methods release the GIL for the duration of the parallel work, so they are
safe to call from a Python thread without blocking other threads. The `threads`
argument sets how many blocks the batch is split into and dispatched onto the
**global** pool — it does *not* create new threads or resize the pool. The actual
concurrency is therefore bounded by the global pool size: if you have called
`set_num_threads(1)` (single-threaded mode), parallel calls run serially
regardless of the `threads` value, and a `threads` value larger than the pool
size simply queues more blocks onto the same workers. Size the pool with
`set_num_threads(n)` first to control how much parallelism these calls can use.

A worked example that uses `integrate_dense_parallel` to fan out from a
heteroclinic trajectory is `examples/python_examples/Heteroclinic.py`.

## Set optimizer NLP partitions

The optimizer's NLP Jacobian can be computed in parallel by splitting the
collocation defects into partitions, each evaluated on a separate thread. Use
the single-argument `set_num_partitions` overload:

::::{tab-set}
:::{tab-item} Python
```python
phase.set_num_partitions(4)   # split the NLP into 4 partitions
```
:::
:::{tab-item} C++
```cpp
phase.set_num_partitions(4);   // split the NLP into 4 partitions
```
:::
::::

There is also a two-argument overload `set_num_partitions(n, qp_threads)` that
simultaneously sets the Pardiso linear-solver thread count. Avoid this overload
unless you specifically need to pin the Pardiso thread count independent of the
partition count; the two arguments conflate distinct resources and are easy to
misuse:

::::{tab-set}
:::{tab-item} Python
```python
# Prefer this:
phase.set_num_partitions(4)

# Avoid this unless you know exactly what qp_threads controls:
# phase.set_num_partitions(4, 2)
```
:::
:::{tab-item} C++
```cpp
// Prefer this:
phase.set_num_partitions(4);

// Avoid this unless you know exactly what qp_threads controls:
// phase.set_num_partitions(4, 2);
```
:::
::::

The partition count is also available as the read-write attribute
`phase.num_partitions`.

## Coordinate thread budgets

When combining batch integration with an optimizer solve in the same script, be
aware that both consume the same pool. A common pattern is to use a high thread
count for the initial-guess sweep and then let the optimizer run at the same
count for its Jacobian partitions:

::::{tab-set}
:::{tab-item} Python
```python
utils.set_num_threads(utils.get_core_count())

# Generate initial guesses in parallel
trajs = integ.integrate_dense_parallel(x0s, tfs, threads=utils.get_num_threads())

# Build and solve the phase — the optimizer uses the same global pool
phase = ode.phase("LGL3", trajs[best_idx], 50)
phase.set_num_partitions(utils.get_num_threads())
phase.optimize()
```
:::
:::{tab-item} C++
```cpp
utils::set_num_threads(utils::get_core_count());

// Generate initial guesses in parallel
auto trajs = integ.integrate_dense_parallel(x0s, tfs, utils::get_num_threads());

// Build and solve the phase — the optimizer uses the same global pool
auto phase = ode.phase(TranscriptionModes::LGL3, trajs[best_idx], 50);
phase.set_num_partitions(utils::get_num_threads());
phase.optimize();
```
:::
::::

If you are running Tycho inside a larger parallel job (e.g. a SLURM task that
already allocates multiple processes), reduce the pool to the cores allocated to
your process rather than the full machine count to avoid oversubscription. This
step reads a process environment variable, so it is shown in Python only; in C++
read it with `std::getenv`:

```python
import os
allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", utils.get_core_count()))
utils.set_num_threads(allocated)
```

## See also

- {doc}`Python reference </reference/python/utilities>` — `utils.set_num_threads`,
  `utils.get_num_threads`, `utils.get_core_count`, and `BumpAllocator`.
- {doc}`Python reference </reference/python/integrators>` — `integrate_parallel`,
  `integrate_dense_parallel`, `integrate_stm_parallel`, and all batch methods.
- {doc}`Choosing an integrator </user_guide/how_to/choosing_an_integrator>` — how to pick
  an algorithm and configure tolerances before running a parallel batch.
