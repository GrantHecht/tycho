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
{doc}`Choosing an integrator </how_to/choosing_an_integrator>`. For the
conceptual overview of Tycho's threading architecture see
{doc}`Integration and parallelism </explanation/integration_and_parallelism>`.

## Query and set the thread count

The utilities submodule exposes three functions for thread management:

```python
from tychopy import utils

cores = utils.get_core_count()     # physical CPU cores
n     = utils.get_num_threads()    # current pool size (default: hardware_concurrency)

utils.set_num_threads(4)           # resize to 4 worker threads
```

`set_num_threads(n)` is a process-global call intended to be made once at
startup, before any parallel work begins. Do not call it from inside a parallel
region or from a worker thread.

Passing `n <= 1` (including zero or negative values) selects single-threaded
mode: the pool stays alive but all work runs inline on the calling thread. This
is useful for reproducible profiling and for checking whether a result changes
under serial execution:

```python
utils.set_num_threads(1)   # single-threaded — pool bypassed, work runs inline
```

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

```python
import numpy as np
from tychopy import utils

x0s = [x0_base + np.random.randn(7) * 1e-4 for _ in range(500)]
tfs  = np.full(500, 10.0)

results = integ.integrate_parallel(x0s, tfs, threads=utils.get_num_threads())
# results[i] is the final state-and-time vector for x0s[i]
```

For trajectory output use `integrate_dense_parallel`:

```python
trajs = integ.integrate_dense_parallel(x0s, tfs, threads=utils.get_num_threads())
# trajs[i] is the dense trajectory for x0s[i]
```

Both methods release the GIL for the duration of the parallel work, so they are
safe to call from a Python thread without blocking other threads. The `threads`
argument controls how many tasks this call dispatches; it does not resize the
global pool set by `set_num_threads`. You can therefore run a batch with a
different thread count than the global default without affecting any concurrent
work.

A worked example that uses `integrate_dense_parallel` to fan out from a
heteroclinic trajectory is `examples/python_examples/Heteroclinic.py`.

## Set optimizer NLP partitions

The optimizer's NLP Jacobian can be computed in parallel by splitting the
collocation defects into partitions, each evaluated on a separate thread. Use
the single-argument `set_num_partitions` overload:

```python
phase.set_num_partitions(4)   # split the NLP into 4 partitions
```

There is also a two-argument overload `set_num_partitions(n, qp_threads)` that
simultaneously sets the Pardiso linear-solver thread count. Avoid this overload
unless you specifically need to pin the Pardiso thread count independent of the
partition count; the two arguments conflate distinct resources and are easy to
misuse:

```python
# Prefer this:
phase.set_num_partitions(4)

# Avoid this unless you know exactly what qp_threads controls:
# phase.set_num_partitions(4, 2)
```

The partition count is also available as the read-write attribute
`phase.num_partitions`.

## Coordinate thread budgets

When combining batch integration with an optimizer solve in the same script, be
aware that both consume the same pool. A common pattern is to use a high thread
count for the initial-guess sweep and then let the optimizer run at the same
count for its Jacobian partitions:

```python
utils.set_num_threads(utils.get_core_count())

# Generate initial guesses in parallel
trajs = integ.integrate_dense_parallel(x0s, tfs, threads=utils.get_num_threads())

# Build and solve the phase — the optimizer uses the same global pool
phase = ode.phase("LGL3", trajs[best_idx], 50)
phase.set_num_partitions(utils.get_num_threads())
phase.optimize()
```

If you are running Tycho inside a larger parallel job (e.g. a SLURM task that
already allocates multiple processes), reduce the pool to the cores allocated to
your process rather than the full machine count to avoid oversubscription:

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
- {doc}`Integration and parallelism </explanation/integration_and_parallelism>` —
  conceptual background on the thread pool, SuperScalar vectorization, and when
  parallelism helps.
- {doc}`Choosing an integrator </how_to/choosing_an_integrator>` — how to pick
  an algorithm and configure tolerances before running a parallel batch.
