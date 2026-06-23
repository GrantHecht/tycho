(explanation-integration-and-parallelism)=
# Integration and parallelism in Tycho

Tycho's integrators propagate ordinary differential equations (ODEs) forward in time
using adaptive embedded Runge-Kutta methods, with dense output, event detection, and
variational sensitivity propagation built in. A work-stealing thread pool makes
batch-parallel and multi-trajectory integration practical without explicit thread
management. This page explains the design behind each of these capabilities and the
reasoning that guides when and how to use them.

It is conceptual rather than exhaustive. For the full API surface see the
{doc}`Python reference </reference/python/integrators>` and the
{doc}`C++ reference </reference/cpp/integrators>`.

The integrators operate on ODEs expressed as
{doc}`VectorFunctions </explanation/vector_function>`. In the optimal-control
setting the same ODE is typically handled through
{doc}`direct collocation </explanation/direct_collocation>`, which avoids
integrating forward at all; shooting-mode use of these integrators is most relevant
for initial-guess generation, sensitivity analysis, and single-arc propagation tasks.

## Adaptive embedded Runge-Kutta integration

### Embedded pairs and error estimation

A Runge-Kutta method advances the ODE state $x(t)$ from $t_0$ to $t_0 + h$ by
evaluating the right-hand side $f(x, t)$ at a number of intermediate stages and
forming a weighted combination of those evaluations. The key idea behind
*adaptive* integration is to carry two such combinations — a high-order solution
$x_{n+1}$ and an embedded lower-order solution $\hat{x}_{n+1}$ — from the same
set of stage evaluations. Their difference

$$
e_{n+1} = x_{n+1} - \hat{x}_{n+1}
$$

is a local error estimate: it measures how much the lower-order method disagrees
with the higher-order one over the step $h$, and thus how much the true solution
has been approximated. The adaptive loop uses this estimate to decide whether to
accept the step and what size to try next.

Because both solutions share the same stage evaluations, the embedded error estimate
is essentially free — it costs only the subtraction and norm computation, not
additional function evaluations. This "free error estimate" is the defining feature
of an embedded pair, and it is what makes all the algorithms in the `IVPAlg` menu
computationally practical.

### The IVPAlg menu

Tycho provides eight user-selectable embedded Runge-Kutta algorithms via the
`IVPAlg` enum:

| Algorithm | Stages | Order pair | FSAL | Notes |
| --------- | ------ | ---------- | ---- | ----- |
| `BS3`     | 4      | 3(2)       | yes  | Low-order, fast, coarse tolerances |
| `Tsit5`   | 7      | 5(4)       | yes  | Efficient general-purpose method |
| `DOPRI54` | 7      | 5(4)       | yes  | Classic Dormand-Prince; Matlab's `ode45` default |
| `BS5`     | 8 + 3  | 5(4)       | no   | Smooth dense output; 3 extra interp stages |
| `DOPRI87` | 13     | 8(7)       | no   | High-order; default for tight tolerances |
| `Vern7`   | 10 + 6 | 7(6)       | no   | High-order + smooth interpolant |
| `Vern8`   | 13 + 8 | 8(7)       | no   | High-order + smooth interpolant |
| `Vern9`   | 16 + 10| 9(8)       | no   | Highest order; expensive per step |

**Order pair notation.** `p(q)` means the main solution has order $p$ (the local
error scales as $h^{p+1}$) and the embedded error estimator has order $q$.

**FSAL (First Same As Last).** In an FSAL method the last stage evaluation of step
$n$ is exactly the first stage evaluation of step $n+1$. This means a step that
gets accepted donates one function evaluation to the next step at no extra cost.
`BS3`, `Tsit5`, and `DOPRI54` are FSAL; the higher-order methods are not.

**Choosing an algorithm.** The tradeoffs are:

- **`BS3`** — use for coarse tolerances ($\gtrsim 10^{-4}$) or when step count matters
  more than per-step cost. Fast FSAL and simple interpolant.
- **`Tsit5` / `DOPRI54`** — general-purpose at moderate tolerances
  ($10^{-6}$–$10^{-8}$). `Tsit5` is often slightly more efficient than `DOPRI54`
  on modern hardware due to optimized coefficients; `DOPRI54` matches Matlab's
  `ode45` and is familiar.
- **`DOPRI87`** — default for tight tolerances or stiff-ish problems where
  fewer, larger steps are preferable to many small ones. 13 stages but the high
  order means it can take much larger steps than a 5th-order method at the same
  accuracy.
- **`Vern7/8/9`** — use when smooth dense output (accurate continuous interpolation
  between accepted steps) is important, such as when the event-detection accuracy
  matters or when a large interpolant table is needed. The extra interpolation
  stages are paid only at accepted steps and produce a high-order polynomial rather
  than the lower-order default midpoint interpolant.

## Step-size control and error norms

### How the controller works

After each attempted step the integrator has an error estimate $e_{n+1}$ and a
user-supplied tolerance vector. It first converts the raw error to a *normalized
error norm* $\text{EEst}$: a single scalar that measures whether the step was
accurate enough. The controller then decides:

- If $\text{EEst} \leq 1$: accept the step.
- If $\text{EEst} > 1$: reject it and retry with a smaller $h$.

In either case the controller proposes a new step size for the next attempt:

$$
h_{\text{new}} = \frac{h}{q}, \qquad
q = \frac{\text{EEst}^{1/(p+1)}}{\gamma}
$$

where $p$ is the order of the embedded error estimator (the lower-order solution),
$\gamma < 1$ is a safety factor (default 0.9), and $q$ is clipped to the range
$[1/q_{\max}, 1/q_{\min}]$ to prevent runaway growth or catastrophic shrinkage.
The exponent $1/(p+1)$ — not $1/p$ — is correct: it matches the relationship between
the estimator order, the truncation error, and the asymptotic step-growth law
derived from assuming $\text{EEst} \approx C h^{p+1}$ and solving for the $h$ that
would give $\text{EEst} = 1$.

### The controller family

Tycho provides three controllers with increasing sophistication, selectable at
runtime via `IVPController`:

**`I` (integral controller).** Uses only the current error estimate:
$q = \text{EEst}^{1/(p+1)} / \gamma$. Simple and robust but can oscillate on
steps near the tolerance boundary.

**`PI` (proportional-integral controller).** Incorporates the previous step's error
as well, dampening oscillations:
$q_{11} = \text{EEst}^{\beta_1}$ and $q = q_{11} / \text{errold}^{\beta_2}$,
where $\beta_1$ and $\beta_2$ are pre-scaled gains (the order factor is absorbed
into the $\beta$ values, matching the Julia `OrdinaryDiffEq` convention). The PI
controller is generally preferred over the pure I controller for smooth problems.

**`PID` (proportional-integral-derivative controller with Söderlind limiter).** Uses
a three-step error history and applies an $\arctan$-based limiter to the raw growth
factor, preventing large $h$ changes even when the error history is very favorable.
The accept condition is $\text{dt\_factor} \geq \text{accept\_safety}$ (default 0.81)
rather than $\text{EEst} \leq 1$, making it slightly more aggressive on marginal
steps.

All three controllers implement a *deadband*: when the proposed step-change factor
falls within the interval $[q_{\text{steady,min}}, q_{\text{steady,max}}]$, the
step size is left unchanged. This avoids unnecessary step resizing on nearly-smooth
segments where the change would be within the controller's noise floor.

### Error norms

The per-component scaled residual for component $i$ is:

$$
r_i = \frac{e_i}{\alpha_i + \max(|x_{0,i}|, |x_{1,i}|) \cdot \rho_i}
$$

where $\alpha_i$ is the absolute tolerance for component $i$, $\rho_i$ is the
relative tolerance, $x_0$ is the pre-step state, and $x_1$ is the post-step state.
This mixed absolute-relative scaling is the standard `calculate_residuals` formula
from Julia's `DiffEqBase` and prevents the denominator from collapsing when a state
component passes through zero.

Two error-norm strategies are available via `ErrorNormType`:

- **`RMS`** (root-mean-square): $\text{EEst} = \sqrt{\frac{1}{n}\sum_i r_i^2}$.
  This is the default and matches Julia's `ODE_DEFAULT_NORM`. It spreads the
  tolerance over all components evenly.
- **`MAX`** (L∞): $\text{EEst} = \max_i |r_i|$. More conservative; the worst
  component must satisfy the tolerance. Useful when one component is safety-critical
  and cannot be allowed to accumulate error even if others are well within bounds.

## Dense output and event detection

### Dense output — continuous trajectories between steps

An adaptive integrator's accepted steps land at irregular time values. Between
accepted times the solution is only defined by interpolation. *Dense output*
provides a continuous polynomial representation that makes the trajectory well-defined
at any time within the integration interval, not just at accepted step endpoints.

Tycho builds a polynomial interpolant from the stage evaluations at each accepted step.
For simpler methods (`BS3`, `Tsit5`, `DOPRI54`) the interpolant is a midpoint-based
polynomial derived from the existing Butcher stages at no additional cost. For
higher-order methods (`BS5`, `Vern7`, `Vern8`, `Vern9`) accurate dense output
requires additional "interpolation stages" — extra evaluations of $f(x, t)$ that are
not part of the main tableau but are needed to build the polynomial. These extra
evaluations are paid once per accepted step and are reflected in the stage counts
listed with the `+` notation in the `IVPAlg` table above. `DOPRI87` uses a single
extra evaluation at $f(x_f, t_f)$ (the final state of each accepted step) since its
last main stage does not land exactly at $(x_f, t_f)$.

The dense interpolant underpins two capabilities: querying the trajectory at arbitrary
times (for post-processing or initial-guess generation), and event detection.

### Event detection

An *event* is a scalar condition on the state that changes sign during the integration.
Physical examples include altitude becoming zero (ground impact), a Lyapunov orbit
crossing a surface of section, or a heliocentric angle reaching a target value.

The user-facing event specification in Tycho is `EventPack`, which bundles:

1. **Event function** — a scalar-output {doc}`VectorFunction </explanation/vector_function>`
   of the ODE state. A zero crossing of this function is the event.
2. **Direction filter** — `Rising` (+1), `Falling` (−1), or `Any` (0). A rising
   filter triggers only when the function passes through zero from negative to
   positive; a falling filter triggers only in the opposite direction.
3. **Stop count** — zero means "record all crossings but never stop"; $N > 0$ means
   "stop integration after the $N$-th recorded crossing of this event."

Detection is a two-phase process handled internally by `EventHandler`:

1. **Bracket.** After each accepted step the event function is evaluated at the new
   state and compared with its previous-step value. A sign change — filtered by
   direction — indicates a crossing in the bracket $[t_{\text{prev}}, t_{\text{next}}]$.
   The bracket is recorded.
2. **Refine.** Each bracketed crossing is refined to the zero location by a
   fast bisection pass followed by Newton's method on the dense interpolant table.
   The result is the state and time at the zero crossing, accurate to the requested
   refinement tolerance. If neither bisection nor Newton can resolve the crossing
   within the bracket (rare; requires a non-finite event function), the crossing is
   recorded as unresolved and counted against `n_failed_refinements`.

Multiple `EventPack` entries can be supplied simultaneously; `EventHandler` monitors
all of them independently per step.

## Variational integration — state-transition matrices

### What the STM is

The *state-transition matrix* (STM) $\Phi(t_f, t_0) \in \mathbb{R}^{n \times n}$ is
the first-order sensitivity of the final state $x_f = x(t_f)$ with respect to the
initial state $x_0$:

$$
\Phi(t_f, t_0) = \frac{\partial x_f}{\partial x_0}
$$

It answers the question: if the initial condition shifts by a small amount $\delta x_0$,
what is the resulting shift $\delta x_f \approx \Phi \, \delta x_0$ in the final state?
The STM is essential for orbit determination, sensitivity analysis, continuation
methods, and as the building block for gradient computations in single-shooting
optimization.

### Tycho's analytic chain-rule approach

Tycho does not compute the STM by perturbing the initial condition and differencing
propagations (finite differences). Instead, each Runge-Kutta stepper is itself a
{doc}`VectorFunction </explanation/vector_function>` — it maps the input state at
$t_0$ to the output state at $t_0 + h$ — and as a VectorFunction it exposes exact
analytic Jacobians through the standard Tycho derivative machinery.

The STM across the full trajectory from $t_0$ to $t_f$ is assembled by the
`STMDriver` via the chain rule:

$$
\Phi(t_f, t_0) = \Phi_N \cdot \Phi_{N-1} \cdots \Phi_2 \cdot \Phi_1
$$

where $\Phi_i = \partial x_{i+1} / \partial x_i$ is the analytic Jacobian of the
$i$-th RK step. Each $\Phi_i$ is obtained by calling `compute_jacobian` on the
stepper VectorFunction, which evaluates the Jacobian without any numerical
perturbation. The product accumulates left-to-right (or right-to-left in the adjoint
path) in a single pass over the dense trajectory states.

This approach gives exact (analytic) sensitivities limited only by floating-point
arithmetic, with no step-size perturbation bias, and is compatible with the
SuperScalar dispatch machinery — the chain-rule multiplication can process multiple
trajectories simultaneously by batching their stage Jacobians into wide SIMD
operations.

### Second-order sensitivities (integrate_stm2)

The `integrate_stm2` interface extends the first-order STM to second-order. It
computes, for a given adjoint vector $\ell_f$:

$$
\frac{\partial^2 (\ell_f \cdot x_f)}{\partial x_0^2}
$$

This is the Hessian of the scalar $\ell_f \cdot x_f$ (the adjoint-weighted final
state) with respect to the initial condition. It is computed by the
`STMDriver::calculate_jacobian_hessian` function using `compute_jacobian_adjointgradient_adjointhessian`
on each stepper VectorFunction — the same mechanism that provides second-order
information throughout the Tycho optimizer — accumulated backwards through the step
sequence via a symmetric recurrence. The result is the exact adjoint Hessian, again
without finite differences.

## Parallelism — the thread pool and dispatch helpers

### The global thread pool

Tycho owns a process-global *work-stealing thread pool* (a Meyers singleton returned
by `thread_pool()`). It is created lazily on first access and initialized to
`std::thread::hardware_concurrency()` worker threads. Each worker has its own
`WorkStealingQueue`: the owner pops from the front (LIFO, favoring cache-local recent
work), and idle workers steal from the back of neighbors' queues (FIFO, reducing
contention with the owner's LIFO pops).

### set_num_threads semantics

`set_num_threads(n)` controls how much of the system the pool is allowed to use:

- **$n \leq 1$** (including 0 or negative): selects *single-threaded mode*. Work
  submitted through the dispatch helpers runs inline on the calling thread. The pool
  itself stays alive but is bypassed by the `use_thread_pool()` guard that all
  dispatch helpers check first. This is useful for debugging, profiling, or
  environments where threading is undesirable.
- **$n > 1$**: resizes the pool to $n$ worker threads (draining any pending work,
  joining existing workers, and relaunching). Use at program startup or at the top
  of a Python script before any parallel work begins.

Before any call to `set_num_threads`, the pool runs at the process default of
`std::thread::hardware_concurrency()` threads.

### Dispatch helpers

Rather than enqueuing raw tasks, parallel work in Tycho goes through three dispatch
helpers that provide automatic fallback, per-dispatch synchronization, and exception
propagation:

**`parallel_sequence(n, func)`** invokes `func(i)` for each `i` in `[0, n)`. The
first $n-1$ calls are dispatched to the pool; the last runs inline on the calling
thread (keeping it productive). All complete before `parallel_sequence` returns.
Falls back to a sequential loop when $n \leq 1$ or the pool is inactive.

**`parallel_blocks(count, func, nparts)`** partitions the range `[0, count)` into
`nparts` blocks and calls `func(start, end)` once per block. Like `parallel_sequence`,
the last block runs inline, the rest are dispatched. Useful for work that is naturally
divided into contiguous index ranges (batch-integrating groups of trajectories, for
example).

**`parallel_task(nparts, pool_work, inline_work)`** runs one callable on a pool
worker and another on the calling thread simultaneously. Useful when there are
exactly two independent units of work and neither is trivially parallelizable into a
loop.

All three helpers use a `std::latch` (C++20) for per-dispatch synchronization.
This means multiple concurrent dispatches from different threads are inherently safe —
each dispatch waits only on its own latch, not a global barrier. Exceptions thrown by
any worker are captured and re-thrown on the calling thread after the latch is
reached.

The helpers guard against two misuse patterns that would produce undefined behavior
or deadlock:

- *Nested dispatch from a pool worker* — a pool worker cannot call a dispatch helper
  because it would block waiting for work it cannot execute (the worker is already
  occupied).
- *Dispatch during `set_num_threads`* — the pool is being torn down and rebuilt;
  enqueueing work during this window could lose tasks or deadlock.

Both conditions throw `std::logic_error` immediately rather than hanging silently.

### Batch and parallel integration

`integrate_parallel` and `integrate_dense_parallel` integrate multiple independent
trajectories concurrently using the thread pool. The ParallelDriver maps the batch
of initial conditions onto the pool via `parallel_sequence`, one trajectory per
dispatch slot. Each trajectory is integrated independently on a worker thread, with no
shared mutable state between trajectories.

The STM batch paths (`calculate_jacobians`, `calculate_jacobians_hessians`) use the
SuperScalar packing mechanism to process multiple trajectories simultaneously inside
a single worker call — `vsize` trajectories at a time, where `vsize` is the width of
`DefaultSuperScalar` (the compile-time SIMD lane count). This reduces per-trajectory
overhead by amortizing the loop over step indices across all lanes.

### The BumpAllocator — allocation-free ODE evaluation

Every VectorFunction evaluation during integration requires temporary Eigen matrices
(intermediate stage values, Jacobian scratch space, Hessian temporaries). Allocating
these from the system heap on every step would be prohibitively expensive and would
introduce lock contention between pool workers.

`BumpAllocator` avoids this with a *per-thread bump arena*: a pre-allocated block of
SIMD-aligned memory that satisfies all temporary requests by advancing a pointer, with
no locking and no system heap calls. Each worker thread owns its own arena, so all
temporaries are thread-local and completely free of contention.

The arena starts with a default capacity of 64 elements per thread. It
*self-tunes*: if an evaluation overflows the arena (because the state is larger than
expected), the overflow lands in a fallback heap block and the high-water mark is
updated. On the next save/restore cycle — at the start of the next trajectory — the
arena resizes itself to the observed high-water mark. After one warm-up trajectory the
arena is correctly sized and subsequent evaluations are allocation-free.

For compile-time-sized VectorFunctions (where all matrix dimensions are known
statically), `BumpAllocator::allocate_run` bypasses the arena entirely and places
temporaries on the real C++ stack with zero overhead.

Call `BumpAllocator::resize(n)` at startup to pre-size all per-thread arenas to `n`
elements and avoid the warm-up overhead for known large problems.

## Putting it together

The conceptual flow of a parallel batch integration is:

1. The user provides a list of initial conditions and an ODE VectorFunction.
2. `integrate_parallel` dispatches one trajectory per pool worker via
   `parallel_sequence`.
3. Each worker runs an `AdaptiveDriver` loop: attempt a step with the selected `IVPAlg`,
   evaluate the error norm, run the controller to accept or reject and propose $h_{\text{new}}$,
   check events via `EventHandler`, record the dense interpolant table.
4. All temporary allocations inside the ODE evaluation go through the worker thread's
   `BumpAllocator` arena — no cross-thread allocation contention.
5. When all workers finish, the calling thread collects results and re-throws any
   accumulated worker exceptions.
6. If STM sensitivity is requested (`integrate_stm`), `STMDriver` traverses the dense
   state sequence, multiplying analytic step Jacobians via the chain rule to produce
   $\Phi(t_f, t_0)$ without finite differences.

The same ODE VectorFunction participates in the
{doc}`direct collocation </explanation/direct_collocation>` transcription when a
`Phase` is constructed, where the solver enforces defect constraints rather than
integrating forward. The integrators are most useful outside that setting — for
initial-guess generation, Monte-Carlo batch propagation, and sensitivity analysis that
operates on a single arc rather than a full NLP.

For a full API listing — `Integrator`, `IVPAlg`, `IVPController`, `ErrorNormType`,
`EventPack`, `set_num_threads`, `BumpAllocator`, and all integrate variants — see the
{doc}`Python reference </reference/python/integrators>`,
{doc}`C++ reference </reference/cpp/integrators>`,
{doc}`Python utilities reference </reference/python/utilities>`, and
{doc}`C++ utilities reference </reference/cpp/utilities>`.
