# How to choose and configure an integrator

Propagating an ODE forward in time requires picking a Runge-Kutta algorithm,
setting tolerances appropriate to the problem, and understanding what trade-offs
each choice makes between cost per step and accuracy per step. The wrong choice
can leave you paying for accuracy you do not need, or accumulating error that
causes phase initial-guess failures or optimizer constraint violations downstream.
This recipe shows how to construct an integrator, configure it, and decide which
algorithm to use for a given task.

This recipe assumes you have a defined ODE — a class that inherits from
{py:class}`~tychopy.optimal_control.ODEBase` or one of the built-in dynamics
models. For the conceptual background on adaptive stepping and the error-control
loop see {doc}`Integration and parallelism </user_guide/how_to/threading_model>`.

## Construct an integrator

Call `ode.integrator(dt)` to get an integrator with the default algorithm
(`DOPRI87`) and a nominal initial step size of `dt`:

```python
integ = ode.integrator(0.01)
```

To select a different algorithm, pass it as the **first** positional argument,
before `dt`. Both an enum value and a string are accepted:

```python
from tychopy import optimal_control as oc

# Enum form
integ = ode.integrator(oc.IVPAlg.DOPRI54, 0.01)

# Equivalent string form
integ = ode.integrator("DOPRI54", 0.01)
```

The method-first, step-second order is fixed — there is no `alg=` keyword
argument. The accepted string names are `"DOPRI54"` / `"DP54"`, `"DOPRI87"` /
`"DP87"`, `"Tsit5"`, `"BS3"`, `"BS5"`, `"Vern7"`, `"Vern8"`, and `"Vern9"`.

## Run an integration

`integrate` returns the final state-and-time vector at `tf`:

```python
xf = integ.integrate(x0, tf)
```

`integrate_dense` returns the full trajectory. Without a count argument it
returns one state vector per accepted adaptive step:

```python
traj = integ.integrate_dense(x0, tf)
```

With an integer `n` it returns exactly `n` evenly-spaced states evaluated on
the polynomial dense-output interpolant:

```python
traj = integ.integrate_dense(x0, tf, 200)
```

Each element of `traj` is an array with the same layout as the ODE input:
`[state..., t]`.

## Set error tolerances

The adaptive controller targets both absolute and relative error bounds on every
state component. By default the integrator uses a pure **absolute** tolerance:
`1e-12` absolute and `0` relative. This is conservative enough for most
orbit-mechanics initial-guess generation. Loosen the absolute tolerance for fast,
rough scans; tighten it (and/or add a relative tolerance) when you need higher
accuracy:

```python
integ.set_abs_tol(1.0e-9)   # uniform absolute tolerance for all states
integ.set_rel_tol(1.0e-9)   # uniform relative tolerance

# Per-component variant — useful when state variables differ wildly in scale
import numpy as np
integ.set_abs_tols(np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9]))  # one per state component (excludes time)
```

Setting `integ.adaptive = False` switches to a fixed step size equal to
`integ.def_step_size`. Use fixed-step mode only when you have a specific reason
to control the step size directly — e.g. comparing trajectories at identical
discrete times, or bit-reproducible integration for testing:

```python
integ.adaptive = False
integ.def_step_size = 0.001
```

## Tune the step-size controller

Two optional knobs give finer control over how the adaptive controller adjusts
step size between accepted steps.

The **controller** selects the feedback scheme:

```python
integ.set_controller("PI")   # I, PI, or PID
```

The default controller is algorithm-dependent: `DOPRI87` (the default algorithm)
uses the basic `I` controller; the other methods default to `PI`. `PI`
incorporates the previous step's error as well as the current one, which dampens
step-size oscillations and is generally preferred for smooth problems. `PID` adds
a further derivative term for very smooth problems at tight tolerances.

The **error norm** determines how per-component errors are combined into the
scalar that drives the controller:

```python
integ.set_error_norm("RMS")   # RMS (root-mean-square) or MAX
```

`RMS` is the default. `MAX` can be useful when one state component is the
binding constraint and you need that component's error to stay below tolerance
even if others are more forgiving.

## Decide which algorithm to use

| Algorithm | Order | Cost per step | When to use |
|-----------|-------|---------------|-------------|
| `DOPRI54` | 5(4)  | Moderate      | General-purpose default for moderate-accuracy needs; classic Dormand-Prince |
| `DOPRI87` | 8(7)  | Higher        | Default when no algorithm is specified; best balance of high accuracy and robustness for orbit mechanics |
| `Tsit5`   | 5(4)  | Low           | Efficient general-purpose propagation; same stage count as DOPRI54 (7) but optimized coefficients give a lower error constant |
| `BS3`     | 3(2)  | Very low      | Cheap initial guesses at loose tolerances; short-duration propagation |
| `BS5`     | 5(4)  | Moderate      | Smooth problems where BS5's coefficient set gives better error constants |
| `Vern7`   | 7(6)  | Moderate–high | High accuracy without the full cost of an 8th-order method |
| `Vern8`   | 8(7)  | High          | Tight tolerances on smooth problems |
| `Vern9`   | 9(8)  | Highest       | Very tight tolerances; benefits are felt only at tolerances below `1e-11` |

For most astrodynamics work — orbit continuation, initial-guess generation,
STM propagation — `DOPRI87` (the default) is the right choice: its higher
order means it takes large steps on smooth Keplerian arcs, and the embedded
7th-order error estimate is well-calibrated. Drop to `Tsit5` or `BS3` when you
are generating a rough initial guess and want to minimize wall time. Switch to
`Vern9` only when the application genuinely requires sub-`1e-11` accuracy, such
as long-arc precision orbit determination.

A complete worked example is `examples/python_examples/OrbitContinuation.py`,
which uses `ode.integrator(dt)` with the default `DOPRI87` algorithm and
`integrate_dense` to produce initial guesses across a family of orbits.

## See also

- {doc}`Python reference </reference/python/integrators>` — the full
  `Integrator` API, including STM propagation, event detection, and
  vectorized batch methods.
- {doc}`Integration and parallelism </user_guide/how_to/threading_model>` —
  conceptual background on adaptive step-size control, dense output, and the
  error norm loop.
- {doc}`Threading model </user_guide/how_to/threading_model>` — how to distribute many
  independent integrations across worker threads with `integrate_parallel` and
  `integrate_dense_parallel`.
