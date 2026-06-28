(tutorial-propagating-an-ode)=
# Propagating an ODE

Before you optimize a trajectory you usually want to *propagate* one: given a
starting state, integrate the dynamics forward in time and see where they go.
Tycho's **integrators** do exactly that — adaptive-step Runge–Kutta propagation
of any ODE you can express as dynamics. Where
{doc}`your first VectorFunction </tutorials/basics/your_first_vectorfunction>`
covered *defining* dynamics, this tutorial uses them: you will wrap a
VectorFunction in an ODE, build an integrator from it, propagate a trajectory,
and check that the result is accurate.

The integrator is also a natural companion to the optimizer. The very same ODE
you propagate here is the one you hand to a `Phase` in the
{doc}`next tutorial </tutorials/basics/your_first_phase>`. Collocation needs an
initial guess to refine, and propagating the dynamics is *one* common way to build
one — alongside analytic approximations, simple interpolation between the boundary
states, continuation from a nearby solution, or a previous result. Propagation is
also how you sanity-check dynamics before you optimize them.

The worked system is the **simple pendulum**: a mass on a rigid rod swinging
under gravity. It has no control input, conserves energy exactly, and oscillates
forever — which makes it a perfect target, because we can *measure* the
integrator's accuracy by how well it conserves that energy.

This is a hands-on, learning-oriented walkthrough. For a task-focused reference on
picking an algorithm and tuning tolerances, see
{doc}`How to choose and configure an integrator </how_to/choosing_an_integrator>`.
Every `{doctest}` block below (the ones showing `>>>` prompts) is executed as part
of Tycho's test suite, so the numbers shown are real; the C++ equivalents appear
alongside in tabs as illustrative builder-API fragments. Each step builds on the
previous one — run them in order in a REPL or script to reproduce the example.

## Setup

We need NumPy for the state arrays, `vector_functions` (`vf`) to build the
dynamics, and `optimal_control` (`oc`) for the ODE and integrator machinery.

```{doctest}
>>> import numpy as np
>>> from tychopy import vector_functions as vf
>>> from tychopy import optimal_control as oc
```

## 1. Define the ODE

The pendulum state is $[\theta, \omega]$ — the angle from vertical and the angular
rate. With gravity $g$ and rod length $L$ the dynamics are

$$
\dot{\theta} = \omega, \qquad
\dot{\omega} = -\frac{g}{L}\,\sin\theta .
$$

To use these dynamics we subclass {py:class}`~tychopy.optimal_control.ODEBase`,
exactly as for a phase. The pattern is the same: declare the variable layout with
{py:class}`~tychopy.optimal_control.ODEArguments` (here **2 states and 0
controls** — nothing steers a free pendulum), pull out the symbolic pieces,
assemble the right-hand side as a single VectorFunction with `vf.stack`, and hand
it to `super().__init__` with the state and control counts.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> class Pendulum(oc.ODEBase):
...     def __init__(self, g, L):
...         args = oc.ODEArguments(2, 0)
...         theta, omega = args.x_vec().tolist()
...         ode = vf.stack([omega, -(g / L) * vf.sin(theta)])
...         super().__init__(ode, 2, 0)
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

double g = 9.81, L = 1.0;
// ODEBuilder(n_state, n_control); the lambda builds the right-hand side.
auto ode = ODEBuilder(2, 0)
               .define([g, L](auto &args) {
                   auto theta = args.x_var(0);
                   auto omega = args.x_var(1);
                   return stack(omega, sin(theta) * (-g / L));
               })
               .var_names({{"theta", 0}, {"omega", 1}, {"t", 2}})
               .build();
```
:::
::::

`ODEArguments(2, 0)` lays the input out as $[\theta, \omega, t]$ — the two state
variables followed by time. `args.x_vec()` is the state sub-vector; with no
control there is nothing else to pull out. Instantiate it with
$g = 9.81\ \mathrm{m/s^2}$ and $L = 1\ \mathrm{m}$:

```{doctest}
>>> g, L = 9.81, 1.0
>>> ode = Pendulum(g, L)
```

## 2. Build an integrator

{py:meth}`~tychopy.optimal_control.ODEBase.integrator` builds a propagator bound
to this ODE. The single argument is the nominal initial step size; the algorithm
defaults to `DOPRI87`, an 8th-order Dormand–Prince pair that is an excellent
general-purpose choice. In C++ the integrator is assembled with a fluent builder.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> integ = ode.integrator(0.01)
```
:::
:::{tab-item} C++
```cpp
auto integ = ode.integrator().step(0.01).build();
```
:::
::::

The step size is only a starting point — the adaptive controller grows and shrinks
it automatically to hold the error within tolerance, so you do not need to guess it
precisely.

## 3. Propagate to a final state

The state Tycho integrates is the **full ODE input** $[\theta, \omega, t]$: the
two states plus the current time as the last component. We start the pendulum at
$\theta_0 = 1$ rad released from rest ($\omega_0 = 0$) at $t_0 = 0$, and propagate
to $t_f = 5$ s. {py:meth}`~tychopy.optimal_control.ode_x.integrator.integrate`
returns the state-and-time vector at `tf`:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> x0 = np.array([1.0, 0.0, 0.0])     # theta0 = 1 rad, omega0 = 0, t0 = 0
>>> xf = integ.integrate(x0, 5.0)
>>> [round(float(c), 4) for c in xf]
[-0.5303, -2.5148, 5.0]
```
:::
:::{tab-item} C++
```cpp
Eigen::Vector3d x0(1.0, 0.0, 0.0);   // theta0, omega0, t0
auto xf = integ.integrate(x0, 5.0);
```
:::
::::

After 5 seconds the bob has swung through several oscillations and is mid-swing at
$\theta \approx -0.53$ rad. The last component of `xf` is just `tf`.

## 4. Get the full trajectory

A single end state is rarely enough — you usually want the path. `integrate_dense`
returns the whole trajectory. Pass an integer `n` to get exactly `n` evenly-spaced
samples, evaluated on the integrator's polynomial dense-output interpolant:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> traj = integ.integrate_dense(x0, 5.0, 6)
>>> len(traj), len(traj[0])
(6, 3)
>>> [round(float(c), 4) for c in traj[-1]]
[-0.5303, -2.5148, 5.0]
```
:::
:::{tab-item} C++
```cpp
auto traj = integ.integrate_dense(x0, 5.0, 6);   // 6 evenly spaced samples
```
:::
::::

Each row has the same layout as the ODE input, $[\theta, \omega, t]$, and the last
row matches the `integrate` result above. Omit the count to get one row per
*accepted adaptive step* instead — the natural sampling the controller chose,
denser where the dynamics move fastest.

## 5. Measure the accuracy

The pendulum conserves total energy $E = \tfrac{1}{2}\omega^2 + \tfrac{g}{L}(1 -
\cos\theta)$ exactly. A numerical integrator does not — but a good one keeps the
drift tiny, which gives us a concrete accuracy gauge. The adaptive controller
targets an absolute and a relative error tolerance on every step; out of the box
the absolute tolerance is a tight `1e-12`. Re-propagate over the controller's own
step grid (omit the count) and watch how little the energy moves:

```{doctest}
>>> dense = integ.integrate_dense(x0, 5.0)     # one row per accepted step
>>> def energy(row):
...     th, om = float(row[0]), float(row[1])
...     return 0.5 * om * om + (g / L) * (1.0 - np.cos(th))
>>> E = np.array([energy(r) for r in dense])
>>> bool(np.max(np.abs(E - E[0])) < 1e-9)
True
```

Over the entire 5-second swing the energy holds to better than one part in a
billion — the integrator is faithfully reproducing the true dynamics, not slowly
leaking accuracy. That fidelity is exactly what the tolerances buy: loosen
`set_abs_tol` for faster, rougher scans, or tighten it (and add `set_rel_tol`)
when you need still more. The full set of knobs — per-component tolerances, the
step-size controller, the error norm — is covered in the
{doc}`integrator how-to </how_to/choosing_an_integrator>`.

## 6. Choose a different algorithm

`DOPRI87` is the default, but you can ask for any of Tycho's embedded Runge–Kutta
pairs by passing the algorithm **first**, before the step size — as an
{py:class}`~tychopy.optimal_control.IVPAlg` enum or an equivalent string:

```{doctest}
>>> fast = ode.integrator(oc.IVPAlg.DOPRI54, 0.01)   # or "DOPRI54"
>>> [round(float(c), 4) for c in fast.integrate(x0, 5.0)]
[-0.5303, -2.5148, 5.0]
```

The lower-order `DOPRI54` agrees to four decimals here — for a smooth problem at
moderate tolerance the cheaper method is plenty. Which pair wins depends on the
accuracy you need and how smooth the dynamics are; the
{doc}`how-to table </how_to/choosing_an_integrator>` lays out the trade-offs for
all of `DOPRI54`, `DOPRI87`, `Tsit5`, `BS3`, `BS5`, `Vern7`, `Vern8`, and `Vern9`.

## 7. Visualize the trajectory

The most informative views of an oscillator are the angle over time and the
**phase portrait** — $\omega$ against $\theta$ — which should trace a single closed
loop for an energy-conserving system. Take a densely sampled trajectory, stack it
into a NumPy array, and slice out the columns.

```python
import matplotlib.pyplot as plt
import numpy as np

T = np.array(integ.integrate_dense(x0, 5.0, 400))  # rows of [theta, omega, t]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
ax1.plot(T[:, 2], T[:, 0])
ax1.set(xlabel="time [s]", ylabel=r"angle $\theta$ [rad]", title="Pendulum swing")

ax2.plot(T[:, 0], T[:, 1], color="C1")
ax2.set(xlabel=r"$\theta$ [rad]", ylabel=r"$\omega$ [rad/s]", title="Phase portrait")

fig.tight_layout()
plt.show()
```

```{eval-rst}
.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from tychopy import vector_functions as vf
   from tychopy import optimal_control as oc

   class Pendulum(oc.ODEBase):
       def __init__(self, g, L):
           args = oc.ODEArguments(2, 0)
           theta, omega = args.x_vec().tolist()
           ode = vf.stack([omega, -(g / L) * vf.sin(theta)])
           super().__init__(ode, 2, 0)

   g, L = 9.81, 1.0
   ode = Pendulum(g, L)
   integ = ode.integrator(0.01)
   x0 = np.array([1.0, 0.0, 0.0])
   T = np.array(integ.integrate_dense(x0, 5.0, 400))

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
   ax1.plot(T[:, 2], T[:, 0])
   ax1.set(xlabel="time [s]", ylabel=r"angle $\theta$ [rad]", title="Pendulum swing")
   ax1.grid(True, alpha=0.3)

   ax2.plot(T[:, 0], T[:, 1], color="C1")
   ax2.set(xlabel=r"$\theta$ [rad]", ylabel=r"$\omega$ [rad/s]", title="Phase portrait")
   ax2.grid(True, alpha=0.3)

   fig.tight_layout()
```

The angle traces a clean, slightly non-sinusoidal oscillation (the large initial
amplitude makes the period longer than the small-angle approximation predicts),
and the phase portrait closes on itself — a visual confirmation of the energy
conservation we measured above.

## What you learned

- Tycho **integrators** propagate any ODE forward in time with adaptive-step
  Runge–Kutta methods. You build one with `ode.integrator(dt)`.
- An ODE for integration is defined exactly like one for a phase — subclass
  `oc.ODEBase`, declare the layout with `oc.ODEArguments`, build the right-hand
  side as a VectorFunction. A system with no control uses `ODEArguments(n, 0)`.
- The integrated state is the full ODE input $[\,\text{state}\dots, t\,]$;
  `integrate` returns the final state-and-time, `integrate_dense` returns the
  whole trajectory (a fixed sample count, or one row per accepted step).
- Accuracy is set by the error tolerances (`set_abs_tol` / `set_rel_tol`) and the
  choice of algorithm. To use an algorithm other than the default `DOPRI87`, pass
  it as the first argument, before the step size.

## Next steps

- {doc}`Setting up a phase </tutorials/basics/your_first_phase>` — feed an ODE
  like this one to the optimizer and solve an optimal-control problem.
- {doc}`How to choose and configure an integrator </how_to/choosing_an_integrator>`
  — algorithm trade-offs, per-component tolerances, controllers, and event
  detection.
- {doc}`Python reference </reference/python/integrators>` — the complete
  integrator API, including state-transition-matrix propagation, event packs, and
  parallel batch integration.
</content>
