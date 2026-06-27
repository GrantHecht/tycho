(integrators-python)=
# Integrators

The Python API for Tycho's integrators subsystem, exposed through
`tychopy.optimal_control`. The subsystem provides adaptive-step Runge-Kutta
integrators that can propagate any ODE defined as a
{py:class}`~tychopy.optimal_control.PhaseInterface` ODE. Users never
construct the ``Integrator`` class directly; instead they call
``ode.integrator(dt)`` on an ODE phase type, which returns a ready-to-use
integrator bound to that ODE.

Three enums control the integrator's algorithm, step-size controller, and
error-norm strategy. ``EventPack`` specifies zero-crossing events to be
located during integration.

```{eval-rst}
.. currentmodule:: tychopy.optimal_control
```

## Enumerations

### ``IVPAlg`` — Runge-Kutta algorithm

Selects the embedded Runge-Kutta pair used to advance the ODE and produce a
local error estimate for adaptive step-size control.

```{eval-rst}
.. autoclass:: IVPAlg
   :members:
   :undoc-members:
```

### ``IVPController`` — step-size controller

Chooses the control law that adjusts the step size between accepted steps.
``I`` is the classic single-term controller; ``PI`` and ``PID`` use
additional derivative/integral terms for smoother step-sequence behaviour
near stiff transitions.

```{eval-rst}
.. autoclass:: IVPController
   :members:
   :undoc-members:
```

### ``ErrorNormType`` — local error norm

Determines how the per-component error estimates are aggregated into a single
scalar norm that is compared against the tolerance target.

```{eval-rst}
.. autoclass:: ErrorNormType
   :members:
   :undoc-members:
```

## Events

### ``EventPack``

Specifies a single zero-crossing event to be located during integration. A
list of ``EventPack`` objects is passed to the ``integrate`` / ``integrate_dense``
family of methods; the integrator tracks each event independently and returns
the bracketed crossing locations.

```{eval-rst}
.. autoclass:: EventPack
   :members:
```

## Integrator

The ``Integrator`` class is constructed by calling ``ode.integrator(dt)`` (which
uses the default algorithm) — or ``ode.integrator(IVPAlg.DOPRI87, dt)`` to select
a method — on any ODE phase type (e.g. ``tychopy.optimal_control.ode_x``,
``tychopy.optimal_control.ode_6``, etc.). The resulting object provides
single-shot, dense, parallel, and state-transition-matrix integration of the
bound ODE.

The representative documentation below is drawn from
``tychopy.optimal_control.ode_x.integrator``; the identical interface is
available on every other ODE submodule's ``integrator`` class.

```{eval-rst}
.. currentmodule:: tychopy.optimal_control.ode_x

.. autoclass:: integrator
   :members:
```
