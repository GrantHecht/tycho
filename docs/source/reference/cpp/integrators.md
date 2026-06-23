(integrators-cpp)=
# Integrators

The C++ API for Tycho's integrators subsystem, declared in the
``tycho::integrators`` namespace and exposed through the
``include/tycho/integrators.h`` umbrella header. Python users should consult
the {doc}`Python reference </reference/python/integrators>`.

The subsystem implements a family of embedded Runge-Kutta adaptive-step
integrators. The design is built around a layered CRTP hierarchy:
``Stepper`` encapsulates a single adaptive step; ``Integrator`` (templated on
an ODE type) composes a stepper with a step-size controller and exposes the
full integration interface including event detection; the higher-level
``AdaptiveDriver``, ``ParallelDriver``, and ``STMDriver`` utilities build
multi-trajectory or sensitivity-matrix workflows on top.

## Enumerations

### ``IVPAlg`` — Runge-Kutta algorithm selector

```{eval-rst}
.. doxygenenum:: tycho::integrators::IVPAlg
   :project: tycho
```

### ``IVPController`` — step-size controller selector

```{eval-rst}
.. doxygenenum:: tycho::integrators::IVPController
   :project: tycho
```

### ``ErrorNormType`` — local error norm

```{eval-rst}
.. doxygenenum:: tycho::integrators::ErrorNormType
   :project: tycho
```

## Step-size controllers

The controller structs implement the control law that maps an estimated local
error ratio to the next step-size multiplier. They are value types: callers
may construct and own them directly, or let ``Integrator`` manage one
internally via ``set_controller()``.

```{eval-rst}
.. doxygenstruct:: tycho::integrators::ControllerOutput
   :project: tycho
   :members:

.. doxygenstruct:: tycho::integrators::IController
   :project: tycho
   :members:

.. doxygenstruct:: tycho::integrators::PIController
   :project: tycho
   :members:

.. doxygenstruct:: tycho::integrators::PIDController
   :project: tycho
   :members:
```

## Error norm helpers

Free functions for computing the scalar error norm that the controller
compares against the tolerance target.

```{eval-rst}
.. doxygenfunction:: tycho::integrators::error_norm
   :project: tycho

.. doxygenfunction:: tycho::integrators::scaled_residuals
   :project: tycho
```

## Events

``EventHandler`` is the user-facing type for specifying a single
zero-crossing event; ``EventPack`` is the scalar VectorFunction wrapper that
the integrator evaluates at each step.

```{eval-rst}
.. doxygenstruct:: tycho::integrators::EventHandler
   :project: tycho
   :members:

.. doxygenclass:: tycho::integrators::EventPack
   :project: tycho
   :members:
```

## Integrator

The central type: a CRTP struct templated on an ODE type that composes an
adaptive stepper with a step-size controller and exposes single-shot,
dense-output, parallel, and event-aware integration.

```{eval-rst}
.. doxygenstruct:: tycho::integrators::Integrator
   :project: tycho
```

## Higher-level drivers

### ``AdaptiveDriver``

Integrates a batch of initial conditions over a shared time grid, optionally
with a vectorised (SuperScalar) inner loop.

```{eval-rst}
.. doxygenclass:: tycho::integrators::AdaptiveDriver
   :project: tycho
   :members:
```

### ``ParallelDriver``

Runs multiple independent integrations in parallel using the thread pool.

```{eval-rst}
.. doxygenclass:: tycho::integrators::ParallelDriver
   :project: tycho
   :members:
```

### ``STMDriver``

Computes state-transition matrices (Jacobians of integrated trajectories) via
finite differences over the stepper output.

```{eval-rst}
.. doxygenstruct:: tycho::integrators::STMDriver
   :project: tycho
   :members:
```
