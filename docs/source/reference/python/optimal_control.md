(optimalcontrol-python)=
# OptimalControl

The Python API for Tycho's optimal-control subsystem, exposed through the
`tychopy.optimal_control` module. Every symbol below is a thin re-export of a
nanobind-bound C++ type or function (or a small pure-Python helper layered on
top of one); C++ users should consult the
{doc}`C++ reference </reference/cpp/optimal_control>` for the underlying types.

```{eval-rst}
.. currentmodule:: tychopy.optimal_control
```

The subsystem turns a continuous optimal-control problem into a
finite-dimensional nonlinear program by direct collocation: you describe the
dynamics as an {py:class}`~tychopy.optimal_control.ODEBase`, discretize them
into a {py:class}`~tychopy.optimal_control.PhaseInterface`, attach constraints
and objectives, and solve — optionally linking several phases together inside
an {py:class}`~tychopy.optimal_control.OptimalControlProblem`. For the
conceptual model see {doc}`Direct collocation in Tycho </explanation/direct_collocation>`; for a hands-on
introduction see the {doc}`Setting up a phase </tutorials/basics/your_first_phase>`.

## ODE definition

The dynamics of a phase are defined by subclassing
{py:class}`~tychopy.optimal_control.ODEBase`.
{py:class}`~tychopy.optimal_control.ODEArguments` is the symbolic argument
vector — time, state, control, and parameters — that a dynamics
VectorFunction is written against.

```{eval-rst}
.. autoclass:: ODEBase
   :members:

.. autoclass:: ODEArguments
   :members:
```

## Phase

A phase is the core optimal-control object: a single contiguous arc of the
trajectory together with its transcription, constraints, and objectives.

```{eval-rst}
.. autoclass:: PhaseInterface
   :members:
```

## Optimal-control problem

A container that links one or more phases (and shared link parameters) into a
single multi-phase problem solved by PSIOPT.

```{eval-rst}
.. autoclass:: OptimalControlProblem
   :members:
```

## Constraints & objectives

State functions attached to a phase region, plus the region selector that
names which discretized state(s) they act on.

```{eval-rst}
.. autoclass:: StateConstraint
   :members:

.. autoclass:: StateObjective
   :members:

.. autoclass:: PhaseRegionFlags
```

## Transcription configuration

Compile-time-style enums that select the collocation scheme, the control
representation, the integral quadrature rule, and the scaling source for a
phase.

```{eval-rst}
.. autoclass:: TranscriptionModes

.. autoclass:: ControlModes

.. autoclass:: IntegralModes

.. autoclass:: ScaleModes
```

## Mesh refinement

The per-iteration mesh-refinement diagnostics, the error estimator, and the
aggregation rule used to reduce per-interval errors to a scalar.

```{eval-rst}
.. autoclass:: MeshIterateInfo
   :members:

.. autoclass:: MeshErrorEstimators

.. autoclass:: MeshErrorAggregation
```

## Interpolation

Lookup-table and continuous-interpolation helpers for representing a
trajectory or control history as a callable VectorFunction.

```{eval-rst}
.. autoclass:: LGLInterpTable
   :members:

.. autoclass:: InterpFunction
   :members:
```
