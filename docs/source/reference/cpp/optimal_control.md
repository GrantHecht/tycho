(optimalcontrol-cpp)=
# OptimalControl

The C++ API for Tycho's optimal-control subsystem, declared in the
`tycho::oc` namespace (with the high-level builder API in `tycho`) and exposed
through the `include/tycho/optimal_control.h` umbrella header. Python users
should consult the {doc}`Python reference </reference/python/optimal_control>`.

The subsystem converts a continuous optimal-control problem into a
finite-dimensional nonlinear program by *transcription*: dynamics are an
{cpp:struct}`tycho::oc::ODEBase`, discretized into an
{cpp:struct}`tycho::oc::ODEPhase`, with constraints and objectives attached and
several phases optionally linked inside an
{cpp:class}`tycho::OptimalControlProblem`. The primary transcription method is
direct collocation (the LGL and Trapezoidal schemes), but central shooting
(`CentralShooting`) is also supported. For the conceptual model see
{doc}`Direct collocation in Tycho </user_guide/phases_and_collocation>`.

## Core ODE types

The dynamics interface a phase is built from.
{cpp:struct}`tycho::oc::ODEArguments` is the symbolic argument vector (time,
state, control, parameters) a dynamics VectorFunction is written against;
{cpp:struct}`tycho::oc::ODEBase` and {cpp:struct}`tycho::oc::StaticODE` are the
runtime- and compile-time-sized ODE wrappers.

```{eval-rst}
.. doxygenstruct:: tycho::oc::ODEArguments
   :project: tycho
   :members:

.. doxygenstruct:: tycho::oc::ODEBase
   :project: tycho

.. doxygenstruct:: tycho::oc::StaticODE
   :project: tycho
```

## Phase and problem

The phase and the multi-phase optimal-control problem container,
with their runtime base classes.

```{eval-rst}
.. doxygenstruct:: tycho::oc::ODEPhaseBase
   :project: tycho
   :members:

.. doxygenstruct:: tycho::oc::ODEPhase
   :project: tycho
   :members:

.. doxygenstruct:: tycho::oc::OptimalControlProblemBase
   :project: tycho
   :members:
```

## Transcription and interpolation

The Legendre-Gauss-Lobatto interpolation table used to represent a
trajectory, and the one- to four-dimensional lookup-table functions.

```{eval-rst}
.. doxygenstruct:: tycho::oc::LGLInterpTable
   :project: tycho
   :members:

.. doxygenstruct:: tycho::oc::InterpTable1D
   :project: tycho
.. doxygenstruct:: tycho::oc::InterpTable2D
   :project: tycho
.. doxygenstruct:: tycho::oc::InterpTable3D
   :project: tycho
.. doxygenstruct:: tycho::oc::InterpTable4D
   :project: tycho
```

## Builder API

The high-level, fluent builder types (in namespace `tycho`, not `tycho::oc`)
that most C++ users construct problems through:
{cpp:class}`tycho::ODEBuilder` assembles a dynamics model,
{cpp:class}`tycho::Phase` wraps a single phase, and
{cpp:class}`tycho::OptimalControlProblem` links phases into a multi-phase
problem.

```{eval-rst}
.. doxygenclass:: tycho::ODEBuilder
   :project: tycho

.. doxygenclass:: tycho::Phase
   :project: tycho

.. doxygenclass:: tycho::OptimalControlProblem
   :project: tycho
```

## Enums

The flag enumerators selecting phase regions, multi-phase links, control and
transcription schemes, integral quadrature, scaling, and mesh-error
estimation. (Authored directly here because the enumerators are grouped under
the `optimal_control` Doxygen module, which Breathe's ``doxygenenum`` cannot
resolve by qualified name; their definitions live in
`detail/optimal_control/core/optimal_control_flags.h`.)

```{eval-rst}
.. cpp:enum-class:: tycho::oc::PhaseRegionFlags

   Region of a phase that a state function (constraint or objective) acts on.

   .. cpp:enumerator:: NotSet

      Sentinel: region not yet assigned.

   .. cpp:enumerator:: Front

      The first (initial-time) node of the phase.

   .. cpp:enumerator:: Back

      The last (final-time) node of the phase.

   .. cpp:enumerator:: FrontandBack

      Both endpoints, front state followed by back state.

   .. cpp:enumerator:: BackandFront

      Both endpoints, back state followed by front state.

   .. cpp:enumerator:: Path

      Every node of the phase (full path constraint).

   .. cpp:enumerator:: InnerPath

      Every interior node, excluding the two endpoints.

   .. cpp:enumerator:: NodalPath

      Every nodal (mesh) point along the path.

   .. cpp:enumerator:: DefectPath

      Every defect (collocation) point along the path.

   .. cpp:enumerator:: PairWisePath

      Each consecutive pair of nodes along the path.

   .. cpp:enumerator:: DefectPairWisePath

      Per adjacent defect-interval pair: all cardinal states spanning the two
      overlapping intervals.

   .. cpp:enumerator:: FrontNodalBackPath

      Per interior nodal point: the phase's first state, that nodal point, and
      the phase's last state (one application per interior node).

   .. cpp:enumerator:: Params

      The combined parameter vector of the phase.

   .. cpp:enumerator:: ODEParams

      The ODE (dynamics) parameter sub-vector.

   .. cpp:enumerator:: StaticParams

      The static (non-ODE) parameter sub-vector.

   .. cpp:enumerator:: Accumulate

      Internal sentinel — not a user-facing region selector.

   .. cpp:enumerator:: BlockDefectPath

      Per defect interval: all cardinal states plus the block-constant control
      value (BlockConstant control mode).

.. cpp:enum-class:: tycho::oc::LinkFlags

   Pairing of phase regions joined by a link (multi-phase) constraint.

   .. cpp:enumerator:: BackToFront

      Back of the first phase to front of the second.

   .. cpp:enumerator:: FrontToBack

      Front of the first phase to back of the second.

   .. cpp:enumerator:: FrontToFront

      Front of the first phase to front of the second.

   .. cpp:enumerator:: BackToBack

      Back of the first phase to back of the second.

   .. cpp:enumerator:: ParamsToParams

      Parameter vector of one phase to that of another.

   .. cpp:enumerator:: LinkParams

      The shared link-parameter vector.

   .. cpp:enumerator:: BackTwoToTwoFront

      Last two states of one phase to first two of another; not yet implemented.

   .. cpp:enumerator:: FrontTwoToTwoBack

      First two states of one phase to last two of another; not yet implemented.

   .. cpp:enumerator:: PathToPath

      Path nodes of one phase to path nodes of another.

   .. cpp:enumerator:: ReadRegions

      Regions are read from an explicit specification.

.. cpp:enum-class:: tycho::oc::ControlModes

   Interpolation/representation scheme for the control history.

   .. cpp:enumerator:: HighestOrderSpline

      Spline matching the transcription's highest order.

   .. cpp:enumerator:: FirstOrderSpline

      Piecewise-linear (first-order) control spline.

   .. cpp:enumerator:: NoSpline

      Independent control values, no inter-node spline.

   .. cpp:enumerator:: BlockConstant

      Control held constant over each transcription block.

.. cpp:enum-class:: tycho::oc::TranscriptionModes

   Collocation/transcription scheme used to discretize a phase.

   .. cpp:enumerator:: LGL3

      3rd-order Legendre-Gauss-Lobatto collocation.

   .. cpp:enumerator:: LGL5

      5th-order Legendre-Gauss-Lobatto collocation.

   .. cpp:enumerator:: LGL7

      7th-order Legendre-Gauss-Lobatto collocation.

   .. cpp:enumerator:: Trapezoidal

      Trapezoidal (2nd-order) collocation.

   .. cpp:enumerator:: CentralShooting

      Central (forward/backward) shooting transcription.

.. cpp:enum-class:: tycho::oc::IntegralModes

   Quadrature rule used to evaluate integral terms over a phase.

   .. cpp:enumerator:: BaseIntegral

      Default quadrature matched to the transcription order.

   .. cpp:enumerator:: SimpsonIntegral

      Simpson's-rule quadrature (unimplemented placeholder; not currently
      dispatched).

   .. cpp:enumerator:: TrapIntegral

      Trapezoidal-rule quadrature.

.. cpp:enum-class:: tycho::oc::ScaleModes

   Source of the variable/equation scaling applied to a phase.

   .. cpp:enumerator:: AUTO

      Scales are computed automatically from the problem.

   .. cpp:enumerator:: CUSTOM

      Scales are supplied explicitly by the user.

   .. cpp:enumerator:: NONE

      No scaling is applied (unit scales).

.. cpp:enum-class:: tycho::oc::MeshErrorEstimators

   Method used to estimate the per-interval mesh (discretization) error.

   .. cpp:enumerator:: DEBOOR

      de Boor collocation-residual error estimate.

   .. cpp:enumerator:: INTEGRATOR

      Error estimate from a reference numerical integrator.

.. cpp:enum-class:: tycho::oc::MeshErrorAggregation

   Reduction used to aggregate per-interval mesh errors into a scalar.

   .. cpp:enumerator:: MAX

      Maximum error over all intervals.

   .. cpp:enumerator:: AVG

      Arithmetic mean of per-interval errors.

   .. cpp:enumerator:: GEOMETRIC

      Geometric mean of per-interval errors.

   .. cpp:enumerator:: ENDTOEND

      End-to-end re-integration trajectory error.
```
