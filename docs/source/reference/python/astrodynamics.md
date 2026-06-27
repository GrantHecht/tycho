(astrodynamics-python)=
# Astrodynamics

The Python API for Tycho's astrodynamics subsystem, exposed through the
`tychopy.astro` module. The subsystem covers three element sets — Cartesian,
classical (Keplerian), and Modified Equinoctial Elements (MEE) — and provides
both numeric conversion/propagation routines (operating on ``numpy`` arrays)
and differentiable {py:class}`~tychopy.vector_functions.VectorFunction`
expressions (for embedding inside optimal-control problem graphs). Dynamics
models ({py:func}`~tychopy.astro.cartesian_dynamics`,
{py:func}`~tychopy.astro.modified_dynamics`,
{py:func}`~tychopy.astro.crtbp_dynamics`) return VectorFunctions suitable for
use as the ODE of a {py:class}`~tychopy.optimal_control.PhaseInterface`.

For the conceptual model — state representations, propagation, and the dynamics
models as phase ODEs — see {doc}`Astrodynamics in Tycho </user_guide/astrodynamics>`.

For hands-on examples see the {doc}`User Guide </user_guide/index>`.

```{eval-rst}
.. currentmodule:: tychopy.astro
```

## Conversions

Numeric conversion routines that operate on ``numpy`` arrays (shape ``(6,)``).
Each function also accepts a {py:class}`~tychopy.vector_functions.VectorFunction`
as its first argument (via an overloaded dispatch); in that case it returns a
VectorFunction expression — use the dedicated VF conversion classes
({py:class}`CartesianToClassic`, {py:class}`ModifiedToCartesian`,
{py:class}`CartesianToMEE`) for embedding conversions inside expression graphs.

```{eval-rst}
.. autofunction:: cartesian_to_classic

.. autofunction:: cartesian_to_classic_true

.. autofunction:: cartesian_to_modified

.. autofunction:: classic_to_cartesian

.. autofunction:: classic_to_modified

.. autofunction:: modified_to_cartesian

.. autofunction:: modified_to_classic
```

## Propagation

Analytic (two-body) propagators that advance a state vector by a given
time-of-flight.  All three routines round-trip through the same universal-variable
LCD kernel under the hood.

```{eval-rst}
.. autofunction:: propagate_cartesian

.. autofunction:: propagate_classic

.. autofunction:: propagate_modified
```

## Lambert

Boundary-value solvers for two-body Lambert's problem via Izzo's algorithm.

```{eval-rst}
.. autofunction:: lambert_izzo

.. autofunction:: lambert_izzo_multirev
```

## Dynamics models

Factory functions that return a differentiable
{py:class}`~tychopy.vector_functions.VectorFunction` encoding the equations of
motion for a phase.  The returned VectorFunction can be passed directly to
{py:class}`~tychopy.optimal_control.PhaseInterface` as the ODE.

```{eval-rst}
.. autofunction:: cartesian_dynamics

.. autofunction:: modified_dynamics

.. autofunction:: crtbp_dynamics

.. autofunction:: j2_cartesian

.. autofunction:: non_ideal_solar_sail
```

## VectorFunction conversions

Differentiable {py:class}`~tychopy.vector_functions.VectorFunction` types for
embedding element-set conversions inside optimal-control expression graphs.
Unlike the numeric routines above, these propagate analytic Jacobians and
Hessians through the conversion.

```{eval-rst}
.. autoclass:: CartesianToClassic
   :members:

.. autoclass:: ModifiedToCartesian
   :members:

.. autoclass:: CartesianToMEE
   :members:
```

## Kepler propagator

The `tychopy.astro.kepler` sub-module (aliased as ``tychopy.astro.Kepler``)
provides the universal-variable Kepler propagator as a VectorFunction, plus
a specialised ODE and phase type for Keplerian two-body problems.

```{eval-rst}
.. currentmodule:: tychopy.astro.kepler

.. autoclass:: KeplerPropagator
   :no-members:

.. autoclass:: ode
   :no-members:

.. autoclass:: phase
   :no-members:
```
