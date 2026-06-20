# How to solve Lambert's problem

Given two position vectors and a time of flight, Lambert's problem asks for
the velocity at the departure point that puts a spacecraft on the two-body arc
connecting them.  This is the fundamental computation behind pork-chop plots,
impulsive $\Delta v$ budgets, and initial-guess generation for low-thrust
trajectories.  Tycho implements Izzo's algorithm, which is robust, fast, and
handles both single-revolution and multi-revolution transfers.

This recipe assumes you have departure and arrival position vectors in the same
coordinate frame and units as `mu`.  For the conceptual background on Lambert's
problem and the long-way vs. short-way distinction see
{doc}`Astrodynamics in Tycho </explanation/astrodynamics>`.

## Single-revolution transfer

`lambert_izzo` returns a tuple `(V1, V2)` of NumPy arrays of shape `(3,)` —
the departure and arrival velocity vectors:

```python
import numpy as np
from tychopy import astro

mu = 3.986004418e5   # km³/s² (Earth)

R1 = np.array([6778.0,  0.0, 0.0])    # departure position, km
R2 = np.array([0.0, 10000.0, 0.0])    # arrival position, km
dt = 3823.4                            # time of flight, s

# Short-way transfer (transfer angle < 180°)
V1_sw, V2_sw = astro.lambert_izzo(R1, R2, dt, mu, longway=False)

# Long-way transfer (transfer angle > 180°)
V1_lw, V2_lw = astro.lambert_izzo(R1, R2, dt, mu, longway=True)
```

The `longway` flag selects between the two single-revolution solutions.
`False` is the short-way transfer (traversing the shorter arc between the
radius vectors); `True` is the long-way transfer (going the other direction,
transfer angle > 180°).  For a given `R1`, `R2`, and `dt` there is exactly
one solution in each sense.

## Multi-revolution transfer

When the time of flight is long enough for the spacecraft to complete one or
more full revolutions before arriving, additional solution families exist.
`lambert_izzo_multirev` (equivalently, the 7-argument overload of
`lambert_izzo`) adds `Nrevs` (the number of complete revolutions) and
`rightbranch` (which of the two branches for that revolution count):

```python
dt_multirev = 4 * np.pi * np.sqrt(((6778.0 + 10000.0) / 2)**3 / mu)

V1_mr, V2_mr = astro.lambert_izzo_multirev(
    R1, R2, dt_multirev, mu,
    longway=False,
    Nrevs=1,
    rightbranch=False,
)
```

For a given `Nrevs ≥ 1` there are in general two solution branches (two
distinct orbits that complete exactly `Nrevs` revolutions in the requested
time), selected by `rightbranch=True` or `rightbranch=False`.  For
`Nrevs = 0` `rightbranch` is ignored and the result is identical to the
single-revolution overload.

## Using the result

The returned `V1` and `V2` are inertial velocity vectors at the departure and
arrival points respectively.  The $\Delta v$ of the maneuver at departure is
`V1 - v_sc` where `v_sc` is the spacecraft's pre-maneuver velocity:

```python
# Departure delta-v (km/s)
v_sc_departure = np.array([0.0, 7.669, 0.0])   # circular LEO velocity
dv1 = np.linalg.norm(V1_sw - v_sc_departure)
```

To use the Lambert arc as an initial guess for an optimal-control phase,
construct a piecewise-linear or analytically propagated trajectory from `R1`
to `R2` using the departure velocity `V1`, then pass that trajectory to
`ode.phase(...)` as the initial guess.

## Non-convergence contract

The scalar overloads raise `RuntimeError` if Izzo's iteration does not
converge within 20 iterations.  This can happen when the problem is
ill-conditioned — for example, when `R1` and `R2` are nearly collinear (the
transfer angle is very close to 0° or 180°), or when `dt` is shorter than the
minimum possible flight time for the requested geometry.  In those cases adjust
the geometry or time of flight.

The vectorized batch overload (`lambert_izzo` with matrix arguments) does
**not** raise on non-convergence; instead it returns a per-problem integer exit
code (0 = converged, 1 = not converged) that you should inspect.

## See also

- {doc}`Python reference </reference/python/astrodynamics>` — complete
  signatures for `lambert_izzo` and `lambert_izzo_multirev`, including the
  batch overload for solving many problems simultaneously.
- {doc}`Astrodynamics in Tycho </explanation/astrodynamics>` — the conceptual
  background on Lambert's problem, the long-way vs. short-way distinction, and
  multi-revolution solutions.
- {doc}`How to propagate a Keplerian orbit analytically </how_to/kepler_propagation>` —
  analytic propagation as a complement to the Lambert solver for building
  initial guesses.
