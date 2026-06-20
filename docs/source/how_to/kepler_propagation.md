# How to propagate a Keplerian orbit analytically

For an unperturbed two-body orbit — no thrust, no drag, no higher-order
gravity — the state at any future time can be computed in closed form without
running a numerical integrator.  Tycho's analytic propagators handle this
case: they are orders of magnitude faster than numerical integration, free of
step-size sensitivity, and exact to floating-point precision.  They are the
natural choice for building initial guesses, setting terminal boundary
conditions, computing reference arcs, and seeding Lambert solvers.

This recipe assumes you have a starting state in one of the three element sets
— Cartesian, classical, or MEE — and a gravitational parameter `mu`.  For the
algorithm behind the propagators (universal-variable / Stumpff functions) see
{doc}`Astrodynamics in Tycho </explanation/astrodynamics>`.  For element-set
conversions see {doc}`How to convert between orbital element sets </how_to/element_conversions>`.

## Choose an element set

All three propagators round-trip through the same universal-variable kernel,
so they produce equivalent results.  The choice is driven by which element set
you are already working in:

| Function              | Input / output           | When to use |
| --------------------- | ------------------------ | ------------ |
| `propagate_cartesian` | Cartesian `[r, v]`       | When your state is in Cartesian coordinates |
| `propagate_classic`   | Classical `[a,e,i,Ω,ω,M]` | When you have classical elements already; faster (updates M only) |
| `propagate_modified`  | MEE `[p,f,g,h,k,L]`    | When your state is in MEE |

## Propagate a Cartesian state

```python
import numpy as np
from tychopy import astro

mu = 3.986004418e5   # km³/s² (Earth)

# Initial Cartesian state [rx, ry, rz, vx, vy, vz]
rv0 = np.array([-6534.7, 801.3, 3548.0,
                -1.208, -7.408,  0.658])

dt = 3600.0   # propagate forward one hour (s)
rv1 = astro.propagate_cartesian(rv0, dt, mu)
```

## Propagate classical elements

For an unperturbed two-body orbit, only the mean anomaly M changes with time;
all other elements are constants of the motion.  `propagate_classic` exploits
this by updating M directly ($M_\text{new} = M + n \cdot \Delta t$, where
$n = \sqrt{\mu / |a|^3}$) without a full Cartesian round-trip:

```python
# Classical elements [a, e, i, Omega, omega, M] — all angles in radians
classic0 = np.array([8000.0, 0.10,
                     np.radians(28.5), np.radians(90.0),
                     np.radians(30.0), np.radians(45.0)])

classic1 = astro.propagate_classic(classic0, dt, mu)
# Elements 0–4 unchanged; element 5 (M) advanced by n*dt.
```

**Classical singularities apply.** Near-circular ($e \approx 0$) or
near-equatorial ($i \approx 0$) orbits may have ill-conditioned ω or Ω in
the input; the propagator advances M correctly regardless, but converting the
result back to Cartesian near those singularities will be inaccurate.  Use MEE
for such orbits.

## Propagate MEE elements

`propagate_modified` internally converts MEE → classical, advances M, then
converts back.  Use it when your state is already in MEE to avoid the extra
conversion step:

```python
mee0 = astro.cartesian_to_modified(rv0, mu)
mee1 = astro.propagate_modified(mee0, dt, mu)
```

## Error conditions

All three propagators raise `ValueError` if `mu <= 0` or if the input state
is non-finite.  They raise `RuntimeError` if the universal-variable iteration
fails to converge — this typically occurs near parabolic trajectories ($e
\approx 1$) or for extremely long time steps on hyperbolic orbits that have
already passed periapsis by a large margin.  If you hit a `RuntimeError`,
verify that your orbit is physically realizable for the requested `dt` and
that `mu` is positive.

```python
# ValueError: mu must be positive
try:
    astro.propagate_cartesian(rv0, dt, mu=-1.0)
except ValueError as e:
    print(e)
```

## See also

- {doc}`Python reference </reference/python/astrodynamics>` — complete
  signatures for `propagate_cartesian`, `propagate_classic`, and
  `propagate_modified`, plus the `KeplerPropagator` VectorFunction for
  embedding Keplerian arcs in expression graphs.
- {doc}`Astrodynamics in Tycho </explanation/astrodynamics>` — the
  universal-variable algorithm and when analytic propagation is the right tool.
- {doc}`How to convert between orbital element sets </how_to/element_conversions>` —
  converting between Cartesian, classical, and MEE before and after
  propagation.
