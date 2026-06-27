(tutorial-orbits-and-elements)=
# Orbits and elements

A spacecraft orbit can be described in several mathematically equivalent but
practically different ways. This tutorial works through the three representations
that Tycho provides — Cartesian, classical, and Modified Equinoctial Elements —
converts between them, verifies that the round-trips are exact, and propagates
a two-body orbit analytically. No optimal-control or phase machinery is needed;
everything here runs with `tychopy.astro` alone.

For the *why* behind each representation — singularities, optimizer
conditioning, when to prefer one over another — read
{doc}`Astrodynamics in Tycho </user_guide/astrodynamics>`. For the full API
listing see the {doc}`Python reference </reference/python/astrodynamics>`.

Every `{doctest}` block below (the ones showing `>>>` prompts) is executed as
part of Tycho's test suite, so the results shown are real. The C++ equivalents
appear alongside in tabs; those are illustrative `tycho::astro` fragments that
assume the headers and `using namespace` lines shown in the first tab (not run
here). Run the Python blocks in order in a REPL or script to follow along.

:::{note}
All results are displayed with `.round(n).tolist()` to keep the output
deterministic and readable. You do not need this in real code — operate on
the NumPy arrays directly.
:::

## Setup

```{doctest}
>>> import numpy as np
>>> from tychopy import astro
```

## 1. A Cartesian state

The most direct way to describe an orbit is a six-vector of position and
velocity in an inertial frame: $[r_x, r_y, r_z, v_x, v_y, v_z]$ in consistent
units (km and km/s here). The gravitational parameter for Earth is
$\mu = 398600.4418\ \text{km}^3/\text{s}^2$.

We start from the classical elements of a representative low-Earth orbit —
semi-major axis 6778 km (roughly 400 km altitude), eccentricity 0.01,
inclination 51.6°, RAAN 45°, argument of perigee 60°, mean anomaly 30° — and
convert to Cartesian with `classic_to_cartesian`. All angles are in radians.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> mu = 398600.4418   # km^3/s^2, Earth gravitational parameter
>>> classic = np.array([
...     6778.0,             # a  [km]   semi-major axis
...     0.01,               # e  [-]    eccentricity
...     np.radians(51.6),   # i  [rad]  inclination
...     np.radians(45.0),   # Ω  [rad]  RAAN
...     np.radians(60.0),   # ω  [rad]  argument of perigee
...     np.radians(30.0),   # M  [rad]  mean anomaly
... ])
>>> rv = astro.classic_to_cartesian(classic, mu)
>>> rv.round(6).tolist()
[-2999.19312, 2903.129203, 5265.737532, -5.452292, -5.48671, -0.030706]
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::astro;

double mu = 398600.4418;   // km^3/s^2, Earth gravitational parameter

// Classical elements [a, e, i, Omega, omega, M] — angles in radians.
Vector6<double> classic;
classic << 6778.0, 0.01, 51.6 * M_PI / 180.0,
           45.0 * M_PI / 180.0, 60.0 * M_PI / 180.0, 30.0 * M_PI / 180.0;

Vector6<double> rv = classic_to_cartesian(classic, mu);
// rv -> [-2999.19312, 2903.129203, 5265.737532, -5.452292, -5.48671, -0.030706]
```
:::
::::

The state is now a NumPy array of position (km) and velocity (km/s) in the
Earth-centered inertial (ECI) frame. The orbit is slightly eccentric and
inclined — neither circular nor equatorial — so all three element sets are
well-defined and the round-trips are clean.

## 2. Classical (Keplerian) elements

Classical elements $[a, e, i, \Omega, \omega, M]$ separate orbit shape and
orientation from the spacecraft's position along the orbit. `cartesian_to_classic`
inverts the conversion exactly: for a non-singular orbit the round-trip
recovers the original elements to floating-point precision.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> classic_rt = astro.cartesian_to_classic(rv, mu)
>>> classic_rt.round(6).tolist()
[6778.0, 0.01, 0.90059, 0.785398, 1.047198, 0.523599]
```
:::
:::{tab-item} C++
```cpp
Vector6<double> classic_rt = cartesian_to_classic(rv, mu);
// classic_rt -> [6778.0, 0.01, 0.90059, 0.785398, 1.047198, 0.523599]
```
:::
::::

These are the same values we started from (to six decimal places):

```{doctest}
>>> np.allclose(classic, classic_rt, atol=1e-6)
True
```

The sixth element is mean anomaly (not true anomaly). For hyperbolic orbits
`a` is negative; `e > 1`; and $M$ is the hyperbolic anomaly equivalent
$M = e\sinh H - H$. All conversion functions follow the same convention.

## 3. Modified Equinoctial Elements (MEE)

Classical elements become numerically ill-conditioned near circular orbits
($e \approx 0$) and near-equatorial orbits ($i \approx 0$). Modified
Equinoctial Elements $[p, f, g, h, k, L]$ replace the problematic angles
with smooth trigonometric components that go to zero gracefully at those
limits. The MEE set removes the circular and equatorial singularities entirely,
retaining only the retrograde-equatorial singularity at exactly $i = 180°$.

`cartesian_to_modified` converts the Cartesian state directly to MEE:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> mee = astro.cartesian_to_modified(rv, mu)
>>> mee.round(6).tolist()
[6777.3222, -0.002588, 0.009659, 0.341829, 0.341829, 2.366304]
```
:::
:::{tab-item} C++
```cpp
Vector6<double> mee = cartesian_to_modified(rv, mu);
// mee -> [6777.3222, -0.002588, 0.009659, 0.341829, 0.341829, 2.366304]
```
:::
::::

The six components are:

| Component | Value | Meaning |
| --------- | ----- | ------- |
| $p$ | 6777.322 km | Semi-latus rectum $p = a(1 - e^2)$ |
| $f$ | −0.002588 | Eccentricity cosine component $e\cos(\omega + \Omega)$ |
| $g$ | 0.009659 | Eccentricity sine component $e\sin(\omega + \Omega)$ |
| $h$ | 0.341829 | Inclination cosine component $\tan(i/2)\cos\Omega$ |
| $k$ | 0.341829 | Inclination sine component $\tan(i/2)\sin\Omega$ |
| $L$ | 2.366304 rad | True longitude $\omega + \Omega + \nu$ |

Note that $\sqrt{f^2 + g^2} = e = 0.01$ and
$2\arctan\!\sqrt{h^2 + k^2} = i = 51.6°$ — the classical values are
recoverable from the MEE components algebraically. At $e = 0$ the pair
$(f, g)$ goes to zero without either becoming undefined; at $i = 0$ the pair
$(h, k)$ does the same. This smooth behavior is what makes MEE the standard
choice for low-thrust trajectory optimization.

The inverse, `modified_to_cartesian`, recovers the Cartesian state exactly:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> rv_rt = astro.modified_to_cartesian(mee, mu)
>>> rv_rt.round(6).tolist()
[-2999.19312, 2903.129203, 5265.737532, -5.452292, -5.48671, -0.030706]
>>> np.allclose(rv, rv_rt, atol=1e-6)
True
```
:::
:::{tab-item} C++
```cpp
Vector6<double> rv_rt = modified_to_cartesian(mee, mu);
// rv_rt matches rv to floating-point precision.
```
:::
::::

## 4. Analytic propagation

For an unperturbed two-body orbit the trajectory from any initial state to any
later time can be computed exactly in closed form. Tycho's propagators use
the universal-variable formulation with Stumpff functions, which handles
elliptic, parabolic, and hyperbolic orbits in a single unified path.

`propagate_cartesian(RV, dt, mu)` advances the Cartesian state by `dt` seconds:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> dt = 1800.0   # 30 minutes
>>> rv_prop = astro.propagate_cartesian(rv, dt, mu)
>>> rv_prop.round(6).tolist()
[-2917.200988, -5671.568494, -2457.29995, 5.416957, -0.535366, -5.310345]
```
:::
:::{tab-item} C++
```cpp
double dt = 1800.0;   // 30 minutes
Vector6<double> rv_prop = propagate_cartesian(rv, dt, mu);
// rv_prop -> [-2917.200988, -5671.568494, -2457.29995, 5.416957, -0.535366, -5.310345]
```
:::
::::

After 30 minutes the spacecraft has moved roughly 5000 km along its orbit.
The orbital energy is conserved: the semi-major axis is unchanged (the orbit
shape is the same; only the position along it has advanced).

`propagate_modified` propagates MEE directly. Under pure two-body gravity the
five shape and orientation elements $[p, f, g, h, k]$ are constants of the
motion; only the true longitude $L$ changes. The propagator updates $L$
via the mean anomaly and the Kepler equation, then returns the new MEE state:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> mee_prop = astro.propagate_modified(mee, dt, mu)
>>> mee_prop.round(6).tolist()
[6777.3222, -0.002588, 0.009659, 0.341829, 0.341829, 4.403587]
```
:::
:::{tab-item} C++
```cpp
Vector6<double> mee_prop = propagate_modified(mee, dt, mu);
// mee_prop -> [6777.3222, -0.002588, 0.009659, 0.341829, 0.341829, 4.403587]
```
:::
::::

The first five elements are identical to the initial values — exactly the
expected behavior for an unperturbed orbit. Only $L$ has advanced. Converting
the propagated MEE back to Cartesian recovers the same result as
`propagate_cartesian`:

```{doctest}
>>> rv_from_mee = astro.modified_to_cartesian(mee_prop, mu)
>>> np.allclose(rv_prop, rv_from_mee, atol=1e-6)
True
```

## 5. Orbit figure

```{eval-rst}
.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from tychopy import astro

   mu = 398600.4418
   classic = np.array([
       6778.0,
       0.01,
       np.radians(51.6),
       np.radians(45.0),
       np.radians(60.0),
       0.0,
   ])
   rv0 = astro.classic_to_cartesian(classic, mu)
   T = 2 * np.pi * np.sqrt(6778.0**3 / mu)

   # Sample the orbit at 200 evenly spaced times over one period
   times = np.linspace(0.0, T, 200)
   states = np.array([astro.propagate_cartesian(rv0, t, mu) for t in times])

   fig = plt.figure(figsize=(6, 5))
   ax = fig.add_subplot(111, projection="3d")
   ax.plot(states[:, 0], states[:, 1], states[:, 2], lw=1.5, color="C0")
   ax.scatter([0], [0], [0], s=60, color="C3", zorder=5, label="Earth center")
   ax.scatter([rv0[0]], [rv0[1]], [rv0[2]], s=40, color="C1", zorder=5,
              label="Initial state")
   ax.set_xlabel("X [km]")
   ax.set_ylabel("Y [km]")
   ax.set_zlabel("Z [km]")
   ax.set_title("LEO orbit (one period)")
   ax.legend(fontsize=8)
   fig.tight_layout()
```

## What you learned

- A spacecraft orbit requires six degrees of freedom. Tycho supports three
  representations: **Cartesian** $[r_x, r_y, r_z, v_x, v_y, v_z]$, **classical
  elements** $[a, e, i, \Omega, \omega, M]$ (mean anomaly), and **MEE**
  $[p, f, g, h, k, L]$ (true longitude).
- `classic_to_cartesian`, `cartesian_to_classic`, `cartesian_to_modified`, and
  `modified_to_cartesian` convert between representations. All are exact and
  invertible for non-singular orbits.
- MEE removes the circular and equatorial singularities of classical elements,
  making it the preferred state for low-thrust trajectory optimization.
- `propagate_cartesian` and `propagate_modified` advance a state analytically
  under two-body gravity using the universal-variable Stumpff-function method.
  For an unperturbed orbit the five non-longitude MEE elements remain constant.

## Next steps

- {doc}`Low-thrust orbit transfer </user_guide/low_thrust_transfer>` —
  use MEE dynamics and PSIOPT to design a minimum-fuel orbit transfer.
- {doc}`Astrodynamics in Tycho </user_guide/astrodynamics>` — the conceptual
  model behind every component used above.
- {doc}`Python reference </reference/python/astrodynamics>` — the complete API,
  including `propagate_classic`, Lambert solvers, and the dynamics VectorFunctions.
