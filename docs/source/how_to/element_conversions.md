# How to convert between orbital element sets

Spacecraft state representations are interchangeable in principle but not
equivalent in practice: classical (Keplerian) elements become numerically
ill-conditioned near circular or equatorial orbits, MEE eliminates those
singularities but is unfamiliar as a boundary-condition language, and
Cartesian coordinates are easy to specify but slow to converge on a coarse
collocation mesh. Knowing which conversion to call, and what to expect at the
edges of each element set's validity, prevents silent numerical failures during
problem setup and post-processing.

This recipe assumes you have a working Python environment with `tychopy`
installed and a Cartesian state or set of orbital elements in hand.  For the
conceptual picture of why element sets differ see
{doc}`Astrodynamics in Tycho </explanation/astrodynamics>`.

The code on this page consists of illustrative fragments — both the Python and
C++ snippets assume the surrounding setup shown in the first tab of each section
(the C++ tabs also assume the headers and `using namespace` lines) and are meant
to be read in order, not as standalone programs.

## Cartesian ↔ classical elements

`cartesian_to_classic` returns `[a, e, i, Ω, ω, M]` where the sixth element
is the **mean anomaly** M.  If you need the **true anomaly** ν instead, call
`cartesian_to_classic_true`:

::::{tab-set}
:::{tab-item} Python
```python
import numpy as np
from tychopy import astro

mu = 3.986004418e5   # km³/s² (Earth)

# An inclined, eccentric orbit
classic = np.array([8000.0,           # a, km
                    0.10,             # e
                    np.radians(28.5), # i
                    np.radians(90.0), # Omega
                    np.radians(30.0), # omega
                    np.radians(45.0)]) # M (mean anomaly), rad

rv = astro.classic_to_cartesian(classic, mu)

# Cartesian -> classical: mean anomaly convention
classic_back = astro.cartesian_to_classic(rv, mu)

# Cartesian -> classical: true anomaly convention
classic_true = astro.cartesian_to_classic_true(rv, mu)
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::astro;

double mu = 3.986004418e5;   // km^3/s^2 (Earth)

// An inclined, eccentric orbit [a, e, i, Omega, omega, M]
Vector6<double> classic;
classic << 8000.0, 0.10, 28.5 * M_PI / 180.0,
           90.0 * M_PI / 180.0, 30.0 * M_PI / 180.0, 45.0 * M_PI / 180.0;

Vector6<double> rv = classic_to_cartesian(classic, mu);

// Cartesian -> classical: mean anomaly convention
Vector6<double> classic_back = cartesian_to_classic(rv, mu);

// Cartesian -> classical: true anomaly convention
Vector6<double> classic_true = cartesian_to_classic_true(rv, mu);
```
:::
::::

**Singularity warning.** Near circular orbits ($e \approx 0$) ω becomes
undefined; near equatorial orbits ($i \approx 0$ or $i \approx \pi$) Ω
becomes undefined.  Both return `NaN` in those slots rather than raising an
exception.  If your orbit passes through near-circular or near-equatorial
conditions, use MEE instead.

## Cartesian ↔ MEE

`cartesian_to_modified` returns `[p, f, g, h, k, L]` where `L` is the **true
longitude** (not mean anomaly).  The conversion is a direct closed-form
algebraic map — no Kepler-equation iteration:

::::{tab-set}
:::{tab-item} Python
```python
mee = astro.cartesian_to_modified(rv, mu)

# Invert back to Cartesian
rv_back = astro.modified_to_cartesian(mee, mu)
```
:::
:::{tab-item} C++
```cpp
Vector6<double> mee = cartesian_to_modified(rv, mu);

// Invert back to Cartesian
Vector6<double> rv_back = modified_to_cartesian(mee, mu);
```
:::
::::

MEE is well-defined for all orbits except exactly retrograde equatorial
($i = 180°$), where $\tan(i/2) \to \infty$ and the $h$, $k$ components
diverge.  Non-equatorial retrograde orbits and all prograde orbits are fully
supported.

## Classical ↔ MEE

`classic_to_modified` converts `[a, e, i, Ω, ω, M]` directly to MEE; `mu`
is accepted for API symmetry but is not used in the algebraic mapping:

::::{tab-set}
:::{tab-item} Python
```python
mee = astro.classic_to_modified(classic, mu)

# Recover classical elements (mean anomaly)
classic_back = astro.modified_to_classic(mee, mu)
```
:::
:::{tab-item} C++
```cpp
Vector6<double> mee = classic_to_modified(classic, mu);

// Recover classical elements (mean anomaly)
Vector6<double> classic_back = modified_to_classic(mee, mu);
```
:::
::::

Because `classic_to_modified` first solves Kepler's equation to extract the
true anomaly, it inherits the classical-element singularities: passing a
nearly circular orbit (e ≈ 0) may produce NaN in the MEE output even though
MEE itself has no such singularity.  When in doubt, convert Cartesian → MEE
directly via `cartesian_to_modified`.

## Quick reference

| From → To             | Function                    | Sixth element |
| ---------------------- | --------------------------- | ------------- |
| Cartesian → classical  | `cartesian_to_classic`      | mean anomaly M |
| Cartesian → classical  | `cartesian_to_classic_true` | true anomaly ν |
| Classical → Cartesian  | `classic_to_cartesian`      | input: M |
| Cartesian → MEE        | `cartesian_to_modified`     | true longitude L |
| MEE → Cartesian        | `modified_to_cartesian`     | — |
| Classical → MEE        | `classic_to_modified`       | input: M |
| MEE → Classical        | `modified_to_classic`       | output: M |

All functions accept a NumPy array of shape `(6,)` and return the same shape.
Each also accepts a `VectorFunction` as its first argument (instead of a
numeric array), in which case it returns a composed `VectorFunction` suitable
for embedding inside an optimal-control expression graph — see the
`CartesianToClassic`, `ModifiedToCartesian`, and `CartesianToMEE` VectorFunction
classes in the reference.

## See also

- {doc}`Python reference </reference/python/astrodynamics>` — complete
  signatures for all conversion functions and the VectorFunction conversion
  classes.
- {doc}`Astrodynamics in Tycho </explanation/astrodynamics>` — the conceptual
  picture of each element set's singularity structure and when to prefer each.
- {doc}`How to use astrodynamics dynamics in a phase </how_to/astro_dynamics_in_a_phase>` —
  using MEE as the ODE representation in an optimal-control phase.
