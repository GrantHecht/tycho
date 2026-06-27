(explanation-astrodynamics)=
# Astrodynamics in Tycho

Tycho's astrodynamics subsystem gives you three orbital state representations,
analytic two-body propagation, Lambert's problem, and a menu of dynamics models
that drop straight in as the ODE of an optimal-control
{doc}`Phase </user_guide/phases_and_collocation>`. This page is practical
orientation: what each piece is for and how to choose between them. For the full
API surface see the {doc}`Python reference </reference/python/astrodynamics>` and
the {doc}`C++ reference </reference/cpp/astrodynamics>`; for runnable walk-throughs
see the astro how-to guides and tutorials.

## State representations

A spacecraft state in two-body mechanics has six degrees of freedom. Tycho
provides three coordinate sets for that state. They are mathematically
equivalent — what differs is their *conditioning*: where each one becomes
ill-conditioned or singular, which decides how well the optimizer can traverse a given
trajectory.

### Cartesian elements

The Cartesian state $[r_x, r_y, r_z, v_x, v_y, v_z]$ — position and velocity in an
inertial frame — is the most direct. There are no angle conventions and no
singularities (beyond the physical $r \neq 0$): every point in $\mathbb{R}^6$ is a
valid state.

Cartesian is the natural output of integrators, the natural input to force models
(gravity, thrust, drag all depend on $\mathbf{r}$ and $\mathbf{v}$ directly), and
the natural language for physical boundary conditions. Its weakness for
trajectory optimization is *scale*, not accuracy: a low-thrust transfer that takes
weeks has slowly evolving orbital geometry but rapidly oscillating Cartesian
coordinates, which a coarse collocation mesh struggles to represent. Such problems
are better conditioned in an element set.

### Classical (Keplerian) elements

The classical elements $[a, e, i, \Omega, \omega, M]$ describe an orbit's shape,
orientation, and position:

| Symbol | Meaning |
| --- | --- |
| $a$ | Semi-major axis |
| $e$ | Eccentricity |
| $i$ | Inclination |
| $\Omega$ | Right ascension of the ascending node (RAAN) |
| $\omega$ | Argument of perigee |
| $M$ | Mean anomaly |

Tycho uses **mean anomaly** as the sixth element (not true anomaly); the
`cartesian_to_classic` / `classic_to_cartesian` conversions and the
`CartesianToClassic` VectorFunction follow this convention.

The appeal of classical elements is that they separate slowly changing orbit
geometry from fast angular position — for an unperturbed orbit only $M$ evolves.
But they carry two well-known singularities: at **circular orbits** ($e = 0$) the
argument of perigee $\omega$ is undefined, and at **equatorial orbits** ($i = 0$
or $\pi$) the RAAN $\Omega$ is undefined. Near those limits — common in practice,
e.g. station-keeping near geostationary orbit — the elements are ill-conditioned:
small physical changes produce large swings in $\omega$ or $\Omega$, which hurts
both propagation accuracy and optimizer convergence.

### Modified Equinoctial Elements (MEE)

Modified Equinoctial Elements $[p, f, g, h, k, L]$ were designed specifically to
remove those singularities:

| Symbol | Meaning |
| --- | --- |
| $p$ | Semi-latus rectum, $p = a(1 - e^2)$ |
| $f$ | Eccentricity component, $f = e\cos(\omega + \Omega)$ |
| $g$ | Eccentricity component, $g = e\sin(\omega + \Omega)$ |
| $h$ | Inclination component, $h = \tan(i/2)\cos\Omega$ |
| $k$ | Inclination component, $k = \tan(i/2)\sin\Omega$ |
| $L$ | True longitude, $L = \omega + \Omega + \nu$ (true anomaly $\nu$) |

At $e = 0$ the components $f$ and $g$ go smoothly to zero; at $i = 0$ the
components $h$ and $k$ go smoothly to zero — no angle becomes undefined. The only
remaining singularity is at exactly $i = 180°$ (a retrograde equatorial orbit);
near-retrograde non-equatorial orbits are fully supported. Removing the
near-circular and near-equatorial singularities is why MEE is the preferred set
for low-thrust optimization: the optimizer can pass through zero eccentricity or
graze equatorial inclination without pathological Jacobians.

### Choosing a representation

- **Cartesian** — force modeling; high-thrust, impulsive, or short arcs; and the
  CR3BP (naturally Cartesian in the synodic frame).
- **Classical** — when the orbit is well away from circular and equatorial and you
  want boundary conditions expressed in orbital geometry.
- **MEE** — low-thrust transfers, especially those that cross or approach zero
  eccentricity or inclination. `MEEDynamics` is the standard choice for
  orbit-raising and orbit-transfer problems.

Tycho provides conversion VectorFunctions — `CartesianToClassic`,
`CartesianToMEE`, `ModifiedToCartesian` — so a trajectory optimized in one
representation can express boundary conditions in another. They are differentiable
with analytic Jacobians, so they compose into boundary-value constraints without
breaking the optimizer's derivative chain. For worked conversions see
{doc}`How to convert between element sets </user_guide/how_to/element_conversions>`.

## Propagation

### Analytic two-body propagation

For an unperturbed two-body orbit — no thrust, no higher-order gravity, no drag —
the trajectory from an initial state to any later time is computed exactly in
closed form. Tycho exposes this through `propagate_cartesian`,
`propagate_classic`, and `propagate_modified`, which share a single
**universal-variable** algorithm (Stumpff functions) that stays well-behaved
across elliptic, parabolic, and hyperbolic regimes without separate cases. The
`KeplerPropagator` VectorFunction takes an initial state and a time-of-flight and
returns the propagated state with exact analytic Jacobian and Hessian.

Analytic propagation is useful in three contexts:

1. **Initial-guess generation** — a Keplerian arc is a good first approximation
   for a coast and is far cheaper than a numerical integration.
2. **Boundary-condition setup** — the exact terminal state of a Keplerian arc can
   set tight bounds or equality constraints.
3. **Pure-Keplerian phase** — `tychopy.astro.kepler.phase` wraps a
   `CentralShooting` phase around `KeplerPropagator`, letting a Keplerian arc
   participate in a multi-phase problem with full sensitivities.

See {doc}`How to propagate a Kepler orbit </user_guide/how_to/kepler_propagation>`.

### Lambert's problem

Lambert's problem is the two-point boundary-value problem: given two position
vectors $\mathbf{r}_1$, $\mathbf{r}_2$ and a time of flight $\Delta t$, find the
velocity $\mathbf{v}_1$ such that the spacecraft travels from $\mathbf{r}_1$ to
$\mathbf{r}_2$ under two-body gravity in time $\Delta t$. Tycho implements Izzo's
algorithm (`lambert_izzo`, `lambert_izzo_multirev`), which is robust and fast.

Two flags select among the multiple solutions that exist for a given geometry: a
`longway` boolean picks the short-way or long-way arc, and for long times of
flight `lambert_izzo_multirev` accepts a revolution count `Nrevs` and a
`rightbranch` flag to select the multi-revolution branch.

Lambert's problem is the enabling computation for **pork-chop plots** and
impulsive maneuver design, and a useful initial-guess generator for low-thrust
problems. See {doc}`How to solve a Lambert transfer </user_guide/how_to/lambert_transfer>`.

## Dynamics models

A dynamics model is a **VectorFunction** mapping the packed input
$[\text{state}(6), \mathbf{a}(3)]$ (IR = 9) to the state derivative $\dot{x}$
(OR = 6). The trailing three inputs are the *non-two-body acceleration* — every
acceleration beyond central-body point-mass gravity. In an optimal-control
problem this is usually the thrust control the optimizer chooses, but it can
equally carry modeled perturbations (J2, drag, SRP) or their sum. This is exactly
the ODE form a {doc}`Phase </user_guide/phases_and_collocation>` expects, and each
model carries analytic Jacobian and Hessian implementations so PSIOPT receives
exact derivatives.

### CartesianDynamics

`CartesianDynamics` is the two-body equations of motion in Cartesian coordinates,
with input $[r_x, r_y, r_z, v_x, v_y, v_z, a_x, a_y, a_z]$:

$$
\begin{pmatrix} \dot{\mathbf{r}} \\ \dot{\mathbf{v}} \end{pmatrix}
=
\begin{pmatrix} \mathbf{v} \\ -\dfrac{\mu}{r^3}\mathbf{r} + \mathbf{a} \end{pmatrix}
$$

The non-two-body acceleration $\mathbf{a}$ enters directly in the inertial frame —
the thrust control in a low-thrust problem, plus any state-dependent
perturbations. Use it where the state is naturally Cartesian: high-thrust or
impulsive transfers, short-arc maneuvers, or CR3BP problems.

### MEEDynamics

`MEEDynamics` is the two-body equations of motion in Modified Equinoctial
Elements (the Gauss variation-of-parameters form), with the non-two-body
acceleration given in the **RSW frame** — input $[p, f, g, h, k, L, u_r, u_t,
u_n]$:

| Component | Direction |
| --- | --- |
| $u_r$ | Radial: along the position vector $\hat{\mathbf{r}}$ |
| $u_t$ | Along-track: in-plane, perpendicular to $\hat{\mathbf{r}}$ |
| $u_n$ | Out-of-plane (normal): along the orbit normal $\hat{\mathbf{h}}$ |

The RSW frame is standard for low-thrust design because each element's
sensitivity to each force direction is transparent: for near-circular orbits,
along-track thrust changes the orbit size most efficiently, out-of-plane thrust
changes inclination, and radial thrust mostly rotates the apse line. The absence
of $e = 0$ and $i = 0$ singularities plus this control parameterization make
`MEEDynamics` the standard choice for orbit-transfer and orbit-raising problems.

### CRTBPDynamics

`CRTBPDynamics` implements the Circular Restricted Three-Body Problem in the
synodic (rotating) frame: a spacecraft of negligible mass under the gravity of two
primaries rotating about their common center of mass. Its parameter $\mu$ is the
**mass ratio** $\mu = m_2/(m_1 + m_2) \in (0,1)$ — not a gravitational parameter;
the larger primary sits at $x = -\mu$ and the smaller at $x = 1 - \mu$ along the
rotating $x$-axis. The state $[x, y, z, v_x, v_y, v_z]$ is in rotating coordinates,
and the equations of motion include both primaries' gravity and the Coriolis and
centrifugal pseudo-forces of the rotating frame.

The CR3BP admits a conserved quantity, the Jacobi constant $C_J$, computable from
the state alone and useful as a constraint or verification check. Periodic orbits
(Lyapunov, halo, and others) are the primary interest in libration-point mission
design, and `CRTBPDynamics` is the model used for them.

### Perturbation and force terms

For problems where additional forces matter, Tycho provides force-term
VectorFunctions that add into the equations of motion:

- **J2 zonal harmonic (`j2_cartesian`)** — the dominant non-spherical
  perturbation for planet-orbit trajectories, causing secular precession of RAAN
  and argument of perigee. `J2Cartesian` takes the position and a (not
  necessarily normalized) north-pole direction (IR = 6) and returns the J2
  acceleration. Add it to the gravity term in `CartesianDynamics` for a perturbed
  model.
- **Solar sail (`non_ideal_solar_sail`)** — the thrust from solar radiation
  pressure on a flat reflective sail. `NonIdealSolarSail` takes position and sail
  normal (IR = 6) and returns the acceleration, parameterized by lightness number
  $\beta$ and optical efficiencies. The sail normal is typically the control.
- **Thrust** — for low-thrust problems the control acceleration in
  `CartesianDynamics` and `MEEDynamics` acts directly as thrust per unit mass.
  Adding a magnitude constraint ($\|\mathbf{u}\| \le a_{\max}$) and an objective on
  cumulative effort recovers the standard minimum-fuel / minimum-time formulation.
  `SimpleLowThrust.py` and `DionysusLowThrust.py` illustrate the pattern.

## From dynamics to a trajectory

Every component on this page connects at one interface: the `Phase`. A dynamics
model is a VectorFunction of $[\text{state}(6), \mathbf{a}(3)]$ returning
$\dot{x}$; passing it to a phase as the ODE registers the dynamics with the
collocation layer. Because the model has no time slot, it is wrapped in an
`ODEBase` whose $[\text{state}, t, \mathbf{a}]$ layout the phase machinery
expects:

```python
from tychopy import optimal_control as oc, vector_functions as vf, astro

class MEEDynamicsODE(oc.ODEBase):
    def __init__(self, mu):
        args = oc.ODEArguments(6, 3)   # [p, f, g, h, k, L, t, u_r, u_t, u_n]
        rhs = astro.modified_dynamics(mu)(vf.stack([args.x_vec(), args.u_vec()]))
        super().__init__(rhs, 6, 3)

ode = MEEDynamicsODE(1.0)              # mu = 1 (non-dimensional)
phase = ode.phase("LGL3", traj_guess, 30)   # 30 LGL3 segments from an initial guess
```

Once a phase holds a dynamics model, the transcription builds defect constraints
from it at every node, boundary and path constraints can be attached in whatever
coordinates are convenient, and PSIOPT solves the resulting NLP — the full
collocation machinery (mesh refinement, control parameterization, Lagrange
objectives) is available regardless of the model. A multi-phase trajectory can
even mix models: a `CartesianDynamics` thrust arc, an `MEEDynamics` heliocentric
leg, and a `CRTBPDynamics` capture, linked into one problem. See
{doc}`How to use astrodynamics dynamics in a phase </user_guide/how_to/astro_dynamics_in_a_phase>`
and the {doc}`low-thrust transfer tutorial </user_guide/low_thrust_transfer>`.

The three ideas to carry away: state representations differ in their singularity
structure, and that choice decides how well the optimizer traverses
low-eccentricity and low-inclination regimes; analytic two-body propagation gives
exact unperturbed trajectories and underpins initial-guess and Lambert
boundary-value work; and the dynamics models are VectorFunctions that plug
directly into a `Phase` with full analytic derivatives.
</content>
