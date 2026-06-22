(explanation-astrodynamics)=
# Astrodynamics in Tycho

Tycho's astrodynamics subsystem provides three orbital state representations,
analytic two-body propagation, Lambert's problem, and a set of dynamics models
that can be used directly as the ODE of an optimal-control
{doc}`Phase </explanation/direct_collocation>`. This page explains the
conceptual model behind each of these components and the reasoning that guides
when to use each one.

It is conceptual rather than exhaustive. For the full API surface see the
{doc}`Python reference </reference/python/astrodynamics>` and the
{doc}`C++ reference </reference/cpp/astrodynamics>`.

## State representations

A spacecraft state in two-body orbital mechanics requires six scalar degrees of
freedom. Tycho provides three coordinate representations of that six-dimensional
state, each with different analytic properties and different suitability for
optimization.

### Cartesian elements

The most direct representation is the Cartesian state vector
$[r_x, r_y, r_z, v_x, v_y, v_z]$: three components of position and three of
velocity, all measured in an inertial frame. No angle conventions, no anomaly
choices, no singularity conditions — the state space is $\mathbb{R}^6$ and
every point in it is a valid state (up to the physical requirement $r \neq 0$).

Cartesian coordinates are the natural output of numerical integrators, the
natural input to force models (gravity, thrust, drag all depend on $\mathbf{r}$
and $\mathbf{v}$ directly), and the natural language for boundary conditions
that specify physical positions and velocities. The `CartesianDynamics` model
accepts and produces Cartesian states, and the Cartesian propagators operate on
them.

The drawback of Cartesian coordinates for trajectory optimization is not
accuracy but scale: low-thrust transfers that take days to months to complete
have slowly evolving orbital elements but rapidly oscillating Cartesian
coordinates, which makes the trajectory hard to represent on a coarse
collocation mesh. For such problems element-set representations are better
conditioned.

### Classical (Keplerian) elements

The classical orbital elements $[a, e, i, \Omega, \omega, M]$ parameterize the
shape, orientation, and instantaneous position of an orbit:

| Symbol | Meaning |
| --- | --- |
| $a$ | Semi-major axis |
| $e$ | Eccentricity |
| $i$ | Inclination |
| $\Omega$ | Right ascension of the ascending node (RAAN) |
| $\omega$ | Argument of perigee |
| $M$ | Mean anomaly |

Tycho uses mean anomaly as the sixth element (not true anomaly); the conversion
functions `cartesian_to_classic` and `classic_to_cartesian` follow this
convention, as does the `CartesianToClassic` VectorFunction.

Classical elements separate the slowly changing orbit geometry ($a$, $e$, $i$,
$\Omega$, $\omega$, and the orbit plane orientation) from the fast-evolving
angular position along that orbit ($M$). For a Keplerian two-body orbit, the
first five elements are constants of the motion and only $M$ changes linearly
with time. This separation can be exploited both for propagation (only $M$
needs updating for the unperturbed case) and for formulating boundary conditions
in terms of the desired final orbit rather than a single final state.

However, classical elements carry two well-known singularities:

- **Circular orbits ($e = 0$):** The argument of perigee $\omega$ becomes
  undefined because there is no perigee to measure an angle from.
- **Equatorial orbits ($i = 0$ or $i = \pi$):** The RAAN $\Omega$ becomes
  undefined because the orbit does not cross the reference equatorial plane.

Near-circular and near-equatorial conditions are common in practical trajectory
problems — station-keeping around geostationary orbit, for example. At these
limits the classical elements are numerically ill-conditioned even if not
strictly undefined, and small physical changes can produce large changes in
$\omega$ or $\Omega$, which is unfavorable for both propagation accuracy and
optimizer convergence.

### Modified Equinoctial Elements (MEE)

Modified Equinoctial Elements $[p, f, g, h, k, L]$ were introduced specifically
to remove the singularities of the classical representation:

| Symbol | Meaning |
| --- | --- |
| $p$ | Semi-latus rectum, $p = a(1 - e^2)$ |
| $f$ | Eccentricity component, $f = e\cos(\omega + \Omega)$ |
| $g$ | Eccentricity component, $g = e\sin(\omega + \Omega)$ |
| $h$ | Inclination component, $h = \tan(i/2)\cos\Omega$ |
| $k$ | Inclination component, $k = \tan(i/2)\sin\Omega$ |
| $L$ | True longitude, $L = \omega + \Omega + \nu$ (true anomaly $\nu$) |

The mapping to classical elements is algebraic and closed-form:
$e = \sqrt{f^2 + g^2}$, $i = 2\arctan\!\sqrt{h^2 + k^2}$ (for $i < 180^\circ$),
$\Omega = \text{atan2}(k, h)$, $\omega = \text{atan2}(gh - fk,\, fh + gk)$.

At $e = 0$ the components $f$ and $g$ both go to zero smoothly — no angle
becomes undefined. At $i = 0$ the components $h$ and $k$ both go to zero
smoothly. The only remaining singularity is at exactly $i = 180°$ (a
retrograde equatorial orbit), where $\tan(i/2) \to \infty$ and $h$, $k$
diverge. Near-retrograde non-equatorial orbits are fully supported; the
singularity is at the single geometric configuration that combines exactly
retrograde inclination with exactly zero out-of-plane component.

Removing the near-circular and near-equatorial singularities is the central
reason MEE is the preferred element set for low-thrust trajectory optimization.
The optimizer can follow a trajectory that passes through zero eccentricity or
grazes equatorial inclination without encountering pathological Jacobians or
numerical blow-up. In practice, most heliocentric and many Earth-orbit low-thrust
trajectories are better conditioned in MEE than in classical elements.

### Choosing a representation

The choice is problem-driven:

- **Cartesian** — use for force modeling, for trajectories that are not
  long-duration low-thrust (high-thrust, impulsive, or short arcs), and as the
  state representation in the CR3BP model (which is naturally Cartesian in the
  synodic frame).
- **Classical elements** — use when the orbit is well away from circular and
  equatorial and you want boundary conditions expressed in terms of orbital
  geometry.
- **MEE** — use for low-thrust transfers, particularly those that cross or come
  near zero eccentricity or zero inclination. The `MEEDynamics` model is
  formulated in MEE and is the standard choice for orbit-raising and
  orbit-transfer problems.

Tycho provides conversion VectorFunctions — `CartesianToClassic`,
`CartesianToMEE`, `ModifiedToCartesian` — so that a trajectory optimized in one
representation can have boundary conditions expressed in another. These are
differentiable and carry analytic Jacobians, so they can be composed into
boundary-value constraints without breaking the optimizer's derivative chain.

## Propagation

### Analytic two-body propagation

For an unperturbed two-body orbit — no thrust, no higher-order gravity, no drag
— the trajectory from an initial state at $t_0$ to any later time $t_0 + \Delta t$
can be computed exactly in closed form. Tycho exposes this through three
propagation functions (`propagate_cartesian`, `propagate_classic`,
`propagate_modified`), all of which share the same underlying algorithm.

The algorithm is the **universal-variable** formulation using Stumpff functions.
The classical approach of separate cases for elliptic and hyperbolic orbits (via
Kepler's equation for the respective anomaly) breaks down near parabolic
trajectories and requires logic to select the right case. The
universal-variable formulation uses a single real variable $X$ — not tied to a
specific anomaly convention — and expresses the propagation equations in terms
of the Stumpff functions $U_0, U_1, U_2, U_3$, which are well-defined and
smoothly varying across elliptic, parabolic, and hyperbolic regimes. The value
$X$ is found by a fixed-point iteration (the **LCD** kernel), and the propagated
Cartesian state is then recovered via the Goodyear $f$-$g$ map. The Cartesian
propagator (`propagate_cartesian`) uses this approach directly; the classical and
MEE propagators convert to Cartesian, propagate, and convert back.

For the `KeplerPropagator` VectorFunction, which takes an initial Cartesian
state and a time-of-flight as inputs and returns the propagated state, the
Jacobian and Hessian are computed via the Implicit Function Theorem (IFT)
applied to the converged LCD fixed-point equation, giving exact analytic
derivatives that propagate into the optimizer without numerical perturbations.

Analytic two-body propagation is useful in three contexts:

1. **Initial-guess generation.** A Keplerian two-body arc is a good first
   approximation for a coast arc, and computing it analytically is orders of
   magnitude faster than running a numerical integrator.
2. **Boundary condition setup.** Given a known initial state and time of flight,
   the terminal state of a purely Keplerian arc is computed exactly and can be
   used to set tight bounds or equality constraints.
3. **Pure-Keplerian phase.** The `tychopy.astro.kepler.phase` wraps a
   `CentralShooting` phase around `KeplerPropagator`, letting a Keplerian arc
   participate in a multi-phase optimal-control problem with full sensitivity
   information.

### Lambert's problem

Lambert's problem is the two-point boundary-value problem of finding the
velocity vector at the departure point of a two-body orbit that passes through
a given arrival point in a given time of flight:

Given $\mathbf{r}_1$, $\mathbf{r}_2$, and $\Delta t$, find $\mathbf{v}_1$
(and consequently $\mathbf{v}_2$) such that the spacecraft travels from
$\mathbf{r}_1$ to $\mathbf{r}_2$ under two-body gravity in time $\Delta t$.

Tycho implements Izzo's algorithm (`lambert_izzo`, `lambert_izzo_multirev`),
which is robust, fast, and handles both single-revolution transfers and
multi-revolution solutions.

**Long-way vs. short-way.** For a given pair of endpoints and time of flight
there are in general two single-revolution solutions: the **short-way** transfer
(traversing the shorter arc between the two radius vectors) and the
**long-way** transfer (traversing the longer arc, going the other direction).
The Tycho API selects between them with the `longway` boolean flag.

**Multi-revolution.** For long enough times of flight a spacecraft can complete
one or more full orbits before arriving. Each additional revolution introduces
two more solutions (associated with two branches of the multi-revolution Lambert
problem). The `lambert_izzo_multirev` function accepts the number of complete
revolutions `Nrevs` and a `rightbranch` flag to select between the two branches
for $\text{Nrevs} \geq 1$.

Lambert's problem is the enabling computation for the **pork-chop plot** and for
impulsive maneuver design, where the departure and arrival positions are
specified and the optimizer needs the velocity increments ($\Delta v$) that
achieve a transfer in a given time. It is also useful as an initial-guess
generator for low-thrust problems: a Lambert arc provides a reasonable Keplerian
shape to initialize a trajectory before the optimizer refines it.

## Dynamics models

A dynamics model in Tycho is a **VectorFunction** that maps the packed input
$[\text{state}(6), \mathbf{a}(3)]$ (IR = 9) to the state derivative
$\dot{x}$ (OR = 6). The trailing three inputs are the *non-two-body
acceleration* — every acceleration on the spacecraft beyond central-body
point-mass gravity. In an optimal-control problem this is usually the thrust
control the optimizer chooses, but it can equally carry modeled perturbations
(J2, atmospheric drag, solar-radiation pressure), or their sum (see
{doc}`How to use astrodynamics dynamics in a phase </how_to/astro_dynamics_in_a_phase>`).
This is exactly the ODE form that a `Phase` expects: the
{doc}`direct collocation </explanation/direct_collocation>` transcription
enforces the defect $\dot{\tilde{x}} = f(\tilde{x}, u)$ at every collocation
point, where $f$ is the dynamics VectorFunction. Each model carries analytic
Jacobian and Hessian implementations, so PSIOPT receives exact first- and
second-derivative information rather than finite-difference approximations.

### CartesianDynamics

`CartesianDynamics` is the two-body equations of motion in Cartesian
coordinates. Its input layout is
$[r_x, r_y, r_z, v_x, v_y, v_z, a_x, a_y, a_z]$
and its output is the state derivative
$[\dot{r}_x, \dot{r}_y, \dot{r}_z, \dot{v}_x, \dot{v}_y, \dot{v}_z]$:

$$
\begin{pmatrix} \dot{\mathbf{r}} \\ \dot{\mathbf{v}} \end{pmatrix}
=
\begin{pmatrix} \mathbf{v} \\ -\dfrac{\mu}{r^3}\mathbf{r} + \mathbf{a} \end{pmatrix}
$$

where $r = \|\mathbf{r}\|$ and $\mu$ is the gravitational parameter. The
non-two-body acceleration $\mathbf{a}$ enters directly in the inertial frame. In
a low-thrust problem it is the thrust control whose magnitude and direction the
optimizer selects at each collocation node; perturbations modeled as functions
of the state are added into the same term.

`CartesianDynamics` is appropriate for problems where the state is most naturally
expressed in Cartesian coordinates: high-thrust or impulsive transfers, short-arc
maneuvers, or problems in the CR3BP where the rotating-frame dynamics are already
Cartesian.

### MEEDynamics

`MEEDynamics` is the two-body equations of motion in Modified Equinoctial
Elements with the non-two-body acceleration given in the **RSW frame** (radial,
along-track, and out-of-plane). The input layout is
$[p, f, g, h, k, L, u_r, u_t, u_n]$ and the output is $[\dot{p}, \dot{f},
\dot{g}, \dot{h}, \dot{k}, \dot{L}]$ as given by the Gauss variation-of-parameters
equations in MEE form.

The three components $[u_r, u_t, u_n]$ are the non-two-body acceleration in the
spacecraft-centered RSW frame (radial, along-track, and out-of-plane) — most
commonly a low-thrust control:

| Component | Direction |
| --- | --- |
| $u_r$ | Radial: along the position vector $\hat{\mathbf{r}}$ |
| $u_t$ | Along-track (tangential): in the orbital plane, perpendicular to $\hat{\mathbf{r}}$ |
| $u_n$ | Out-of-plane (normal): along the orbit normal $\hat{\mathbf{h}}$ |

The RSW frame is standard for low-thrust trajectory design because the
sensitivity of each element to each force direction is transparent. As a rule of
thumb for near-circular orbits, along-track thrust is the most efficient way to
change the orbit's size (semi-major axis), out-of-plane thrust changes its
inclination, and radial thrust — the least efficient of the three axes —
primarily rotates the apse line and adjusts eccentricity.

The absence of singularities at $e = 0$ and $i = 0$, combined with the RSW
control parameterization, makes `MEEDynamics` the standard choice for low-thrust
orbit transfer and orbit-raising problems.

### CRTBPDynamics

`CRTBPDynamics` implements the equations of motion for the Circular Restricted
Three-Body Problem (CR3BP) in the synodic (rotating) frame. The problem models
a spacecraft of negligible mass under the gravity of two primaries (mass $m_1$
and $m_2$) that rotate in circular orbits about their common center of mass.

The model parameter $\mu$ is the **mass ratio** $\mu = m_2/(m_1 + m_2) \in
(0, 1)$; it is not a gravitational parameter in the two-body sense. The position
of the larger primary is at $x = -\mu$ along the rotating $x$-axis; the smaller
primary is at $x = 1 - \mu$. The state is expressed in rotating (synodic)
Cartesian coordinates $[x, y, z, v_x, v_y, v_z]$, and the equations of motion
include both the gravitational attraction of both primaries and the
pseudo-forces (Coriolis and centrifugal) arising from the rotating reference
frame:

$$
\begin{pmatrix} \dot{\mathbf{r}} \\ \dot{\mathbf{v}} \end{pmatrix}
=
\begin{pmatrix}
  \mathbf{v} \\[4pt]
  x - \dfrac{(1-\mu)(x+\mu)}{r_1^3} - \dfrac{\mu(x+\mu-1)}{r_2^3} + 2v_y + a_x \\[4pt]
  y - \dfrac{(1-\mu)\,y}{r_1^3} - \dfrac{\mu\,y}{r_2^3} - 2v_x + a_y \\[4pt]
  -\dfrac{(1-\mu)\,z}{r_1^3} - \dfrac{\mu\,z}{r_2^3} + a_z
\end{pmatrix}
$$

where $r_1 = \|\mathbf{r} - \mathbf{r}_1\|$ and $r_2 = \|\mathbf{r} - \mathbf{r}_2\|$
are the distances to the two primaries.

The CR3BP admits a conserved quantity, the Jacobi constant $C_J$, which can be
computed from the state alone and used as a constraint or a solution-verification
check. Periodic orbits in the CR3BP — Lyapunov orbits, halo orbits, and others
— are of primary interest in libration-point mission design, and
`CRTBPDynamics` is the dynamics model used for all such problems.

### Perturbation and force terms

The dynamics models above represent the unperturbed or controlled-thrust
equations of motion. For problems where additional forces are significant, Tycho
provides force-term VectorFunctions that can be added to the equations of motion.

**J2 zonal harmonic (`j2_cartesian`).** The J2 term is the dominant
non-spherical perturbation for Earth-orbit (and other planet-orbit) trajectories.
It arises from the equatorial bulge of the planet and causes secular precession
of the RAAN and argument of perigee. The `J2Cartesian` VectorFunction takes the
spacecraft position and a north-pole direction vector (IR = 6) and returns the
J2 acceleration as a 3-vector. The inputs are $[r_x, r_y, r_z,
n_x, n_y, n_z]$, where $\mathbf{n}$ is the planet's north-pole direction and
need not be pre-normalized; the acceleration depends on the angle between the
position vector and the rotation axis. The J2 term can be added to the
gravity term in `CartesianDynamics` to construct a perturbed Cartesian dynamics
model.

**Solar sail (`non_ideal_solar_sail`).** The solar-sail acceleration model
computes the thrust produced by solar radiation pressure on a flat reflective
sail. The `NonIdealSolarSail` VectorFunction takes the position vector and sail
normal direction (IR = 6) as inputs and returns the sail acceleration (3-vector)
parameterized by the lightness number $\beta$ and optical efficiency coefficients
$(n_1, n_2, t_1)$ for normal-force efficiency, absorption efficiency, and
tangential-force efficiency respectively. In solar-sail trajectory design the
sail normal is typically the control variable, whose direction the optimizer
determines at each collocation node.

**Thrust.** For low-thrust problems the control acceleration in `CartesianDynamics`
and `MEEDynamics` acts directly as the thrust force per unit mass. Adding a
thrust-magnitude constraint (for example, $\|\mathbf{u}\| \leq a_{\max}$) and
an objective on cumulative thrust effort recovers the standard minimum-fuel or
minimum-time low-thrust formulation. The examples `SimpleLowThrust.py` and
`DionysusLowThrust.py` illustrate this pattern: the control acceleration is
stacked with the state into the ODE input, and path constraints enforce the
thrust bound.

## From dynamics to a trajectory

Every component described in this page — state representation, propagator,
dynamics model — connects at a single interface: the `Phase`.

A dynamics model such as `MEEDynamics` or `CartesianDynamics` is a
{doc}`VectorFunction </explanation/vector_function>` of the packed
$[\text{state}(6), \mathbf{a}(3)]$ input returning $\dot{x}$. Passing it to
a `Phase` as the ODE registers the dynamics with the collocation layer:

`modified_dynamics` has the input layout $[\text{state}(6), \mathbf{a}(3)]$
with no time slot, so it is wrapped in an `ODEBase` whose argument layout
$[\text{state}, t, \mathbf{a}]$ the phase machinery expects:

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

The {doc}`low-thrust transfer tutorial </tutorials/astrodynamics/low_thrust_transfer>`
walks through this composition end to end with a runnable solve.

Once a `Phase` holds a dynamics model, the transcription builds defect
constraints from it at every collocation node, boundary-value and path
constraints can be attached (expressed in whatever coordinate system is
convenient), and PSIOPT solves the resulting NLP. The full collocation
machinery — mesh refinement, control parameterization, Lagrange objectives —
is available regardless of which dynamics model is used. The dynamics model
defines the physics; the phase owns the discretization.

A multi-phase trajectory can mix dynamics models. A transfer from low Earth
orbit to a high-energy departure might use a `CartesianDynamics` phase for the
initial thrust arc (where Cartesian coordinates are well-conditioned), an
`MEEDynamics` phase for the long low-thrust heliocentric leg, and a
`CRTBPDynamics` phase for capture into a libration-point orbit. The
{doc}`direct collocation explanation </explanation/direct_collocation>` covers
how phases are linked and how the NLP is assembled from multiple phase blocks.

For a full API listing of the astrodynamics subsystem — conversion functions,
propagators, dynamics classes, and their parameters — see the
{doc}`Python reference </reference/python/astrodynamics>` and the
{doc}`C++ reference </reference/cpp/astrodynamics>`.

The three organizing ideas of this page are these: orbital state
representations differ in their singularity structure, and the choice of
representation determines how well the optimizer can traverse low-eccentricity
and low-inclination regimes; analytic two-body propagation provides exact
trajectories for the unperturbed case and is the foundation for initial-guess
generation and the Lambert boundary-value solver; and the dynamics models are
VectorFunctions that plug directly into a `Phase`, giving the transcription and
the optimizer full analytic derivative access to the equations of motion.
