(explanation-direct-collocation)=
# Direct collocation in Tycho

A **phase** is Tycho's representation of one optimal-control problem: find a
trajectory that obeys some dynamics, satisfies the constraints you attach, and
minimizes an objective. Where a
{doc}`VectorFunction </user_guide/vectorfunctions>` describes *what* a quantity
is — the dynamics, a constraint, an objective — a phase describes *what to do
with them*. This page is the practical picture: what a phase is made of, how you
set one up, and what **direct collocation** does for you. The heavier
mathematics — the optimal-control problem in full, the defect equations, the
collocation node theory — lives in [Under the hood](#under-the-hood) at the end;
you can set up and solve phases without reading it.

For a guided, runnable build see the
{doc}`first-phase tutorial </user_guide/first_phase>`. For the complete API
surface see the {doc}`Python reference </reference/python/optimal_control>` and
the {doc}`C++ reference </reference/cpp/optimal_control>`.

## What a phase is made of

A phase is built from two things you supply and a discretization it owns for you.

**The dynamics, as an ODE.** You give the equations of motion as an **ODE**: a
VectorFunction of the packed argument $[x, t, u, p]$ (state, time, control, and
optional static parameters) that returns the state derivative $\dot{x}$. Any of
the {doc}`astrodynamics models </user_guide/astrodynamics>` can serve directly as
an ODE, or you can compose your own.

**A `Phase`, built from that ODE.** The phase owns the discretization (the mesh
and transcription order), holds the constraints and objective you attach, and
drives the solve. The typical flow is: define the ODE, create a phase from it
with a transcription mode and an initial mesh, attach the boundary values, path
constraints, and objective, then call `solve` or `optimize`.

Everything you attach to a phase — a boundary value, a path constraint, an
integral objective — is itself a VectorFunction, evaluated over some region of
the trajectory. *Which* region is named by the `PhaseRegionFlags` enum: a
constraint can act on the `Front` node, the `Back` node, every node of the
`Path`, only the `InnerPath` interior, the parameter blocks, and so on. The key
mental model is that a phase is **not** a monolithic problem object — it is a
collection of VectorFunctions, each tied to a part of the trajectory, that the
transcription assembles into one large system.

A phase can also carry **static parameters** $p$: time-independent unknowns
shared across the whole trajectory — an unknown mass, a thrust magnitude, a free
coast duration. Like free endpoint times, they are optimization variables, not
fixed inputs, and may enter the dynamics, the constraints, and the objective. The
related *link parameters* on an
{py:class}`~tychopy.optimal_control.OptimalControlProblem` play the same role but
are shared across several phases.

## What direct collocation does for you

A continuous trajectory $x(t)$ has a value at every one of uncountably many
instants — no finite optimizer can search that space. **Direct collocation** is
the bridge: it replaces the continuous trajectory with a finite set of unknowns
(the state and control at chosen instants) and replaces the differential equation
with a finite set of algebraic conditions — **defect constraints** — that pin
those unknowns to the dynamics. The result is an ordinary, finite nonlinear
program (NLP) that the optimizer can solve.

You do not assemble any of this by hand. You choose a mesh (how many intervals)
and a transcription order; the phase builds the defect constraints, folds in the
constraints and objective you attached, and hands the whole thing to the solver.
The practitioner-facing choices are the transcription mode, the control
parameterization, and the mesh-refinement settings — covered next. The exact form
of the defect and why it works is in [Under the hood](#under-the-hood).

The one property worth knowing up front is that the resulting NLP is **sparse and
structured**: each defect block touches only the few nodes of its own interval,
and each VectorFunction tracks the inputs it actually reads. The Jacobian of the
whole discretized system is therefore mostly zeros, with a banded structure the
sparse solver exploits directly. This is what keeps large collocation problems
tractable.

## Choosing a transcription

The transcription is *how* the dynamics get enforced on the mesh. The choice
trades per-interval cost against how few intervals you need, and how robust the
scheme is on hard dynamics.

**Collocation orders** (`TranscriptionModes`) approximate the state by a
polynomial on each interval and enforce the dynamics at interior points:

- **LGL3** — a third-order (cubic) polynomial per interval. This is the
  Hermite–Simpson scheme: low per-interval cost, robust convergence, and a sound
  default for most problems.
- **LGL5** — a fifth-order polynomial. Each interval can represent more curved
  dynamics accurately, so a coarser mesh suffices.
- **LGL7** — a seventh-order polynomial, for smooth problems where you want very
  few, very accurate intervals.

Higher order trades more work and more decision variables *per interval* for
fewer intervals at a given accuracy. Which side wins depends on smoothness:
high-order polynomials represent smooth arcs efficiently, while trajectories with
sharp control switches are often better served by a finer mesh of lower-order
intervals.

**Other transcriptions:**

- **`Trapezoidal`** — a second-order scheme (a trapezoidal-rule integration of
  the dynamics). Cheap and very robust; useful as a first pass or for coarse
  feasibility studies.
- **`CentralShooting`** — a fundamentally different idea: instead of a polynomial
  defect, it *integrates* the dynamics with a numerical propagator, shooting both
  bounding nodes of an interval to its midpoint and constraining them to agree.
  Each defect still couples only the two bounding nodes, so the NLP keeps the same
  interval-banded sparsity; but the per-interval Jacobian blocks are dense and
  each evaluation runs a full propagation, so iterations are more expensive.
  Shooting can be more robust for stiff or highly nonlinear dynamics where a
  fixed-order polynomial struggles to represent the arc.

The phase abstraction is the same in every case; only the residual that enforces
the dynamics changes.

## Parameterizing the control

The state's representation is fixed by the transcription, but the **control**
history has more freedom — it need not be as smooth as the state. `ControlModes`
chooses how $u(t)$ is represented between nodes, and the choice is a modeling
decision about the physics, not just a numerical one:

- **`HighestOrderSpline`** — the control is a spline whose order matches the
  transcription's. Appropriate when the physical control genuinely varies
  smoothly.
- **`FirstOrderSpline`** — a piecewise-linear control; a common, well-behaved
  middle ground.
- **`NoSpline`** — the control values at the nodes are independent decision
  variables with no inter-node interpolation. Maximum flexibility, useful when the
  optimal control is expected to be irregular (bang-bang, discontinuous).
- **`BlockConstant`** — the control is held *constant* over each interval. Models
  a physically piecewise-constant control — a thruster firing at a fixed level
  over a segment — and adjusts the transcription's bookkeeping accordingly.

A smoother parameterization has fewer effective degrees of freedom and tends to
regularize the solution; a looser one can represent discontinuous controls but
enlarges the search space. The mode is independent of the transcription order:
you can pair a high-order state polynomial with a piecewise-linear control.

## Mesh refinement

A single fixed mesh is only a first guess at the discretization. The defects can
be driven to zero on a coarse mesh and the *discrete* solution will look
converged — yet the underlying polynomial may still be a poor approximation of the
true trajectory *between* the nodes. The defects being zero says the polynomial
solves the dynamics at the collocation points; it says nothing about the error in
between.

Tycho closes this gap with an **estimate–refine loop** (adaptive mesh
refinement). After a solve, the phase estimates the discretization error in each
interval, and if the worst (or aggregated) error exceeds the target tolerance, it
builds a finer mesh — adding intervals where the error is concentrated — and
re-solves, repeating until the tolerance is met or an iteration cap is hit. Two
settings shape it:

- **`MeshErrorEstimators`** — how per-interval error is estimated. `DEBOOR`
  reuses the solve you already did (cheap); `INTEGRATOR` re-integrates the
  dynamics with a reference propagator and compares (more expensive, more
  independent).
- **`MeshErrorAggregation`** — how the per-interval errors reduce to the scalar
  the convergence test uses. `MAX` is conservative (converged only when *every*
  interval is below tolerance); `AVG` and `GEOMETRIC` are averaged measures;
  `ENDTOEND` compares a full end-to-end re-integration.

The loop is governed by a target tolerance (`mesh_tol`) and bounded by a maximum
iteration count and min/max segment counts, so it always terminates. A common,
deliberate pattern is to *solve* (feasibility only) on the first coarse mesh and
switch to full *optimize* once the mesh is fine enough to trust — cheap iterations
early, accurate iterations late. For a step-by-step recipe see
{doc}`How to refine a mesh </user_guide/how_to/mesh_refinement>`.

The broader point is that mesh refinement is what makes the discrete answer a
reliable surrogate for the continuous one: collocation produces a finite problem,
and mesh refinement establishes that its solution approximates the true trajectory
to the tolerance you asked for.

## Under the hood

Everything above is enough to set up and solve phases. This section gives the
mathematics behind the discretization — useful when you read the transcription
code, debug a convergence problem, or want to know exactly what the optimizer is
being handed. Skip it if you only assemble and solve phases.

### The problem a phase represents

The continuous problem a phase represents is the standard optimal-control
problem: find a state trajectory $x(t)$ and control history $u(t)$ over $[t_0,
t_f]$ that

$$
\min_{x(\cdot),\,u(\cdot),\,p,\,t_0,\,t_f}\;
  \underbrace{\phi\big(x(t_0), t_0, x(t_f), t_f, p\big)}_{\text{Mayer term}}
  \;+\;
  \underbrace{\int_{t_0}^{t_f} L\big(x(t), u(t), t, p\big)\,dt}_{\text{Lagrange term}}
$$

subject to the dynamics, path, and boundary constraints

$$
\dot{x}(t) = f\big(x(t), u(t), t, p\big), \qquad
g\big(x(t), u(t), t, p\big) \le 0, \qquad
b\big(x(t_0), x(t_f), p\big) = 0 .
$$

The objective can carry a **Mayer** term (a function of the boundary states and
times) and a **Lagrange** term (an integral along the trajectory). A minimum-time
problem is a pure Mayer objective on $t_f$; a minimum-fuel problem is typically a
Lagrange integral of control effort. The endpoints $t_0$, $t_f$ may be fixed or
free; when free they become unknowns the optimizer solves for.

### The defect constraint

The first step is to split $[t_0, t_f]$ into a **mesh** of intervals. Within each
interval the state is approximated by a polynomial whose coefficients are fixed by
the state values at a set of **Legendre–Gauss–Lobatto (LGL)** nodes — a specific,
non-uniform placement clustered toward the endpoints, chosen because they make the
interpolating polynomial and its quadrature unusually accurate. The nodes that
carry decision variables are the **cardinal** points; additional **interior**
points are where the approximation is *tested*.

The central idea is the **defect**. If the interval polynomial $\tilde{x}(t)$
truly solves the dynamics, then at each interior (collocation) point its
derivative must equal the dynamics evaluated there. The defect is the amount by
which this fails:

$$
\delta_c \;=\; \dot{\tilde{x}}(t_c) - f\big(\tilde{x}(t_c), u(t_c), t_c, p\big).
$$

Forcing every defect to zero, $\delta_c = 0$, is exactly the statement that the
polynomial is a discrete solution of the ODE on that interval. In practice the
polynomial derivative is a weighted combination of the cardinal state values and
the dynamics at the cardinal nodes, so the defect reads schematically as

$$
\delta \;=\; \sum_j w^{x}_{j}\, x_j \;+\; h\sum_j w^{f}_{j}\, f(x_j, u_j, t_j, p) \;=\; 0,
$$

where $h$ is the interval duration and the weights $w$ come from the LGL
coefficient tables for the chosen order. Tycho assembles this defect as a single
VectorFunction (`LGLDefects`) carrying analytic derivatives, so the optimizer
receives not just the residual but its Jacobian and Hessian. The defect residuals
across all intervals become the **dynamics constraint** of the discrete problem —
one equality block per interval.

`CentralShooting` replaces this polynomial defect with an integrated one: for each
interval $[t_k, t_{k+1}]$ it propagates the left node forward and the right node
backward to the midpoint $t_m = \tfrac{1}{2}(t_k + t_{k+1})$, and the defect
constrains the two propagated states to agree at $t_m$. Matching at the midpoint
rather than an endpoint is what makes it *central* and keeps the residual
symmetric. The per-interval Jacobian blocks are filled from the integrator's
state-transition matrix.

### What transcription produces

Once a mesh and order are fixed, the three ingredients of the continuous problem
map cleanly onto the three ingredients of an NLP:

| Continuous problem | Discrete NLP |
| --- | --- |
| $x(t)$, $u(t)$, free $t_0/t_f$, parameters $p$ | **Decision variables**: state and control at every node, the phase parameters, and any free times |
| $\dot{x} = f$, $g \le 0$, $b = 0$ | **Constraints**: defect (dynamics) blocks + boundary + path/integral constraints |
| Mayer + Lagrange objective | **Objective**: a scalar function of the decision variables |

The decision vector concatenates every node's state and control, the phase's
parameter blocks, and the free endpoint times into one long vector. The constraint
set is the union of the per-interval defect blocks with every user-attached
boundary value and path constraint, each evaluated over the nodes its
`PhaseRegionFlags` region selects. The Lagrange integral is discretized by a
quadrature rule matched to the transcription (`IntegralModes`); the Mayer term is
a plain function of the boundary nodes. The transcription does not just produce
*an* NLP — it produces one whose sparsity pattern mirrors the locality of the
dynamics.

### Solving with PSIOPT

The transcribed NLP is handed to **PSIOPT**, Tycho's bundled sparse interior-point
nonlinear optimizer. It consumes the decision vector, constraint residuals,
objective, and their analytic derivatives — all delivered by the assembled
VectorFunctions — and returns a decision vector that is locally optimal and
feasible to a requested tolerance.

PSIOPT solves the Karush–Kuhn–Tucker (KKT) conditions by following an
interior-point path, assembling a sparse KKT linear system at each iteration from
the Jacobians and adjoint Hessians the VectorFunctions provide, and factoring it
with a sparse linear solver. The fused value-Jacobian-gradient-Hessian call that
every VectorFunction supports exists precisely so this inner loop never has to
reconstruct derivatives numerically. The interior-point internals — barrier
parameters, line searches, the KKT factorization, convergence flags — are a topic
of their own. For understanding a phase, the essential contract is:
**transcription produces a sparse, differentiable NLP; PSIOPT solves it.** A phase
can run PSIOPT in *solve* mode (drive constraints to feasibility) or *optimize*
mode (feasibility plus minimizing the objective).

## Where to go next

This page covered the practical pipeline: ODE + phase → collocation mesh and
defects → sparse NLP → PSIOPT solve → mesh refinement. To go further:

- **Tutorial.** For a guided, runnable build of a phase from an ODE through to a
  solved trajectory, start with the
  {doc}`first-phase tutorial </user_guide/first_phase>`.
- **Reference.** The complete API for phases, ODEs, transcription modes, control
  modes, and mesh settings is in the
  {doc}`Python reference </reference/python/optimal_control>` and the
  {doc}`C++ reference </reference/cpp/optimal_control>`.
- **The dynamics layer.** Every defect, constraint, and objective in a phase is a
  VectorFunction; the {doc}`VectorFunction model </user_guide/vectorfunctions>`
  explains the differentiable, symbolic machinery the transcription is built on.

The two organizing ideas: a phase turns a continuous optimal-control problem into
a sparse, differentiable NLP by enforcing the dynamics as **defect constraints**
on a collocation mesh, and **mesh refinement** iterates that discretization until
the discrete solution faithfully represents the continuous trajectory.
</content>
</invoke>
