(explanation-direct-collocation)=
# Direct collocation in Tycho

A **phase** is Tycho's discretization of one continuous optimal-control problem.
Where a {doc}`VectorFunction </explanation/vector_function>` describes *what* a
quantity is — the dynamics, a constraint, an objective — a phase describes *what
to do with them*: find a trajectory that obeys the dynamics, satisfies the
constraints, and minimizes the objective. This page explains the conceptual
model that takes you from a continuous problem statement to a number the solver
can optimize: the continuous optimal-control problem, its **direct collocation**
transcription into a finite-dimensional nonlinear program (NLP), the solve, and
the mesh-refinement loop that makes the discrete answer faithful to the
continuous one.

It is conceptual rather than exhaustive. For the full API surface see the
{doc}`Python reference </reference/python/optimal_control>` and the
{doc}`C++ reference </reference/cpp/optimal_control>`; for a guided build, the
{doc}`Tutorials </tutorials/index>`.

## The problem a phase solves

The continuous problem a phase represents is the standard optimal-control
problem. We seek a state trajectory $x(t)$ and a control history $u(t)$ over a
time interval $[t_0, t_f]$ that

$$
\min_{x(\cdot),\,u(\cdot),\,p,\,t_0,\,t_f}\;
  \underbrace{\phi\big(x(t_0), t_0, x(t_f), t_f, p\big)}_{\text{Mayer term}}
  \;+\;
  \underbrace{\int_{t_0}^{t_f} L\big(x(t), u(t), t, p\big)\,dt}_{\text{Lagrange term}}
$$

subject to the dynamics, boundary conditions, and path constraints

$$
\dot{x}(t) = f\big(x(t), u(t), t, p\big), \qquad
g\big(x(t), u(t), t, p\big) \le 0, \qquad
b\big(x(t_0), x(t_f), p\big) = 0 .
$$

The objective can carry a **Mayer** term — a function of the boundary states and
times — and a **Lagrange** term — an integral accumulated along the trajectory.
Either or both may be present; a minimum-time problem is a pure Mayer objective
on $t_f$, a minimum-fuel problem is typically a Lagrange integral of control
effort. The endpoints $t_0$ and $t_f$ may be fixed or free, and when free they
become unknowns the optimizer solves for alongside the trajectory itself.

The vector $p$ holds **static parameters**: time-independent unknowns shared
across the whole trajectory — an unknown mass, a thrust magnitude, a free coast
duration. Like the endpoint times, they are optimization variables rather than
fixed inputs, and they may enter the dynamics, the path and boundary
constraints, and the objective. A phase carries its own static parameters; the
related *link parameters* on an {py:class}`~tychopy.optimal_control.OptimalControlProblem`
play the same role but are shared across several phases.

In Tycho this maps onto two objects. The dynamics $f$ are supplied as an **ODE**:
a VectorFunction of the packed argument $[x, t, u, p]$ (state, time, control, and
optional parameters) returning $\dot{x}$. The ODE is then handed to a **`Phase`**,
which owns the discretization, the constraints, and the objective, and ultimately
drives the solve. Everything you attach to a phase — a boundary value, a path
constraint, an integral objective — is itself a VectorFunction evaluated over
some region of the discretized trajectory. That regional structure is captured by
the `PhaseRegionFlags` enum: a constraint can act on the `Front` node, the `Back`
node, every node of the `Path`, only the `InnerPath` interior, the parameter
blocks, and so on. The conceptual point is that a phase is *not* a monolithic
problem object — it is a collection of VectorFunctions, each tied to a part of the
trajectory, that the transcription will assemble into one large system.

## From continuous to discrete: collocation

A continuous trajectory is an infinite-dimensional object: $x(t)$ has a value at
every one of uncountably many instants. No finite optimizer can search that space
directly. **Direct transcription** replaces the continuous trajectory with a
finite set of unknowns — the state and control at chosen instants — and replaces
the differential equation with a finite set of algebraic conditions that pin
those unknowns to the dynamics. Tycho's default transcription family is **direct
collocation**.

The first step is to split $[t_0, t_f]$ into a **mesh** of intervals (segments).
Within each interval, the state trajectory is approximated by a polynomial, and
the polynomial's coefficients are fixed by the state values at a set of
**Legendre–Gauss–Lobatto (LGL)** nodes. LGL nodes are a specific, non-uniform
placement of points within the interval — clustered toward the endpoints — chosen
because they make the interpolating polynomial and its associated quadrature
unusually accurate. The nodes that carry decision variables are the **cardinal**
points; additional **interior** points are where the approximation is *tested*.

The key idea — the heart of collocation — is the **defect constraint**. If the
interval polynomial $\tilde{x}(t)$ truly represents a solution of the dynamics,
then at each interior (collocation) point its time derivative must equal the
dynamics evaluated there:

$$
\dot{\tilde{x}}(t_c) \;=\; f\big(\tilde{x}(t_c), u(t_c), t_c, p\big).
$$

The **defect** is the amount by which this fails to hold:

$$
\delta_c \;=\; \dot{\tilde{x}}(t_c) - f\big(\tilde{x}(t_c), u(t_c), t_c, p\big).
$$

Forcing every defect to zero, $\delta_c = 0$, is exactly the statement that the
collocation polynomial is a discrete solution of the ODE on that interval. In
practice the polynomial derivative is expressed as a weighted combination of the
cardinal state values and the dynamics evaluated at the cardinal nodes, so the
defect for an interval reads schematically as

$$
\delta \;=\; \sum_j w^{x}_{j}\, x_j \;+\; h\sum_j w^{f}_{j}\, f(x_j, u_j, t_j, p) \;=\; 0,
$$

where $h$ is the interval duration and the weights $w$ come from the LGL
coefficient tables for the chosen order. Tycho assembles this defect as a single
VectorFunction (`LGLDefects`) carrying analytic derivatives, so the optimizer
receives not just the residual but its Jacobian and Hessian. The defect residuals
across all intervals become the **dynamics constraint** of the discrete problem —
one equality block per interval, stitched together over the whole phase.

The **order** of the collocation polynomial sets the accuracy of this
approximation per interval, and Tycho exposes it through `TranscriptionModes`:

- **LGL3** — a third-order (cubic) polynomial per interval, two cardinal states
  and one collocation point. This is the Hermite–Simpson scheme: lightweight,
  robust, a good default.
- **LGL5** — a fifth-order polynomial, more cardinal states and collocation
  points per interval. Higher order means each interval can represent more curved
  dynamics accurately, so a coarser mesh suffices.
- **LGL7** — a seventh-order polynomial; higher order still, for smooth problems
  where you want very few, very accurate intervals.

Higher order trades more work and more decision variables *per interval* for
fewer intervals at a given accuracy. Which side of that trade wins depends on how
smooth the trajectory is: smooth arcs love high order, while trajectories with
sharp control switches are often better served by a finer mesh of lower-order
intervals.

Collocation is not the only transcription Tycho offers. `Trapezoidal` is a
second-order scheme — the interval polynomial is effectively linear and the
defect is a trapezoidal-rule integration of the dynamics — cheap and very robust,
useful as a first pass or for coarse feasibility studies. `CentralShooting` is a
fundamentally different idea: instead of enforcing a polynomial defect, it
*integrates* the dynamics with a numerical propagator. For each interval
$[t_k, t_{k+1}]$ it shoots both bounding nodes to the interval **midpoint**
$t_m = \tfrac{1}{2}(t_k + t_{k+1})$ — the left node propagated forward from
$(x_k, t_k)$, the right node propagated backward from $(x_{k+1}, t_{k+1})$ — and
the defect constrains the two propagated states to agree at $t_m$. (Matching at
the midpoint rather than at one endpoint is what makes it *central* shooting,
and keeps the residual symmetric in the two nodes.) Each defect still couples
only the two bounding nodes, so the NLP keeps the same interval-banded
sparsity as collocation; but the per-interval Jacobian blocks are dense —
filled from the integrator's state-transition matrix — and each evaluation
runs a full numerical propagation, so iterations are more expensive. Shooting
can be more robust for stiff or highly nonlinear dynamics where a fixed-order
polynomial struggles to represent the arc. The phase abstraction is the same in
every case; only the residual that enforces the dynamics changes.

## What transcription produces

Once a mesh and a transcription order are fixed, the phase turns the continuous
problem into a concrete, finite **nonlinear program**. The three ingredients of
the continuous problem map cleanly onto the three ingredients of an NLP:

| Continuous problem | Discrete NLP |
| --- | --- |
| $x(t)$, $u(t)$, free $t_0/t_f$, parameters $p$ | **Decision variables**: state and control at every node, the phase parameters, and any free times |
| $\dot{x} = f$, $g \le 0$, $b = 0$ | **Constraints**: defect (dynamics) blocks + boundary + path/integral constraints |
| Mayer + Lagrange objective | **Objective**: a scalar function of the decision variables |

The decision vector is the concatenation of every node's state and control, the
phase's parameter blocks, and the free endpoint times — a single long vector. The
constraint set is the union of the per-interval defect blocks (the discretized
dynamics) with every user-attached boundary value and path constraint, each
evaluated over the nodes its `PhaseRegionFlags` region selects. The Lagrange
integral in the objective is discretized by a quadrature rule matched to the
transcription (`IntegralModes`), accumulating contributions interval by interval;
the Mayer term is a plain function of the boundary nodes.

This finite NLP is the bridge to the solver. Crucially, it is **sparse and
structured**: each defect block touches only the few nodes of its own interval,
and each VectorFunction tracks the input domain it actually reads (see the
{doc}`VectorFunction model </explanation/vector_function>`). The Jacobian of the
whole discretized system is therefore mostly zeros, with a banded, near-block
structure that the sparse solver exploits directly. The transcription does not
just produce *an* NLP — it produces one whose sparsity pattern mirrors the
locality of the dynamics, which is what keeps large collocation problems
tractable.

## Solving with PSIOPT

The transcribed NLP is handed to **PSIOPT**, Tycho's bundled sparse
interior-point nonlinear optimizer. From the phase's point of view, PSIOPT is the
component that consumes the decision vector, constraint residuals, objective, and
their analytic derivatives — all delivered by the assembled VectorFunctions — and
returns a decision vector that is locally optimal and feasible to a requested
tolerance.

PSIOPT solves the Karush–Kuhn–Tucker (KKT) conditions of the NLP by following an
interior-point path, assembling a sparse KKT linear system at each iteration from
the Jacobians and adjoint Hessians the VectorFunctions provide, and factoring it
with a sparse linear solver. The fused value-Jacobian-gradient-Hessian call that
every VectorFunction supports exists precisely so this inner loop never has to
reconstruct derivatives numerically.

The internals of the interior-point method — barrier parameters, line searches,
the KKT factorization, convergence flags — are a topic in their own right and are
covered in a separate Solvers explanation (Plan 3). For the purposes of
understanding a phase, the essential fact is the contract: **transcription
produces a sparse, differentiable NLP; PSIOPT solves it.** A phase can run PSIOPT
in *solve* mode (drive the constraints to feasibility) or *optimize* mode
(feasibility plus minimizing the objective), and the mesh loop below uses both.

## Control parameterization

The state trajectory's representation is dictated by the collocation polynomial,
but the **control** history has more freedom, because the control need not be as
smooth as the state. `ControlModes` chooses how $u(t)$ is represented between
nodes, and the choice is a modeling decision about the physics, not just a
numerical one:

- **`HighestOrderSpline`** — the control is interpolated by a spline whose order
  matches the transcription's. The control is as smooth as the state
  representation; appropriate when the physical control genuinely varies smoothly.
- **`FirstOrderSpline`** — a piecewise-linear control. The control varies linearly
  between nodes regardless of the state polynomial's order; a common, well-behaved
  middle ground.
- **`NoSpline`** — the control values at the nodes are independent decision
  variables with no inter-node interpolation tying them together. Maximum
  flexibility, useful when the optimal control is expected to be irregular.
- **`BlockConstant`** — the control is held *constant* over each interval (block).
  This models a control that is physically piecewise-constant — a thruster that
  fires at a fixed level over a segment, for instance — and changes the
  transcription's bookkeeping accordingly (the defect block carries a single
  block control value, reflected in the `BlockDefectPath` region).

Conceptually, the control mode decides how many control decision variables there
are and how they relate across an interval. A smoother parameterization has fewer
effective degrees of freedom and tends to regularize the solution; a looser one
(`NoSpline`) can capture bang-bang or discontinuous controls but gives the
optimizer more to chew on. The mode is independent of the transcription order: you
can pair a high-order state polynomial with a piecewise-linear control.

## Mesh refinement

A single fixed mesh is a guess. You choose some number of intervals up front, but
you do not yet know whether they are placed where the trajectory needs them. If
the dynamics are nearly linear over most of the flight but turn sharply during a
short maneuver, a uniform mesh wastes intervals on the easy regions and
under-resolves the hard one. The defects can be driven to zero on a coarse mesh
and the *discrete* solution will look converged — yet the underlying collocation
polynomial may still be a poor approximation of the true continuous trajectory
between the nodes. The defects being zero says the polynomial solves the dynamics
*at the collocation points*; it says nothing about the error in between.

Tycho closes this gap with an **estimate–refine loop**, adaptive mesh refinement.
After a solve on the current mesh, the phase estimates the discretization error in
each interval, and if the worst (or aggregated) error exceeds a target tolerance,
it builds a finer mesh — adding intervals where the error is concentrated — and
re-solves. The loop repeats until the error tolerance is met or an iteration cap
is reached. The driving quantities are:

- **`MeshErrorEstimators`** — how the per-interval error is estimated.
  `DEBOOR` uses a de Boor-style estimate derived from the collocation residual /
  higher-order interpolation of the existing solution: cheap, because it reuses
  the solve you already did. `INTEGRATOR` instead re-integrates the dynamics with
  a reference numerical propagator and compares against the collocation
  polynomial: more expensive but a more independent measure of how far the
  discrete trajectory has drifted from a true one.
- **`MeshErrorAggregation`** — how the vector of per-interval errors is reduced to
  the scalar that the convergence test and the refinement distribution use.
  `MAX` is the conservative choice (the mesh is converged only when *every*
  interval is below tolerance); `AVG` and `GEOMETRIC` are averaged measures; and
  `ENDTOEND` compares a full re-integration of the trajectory end to end.

The loop is governed by a target tolerance (`mesh_tol`) and bounded by a maximum
iteration count and minimum/maximum segment counts, so it always terminates. A
common and deliberate pattern is to *solve* (feasibility only) on the first,
coarse mesh and switch to full *optimize* once the mesh is fine enough to trust —
cheap iterations early, accurate iterations late. The result is a mesh shaped by
the problem: dense where the trajectory is hard, sparse where it is easy, and
demonstrably accurate to the tolerance you asked for.

The conceptual payoff is that mesh refinement is what makes the discrete answer a
trustworthy stand-in for the continuous one. Collocation gives you a finite
problem; mesh refinement gives you confidence that the finite problem's solution
actually approximates the infinite-dimensional trajectory you set out to find.

## Where to go next

This page covered the conceptual pipeline: continuous problem → collocation mesh
and defects → sparse NLP → PSIOPT solve → mesh refinement. To go further:

- **Reference.** The complete API for phases, ODEs, transcription modes, control
  modes, and mesh settings lives in the
  {doc}`Python reference </reference/python/optimal_control>` and the
  {doc}`C++ reference </reference/cpp/optimal_control>`.
- **The dynamics layer.** Every defect, constraint, and objective in a phase is a
  VectorFunction; the {doc}`VectorFunction model </explanation/vector_function>`
  explains the differentiable, symbolic machinery the transcription is built on.
- **Tutorials.** For a guided, runnable build of a phase from an ODE through to a
  solved trajectory, see the {doc}`Tutorials </tutorials/index>`.

The two ideas to keep in mind are the ones this page opened and closed with: a
phase turns a continuous optimal-control problem into a sparse, differentiable
NLP by enforcing the dynamics as **defect constraints** on a collocation mesh,
and **mesh refinement** iterates that discretization until the discrete solution
faithfully represents the continuous trajectory. Everything else — transcription
order, control parameterization, error estimators — is detail in service of those
two ideas.
