---
sd_hide_title: true
---

# Tycho

```{image} _static/tycho_transfer_named_stacked_light.svg
:alt: Tycho
:width: 220px
:align: center
:class: only-light sd-pt-4
```

```{image} _static/tycho_transfer_named_stacked_dark.svg
:alt: Tycho
:width: 220px
:align: center
:class: only-dark sd-pt-4
```

```{div} sd-text-center sd-fs-4 sd-text-muted sd-pt-3
Trajectory design and optimal control in C++ and Python
```

```{div} sd-text-center sd-pt-2 sd-pb-1 tycho-lead
A high-performance library for general optimal-control problems and space
trajectory optimization. Problems are defined with composable, self-
differentiating **VectorFunctions**, transcribed by **direct collocation**, and
solved with the bundled **PSIOPT** interior-point optimizer.
```

::::{div} sd-text-center sd-pt-3 sd-pb-2

```{button-ref} getting_started/index
:ref-type: doc
:color: primary
:shadow:
:class: sd-px-4 sd-me-2
Get started
```

```{button-ref} tutorials/examples/index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4 sd-me-2
Worked examples
```

```{button-ref} reference/index
:ref-type: doc
:color: secondary
:outline:
:class: sd-px-4
API reference
```

::::

---

## Tycho at a glance

A low-thrust spacecraft spiralling from one circular orbit to another in minimum
time — defined, transcribed, and solved in one short script. Write the dynamics
as a self-differentiating VectorFunction, transcribe them onto a collocation
mesh, bound the thrust, pin the orbits, and let PSIOPT do the rest. Exact
Jacobians and Hessians are generated for you.

```python
import numpy as np
import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments

# Dynamics: two-body gravity + bounded thrust
class LowThrust(oc.ODEBase):
    def __init__(self, mu=1.0, acc=0.02):
        a = oc.ODEArguments(6, 3)
        r, v, u = a.head3(), a.segment3(3), a.tail3()
        accel = r.normalized_power3() * (-mu) + u * acc
        super().__init__(vf.stack([v, accel]), 6, 3)

ode = LowThrust(mu=1.0)

# Circular start (r=1) and target (r=2) states
X0 = np.zeros(7); X0[0], X0[4] = 1.0, 1.0
Xf = np.zeros(6); Xf[0], Xf[4] = 2.0, np.sqrt(0.5)

# Spiral guess from a tangential-thrust propagation
XIG = np.zeros(10); XIG[:7] = X0
integ = ode.integrator(0.01, Args(3).normalized() * 0.8, [3, 4, 5])
guess = integ.integrate_dense(XIG, 6.4 * np.pi, 100)

# Transcribe, bound |thrust|<=1, minimize transfer time
phase = ode.phase("LGL3", guess, 256)
phase.add_boundary_value("Front", range(0, 7), X0)
phase.add_lu_norm_bound("Path", [7, 8, 9], 0.001, 1, 1.0)
phase.add_boundary_value("Back", range(0, 6), Xf)
phase.add_delta_time_objective(1.0)

phase.optimize()              # PSIOPT
traj = phase.return_traj()    # reaches r=2 in ~18.26 time units
```

```{eval-rst}
.. plot:: _plots/landing_low_thrust.py
```

```{div} sd-text-center sd-pt-1 sd-pb-2
See the full write-up in the
{doc}`low-thrust worked example </tutorials/examples/simple_low_thrust>`, or
start from the {doc}`first-phase tutorial </tutorials/basics/your_first_phase>`.
```

## Key features

::::{grid} 1 2 2 3
:gutter: 3
:class-container: sd-pt-2

:::{grid-item-card} {octicon}`pulse;1.4em;sd-mr-1` Self-differentiating dynamics
Define dynamics, constraints, and objectives as composable **VectorFunctions**.
Exact Jacobians and Hessians are produced automatically — no finite differences,
no hand-coded derivatives.
:::

:::{grid-item-card} {octicon}`graph;1.4em;sd-mr-1` Direct collocation
High-order Legendre–Gauss–Lobatto transcription turns continuous optimal-control
problems into sparse NLPs, with adaptive mesh refinement to certify the result.
:::

:::{grid-item-card} {octicon}`zap;1.4em;sd-mr-1` PSIOPT optimizer
A bundled, high-performance sparse interior-point solver built for the large,
structured nonlinear programs that trajectory optimization produces.
:::

:::{grid-item-card} {octicon}`rocket;1.4em;sd-mr-1` Astrodynamics-ready
Built-in dynamical models, reference frames, ephemeris access, and Lambert /
Kepler tooling make space-trajectory design a first-class use case.
:::

:::{grid-item-card} {octicon}`link;1.4em;sd-mr-1` Multi-phase problems
Link phases for staged launches, flybys, and coast arcs — arbitrarily complex
scenarios are assembled into a single NLP and solved together.
:::

:::{grid-item-card} {octicon}`code-square;1.4em;sd-mr-1` C++ and Python
A modern C++20 core with a first-class Python API. Prototype interactively in
Python; deploy the identical models in compiled C++.
:::

::::

## Explore the documentation

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`milestone;1.2em;sd-mr-1` Getting Started
:link: getting_started/index
:link-type: doc
Install Tycho into a conda environment, then write your first program in a few
lines.
:::

:::{grid-item-card} {octicon}`mortar-board;1.2em;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc
Hands-on walkthroughs — build a VectorFunction, propagate an ODE, solve your
first phase.
:::

:::{grid-item-card} {octicon}`checklist;1.2em;sd-mr-1` How-to Guides
:link: how_to/index
:link-type: doc
Short, problem-first recipes for common tasks, from mesh refinement to
threading.
:::

:::{grid-item-card} {octicon}`book;1.2em;sd-mr-1` Reference
:link: reference/index
:link-type: doc
The complete Python and C++ API across every subsystem.
:::

:::{grid-item-card} {octicon}`light-bulb;1.2em;sd-mr-1` Explanation
:link: explanation/index
:link-type: doc
The concepts and theory — VectorFunctions, direct collocation, astrodynamics.
:::

:::{grid-item-card} {octicon}`git-pull-request;1.2em;sd-mr-1` Contributing
:link: contributing/index
:link-type: doc
Build from source, the style guides, and testing conventions.
:::

::::

## Built on ASSET

Tycho is an independently maintained fork of
[**ASSET** (Astrodynamics Software and Science Enabling Toolkit)](https://github.com/AlabamaASRL/asset_asrl),
originally developed by the Astrodynamics and Space Research Laboratory at the
University of Alabama. Full credit and thanks are owed to the original authors
for the foundation this project builds upon. Original ASSET development was
funded by NASA under Grant No. 80NSSC19K1643.

If you use Tycho in published work, please cite the original ASSET paper:

```{div} sd-fs-6
Pezent, J. B., Sikes, J., Ledbetter, W., Sood, R., Howell, K. C., & Stuart, J. R.
(2022). *ASSET: Astrodynamics Software and Science Enabling Toolkit.* AIAA
SciTech 2022 Forum, AIAA 2022-1131.
[doi:10.2514/6.2022-1131](https://doi.org/10.2514/6.2022-1131)
```

```{toctree}
:hidden:
:maxdepth: 1

getting_started/index
tutorials/index
how_to/index
reference/index
explanation/index
contributing/index
```
</content>
