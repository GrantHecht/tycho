---
sd_hide_title: true
---

# Tycho

```{div} sd-text-center sd-fs-3 sd-font-weight-bold sd-pt-3
Tycho
```

```{div} sd-text-center sd-fs-5 sd-text-muted
Trajectory design and optimal control in C++ and Python
```

```{div} sd-text-center sd-pt-2 sd-pb-2
A high-performance library for general optimal-control problems and space
trajectory optimization. Problems are defined with composable, self-
differentiating **VectorFunctions**, transcribed by direct collocation, and
solved with the built-in **PSIOPT** interior-point solver.
```

::::{grid} 1 2 2 2
:class-container: sd-text-center sd-pt-4

:::{grid-item-card} Getting Started
:link: getting_started/index
:link-type: doc
Install Tycho into a conda environment, then write your first program — a
self-differentiating dynamics model in a few lines.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Hands-on walkthroughs. Build and differentiate your first VectorFunction.
:::

:::{grid-item-card} How-to Guides
:link: how_to/index
:link-type: doc
Short, problem-first recipes for common tasks — a growing collection.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc
Python and C++ API reference for the VectorFunction subsystem.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc
Concepts and theory, starting with the VectorFunction model.
:::

:::{grid-item-card} Contributing
:link: contributing/index
:link-type: doc
Build from source, style guides, testing conventions.
:::

::::

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
