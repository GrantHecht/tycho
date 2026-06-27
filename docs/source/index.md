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

:::{grid-item-card} User Guide
:link: user_guide/index
:link-type: doc
Concepts, hands-on tutorials, how-to recipes, and worked examples —
everything to learn and use Tycho.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc
Python and C++ API reference for the VectorFunction subsystem.
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
user_guide/index
reference/index
contributing/index
```
