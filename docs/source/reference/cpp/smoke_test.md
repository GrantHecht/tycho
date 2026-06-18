---
orphan: true
---

# Smoke Test (Pipeline Verification)

Minimal Breathe directives that exercise the Doxygen → Breathe → Sphinx
pipeline on different C++ symbol shapes (enum + templated function), so
that a Doxygen or Breathe version bump that breaks rendering fails CI
immediately rather than during content rollout.

```{eval-rst}
.. doxygenenum:: tycho::integrators::ErrorNormType
   :project: tycho

.. doxygenfunction:: tycho::integrators::check_state_finite_or_throw
   :project: tycho
```
