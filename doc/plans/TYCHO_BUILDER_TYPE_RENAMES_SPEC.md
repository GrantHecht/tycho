# Spec: Builder API Type Renames

> Promote the builder API types to primary C++ entry points with natural names.

## Motivation

The builder API (`RuntimeODE`, `OCP`, `Phase`) is the primary C++ interface for
defining optimal control problems. The current names reflect implementation history
(runtime vs static) rather than user intent. Before writing ~20 new C++ examples,
rename the types so all new code uses the final API names.

## Renames

| Current Name | New Name | File |
|---|---|---|
| `ODE<Derived, SX, CX, PX>` | `StaticODE<...>` | `include/tycho/detail/optimal_control/phase/ode.h` |
| `ODE_Expression<...>` | `StaticODE_Expression<...>` | same |
| `ODE_DerivModeWrapper<...>` | `StaticODE_DerivModeWrapper<...>` | same |
| `RuntimeODE` | `ODE` | `include/tycho/detail/optimal_control/builder/runtime_ode.h` |
| `ODEBuilder` | `ODEBuilder` | (no change) |
| `OptimalControlProblem` (base) | `OptimalControlProblemBase` | `include/tycho/detail/optimal_control/phase/optimal_control_problem.h` |
| `OCP` (builder wrapper) | `OptimalControlProblem` | `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` |
| `Phase` (builder wrapper) | `Phase` | (no change) |

## Backward Compatibility

### Deprecated aliases

No deprecated `using` aliases for the static DSL `ODE<>` template. Reason: a
deprecated `ODE` alias in `tycho::oc` would collide with the new builder `ODE`
in `tycho` whenever both namespaces are imported (common in tests, benchmarks,
and examples). Since the static DSL types are internal infrastructure (not a
user-facing API), a clean rename with no alias is appropriate.

For `OptimalControlProblemBase` (the renamed base struct): no deprecated alias
either, for the same collision reason — the builder wrapper takes the
`OptimalControlProblem` name.

For `RuntimeODE`: provide a deprecated alias `using RuntimeODE = ODE;` in
`runtime_ode.h` since there is no collision risk (RuntimeODE is unique).

### Python API stability

The Python-facing name `oc.OptimalControlProblem` must NOT change. The nanobind
binding string in `optimal_control_problem_bind.cpp` stays as
`"OptimalControlProblem"` — it binds the renamed C++ `OptimalControlProblemBase`
to the existing Python name. No Python user code changes needed.

### Macros

The `BUILD_ODE_FROM_EXPRESSION`, `BUILD_ODE_FROM_EXPRESSION_FD`, and
`BUILD_ODE_FROM_EXPRESSION_FWAD` macro names stay unchanged — only their
internal expansion updates from `ODE_Expression`/`ODE_DerivModeWrapper` to
`StaticODE_Expression`/`StaticODE_DerivModeWrapper`. These macros are used in
20+ files; renaming them would be churn with no user benefit.

### Header filenames

Header filenames stay as-is (`runtime_ode.h`, `ocp_wrapper.h`,
`optimal_control_problem.h`, `ode.h`). Renaming them would cascade into every
`#include` directive and risks collisions. The internal filename is not
user-facing.

## Scope

### Files to modify (rename + update references)

**Headers (definitions):**
- `include/tycho/detail/optimal_control/phase/ode.h` — rename struct templates
- `include/tycho/detail/optimal_control/builder/runtime_ode.h` — rename class
- `include/tycho/detail/optimal_control/phase/optimal_control_problem.h` — rename struct
- `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` — rename class

**Headers (references — builder subsystem):**
- `include/tycho/detail/optimal_control/builder/ode_builder.h` — returns RuntimeODE
- `include/tycho/detail/optimal_control/builder/phase_wrapper.h`
- `include/tycho/detail/optimal_control/builder/var_registry.h` — comments

**Headers (references — OC core):**
- `include/tycho/detail/optimal_control/phase/ode_phase.h`
- `include/tycho/detail/optimal_control/phase/ode_phase_base.h`
- All other OC headers that reference these types

**Headers (references — outside OC):**
- `include/tycho/detail/astro/kepler_model.h` — `using oc::ODE_Expression`, inherits from it

**Source (implementation):**
- `src/optimal_control/runtime_ode.cpp` — RuntimeODE method definitions + error messages

**Source (bindings):**
- `src/bindings/optimal_control/*.cpp` — type references
- `src/bindings/optimal_control/ode_bind.h` — `ODE_ExpressionBuild` references
- `src/bindings/optimal_control/optimal_control_problem_bind.cpp` — bind C++ `OptimalControlProblemBase` to Python name `"OptimalControlProblem"`

**Examples:**
- `examples/cpp_examples/builder/*/main.cpp` (4 files)
- `examples/cpp_examples/static/*/main.cpp` (4 files)

**Tests:**
- `tests/cpp/optimal_control/*.cpp` (including test_control_modes, test_builder_brachistochrone, test_ode_builder)
- `tests/cpp/vector_functions/test_ode_syntax_prototypes.cpp`
- `tests/cpp/vector_functions/test_multi_param_ode.cpp`
- `tests/cpp/test_utils.h` — `BUILD_ODE_FROM_EXPRESSION(BrachODE, ...)`
- `tests/cpp/integrators/integrator_test_utils.h` — `BUILD_ODE_FROM_EXPRESSION(SHO, ...)`

**Benchmarks:**
- `bench/cpp/bench_odes.h` — `BUILD_ODE_FROM_EXPRESSION` uses
- `bench/cpp/bench_phases.h` — `BUILD_ODE_FROM_EXPRESSION` uses

**Documentation:**
- `doc/Bindings.md` — references to `ODE_ExpressionBuild`, `ODE_Expression`

**Error message strings:**
- `runtime_ode.h` — 8 occurrences of `"RuntimeODE:"` prefix -> `"ODE:"`
- `ocp_wrapper.h` — occurrences of `"OCP::"` prefix -> `"OptimalControlProblem::"`

### Non-goals (excluded from scope)

- Historical plan documents (`doc/plans/*.md`) — describe past decisions, not live code
- `PAIN_POINTS.md` files — historical record
- Python-facing API names — stay unchanged

## Verification

1. `ninja -j8 all` compiles cleanly
2. `ctest --output-on-failure` — all C++ tests pass
3. `conda run -n tycho python -c "import tychopy"` — Python import works
4. All 4 existing C++ builder examples build and converge
5. All 4 existing C++ static DSL examples build and converge
6. `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"` — all 38 Python examples pass
