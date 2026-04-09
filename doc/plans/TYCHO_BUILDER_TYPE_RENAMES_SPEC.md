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

- Add `using` aliases at the old names in the original headers, marked `[[deprecated]]`
- Python bindings: internal binding code references old names; update to new names
- Existing 4 C++ builder examples: update to new names
- Existing static DSL examples: update `ODE` -> `StaticODE`
- C++ tests referencing these types: update

## Scope

### Files to modify (rename + update references)

**Headers (definitions):**
- `include/tycho/detail/optimal_control/phase/ode.h` — rename struct templates
- `include/tycho/detail/optimal_control/builder/runtime_ode.h` — rename class
- `include/tycho/detail/optimal_control/phase/optimal_control_problem.h` — rename struct
- `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` — rename class

**Headers (references):**
- `include/tycho/detail/optimal_control/phase/ode_phase.h`
- `include/tycho/detail/optimal_control/phase/ode_phase_base.h`
- `include/tycho/detail/optimal_control/builder/phase_wrapper.h`
- All other OC headers that reference these types

**Source (bindings):**
- `src/bindings/optimal_control/*.cpp` — update type references

**Examples:**
- `examples/cpp_examples/builder/*/main.cpp` (4 files)
- `examples/cpp_examples/static/*/main.cpp` (4 files)

**Tests:**
- `tests/cpp/optimal_control/*.cpp`
- `tests/cpp/vector_functions/test_ode_syntax_prototypes.cpp`
- `tests/cpp/vector_functions/test_multi_param_ode.cpp`

**Python (if exposed):**
- `tychopy/OptimalControl/__init__.py` — check for exposed names

## Verification

1. `ninja -j8 all` compiles cleanly
2. `ctest --output-on-failure` — all C++ tests pass
3. `conda run -n tycho python -c "import tychopy"` — Python import works
4. All 4 existing C++ builder examples build and converge
5. `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"` — all 38 Python examples pass
