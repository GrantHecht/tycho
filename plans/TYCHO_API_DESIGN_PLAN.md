# Tycho C++ API Design Plan

**Date:** 2026-03-16
**Status:** Approved

## Motivation

Tycho's Python bindings (`tychopy`) are the primary interface today, but the C++ API
should be an equally viable way to use the library — offering higher performance at
the expense of requiring compilation and stricter syntax. Currently, the C++ API is
difficult to use, incomplete, and not packaged for external consumption.

This plan addresses three goals:

1. **Establish Tycho as a proper C++ library** — installable, linkable, with clean
   public headers and a well-defined API surface.
2. **Provide two C++ ODE definition paths** — a runtime "builder" API matching
   Python's ergonomics, and an improved static DSL for maximum compile-time
   performance.
3. **Rename the Python package** from `tycho` to `tychopy`, so that "tycho" refers
   to the C++ library itself.

---

## Phase 1: Python Package Rename (`tycho` -> `tychopy`)

### Scope

Full rename of the Python package, nanobind module, and all references. Clean break
with no compatibility shim (no external users yet).

### Changes

| Component | Before | After |
|-----------|--------|-------|
| Python package directory | `tycho/` | `tychopy/` |
| nanobind module | `_tycho` | `_tychopy` |
| Python import | `import tycho` | `import tychopy as typy` |
| CMake target | `_tycho` | `_tychopy` |
| `pyproject.toml` | package name `tycho` | package name `tychopy` |

### Details

- All 38 Python examples, test scripts, and `scripts/run_examples.py` updated.
- Standard import convention: `import tychopy as typy`.
- Submodule access: `typy.VectorFunctions`, `typy.OptimalControl`, `typy.Astro`, etc.
- `CLAUDE.md` and documentation updated to reflect new naming.
- PCH targets and internal CMake names for bindings stay as build-internal details.

### What stays unchanged

- C++ namespace (`Tycho` — renamed in phase 2).
- C++ source directory (`src/`).
- C++ library targets (`tycho_core`, `psiopt`, etc. — renamed in phase 2-3).

### Validation

- All 38 Python examples pass with `import tychopy as typy`.
- `python scripts/run_examples.py` exits 0.
- C++ brachistochrone example still converges.

---

## Phase 2: Public Header Separation + Namespace Migration

### Scope

Separate public API headers from implementation details and create a proper
`include/tycho/` header tree.

### Public header layout

```
include/
  tycho/
    tycho.h                           # umbrella header (includes all below)
    vector_functions.h                # VF DSL public API
    optimal_control.h                 # ODE, phase, OCP public API
    solvers.h                         # PSIOPT configuration + convergence
    integrators.h                     # Integrator<ODE> public API
    astro.h                           # astrodynamics public API
    utils.h                           # public utilities (thread pool, etc.)
    typedefs.h                        # Eigen type aliases
    detail/                           # template implementation bodies
      vector_functions_impl.h
      optimal_control_impl.h
      integrators_impl.h
      ...
```

### Public vs. private split

| Module | Public (`include/tycho/`) | Private (stays in `src/`) |
|--------|---------------------------|---------------------------|
| VectorFunctions | `Arguments`, common functions (sin, cos, stack, etc.), `GenericFunction`, `ODEArguments`, operator overloads, expression types | Autodiff internals, `INPUT_DOMAIN` machinery, super-scalar dispatch details |
| OptimalControl | `ODE`, `ODESize`, `ODEPhase` (via base), `OptimalControlProblem`, enums (`TranscriptionModes`, `PhaseRegionFlags`, `ControlModes`) | Collocation internals, FD derivative implementations, mesh refinement internals |
| Solvers | `PSIOPT` (configuration + convergence flags), `OptimizationProblem` | NLP structure, KKT system internals, linear solver backends |
| Integrators | `Integrator<ODE>`, RK method names | RK coefficient tables, stepper internals |
| Astro | Kepler utilities, Lambert solvers, CR3BP, MEE dynamics, frame conversions | Internal propagator details |
| Utils | Thread pool (`set_num_threads`, `get_num_threads`, `parallel_blocks`, `parallel_task`), `FlatMap`, type aliases | Memory management, color output, internal helpers |

### Template handling

Tycho is heavily templated. The strategy:

- **`detail/` headers** — Public headers in `include/tycho/` declare the API.
  Template implementation bodies live in `include/tycho/detail/` and are included
  at the bottom of the public header. The `detail/` convention signals "don't use
  directly." This is the primary strategy for most of the codebase.
- **Explicit instantiation** — For non-template types (e.g., `PSIOPT`), the public
  header contains only declarations; implementations stay in `src/` as `.cpp` files.

### Library target rename

- `tycho_core` -> `tycho` (the main library target).
- CMake namespace for consumers: `tycho::tycho`.
- Internal build targets (`psiopt`, `optimalcontrol`, `astro`, `utils`,
  `vectorfunctions`) remain as build-internal — only `tycho` is exported.

### CMake changes

- `include/` added to public include path for the `tycho` target.
- External consumers see `#include <tycho/tycho.h>`.
- Internal code in `src/` continues using internal paths during build.

### Validation

- All C++ tests pass.
- All 38 Python examples pass.
- C++ brachistochrone example converges.
- External consumer test: a minimal project that does `#include <tycho/tycho.h>`
  and links `tycho::tycho` compiles and runs.

---

## Phase 3: CMake Library Packaging

### Scope

Make Tycho consumable by external C++ projects via both
`FetchContent`/`add_subdirectory` and `find_package` after installation.

### `FetchContent` / `add_subdirectory` support (first)

- Guard `project()` to work as a subdirectory.
- Expose `tycho::tycho` as an alias target.
- Public include path via `$<BUILD_INTERFACE:...>` generator expression.
- `TYCHO_MASTER_PROJECT` variable (true when Tycho is the top-level project)
  controls defaults:
  - `BUILD_CPP_TESTS` defaults OFF when not master project.
  - `BUILD_CPP_BENCHMARKS` defaults OFF when not master project.
  - `ENABLE_PYTHON_BINDINGS` defaults OFF when not master project.
  - `BUILD_CPP_EXAMPLES` defaults OFF when not master project.
- Vendored deps in `dep/` work naturally in this mode.

### `install()` + `find_package` support (second)

- `install(TARGETS tycho EXPORT tychoTargets ...)` for the library.
- `install(DIRECTORY include/tycho DESTINATION include)` for public headers.
- Generate and install `tychoConfig.cmake` + `tychoConfigVersion.cmake` via
  `CMakePackageConfigHelpers`.
- Version compatibility: `SameMajorVersion`.

### Dependency handling

**Eigen is the only dependency that matters for C++ consumers.** The others
(fmt, autodiff, nanobind) are implementation details statically linked into
Tycho, with no symbols exposed in public headers.

Strategy: **Hybrid** — bundled Eigen by default, `TYCHO_USE_SYSTEM_EIGEN=ON`
to use a system-installed Eigen instead.

When system Eigen is used, the generated `tychoConfig.cmake` includes:
```cmake
include(CMakeFindDependencyMacro)
find_dependency(Eigen3)
```

When bundled, the Eigen headers are installed alongside Tycho's headers.

### Minimal consumer example

```cmake
# Via FetchContent
include(FetchContent)
FetchContent_Declare(tycho
    GIT_REPOSITORY https://github.com/GrantHecht/tycho.git
    GIT_TAG main
)
FetchContent_MakeAvailable(tycho)
target_link_libraries(my_app PRIVATE tycho::tycho)

# Via installed package
find_package(tycho REQUIRED)
target_link_libraries(my_app PRIVATE tycho::tycho)
```

### Validation

- A standalone test project successfully builds and links via `FetchContent`.
- `cmake --install` followed by `find_package(tycho)` in a separate project works.
- Both bundled and system Eigen paths tested.

---

## Phase 4: C++ Examples (Static DSL)

### Scope

Implement three new C++ examples using the current static VF DSL, mirroring their
Python counterparts. These serve dual purpose: validating the C++ API end-to-end
and surfacing pain points that inform phases 5-7.

### Examples

**4a. Zermelo (`examples/cpp_examples/static/zermelo/`)**
- 2-state, 1-control, time-dependent ODE.
- Wind functions as composable VectorFunctions (template the ODE on wind type or
  use `GenericFunction` for runtime dispatch).
- Exercises: `LGL3` collocation, `solve_optimize`, boundary values, variable bounds,
  delta-time objective.
- Demonstrate at least 2 wind models (uniform + variable-direction).

**4b. Delta3Launch (`examples/cpp_examples/static/delta3_launch/`)**
- 4-phase launch vehicle, 7 states + 3 controls.
- Exercises: `OptimalControlProblem`, `addPhase`, `addForwardLinkEqualCon`, adaptive
  mesh refinement, custom VectorFunction constraints (`TargetOrbit`), mass staging,
  multiple ODE instances with different parameters.
- Most complex example — will heavily stress the multi-phase API and expose the
  manual index-tracking pain point.
- Template cost: `Arguments<11>` (~8-10 GB based on scaling data in TODO.md).

**4c. HyperSensLong (`examples/cpp_examples/static/hypersens/`)**
- 1-state, 1-control ODE (trivially small).
- Exercises: integral objectives (`addIntegralObjective`), extensive mesh refinement
  configuration (`setMeshTol`, `setMeshErrorEstimator`, `setMeshErrorCriteria`,
  `setMeshIncFactor`, `setMeshRedFactor`, `setMeshErrFactor`), `MINDEG` ordering,
  `MeshConverged` flag.

### Structure

Examples are **standalone projects** that consume Tycho via `find_package(tycho)`.
This validates the phase 3 packaging and shows users how to set up their own projects.

```
examples/cpp_examples/
  static/
    brachistochrone/
      CMakeLists.txt          # standalone: find_package(tycho REQUIRED)
      main.cpp
    zermelo/
      CMakeLists.txt
      main.cpp
    delta3_launch/
      CMakeLists.txt
      main.cpp
    hypersens/
      CMakeLists.txt
      main.cpp
```

Each example's `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.20)
project(zermelo)
find_package(tycho REQUIRED)
add_executable(zermelo main.cpp)
target_link_libraries(zermelo PRIVATE tycho::tycho)
```

The existing brachistochrone example moves into `static/brachistochrone/`.

For Tycho's own CI/testing, the top-level `CMakeLists.txt` can optionally build
these in-tree via the `BUILD_CPP_EXAMPLES` flag.

### Validation

- Each example converges to the same solution as its Python counterpart.
- Pain points encountered during implementation documented as comments in the code.
- Compile times and peak memory usage recorded for each example.

---

## Phase 5: Builder API (Runtime ODE Interface)

### Scope

A C++ equivalent of Python's `ODEBase` — ergonomic ODE definition with
string-named variables, runtime construction, and convenience methods. Uses
`GenericFunction<-1,-1>` under the hood.

### Design principles

- **Zero cost when not used.** The builder and named-variable features are layered
  on top of the existing API. Users who don't use names pay no overhead.
- **Two-tier phase API:**
  - `ODEPhaseBase` / `ODEPhase<DODE>` — existing, unchanged, index-based only.
  - `Tycho::Phase` — wrapper that adds named-variable overloads, holds a variable
    registry. Named overloads resolve strings to indices at call time (map lookup),
    then delegate to the underlying index-based methods.

### `Tycho::ODEBuilder`

```cpp
// Lambda form (inline definition)
auto ode = Tycho::ODEBuilder(/*XVars=*/3, /*UVars=*/1)
    .define([](auto& args) {
        auto [x, y, v] = args.XVec();
        auto theta = args.UVar(0);
        return Tycho::stack(
            Tycho::sin(theta) * v,
            -1.0 * Tycho::cos(theta) * v,
            9.81 * Tycho::cos(theta)
        );
    })
    .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 3}})
    .build();

// Pre-built expression form
auto XtU = Tycho::ODEArguments(3, 1);
// ... build expression ...
auto ode = Tycho::ODEBuilder(3, 1)
    .from(ode_expr)
    .var_names(...)
    .build();
```

Both forms produce the same internal representation. The lambda is syntactic sugar.

### API surface

| Python (`ODEBase`) | C++ (`ODEBuilder` / `RuntimeODE`) |
|---------------------|-----------------------------------|
| `ode.phase("LGL3", traj, 32)` | `ode.phase(TranscriptionModes::LGL3, traj, 32)` |
| `ode.make_input(p=Lstar, w=1.0)` | `ode.make_input({{"p", Lstar}, {"w", 1.0}})` |
| `ode.make_units(p=Lstar, t=Tstar)` | `ode.make_units({{"p", Lstar}, {"t", Tstar}})` |
| `ode.integrator(step, ulaw, "MEEs")` | `ode.integrator(step, ulaw, "MEEs")` |
| `ode.get_vars(["w", "t"], state)` | `ode.get_vars({"w", "t"}, state)` |
| `Vgroups={"MEEs": MEEs}` | `.var_group("MEEs", {0, 5})` |

### Named-variable phase usage

```cpp
auto phase = ode.phase(TranscriptionModes::LGL5, initial_guess, 16);

// Named overloads (string lookup, then delegate to index-based)
phase.addBoundaryValue(PhaseRegionFlags::Front, {"MEEs", "w", "t"}, vals);
phase.addEqualCon(PhaseRegionFlags::Path, norm_constraint, "U");
phase.addLUFuncBound(PhaseRegionFlags::Path, rad_func, "MEEs", Re, 10*Re);

// Index overloads still work (direct passthrough, no lookup)
phase.addBoundaryValue(PhaseRegionFlags::Front, indices, values);
```

### API constants

All API constants (phase regions, transcription modes, control modes, mesh
estimators, etc.) use **enums only** — no string alternatives:

- `PhaseRegionFlags::Front`, `PhaseRegionFlags::Back`, `PhaseRegionFlags::Path`
- `TranscriptionModes::LGL3`, `TranscriptionModes::LGL5`
- `ControlModes::FirstOrderSpline`, `ControlModes::BlockConstant`
- etc.

Strings are used **only** for user-defined variable names.

### Internal representation

- `ODEBuilder::build()` returns a `Tycho::RuntimeODE` that wraps a
  `GenericODE<GenericFunction<-1,-1>, ...>` plus a variable name registry.
- `RuntimeODE` exposes the underlying `GenericODE` for users who want to
  drop down to the raw API:
  ```cpp
  auto& raw_ode = built.generic_ode();     // no names, no wrapper
  auto phase = built.phase(...);            // convenience, with names
  ```

### `Tycho::Phase` wrapper

```
ODEPhaseBase / ODEPhase<DODE>    (existing, unchanged, index-based only)
        ^
        | held by (shared_ptr)
Tycho::Phase                     (wrapper, adds named-variable overloads)
```

- Holds `std::shared_ptr<ODEPhaseBase>` + variable registry.
- Named overloads resolve strings -> `Eigen::VectorXi` indices, then call
  the underlying `ODEPhaseBase` methods.
- Index-based overloads are direct passthroughs with zero overhead.
- Exposes the underlying phase pointer for advanced use.

### Validation

- Builder API produces identical solutions to static DSL and Python for all
  test problems.
- Named-variable resolution is correct for single variables and variable groups.
- Index-based path through `Tycho::Phase` has zero overhead vs. using
  `ODEPhaseBase` directly.

---

## Phase 6: Builder API Examples

### Scope

Re-implement the four C++ examples using the builder API. Validates that the
runtime API is functionally complete and ergonomic.

### Examples

| Example | Key features exercised |
|---------|----------------------|
| Brachistochrone | Basic builder usage, named variables, boundary values |
| Zermelo | Composable wind functions via `from()`, multiple problem variants |
| Delta3Launch | `var_group()`, `OptimalControlProblem`, `make_input()`, `make_units()`, multi-phase |
| HyperSensLong | Integral objectives with named variable selection, mesh refinement |

### Structure

```
examples/cpp_examples/
  static/
    brachistochrone/
    zermelo/
    delta3_launch/
    hypersens/
  builder/
    brachistochrone/
      CMakeLists.txt
      main.cpp
    zermelo/
      CMakeLists.txt
      main.cpp
    delta3_launch/
      CMakeLists.txt
      main.cpp
    hypersens/
      CMakeLists.txt
      main.cpp
```

Builder examples are also standalone projects using `find_package(tycho)`,
identical pattern to the static examples.

### Validation

- Each builder example converges to the same solution as its static DSL and
  Python counterparts.
- API gaps discovered during implementation feed back into the builder API design
  before finalizing.

---

## Phase 7: Static DSL Improvements

### Scope

Address the five known pain points in the static VF DSL and explore cleaner ODE
definition syntax. This phase involves the most design iteration.

### Pain point 1: Template memory explosion

The Jacobian/Hessian template expansions create exponentially deep type trees.
Three mitigation strategies (all to be implemented as user-selectable options):

**A. Firebreak mechanism (new)**
```cpp
// Insert a type-erasure barrier to cap template depth
auto mee_dynamics = Tycho::barrier(raw_mee_expr);  // evaluates to GenericFunction
auto full_ode = Tycho::stack(mee_dynamics(mee_input), weight_dot);
```

`Tycho::barrier(expr)` evaluates `expr` into a `GenericFunction`, cutting the type
tree. Virtual dispatch at the barrier point trades a small runtime cost for
dramatically reduced template depth. Users place barriers at natural subexpression
boundaries (e.g., `MEEDynamics`, `ZonalGrav`).

**B. FD Jacobian mode as first-class option**

Make it trivially easy to select finite-difference Jacobians for an ODE. This
already exists in the VF infrastructure (`DenseDerivativeMode::ForwardFiniteDiff`)
but the ODE macros and definition patterns don't expose it cleanly. FD avoids the
template explosion entirely.

**C. Forward-mode AD option**

Surface the existing `autodiff::dual` forward-mode path as an easy toggle. Cheaper
template instantiation than the expression-level AD, still automatic derivatives.

### Pain point 2: `BUILD_ODE_FROM_EXPRESSION` single-parameter limitation

The macro currently only accepts a single constructor argument type.

**Fix:** Variadic macro — `BUILD_ODE_FROM_EXPRESSION(Name, Impl, T1, T2, ...)`
generates a constructor taking `(T1, T2, ...)`. Also fix the struct parameter
approach so `Input<Scalar>`/`Output<Scalar>` aliases are correctly generated.

### Pain point 3: No named variables in static DSL

Compile-time named variable accessors:
```cpp
struct MyODE : Tycho::StaticODE<3, 1, 0> {
    static constexpr auto x     = XVar<0>;
    static constexpr auto y     = XVar<1>;
    static constexpr auto v     = XVar<2>;
    static constexpr auto theta = UVar<0>;

    static auto Definition(double g) {
        auto args = Arguments<5>();
        return Tycho::stack(
            Tycho::sin(args[theta]) * args[v],
            ...
        );
    }
};
```

`XVar<I>` and `UVar<I>` are lightweight compile-time index tags. They provide
semantic meaning and eliminate raw index errors without any runtime cost.

### Pain point 4: `Scaled<Scaled<...>>` nesting bugs

Implementation fix in `OperatorOverloads.h`: detect `Scaled<Scaled<...>>` nesting
and collapse to a single `Scaled` with the combined coefficient. Purely internal
change, no API impact.

### Pain point 5: Verbose ODE definition ceremony

Currently requires: `struct Impl : ODESize<...>` + `static auto Definition(...)` +
`BUILD_ODE_FROM_EXPRESSION(Name, Impl, ...)`. Explore two alternatives:

**Option A — Single-struct, macro-free:**
```cpp
struct Brachistochrone : Tycho::StaticODE<3, 1, 0> {
    double g;
    Brachistochrone(double g) : g(g) {}

    auto ode() const {
        auto args = Arguments<5>();
        auto [x, y, v] = args.head<3>();
        auto theta = args[4];
        return Tycho::stack(
            Tycho::sin(theta) * v,
            -1.0 * Tycho::cos(theta) * v,
            g * Tycho::cos(theta)
        );
    }
};
```

**Option B — Refined macro:**
```cpp
TYCHO_ODE(Brachistochrone, XVars=3, UVars=1, Params=(double g)) {
    auto [x, y, v] = args.XVec();
    auto theta = args.UVar(0);
    return Tycho::stack(
        Tycho::sin(theta) * v,
        -1.0 * Tycho::cos(theta) * v,
        g * Tycho::cos(theta)
    );
};
```

**Iteration strategy:** Implement both options for the Brachistochrone ODE, compare
ergonomics, compile characteristics, and readability, then settle on one (or keep
both if they serve different use cases).

### Validation

- All static DSL examples still converge after improvements.
- Compile times and peak memory measured before and after.
- `Scaled<Scaled<...>>` expressions that previously failed now work correctly.
- Both ODE definition syntax options prototyped and evaluated.

---

## Phase 8: Refine Static DSL Examples

### Scope

Update the phase 4 static DSL examples to use the improvements from phase 7.

### Changes

- Rewrite each example using the settled-upon ODE definition syntax.
- Replace manual index constants with compile-time named variables (`XVar<I>`,
  `UVar<I>`).
- Apply firebreaks where template cost was problematic (Delta3Launch especially).
- Use fixed variadic `BUILD_ODE` macro or single-struct pattern for
  multi-parameter ODEs.
- Verify the `Scaled<Scaled<...>>` fix across all expressions.

### Validation

- All examples converge to the same solutions as before.
- Compile times and memory usage documented and compared against pre-improvement
  baselines from phase 4.
- The code reads cleanly and is suitable as canonical reference documentation.

---

## Phase 9: Future Work Backlog

A living list of follow-up work, to be updated as phases 1-8 progress.

### Near-term

- **BettsLowThrust C++ example** — The big integration test from TODO.md. Depends
  on control-law integration and multi-parameter ODE fixes. Candidate for both
  static DSL (with firebreaks) and builder API versions.
- **Control-law integration in C++** — Extend `Integrator<ODE>` to accept a
  control law VectorFunction for initial guess generation. Needed for BettsLT and
  any problem where linear interpolation doesn't converge.
- **C++ `OptimalControlProblem` builder** — Named-phase builder for multi-phase
  problems (builder API for the OCP itself, not just individual ODEs).

### Medium-term

- **Documentation** — Sphinx/Doxygen for the C++ API. The public header separation
  from phase 2 makes this straightforward.
- **CI/CD** — Automated testing of C++ examples, benchmarks, and Python examples
  on PR.
- **PyPI packaging for tychopy** — Publish the Python package.
- **Benchmarking builder vs. static DSL** — Quantify runtime performance gap to
  help users choose the right API for their problem.

### Longer-term

- **vcpkg / Conan packaging** — Package manager distribution for the C++ library.
- **Julia bindings (Tycho.jl)** — As mentioned.
- **Expression tree simplification** — Beyond the `Scaled<Scaled>` fix: general
  compile-time expression optimization passes.
- **Explicit Jacobian ODEs** — Allow hand-coded Jacobians/Hessians for users who
  want maximum control over derivative computation.
- **Additional static DSL pain points** — To be identified during phases 4 and 8.

---

## Summary

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| 1 | `tycho/` -> `tychopy/` rename | — |
| 2 | Public headers | Phase 1 |
| 3 | CMake packaging (FetchContent + install) | Phase 2 |
| 4 | C++ examples (static DSL): Zermelo, Delta3Launch, HyperSensLong | Phase 3 |
| 5 | Builder API (runtime ODE interface) | Phase 4 |
| 6 | Builder API examples | Phase 5 |
| 7 | Static DSL improvements (pain points 1-5) | Phase 4 |
| 8 | Refine static DSL examples | Phase 7 |
| 9 | Future work backlog (living document) | — |

Each phase produces a mergeable PR. Phases 5 and 7 can potentially proceed in
parallel since they address independent APIs, but phase 6 depends on phase 5 and
phase 8 depends on phase 7.
