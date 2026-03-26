# C++ Builder API Pain Points

Documented during Phase 6 builder example implementation. These findings inform
design priorities for future Builder API iterations.

## Pain Point 1: Dynamic VF DSL operators are incomplete

**Severity: High** — `ODEArguments<-1,-1,-1>` produces `Segment<-1,-1,-1>` types
that do not support all VF DSL operators. Specifically, `operator+` between a
dynamic Segment and a CrossProduct does not compile:

```cpp
auto args = ODEArguments<-1, -1, -1>(7, 3, 0);
auto R = args.segment(0, 3);
auto V = args.segment(3, 3);
auto omega = Constant<-1, 3>(11, omega_val);
auto Vr = V + R.cross(omega);  // COMPILE ERROR
```

This means any ODE using vector operations (cross products, normalized vectors,
norm-based expressions composed with arithmetic) must fall back to
`Arguments<N>()` with compile-time segments, losing the dynamic-sizing
advantage of the Builder API.

**Impact:** The Delta3Launch example's `make_rocket_ode()` had to use
`Arguments<11>()` instead of `ODEArguments<-1,-1,-1>()`, which means the
expression template tree is still statically typed. The Builder API's `from()`
still helps (no `BUILD_ODE_FROM_EXPRESSION` macro, named variables, runtime
parameterization), but the compile cost from `Arguments<11>` remains.

**Workaround:** Use `Arguments<N>()` with compile-time `.head<K>()`,
`.segment<S,K>()`, `.coeff<I>()` for complex vector ODEs, and reserve
`ODEBuilder::define()` with `ODEArgsProxy` for simpler scalar ODEs.

## Pain Point 2: `define()` lambda only offers scalar accessors — RESOLVED

**Status: Fixed** — Added `XVec(start, count)`, `UVec(start, count)`, and
`PVec(start, count)` overloads to `ODEArgsProxy`. These return dynamic
sub-segments within each variable group (0-based indexing within the group).

Note: sub-segments are still dynamic-sized (`Segment<-1,-1,-1>`), so they
have the same operator limitations as pain point 1. For simple operations
(`.coeff(i)`, scalar arithmetic), they work well. For vector operations
(`cross`, `normalized_power`), `from()` with `Arguments<N>()` is still needed.

## Pain Point 3: OCP link constraints require index-based API — RESOLVED

**Status: Fixed** — Added `OCP` wrapper class that accepts Phase wrappers
directly and resolves named variables for link constraints:

```cpp
OCP ocp;
ocp.addPhase(phase1);                                          // no base_ptr()
ocp.addForwardLinkEqualCon(phase1, phase4, {"R", "V", "t", "u"});  // named vars
ocp.optimizer().set_PrintLevel(1);
ocp.solve_optimize();
```

Use `ocp.base()` to access the underlying `OptimalControlProblem` for methods
not yet wrapped.

## Pain Point 4: `base_ptr()` required for OCP integration — RESOLVED

**Status: Fixed** — The `OCP` wrapper's `addPhase(Phase&)` accepts Phase
wrappers directly, eliminating all `base_ptr()` calls for the common workflow.

## Positive Findings

### Composable wind functions (Zermelo)

The Builder API's primary advantage shines in the Zermelo example. The static
version requires 4 separate ODE types (one per wind model) and a templated
`navigate<ODE>()` function. The builder version has:
- 4 factory functions returning `RuntimeODE`
- A single, non-templated `navigate(RuntimeODE&, ...)` function
- Each wind model composed at runtime with captured parameters

This is a clear ergonomic win: **4 ODE types + template → 4 factory functions + 1 plain function**.

### Named variables eliminate index errors

Every constraint, objective, and boundary condition uses string names:
```cpp
phase.addBoundaryValue(PhaseRegionFlags::Front, {"x", "y", "v", "t"}, vals);
phase.addLUVarBound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);
phase.addDeltaTimeObjective(1.0);
```

Compare to the static version:
```cpp
Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);
phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
```

### var_group() for structured state vectors

The Delta3Launch example uses groups effectively:
```cpp
.var_group("R", 0, 3)
.var_group("V", 3, 3)
.var_group("u", 8, 3)
```

Then constraints use group names:
```cpp
phase.addLUNormBound(PhaseRegionFlags::Path, {"u"}, 0.5, 1.5);
phase.addLowerNormBound(PhaseRegionFlags::Path, {"R"}, Re);
phase.addEqualCon(PhaseRegionFlags::Back, target_orbit_fn, {"R", "V"});
```

### Simple ODEs are trivially clean

Brachistochrone and HyperSens are the ideal Builder API use cases:
```cpp
auto ode = ODEBuilder(1, 1)
    .define([](auto& args) {
        return args.UVar(0) - args.XVar(0);
    })
    .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
    .build();
```

No macros, no structs, no template parameters.
