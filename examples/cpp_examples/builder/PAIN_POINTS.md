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

## Pain Point 2: `define()` lambda only offers scalar accessors

**Severity: Medium** — `ODEArgsProxy` provides `XVar(i)`, `UVar(i)`, `PVar(i)`
(individual scalars) and `XVec()`, `UVec()`, `PVec()` (full group segments).
There is no way to get a sub-segment of the state vector, e.g., the first 3
elements of a 7-element state.

For simple ODEs (Brachistochrone: 3 states, HyperSens: 1 state), scalar
accessors are perfectly ergonomic. For ODEs with structured state vectors
(Delta3Launch: R(3)+V(3)+m = 7 states), you need sub-segments with vector
operations — which requires falling back to `from()` with `Arguments<N>()`.

**Suggestion:** Add `XVec(start, count)` or `XSeg(start, count)` to
`ODEArgsProxy` that returns a sub-segment of the state vector with proper VF
DSL support. Similarly for `USeg(start, count)`.

## Pain Point 3: OCP link constraints require index-based API

**Severity: Medium** — `OptimalControlProblem::addForwardLinkEqualCon()` takes
`shared_ptr<ODEPhaseBase>` and `VectorXi` index arrays. It does not know about
the Phase wrapper's named variables. This means multi-phase problems still
require manual index construction for phase linking:

```cpp
// Named variables used throughout, then suddenly:
auto link_vars = make_idx({0, 1, 2, 3, 4, 5, 7, 8, 9, 10});
ocp.addForwardLinkEqualCon(phase1.base_ptr(), phase4.base_ptr(), link_vars);
```

**Suggestion:** Add OCP-aware helpers that accept Phase wrappers and named
variables, e.g.:
```cpp
ocp.addForwardLinkEqualCon(phase1, phase4, {"R", "V", "t", "u"});
```

## Pain Point 4: `base_ptr()` required for OCP integration

**Severity: Low** — Every `ocp.addPhase()` call requires `phase.base_ptr()`
to extract the underlying `shared_ptr<ODEPhaseBase>`. This is a minor
ergonomic friction that could be avoided with an `addPhase(Phase&)` overload.

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
