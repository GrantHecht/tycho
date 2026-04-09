# C++ Builder API Pain Points

Documented during Phase 6 builder example implementation. These findings inform
design priorities for future Builder API iterations.

## Pain Point 1: Dynamic VF DSL operators are incomplete — RESOLVED

**Status: Fixed** — Three complementary fixes address this:

1. **Template segment accessors** (`head<3>()`, `segment<3>(start)`, `tail<3>()`)
   on dynamic `ODEArguments<-1,-1,-1>` produce segments with static output sizes
   (ORC=3), enabling composition with `cross()`, `normalized_power<N>()`, etc.

2. **Relaxed operator signatures** — `operator+`, `operator-`, `operator*`,
   `operator/`, and comparison operators now accept operands with mixed
   static/dynamic compile-time sizes. `static_assert` guards preserve
   compile-time error detection when both operands are fully static.

3. **`pack()` method** — `DenseFunctionBase::pack()` type-erases an expression
   into `GenericFunction<IR, OR>` (like Eigen's `.eval()`), giving users
   opt-in control over compile-time complexity for deeply nested expressions.

Template overloads `XVec<SZ>()`, `UVec<SZ>()`, `PVec<SZ>()` were also added
to `ODEArgsProxy` for use in the `define()` lambda path.

The Delta3Launch example now uses `ODEArguments<-1,-1,-1>` with template
segment accessors:

```cpp
auto args = ODEArguments(7, 3, 0);
auto R = args.head<3>();           // ORC=3
auto V = args.segment<3>(3);      // ORC=3
auto Vr = V + R.cross(omega);     // compiles — both ORC=3
```

## Pain Point 2: `define()` lambda only offers scalar accessors — RESOLVED

**Status: Fixed** — Added `XVec(start, count)`, `UVec(start, count)`, and
`PVec(start, count)` overloads to `ODEArgsProxy`. These return dynamic
sub-segments within each variable group (0-based indexing within the group).

Note: sub-segments are dynamic-sized, producing dynamic ORC.
Since pain point 1 is now resolved (operators accept mixed static/dynamic
sizes), sub-segments work with all operations including `cross()` and
`normalized_power()`. Template overloads `XVec<SZ>()`, `UVec<SZ>()`,
`PVec<SZ>()` are also available for compile-time-sized segments.

## Pain Point 3: OptimalControlProblem link constraints require index-based API — RESOLVED

**Status: Fixed** — Added `OptimalControlProblem` wrapper class that accepts Phase wrappers
directly and resolves named variables for link constraints:

```cpp
OptimalControlProblem ocp;
ocp.addPhase(phase1);                                          // no base_ptr()
ocp.addForwardLinkEqualCon(phase1, phase4, {"R", "V", "t", "u"});  // named vars
ocp.optimizer().set_PrintLevel(1);
ocp.solve_optimize();
```

Use `ocp.base()` to access the underlying `OptimalControlProblem` for methods
not yet wrapped.

## Pain Point 4: `base_ptr()` required for OptimalControlProblem integration — RESOLVED

**Status: Fixed** — The `OptimalControlProblem` wrapper's `addPhase(Phase&)` accepts Phase
wrappers directly, eliminating all `base_ptr()` calls for the common workflow.

## Positive Findings

### Composable wind functions (Zermelo)

The Builder API's primary advantage shines in the Zermelo example. The static
version requires 4 separate ODE types (one per wind model) and a templated
`navigate<ODE>()` function. The builder version has:
- 4 factory functions returning `ODE`
- A single, non-templated `navigate(ODE&, ...)` function
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
