# Dynamic VF DSL Operator Completeness

**Date:** 2026-03-25
**Status:** Approved
**Pain Point:** PAIN_POINTS.md #1 — Dynamic VF DSL operators are incomplete
**Branch:** feat/builder-examples

## Problem

`ODEArguments<-1,-1,-1>` produces `Segment<-1,-1,-1>` types (ORC=-1) that cannot
compose with VF operations that produce static output sizes. Specifically,
`operator+` between a dynamic segment and a `FunctionCrossProduct` (ORC=3) fails
to compile because the operator template requires matching compile-time `OR`:

```cpp
// OperatorOverloads.h:202 — both sides must have same IR and OR
template <class Func1, int IR, int OR, class Func2>
operator+(DenseFunctionBase<Func1, IR, OR>, DenseFunctionBase<Func2, IR, OR>)
```

`Segment<-1,-1,-1>` has ORC=-1; `FunctionCrossProduct` has ORC=3. Template
deduction fails: `-1 != 3`.

This forces Builder API users to fall back to `Arguments<N>()` with compile-time
segments for any ODE involving vector operations (cross, normalized_power, etc.),
losing the dynamic-sizing advantage.

## Root Cause

The operator signature is unnecessarily restrictive. `TwoFunctionSum_Impl`
already handles mixed compile-time sizes internally via
`SZ_MAX<Func1::ORC, Func2::ORC>`, and validates actual sizes at runtime. The
bottleneck is purely the operator template signature, not the implementation.

`SZ_MAX` uses `SZ_MAXOP` semantics: if either argument is negative, the result
is -1. So mixed compositions produce fully dynamic results. The existing
runtime validation in `TwoFunctionSum_Impl`'s constructor (checking
`func1.IRows() == func2.IRows()` and `func1.ORows() == func2.ORows()`) is
sufficient to catch actual size mismatches — no changes needed there.

## Design

Four changes, ordered by dependency:

### Part A: Demonstrate existing template segment accessors

`DenseFunctionBase` already provides `segment<SZ>(start)`, `head<SZ>()`, and
`tail<SZ>()` (DenseFunctionBase.h:114-135). These produce segments with static
ORC, enabling composition with cross products and other fixed-output operations.

Update the `delta3_launch` builder example to use `ODEArguments<-1,-1,-1>` with
these template accessors instead of falling back to `Arguments<11>()`.

```cpp
auto args = ODEArguments<-1, -1, -1>(7, 3, 0);
auto R = args.head<3>();           // IRC=-1, ORC=3
auto V = args.segment<3>(3);      // IRC=-1, ORC=3
auto u = args.tail<3>().normalized();
auto Vr = V + R.cross(omega);     // ORC=3 == ORC=3 -> compiles
```

**Architecture:** Dynamic shell (IRC=-1 at root), static computation core
(ORC=3 on interior nodes). Eigen uses fixed-size operations internally
(segment<3>, middleCols<3>) for stack allocation and SIMD paths. Type erasure
at the RuntimeODE boundary wraps the statically-typed expression tree without
affecting internal evaluation.

### Part B: Relax binary operator template signatures

Replace the 6 affected operator signatures in `OperatorOverloads.h` to accept
independent compile-time sizes:

```cpp
// Before:
template <class Func1, int IR, int OR, class Func2>
operator+(DenseFunctionBase<Func1, IR, OR>, DenseFunctionBase<Func2, IR, OR>)

// After:
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
operator+(DenseFunctionBase<Func1, IR1, OR1>, DenseFunctionBase<Func2, IR2, OR2>)
```

**Static safety preserved via static_assert:** When both operands have fully
static (positive) sizes that disagree, a compile-time error is emitted:

```cpp
static_assert(OR1 == OR2 || OR1 < 0 || OR2 < 0,
              "Output size mismatch: both functions have static sizes that don't match");
static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
              "Input size mismatch: both functions have static sizes that don't match");
```

| LHS ORC | RHS ORC | Result |
|---------|---------|--------|
| 3 | 3 | Compiles (static match) |
| 3 | 4 | **Compile error** (static mismatch) |
| -1 | 3 | Compiles (dynamic + static allowed) |
| -1 | -1 | Compiles (both dynamic, runtime check) |

**Affected signatures — arithmetic operators (critical, 8 total):**

1. `operator+` — TwoFunctionSum (line 202): shared IR and OR
2. `operator+` — MultiFunctionSum 3-arg (line 208): IR/OR match between
   `DenseFunctionBase<Func3, IR, OR>` and the `TwoFunctionSum`'s deduced
   IRC/ORC (which is already computed via SZ_MAX internally)
3. `operator+` — MultiFunctionSum 4-arg (line 213): same pattern as #2
4. `operator-` — FunctionDifference (line 220): shared IR and OR
5. `operator*` — VFunc * SFunc (line 177): shared IR
6. `operator*` — SFunc * VFunc (line 182): shared IR
7. `operator*` — SFunc * SFunc, both OR=1 (line 187): shared IR
8. `operator/` — VFunc / SFunc (line 192): shared IR

For operators 5-8 (vector*scalar, vector/scalar), the OR values are already
independent (vector has OR, scalar has 1). Only the IR parameters need the
same relaxation treatment with static_assert guards.

**Affected signatures — comparison operators (secondary, 5 total):**

9-13. `operator<`, `<=`, `>`, `>=`, `==` (lines 229-258): shared IR between
      two scalar VFs. Both sides have OR=1 (hardcoded), so only IR can
      mismatch. In practice, both sides of a comparison typically originate
      from the same expression tree (matching IR), but the same relaxation
      should be applied for consistency. The VF-vs-double overloads (lines
      260-298) are unaffected — they create `Constant<IR, 1>` with the
      same IR as the left operand.

### Part C: Template overloads on ODEArgsProxy

Add `XVec<SZ>()`, `UVec<SZ>()`, `PVec<SZ>()` template overloads to
`ODEArgsProxy` so the `define()` lambda path can also produce statically-sized
segments:

```cpp
template <int SZ>
auto XVec(int start = 0) const {
    // runtime bounds validation
    return args_.segment<SZ>(start);
}

template <int SZ>
auto UVec(int start = 0) const {
    // runtime bounds validation
    return args_.segment<SZ>(args_.XtVars() + start);
}

template <int SZ>
auto PVec(int start = 0) const {
    // runtime bounds validation
    return args_.segment<SZ>(args_.XtUVars() + start);
}
```

No overload ambiguity: `XVec<3>(0)` (template + 1 arg) is distinct from
`XVec(0, 3)` (non-template + 2 args) and `XVec()` (no args).

### Part D: `pack()` method on DenseFunctionBase

Add a method to `DenseFunctionBase` that type-erases the expression into a
`GenericFunction<IR, OR>`, analogous to Eigen's `.eval()`:

```cpp
GenericFunction<IR, OR> pack() const {
    return GenericFunction<IR, OR>(this->derived());
}
```

This allows users to control compile-time complexity by cutting the expression
tree at intermediate points:

```cpp
auto Vr = (V + R.cross(omega)).pack();          // GenericFunction<-1, 3>
auto grav = ((-mu) * R.normalized_power<3>()).pack();  // GenericFunction<-1, 3>
```

**Properties:**
- Preserves compile-time IR/OR (deduced from the expression)
- Never required — purely opt-in for users hitting compile-time walls
- Trade-off: virtual dispatch overhead (~1-2 ns/call) vs reduced compile time/RAM

## Files Modified

| File | Change |
|------|--------|
| `include/tycho/detail/OperatorOverloads.h` | Relax 13 operator signatures + static_assert |
| `include/tycho/detail/DenseFunctionBase.h` | Add `pack()` method |
| `include/tycho/detail/ode_builder.h` | Add template XVec/UVec/PVec to ODEArgsProxy |
| `examples/cpp_examples/builder/delta3_launch/main.cpp` | Use ODEArguments<-1,-1,-1> with template accessors |
| `examples/cpp_examples/builder/PAIN_POINTS.md` | Update pain point 1 status |

## Testing

- All existing C++ unit tests must pass (no regressions from operator changes)
- All 38 Python examples must pass
- `delta3_launch` builder example must converge with the new ODEArguments approach
- C++ brachistochrone example must still converge
- New test coverage for mixed-size operator composition (static+dynamic)
- New test coverage for `pack()` correctness (value, Jacobian, adjoint-Hessian)
- New test coverage for ODEArgsProxy template overloads

## Notes

- The existing runtime size checks in `TwoFunctionSum_Impl`,
  `MultiFunctionSum_Impl`, `FunctionCrossProduct_Impl`, etc. are sufficient
  for catching actual size mismatches at construction time. No modifications
  to these runtime checks are needed.
- `GenericFunction<IR, OR>` is used throughout the codebase with various IR/OR
  combinations (e.g., `<-1, -1>`, `<-1, 1>` in Python bindings). The `pack()`
  method's return type `GenericFunction<IR, OR>` should work for all
  combinations, but this should be verified during implementation for less
  common static IR/OR pairs.

## Out of Scope

- Templating ODEBuilder or RuntimeODE (unnecessary; type erasure at the builder
  boundary is the correct design)
- Changes to GenericFunction, GFStorage, or GFTypeErasure internals
- Changes to SZ_MAX/SZ_MAXOP semantics
