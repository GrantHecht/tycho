# BumpAllocator Jacobian Bug

**Date discovered:** 2026-03-23
**Status:** Open — temporary workaround applied, root cause not fully resolved
**Affected code:** `include/tycho/detail/memory_management.h` (BumpAllocator)
**Symptom:** Wrong Jacobians in VectorFunction expressions containing sums of segments inside nested functions (e.g., `cos(2*(x+y))`)

## Summary

The BumpAllocator rewrite (commit `34fe739`) introduced a bug that causes
incorrect Jacobian computation in certain VectorFunction expression trees.
The bug produces correct **function values** but wrong **derivatives**,
leading to degraded solver convergence (10x more iterations) or convergence
to the wrong solution.

The original ASSET codebase (`asset_asrl`) does not exhibit this bug.

## How to Reproduce

```cpp
#include <tycho/tycho.h>
using namespace Tycho;

auto a = Arguments<2>();
auto x = a.coeff<0>();
auto y = a.coeff<1>();
auto ang = 2.0 * (x + y);
auto f = cos(ang);

GenericFunction<2, 1> gf(f);
Eigen::VectorXd tp(2);
tp << 0.2, -0.5;
Eigen::MatrixXd jac(1, 2);
gf.jacobian(tp, jac);

// Expected: [1.12928, 1.12928]
// Actual:   [1.12928, 3.12928]  ← dy is wrong by +2.0
```

The equivalent Python/asset_asrl code produces the correct Jacobian.

## Affected Expression Patterns

The bug triggers when ALL of the following conditions are met:

1. A **sum of segments** exists: `x + y` where `x = args.coeff<0>()`,
   `y = args.coeff<1>()` (creates `TwoFunctionSum` with `is_sum_of_segments = true`)
2. The sum is used inside a **nested function**: e.g., `cos(2*(x+y))`,
   which creates `NestedFunction<Cos, Scaled<TwoFunctionSum<Segment, Segment>>>`
3. The nested function's Jacobian is computed via the chain rule in
   `NestedFunction::compute_jacobian`, which calls
   `inner_func.right_jacobian_product(jx_, jx_outer, jx_inner, DirectAssignment(), false)`

Simple expressions like `x + y`, `2*(x+y)`, `sin(sqrt(x*x + y*y))` are
NOT affected. Only the combination of a segment sum inside a nested function
triggers the bug.

## Root Cause Analysis

### The Proximate Mechanism

In `NestedFunction::compute_jacobian` (NestedFunction.h, line 118–122):

```cpp
auto Impl = [&](auto &fx_inner, auto &jx_inner, auto &jx_outer) {
    this->inner_func.compute_jacobian(x, fx_inner, jx_inner);   // (1)
    this->outer_func.compute_jacobian(fx_inner, fx_, jx_outer);  // (2)
    this->inner_func.right_jacobian_product(                     // (3)
        jx_, jx_outer, jx_inner, DirectAssignment(), false);
};
```

Step (3) passes `jx_` (the output Jacobian) as `target` and `jx_inner`
(the inner function's Jacobian) as `right` to
`TwoFunctionSum::right_jacobian_product`.

**Diagnostic finding:** At the point where `TwoFunctionSum::right_jacobian_product`
executes, `target.data() == right.data()` — the two buffers are **aliased**.
This means `jx_` and `jx_inner` point to the same memory.

The `Aliased` template parameter is `false`, so the code assumes no aliasing.
`TwoFunctionSum::right_jacobian_product` (Summation.h) then:

1. Calls `func1.right_jacobian_product(target, left, right, DirectAssignment)` —
   writes to **column 0** of target (which is also column 0 of right)
2. Calls `func2.right_jacobian_product(target, left, right, PlusEqualsAssignment)` —
   reads **column 1** of right (which still contains the original `jx_inner` value
   of 2.0, because func1 only wrote to column 0), then **adds** to column 1 of target

Result: column 1 gets `jx_inner[1] + correct_value` instead of just `correct_value`.
The extra `jx_inner[1] = 2.0` produces the observed error.

### The Underlying Cause: BumpAllocator Rewrite

The aliasing of `jx_` and `jx_inner` does NOT occur in the original ASSET codebase.
Both codebases have the same `ExactTempPack` optimization (stack-allocating
fixed-size temporaries) and the same `TwoFunctionSum::right_jacobian_product`
code. Yet ASSET produces correct Jacobians and Tycho does not.

**Key finding:** Disabling `ExactTempPack` (forcing the arena path) fixes the
simple `cos(2*(x+y))` case but NOT the more complex ODE expression
`sin(sqrt(x²+y²)) * cos(2*(x+y))`. This means the arena path (`run_arena_impl`)
also has the aliasing problem.

**Key finding:** Building at `-O0` (no optimization) with `ExactTempPack` enabled
produces **correct** Jacobians. The bug only manifests at `-O3`.

This points to the BumpAllocator rewrite changing something about how memory
is allocated or mapped that interacts with compiler optimizations. The original
ASSET `MemoryManager` has a different arena implementation (`TempStack` with
`getBlock`/`freeBlock` semantics) vs Tycho's `BumpStack` with `save`/`restore`.

### What We Know

| Condition | Jacobian |
|-----------|----------|
| asset_asrl (pip), any optimization level | Correct |
| Tycho, `-O0`, ExactTempPack enabled | Correct |
| Tycho, `-O3`, ExactTempPack disabled (arena only) | **Wrong** for complex expressions |
| Tycho, `-O3`, ExactTempPack enabled | **Wrong** |
| Tycho, `-O3`, ExactTempPack disabled + Summation.h fix | Correct |

### What We Don't Know

- The exact mechanism by which the BumpAllocator rewrite causes `jx_` and
  `jx_inner` to alias. The save/restore semantics should prevent this, and
  `jx_` comes from the caller (not the arena) while `jx_inner` comes from
  the arena.
- Whether the issue is in `BumpStack::allocate`, `make_temps`, `save`/`restore`,
  or in how Eigen Map types interact with the arena memory.
- Whether the compiler's alias analysis is making incorrect assumptions due to
  the `const_cast_derived()` pattern used throughout the Eigen-based VF DSL.

## Temporary Workaround

Two changes have been applied that together eliminate the bug:

### 1. Disable ExactTempPack (memory_management.h)

```cpp
template <class Func, class... TempSpecs>
static void allocate_run(Func &&f, const TempSpecs &...tspecs) {
    // Always use the arena — ExactTempPack disabled due to aliasing bug.
    BumpAllocator::run_arena_impl(f, tspecs...);
}
```

This alone does NOT fully fix the problem (complex expressions still fail),
but it eliminates the `-O0` vs `-O3` discrepancy.

### 2. Fix TwoFunctionSum::right_jacobian_product (Summation.h)

Changed func2 from `PlusEqualsAssignment` to `DirectAssignment` when the
incoming assignment is `DirectAssignment`:

```cpp
// Both segments write to non-overlapping columns — each uses the
// incoming assignment mode for its own columns.
this->func1.right_jacobian_product(target_, left, right, assign, aliased);
this->func2.right_jacobian_product(target_, left, right, assign, aliased);
```

This is safe because the segments write to non-overlapping column ranges.
It makes the code robust against target/right aliasing.

**Note:** This Summation.h change deviates from the original ASSET code. ASSET
uses `PlusEqualsAssignment` for func2 and works correctly because its
`MemoryManager` does not produce the aliasing. The Summation.h change is
a defense-in-depth measure that should be reverted once the BumpAllocator
root cause is fixed.

## Verification

- All 290 C++ unit tests pass with both fixes applied
- Python Zermelo example (`examples/Zermelo.py`) passes
- C++ Zermelo example produces correct Jacobians matching Python and asset_asrl
- Variable-direction wind case converges in 8+10 iterations (matching Python),
  vs 57+102 iterations with the bug

## Next Steps

1. **Compare `BumpStack` with ASSET's `TempStack`** — the arena implementations
   differ (`save`/`restore` vs `getBlock`/`freeBlock`). One of these differences
   likely causes the aliasing.
2. **Add a Jacobian unit test** for `cos(2*(x+y))` and `sin(r)*cos(ang)` to
   catch regressions.
3. **Once root cause is found**, revert the Summation.h change and re-enable
   ExactTempPack if it can be done safely.
4. **Profile** to determine if ExactTempPack provides measurable performance
   benefit — if not, removing it permanently is the simplest fix.

## Related Commits

- `34fe739` — BumpAllocator rewrite that introduced the bug
- `a31bf0a` — Initial Summation.h fix (on `feat/cpp-examples` branch)
- Subsequent commits on `feat/cpp-examples` — ExactTempPack disable + investigation
