# BumpAllocator Jacobian Bug — Root Cause Analysis

## The Real Bug: TwoFunctionSum ignores the Aliased flag

The bug is NOT in BumpAllocator. It is in `TwoFunctionSum::right_jacobian_product`
(and `MultiFunctionSum::right_jacobian_product`) in `include/tycho/detail/Summation.h`.

### Mechanism

`CwiseFunctionOperator::compute_jacobian_impl` (CwiseOperators.h:879-900) computes
the Jacobian of `cos(f(x))` by first computing the inner Jacobian into `jx_`, then
scaling it in-place:

```cpp
this->func.compute_jacobian(x, fxt, jx_);   // (1) jx_ = Jacobian of inner func
...
// (3) Scale jx_ in-place: jx_ = diag(-sin(fxt)) * jx_
this->func.right_jacobian_product(jx_, jxdiag, jx_, DirectAssignment(), true);
//                                 ^^target     ^^right — SAME buffer, Aliased=true
```

Step (3) passes `jx_` as **both target and right** — intentional self-aliasing for
in-place scaling. The `Aliased=true` flag is set correctly.

For `cos(2*(x+y))`, this dispatches through `Scaled::right_jacobian_product` to
`TwoFunctionSum::right_jacobian_product`. Since `is_sum_of_segments = true`, the
code decomposes the product into per-segment operations:

```cpp
// Summation.h, original (buggy) code:
this->func1.right_jacobian_product(target_, left, right, DirectAssignment(), aliased);
this->func2.right_jacobian_product(target_, left, right, PlusEqualsAssignment(), aliased);
```

In the non-aliased case, `target` columns are zero-initialised, so
`PlusEqualsAssignment` (adding to zero) produces the correct result.

In the aliased case (`target == right`), `target.col(1)` holds the original
`right.col(1)` value (the inner Jacobian entry, e.g. 2.0), NOT zero.
`PlusEqualsAssignment` then adds the correct scaled value on top of the non-zero
original, producing `2.0 + 1.12928 = 3.12928` instead of `1.12928`.

### Why the Bug Doc's BumpAllocator Theory is Wrong

The bug doc hypothesizes that BumpAllocator causes `jx_` and `jx_inner` to alias
in `NestedFunction::compute_jacobian_impl`. This is incorrect:

1. The actual expression `cos(2*(x+y))` creates `CwiseCos<Scaled<TwoFunctionSum<...>>>`,
   NOT `NestedFunction<CosFunction, Scaled<...>>`. The `cos()` overload produces `CwiseCos`.

2. `CwiseFunctionOperator::compute_jacobian_impl` does the chain rule entirely
   differently from NestedFunction — it scales the Jacobian in-place using
   `right_jacobian_product(jx_, diag, jx_, ...)` where `jx_` IS both target and right.

3. The `ExactTempPack` disable was a red herring. The bug manifests at all
   optimisation levels and with any allocation strategy because the issue is in
   how `TwoFunctionSum::right_jacobian_product` handles the `Aliased` flag.

### The Fix

In `TwoFunctionSum::right_jacobian_product` (and `MultiFunctionSum`): when
`Aliased=true` and the incoming assignment is `DirectAssignment`, use
`DirectAssignment` (not `PlusEqualsAssignment`) for func2 and subsequent segments.
Each segment writes to non-overlapping columns, and with aliasing the target
columns are not zero — they hold the original right values.

The non-aliased case is unchanged (PlusEquals on zero-initialised target).

```cpp
if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
    if constexpr (Aliased) {
        this->func2.right_jacobian_product(target_, left, right, DirectAssignment(), aliased);
    } else {
        this->func2.right_jacobian_product(target_, left, right, PlusEqualsAssignment(), aliased);
    }
}
```

### ExactTempPack

Re-enabled since the real fix is in Summation.h, not in the allocator.

## Verified

- [x] Bug reproduces in diagnostic (`bug-fix/jac_alias_diag.cpp`) — 7 of 9 tests fail before fix
- [x] Test 8 ("direct, no GF") confirms bug exists without GenericFunction/virtual dispatch
- [x] Error is exactly +2.0 in the simple case, consistent with the aliasing mechanism
- [x] Fix verified — all 9 diagnostic tests pass
- [x] 290/290 C++ unit tests pass
- [x] 30/30 Python examples pass (8 skipped for basemap)
- [x] C++ brachistochrone converges: "Optimal Solution Found", t=1.8013 s
- [x] ExactTempPack re-enabled successfully
