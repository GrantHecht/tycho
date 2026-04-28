# Enzyme SIMD Trig Findings

This document is a follow-up review of `EnzymeTrig.md` after testing both the
LLVM 22 Enzyme build and a rebuilt Enzyme using the system LLVM 21 toolchain.

The main conclusion is that Approach A is not the best real fix available right
now. The better long-term fix is an Enzyme patch for the x86 SIMD IR emitted by
Eigen 5 packet trig. LLVM 22 introduced an additional compatibility issue, but
LLVM 21 proves that the core blocker is not only LLVM 22 compatibility.

## Current Recommendation

Use Approach D only as a short-term unblocker. It is appropriate for getting
affected tests and benchmarks moving again, but it should be documented as
disabling the direct SIMD Enzyme path for those fixtures.

For the real fix, prefer an expanded Approach B:

1. Add Enzyme support for the x86 vector round intrinsics emitted by Eigen's
   SIMD trig implementation.
2. Extend Enzyme's handling of Eigen's packed integer bit-manipulation idioms
   used to implement floating-point absolute value, sign extraction, sign
   toggling, and range-reduction bookkeeping.
3. Preserve AVX and the direct SIMD Enzyme path. Disabling AVX is not an
   acceptable Tycho solution.
4. Keep the LLVM 22 masked load/store compatibility fix in scope if Tycho wants
   to support LLVM 22 again.
5. Add targeted Enzyme regression tests for the reduced LLVM IR patterns Tycho
   emits under LLVM 21 and LLVM 22.

Approach A should be kept as a fallback or temporary Tycho-local workaround, not
as the primary long-term answer.

## Bottom Line On LLVM Versions

Rebuilding Enzyme with LLVM 21 avoids the LLVM 22-specific masked-load/store
crash, but it does not make the direct SIMD Enzyme path work.

Under LLVM 21, the first hard failure is:

```text
error: Enzyme: cannot handle (forward) unknown intrinsic
llvm.x86.avx.round.pd.256
```

So the problem is not simply "LLVM 22 plus Enzyme are incompatible." LLVM 22
adds one extra incompatibility, but the underlying Eigen packet trig lowering is
still unsupported by Enzyme under LLVM 21.

Because Tycho only requires full C++20 support, using LLVM 21 is reasonable from
a language-support standpoint. However, older LLVM alone is not a reliable fix.
LLVM 21 already shows the same core SIMD trig failure. Older LLVM versions may
change the exact IR shape, but the diagnostic runs below show that Enzyme also
struggles with the lower-level vector bit operations used by Eigen's packet math.

## Active LLVM 21 Configuration

The active Tycho build was using the system Clang/LLVM 21 toolchain:

```text
/usr/bin/clang++ --version
clang version 21.1.8 (Fedora 21.1.8-4.fc43)
```

The Enzyme install used by Tycho was also rebuilt/configured for LLVM 21:

```cmake
set(Enzyme_LLVM_VERSION_MAJOR "21")
set(Enzyme_LLVM_VERSION_MINOR "1")
set(Enzyme_LLVM_VERSION_PATCH "8")
set(Enzyme_LLVM_DIR "/usr/lib64/llvm21/lib64/cmake/llvm")
set(Enzyme_CLANG_EXE "/usr/bin/clang-21")
```

Tycho's CMake checks require the active Clang major version to match Enzyme's
LLVM major version, so this was not an accidental mixed LLVM 22/21 setup.

## LLVM 21 Build Failure

Rebuilding the affected targets with LLVM 21:

```text
cmake --build build --target tycho_enzyme_tests bench_enzyme -j2
```

failed in both the Enzyme tests and the benchmark target. The first visible
failure in `test_enzyme_vectorized.cpp` was:

```text
error: Enzyme: cannot handle (forward) unknown intrinsic
llvm.x86.avx.round.pd.256
  %24 = call reassoc nsz arcp contract afn noundef <4 x double>
        @llvm.x86.avx.round.pd.256(<4 x double> %23, i32 1)
```

`bench_enzyme.cpp` failed in the same way:

```text
error: Enzyme: cannot handle (forward) unknown intrinsic
llvm.x86.avx.round.pd.256
  %17 = tail call reassoc nsz arcp contract afn noundef <4 x double>
        @llvm.x86.avx.round.pd.256(<4 x double> %16, i32 1)
```

After reporting the unsupported intrinsic, Clang aborted during the Enzyme pass
with:

```text
fatal error: error in backend: function failed verification (4)
```

LLVM 21 crash reproducers were emitted under `/tmp`, including:

```text
/tmp/test_enzyme_vectorized-70262a.cpp
/tmp/test_enzyme_vectorized-70262a.sh
/tmp/bench_enzyme-d95eb6.cpp
/tmp/bench_enzyme-d95eb6.sh
```

## LLVM 21 IR Without Enzyme

Generating LLVM IR without the Enzyme plugin under LLVM 21 produced:

```text
/tmp/bench_enzyme_llvm21_noenzyme.ll
```

That file had 76,187 lines and confirmed the failing Eigen packet trig shape.

The LLVM 21 IR still contains old-style masked load/store intrinsics with an
explicit alignment immarg:

```llvm
%95 = call <2 x double> @llvm.masked.load.v2f64.p0(
  ptr nonnull %48,
  i32 1,
  <2 x i1> <i1 true, i1 false>,
  <2 x double> poison)

call void @llvm.masked.store.v2f64.p0(
  <2 x double> %96,
  ptr nonnull %47,
  i32 1,
  <2 x i1> <i1 true, i1 false>)
```

The declarations also include the alignment operand:

```llvm
declare <2 x double> @llvm.masked.load.v2f64.p0(
  ptr captures(none),
  i32 immarg,
  <2 x i1>,
  <2 x double>)

declare void @llvm.masked.store.v2f64.p0(
  <2 x double>,
  ptr captures(none),
  i32 immarg,
  <2 x i1>)
```

This is compatible with Enzyme's current masked-load/store operand indexing.
Therefore, the LLVM 22 masked-load/store crash is not the active LLVM 21
failure.

The same LLVM 21 IR also confirms the Eigen AVX trig lowering:

```llvm
%17 = tail call reassoc nsz arcp contract afn noundef <4 x double>
      @llvm.x86.avx.round.pd.256(<4 x double> %16, i32 1)

%28 = tail call noundef <4 x i64>
      @llvm.x86.avx2.psrlv.q.256(<4 x i64> %26, <4 x i64> %27)

%29 = tail call noundef <4 x i64>
      @llvm.x86.avx2.psllv.q.256(<4 x i64> splat (i64 1), <4 x i64> %25)

%113 = and <4 x i64> %110, splat (i64 -9223372036854775808)
%114 = xor <4 x i64> %113, %112
```

These are part of Eigen's double-precision packet trig implementation:
argument/range reduction, quadrant bookkeeping, and sign application.

## LLVM 22-Specific Masked Load/Store Issue

The original LLVM 22 investigation remains valid, but it is a separate issue.

Under LLVM 22, Tycho emitted masked load/store intrinsics with no alignment
operand. Alignment lives on the pointer argument:

```llvm
%95 = call <2 x double> @llvm.masked.load.v2f64.p0(
  ptr nonnull align 16 %48,
  <2 x i1> <i1 true, i1 false>,
  <2 x double> poison)

call void @llvm.masked.store.v2f64.p0(
  <2 x double> %96,
  ptr nonnull align 8 %47,
  <2 x i1> <i1 true, i1 false>)
```

LLVM 22's own release notes describe this IR change: the alignment argument was
removed from masked load/store/gather/scatter intrinsics and replaced by an
`align` attribute on the pointer argument.

Current Enzyme source still assumes the older intrinsic shape:

```cpp
if (ID == Intrinsic::masked_store) {
  auto align0 = cast<ConstantInt>(I.getOperand(2))->getZExtValue();
  ...
}

if (ID == Intrinsic::masked_load) {
  auto align0 = cast<ConstantInt>(I.getOperand(1))->getZExtValue();
  ...
}
```

With LLVM 22, those operands are masks, not `ConstantInt` values. That explains
the LLVM 22 crash:

```text
llvm::cast<ConstantInt>(From*) argument of incompatible type
```

This masked-load/store issue should be fixed if Tycho wants LLVM 22 support,
but LLVM 21 demonstrates that fixing it is not sufficient for Eigen SIMD trig.

## Diagnostic Feature Tests

These feature tests were diagnostic only. They should not be interpreted as
candidate Tycho solutions. Tycho should keep AVX enabled.

### AVX Disabled, SSE4.1 Enabled

Generating IR with AVX disabled but SSE4.1 enabled changed the packet width from
`<4 x double>` to `<2 x double>`, but did not eliminate the class of problem.
Eigen emitted the SSE4.1 round intrinsic instead:

```llvm
%16 = tail call reassoc nsz arcp contract afn noundef <2 x double>
      @llvm.x86.sse41.round.pd(<2 x double> %15, i32 1)

%97 = and <2 x i64> %94, splat (i64 -9223372036854775808)
%98 = xor <2 x i64> %97, %96
```

This confirms that the problem is not uniquely AVX. It is Enzyme's handling of
Eigen's x86 packet trig lowering.

### AVX And SSE4.1 Disabled

Generating IR with AVX and SSE4.1 disabled removed the target-specific round
intrinsic, but Enzyme still failed on Eigen's packet bit operations.

The Enzyme failure included:

```text
cannot handle unknown binary operator:
  %4 = and <2 x i64> %3, splat (i64 9223372036854775807)
```

That pattern is the vector floating-point absolute-value mask, clearing the sign
bit. The same diagnostic compile later asserted in Enzyme:

```text
GradientUtils.cpp:6594:
Value *GradientUtils::invertPointerM(...):
Assertion `0 && "cannot find deal with ptr that isnt arg"' failed.
```

The corresponding crash reproducer was:

```text
/tmp/bench_enzyme-9d0d3b.cpp
/tmp/bench_enzyme-9d0d3b.sh
```

This diagnostic matters because it shows that even if the target-specific round
intrinsic is avoided, Enzyme still needs better handling for Eigen's packed
integer representation of floating-point operations.

### `-enzyme-loose-types=1`

Testing Enzyme's loose type mode with AVX still enabled was not viable.

It still reported the unknown AVX round intrinsic:

```text
error: Enzyme: cannot handle (forward) unknown intrinsic
llvm.x86.avx.round.pd.256
```

It also emitted warnings that it was treating integer bit operations as
constant:

```text
warning: binary operator is integer and constant:
  %110 = xor <4 x i64> %109, %10

warning: binary operator is integer and constant:
  %113 = and <4 x i64> %110, splat (i64 -9223372036854775808)
```

Then it generated invalid IR:

```text
Invalid bitcast
  %2026 = bitcast <4 x i64> %2014 to double

Invalid operands for select instruction!
  %2028 = select fast <4 x i1>, double %2026, double %2027
```

The backend then failed verification. This is not a safe workaround.

## Why Disabling AVX Is Not A Solution

Tycho should not disable AVX.

The AVX-disabled runs were only probes to expose the next failure mode after the
AVX round intrinsic. They show that:

1. SSE4.1 has an analogous target-specific round intrinsic.
2. Even older/non-SSE4.1 lowering still contains Eigen packet bit tricks that
   Enzyme cannot currently differentiate robustly.
3. Disabling AVX would reduce performance and still would not provide a
   principled direct SIMD Enzyme path.

Therefore the fix should preserve AVX and improve Enzyme's support for the IR
that Eigen emits.

## Why Approach A Is Not Ideal

Approach A hides Eigen's SIMD trig body behind Tycho math wrappers and custom
Enzyme derivative registration. That can be useful as containment, but it has
several important problems.

First, it requires user opt-in. Any differentiated code path that calls ordinary
`sin` or `cos`, or reaches Eigen's packet trig through another route, can still
hit the same Enzyme limitations.

Second, the custom derivative ABI is not proven for the proposed C++ wrapper
shape. Existing Enzyme custom derivative examples tend to use scalar functions
or C-style pointer-out functions. A wrapper returning or accepting
`Eigen::Array<double, W, 1>` may lower through sret/byval/aggregate conventions,
and the registered derivative must match that lowered ABI exactly.

Third, A only addresses the trig body. The diagnostic runs show that Enzyme also
has trouble with Eigen's vector bit operations. Wrapping only `sin` and `cos`
does not make the direct SIMD Enzyme path generally robust.

Fourth, A creates a Tycho-specific math API burden. Users must remember to call
the wrapper functions. If they do not, the failure returns.

Approach A remains a possible temporary Tycho-side escape hatch if the Enzyme
patch proves unexpectedly deep, but it should not be the preferred long-term
design.

## Enzyme Source Observations

Relevant Enzyme source observations:

- `AdjointGenerator.h` has explicit handling for `masked_load` and
  `masked_store`, but the current operand indexing assumes the LLVM 21-and-older
  intrinsic ABI.
- `ActivityAnalysis.cpp` marks generic rounding intrinsics like `round`,
  `roundeven`, `floor`, `ceil`, `trunc`, `rint`, and `nearbyint` as inactive,
  but it does not cover x86-specific vector round intrinsics such as
  `x86_avx_round_pd_256` or `x86_sse41_round_pd`.
- The forward/reverse unknown-intrinsic path in `AdjointGenerator.h` reports
  `cannot handle (forward) unknown intrinsic` for the x86 round intrinsic.
- `AdjointGenerator.h` has narrow `and` handling. It does not robustly cover
  vector splat masks used by Eigen for floating-point absolute value
  (`0x7fff...`) or sign-bit extraction (`0x8000...`).
- Existing `xor` handling is closer because Enzyme has helper logic for
  top-bit-only constants, but the observed invalid IR under
  `-enzyme-loose-types=1` indicates vector type propagation is still not robust
  for the Eigen bitcast graph.
- `Utils.h` already contains `containsOnlyAtMostTopBit`, which can recognize
  scalar and vector top-bit constants. That helper is relevant but not currently
  enough for all of Eigen's packet patterns.
- Enzyme already has generic intrinsic derivative machinery for mathematical
  `sin` and `cos`, including vector types. The missing piece is support for the
  lowered target-specific IR used inside Eigen's packet implementation.

## Tycho Dispatch Observation

Tycho's Enzyme dispatch is important for interpreting Approach D.

When a function is `Vectorizable` and the scalar type is a SuperScalar,
`dense_enzyme.h` routes to the direct SIMD Enzyme implementation. If
`is_vectorizable=false`, `DenseFunctionBase::compute_jacobian` scalarizes
per-lane before Enzyme sees the function body.

Therefore D is a valid unblocker because it avoids the direct SIMD Enzyme path,
but it also means those fixtures no longer exercise the behavior Tycho
eventually wants to support.

## Proposed Real Fix Scope

The best next implementation target is a bounded Enzyme patch.

### Required For LLVM 21

1. Treat x86 vector round intrinsics used by Eigen trig as inactive or give them
   an explicit zero derivative. At minimum this includes:
   - `llvm.x86.avx.round.pd.256`
   - `llvm.x86.sse41.round.pd`

   It may also be worth checking the corresponding `ps` forms while adding
   tests, even though the current Tycho failure is double-precision packet trig.

2. Extend `and` handling for vector floating-point bit masks:
   - sign-bit extraction masks such as `0x8000000000000000`;
   - absolute-value masks such as `0x7fffffffffffffff`;
   - vector splat forms represented as `ConstantVector` or `ConstantDataVector`.

3. Ensure `xor` sign-bit toggling is recognized for Eigen's vector bitcast
   patterns and keeps the correct vector floating type, not a scalar `double`.

4. Account for AVX2 integer shift intrinsics used in Eigen's range-reduction
   bookkeeping:
   - `llvm.x86.avx2.psrlv.q.256`
   - `llvm.x86.avx2.psllv.q.256`

   These are part of producing the integer quadrant/reduction values. Their
   derivatives are inactive almost everywhere, but Enzyme needs to propagate
   that fact without producing invalid IR.

5. Add LLVM 21 lit tests using reduced IR that reproduces:
   - AVX double packet trig range reduction;
   - SSE4.1 double packet round;
   - vector sign-bit `and`/`xor`;
   - vector absolute-value bit masks.

### Required For LLVM 22 Support

6. Update `masked_load` and `masked_store` handling to support LLVM 22's
   pointer-alignment attribute representation.

   Enzyme should derive alignment from the pointer operand's attributes when the
   alignment immarg is absent, while retaining support for LLVM 21-and-older
   operand layouts.

7. Add LLVM 22 lit tests for masked load/store intrinsic signatures without the
   alignment operand.

## Expected Outcome

This fix belongs in Enzyme because it addresses the actual lowered IR, preserves
ordinary user code, preserves Tycho's direct SIMD Enzyme path, and keeps AVX
enabled.

The practical sequencing should be:

1. Use D only where needed to unblock unrelated Tycho work.
2. Patch Enzyme for the LLVM 21 Eigen packet trig failures.
3. Add the LLVM 22 masked-load/store compatibility patch if LLVM 22 remains a
   target.
4. Re-enable the affected vectorized fixtures and benchmarks so they again test
   direct SIMD Enzyme rather than scalarized fallback behavior.
