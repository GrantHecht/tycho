# Tycho upstream Enzyme/Eigen canary

A pair of compile-time tests that detect when upstream gains support for
SIMD trig differentiation under Enzyme — the constraint that drove the
current `tycho::math::cos / sin` design (per-lane scalar libm calls in
[`include/tycho/detail/utils/simd_math.h`](../../include/tycho/detail/utils/simd_math.h)).

## What's checked

| Test | What it tries | If it passes |
|---|---|---|
| **A** | `Eigen::Array<double, 4, 1>::cos()` differentiated by Enzyme at `enzyme_width=4`. Today fails because Eigen 5's `pcos<Packet4d>` lowers to bithack IR Enzyme can't handle (issues [#689](https://github.com/EnzymeAD/Enzyme/issues/689), [#2679](https://github.com/EnzymeAD/Enzyme/issues/2679), [#2794](https://github.com/EnzymeAD/Enzyme/issues/2794)). | Drop the per-lane scalar wrappers in `simd_math.h`; call `x.cos() / x.sin()` directly. Larger SIMD trig win available. |
| **B** | A custom-registered Enzyme derivative invoked under `enzyme_width=4`. Today fails because Enzyme's auto-generated `fixderivative_*` bridge produces invalid IR when widening a registered scalar-shape derivative to take `[4 x ptr]` shadow aggregates. | Alternative architecture becomes viable: keep Eigen's SIMD packet trig, register a custom derivative, let Phase 3 batching amortise across BW tangents. |

## How to run

```bash
scripts/upstream_canary/check.sh
```

Exit code:
- **0**: status quo unchanged (both tests still fail at compile time).
- **non-zero**: at least one test compiles or runs — upstream has changed
  and the dependent code in `simd_math.h`, `dense_enzyme.h`, and
  `EnzymeTrig.md` should be revisited.

Environment overrides (defaults shown):

```
CXX=clang++
ENZYME_PLUGIN=$HOME/Software/Enzyme/install/lib/ClangEnzyme-21.so
EIGEN_INCLUDE=<repo>/dep/eigen
```

## When to run

Manually before bumping the Enzyme submodule or Eigen submodule. Or
schedule a periodic agent (e.g. monthly) to detect upstream changes.

## What to do if a test passes

1. Re-read the current implementation rationale in `CLAUDE.md` (search for
   "tycho::math") and in `EnzymeTrig.md` at the repo root.
2. Verify the win is real with the existing benchmarks under
   `bench/cpp/bench_enzyme.cpp` (compare `BM_JacobianVecSIMD_Enzyme/*`
   numbers before vs after the simplification).
3. Update `simd_math.h`, the affected fixtures, and the rule-of-thumb
   section of `CLAUDE.md`.
