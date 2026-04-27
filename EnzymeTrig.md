# Enzyme + Eigen 5 SIMD Trig — Resolution Plan

**Status:** investigation complete; awaiting decision on path A / D / D-then-A
**Branch:** `eigen-upgrade` (off `worktree-enzyme-ad-plan`)
**Eigen pin:** 5.0.1 (libeigen upstream) — **not reverting**
**Date:** 2026-04-27

---

## 1. Problem

After upgrading `dep/eigen` from 3.4 (wgledbetter fork) to 5.0.1 (libeigen
upstream), three Enzyme-touching translation units fail to build:

- `bench/cpp/bench_enzyme.cpp`
- `tests/cpp/enzyme/test_enzyme_hessian.cpp`
- `tests/cpp/enzyme/test_enzyme_vectorized.cpp`

Errors:
```
Enzyme: cannot handle (forward) unknown intrinsic llvm.x86.avx.round.pd.256
cannot handle unknown binary operator: %... = xor <4 x i64> ..., splat (i64 -9223372036854775808)
cannot handle unknown binary operator: %... = and <4 x i64> ..., splat (i64 -9223372036854775808)
```

### Root cause

Eigen 5.0.1 ships vectorised `psin<Packet4d>` / `pcos<Packet4d>` in
`dep/eigen/Eigen/src/Core/arch/AVX/MathFunctions.h`. The polynomial
range-reduction path lowers to:

- `llvm.x86.avx.round.pd.256` (packed double round-to-int — locally
  constant, derivative is 0)
- `xor <4 x i64> v, splat(0x8000000000000000)` (packed-double sign flip,
  i.e. element-wise negation)
- `and <4 x i64> v, splat(0x8000000000000000)` (packed-double sign-bit
  extraction)

Enzyme 22.1 has handlers for the *scalar* equivalents but not the
packed-i64 splat patterns and not `x86_avx_round_pd_256`, so the AD pass
aborts.

### Scope

Only Phase 5b (`is_vectorizable=true` VFs paired with `EnzymeAD` and
trig-bearing bodies) hits this. Real user code is unaffected:

- `mee_dynamics.h:37` keeps `is_vectorizable=true`. Its **analytic /
  non-Enzyme** integration path now benefits from Eigen 5 SIMD trig as
  intended.
- The CLAUDE.md "trig-heavy bodies regress under Phase 5b" caveat
  already directs Enzyme users to `is_vectorizable=false`, which routes
  through Phase 5a scalarise-per-lane → libm `cos`/`sin` → Enzyme handles
  it cleanly.

The failing TUs are micro-benchmarks and regression tests for a path
the docs already deprecate as suboptimal.

---

## 2. Affected fixtures

| Fixture | File | Line | Trig? | What it tests |
|---|---|---|---|---|
| `BrachBench<Jm,Hm>` | `bench/cpp/bench_enzyme.cpp` | 98 | yes | Phase 5b SIMD Jacobian primal + Enzyme tangents |
| `CR3BPBench<Jm,Hm>` | `bench/cpp/bench_enzyme.cpp` | 124 | **no** (sqrt only) | Phase 5b SIMD Jacobian — should still build |
| `MEEBench<Jm,Hm>` | `bench/cpp/bench_enzyme.cpp` | 157 | yes | Phase 5b SIMD Jacobian + Enzyme tangents |
| `BrachVectorizable<Jm,Hm>` | `tests/cpp/enzyme/test_enzyme_vectorized.cpp` | 21 | yes | Phase 5a scalarise-per-lane correctness |
| (`enzyme_test_dynamics.h` fixtures used by `test_enzyme_hessian.cpp`) | n/a | — | yes (in `MEEEnzymeFull`) | scalar Enzyme; default `is_vectorizable=false` — should not hit Phase 5b but currently fails; needs verification |

`mee_dynamics.h:37` — **stays `is_vectorizable=true`**; not in scope.

---

## 3. Solution space (no Eigen revert)

### Approach A — opaque wrapper + registered Enzyme derivative

Wrap SIMD trig behind a `__attribute__((noinline))` shim and register a
custom forward + reverse derivative with Enzyme.

```cpp
// noinline so the call survives Enzyme's pass
__attribute__((noinline)) Eigen::Array<double,4,1>
tycho_simd_cos_W4(const Eigen::Array<double,4,1>& x);

// Forward derivative: d cos(x)/dx · dx = -sin(x) * dx
void tycho_simd_cos_W4_fwd(const Eigen::Array<double,4,1>* x,
                           const Eigen::Array<double,4,1>* dx,
                           Eigen::Array<double,4,1>* y,
                           Eigen::Array<double,4,1>* dy);

// Reverse augment + gradient (reverse mode used for Hessian outer pass)
void* tycho_simd_cos_W4_aug(...);
void  tycho_simd_cos_W4_grad(...);

void* __enzyme_register_derivative_tycho_simd_cos_W4[] = {
    (void*)tycho_simd_cos_W4, (void*)tycho_simd_cos_W4_fwd
};
void* __enzyme_register_gradient_tycho_simd_cos_W4[] = {
    (void*)tycho_simd_cos_W4, (void*)tycho_simd_cos_W4_aug, (void*)tycho_simd_cos_W4_grad
};
```

**Public surface:** add `tycho::math::cos(x)` / `sin(x)` overloads in
`include/tycho/detail/utils/math.h`. For `double` Scalar dispatches to
`std::cos`; for `Eigen::Array<double,W,1>` dispatches to the registered
SIMD wrapper. User VFs that opt in write `tycho::math::cos(x[5])` instead
of `cos(x[5])`.

**Plumbing source (Enzyme):**
- `enzyme/Enzyme/PreserveNVVM.cpp:60–325` — registration handler scans
  `__enzyme_register_derivative_*` / `_gradient_*` / `_splitderivative_*`
  globals; sets `enzyme_derivative` / `enzyme_augment` /
  `enzyme_gradient` metadata on the original; force-applies `NoInline` and
  `ExternalLinkage`.
- `enzyme/Enzyme/Enzyme.cpp:3216` — pass registered via
  `registerOptimizerEarlyEPCallback`, runs before Enzyme proper, so the
  `noinline` survives.
- `enzyme/Clang/EnzymeClang.cpp:200–214` — auto-applies `UsedAttr` on
  globals matching the registration prefixes.

**Authoritative test references:**
- `enzyme/test/Integration/ForwardMode/customfwd.c:22–47` — scalar
  forward mode
- `enzyme/test/Integration/ReverseMode/customcombined.c:22–49` — scalar
  reverse mode
- `enzyme/test/Enzyme/ForwardMode/v_fmax4.ll:4–17` — `<4 x double>`
  precedent

**Confidence:** high. Maintenance: zero (no Enzyme fork).
**Effort:** ~half-day (3 wrappers × {fwd, aug, grad} × {W=2, W=4, W=8}).

### Approach B — patch Enzyme

Three localised changes:

1. Add `Intrinsic::x86_avx_round_pd_256` (and `x86_sse41_round_pd`) to
   `KnownInactiveIntrinsics` in
   `enzyme/Enzyme/ActivityAnalysis.cpp:308–360`. Mirrors existing entries
   for `Intrinsic::round`/`floor`/`ceil`/`nearbyint`. Trivial.
2. Extend `And` handler in `enzyme/Enzyme/AdjointGenerator.h:2438–2467`
   (reverse) and the parallel block at ~2674 (forward) to recognise
   splat-of-sign-bit on integer-vector types and emit zero derivative.
   Today only a single hard-coded `0xF8000000` mantissa-truncation
   pattern is recognised.
3. The `Xor` handler at `AdjointGenerator.h:2469–2506` *is* in principle
   capable via `containsOnlyAtMostTopBit` (`Utils.h:2054–2128`, which
   already covers `ConstantVector` and `ConstantDataVector` splats), but
   relies on `TR.query(&BO).IsAllFloat(...)` succeeding through Eigen's
   bitcast graph. If type analysis fails, add a `looseTypeAnalysis`
   fallback assuming float when the splat matches sign-bit.

**Total patch:** ~50 lines across 2 files. Optionally also add
`x86_avx_round_pd_256` as an `IntrPattern` with `(InactiveArg)` in
`enzyme/Enzyme/InstructionDerivatives.td:1001–1017` (existing precedent:
`x86_sse2_min_pd`, `x86_sse2_max_pd`).

**Confidence:** medium. The Xor type-analysis hop is the unknown.
**Maintenance:** downstream Enzyme patch maintained in tycho until
upstreamed.
**Effort:** ~1 day to implement + verify; longer to upstream.

### Approach C — `enzyme_function_like("cos")`

Alias a wrapper to Enzyme's existing `cos` dispatch table.

**Confidence:** low. Existing `cos` handler is registered against scalar
`double` (`IntrinsicDerivatives.inc:958` — `Intrinsic::cos`); a
`<4 x double>`-returning wrapper will mismatch.
**Effort:** ~half-day spike.

### Approach D — set `is_vectorizable=false` on offending fixtures

Drop the unsupported combination from the test surface. Phase 5b + trig
+ Enzyme stops being a regression-tested path. Real users follow the
documented `is_vectorizable=false` rule for trig bodies; nothing changes
for them.

`mee_dynamics.h:37` is **untouched** — its analytic-integration path
keeps `is_vectorizable=true` and benefits from Eigen 5 SIMD trig.

**Confidence:** high. Build clears immediately.
**Effort:** ~30 min: edit four fixtures, rerun gate.
**Cost:** removes a regression-test for a slow path; the slow path was
already documented as inferior.

---

## 4. Recommended path

**D first, then A.**

1. **D unblocks Phase 2.** It deletes a regression-test for a path the
   docs already declare worse than the alternative. It is *not* a revert
   — Eigen 5 stays, mee_dynamics analytic SIMD trig stays, real user
   workflow unchanged.

2. **A delivers a faster path than D removed.** With `tycho::math::cos`
   + registered Enzyme derivatives, Phase 5b + Enzyme + trig becomes a
   *supported* combination at full SIMD speed. Targets the same
   benchmark slot MEE used to live in (Phase 5b was 142 ns vs Phase 3
   scalar 24 ns; A should beat both).

3. **B is held in reserve** for upstreaming. Worth doing *eventually*
   but not on the critical path.

### Phase D — fixture cleanup (this PR)

| Step | File | Change |
|---|---|---|
| D.1 | `bench/cpp/bench_enzyme.cpp:103` | `BrachBench`: `is_vectorizable = false` |
| D.2 | `bench/cpp/bench_enzyme.cpp:162` | `MEEBench`: `is_vectorizable = false` |
| D.3 | `tests/cpp/enzyme/test_enzyme_vectorized.cpp:27` | `BrachVectorizable`: `is_vectorizable = false` |
| D.4 | `tests/cpp/enzyme/test_enzyme_hessian.cpp` | verify build clears; if `MEEEnzymeFull` still fails, investigate (default is already `false`) |
| D.5 | `bench/cpp/bench_enzyme.cpp:124` | `CR3BPBench` left alone — confirms Phase 5b SIMD still works for arithmetic-only bodies |
| D.6 | rerun pre-merge gate (Tasks 2.3 / 2.4) | verify ctest, examples, brachistochrone, bench_enzyme |
| D.7 | CLAUDE.md update | upgrade the existing "trig-heavy regress under Phase 5b" note from a *recommendation* to a *hard rule* under Eigen 5; mention Approach A as future work |

### Phase A — supported SIMD trig under Enzyme (follow-up PR, separate from Eigen 5 merge)

| Step | Deliverable |
|---|---|
| A.1 | New header `include/tycho/detail/utils/simd_math.h` with `tycho_simd_{cos,sin}_W{2,4,8}` declarations |
| A.2 | New TU `src/utils/simd_math.cpp` defining the noinline wrappers (just `return x.cos();` etc., relying on Eigen 5's `pcos<Packet4d>`) |
| A.3 | Register forward + reverse derivatives via `__enzyme_register_derivative_*` / `__enzyme_register_gradient_*` globals in same TU |
| A.4 | New header `include/tycho/detail/utils/math.h` exposing `tycho::math::cos(x)` / `sin(x)` overloads — `double` → `std::cos`; `Eigen::Array<double,W,1>` → `tycho_simd_cos_WK` (`K∈{2,4,8}`) |
| A.5 | Restore Phase 5b coverage: revert D.1/D.2/D.3 and migrate the trig calls to `tycho::math::cos` / `tycho::math::sin` |
| A.6 | New unit test `test_simd_math_enzyme_derivative` validating: `__enzyme_fwddiff(tycho_simd_cos_W4)` ≈ `-sin(x) * dx` to 1e-13 |
| A.7 | Bench: `simd_math_jacobian_bench` measuring Phase 5b + Enzyme + trig on MEEBench-with-`tycho::math` vs Phase 3 scalar W=8 |
| A.8 | Doc update: CLAUDE.md "Vectorizable VFs with EnzymeAD" section adds `tycho::math::cos/sin` as the supported way to do SIMD trig under Enzyme |

---

## 5. Open questions for user

1. **Path:** D-only, D-then-A, or skip D and go straight to A? *(Recommendation: D-then-A.)*
2. **`tycho::math` namespace placement:** under `tycho::math` (new namespace) or `tycho::detail::simd_math`? Public-facing because users of EnzymeAD-vectorizable VFs need it.
3. **W coverage:** start with W=4 only (current default on AVX2), add W=2 / W=8 in a follow-up? Or all three at once?
4. **`MEEEnzymeFull` (`test_enzyme_hessian.cpp`) failure:** if it still breaks under default `is_vectorizable=false` after D, that means MEE-trig is reaching Eigen's vectorised `pcos<Packet4d>` via *some other* path (likely SuperScalar Scalar of `Eigen::Matrix<Array<double,4,1>, ...>` *and* Enzyme — the Phase 5a path?). Needs investigation in D.4.
5. **B (upstream Enzyme patch):** worth filing now as a future-work issue against Enzyme, or wait until after A is shipped?

---

## 6. Investigation evidence

### Failing fixtures characterized

(Subagent investigation, Explore on `/home/ghecht/Projects/tycho`.)

> "Trig is essential — the brachistochrone *is* the physics being
> tested." (`BrachBench`, `bench_enzyme.cpp:98`)
>
> "MEE is the 9-state orbital-element model; trig appears once per
> evaluation and is inherent to the physics." (`MEEBench`,
> `bench_enzyme.cpp:157`)
>
> "`mee_dynamics.h` line 37: `static constexpr bool is_vectorizable =
> true`. This is correct — MEE is used with Analytic (non-Enzyme)
> integration via the bare `compute_impl`."

### Enzyme custom-derivative API verified

(Subagent investigation, general-purpose on `/home/ghecht/Software/Enzyme`.)

> "Public custom-derivative API summary: `__enzyme_register_derivative_<NAME>`
> — forward mode. Array of 2 function pointers: `{ original,
> fwd_derivative }`. `__enzyme_register_gradient_<NAME>` — reverse mode.
> Array of 3 function pointers: `{ original, augmented_primal, gradient
> }`. Plumbing source: `Enzyme/PreserveNVVM.cpp:100–325, 444–462`."
>
> "`__attribute__((noinline))` on `tycho_simd_cos` is sufficient — every
> official integration test does it that way."
>
> "Forward, scalar: `customfwd.c:22–47`. Reverse, scalar:
> `customcombined.c:22–49`. Vector `<4 x double>` value passing:
> `v_fmax4.ll:4–17`."

Confidence ratings: A = high, B = medium, C = low, D = high.
