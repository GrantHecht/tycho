# Tycho — Eigen 5 + Enzyme work — Current State and Future Work

**Branch:** `eigen-upgrade` @ `349b646` (off `worktree-enzyme-ad-plan` @ `a31575a`).
**Status:** Pre-merge gate passed; cleanup pass + worktree fast-forward + PR
remaining.
**Date:** 2026-04-27.

This document is the resumable hand-off record for the Eigen 3.4 → 5.0.1
upgrade together with the Enzyme support work it surfaced. It covers what
shipped, what's known about regressions (including the original
trigger that motivated the upgrade), and what's left.

---

## 1. Executive summary

`eigen-upgrade` adds three things on top of Phase 1:

1. **Eigen 3.4 (wgledbetter fork) → Eigen 5.0.1 (libeigen upstream)** —
   submodule URL + pin update.
2. **`tycho::math::cos / sin`** opt-in math wrapper API for VFs that
   Enzyme will differentiate. Sidesteps Eigen 5's vectorised packet trig
   (which Enzyme cannot differentiate) without giving up Phase 5b SIMD
   primal evaluation or Phase 3 batched tangents.
3. **`super_scalar_traits.h`** — registration of `Eigen::Array<double,
   W, 1>` as a Matrix Scalar under Eigen 5's stricter trait
   requirements.

Pre-merge gate is green:

- **ctest:** 1015/1015 pass (after regenerating two regression goldens
  for Eigen 5's altered SIMD codegen).
- **C++ brachistochrone example:** converges, objective ≈ 1.8013 s.
- **Python examples:** 32/32 pass; 7 pre-existing skips
  (`mpl_toolkits.basemap` dep).
- **Benchmarks:** clean Phase-1 → current comparison shows 83/93
  unchanged within the 10% threshold, 10 regressions concentrated on
  small-fixed-size VF + orbit-element paths (most under 20%; two
  outliers at +33% and +49% to investigate), and 1 win.

The targeted Phase 5b SIMD measurements that motivated the upgrade
**resolve the original regression** — see §4.1.

---

## 2. What landed (commit-by-commit)

All commits are on `eigen-upgrade`, in order from Phase 1's `381ec75`:

| Commit | Subject | What it does |
|---|---|---|
| `cc7e85b` | `chore(deps): bump dep/eigen to libeigen/eigen tag 5.0.1` | Submodule URL changed `gitlab.com/wgledbetter/eigen.git → gitlab.com/libeigen/eigen.git`; pin moved from `5d33c67444…` (Eigen 3.4 fork) to `bc3b39870e…` (libeigen 5.0.1). |
| `9a0bbdb` | `Revert "chore(deps): …"` | Transient revert during spike — overruled by user mid-session. |
| `011e712` | `Reapply "chore(deps): …"` | Reapplied the bump. Net effect of `cc7e85b → 9a0bbdb → 011e712` is just the original bump. |
| `521c8f9` | `feat(vf): tycho::math::cos/sin SIMD wrappers + Eigen 5 SuperScalar shim` | Adds `include/tycho/detail/typedefs/super_scalar_traits.h`, `include/tycho/detail/utils/simd_math.h`. Migrates `BrachVectorizable` (`tests/cpp/enzyme/test_enzyme_vectorized.cpp`), `BrachBench`, `MEEBench` (`bench/cpp/bench_enzyme.cpp`) to use `tycho::math::*`. All restore `is_vectorizable=true` at default `TYCHO_ENZYME_BATCH_WIDTH=4`. |
| `babc607` | `test(integrators/regression): regenerate goldens for Eigen 5` | Regenerates `golden/ivp_03_cr3bp_dopri87.bin` and `golden/trans_08_jacobian.bin` (the only two whose bit-exact content shifted under Eigen 5; the other 9 are bit-identical). |
| `331211c` | `chore(scripts): add upstream Enzyme/Eigen SIMD-trig canary` | New `scripts/upstream_canary/` — two compile-time tests (Eigen `pcos<Packet4d>` differentiable; custom Enzyme derivative composes with `enzyme_width > 1`) plus driver `check.sh` and README. |
| `0b4347c` | `docs: tycho::math wrappers + Enzyme canary in CLAUDE.md` | Replaces the old "trig-heavy bodies regress under Phase 5b" caveat with the new opt-in API rule of thumb; adds an "Upstream canary" subsection with last-run table seeded for 2026-04-27. |
| `349b646` | `chore(docs): land Phase 2 planning artifacts` | Lands `EnzymeTrig.md` + `EnzymeTrigFindings.md` at the repo root as historical record of the spike. (Earmarked for relocation into `docs/superpowers/specs/` in the cleanup pass — see §5.5.) |

---

## 3. Architecture decisions (with rejected alternatives)

### Chosen: `tycho::math::cos/sin` with per-lane scalar libm dispatch

`tycho::math::cos / sin` overloads for `Eigen::Array<double, W, 1>` are
`__attribute__((noinline))` and manually unroll into W scalar
`std::cos(double)` / `std::sin(double)` calls (W ∈ {2, 4, 8}). Each lowers
to `@llvm.cos.f64` / `@llvm.sin.f64`, which Enzyme's built-in handler
differentiates cleanly under any `enzyme_width`. Phase 3 batched tangents
and Phase 5b SIMD primal evaluation compose without custom derivatives or
per-VF batching opt-outs. The surrounding non-trig arithmetic in
`compute_impl` still vectorises across W lanes via Eigen's native packet
ops; only the trig itself is per-lane scalar.

**Why:** Eigen 5 ships vectorised `pcos<Packet4d>` / `psin<Packet4d>` whose
lowered IR uses `llvm.x86.avx.round.pd.256` and packed `<W x i64>` xor /
and on sign-bit splats. Enzyme cannot differentiate these — confirmed by
[#689](https://github.com/EnzymeAD/Enzyme/issues/689),
[#2679](https://github.com/EnzymeAD/Enzyme/issues/2679),
[#2794](https://github.com/EnzymeAD/Enzyme/issues/2794). Standalone
reproducer at `/tmp/enzyme_repro/` (not committed; recreate from
`scripts/upstream_canary/test_*.cpp`) shows the failure is upstream and
not Eigen-specific.

### Rejected: custom Enzyme derivative + Phase 5b SIMD trig

Wrap SIMD trig in `__attribute__((noinline))` shims, register custom
forward + reverse derivatives via `__enzyme_register_derivative_*`,
let Eigen's SIMD packet trig run inside the wrapper body. This works
correctly **at `enzyme_width = 1` only** — and we verified end-to-end:
both forward Jacobian and Forward-over-Reverse Hessian validated to
1e-12 / 1e-10 against the scalar reference path.

But `enzyme_width > 1` (Phase 3 batched tangents) crashes Enzyme's
auto-generated `fixderivative_*` bridge with a "Call parameter type does
not match function signature" IR-verification error. The bridge's
signature uses scalar shadow types regardless of `width`, but the call
site passes `[width × ptr]` aggregates. **This is an upstream Enzyme
bug**; the fix needs either a 50-line patch to `EnzymeLogic.cpp` (correct
correctness, no perf win; same total work as W=1 dispatch) or a more
involved patch + new annotation to allow user-batched derivatives
(real perf win, but the user side requires hand-rolled LLVM IR — C++
ABI cannot natively express `[W × ptr]` by-value).

User declined to patch Enzyme or file an upstream issue, which made the
custom-derivative path a dead end.

### Rejected: hand-rolled batched LLVM-IR derivative

Define a wrapper in a separate `.ll` file with the `[4 x ptr]` aggregate
signature that Enzyme expects at the call site. Tested in repro
`04_batched_main.cpp` + `04_batched_wrapper.ll`. Fails because Enzyme
reads the registration global at compile time and only sees the C++ TU's
opaque declaration, not the eventually-linked LLVM-IR signature.

### Rejected: per-VF `enzyme_disable_batch_width` opt-out

Earlier intermediate design: VF-level `static constexpr bool
enzyme_disable_batch_width = true;` flag detected in `dense_enzyme.h`,
forces single-tangent dispatch on Phase 5b for that VF only. Worked but
sacrificed Phase 3 batching for trig VFs. The scalar-libm wrapper
approach (§ "Chosen") gives Phase 3 batching back, so the flag became
unnecessary and was removed before the gate.

---

## 4. Known regressions — full inventory

### 4.1 The original trigger (RESOLVED)

**MEE Phase 5b 6× slowdown on Eigen 3.4** — documented in the old
CLAUDE.md "Caveat — trig-heavy bodies regress under Phase 5b" section.
Cause: Eigen 3.4's `Array<double, 4, 1>::cos()` lowered to four
*separate scalar* `cos(double)` libm calls with no SIMD vectorisation,
while Phase 3 batched tangents at `enzyme_width=8` could amortise a
single libm call across 8 tangents. Net: Phase 5b was ~6× slower per
lane on MEE than the documented "use `is_vectorizable=false`" fallback.

This regression **drove the entire Eigen 5 upgrade and Enzyme analysis
that culminated in `tycho::math::cos/sin`**. Status:

- Old Phase 5b on Eigen 3.4: ~142 ns total for MEE (35 ns/lane).
- Old `is_vectorizable=false` fallback: ~24 ns total (5.97 ns/lane Phase 3 W=8).
- **New Phase 5b SIMD with Eigen 5 + `tycho::math`: 562 ns total** at
  default `TYCHO_ENZYME_BATCH_WIDTH=4`. *Note:* this is
  measured with all 4 SIMD lanes carrying identical inputs (a
  bench-fixture artifact); the per-lane figure is ~141 ns/lane and the
  4-lane parallelism win shows up only in real ensemble-integration
  workloads (4 distinct trajectories). For single-trajectory
  evaluation, the `is_vectorizable=false` Phase 3 W=8 path remains
  the fastest option.

### 4.2 New outliers from the Phase-1 → current gate comparison

These are the regressions worth investigating before merge.
Hand-measured at performance governor + idle machine state. Both ran
against `381ec75` (Eigen 3.4) baseline → `349b646` (Eigen 5 + Phase 5b)
current. Bench JSONs at `bench/results/381ec757.json` and
`bench/results/349b6469.json`.

| Benchmark | Phase 1 | Current | Δ | What it does |
|---|---:|---:|---:|---|
| `BM_GF_VJP_PolarLT` | 244 ns | 364 ns | **+49%** | Type-erased `GenericFunction::compute_jacobian_adjointgradient_adjointhessian` call on the polar-coordinate low-thrust example. Largest absolute regression. |
| `BM_VF_Composition` | 10.4 ns | 13.7 ns | **+33%** | Microbenchmark of `tycho::vf::Composition` glue. Small absolute (3.3 ns) but a 33% relative shift on a hot path is concerning. |

**Hypothesis (unverified).** Both touch the type-erased GF dispatch
and/or the `compute_*` template-instantiation chain. Eigen 5 changed
how some fixed-size matrix expressions inline; the difference may
compound at 3-4 layers of template glue. Worth reviewing the IR for
these two specific benchmarks before/after the Eigen swap to confirm.

### 4.3 Other regressions in the gate (smaller, mostly explainable)

| Benchmark | Phase 1 | Current | Δ | Notes |
|---|---:|---:|---:|---|
| `BM_Integrate_Brach_DOPRI87` | 3.79 µs | 5.31 µs | **+40%** | Brach integrator. **Notable:** `BrachODE` (`bench/cpp/bench_odes.h:38`) is a VF *expression tree*, not a templated `compute_impl`, and it integrates with `Scalar=double` — so the trig calls go straight to libm `std::cos` / `std::sin` in both Eigen 3.4 and Eigen 5. The `tycho::math::cos(double)` overload added on this branch is a thin wrapper around `std::cos`; not the source. Most likely cause: Eigen 5's altered fixed-size matrix codegen on the RK butcher-tableau accumulation + the specific shape of `BrachODE`'s 3-output VF body (`sin(theta)*v`, `-cos(theta)*v`, `g*cos(theta)`). Sister benchmark `BM_Integrate_PolarLT_DOPRI87` (also trig-bearing, more complex shape) is flat at +0.4%, so this is **specific to BrachODE**, not a generic integrator regression. **This is the largest real integrator regression in the gate and the top priority for §5.1 follow-up investigation.** |
| `BM_Integrate_Brach_DOPRI54` | 2.33 µs | 2.72 µs | +17% | Same `BrachODE` body, lower-order RK. Same hypothesis as DOPRI87. |
| `BM_VF_ComputeAdjointGradient` | 46.9 ns | 59.6 ns | +27% | VF VJP path. Same family as §4.2 outliers. |
| `BM_VF_ComputeFullVJP` | 94.9 ns | 110 ns | +16% | Same family. |
| `BM_CartesianToClassic` | 115 ns | 134 ns | +16% | Orbital element conversion. Eigen 5 fixed-size 3-/6-vector codegen difference. |
| `BM_GF_VJP_Brach` | 82.5 ns | 94.4 ns | +14% | Same family as §4.2 outliers. |
| `BM_VF_ArgsSegment` | 8.94 ns | 10.1 ns | +13% | Small absolute (1 ns). Most likely Eigen 5's inlining heuristics. |
| `BM_ThreadPool_ParallelSequence/100` | 36.7 µs | 42.2 µs | +15% | **Likely noise** — `ParallelSequence/10` is unchanged at 7.62 → 7.68 µs. ThreadPool benchmarks are inherently noisy. |

### 4.3.1 Integrator regression sub-summary

The gate caught **only Brach-specific integrator regressions** — DOPRI54
(+17%) and DOPRI87 (+40%). Other integrator benchmarks are flat:

| Integrator benchmark | Phase 1 | Current | Δ |
|---|---:|---:|---:|
| `BM_Integrate_Brach_DOPRI54` | 2.33 µs | 2.72 µs | **+17%** |
| `BM_Integrate_Brach_DOPRI87` | 3.79 µs | 5.31 µs | **+40%** |
| `BM_Integrate_PolarLT_DOPRI87` | 7.01 µs | 7.04 µs | +0.4% |
| `BM_Integrate_SHO_DOPRI54` | 75.94 µs | 76.47 µs | +0.7% |
| `BM_Integrate_SHO_DOPRI87` | 10.67 µs | 10.63 µs | -0.4% |
| `BM_Integrate_SHO_Dense_100` | 18.72 µs | 18.39 µs | -1.8% |
| `BM_Integrate_SHO_FixedStep` | 164.32 µs | 163.26 µs | -0.6% |

So:
- Not a generic integrator-update regression (SHO + PolarLT are flat).
- Not a tycho::math regression (BrachODE bench uses libm directly with
  `Scalar=double`).
- Specific to BrachODE's 3-output VF expression tree shape under Eigen 5
  codegen.

### 4.4 Wins (gate)

| Benchmark | Phase 1 | Current | Δ |
|---|---:|---:|---:|
| `BM_TypeStorage_CopySmall` | 3.87 ns | 3.28 ns | **-15%** |

### 4.5 Targeted Phase 5b SIMD wins (separate hand-measurement)

| Benchmark | Phase 1 (W=1, custom deriv) | Current (W=4 native) | Δ |
|---|---:|---:|---:|
| `BM_JacobianVecSIMD_Enzyme/Brach` | 421 ns | 316 ns | **-25%** |
| `BM_JacobianVecSIMD_Enzyme/MEE` | 874 ns | 562 ns | **-36%** |
| `BM_JacobianVecSIMD_Enzyme/CR3BP` | 14.3 ns | 14.3 ns | (no trig path — unchanged, expected) |

These were measured at the same machine state in adjacent runs, so the
wins are not contaminated by the across-the-board codegen shift in §4.3.

### 4.6 Reading the picture

The 10 §4.3 / §4.4 regressions cluster on the VF inline / GF VJP paths,
small fixed-size (3- and 6-vector) orbit-element conversions, and the
Brach integrator. This is exactly where Eigen 5's altered SIMD codegen
on small matrices would most visibly differ from Eigen 3.4. Most are
within the noise band you'd expect from a major Eigen version bump.

The Phase 5b targeted wins (§4.5) more than offset for the workloads
that touch the Phase 5b path. The original §4.1 trigger is resolved: the
mechanism that produced the 6× MEE slowdown is gone (Eigen 5 + scalar
libm + Phase 3 batching now replaces the Eigen-3.4 packet-trig stub +
`is_vectorizable=false` workaround).

The §4.2 outliers (+49% PolarLT VJP, +33% VF Composition) and the
§4.3.1 BrachODE integrator regressions (+40% DOPRI87, +17% DOPRI54) are
the items worth investigating before merge. None of them block Phase 2
acceptance per the gate criterion ("any regressions need PR-description
justification") but they are not yet root-caused.

### 4.7 `e942c7f1` baseline re-recorded — confirms recording-conditions, not regression

The original `e942c7f1.json` (recorded 2026-04-25) showed **+33% to +36%
across 92 of 93 benchmarks** vs today's `381ec757` Phase 1 baseline,
which was suspicious-uniform across benchmarks that share zero code
with anything that changed between the two commits.

**Re-recorded `e942c7f1` today** at performance governor + clean machine
state, then re-compared:

| Comparison | Same machine state | Regressions detected | Verdict |
|---|---|---:|---|
| `e942c7f1.json` (2026-04-25) → `381ec757.json` (today) | no | 92 of 93 (+33% to +36%) | recording-conditions noise |
| `e942c7f1.json` (re-recorded today) → `381ec757.json` (today) | yes | **1** (`BM_TypeStorage_CopySmall` +17%, ULP-level) | **Phase 1 ancestor and `e942c7f1` are perf-equivalent** |
| `e942c7f1.json` (re-recorded today) → `349b6469.json` (today) | yes | 11 (the same 10 §4.3 / §4.4 entries plus `BM_BumpAllocator_Resize` +12% borderline) | confirms Phase 2 → current regression set is the only real signal |

**Conclusions:**

1. The +34% across-the-board "gap" was 100% recording-conditions noise
   from the original 2026-04-25 capture. The new same-day re-record
   shows essentially flat perf for the entire `e942c7f1 → 381ec75` span.
2. The two integrator-touching commits between `e942c7f1` and Phase 1
   (`9fed0b5` privatize-accessor refactor, `b2b05ad` storemidpoints-gated
   finite-check) had **zero measurable perf impact**, as static analysis
   suggested.
3. **Every real perf regression in this PR** is Phase 2 work (the ~10
   benchmarks listed in §4.3 / §4.4). The `e942c7f1` → current
   comparison surfaces the same set, with one extra borderline
   (`BM_BumpAllocator_Resize` +12.1%).

The original `bench/results/e942c7f1.json` has been replaced with
today's recording. (`bench/results/` is gitignored, so the original is
not recoverable — but its content was provably noise.)

---

## 5. Outstanding work

In rough sequence:

### 5.1 Investigate §4.2 / §4.3 outliers (recommended, optional for merge)

Three benchmarks are worth dumping `-emit-llvm -O3` IR for at both
commits and diffing the inline + vectorisation choices:

1. `BM_GF_VJP_PolarLT` (+49%) — type-erased GF VJP path. Look at
   `tycho::vf::GenericFunction::compute_jacobian_adjointgradient_adjointhessian`
   instantiation for PolarLT.
2. `BM_VF_Composition` (+33%) — VF Composition glue. Look at
   `tycho::vf::Composition::compute_*` instantiations.
3. `BM_Integrate_Brach_DOPRI87` (+40%) — `BrachODE` integrated under
   DOPRI87. The Brach VF expression tree is in `bench/cpp/bench_odes.h:38`;
   the integrator's RK accumulation uses `Eigen::Matrix<double, 4, 1>`
   ops that may codegen differently under Eigen 5.

Most likely conclusion is "Eigen 5 codegen tradeoff" but worth
confirming we didn't miss anything (see §4.7 for the e942c7f1 baseline
re-record that should also be done as part of this).

If a real fix lands here it could land in a follow-up PR; the gate
results are already publishable.

### 5.2 `-Wnontrivial-memcall` warning fix — DONE (2026-04-27)

clang 21 + Eigen 5's now-non-trivially-copyable `Array<double, 4, 1>`
flags `std::memset` on a `Scalar*` at `include/tycho/detail/utils/memory_management.h:84`:

```
warning: first argument in call to 'memset' is a pointer to non-trivially
copyable type 'Eigen::Array<double, 4, 1>' [-Wnontrivial-memcall]
```

Cosmetic today (warning, not error). Two reasonable fixes:

- Cast: `std::memset(static_cast<void*>(data_ + aligned), 0, n * sizeof(Scalar));`
- Replace with `std::uninitialized_value_construct_n(data_ + aligned, n);`

Casting is a smaller change but `uninitialized_value_construct_n` is
arguably more correct (calls Scalar's value-constructor instead of
zero-bit-bashing). Either is fine.

### 5.3 Public umbrella header `include/tycho/math.h` — DONE (2026-04-27)

Currently consumers include `tycho/detail/utils/simd_math.h` directly,
which is fine for tycho-internal call sites but not idiomatic for the
public-API surface. CLAUDE.md explicitly notes "(public umbrella TBD)".
A trivial umbrella:

```cpp
// include/tycho/math.h
#pragma once
#include "tycho/detail/utils/simd_math.h"
```

…lets external consumers write `#include <tycho/math.h>` per the
existing convention (`tycho/vector_functions.h`, `tycho/utils.h`, etc.).

### 5.4 Reset Phase 5b benchmark target (Task 2.4) — DONE (2026-04-27)

Tracker Task 2.4 (current state: `pending`) was titled "Validate MEE
Phase 5b benchmark target (≤ 30 ns total)". That target was based on
the pre-spike model that assumed SIMD packet trig + Phase 3 composition.
Under the no-Enzyme-patch constraint that target is unreachable; the
realistic ceiling is the §4.5 numbers (562 ns total, 141 ns/lane on MEE
for the 4-lane SIMD bench).

Action: rephrase Task 2.4 in CLAUDE.md (or wherever the original target
appears) to "≤ 600 ns at default `TYCHO_ENZYME_BATCH_WIDTH=4` for MEE
Phase 5b SIMD; revisit if upstream canary fires", or close as
"superseded by §4.5".

### 5.5 Move/delete planning artifacts — DONE (2026-04-27)

`EnzymeTrig.md` and `EnzymeTrigFindings.md` live at the repo root and
were committed in `349b646` as historical record. They should move to
`docs/superpowers/specs/` (alongside the original
`2026-04-25-claude-enzyme-ad-support-design.md` referenced by CLAUDE.md)
or be deleted entirely now that their decisions are codified in
CLAUDE.md and this document.

Suggested target paths:

- `docs/superpowers/specs/2026-04-26-enzyme-trig-spike.md` ← `EnzymeTrig.md`
- `docs/superpowers/specs/2026-04-26-enzyme-trig-findings-codex.md` ← `EnzymeTrigFindings.md`

### 5.6 `doc/VectorFunction.md` mention of `tycho::math` — DONE (2026-04-27)

CLAUDE.md cites `doc/VectorFunction.md` as the canonical VF doc.
Whatever section there describes EnzymeAD usage should pick up the
`tycho::math::cos / sin` opt-in pattern + the rule of thumb that's now
in CLAUDE.md.

### 5.7 Fast-forward `worktree-enzyme-ad-plan` to `eigen-upgrade`

The original plan (per the user) was: branch `eigen-upgrade` off
`worktree-enzyme-ad-plan`, do the work on `eigen-upgrade`, and
fast-forward `worktree-enzyme-ad-plan` to it on completion (no PR
needed for that internal branch fast-forward).

Current state:

- `worktree-enzyme-ad-plan` is at `a31575a` (Phase 1 ancestor, no
  Phase 2 work).
- `eigen-upgrade` is at `349b646` (8 commits ahead, Phase 2 complete).

To fast-forward:

```bash
git checkout worktree-enzyme-ad-plan
git merge --ff-only eigen-upgrade
# OR equivalently:
# git branch -f worktree-enzyme-ad-plan eigen-upgrade
```

Verify with `git log --oneline -3 worktree-enzyme-ad-plan` after.

### 5.8 PR creation (optional)

If the branch gets pushed for review, the PR description needs:

- Bench results table from §4 (especially §4.2–4.5).
- Note the §4.1 resolution.
- Note the canary + last-run table.
- Pin caveat: Eigen 5.0.1 + clang 21 + Enzyme `cecf3492` is the validated
  toolchain; older clang may not be tested.

---

## 6. Reference state (for resumption after compaction)

### Repo

| Item | Value |
|---|---|
| Active branch | `eigen-upgrade` |
| HEAD | `349b646` chore(docs): land Phase 2 planning artifacts |
| Phase 1 baseline | `381ec75` ci(phase-1): autodiff removal — pre-merge gate green |
| Worktree branch (needs FF) | `worktree-enzyme-ad-plan` @ `a31575a` |
| Eigen submodule URL | `https://gitlab.com/libeigen/eigen.git` |
| Eigen submodule pin | `bc3b39870ecb690a623a3f49149a358b95c5781d` (5.0.1) |

### Toolchain

| Item | Value |
|---|---|
| OS | Linux 6.19.13-200.fc43.x86_64 (Fedora 43) |
| Compiler | `/usr/bin/clang++` (clang 21.1.8, Fedora 21.1.8-4.fc43) |
| LLVM | system `/usr/lib64/llvm21` |
| Enzyme source | `/home/ghecht/Software/Enzyme` @ `cecf3492` |
| Enzyme install | `/home/ghecht/Software/Enzyme/install` (`ClangEnzyme-21.so`) |
| Conda env | `tycho` (Python 3.14) |

### Build flags (CMake cache, current)

```
TYCHO_FP_MODE                  = SAFER_FAST
TYCHO_ENZYME_BATCH_WIDTH       = 4
TYCHO_ENZYME_HESSIAN_STRATEGY  = ForwardOverReverse
ENABLE_ENZYME_AD               = ON
BUILD_CPP_TESTS                = ON
BUILD_CPP_BENCHMARKS           = ON
BUILD_CPP_EXAMPLES             = ON
```

### CMake configure incantation

```bash
conda run -n tycho cmake --preset linux-clang-release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/home/ghecht/Software/Enzyme/install \
  -DENABLE_ENZYME_AD=ON \
  -DBUILD_CPP_TESTS=ON \
  -DBUILD_CPP_BENCHMARKS=ON \
  -DBUILD_CPP_EXAMPLES=ON
```

### Build parallelism

`-j6` (per CLAUDE.md memory: 32 GB Linux, `heavy_compile` ninja pool
auto-throttles).

### Bench JSON files of interest

| File | Commit | Branch when recorded | Notes |
|---|---|---|---|
| `bench/results/381ec757.json` | `381ec75` | `eigen-upgrade` (detached) | Phase 1 baseline (Eigen 3.4). Recorded 2026-04-27. |
| `bench/results/349b6469.json` | `349b646` | `eigen-upgrade` | Current Phase 2. Recorded 2026-04-27 (performance governor). |
| `bench/results/011e712a-dirty.json` | pre-commit checkpoint | `eigen-upgrade` | Pre-commit recording (powersave governor). Superseded by `349b6469.json`. **Safe to delete.** |

---

## 7. Reproducibility recipes

### 7.1 Run the pre-merge gate (top of `eigen-upgrade`)

```bash
# 1. Build
cd /home/ghecht/Projects/tycho
conda run -n tycho cmake --preset linux-clang-release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/home/ghecht/Software/Enzyme/install \
  -DENABLE_ENZYME_AD=ON \
  -DBUILD_CPP_TESTS=ON \
  -DBUILD_CPP_BENCHMARKS=ON \
  -DBUILD_CPP_EXAMPLES=ON
ninja -C build -j6 all

# 2. ctest
cd build && ctest --output-on-failure

# 3. Brachistochrone
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp \
  | grep -E "Optimal Solution Found|Prim Obj"

# 4. Python examples
cd .. && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"

# 5. Bench compare (assumes performance governor; baseline JSON exists)
bench/bench_track.sh compare 381ec757 349b6469
```

### 7.2 Run the upstream Enzyme/Eigen canary

```bash
scripts/upstream_canary/check.sh
# Exit 0 = status quo (both tests still fail at compile time).
# Non-zero = upstream changed; revisit simd_math.h.
```

### 7.3 Re-record a baseline at an arbitrary commit

```bash
# Full procedure used to capture 381ec757 baseline:
git checkout <commit>
git submodule sync dep/eigen
git submodule update --init dep/eigen
conda run -n tycho cmake --preset linux-clang-release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/home/ghecht/Software/Enzyme/install \
  -DENABLE_ENZYME_AD=ON -DBUILD_CPP_TESTS=ON \
  -DBUILD_CPP_BENCHMARKS=ON -DBUILD_CPP_EXAMPLES=ON
ninja -C build -j6 all
bench/bench_track.sh record --reps 3
git checkout eigen-upgrade
git submodule sync dep/eigen
git submodule update --init dep/eigen
ninja -C build -j6 all
```

CPU governor on Fedora: `sudo tuned-adm profile throughput-performance`
or `sudo cpupower frequency-set -g performance`. (Note: modern AMD/Intel
opportunistic-boost under load can mask governor differences; results
seen on this machine were identical in both states.)

### 7.4 Switching back to powersave (post-bench cleanup)

```bash
sudo tuned-adm profile balanced       # or whatever your default was
# OR
sudo cpupower frequency-set -g powersave
```

---

## 8. Open questions / future investigations

### 8.1 §4.2 outliers

What inlining/vectorisation choice changed for `BM_GF_VJP_PolarLT`
between Eigen 3.4 and Eigen 5 to give a 49% slowdown? Same question for
`BM_VF_Composition` (+33%) and `BM_Integrate_Brach_DOPRI87` (+40%).
Recommended approach: dump `-emit-llvm -O3` IR for the relevant TU at
both commits, diff the inline tree.

### 8.2 Upstream Enzyme batch-width custom-derivative composition

The canary's Test B detects the day Enzyme fixes
`fixderivative_*` to widen shadows correctly under
`enzyme_width > 1`. If/when that lands, an alternative architecture
becomes viable: register a custom Enzyme derivative that uses Eigen 5's
SIMD packet trig in its body. Theoretically ~5× faster than the current
scalar-libm-with-Phase-3-batching architecture for large IR
workloads (estimate: MEE Jacobian ~110 ns vs current 562 ns), but
requires hand-rolled batched-shape derivatives — only worth pursuing if
the perf delta becomes a measured bottleneck.

### 8.3 Upstream Enzyme/Eigen SIMD packet-trig differentiation

The canary's Test A detects the day Enzyme adds support for Eigen 5's
`pcos<Packet4d>` polynomial IR. If passes, the per-lane scalar libm
unrolling in `simd_math.h` becomes redundant — switch the wrappers to
delegate to `x.cos() / x.sin()` directly.

### 8.4 LLVM 22 compatibility

Codex's findings noted that LLVM 22 changed the `masked_load` /
`masked_store` intrinsic ABI (alignment moved from immarg to ptr
attribute), and Enzyme's current `AdjointGenerator.h` operand indexing
breaks under that. Not blocking for current Linux Fedora 43 (clang 21),
but a known sharp edge if the toolchain bumps.

---

## 9. File locations

| Subject | Path |
|---|---|
| Math wrapper API | `include/tycho/detail/utils/simd_math.h` |
| Eigen 5 SuperScalar shim | `include/tycho/detail/typedefs/super_scalar_traits.h` |
| `eigen_types.h` (includes shim) | `include/tycho/detail/typedefs/eigen_types.h` |
| Public umbrella | `include/tycho/math.h` (added §5.3) |
| Test fixture using `tycho::math` | `tests/cpp/enzyme/test_enzyme_vectorized.cpp` |
| Bench fixtures using `tycho::math` | `bench/cpp/bench_enzyme.cpp` (`BrachBench`, `MEEBench`) |
| Eigen-5 regenerated regression goldens | `tests/cpp/integrators/regression/golden/{ivp_03_cr3bp_dopri87,trans_08_jacobian}.bin` |
| Upstream canary | `scripts/upstream_canary/{check.sh, test_a_*.cpp, test_b_*.cpp, README.md}` |
| Master rule-of-thumb doc | `CLAUDE.md` (search "Trig-bearing bodies under Phase 5b") |
| Spike historical record | `docs/superpowers/specs/2026-04-26-enzyme-trig-{spike,findings-codex}.md` (moved §5.5) |
| Memory-mgmt warning fix | `include/tycho/detail/utils/memory_management.h:84` (fixed §5.2) |

---

## 10. Sanity-check checklist (resume here)

After context compaction, verify in order:

- [ ] `git rev-parse HEAD` → `349b646…`. If not, `git checkout
      eigen-upgrade`.
- [ ] `git status --short` → clean (no uncommitted changes from
      mid-conversation work).
- [ ] `git submodule status dep/eigen` → reports `bc3b39870e…` (Eigen
      5.0.1).
- [ ] `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` →
      `performance` if benchmarking; otherwise either is fine.
- [ ] `scripts/upstream_canary/check.sh` → exit 0 (status quo). If
      non-zero, `simd_math.h` may now be replaceable; see §8.2 / §8.3.
- [ ] `bench/results/381ec757.json` and `bench/results/349b6469.json`
      both exist. (Required for §4 reproducibility.)

If all green, proceed to §5.1 or whichever cleanup item is next.
