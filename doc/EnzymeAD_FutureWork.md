# EnzymeAD — Future Work

This document records the genuine future-work items remaining on the EnzymeAD
derivative-mode subsystem after the initial five-phase rollout (see
`include/tycho/detail/vf/derivatives/dense_enzyme.h` and CLAUDE.md for the
shipping state).

The five-phase plan executed all 38 original tasks. Phase 3 (batched forward
mode via `enzyme_width`) and Phase 5b (direct-SIMD differentiation of
SuperScalar bodies) were initially off-ramped and later recovered after
identifying the actual Enzyme calling convention. The remaining items below
are genuine extensions, not bug fixes.

---

## 1. SIMD trig overload for `Eigen::Array<double, W, 1>` — RESOLVED (2026-04-27)

> **Resolution.** The Eigen 3.4 → 5.0.1 upgrade plus the opt-in
> `tycho::math::cos / sin` wrapper API ships in `eigen-upgrade`. Eigen 5
> *does* emit vectorised `pcos<Packet4d>` / `psin<Packet4d>` for the
> analytic-integration paths that need it; it just turns out Enzyme can't
> differentiate the resulting bithack IR yet (issues
> [#689](https://github.com/EnzymeAD/Enzyme/issues/689),
> [#2679](https://github.com/EnzymeAD/Enzyme/issues/2679),
> [#2794](https://github.com/EnzymeAD/Enzyme/issues/2794)). VFs that pair
> trig with `EnzymeAD` route their `cos`/`sin` calls through
> `tycho::math::*` (umbrella `<tycho/math.h>`); the wrappers unroll into
> per-lane scalar libm calls Enzyme handles cleanly under any
> `enzyme_width`. Net Phase 5b numbers: MEE 562 ns total at W=4 (down
> from prior 874 ns W=1 fallback), Brach 316 ns (down from 421 ns).
> See `CurrentStateAndFutureWork.md` §3 (architecture decisions) and
> `docs/superpowers/specs/2026-04-26-enzyme-trig-spike.md` for the spike
> writeup. The compile-time canary at `scripts/upstream_canary/` detects
> when Enzyme adds support for SIMD packet trig — at which point the
> wrappers can be replaced with direct `x.cos() / x.sin()` calls.

**Historical context (superseded by `tycho::math` wrappers in
`include/tycho/detail/utils/simd_math.h`).** Eigen 3.4's
`Array<double, 4, 1>::cos()` / `::sin()` / `::exp()` / `::log()` were
unrolled per-element and lowered to four separate scalar libm calls
(`@cos(double)` × 4, etc.) rather than a single vectorised
`cos<4xdouble>` intrinsic. This killed Phase 5b SIMD performance on
bodies whose math involves transcendentals — on MEE (9→6, one cos +
one sin per body), the scalar Phase 3 path with `enzyme_width=8` was
5.97 ns vs Phase 5b SIMD at 142 ns. Adding `-fveclib=libmvec` made
MEE Phase 5b *worse* (319 ns) because Eigen unrolls before Clang's
vector-loop pass runs. Both attempts reverted; resolved by Eigen 5
upgrade + the per-lane scalar-libm `tycho::math::cos/sin` wrappers
shipping in `<tycho/math.h>`.

---

## 2. Direct-SIMD Hessian path — RESOLVED for FoR (2026-04-28)

**Status.** Phase 6 ships direct-SIMD Forward-over-Reverse Hessian for
`is_vectorizable=true` EnzymeAD VFs at SuperScalar input.  Gated by
`TYCHO_ENZYME_SIMD_HESSIAN` (default ON); flag OFF falls back to the
Phase 5a scalarize-per-lane path.  `ForwardOverReverse` is the only
cmake-selectable Hessian strategy; the Phase 7 / 7+ Forward-over-Forward
prototype was archived (see item 3 below).

**What landed.**

- `enzyme_for_outer_wrapper_simd<Derived, SSType>` — SIMD twin of
  the scalar FoR wrapper.  Inner `__enzyme_autodiff` runs reverse mode
  on SuperScalar arithmetic, propagating W lane-local cotangents.
- `compute_adjoint_hessian_for_simd_` — SS-typed FoR Hessian helper.
  Outer `__enzyme_fwddiff` with `enzyme_width=BW` chains Phase 3
  batching with Phase 5b SIMD, producing BW Hessian columns per call.
- `simd_compute_jacobian_adjointgradient_adjointhessian_impl` — public
  SIMD entry; tests call it directly to A/B against Phase 5a in the
  same binary.
- Per-VF opt-out via `tycho::vf::EnzymeSimdHessianUnsupported` marker
  base class — for VFs whose body trips Enzyme's SS reverse-mode IR
  type-deduction (composite trig+sqrt+division composites; MEE is the
  canonical case in tycho today).  The dispatch consults
  `enzyme_simd_hessian_supported_v<Derived>` (true unless Derived
  inherits from the marker).
- Tests `EnzymeVectorized.HessianSIMDMatchesScalarized_{Brach,CR3BP}`
  validate the SIMD path matches the per-lane scalar reference to
  1e-10.

**Phase 7 follow-on archived.**  FoF SIMD + combined J+H was prototyped
in Phase 7 / 7+ and is preserved as reference code in `dense_enzyme.h`
(see item 3 below); it is no longer cmake-selectable.  FoR-SIMD wins on
every measured non-MEE body, and MEE-class workloads do not dominate
any current PSIOPT use case.

**Open item — MEE-class fallback.**  The per-VF opt-out is shipped, but
the upstream Enzyme limitation that forces it is real: composite bodies
with cos/sin/sqrt/division saturate the type-deduction tape under
SuperScalar reverse mode.  Track upstream Enzyme issues; if a future
release makes SS reverse-mode robust to these patterns, the opt-out can
be removed.

---

## 3. FoF Hessian path — ARCHIVED (research scaffolding only)

**Status (2026-04-28).** FoF (Forward-over-Forward) Hessian SIMD with
combined-J+H output, plus the doubly-batched variant, were prototyped
in Phase 7 / 7+ and have been **archived as reference code in
`include/tycho/detail/vf/derivatives/dense_enzyme.h`**.  All FoF tests
and benchmarks have been removed and `ForwardOverForward` is no longer
cmake-selectable — the helpers sit inside
`#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)` blocks
that the build never defines.  Reviving requires re-adding the cmake
strategy STRINGS entry and the dispatch branches stripped from
`dense_enzyme.h`'s scalar and SIMD dispatch sites.

**What was prototyped.**

- Scalar combined J+H FoF helper that reads column i of J from the
  outer fwddiff's fx-shadow (previously discarded) — saves the up-front
  Phase-1 forward sweep.
- SIMD combined J+H FoF helper (singly-batched outer at width BW).
- Doubly-batched FoF SIMD helper (BW outer × BW inner = BW² Hessian
  elements per outer call); confirms Enzyme accepts nested
  `__enzyme_fwddiff(enzyme_width)` composition.
- Per-VF MEE-class inner wrapper variant for the trig+sqrt+division
  body that fails Enzyme's TypeAnalysis at -O3 (workaround: separate
  TU at -O1).

**Why archived.**  Bench numbers at archival (BW=4, AVX2):

| VF      | FoR-SIMD | FoF-SIMD (singly) | FoF-SIMD (doubly) |
|---------|----------|-------------------|-------------------|
| Brach   |  1038 ns |  2102 ns          | n/a (IR=5 mod 4)  |
| CR3BP   |   291 ns |  1545 ns          | n/a (IR=7 mod 4)  |
| MEE     | fallback |  9812 ns @ -O1    | n/a (IR=9 mod 4)  |
| Poly8x4 |    n/a   |   769 ns          |   727 ns (-5%)    |

FoR wins decisively on every body where it can run.  MEE has no real
PSIOPT workload demanding SIMD Hessian today.  The doubly-batched
variant proved the W² claim but yielded only ~5% — per-call Enzyme
overhead is not the dominant cost; body work is (same conclusion as
item 4 below).

**When to revive.**
1. **MEE-class workload dominance.**  Real workload's runtime
   dominated by composite trig+sqrt+division Hessians where FoR-SIMD
   can't go SIMD.  Compile MEE-class TUs at -O1 (see archived header
   docstrings).
2. **Per-call overhead becomes bottleneck.**  Future Enzyme/clang
   reduces body cost so per-call dominates.
3. **Cache-locality in fused J+H consumer.**  Solver reads J and H
   in the same iteration; combined-J+H FoF saves a redundant
   forward sweep + can produce both in one cache window.

**Reference commits** for revival (annotated git tags pinned locally; push
when this branch lands so collaborators can resolve them):
- tag `enzyme-fof-phase7` (commit `e8651d2`) — Phase 7 direct-SIMD FoF
  Hessian + combined J+H
- tag `enzyme-fof-mee-bench-split` (commit `4cd7391`) — MEE FoF SIMD
  Hessian bench TU split for -O1
- tag `enzyme-fof-phase7plus` (commit `e777544`) — Phase 7+ doubly-batched
  FoF SIMD helper
- tag `enzyme-fof-archive-2026-04-28` (commit `79fc589`) — archive point
  where the FoF tests/benches were removed

Test fixtures `MEEVectorizable` and `PolyVectorizable8x4` and benches
`BM_Poly8x4_HessianFoFSIMD_singly|doubly` were removed at the
`enzyme-fof-archive-2026-04-28` tag; the FoF-strategy registration of
`BM_MEE_HessianSIMD_Enzyme` was also dropped (the bench itself still
ships under the FoR strategy via the Phase 5a fallback).  Cherry-pick
the fixtures and benches from the reference tags above when reviving.

---

## 4. `enzyme_width` × Phase 5b composition study

**State today.** `simd_compute_jacobian_impl` already includes a Phase 3
batching prelude (`if constexpr (BW > 1)`) that uses `enzyme_width=BW`
on the SuperScalar-typed wrapper. In bench numbers, this composes
*neutrally* — the combined path measures the same as Phase 5b alone.

**Hypothesis.** Per-call Enzyme overhead is small once the body is in
icache, so reducing call count via Phase 3 doesn't help when Phase 5b's
SIMD work is already saturating the body cost. On larger bodies (>20 ops
per scalar dim, e.g., MEE if item 1 lifts the trig limitation), the
composition might become meaningful again.

**Investigation plan.**

1. Profile a single Phase 5b call vs a single Phase 3+5b combined call
   with `perf` to see where the cycles go.
2. Vary the test body size (synthetic ODE with N arithmetic ops per
   compute_impl, parameter N) to find the crossover point where Phase 3
   batching starts paying off again.
3. If a regime exists where composition wins by ≥1.5×, add a
   per-workload selector or crossover threshold.

**Estimated effort.** 0.5 day. Mostly a measurement exercise.

**Decision criterion.** Low priority unless real workloads land in the
"large body" regime.

---

## 5. Default `TYCHO_ENZYME_BATCH_WIDTH = 8` for IR≥8 workloads

**State today.** The default is `W=4`. W=8 is selectable but silently
falls back to the W=1 tail loop when IR<8 (so doesn't degrade Brach/CR3BP).
On MEE (IR=9), W=8 gives **60× faster Jacobian than the FDiff reference** (5.96 ns
vs 359 ns).

**Proposed change.** Make `W=8` the default. Pros:

- Big win on large-IR workloads; no regression on small-IR (falls
  through to the same tail loop as W=1).
- Aligns with current AVX-512 capability of dev-machine CPUs.

Cons:

- Slight memory bump for shadow buffers (`Eigen::Matrix<double, IR, 8>`
  on stack vs `Eigen::Matrix<double, IR, 4>`). IR=9, OR=6 → 9×8=72 doubles
  + 6×8=48 doubles = 960 bytes, well within stack budget.
- Untested on AVX2-only CPUs (W=8 would force LLVM to lower to two AVX2
  ops; possibly slower than W=4 for AVX2 hardware).

**Decision criterion.** Run the W=4-vs-W=8 sweep on AVX2-only hardware
(if any in the dev fleet). If W=8 doesn't regress AVX2, flip the default.

**Estimated effort.** 1 hour to flip + re-bench, plus the AVX2 box.

---

## 6. Eigen upstream issue — RESOLVED (2026-04-27)

> **Resolution.** Eigen 5.0.1 ships vectorised `pcos<Packet4d>` /
> `psin<Packet4d>` for `Array<double, W, 1>`, addressing the original
> per-element scalar-libm pattern documented here. Code paths *not*
> differentiated by Enzyme (e.g. `include/tycho/detail/astro/mee_dynamics.h`)
> benefit directly. The remaining Enzyme-side limitation is tracked by
> upstream Enzyme issues
> [#689](https://github.com/EnzymeAD/Enzyme/issues/689),
> [#2679](https://github.com/EnzymeAD/Enzyme/issues/2679),
> [#2794](https://github.com/EnzymeAD/Enzyme/issues/2794) and by the
> Tycho-side compile-time canary at `scripts/upstream_canary/`. No new
> Eigen issue needs filing.

The `Array<double, W, 1>::cos()` per-element scalar-libm pattern was a
genuine Eigen ecosystem issue under Eigen 3.4. Eigen 5 fixes it.

---

## 7. `TYCHO_ENZYME_BATCH_WIDTH ∈ {1, 4, 8}` automated correctness coverage

**State today.** Tests run with whatever `TYCHO_ENZYME_BATCH_WIDTH` cmake
selected (default `4`).  The W=1 (no batching) and W=8 paths — and
specifically the W=8 silent fallback when IR<8 (Brach IR=5: `ir_chunked = 0`,
all work routes through the tail loop; CR3BP IR=7: same) — have no
automated correctness gate.  The tail loop is exercised at the default
W=4 (Brach: chunked=4, tail=1) but the all-tail-loop case is not.

**Architectural obstacle.**  `TYCHO_ENZYME_BATCH_WIDTH` is a global
preprocessor define applied via `add_compile_definitions` in
`CMakeLists.txt`, and `src/vf/extern_template_instantiations.cpp`
(`vf_instantiations` static lib) instantiates downstream templates with
that single global value.  Building per-W test executables that link
against `vf_instantiations` would either (a) ODR-violate when the test TU
sees a different W than the static lib, or (b) require the test target
to compile its own copies of all transitively-instantiated templates with
its own W.  Neither is a small change.

**Suggested approach (when revisited).**

1. Refactor `vf_instantiations` so the EnzymeAD-specific extern templates
   live in their own translation unit isolated from the global W
   definition (or are instantiated in headers only, paying the cost
   per-TU).
2. Add three test executables `tycho_enzyme_tests_w{1,4,8}.cpp`, each
   built with `target_compile_options(... PRIVATE -UTYCHO_ENZYME_BATCH_WIDTH
   -DTYCHO_ENZYME_BATCH_WIDTH=${W})` and `DISABLE_PRECOMPILE_HEADERS ON`,
   linking only against the EnzymeAD-clean headers.
3. Each TU asserts Jacobian correctness for the matrix
   {Brach(IR=5), CR3BP(IR=7), MEE(IR=9)} × {W=1, W=4, W=8}, especially
   the W=8/IR<8 silent-fallback cells.

**Estimated effort.** 1 day (the harder half is the cmake refactor for
`vf_instantiations`).

**Decision criterion.**  Pursue if any future change to the chunked-loop
or tail-loop dispatch lands without a regression-catch test, or if a
real workload with IR≥8 ships through PSIOPT and needs W=8 confidence.

---

## 8. End-to-end PSIOPT test for CR3BP under `<EnzymeAD, EnzymeAD>`

**State today.**  `tests/cpp/enzyme/test_enzyme_phase_e2e.cpp` exercises
the full Enzyme pipeline only for brachistochrone (IR=5, OR=3, scalar
trig).  CR3BP (IR=7, OR=6, sqrt-only — the body that actually wins from
Phase 6 SIMD Hessian) has no end-to-end PSIOPT integration test; only
the unit-level Jacobian/Hessian comparison tests.

**Suggested approach.**

1. Define a CR3BP optimal control problem (e.g., minimum-time L1 ⇄ L2
   transfer, or a Halo orbit station-keeping problem).
2. Solve under `<EnzymeAD, EnzymeAD>`, compare convergence + cost
   against the `<FDiffCentArray, FDiffFwd>` reference to 1e-6.

**Estimated effort.** 0.5 day for the OCP setup + verification.

**Decision criterion.**  Pursue if a future PSIOPT internals change lands
that could affect the nested-AD path on real-CR3BP-style workloads
without being caught by the Brach E2E.  The unit tests cover the
correctness invariants; this would catch dispatch-pipeline integration
regressions specifically.

---

## 9. Extract FoF archive into a separate header

**State today.**  The FoF archive lives in two `#if defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)`
blocks inside `include/tycho/detail/vf/derivatives/dense_enzyme.h`:

1. Free-function FoF inner wrappers (lines ~282–365, ~80 lines).
2. FoF Hessian helper member functions inside the
   `DenseSecondDerivatives<..., EnzymeAD>` class body (lines ~987–1395,
   ~410 lines).

The blocks are gated by a sentinel macro that nothing currently defines, so
they are zero-cost in builds.  The header readability cost is real, however:
~500 lines of archive code dominates the live FoR path.

**Architectural obstacle.**  Block 2's helpers are declared as member
functions inside the `DenseSecondDerivatives` class body, so they cannot be
moved into a separate header without first refactoring them to out-of-class
definitions.  The refactor itself is mechanical but not trivial — each
helper would need its template parameter list re-stated outside the class
and its `friend`-style access patterns verified.

**Suggested approach (when revisited).**

1. Refactor Block 2's member functions to out-of-class definitions inside
   the same `#if` gate.  Verify build under hypothetical revival
   (`-DTYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward` reinserted in
   CMakeLists.txt + `TYCHO_ENZYME_FOF_ARCHIVE_REVIVED` sentinel).
2. Move both blocks (now both at file scope) plus the archive docstring
   into `include/tycho/detail/vf/derivatives/dense_enzyme_fof_archive.h`.
3. Replace the blocks in `dense_enzyme.h` with `#include
   "dense_enzyme_fof_archive.h"`.

**Estimated effort.**  0.5 day for the out-of-class refactor + extraction.

**Decision criterion.**  Pursue when `dense_enzyme.h` readability becomes
a maintenance pain point on the live path, or as a prerequisite to actual
FoF revival (the extraction makes revival mechanically simpler).

---

## Cross-references

- Initial spec: `docs/superpowers/specs/2026-04-25-claude-enzyme-ad-support-design.md` (gitignored).
- Implementation plan: `docs/superpowers/plans/2026-04-25-enzyme-ad-support.md` (gitignored).
- Phase 1/2 PR descriptions and decision logs: `/tmp/enzyme_phase{1..5}_*.md` (transient; preserved across pushes via the PR descriptions).
- Phase 3 recovery: commit `5ff5caa`.
- Phase 5b recovery: commit `ecc721a`.
- MEE root-cause: commit `ae0ea1d` and `/tmp/enzyme_phase5b_findings.md`.
- Migration trigger evaluation: `/tmp/enzyme_migration_evaluation.md`.

The spec and plan files are gitignored research artifacts. The shipping
contract is encoded in CLAUDE.md, the dense_enzyme.h docstrings, and the
test/bench coverage.
