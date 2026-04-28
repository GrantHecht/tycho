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

**Problem (historical).** Eigen 3.4's `Array<double, 4, 1>::cos()` /
`::sin()` / `::exp()` / `::log()` were unrolled per-element and lowered to
four separate scalar libm calls (`@cos(double)` × 4, etc.) rather than a
single vectorised `cos<4xdouble>` intrinsic. This killed Phase 5b SIMD
performance on bodies whose math involves transcendentals.

**Concrete impact.** On the MEE dynamics test case (9→6 with one `cos(x5)`
and one `sin(x5)` per body invocation):

- Scalar Phase 3 path with `enzyme_width=8`: 5.97 ns total. `cos(x5)` is
  computed *once* per primal pass and amortised across 8 tangents.
- Phase 5b SIMD primal: 142 ns total. Each lane has its own `x5_lane`,
  so per body invocation `cos` is called 4 times. For IR=9 tangents,
  that's ~36 cos + ~36 sin libm calls per Jacobian.

The 18× libm-call ratio maps cleanly onto the ~6× per-lane runtime
regression observed.

**What was tried and didn't work.** Adding `-fveclib=libmvec` to the bench
target made MEE Phase 5b *worse* (142 → 319 ns). Eigen unrolls
`Array::cos()` to scalar libm calls in the C++ template *before* Clang's
vector-loop pass runs, so libmvec's vector-loop-replacement pass has no
vector loop to match against. Reverted.

**Workaround in tree (CLAUDE.md).** Document a per-VF rule:

| Body type | `is_vectorizable` | Path |
|-----------|-------------------|------|
| Arithmetic / `sqrt` only | `true` | Phase 5b SIMD (1.5–1.8× per-lane win) |
| Trig / `exp` / `log` heavy | `false` | Base class scalarises → fast Phase 3 W=8 |

**Proposed fix.** A Tycho-side header defining SIMD overloads for
`Eigen::Array<double, 4, 1>` math functions, calling Sleef
(<https://sleef.org/>) or libmvec's `_ZGVdN4v_*` symbols directly. Would
need:

1. New header `include/tycho/detail/vf/derivatives/enzyme_simd_math.h`
   defining `cos`/`sin`/`tan`/`exp`/`log`/etc. for `Array<double, 4, 1>`
   that emit a single SIMD library call.
2. Specialisations gated on AVX2/AVX-512 detection (CMake check) so the
   default build stays portable.
3. Verification that Enzyme can differentiate through the new SIMD calls
   (probably requires registering them as known math functions in
   Enzyme's `KnownFunctions` table — Enzyme has rules for `_ZGVdN4v_*`
   at the IR level).

**Estimated effort.** 1–2 days. Maintenance burden: a header per math
function family, plus a small CMake feature-detection script. Payoff:
makes `is_vectorizable = true` universally beneficial under EnzymeAD,
removing the trig-body asterisk from the migration trigger evaluation.

**Decision criterion.** Pursue if any astrodynamics workload shipping
through Tycho has Vectorizable + trig in the inner loop. The existing
MEE example is one such workload.

---

## 2. Direct-SIMD Hessian path — RESOLVED for both FoR and FoF (2026-04-28)

**Status.** Phase 6 ships direct-SIMD Forward-over-Reverse Hessian and
Phase 7 ships direct-SIMD Forward-over-Forward (combined J+H) for
`is_vectorizable=true` EnzymeAD VFs at SuperScalar input.  Gated by
`TYCHO_ENZYME_SIMD_HESSIAN` (default ON); flag OFF falls back to the
Phase 5a scalarize-per-lane path under either strategy.  Strategy
selected at cmake-time via `TYCHO_ENZYME_HESSIAN_STRATEGY`
(`ForwardOverReverse` default, `ForwardOverForward` opt-in for
MEE-class workloads — see item 3 below for the FoF perf table).

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
- Per-VF opt-out marker `static constexpr bool
  enzyme_simd_hessian_supported = false;` — for VFs whose body trips
  Enzyme's SS reverse-mode IR type-deduction (composite trig+sqrt+
  division composites; MEE is the canonical case in tycho today).
  Eligible VFs are detected via `enzyme_simd_hessian_eligible<Derived>()`
  in the dispatch.
- Tests `EnzymeVectorized.HessianSIMDMatchesScalarized_{Brach,CR3BP}`
  validate the SIMD path matches the per-lane scalar reference to
  1e-10.

**Phase 7 follow-on shipped.**  FoF SIMD + combined J+H is now in tree
(see item 3 below).  FoR remains the production default — it's faster
on every measured non-MEE body — but FoF is the only SIMD-capable
strategy for MEE-class composites.  Switch via
`TYCHO_ENZYME_HESSIAN_STRATEGY=ForwardOverForward` at cmake time when
MEE-class VFs dominate the workload.

**Open item — MEE-class fallback.**  The per-VF opt-out is shipped, but
the upstream Enzyme limitation that forces it is real: composite bodies
with cos/sin/sqrt/division saturate the type-deduction tape under
SuperScalar reverse mode.  Track upstream Enzyme issues; if a future
release makes SS reverse-mode robust to these patterns, the opt-out can
be removed.

---

## 3. FoF combined-J+H optimisation (spec §4.1 future work)

**Status.** RESOLVED via Phase 7 (2026-04-28).

**What landed.**

- `compute_jacobian_adjoint_hessian_fof_` (scalar) replaces
  `compute_adjoint_hessian_fof_` under FoF strategy.  The outer fwddiff
  reads J(x)·e_i from the previously-discarded fx-shadow on the first
  inner iteration (j == 0); subsequent j iterations recompute the same
  column and are ignored.
- `compute_jacobian_adjoint_hessian_fof_simd_` (SIMD twin) ships under
  Phase 7 alongside `enzyme_fof_inner_wrapper_simd`.  Phase 3 batching
  applies to the OUTER tangent (BW columns of J + BW Hessian column
  entries per call); inner-direction batching deferred (would yield W²
  Hessian elements per call but adds index-arithmetic complexity).
- Dispatch under `TYCHO_ENZYME_HESSIAN_STRATEGY=ForwardOverForward` +
  `TYCHO_ENZYME_SIMD_HESSIAN=ON` routes Vectorizable+SS+EnzymeAD
  inputs to the SIMD FoF path.  No `enzyme_simd_hessian_eligible<>`
  gate — FoF avoids the SS reverse-mode tape that forced MEE-class
  VFs to opt out of FoR SIMD.
- Tests `EnzymeVectorized.HessianFoFSIMDMatchesScalarized_{Brach,
  CR3BP,MEE}` validate the path.  MEE works for the first time under
  any SIMD Hessian strategy.

**Bench result.**  Phase 6 FoR-SIMD remains faster on Brach (1044 ns)
and CR3BP (291 ns) than Phase 7 FoF-SIMD (2109 ns and 1543 ns
respectively).  FoF wins qualitatively on MEE-class bodies (only
viable SIMD Hessian path).  FoR remains the cmake default; FoF is the
opt-in alternative for MEE-dominated workloads.

**Phase 7+ — Inner-direction batching (proof of concept, 2026-04-28).**
Doubly-batched FoF SIMD helper `compute_jacobian_adjoint_hessian_fof_simd_db_`
ships alongside the singly-batched one.  Outer `__enzyme_fwddiff(enzyme_width=BW)`
over `enzyme_fof_inner_wrapper_simd_innerbatch<...,IBW>` (which itself does
`__enzyme_fwddiff(enzyme_width=IBW)`) — confirmed Enzyme accepts nested
enzyme_width composition.  Per outer call: BW outer × BW inner = BW²
Hessian elements + BW columns of J.  Validated by
`EnzymeVectorized.HessianFoFSIMDdb_PolyMatchesScalarized` on a synthetic
IR=8 OR=4 fixture.

Bench (Poly8x4, BW=4): singly-batched 769 ns vs doubly-batched 727 ns —
**only 5% speedup**.  Per-call Enzyme overhead is not the dominant cost;
body work dominates.  Same conclusion as item 4 below ("`enzyme_width` ×
Phase 5b composition is neutral").  The 4× call-count reduction is
absorbed by the 4× body work per call.

The helper is shipped as a research path (not the default) — restricted to
ir divisible by BW (no tail handling).  Promote when a workload arises
where call overhead is the bottleneck.  Index arithmetic for the BW²
shadow unpack is documented in the helper docstring.

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
