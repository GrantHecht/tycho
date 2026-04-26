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

## 1. SIMD trig overload for `Eigen::Array<double, W, 1>`

**Problem.** Eigen's `Array<double, 4, 1>::cos()` / `::sin()` / `::exp()` /
`::log()` are unrolled per-element and lower to four separate scalar libm
calls (`@cos(double)` × 4, etc.) rather than a single vectorised
`cos<4xdouble>` intrinsic. This kills Phase 5b SIMD performance on bodies
whose math involves transcendentals.

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

## 2. Direct-SIMD Hessian path

**State today.** `DenseSecondDerivatives<..., EnzymeAD>::compute_jacobian_adjointgradient_adjointhessian_impl`
uses `if constexpr (IsSuperScalar<Scalar>)` to route SuperScalar input
through scalarise-per-lane. Each lane runs the scalar Phase 2 Hessian via
the FoR (Forward-over-Reverse) outer wrapper.

**Proposed extension.** Mirror Phase 5b for the Hessian: a SuperScalar-typed
FoR wrapper that has Enzyme differentiate through SIMD ops directly.

```cpp
template <class Derived, class SSType>
inline void enzyme_for_outer_wrapper_simd(const Derived* self,
                                          const SSType* x_data,
                                          SSType* g_data,
                                          SSType* fx_scratch,
                                          SSType* lam_scratch,
                                          int n_in, int n_out) {
    __enzyme_autodiff<void>(
        reinterpret_cast<void*>(
            &enzyme_compute_wrapper_simd<Derived, SSType>),
        enzyme_const, self,
        enzyme_dup,   x_data,     g_data,
        enzyme_dup,   fx_scratch, lam_scratch,
        enzyme_const, n_in,
        enzyme_const, n_out);
}
```

Then in `compute_adjoint_hessian_for_`, route Vectorizable + SS through
this SIMD wrapper instead of per-lane scalarisation.

**Risks.**

- Same Eigen-trig limitation as Phase 5b — bodies with `cos`/`sin` will
  see the same regression.
- Reverse-mode (`__enzyme_autodiff`) over SIMD types is less exercised in
  Enzyme's test suite than forward-mode. May hit IR-analysis issues
  similar to (but distinct from) the i128-shift hazard from the original
  Phase 2 work.

**Estimated effort.** 0.5–1 day for the symmetric reverse-mode wrapper +
benchmark. Phase 5b's calling convention is the model.

**Decision criterion.** Pursue after item 1 (SIMD trig) is in tree, so the
Hessian path can benefit cleanly. Without item 1, the per-lane libm cost
will dominate on any non-trivial body just as it does on Jacobian.

---

## 3. FoF combined-J+H optimisation (spec §4.1 future work)

**State today.** The Phase 2 FoF helper is retained as research scaffolding
(per spec §4.1 revised; see commit `0fadf4a`). The FoF outer
`__enzyme_fwddiff` *naturally* produces `J(x)·e_i` as a free byproduct in
the `fx`-shadow, alongside the Hessian element in the `dfx_inner`-shadow.

The current FoF caller throws this Jacobian away — it calls
`compute_jacobian_impl` separately up-front. A planned future commit can
read the column of J from the outer fwddiff's shadow instead of running
the Phase 1 forward pass redundantly.

**Why it matters.** Saves IR Phase-1 forward sweeps per Hessian computation:

- Today: `compute_jacobian_impl` (IR fwddiff calls) + IR² FoF outer calls = IR + IR² Enzyme calls per Hessian.
- Optimised: just IR² FoF outer calls; J is extracted from the same passes.
- Savings: ~IR/(IR+1) ≈ small for IR=5–9, but composes with Phase 3
  batching (`enzyme_width=W` over the FoF outer pass) to give a W-column
  J slab and a W² H block per outer call.

**Rough plan.**

1. Modify `compute_adjoint_hessian_fof_` to allocate an `fx`-shadow buffer
   and pass it via `enzyme_dup` instead of letting Enzyme synthesise one.
2. Read `J(x)·e_i` from the `fx`-shadow at each outer iteration.
3. Skip the up-front `compute_jacobian_impl` call.
4. Add a benchmark variant `BM_HessianFoF_combined_*` to measure the
   saving vs the current FoR default.

**Estimated effort.** 0.5 day for the J-extraction + 0.5 day for benches.
Total ~1 day.

**Decision criterion.** Pursue if FoF becomes a real production strategy
again (currently retained as private research scaffolding). The Phase 2
benchmark already showed FoR wins by 4–9× on Hessians, so reviving FoF
needs the combined-output property to flip the verdict — i.e., this
optimisation needs to recover at least 5× to make FoF competitive.

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
On MEE (IR=9), W=8 gives **60× faster Jacobian than autodiff** (5.96 ns
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

## 6. Eigen upstream issue

The `Array<double, W, 1>::cos()` per-element scalar-libm pattern is a
genuine Eigen ecosystem issue. Filing an upstream issue with the
distillation from item 1 above would benefit any project using Eigen's
SuperScalar arithmetic with libm transcendentals.

**Status.** Not filed. Tycho can carry the workaround (item 1) regardless,
but a proper upstream fix would let other projects benefit.

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
