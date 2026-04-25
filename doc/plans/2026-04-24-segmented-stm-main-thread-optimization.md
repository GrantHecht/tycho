# Segmented STM main-thread Jacobian elimination — design plan

**Status:** design only. No code change in PR #44 (the PR that opened this branch). Intended for a follow-up PR once the design is reviewed and any related work that ties into it is folded in.

**Branch:** to be created off `main` once PR #44 lands. Working name: `feat/segmented-stm-main-thread-state-only`.

**Goal:** Eliminate redundant Jacobian computation on the main thread of `integrate_stm_parallel(x0, tf, n_parts)` (the segmented single-trajectory overload) without perturbing the chain-rule product, the FP-determinism guarantee, or the latency-hiding pattern that motivates main-thread propagation.

**Tech stack:** C++17, Eigen, nanobind, Google Test, Google Benchmark.

---

## 1. Context / Problem Statement

### Where the code lives

`integrate_stm_parallel(const ODEState<double> &x0, double tf, int n_parts)` — the segmented single-trajectory overload — is declared at `include/tycho/detail/integrators/integrator.h:1896-2031`. It propagates a single trajectory `[t0, tf]` partitioned into `n_parts` segments, dispatches each segment to a worker thread that computes the per-segment STM `(xf_i, jx_i)`, and composes the global STM via the chain-rule product `J_total = J_n × J_{n-1} × ... × J_1`.

### Why the path exists

For a single trajectory of duration `T` and an ODE of dimension `D`, an unsegmented STM costs `O(T · D² · stages)` per step in the Jacobian and `O(D² · M)` accumulated chain-rule. With `n_parts` workers, segmenting cuts each worker's per-segment Jacobian work to `O(T/n_parts · D² · stages)` and the workers run in parallel — ideally an `n_parts`× wall-clock reduction.

### The performance puzzle

The threadpool branch of the implementation runs the main thread in lock-step with the workers: for each non-final segment `i`, the main thread computes `xs[i+1]` synchronously while workers are in-flight, so worker `(i+1)` has its starting state ready by the time it dequeues. The main-thread propagation is implemented by calling `integrate_stm_core` on the segment and discarding everything except the resulting state:

```cpp
// integrator.h:1935-1944
// Use integrate_stm_core (discarding the STM) so the
// main thread exercises the same FP arithmetic as the
// workers. xs[i+1] becomes worker (i+1)'s starting
// state; if it diverged from worker i's xf in FP, the
// chain-rule product in jxall would reflect sensitivity
// at a mismatched linearization point. Identical
// codepaths ⇒ bit-identical xs ⇒ exact chaining.
auto stm_result =
    this->integrate_stm_core(xs[i], ts[i + 1], main_ctrl, main_na, main_nr);
xs[i + 1] = std::get<0>(stm_result);
```

The Jacobian portion of each main-thread call is computed and immediately discarded. For `n_parts` segments, the main thread pays `n_parts - 1` redundant Jacobian computations purely to feed the next worker's starting state.

This document scopes the redesign that eliminates those redundant Jacobians without weakening the bit-equality guarantee.

---

## 2. Code path map (ground truth)

### 2.1 The decomposition discovery

`integrate_stm_core` is **literally** `integrate_dense_core + calculate_jacobian`:

```cpp
// integrator.h:1167-1172
STMRet integrate_stm_core(const ODEState<double> &x0, double tf, ControllerVariant &controller,
                          int &naccept, int &nreject) const {
    auto xs = this->integrate_dense_core(x0, tf, controller, naccept, nreject);
    Jacobian<double> jx = this->calculate_jacobian(xs);
    return std::tuple{xs.back(), jx};
}
```

State propagation is wholly contained in `integrate_dense_core`. The Jacobian is computed in a separate, **post-hoc, read-only** pass over the stored trajectory `xs`. Jacobian computation does not perturb state.

### 2.2 State propagation call chain

`integrate_dense_core` reduces directly to `integrate_impl`:

```cpp
// integrator.h:1054-1067
DenseRet integrate_dense_core(const ODEState<double> &x0, double tf,
                              ControllerVariant &controller, int &naccept, int &nreject) const {
    ODEState<double> xf;
    std::vector<EventPack> events;
    std::vector<std::vector<Eigen::Vector2d>> eventtimes;
    bool storestates = true;
    bool storederivs = false;
    bool storemidpoints = false;
    std::vector<ODEState<double>> xs;
    std::vector<ODEDeriv<double>> d_xs;
    xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                              storemidpoints, xs, d_xs, controller, naccept, nreject);
    return xs;
}
```

`integrate_impl` dispatches on the RK method (`with_public_ivp_alg(rk_method_, ...)`) and forwards to `AdaptiveDriver::integrate`, which in turn drives `Stepper::step<bool ComputeMidpoint>`. The full chain for state propagation is:

```
integrate_dense_core → integrate_impl → AdaptiveDriver::integrate → Stepper::step<false>
```

Whether the caller is `integrate_stm_core` (which wraps `integrate_dense_core` then layers Jacobian work on top) or `integrate_dense_core` directly, **the state-update arithmetic flows through the same compiled code**.

### 2.3 RK state accumulation

In `Stepper::step<>` (`include/tycho/detail/integrators/stepper.h`), the RK update is the standard `xf = x + h · Σ B[i] · k[i]` accumulation:

```cpp
// stepper.h ~lines 142-150
for (int i = 0; i < Stages; i++) {
    xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
        Scalar(RKData::B[i]) * k_vals[i];
}
```

The `bool ComputeMidpoint` template parameter affects only the midpoint/extra-stage branches (`if constexpr (ComputeMidpoint)` block ~lines 171-233). The main `xf` accumulator is not touched by `ComputeMidpoint` branches.

### 2.4 Jacobian assembly (the discardable portion)

`calculate_jacobian` delegates to `STMDriver::calculate_jacobian`:

```cpp
// integrator.h:1276-1281
Jacobian<double> calculate_jacobian(const std::vector<ODEState<double>> &xs) const {
    Jacobian<double> out;
    out = STMDriver::calculate_jacobian(this->stepper_, this->ode_, xs, this->input_rows(),
                                        this->output_rows(), this->enable_vectorization_);
    return out;
}
```

`STMDriver::calculate_jacobian` (in `include/tycho/detail/integrators/stm_driver.h`) iterates over `xs[i] → xs[i+1]` step pairs, computes the per-step Jacobian via the bound stepper's `compute_jacobian`, and accumulates the chain-rule product:

```
for i in 0 .. M-1:
    stepper.compute_jacobian(xs[i] → xs[i+1], stepper_jacobian)
    jxall.topRows(output_rows) = (stepper_jacobian * jxall).eval()
```

For a segment of `M` accepted steps and ODE of input dimension `N` (state + auxiliary), the cost is `O(M · N² · stages)` for the per-step Jacobians plus `O(M · N²)` for the chain-rule multiplications. For typical optimal-control phases (`N ∈ [50, 200]`, `M ∈ [50, 500]`), this is the dominant per-segment cost — and it is exactly what the main thread computes and discards.

### 2.5 Worker dispatch and chain-rule product

The threadpool branch dispatches each segment via `tycho::utils::thread_pool().submit_task` (futures returned). Workers each call `integrate_stm_core` for their assigned segment. The final chain-rule composition runs on the main thread after all futures resolve:

```cpp
// integrator.h ~line 1983
auto [xf, jx] = results[i].get();
jxall.topRows(this->output_rows()) = (jx * jxall).eval();
```

### 2.6 Counter handling (asymmetric across branches by design)

The threadpool branch keeps counters strictly local — main thread uses `int main_na = 0, main_nr = 0;` (`integrator.h:1924`), workers use locals inside `stm_op` (`integrator.h:1914`). Counters are never published to `naccept_`/`nreject_` members. The non-threadpool fallback uses `CounterWriteback _writeback{*this, total_na, total_nr};` (`integrator.h:2013`) and **does** publish.

This asymmetry is documented at `integrator.h:1891-1895` and is intentional: cross-thread member writes would require atomics or aggregation. The batch overloads (`integrate_stm_parallel(x0s, tfs, ...)`) take the aggregation path via `BatchCounterWriteback` (`integrator.h:1873, 1885-1886`). The segmented single-trajectory path could adopt the same pattern but currently doesn't — see Section 7.

### 2.7 Latency-hiding pattern (commit `3b4794f`)

Commit `3b4794f` (Phase 4) explicitly removed redundant per-segment re-propagation from the *non-threadpool* serial branch but left the threadpool branch's main-thread call intact. The commit message notes:

> The threadpool branch keeps its re-propagation: in-flight futures can't be joined before the next submit, so it has a real reason to integrate `xs[i+1]` synchronously on the main thread while worker segments run.

The latency-hiding rationale stands: the main thread overlaps with workers, so eliminating *only the Jacobian portion* (not the entire main-thread call) preserves the overlap.

---

## 3. The optimization opportunity

### 3.1 What changes

Replace the main-thread `integrate_stm_core` call with `integrate_dense_core` and use `xs.back()` for `xs[i+1]`:

```cpp
// proposed replacement at integrator.h:1932-1944
main_ctrl = this->make_worker_controller();
main_na = 0;
main_nr = 0;
// Main-thread state-only propagation. Bit-equality with workers' xf is
// guaranteed by code identity: workers call integrate_stm_core, which
// itself calls integrate_dense_core; the state-update arithmetic flows
// through identical compiled code in either path.
auto xs_main = this->integrate_dense_core(xs[i], ts[i + 1], main_ctrl, main_na, main_nr);
xs[i + 1] = xs_main.back();
```

### 3.2 Why this is bit-safe

State propagation goes through the same `integrate_dense_core → integrate_impl → AdaptiveDriver::integrate → Stepper::step<false>` chain regardless of whether the caller wraps it in `integrate_stm_core`. `xs[i+1]` produced on the main thread and `xf` produced by worker `i` are products of identical compiled code operating on identical inputs (same `xs[i]`, same `ts[i+1]`, same controller via `make_worker_controller()`).

The bit-equality is **structural**, not maintained by FP-arithmetic-mirroring discipline. The current "discarded Jacobian" pattern conveys the intent ("run identical code") but pays for it in actual Jacobian work.

### 3.3 Cost saved

Per non-final segment, the main thread skips:
- `M` per-step Jacobians (`O(M · N² · stages)`)
- `M` chain-rule multiplications (`O(M · N²)`)
- One `Eigen::MatrixXd` of size `N × N` allocated and returned
- The `STMDriver::calculate_jacobian` finite-check at the tail (`detail::check_stm_finite_or_throw`)

Workers' work is unchanged. The main-thread critical path becomes proportional to state-only RK cost, which is a small constant of the full STM cost.

### 3.4 What does NOT change

- The chain-rule product `J_total = J_n × ... × J_1` — still assembled from worker results.
- The latency-hiding pattern — main thread still propagates `xs[i+1]` synchronously while workers run in parallel.
- The exception-aggregation logic at `integrator.h:1947-2006` — purely main-thread control flow.
- The `make_worker_controller()` reset for fresh per-call controller state.
- The `BatchEventCounterWriteback` / `BatchCounterWriteback` semantics on the *batch* overloads (different code path).
- The non-threadpool serial fallback at `integrator.h:2007-2025` — already optimal post-`3b4794f`.

---

## 4. Verification & risk

### 4.1 Bit-equality verification harness (proposed)

Add a probe test that pins the structural bit-equality between state-only and STM-augmented propagation paths:

```cpp
// tests/cpp/integrators/test_stm_state_bit_equality.cpp (new)
TEST(STMStateBitEquality, IntegrateDenseMatchesIntegrateSTMStateBit) {
    // Same ODE, same x0, same tf, same controller seed.
    // Run integrate_dense_core(x0, tf) and integrate_stm_core(x0, tf) twice.
    // Compare xs.back() bit-for-bit (memcmp on raw double bytes,
    // or std::bit_cast<uint64_t> elementwise).
    // Run inside the thread pool too — once on the main thread,
    // once on a worker — and verify all four xf are bit-equal.
}
```

This test should be CI-gated (not just one-shot) so a future inlining or compiler-flag change cannot quietly break the bit-equality contract.

Run the harness under each `TYCHO_FP_MODE` ∈ {STRICT, SAFER_FAST, FAST} on each platform (Linux-clang, macOS-llvm, Windows-clang-cl). Add a CMake test matrix or a single test compiled three times with different flag sets — whichever is mechanically simpler.

### 4.2 Existing regression coverage

`tests/cpp/integrators/test_segmented_stm_serial.cpp` pins serial-branch correctness:

- `SegmentedSTMSerial_EndpointMatchesSingleShot` — endpoint state to `1e-10`, Jacobian block to `1e-6`. After the optimization lands, tighten the Jacobian tolerance — ideally to bit-exact equality, since the per-worker Jacobians and the chain-rule composition are unchanged.
- `SegmentedSTMSerial_StepCountNotDoubled` — bounds segmented step count below `2 × single_total`. Pre-fix: ~2.3×; post-fix: ~1.7×. After this optimization the threadpool branch no longer doubles the main-thread step work, but this test only exercises the serial branch via `ForceSerialThreadPool`, so it is unaffected.

The threadpool path is exercised by `tests/cpp/integrators/test_integration_stm.cpp` (`n_parts=4`) and `tests/cpp/integrators/test_parallel_stm_errors.cpp` (`n_parts ∈ {4, 8}`). Neither asserts bit-exact equivalence on the threadpool path today; both should be inspected for whether tightening from "tolerance-based" to "bit-exact" makes sense post-optimization.

### 4.3 Build-matrix concerns

Default build uses `TYCHO_FP_MODE=SAFER_FAST` (`CMakeLists.txt:74`). On Unix, this expands to `-ffast-math -fno-finite-math-only`. On Windows-clang-cl, it expands to `/fp:fast /clang:-fno-finite-math-only`. FMA contraction is permitted under SAFER_FAST and FAST; STRICT disables it (Windows: `/fp:strict`; Unix: relies on the compiler's strict default).

Within a single binary, FMA contraction is deterministic — the compiler's choice of which `a*b+c` triples to fuse is fixed at build time. So intra-binary bit-equality between two calls to the same compiled function is preserved under all three FP modes. The bit-equality probe (Section 4.1) confirms this empirically across the matrix.

Cross-binary determinism (different builds, different platforms, different compiler versions) is **not in scope** — the optimization preserves the existing intra-binary contract and does not strengthen the cross-binary contract.

### 4.4 Failure modes the optimization does NOT introduce

- Chain-rule product `J_n × ... × J_1` is unchanged (workers still compute their full Jacobians).
- Counter handling is unchanged.
- Exception aggregation is unchanged.
- The latency-hiding pattern (main-thread propagation overlapping with workers) is preserved.
- Cross-thread synchronization is unchanged — the futures-based dispatch is untouched.

### 4.5 Failure modes to watch for in review

- Compiler inlining differences: if the new call site is inlined differently than the old one, the FP-arithmetic *order* could shift. The bit-equality probe catches this.
- Vectorization differences: `enable_vectorization_` is consulted inside `STMDriver::calculate_jacobian` but not on the state-only path. Confirm `AdaptiveDriver::integrate` does not gate on `enable_vectorization_` for state computation in a way that diverges from the STM path's state computation.
- Eigen expression-template evaluation differences: if `AdaptiveDriver::integrate` uses different Eigen idioms when called from `integrate_stm_core` vs `integrate_dense_core`, FP order could differ. Static analysis: the call signature is identical (`integrate_impl` is a template instance), so this should not occur.

---

## 5. Benchmark plan

### 5.1 Current state

No existing benchmark exercises `integrate_stm_parallel(x0, tf, n_parts)` for `n_parts > 1`. The closest is `bench/cpp/integrators/bench_method_comparison.cpp`, which runs single-shot Kepler at three tolerance tiers (1e-6, 1e-9, 1e-12) and does not segment. This means there is no quantitative measurement of the discarded-Jacobian cost today.

### 5.2 New benchmark: `bench/cpp/integrators/bench_segmented_stm.cpp`

**Test problems:**
- Kepler LEO two-body propagation, `D=6`, half-period.
- CRTBP halo-orbit propagation, `D=6`, ~one full period.
- Optionally a higher-dimensional ODE (e.g., a Cartesian dynamics with augmented sensitivity rows) to make the per-step Jacobian cost dominate state cost.

**Configurations:**
- `n_parts ∈ {1, 2, 4, 8, 16}` to characterize main-thread overhead scaling.
- Tolerance tier: `1e-9` (representative).
- Methods: DOPRI54 and DOPRI87 (the most-used adaptive methods for these problems).

**Metrics:**
- Wall-clock total per call.
- Wall-clock per segment on the main thread.
- Worker wall-clock (median across lanes) per segment.
- Step counts per segment.

The benchmark records baseline measurements before the optimization lands; a follow-up record shows the improvement. Use `bench/bench_track.sh baseline` then `bench/bench_track.sh compare` to lock in the comparison.

### 5.3 Acceptance gate for the follow-up PR

The optimization must show a measurable wall-clock reduction (target: ≥10% at `n_parts=8` for the Kepler/CRTBP problems, larger for higher-dimensional problems where Jacobian cost dominates) without regressing any single-shot benchmark in `bench_method_comparison.cpp` beyond bench-tracker noise (~±1%).

---

## 6. Implementation sketch (for the follow-up PR)

### 6.1 Core change

One ~5-line replacement at `integrator.h:1932-1944`:

```cpp
// Before
main_ctrl = this->make_worker_controller();
main_na = 0;
main_nr = 0;
auto stm_result =
    this->integrate_stm_core(xs[i], ts[i + 1], main_ctrl, main_na, main_nr);
xs[i + 1] = std::get<0>(stm_result);

// After
main_ctrl = this->make_worker_controller();
main_na = 0;
main_nr = 0;
auto xs_main = this->integrate_dense_core(xs[i], ts[i + 1], main_ctrl, main_na, main_nr);
xs[i + 1] = xs_main.back();
```

Update the comment to describe the new design: state-only main-thread propagation with bit-equality guaranteed by code identity, not by mirroring discipline.

### 6.2 Test additions

- `tests/cpp/integrators/test_stm_state_bit_equality.cpp` (new) — bit-equality probe (Section 4.1).
- Strengthen `test_segmented_stm_serial.cpp::SegmentedSTMSerial_EndpointMatchesSingleShot` Jacobian tolerance from `1e-6` toward bit-exact.
- Add a threadpool-branch Jacobian regression in `test_integration_stm.cpp` if not already present.

### 6.3 Benchmark additions

- `bench/cpp/integrators/bench_segmented_stm.cpp` per Section 5.

### 6.4 Doc updates

- Update `integrator.h:1891-1895` if the optimization affects the documented behavior of the segmented-STM threadpool branch (it should not — counter handling is unchanged).
- Update `integrator.h:1935-1944` comment to reflect the new design.

---

## 7. Open questions / decision points

1. **Endpoint-only vs full-trajectory return:** `integrate_dense_core` returns the full `xs` trajectory; the main-thread caller only needs `xs.back()`. A leaner `integrate_state_endpoint(x0, tf)` helper would skip per-step trajectory storage entirely. Estimated additional saving: `O(M · N)` allocations and writes per segment. **Decision needed:** scope this in the same follow-up PR, or punt to a third PR after measuring whether the saving is significant?

2. **Counter-handling unification:** The segmented single-trajectory threadpool branch currently keeps counters local; the batch overloads aggregate via `BatchCounterWriteback`. After the main-thread Jacobian is removed, adopting the same aggregation pattern for the segmented overload would publish per-call totals to `naccept_`/`nreject_` and remove the asymmetry documented at `integrator.h:1891-1895`. **Decision needed:** include in the same follow-up PR (one cohesive segmented-STM cleanup) or separate PR?

3. **Bit-equality probe permanence:** Should the bit-equality probe (Section 4.1) be CI-gated forever, or recorded as a one-time verification with the commit's bench output and dropped? Permanent: catches future regressions but adds CI surface area. Once-and-done: relies on the structural argument holding.

4. **Interaction with the future SuperScalar batch path (`ParallelDriver` placeholder):** SP1 introduced a `ParallelDriver` placeholder for the SuperScalar batch path. If that path eventually has a segmented-STM equivalent, the same optimization applies. **Decision needed:** plan ahead so the design doesn't have to be re-derived?

5. **n_parts=1 edge case:** When `n_parts=1`, the threadpool branch is bypassed (`if (n_parts > 1 && tycho::utils::use_thread_pool())` at line 1920); no main-thread propagation runs. Confirmed unaffected.

---

## 8. Related work

*(Reserved for the user / repo maintainer to fold in additional related work that builds on this design. Likely topics: counter-aggregation unification across overloads, ParallelDriver SuperScalar segmentation, additional STM API ergonomics, broader integrator/STM API rationalization.)*

---

## 9. Files referenced

| Path | Role |
|------|------|
| `include/tycho/detail/integrators/integrator.h` | Owns `integrate_stm_parallel`, `integrate_stm_core`, `integrate_dense_core`, `calculate_jacobian`. The single-line replacement at lines 1932-1944 is the optimization. |
| `include/tycho/detail/integrators/stm_driver.h` | Owns `STMDriver::calculate_jacobian` (the per-step Jacobian + chain-rule loop that the main thread currently runs and discards). |
| `include/tycho/detail/integrators/stepper.h` | Owns `Stepper::step<bool ComputeMidpoint>` — the RK kernel; state accumulation lines ~142-150 are the canonical FP-equality anchor. |
| `include/tycho/detail/integrators/adaptive_driver.h` | Owns `AdaptiveDriver::integrate` — the driver that both state-only and STM-augmented paths share. |
| `include/tycho/detail/utils/thread_pool.h` | Owns the `submit_task` primitive used for worker dispatch. |
| `tests/cpp/integrators/test_segmented_stm_serial.cpp` | Pins serial-branch endpoint and step-count behavior. Jacobian tolerance to be tightened post-optimization. |
| `tests/cpp/integrators/test_integration_stm.cpp` | Threadpool-branch coverage at `n_parts=4`. |
| `tests/cpp/integrators/test_parallel_stm_errors.cpp` | Threadpool-branch exception aggregation at `n_parts ∈ {4, 8}`. |
| `bench/cpp/integrators/bench_method_comparison.cpp` | Existing single-shot integrator benchmark; new `bench_segmented_stm.cpp` should sit alongside it. |
| `bench/cpp/CMakeLists.txt` | Wire up the new benchmark target. |
| `CMakeLists.txt` (root, lines ~70-294) | `TYCHO_FP_MODE` definition and per-mode flag handling — relevant to the bit-equality build matrix in Section 4. |

---

## Appendix A — Why "discard the Jacobian" was chosen originally

The original design at `integrator.h:1935-1944` predates the discovery that `integrate_stm_core` is structurally `integrate_dense_core + calculate_jacobian`. The author wrote the comment to convey the intent ("run the same FP arithmetic on the main thread as the workers") and chose `integrate_stm_core` to enforce that intent at the call site. Once the decomposition is recognized, the structural identity makes the workaround unnecessary — `integrate_dense_core` IS what `integrate_stm_core` runs for state, by construction.

This is not a bug or oversight; it is a refactor opportunity that surfaces only after the SP1 driver decomposition (which made the call structure explicit). The current code is correct and conservative; the proposed change is correct and faster.
