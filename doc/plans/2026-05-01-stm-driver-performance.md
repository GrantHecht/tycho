# `stm_driver.h` performance optimization plan

**Status:** design only. No code change yet — this plan is the spec for the follow-up PR.

**Branch:** to be created off `main` once the parent branch (`feat-direct-cartesian-to-mee`) lands. Working name: `perf/stm-driver-hot-loop`.

**Goal:** Reduce per-step and per-trajectory overhead in the `STMDriver` chain-rule loops without changing numerical results, then quantify each change in a new dedicated benchmark.

**Tech stack:** C++20, Eigen, Google Test, Google Benchmark.

---

## 1. Context

### 1.1 Where the code lives

`STMDriver` is a header-only static-method holder at `include/tycho/detail/integrators/stm_driver.h:52-483`. It owns no state; callers pass in a `vf::GenericFunction<-1, -1>` (the type-erased single-step Jacobian / adjoint-Hessian VF), the dynamics object, and pre-computed dense states.

The four entry points:

| Method | File:Line | Purpose |
| --- | --- | --- |
| `calculate_jacobian` | `stm_driver.h:57-133` | Single trajectory, scalar + SuperScalar mixed |
| `calculate_jacobian_hessian` | `stm_driver.h:137-203` | Single trajectory, scalar only |
| `calculate_jacobians` | `stm_driver.h:376-482` | Batch trajectories, SuperScalar packed |
| `calculate_jacobians_hessians` | `stm_driver.h:210-372` | Batch trajectories + adjoint Hessian, SuperScalar packed |

### 1.2 Call sites

All four methods are called from a single place — the `Integrator` template in `include/tycho/detail/integrators/integrator.h`:

```
integrator.h:1314 → STMDriver::calculate_jacobians
integrator.h:1324 → STMDriver::calculate_jacobian
integrator.h:1338 → STMDriver::calculate_jacobian_hessian
integrator.h:1350 → STMDriver::calculate_jacobians_hessians
```

The `input_rows`, `output_rows`, `t_var()`, and `enable_vectorization_` arguments come straight from the integrator's ODE wrapper — `output_rows()` is the dynamics state dimension, `input_rows() = output_rows() + 1` (the trailing slot is the time variable), and `t_var() == output_rows()` (zero-indexed).

### 1.3 The hot loops

Per-step work, scalar `calculate_jacobian` (`stm_driver.h:86-96`):

```cpp
auto scalar_impl = [&](int i) {
    stepper_input.head(ode.input_rows()) = xs[i];
    stepper_input[ode.input_rows()] = xs[i + 1][ode.t_var()];

    stepper_output.setZero();
    stepper_jacobian.setZero();

    stepper.compute_jacobian(stepper_input, stepper_output, stepper_jacobian);
    jactmp.noalias() = stepper_jacobian * jxall;
    jxall.topRows(output_rows) = jactmp;        // <-- redundant copy
};
```

Per-step work, `calculate_jacobian_hessian` (`stm_driver.h:175-197`):

```cpp
for (int i = 0; i < numsteps; i++) {
    stepper_input.head(ode.input_rows()) = xs[numsteps - i - 1];
    stepper_input[ode.input_rows()] = xs[numsteps - i][ode.t_var()];

    stepper_output.setZero();
    stepper_jacobian.setZero();
    stepper_grad.setZero();
    stepper_hessian.setZero();

    stepper.compute_jacobian_adjointgradient_adjointhessian(
        stepper_input, stepper_output, stepper_jacobian, stepper_grad, stepper_hessian,
        stepper_adjvars);

    jtwist.topRows(output_rows) = stepper_jacobian;
    jxall = jxall * jtwist;                     // <-- aliased mul
    if (i == 0) {                               // <-- branch in inner loop
        jxall.rightCols(1) = stepper_jacobian.rightCols(1);
    }

    hxall = jtwist.transpose() * hxall * jtwist;  // <-- triple product, aliased
    hxall += stepper_hessian;
    stepper_adjvars = stepper_grad.head(output_rows);
}
```

The dominant cost is the `stepper.compute_*` call (Enzyme/AD work over an RK step: typical 10–100k flops per call). The chain-rule matmul around it is `O(input_rows^3)` per step (~500–3500 flops for input_rows ∈ [8, 15]). So matrix-algebra cleanup is **second-tier**: even doing all of it perfectly is bounded by the matmul share of total runtime, which is in the low single-digit percent on representative ODEs. The plan still does the work — small wins compound, and the changes also clean up readability — but acceptance gates are sized accordingly.

### 1.4 Why the SuperScalar Hessian path can't pack consecutive steps

`calculate_jacobian_hessian` (single trajectory, scalar) cannot SuperScalar-batch consecutive steps because each iteration's `stepper_adjvars` depends on the previous iteration's `stepper_grad.head(output_rows)` (`stm_driver.h:196`). This is a true reverse-mode adjoint chain — `λ_n = J_n^T λ_{n+1}` — and is the fundamental reason the batched Hessian path packs across **independent trajectories** instead of consecutive steps. This plan does not attempt to break that data dependency.

---

## 2. Findings (prioritized)

Findings are referenced as `F1`–`F10` in §3 (tasks). Order is rough impact estimate, highest first within each tier.

### Tier 1 — Per-step hot loop (multiplied by `numsteps × ntrajs`)

**F1. Redundant chain-rule copy.** `jactmp.noalias() = stepper_jacobian * jxall;` followed by `jxall.topRows(output_rows) = jactmp;` (`stm_driver.h:94-95, 116-117, 448-449`). The copy is one (output_rows × input_rows) memcpy per step, only present because Eigen aliasing rules forbid writing into `jxall` while it's an operand. Fix: ping-pong two `MatrixXd` buffers and `std::swap` them — zero copies in the hot path.

**F2. Triple product `jtwist.T * hxall * jtwist` written as one expression** (`stm_driver.h:194, 319`). Eigen *probably* breaks this into two products with an internal temp, but expression-template evaluation order isn't part of the public contract; under aliasing it's free to allocate. Fix: explicit named-temp matmuls, `noalias()` on both, ping-pong two `hxall` buffers to eliminate the implicit temp.

**F3. Branch in the inner loop:** `if (i == 0) jxall.rightCols(1) = stepper_jacobian.rightCols(1);` (`stm_driver.h:190-192, 315-317`). The condition is true exactly once per trajectory but is evaluated `numsteps` times. Fix: peel the first iteration outside the loop. Marginal but free.

### Tier 2 — Per-trajectory allocation overhead (multiplied by `ntrajs`)

**F4. `Eigen::MatrixXd jxall_scalar(input_rows, input_rows)` allocated per-lane** (`stm_driver.h:455`) — this is inside the per-pack inner loop in `calculate_jacobians`, so it allocates on every trajectory. Fix: hoist to outer scope, `setZero` on entry to each pack lane.

**F5. `jxs[idx].resize(...)` and `hxs[idx].resize(...)` inside the per-lane loop** (`stm_driver.h:326-327`). Each resize is one heap allocation. Fix: pre-size all of `jxs` and `hxs` in a single pass at function entry. Same applies to `jxs` in `calculate_jacobians`.

**F6. `std::vector xs_tail(begin, end-ncalls)` copy** (`stm_driver.h:344-345, 470-471`). Tail-handling for heterogeneous trajectory lengths copies a slice of the dense-states vector before recursing into the scalar method. Fix: add an iterator-pair (or `std::span`) overload of `calculate_jacobian` / `calculate_jacobian_hessian` and call it directly. Drops both the copy and the `std::vector` heap alloc per tail.

### Tier 3 — Conditional / contract-bearing changes

**F7. Pre-call `setZero()` on stepper output buffers** (lines 90-91, 99-100, 179-182, 304-307, 445-446). The `vf::GenericFunction` concept does not guarantee whether `compute_jacobian` and `compute_jacobian_adjointgradient_adjointhessian` overwrite or accumulate into their outputs. Inspection of `generic_function.h:108-151` shows direct passthrough to the underlying VF's `compute_jacobian` / `compute_jacobian_adjointgradient_adjointhessian`, but composite VFs (sums, scaled, etc.) may legitimately accumulate. **Until the contract is documented and audited**, leave these in. Fix: separate task — audit the `GenericFunction` Jacobian/Hessian write contract end-to-end, document it in `doc/VectorFunction.md`, and only then decide whether the `setZero` calls are removable. If kept, they remain a correctness backstop with negligible cost relative to the AD work that follows.

**F8. SS-to-scalar lane unpack via triple-nested manual loop** (`stm_driver.h:110-115, 329-338`). Each `(l, k)[j]` access is a strided load (stride = vsize doubles within an SoA Eigen pack). Inherent to the layout — but the manual loop can be replaced with an `Eigen::Map<Eigen::Matrix<double, output_rows, input_rows, ColMajor>, 0, Eigen::Stride<vsize, 0>>` that lets Eigen vectorize the gather where possible. Likely a small win, mainly readability.

**F9. `jtwist` sparsity.** The bottom `(input_rows - output_rows)` rows of `jtwist` are all zero except for `jtwist(input_rows-1, input_rows-1) = 1`. Both `jtwist * X` and `jtwist.T * X` could be specialized to `output_rows × input_rows` workblocks plus a copy of the last row/column. Concrete savings on a (8 × 8) matrix: ~12% fewer multiplies for the symmetric case, less for the rectangular Jacobian case. Marginal at typical sizes.

### Tier 4 — Algorithmic / structural

**F10. Bucket scheduling for heterogeneous trajectory lengths.** The current sort (`stm_driver.h:255`) groups packs by length, but a pack still finishes in `lane_min(size)` SS iterations and pays scalar tail-handling for any longer lanes. A bucket-scheduler would group trajectories into length ranges so within a bucket all lanes finish near-simultaneously. Defer until a real workload demonstrates the cost matters — current users are largely homogeneous-length (collocation transcription).

---

## 3. Tasks

Each task is independently testable, independently committable, and matched to one or more findings. Tasks 1 and 10 are the bench bookends; tasks 2-9 are the optimizations. Tasks 2-9 may be reordered freely except where blocking is noted.

### Task 1: Add `bench_stm_driver` baseline benchmark

**Goal:** Add a Google Benchmark file dedicated to `STMDriver` so each subsequent task can record a measurable delta.

**Files:**
- Create: `bench/cpp/integrators/bench_stm_driver.cpp`
- Modify: `bench/cpp/integrators/CMakeLists.txt` (register the new TU into `bench_all`)

**Why a new bench:** `bench/cpp/integrators/bench_integrators.cpp` exercises the full integrator including state propagation; it doesn't isolate the `STMDriver` chain-rule loops from the AD-heavy stepper calls. To attribute wins from F1–F9 cleanly we need a benchmark that runs `STMDriver::calculate_*` over a fixed pre-computed `xs` so the only varying cost is the chain-rule path itself.

**Acceptance criteria:**
- [ ] One TU compiles and links into `bench/cpp/bench_all`.
- [ ] Benchmarks for `calculate_jacobian` (CR3BP, MEE), `calculate_jacobian_hessian` (CR3BP), `calculate_jacobians` (CR3BP × 32 trajectories), and `calculate_jacobians_hessians` (CR3BP × 32 trajectories) — these four match the four entry points.
- [ ] Each benchmark uses a pre-computed `xs` (or `xs_s`) generated once in `static const`-style state, so state propagation cost is excluded.
- [ ] Each benchmark reports per-step time via `state.SetItemsProcessed(state.iterations() * numsteps)`.
- [ ] `bench/bench_track.sh baseline` records on the parent commit.

**Verify:**
```bash
cd build && ninja -j6 bench_stm_driver
./bench/cpp/bench_all --benchmark_filter='STMDriver/'
bench/bench_track.sh baseline
```

**Steps:**

1. Create `bench/cpp/integrators/bench_stm_driver.cpp`:

```cpp
#include "tycho/detail/integrators/integrator.h"
#include "tycho/detail/integrators/stm_driver.h"
#include "tycho/detail/astro/cr3bp.h"
#include "tycho/detail/astro/mee_dynamics.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace tycho;
using namespace tycho::integrators;

namespace {

// Produce a representative xs (dense states) for CR3BP propagated for
// `n_steps` adaptive RK steps. Cached in a function-static so each
// benchmark sample reuses the same trajectory.
const auto &cr3bp_xs() {
    static const auto xs = [] {
        astro::CR3BP ode(/*mu=*/0.01215);
        Integrator<astro::CR3BP> integ(ode, IVPAlg::DOPRI87, 0.01);
        integ.set_abs_tol(1e-12);
        integ.set_rel_tol(1e-12);
        Eigen::Matrix<double, 7, 1> x0;
        x0 << 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0;
        return integ.integrate_dense(x0, 5.0);
    }();
    return xs;
}

void BM_STMDriver_calculate_jacobian_cr3bp(benchmark::State &state) {
    astro::CR3BP ode(0.01215);
    Integrator<astro::CR3BP> integ(ode, IVPAlg::DOPRI87, 0.01);
    auto stepper = integ.stepper();   // accessor introduced in this task if not already exposed
    int input_rows = ode.input_rows() + 1;
    int output_rows = ode.input_rows();
    const auto &xs = cr3bp_xs();
    for (auto _ : state) {
        auto jx = STMDriver::calculate_jacobian(stepper, ode, xs, input_rows, output_rows,
                                                /*enable_vectorization=*/true);
        benchmark::DoNotOptimize(jx);
    }
    state.SetItemsProcessed(state.iterations() * (xs.size() - 1));
}
BENCHMARK(BM_STMDriver_calculate_jacobian_cr3bp);

// ... analogous for MEE, jacobian_hessian, batched variants
}  // namespace
```

2. If `Integrator::stepper()` is not currently a public accessor, add one that returns `const GenericFunction<-1, -1>&`. (Verify before writing — it may already exist as a friend or via internal exposure.) If exposing it broadens the public surface in an unwelcome way, the bench can construct a `GenericFunction` from an in-place stepper VF the same way `Integrator` does internally — pick whichever is less invasive.

3. Append to `bench/cpp/integrators/CMakeLists.txt` so the file participates in `bench_all`. Mirror the registration pattern used by `bench_integrators.cpp`.

4. Build and run:

```bash
cmake --preset linux-clang-release -DBUILD_CPP_BENCHMARKS=ON
cd build && ninja -j6 bench_all
./bench/cpp/bench_all --benchmark_filter='STMDriver/'
```

5. Record baseline:

```bash
bench/bench_track.sh baseline --reps 5
```

6. Commit:

```bash
git add bench/cpp/integrators/bench_stm_driver.cpp bench/cpp/integrators/CMakeLists.txt
git commit -m "bench: add stm_driver chain-rule micro-benchmark"
```

---

### Task 2: F1 — eliminate redundant chain-rule copy via ping-pong

**Goal:** Replace the `jactmp` write-back into `jxall.topRows(output_rows)` pattern with a two-buffer ping-pong, removing one (output_rows × input_rows) memcpy per step.

**Files:**
- Modify: `include/tycho/detail/integrators/stm_driver.h:67-118` (`calculate_jacobian` scalar + vector lambdas) and `stm_driver.h:397-451` (`calculate_jacobians`)

**Acceptance criteria:**
- [ ] Per-step path no longer contains `jxall.topRows(output_rows) = jactmp;`.
- [ ] All STM tests in `tests/cpp/integrators/test_integration_stm.cpp` still pass.
- [ ] Bench (`STMDriver/calculate_jacobian/cr3bp`) shows non-positive change vs. baseline; small positive change (≤2%) is acceptable.

**Verify:**
```bash
cd build && ninja -j6 tycho_tests bench_all
ctest --test-dir tests/cpp/integrators --output-on-failure
./bench/cpp/bench_all --benchmark_filter='STMDriver/calculate_jacobian'
bench/bench_track.sh compare
```

**Steps:**

1. In `calculate_jacobian` replace the local `Eigen::MatrixXd jactmp(output_rows, input_rows);` with a pair of full-size buffers and swap them every step. The bottom rows of `jxall` represent the time-variable pass-through identity row and must be preserved across swaps:

```cpp
Eigen::MatrixXd jxall_a(input_rows, input_rows);
Eigen::MatrixXd jxall_b(input_rows, input_rows);
jxall_a.setIdentity();
jxall_b = jxall_a;   // matching identity bottom row that gets preserved by the convention
Eigen::MatrixXd *jxall_cur = &jxall_a;
Eigen::MatrixXd *jxall_nxt = &jxall_b;

auto scalar_impl = [&](int i) {
    stepper_input.head(ode.input_rows()) = xs[i];
    stepper_input[ode.input_rows()] = xs[i + 1][ode.t_var()];

    stepper_output.setZero();
    stepper_jacobian.setZero();
    stepper.compute_jacobian(stepper_input, stepper_output, stepper_jacobian);

    // Top output_rows rows: chain-rule update. Bottom rows: unchanged
    // (preserved across swaps because both buffers were initialized the
    // same way and the bottom rows are never written).
    jxall_nxt->topRows(output_rows).noalias() = stepper_jacobian * (*jxall_cur);
    std::swap(jxall_cur, jxall_nxt);
};
```

2. Apply the same pattern in `vector_impl` (where the swap happens once per lane inside the j loop).

3. The final result is read from `*jxall_cur` after the loop:

```cpp
jx = jxall_cur->topRows(output_rows);
```

4. Apply the same change in `calculate_jacobians` (`stm_driver.h:397-451`) for the SuperScalar `jxall_ss` buffer — two `Matrix<DefaultSuperScalar, -1, -1>` buffers ping-ponged.

5. Verify and commit:

```bash
cd build && ninja -j6 tycho_tests bench_all
ctest --test-dir tests/cpp/integrators -R STM --output-on-failure
./bench/cpp/bench_all --benchmark_filter='STMDriver/calculate_jacobian'
bench/bench_track.sh record
git add include/tycho/detail/integrators/stm_driver.h
git commit -m "perf(stm_driver): ping-pong chain-rule buffer to drop per-step copy"
```

---

### Task 3: F2 — explicit named temps for `jtwist.T * hxall * jtwist`

**Goal:** Replace the implicit-temp triple product with two explicit `noalias()` matmuls and a ping-pong on `hxall`. Guarantees no internal Eigen allocation in the hot path.

**Files:**
- Modify: `include/tycho/detail/integrators/stm_driver.h:158-197` (`calculate_jacobian_hessian`), `stm_driver.h:289-321` (`calculate_jacobians_hessians`)

**Acceptance criteria:**
- [ ] No expression-template triple product remains; every matmul is on its own line with `.noalias()`.
- [ ] STM Hessian tests pass.
- [ ] Bench (`STMDriver/calculate_jacobian_hessian/cr3bp`) shows non-positive change vs. record from Task 2.

**Verify:**
```bash
cd build && ninja -j6 tycho_tests bench_all
ctest --test-dir tests/cpp/integrators -R "STM|Hessian" --output-on-failure
./bench/cpp/bench_all --benchmark_filter='STMDriver/calculate_jacobian_hessian'
bench/bench_track.sh compare
```

**Steps:**

1. In `calculate_jacobian_hessian` add a single scratch buffer and a second `hxall`:

```cpp
Eigen::MatrixXd hxall_a(input_rows, input_rows);
Eigen::MatrixXd hxall_b(input_rows, input_rows);
Eigen::MatrixXd hscratch(input_rows, input_rows);
hxall_a.setZero();
hxall_b.setZero();
Eigen::MatrixXd *hxall_cur = &hxall_a;
Eigen::MatrixXd *hxall_nxt = &hxall_b;
```

2. Replace `hxall = jtwist.transpose() * hxall * jtwist;` with:

```cpp
hscratch.noalias() = jtwist.transpose() * (*hxall_cur);
hxall_nxt->noalias() = hscratch * jtwist;
*hxall_nxt += stepper_hessian;
std::swap(hxall_cur, hxall_nxt);
```

3. After the loop, `hxall_cur` is the final Hessian — return `*hxall_cur` in the existing tuple.

4. Repeat for `calculate_jacobians_hessians` with `Matrix<DefaultSuperScalar, -1, -1>` buffers.

5. Same for the `hxs[idx] = jtwist.transpose() * hxs[idx] * jtwist;` in tail handling at `stm_driver.h:359-360` — but the tail is a one-shot, so plain named temps suffice (no ping-pong needed):

```cpp
Eigen::MatrixXd hscratch_tail(input_rows, input_rows);
hscratch_tail.noalias() = jtwist.transpose() * hxs[idx];
hxs[idx].noalias() = hscratch_tail * jtwist;
hxs[idx] += htmp;
```

6. Verify and commit.

---

### Task 4: F3 — peel the `i == 0` branch out of the per-step loop

**Goal:** Remove the `if (i == 0) jxall.rightCols(1) = stepper_jacobian.rightCols(1);` test from the inner loop body in both `calculate_jacobian_hessian` (`stm_driver.h:190-192`) and `calculate_jacobians_hessians` (`stm_driver.h:315-317`).

**Files:**
- Modify: `include/tycho/detail/integrators/stm_driver.h`

**Acceptance criteria:**
- [ ] Inner loop body has no `if (i == 0)` test.
- [ ] First-iteration semantics identical (verified by existing STMComposition / SHO STM tests).

**Verify:** same as Task 3.

**Steps:**

1. In `calculate_jacobian_hessian`, run iteration `i = 0` outside the loop:

```cpp
// Iteration 0 (peeled): identical to the loop body but with the
// jxall.rightCols(1) update that only fires once.
{
    stepper_input.head(ode.input_rows()) = xs[numsteps - 1];
    stepper_input[ode.input_rows()] = xs[numsteps][ode.t_var()];
    stepper_output.setZero();
    stepper_jacobian.setZero();
    stepper_grad.setZero();
    stepper_hessian.setZero();
    stepper.compute_jacobian_adjointgradient_adjointhessian(
        stepper_input, stepper_output, stepper_jacobian, stepper_grad, stepper_hessian,
        stepper_adjvars);

    jtwist.topRows(output_rows) = stepper_jacobian;
    jxall_nxt->noalias() = (*jxall_cur) * jtwist;
    jxall_nxt->rightCols(1) = stepper_jacobian.rightCols(1);   // first-iter only
    std::swap(jxall_cur, jxall_nxt);

    hscratch.noalias() = jtwist.transpose() * (*hxall_cur);
    hxall_nxt->noalias() = hscratch * jtwist;
    *hxall_nxt += stepper_hessian;
    std::swap(hxall_cur, hxall_nxt);

    stepper_adjvars = stepper_grad.head(output_rows);
}

for (int i = 1; i < numsteps; i++) {
    // ... same body, no rightCols(1) update, no `if`
}
```

2. Repeat in `calculate_jacobians_hessians`.

3. Verify and commit.

**Note:** Tasks 2, 3, 4 all touch the same loops. They can be merged into one commit if reviewer prefers a single coherent rewrite — but keeping them as separate commits gives one bench delta per change, which is the auditable form.

---

### Task 5: F4, F5 — hoist `jxall_scalar` and pre-size output vectors

**Goal:** Eliminate per-trajectory heap allocations in `calculate_jacobians` and `calculate_jacobians_hessians`.

**Files:**
- Modify: `include/tycho/detail/integrators/stm_driver.h:263-264, 326-327, 412, 455`

**Acceptance criteria:**
- [ ] No `Eigen::MatrixXd <name>(input_rows, ...)` declarations inside per-pack-lane loops.
- [ ] `jxs` and `hxs` resized to the correct dimensions in a single pre-pass; inner loop uses `=` assignment, not `resize` + assign.
- [ ] Bench `STMDriver/calculate_jacobians/cr3bp_x32` shows non-positive change vs. record from Task 4.

**Verify:**
```bash
ctest --test-dir tests/cpp/integrators -R "STM|Jacobian" --output-on-failure
./bench/cpp/bench_all --benchmark_filter='STMDriver/calculate_jacobians'
bench/bench_track.sh compare
```

**Steps:**

1. In `calculate_jacobians_hessians`, pre-size both output vectors at function entry:

```cpp
std::vector<Eigen::MatrixXd> jxs(ntrajs);
std::vector<Eigen::MatrixXd> hxs(ntrajs);
for (int i = 0; i < ntrajs; ++i) {
    jxs[i].resize(output_rows, input_rows);
    hxs[i].resize(input_rows, input_rows);
}
```

2. Drop the per-lane `jxs[idx].resize(...)` / `hxs[idx].resize(...)` at `stm_driver.h:326-327` — now redundant.

3. In `calculate_jacobians`, hoist `jxall_scalar` out of the inner per-lane loop:

```cpp
// Outside the n / v loops:
Eigen::MatrixXd jxall_scalar(input_rows, input_rows);

// Inside the v loop, before the unpack:
jxall_scalar.setZero();   // optional, fully overwritten below
for (int k = 0; k < input_rows; k++) {
    for (int l = 0; l < input_rows; l++) {
        jxall_scalar(l, k) = jxall_ss(l, k)[v];
    }
}
```

4. Verify and commit.

---

### Task 6: F6 — iterator-pair tail overload for `xs`

**Goal:** Avoid the `std::vector<ODEState> xs_tail(begin, end)` copy in tail handling by accepting iterator ranges directly.

**Files:**
- Modify: `include/tycho/detail/integrators/stm_driver.h` — add overload taking `(It begin, It end, ...)` for `calculate_jacobian` and `calculate_jacobian_hessian`; rewrite the existing `vector` overloads to delegate to the iterator overload.
- Modify: `tests/cpp/integrators/test_integration_stm.cpp` — add a test that exercises the iterator overload directly (see Task 6 step 3 below).

**Acceptance criteria:**
- [ ] No `std::vector<typename DODE::template Input<double>> xs_tail(begin, end);` remains in `calculate_jacobians` or `calculate_jacobians_hessians` tail-handling paths.
- [ ] Existing `vector`-based public API unchanged at call sites in `integrator.h`.
- [ ] Equivalence test passes: `calculate_jacobian(stepper, ode, xs.begin(), xs.end(), ...) == calculate_jacobian(stepper, ode, xs, ...)`.

**Verify:**
```bash
cd build && ninja -j6 tycho_tests bench_all
ctest --test-dir tests/cpp/integrators -R STM --output-on-failure
./bench/cpp/bench_all --benchmark_filter='STMDriver/'
```

**Steps:**

1. Add the iterator overload as the primary implementation:

```cpp
template <class DODE, class It>
static Eigen::MatrixXd
calculate_jacobian(const GenericFunction<-1, -1> &stepper, const DODE &ode,
                   It xs_begin, It xs_end, int input_rows, int output_rows,
                   bool enable_vectorization) {
    auto numsteps_total = std::distance(xs_begin, xs_end) - 1;
    if (numsteps_total < 1)
        throw std::invalid_argument("STMDriver::calculate_jacobian: requires at least 2 states ...");
    // ... body uses xs_begin[i] in place of xs[i] (operator[] on random-access iterators)
}
```

2. Make the existing `std::vector` overload a thin forwarder:

```cpp
template <class DODE>
static Eigen::MatrixXd
calculate_jacobian(const GenericFunction<-1, -1> &stepper, const DODE &ode,
                   const std::vector<typename DODE::template Input<double>> &xs,
                   int input_rows, int output_rows, bool enable_vectorization) {
    return calculate_jacobian(stepper, ode, xs.cbegin(), xs.cend(), input_rows, output_rows,
                              enable_vectorization);
}
```

3. Add an equivalence test:

```cpp
// tests/cpp/integrators/test_integration_stm.cpp
TEST_F(IntegratorTest, STMDriverIteratorOverloadEquivalent) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    auto xs = integ.integrate_dense(x0, 1.0);

    auto jx_vec = STMDriver::calculate_jacobian(integ.stepper(), ode, xs,
                                                ode.input_rows() + 1, ode.input_rows(), false);
    auto jx_it  = STMDriver::calculate_jacobian(integ.stepper(), ode, xs.cbegin(), xs.cend(),
                                                ode.input_rows() + 1, ode.input_rows(), false);
    EXPECT_TRUE(jx_vec.isApprox(jx_it, 1e-15));
}
```

4. In `calculate_jacobians_hessians` tail handling, replace:

```cpp
std::vector<typename DODE::template Input<double>> xs_tail(
    xs_s[idx].begin(), xs_s[idx].end() - ncalls);
auto [jtmp, htmp] = STMDriver::calculate_jacobian_hessian(
    stepper, ode, xs_tail, lf_tail, input_rows, output_rows);
```

with:

```cpp
auto [jtmp, htmp] = STMDriver::calculate_jacobian_hessian(
    stepper, ode, xs_s[idx].cbegin(), xs_s[idx].cend() - ncalls, lf_tail,
    input_rows, output_rows);
```

5. Same in `calculate_jacobians` tail at `stm_driver.h:470-475`.

6. Verify and commit.

---

### Task 7: F8 — `Eigen::Map` for the SS-to-scalar lane unpack (optional polish)

**Goal:** Replace the manual triple-nested unpack with `Eigen::Map` so Eigen handles the strided gather.

**Files:**
- Modify: `include/tycho/detail/integrators/stm_driver.h:110-115, 329-338`

**Acceptance criteria:**
- [ ] Manual `for (j) for (k) for (l) stepper_jacobian(l, k) = stepper_jacobian_ss(l, k)[j];` replaced with an `Eigen::Map`-based assignment.
- [ ] STM tests pass to within 1 ulp of pre-task baseline (this is a layout-only change, results must be bitwise-identical).
- [ ] Bench shows non-positive change.

**Verify:** same harness as previous tasks; add a bitwise-equality assertion in a new test that runs the SS path with `enable_vectorization=true` and the scalar path, both on the same `xs`, and asserts `(jx_ss - jx_scalar).cwiseAbs().maxCoeff() < 1e-13`.

**Steps:**

1. Investigate the underlying buffer layout of `Eigen::Matrix<DefaultSuperScalar, -1, -1>` — `DefaultSuperScalar` is `Eigen::Array<double, vsize, 1>` so the matrix's underlying contiguous double buffer is laid out as `(elem_index_in_matrix, lane_index_within_pack)` with stride `sizeof(double) * vsize` between elements and stride `sizeof(double)` between lanes within an element.

2. Define a helper:

```cpp
namespace detail {
template <int OR, int IR>
inline void unpack_lane(const Eigen::Matrix<DefaultSuperScalar, -1, -1> &src, int v,
                        Eigen::MatrixXd &dst) {
    constexpr int vsize = DefaultSuperScalar::SizeAtCompileTime;
    const double *base = reinterpret_cast<const double *>(src.data()) + v;
    Eigen::Map<const Eigen::Matrix<double, -1, -1>, 0, Eigen::Stride<-1, -1>>
        view(base, src.rows(), src.cols(),
             Eigen::Stride<-1, -1>(src.rows() * vsize, vsize));
    dst = view;
}
}  // namespace detail
```

3. Replace the manual loop with `detail::unpack_lane(stepper_jacobian_ss, j, stepper_jacobian);` in `vector_impl` and the equivalent calls in `calculate_jacobians`/`calculate_jacobians_hessians`.

4. Add a unit test that constructs a synthetic SS matrix with known per-lane content and verifies `unpack_lane` matches a reference manual unpack bitwise. Place in `tests/cpp/integrators/test_integration_stm.cpp` or a new `test_stm_driver_internals.cpp` if it doesn't fit the existing file's scope.

5. Verify and commit.

**Risk note:** `reinterpret_cast<const double *>(src.data())` relies on `DefaultSuperScalar` being layout-compatible with `double[vsize]`. Eigen's `Array<double, vsize, 1>` with the default `Aligned` storage option *is* layout-compatible, but verify with a `static_assert(sizeof(DefaultSuperScalar) == sizeof(double) * vsize)` at the top of the helper to fail loudly if a future Eigen update breaks this assumption.

---

### Task 8: F7 — audit `setZero()` contract (research, not optimization)

**Goal:** Document the `vf::GenericFunction::compute_jacobian` and `compute_jacobian_adjointgradient_adjointhessian` write contract end-to-end. Decide whether the pre-call `setZero()` calls in `STMDriver` are necessary.

**Files:**
- Read: `include/tycho/detail/vf/type_erasure/generic_function.h`, all VF `compute_jacobian_impl` / `compute_jacobian_adjointgradient_adjointhessian_impl` overrides under `include/tycho/detail/vf/`.
- Modify: `doc/VectorFunction.md` — add a "Output buffer contract" section.
- Modify: `include/tycho/detail/integrators/stm_driver.h` — only if audit confirms outputs are unconditionally overwritten.

**Acceptance criteria:**
- [ ] `doc/VectorFunction.md` documents whether the Jacobian/Hessian outputs are required to be zeroed by the caller, by the callee, or by neither.
- [ ] If the contract is "callee must overwrite":
  - [ ] All `setZero()` calls on `stepper_output*` / `stepper_jacobian*` / `stepper_grad*` / `stepper_hessian*` immediately before `stepper.compute_*` calls are removed.
  - [ ] Bench shows non-positive change.
- [ ] If the contract is "caller must zero" or undocumented/inconsistent:
  - [ ] No code change in `STMDriver`. Document the reasoning in the doc and in a `// CONTRACT:` comment above each `setZero` block.

**Verify:** All existing tests + the full Pre-Merge Verification Sequence in CLAUDE.md (C++ tests + Python examples + brachistochrone). This is a contract change touching every VF.

**Steps:**

1. Catalogue every concrete VF that implements `compute_jacobian_impl` and `compute_jacobian_adjointgradient_adjointhessian_impl`:

```bash
rg -n "compute_jacobian_impl\b|compute_jacobian_adjointgradient_adjointhessian_impl\b" \
   include/tycho/detail/vf/
```

2. For each match, read the body and classify: does it assign-to (`out = ...`), accumulate-into (`out += ...`), or do partial writes (writes only some rows/cols)? Composite VFs (sums, scaled) typically accumulate; primitive VFs typically overwrite.

3. Decide the contract. Note that even if every leaf VF overwrites, a composite that accumulates would break the contract — so the contract has to bind the entire `GenericFunction` chain, not just the leaves.

4. If the audit lands on "always overwrite": remove `setZero()` and add a `static_assert` or runtime-debug `assert` that confirms output buffers are fully written (e.g. seed with NaN before, check for finite after). If "caller must zero": document in `VectorFunction.md` and leave `STMDriver` alone.

5. Commit.

**Why this is a research task:** The optimization is contingent on the audit outcome. Splitting it into "audit + decide" lets the audit produce documentation value even if the bench delta is zero.

---

### Task 9: F9 — exploit `jtwist` sparsity (deferred, conditional)

**Goal:** Specialize the `jtwist * X` and `jtwist.T * X` matmuls to skip the structurally-zero last-row / last-col multiplications.

**Decision rule:** Run after Tasks 2-7 are merged. If the cumulative bench delta from those tasks is ≥10% on `STMDriver/calculate_jacobian_hessian/cr3bp`, **drop this task** — the matmul is no longer the bottleneck. If <5%, **execute** to chase the remaining few percent. Between 5% and 10%, judgement call.

**Files (if executed):**
- Modify: `include/tycho/detail/integrators/stm_driver.h`

**Acceptance criteria (if executed):**
- [ ] Bench shows ≥3% improvement on the target benchmark vs. record after Task 8.
- [ ] All STM tests pass with bitwise-identical outputs (this is a sparse equivalent of dense matmul, must reproduce exactly).

**Sketch:**

```cpp
// jtwist.transpose() * H, where jtwist's bottom (input_rows - output_rows)
// rows are zero except for jtwist(input_rows-1, input_rows-1) = 1.
// Equivalent (assuming output_rows == input_rows - 1):
//   result.topRows(input_rows - 1).noalias() = stepper_jacobian.transpose() * H.topRows(output_rows);
//   result.row(input_rows - 1) = H.row(output_rows);   // identity row pulls through unchanged
```

Worth checking the `output_rows == input_rows - 1` invariant — it's the only configuration in current Tycho but `STMDriver` is templated on `DODE` so future ODE shapes might break this. Guard with `if constexpr` or an `assert`.

---

### Task 10: Final bench compare and PR

**Goal:** Wrap the perf series, document the cumulative delta, open the PR.

**Files:**
- Update: `bench/results/` (committed bench JSONs from each Task)
- Update: `doc/plans/2026-05-01-stm-driver-performance.md` — append a "Results" section with measured per-task deltas
- Open: PR against `main`

**Acceptance criteria:**
- [ ] All Pre-Merge Verification Sequence steps pass (CLAUDE.md §Testing → "Pre-Merge Verification Sequence").
- [ ] PR description includes the cumulative bench delta vs. baseline (Task 1) for every benchmark in `bench_stm_driver.cpp`.
- [ ] Any benchmark regression (negative delta beyond noise threshold) is justified or rolled back.

**Steps:**

1. Run the full pre-merge sequence:

```bash
# 1. C++ tests
cd build && ninja -j6 tycho_tests tycho_tests_light
ctest --output-on-failure

# 2. Python examples
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"

# 3. C++ brachistochrone
cmake --preset linux-clang-release -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j6 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp

# 4. Bench compare
bench/bench_track.sh compare
```

2. Append a Results section to this plan:

```markdown
## Results (filled in after execution)

| Task | Bench (CR3BP) | ns/step before | ns/step after | Δ |
| ---- | ------------- | -------------- | ------------- | -- |
| T2 (F1)  | calculate_jacobian        | ... | ... | ...% |
| T3 (F2)  | calculate_jacobian_hessian| ... | ... | ...% |
| T4 (F3)  | calculate_jacobian_hessian| ... | ... | ...% |
| T5 (F4-5)| calculate_jacobians       | ... | ... | ...% |
| T6 (F6)  | (tail-handling cases)     | ... | ... | ...% |
| T7 (F8)  | calculate_jacobian (SS)   | ... | ... | ...% |
| T8 (F7)  | (per outcome)             | ... | ... | ...% |
| T9 (F9)  | (if executed)             | ... | ... | ...% |
| **Cumulative** | **all four**        | ... | ... | ...% |
```

3. Open PR with `gh pr create`. Cite this plan document by path. Include the Results table in the PR body.

---

## 4. Verification strategy

### 4.1 Unit tests

The existing `tests/cpp/integrators/test_integration_stm.cpp` covers:

- `SHOSTMVsAnalytical` — analytic SHO STM (rotation matrix). Failing this means the chain-rule product is broken.
- `STMIdentityAtZeroTime` — `xs.size() == 1` boundary (currently throws — verify the throw still fires after iterator overload).
- `STMComposition` — `Φ(0→1) ≈ Φ(0.5→1) · Φ(0→0.5)`. Catches first-iteration handling regressions (relevant to Task 4 branch peel).

If the `STMIdentityAtZeroTime` test currently passes the precondition check by happening to call a different code path, this plan's iterator overload should preserve that. Re-read the test before Task 6 to confirm.

### 4.2 Integration tests

The 38 Python examples (`examples/python_examples/`, run via `scripts/run_examples.py`) exercise full trajectory optimization with PSIOPT consuming `STMDriver` Jacobians and Hessians. These are the project's acceptance gate per CLAUDE.md and must all pass before merge.

### 4.3 Numerical tolerance

All optimizations in this plan are **algebraic identities**: they reorder operations or eliminate intermediate copies without changing the floating-point sequence applied to the data. Expected tolerance vs. baseline:

| Change | Expected delta |
| ------ | -------------- |
| F1 (ping-pong)      | bitwise identical |
| F2 (named temps)    | bitwise identical |
| F3 (branch peel)    | bitwise identical |
| F4-F6 (alloc hoist) | bitwise identical |
| F7 (`setZero` drop) | bitwise identical *if* contract is "always overwrite" |
| F8 (`Eigen::Map`)   | bitwise identical |
| F9 (sparsity)       | bitwise identical |

**No task in this plan is expected to change numerical output.** Any test that fails on numerical-equality grounds indicates a real regression, not tolerance creep.

---

## 5. Benchmark plan

### 5.1 New benchmark file

`bench/cpp/integrators/bench_stm_driver.cpp` (created in Task 1). Cases:

| Case ID | Method | ODE | Setup | numsteps | ntrajs |
| ------- | ------ | --- | ----- | -------- | ------ |
| S1 | `calculate_jacobian`           | CR3BP | adaptive, tol 1e-12 | ~200 | 1 |
| S2 | `calculate_jacobian`           | MEE   | adaptive, tol 1e-12 | ~200 | 1 |
| S3 | `calculate_jacobian_hessian`   | CR3BP | adaptive, tol 1e-12 | ~200 | 1 |
| B1 | `calculate_jacobians`          | CR3BP | adaptive, tol 1e-12 | ~200 | 32 |
| B2 | `calculate_jacobians_hessians` | CR3BP | adaptive, tol 1e-12 | ~200 | 32 |

Heterogeneous-length variants (50% trajectories at half length) for B1, B2 to exercise the tail-handling path that Task 6 (F6) addresses.

### 5.2 Acceptance gate

Tasks 2-7 each set their own threshold (typically "non-positive change" — i.e. no regression — with any improvement counted as a Task win). Cumulative target across the series: **≥5% wall-clock improvement on at least one of S1, S3, B1, B2**, and **no regression >1% on any case**. If the cumulative result misses the target, the plan still succeeds as a code-clarity exercise — the matmul work was always second-tier behind the AD cost (§1.3).

### 5.3 Reproducibility

```bash
# On the parent commit (before Task 2):
bench/bench_track.sh baseline --reps 5

# After each Task:
bench/bench_track.sh record --reps 5

# At the end of the series:
bench/bench_track.sh compare
```

`bench/bench_track.sh` is documented in `bench/MACBENCH.md` / `bench/WINBENCH.md`. CPU governor / thermal-state notes in those docs apply here.

---

## 6. Risks

| Risk | Mitigation |
| ---- | ---------- |
| Eigen aliasing rule violated by ping-pong (F1, F2) | All `noalias()` writes target the *next* buffer, never the current one. Tests catch this immediately if violated. |
| Branch peel changes first-iteration semantics (F3) | The peeled iteration is a verbatim copy of the loop body except for the `i==0` condition being unconditionally true. STMComposition test catches asymmetry. |
| `setZero()` removal breaks composite VFs (F7) | This is exactly why Task 8 is structured as audit-then-decide rather than blind removal. |
| `Eigen::Map` reinterpret_cast violates strict aliasing (F8) | `static_assert(sizeof(DefaultSuperScalar) == sizeof(double) * vsize)` plus an explicit unit test against manual unpack. If the assert fails on a future toolchain, the helper falls back to the manual loop via `if constexpr`. |
| Iterator overload deduction breaks existing call sites (F6) | The existing `vector` overload is preserved as a thin forwarder. Call sites in `integrator.h` are unchanged. |
| Bench noise hides small wins | `--reps 5` and the threshold logic in `bench_track.sh` already account for this. If a single task delta is in the noise, document it as "no measurable change" rather than rolling back the cleanup — the change is still worth keeping for clarity. |

---

## 7. Files referenced

- `include/tycho/detail/integrators/stm_driver.h` (the file under optimization)
- `include/tycho/detail/integrators/integrator.h:1314,1324,1338,1350` (call sites)
- `include/tycho/detail/integrators/stepper.h:142-150` (RK update referenced for context)
- `include/tycho/detail/vf/type_erasure/generic_function.h:108-151` (Jacobian/Hessian dispatch)
- `tests/cpp/integrators/test_integration_stm.cpp` (existing test coverage)
- `tests/cpp/integrators/integrator_test_utils.h` (SHO / Kepler test fixtures)
- `bench/cpp/integrators/CMakeLists.txt` (bench registration)
- `bench/cpp/integrators/bench_integrators.cpp` (existing bench, pattern reference)
- `bench/bench_track.sh` (delta-tracking harness)
- `doc/VectorFunction.md` (target for F7 contract documentation)

---

## 8. Out of scope

- **Discrete-algorithm derivative through adaptive step controller** (the original conversation prompt). Decided not worth fixing; deferred indefinitely.
- **Reducing `compute_jacobian*` call count.** Each step requires its own AD pass; the call count is structurally minimum.
- **Multithreading the chain-rule product.** The reverse adjoint chain in `calculate_jacobian_hessian` is sequential by data dependency (§1.4); the forward Jacobian chain in `calculate_jacobian` is sequential by chain-rule structure. Parallelism opportunities live at a higher level (multiple trajectories, already exploited by `calculate_jacobians[_hessians]`) or at a lower level (inside the stepper VF's own AD work, already vectorized via SuperScalar).
- **Tree-reduction of per-step Jacobians.** Saves no flops on a sequential CPU; only useful if the per-step matmul exposes parallelism that current Eigen doesn't already exploit. Not worth it at typical input_rows ~ 8.
