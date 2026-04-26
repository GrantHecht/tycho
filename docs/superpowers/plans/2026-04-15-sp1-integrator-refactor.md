# SP1: Integrator Refactor — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `integrator.h` (2207 lines) into 8 focused modules with proper encapsulation, normalize tableau interface, clean up transcription stepper dead code — all without changing behavior.

**Architecture:** Strangler-fig approach. Create new modules alongside the monolith, unit-test them in isolation, then switch `Integrator<DODE>` to delegate to the new modules. The golden regression corpus (generated first) validates bit-identical behavior at each stage.

**Tech Stack:** C++20, CMake/ninja, Google Test, Eigen, nanobind (Python bindings)

**Spec:** `docs/superpowers/specs/2026-04-15-ivp-solver-modernization-design.md`

**Branch:** `ivp-solver-modernization`

**Build notes:**
- Linux: `cmake --preset linux-clang-release && cd build && ninja -j8 all`
- macOS: `cmake --preset macos-llvm-release && cd build && ninja -j6 all`
- Always `conda activate tycho` before building
- NEVER run two builds simultaneously

---

## Task 0: Generate Golden Regression Corpus

**Purpose:** Create the oracle that validates every subsequent refactoring step produces identical results.

**Files:**
- Create: `tests/cpp/integrators/regression/generate_golden.cpp`
- Create: `tests/cpp/integrators/regression/test_regression_ivp.cpp`
- Create: `tests/cpp/integrators/regression/test_regression_transcription.cpp`
- Modify: `tests/cpp/CMakeLists.txt` (add new test targets)

**Reference:** Spec section "Regression Testing Strategy" — 14 test cases across IVP, transcription, and end-to-end OC paths.

- [ ] **Step 0.1: Create `generate_golden.cpp`**

This executable runs the 12 numerical test cases from the spec's regression table and serializes results (final states, trajectories, Jacobians, step counts) to binary files under `tests/cpp/integrators/regression/golden/`.

Use a simple binary format: write array sizes as `int32_t`, then raw `double` data. Each test case writes to a separate file (e.g., `golden/ivp_01_twobody_dopri54.bin`).

Test cases to generate (from spec):
1. Two-body DOPRI54 — final state + trajectory + step count
2. Two-body DOPRI87 — final state + trajectory + step count
3. CR3BP L1 halo DOPRI87 — final state + step count
4. Forced control DOPRI54 — final state (exercises `update_control`)
5. Altitude crossing event DOPRI54 — final state + event times
6. Batch N=8 trajectories DOPRI87 — all final states (SuperScalar path)
7. Backward integration DOPRI54 — final state
8. Two-body `calculate_jacobian` — Jacobian matrix
9. Two-body `calculate_jacobian_hessian` — Jacobian + Hessian
10. CR3BP `calculate_jacobians` batch — all Jacobian matrices
11. Two-body `RKStepper<DOPRI54>::compute_jacobian_impl` isolated — output + Jacobian
12. Two-body `RKStepper<DOPRI87>::compute_jacobian_adjointgradient_adjointhessian_impl` — output + Jacobian + adjgrad + adjhess

For ODE definitions, use the existing test utilities in `tests/cpp/integrators/integrator_test_utils.h`. If a two-body or CR3BP ODE isn't already defined there, define one using `tycho::astro::CartesianDynamics` or `tycho::astro::CRTBPDynamics`.

- [ ] **Step 0.2: Create `test_regression_ivp.cpp`**

Google Test file that loads golden data and compares against live integration results. Use `EXPECT_NEAR` with tolerance 0 for bit-identical cases (absolute tolerance `0.0`), and `1e-14` for FP-noise-tolerant cases. Each test case from the table above (1-7) gets its own `TEST()`.

- [ ] **Step 0.3: Create `test_regression_transcription.cpp`**

Same pattern for test cases 8-12. Tests `calculate_jacobian`, `calculate_jacobians`, `calculate_jacobian_hessian`, and isolated `RKStepper` derivative evaluation.

- [ ] **Step 0.4: Add CMake targets**

In `tests/cpp/CMakeLists.txt`, add:
- `generate_golden` executable (run once to create golden data)
- `tycho_regression_ivp` test executable
- `tycho_regression_transcription` test executable

Both test executables link against `tycho` and Google Test. Mark them in the `heavy_compile` pool since they instantiate integrator templates.

- [ ] **Step 0.5: Build, generate golden data, verify tests pass**

```bash
cd build && ninja -j8 generate_golden tycho_regression_ivp tycho_regression_transcription
./tests/cpp/integrators/regression/generate_golden
ctest -R regression --output-on-failure
```

All regression tests must pass (they compare against data they just generated).

**Note on cases 13-14 (end-to-end OC):** The brachistochrone (case 13) and
Python shooting problem (case 14) are deferred to Task 11's final verification
rather than the automated regression corpus. Rationale: these are full OC
solve loops that exercise the optimizer + transcription pipeline far beyond
the integrator, making them slow and noisy as intermediate gates. The 12
numerical regression cases above are sufficient to catch integrator-level
regressions at each task.

**Note on portability:** Golden data is raw binary (platform-native endianness).
Generated and consumed on the same machine. Not portable across platforms.

- [ ] **Step 0.6: Commit golden corpus**

```bash
git add tests/cpp/integrators/regression/ tests/cpp/CMakeLists.txt
git commit -m "test: add golden regression corpus for integrator refactor"
```

---

## Task 1: Normalize `rk_coeffs.h`

**Purpose:** Standardize the tableau interface so all downstream modules use consistent field names and every specialization exposes `Order`, `ErrorOrder`, `FSAL`, `HasEmbedded`.

**Files:**
- Modify: `include/tycho/detail/integrators/rk_coeffs.h`
- Modify: `include/tycho/detail/integrators/integrator.h` (update field references)
- Modify: `include/tycho/detail/integrators/rk_steppers.h` (update field references)
- Modify: `src/bindings/integrators/rk_coeffs_bind.cpp` (if enum changes)

- [ ] **Step 1.1: Add new fields to all `RKCoeffs` specializations**

Add to each specialization (`RK4Classic`, `DOPRI54`, `DOPRI5`, `DOPRI87`):
```cpp
static constexpr int Order = ...;
static constexpr int ErrorOrder = ...;
static constexpr bool FSAL = ...;
static constexpr bool HasEmbedded = ...;
```

Values:
- `RK4Classic`: `Order=4, ErrorOrder=0, FSAL=false, HasEmbedded=false`
- `DOPRI54`: `Order=5, ErrorOrder=4, FSAL=true, HasEmbedded=true`
- `DOPRI5`: `Order=5, ErrorOrder=0, FSAL=false, HasEmbedded=false`
- `DOPRI87`: `Order=8, ErrorOrder=7, FSAL=false, HasEmbedded=true`

For `Bmid` sizing, add to DOPRI54 and DOPRI87 only:
```cpp
static constexpr int BmidSize = FSAL ? Stages : Stages + 1;
```

And change `MidCoeffs` type to `std::array<double, BmidSize>` (DOPRI54 and
DOPRI87 only). `RK4Classic` and `DOPRI5` do not have midpoint coefficients
— do NOT add `Bmid`/`BmidSize` to them.

Note: for DOPRI5 and RK4Classic, `FSAL`, `Order`, `ErrorOrder`, and
`HasEmbedded` are being **added** (they don't have these fields today).
For DOPRI54 and DOPRI87, `HasEmbedded` is a **rename** from
`EmbeddedCorrector`, and the others are additions.

- [ ] **Step 1.2: Rename fields**

In `rk_coeffs.h`, rename all fields in all specializations:
- `ACoeffs` -> `A`
- `BCoeffs` -> `B`
- `CCoeffs` -> `Bhat`
- `Times` -> `C`
- `MidCoeffs` -> `Bmid`
- `EmbeddedCorrector` -> `HasEmbedded` (already added)
- Remove `is_diag_`

Also remove the `template <class T, int SZ> using STDarray` alias from all
specializations — rewrite ALL coefficient field declarations to use
`std::array<std::array<double, N>, M>` and `std::array<double, N>` directly.
This affects every specialization's `A`, `B`, `Bhat`, `C`, `Bmid` declarations.

- [ ] **Step 1.3: Update all consumers of old field names**

Search and replace across:
- `integrator.h`: references to `ACoeffs`, `BCoeffs`, `CCoeffs`, `Times`, `MidCoeffs`, `EmbeddedCorrector`, `is_diag_`
- `rk_steppers.h`: same references

Use global find-replace within each file. For `is_diag_`, replace:
```cpp
const int js = is_diag_ ? i : 0;
```
with:
```cpp
const int js = 0;
```
since all methods used in practice are non-diagonal (RK4Classic is internal-only and its diagonal optimization is subsumed by compile-time zero-skipping on `A[i][j]`).

- [ ] **Step 1.4: Build and run regression tests**

```bash
cd build && ninja -j8 all && ctest -R regression --output-on-failure
```

All must pass — this is a pure rename.

- [ ] **Step 1.5: Commit**

```bash
git add include/tycho/detail/integrators/rk_coeffs.h \
        include/tycho/detail/integrators/integrator.h \
        include/tycho/detail/integrators/rk_steppers.h
git commit -m "refactor: normalize RKCoeffs tableau interface (standard Butcher notation)"
```

---

## Task 2: Extract `step_controller.h`

**Purpose:** Create the standalone step-size controller module. This has zero dependencies on the integrator — it's pure math.

**Files:**
- Create: `include/tycho/detail/integrators/step_controller.h`
- Create: `tests/cpp/integrators/test_step_controller.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 2.1: Create `step_controller.h`**

```cpp
#pragma once
#include <cmath>

namespace tycho::integrators {

struct IController {
    double safety = 0.9;
    double exponent_bias = 1.0;  // matches current err_pow_fac_ = 1
    double prev_err_norm_ = 0.0;

    double propose_step(double h, double err_norm, int order) const {
        return safety * h * std::pow(1.0 / err_norm, 1.0 / (order + exponent_bias));
    }

    void accept(double err_norm) { prev_err_norm_ = err_norm; }
    void reject(double err_norm) { prev_err_norm_ = err_norm; }
    void reset() { prev_err_norm_ = 0.0; }
};

} // namespace tycho::integrators
```

- [ ] **Step 2.2: Write unit tests**

Test `IController::propose_step` against the current `calc_hnext` formula:
```cpp
// Current formula: step_frac_ * h * pow(accerr / err, 1.0 / (error_order_ + err_pow_fac_))
// New formula:     safety * h * pow(1.0 / err_norm, 1.0 / (order + exponent_bias))
// where err_norm = err / accerr
// These are equivalent: pow(accerr/err, 1/k) = pow(1/(err/accerr), 1/k) = pow(1/err_norm, 1/k)
```

Test cases:
- `err_norm < 1` (step accepted, h should increase)
- `err_norm > 1` (step rejected, h should decrease)
- `err_norm = 1` (boundary case)
- Various orders (5, 8)
- Default parameters match current: `safety=0.9`, `exponent_bias=1.0` (note: current `err_pow_fac_=1`, mapped to `exponent_bias`)

Wait — check the current default. In `integrator.h` line 351: `double err_pow_fac_ = 1;` and line 339: `double error_order_ = 7.0;`. The formula is `pow(accerr/err, 1.0/(error_order_ + err_pow_fac_))`. For DOPRI87, `error_order_=7`, so the exponent is `1/(7+1) = 1/8`. The `IController` formula uses `order` directly (passed by the driver from `Stepper::error_order()`), and `exponent_bias` absorbs the `+1` shift. So `exponent_bias = 1.0` to match current behavior.

- [ ] **Step 2.3: Build and run controller tests**

```bash
cd build && ninja -j8 tycho_tests && ctest -R step_controller --output-on-failure
```

- [ ] **Step 2.4: Commit**

```bash
git add include/tycho/detail/integrators/step_controller.h \
        tests/cpp/integrators/test_step_controller.cpp \
        tests/cpp/CMakeLists.txt
git commit -m "refactor: extract IController into step_controller.h"
```

---

## Task 3: Extract `event_handler.h`

**Purpose:** Extract event detection and refinement logic from the inline adaptive loop into a stateless utility.

**Files:**
- Create: `include/tycho/detail/integrators/event_handler.h`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 3.1: Create `event_handler.h`**

Extract two pieces from `integrator.h`:

1. **Sign-change detection** (currently inline in both adaptive loops, ~lines 694-740 and ~920-970): For each event function, evaluate at `x_next`, check sign change against `prev_vals`, respect direction flag, record crossing windows.

2. **Event refinement** (currently `find_events` method, ~lines 1039-1143): Bisection + Newton refinement using `LGLInterpTable`.

Both become static methods on `EventHandler`. The interface is specified in the design spec section "Event Handler".

Key includes needed: VF types (`GenericFunction`, `LGLInterpTable`), Eigen types.

- [ ] **Step 3.2: Build**

```bash
cd build && ninja -j8 all
```

No dedicated unit test for EventHandler at this stage — it will be integration-tested through the regression corpus when integrated into the driver (Task 6).

- [ ] **Step 3.3: Commit**

```bash
git add include/tycho/detail/integrators/event_handler.h
git commit -m "refactor: extract EventHandler into event_handler.h"
```

---

## Task 4: Extract `stepper.h`

**Purpose:** Extract the IVP RK stage computation from `stepper_compute_impl` into a stateful `Stepper<Alg, DODE, Scalar>` struct.

**Spec deviation:** The spec says the Stepper should receive a pre-composed
ODE (control function baked in via `NestedFunction`). SP1 defers this to
avoid FP-noise risk during the refactor — the Stepper calls `update_control`
per stage, matching current behavior exactly. This will be aligned with the
spec in SP2 when we also have the Julia comparison harness to validate any
FP differences.

**Files:**
- Create: `include/tycho/detail/integrators/stepper.h`
- Create: `tests/cpp/integrators/test_stepper.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 4.1: Create `stepper.h`**

Extract from `integrator.h` lines 416-560 (`stepper_compute_impl` + `stepper_compute` dispatcher).

The `Stepper<Alg, DODE, Scalar>` struct:
- Holds `k_` stage buffer (`std::array<ODEDeriv<Scalar>, Stages>`), `k_fsal_`, `fsal_valid_`
- `step()` method is the body of `stepper_compute_impl`, reading from `RKCoeffs<Alg>` using the new field names (`A`, `B`, `Bhat`, `C`, `Bmid`)
- FSAL handling: if `fsal_valid_`, skip first `ode_.compute()` and use `k_fsal_`. After each step on FSAL methods, copy the last stage derivative to `k_fsal_`
- Midpoint computation: when `compute_midpoint=true`, compute `xf_mid` using `Bmid` coefficients. For non-FSAL methods, perform an extra ODE evaluation at `xf` for the `Stages+1`th midpoint coefficient
- `reset_fsal()`: sets `fsal_valid_ = false`

The `step()` method must also call `update_control` at each stage (matching current behavior). For SP1 we keep the direct mutation approach — the pre-composition change from the spec is deferred to avoid FP-noise risk. Add a `// TODO(SP2): pre-compose control function` comment.

Reference the existing `stepper_compute_impl` body closely — copy the loop structure exactly to ensure bit-identical results.

- [ ] **Step 4.2: Write stepper comparison tests**

Test that `Stepper<DOPRI54, TwoBodyODE, double>::step()` produces identical `xf`, `xf_est`, and `xf_mid` to the current `stepper_compute_impl<IVPAlg::DOPRI54>` for the same inputs. Same for `DOPRI87`.

Use a fixed initial state and step size (no adaptive loop — just one step). Compare all output vectors element-by-element with `EXPECT_DOUBLE_EQ` (bit-identical).

Test cases:
- DOPRI54: one step, check `xf`, `xf_est`, `xf_mid`, FSAL cache populated
- DOPRI54: two consecutive steps, verify FSAL reuse (first stage of step 2 = last stage of step 1)
- DOPRI54: step + `reset_fsal()` + step, verify FSAL NOT reused
- DOPRI87: one step, check `xf`, `xf_est`, midpoint computation with extra derivative
- DOPRI87: verify `fsal_valid_` stays false

- [ ] **Step 4.3: Build and run stepper tests**

```bash
cd build && ninja -j8 tycho_tests && ctest -R test_stepper --output-on-failure
```

- [ ] **Step 4.4: Commit**

```bash
git add include/tycho/detail/integrators/stepper.h \
        tests/cpp/integrators/test_stepper.cpp \
        tests/cpp/CMakeLists.txt
git commit -m "refactor: extract Stepper<Alg,DODE,Scalar> into stepper.h"
```

---

## Task 5: Extract `adaptive_driver.h`

**Purpose:** Extract the scalar adaptive integration loop into `AdaptiveDriver`, unifying the duplicated loops from `integrate_impl` (~lines 560-740) and the scalar fallback path.

**Files:**
- Create: `include/tycho/detail/integrators/adaptive_driver.h`

- [ ] **Step 5.1: Create `adaptive_driver.h`**

`AdaptiveDriver<Alg, Controller, DODE, Scalar>` holds:
- `Stepper<Alg, DODE, Scalar> stepper_`
- `Controller controller_`
- Tolerances (`abs_tols_`, `rel_tols_`), step bounds, `max_step_change_`
- Error buffers (allocated once)

The `integrate()` method is the body of `integrate_impl` (~lines 560-740):
- Initialize `h` from `def_step_size_`
- `stepper_.reset_fsal()`, `controller_.reset()`
- Loop: clamp `h` to not overshoot `tf` -> `stepper_.step()` -> compute error norm -> `controller_.propose_step()` -> clamp -> accept/reject -> `EventHandler::check_crossings()` -> store states/derivs -> continue

Copy the loop structure from `integrate_impl` exactly. The error norm computation (element-wise max of `(xf-xf_est).cwiseAbs() / (abs_tols + xf.cwiseAbs() * rel_tols)`) stays inline in the driver.

The driver takes `const DODE& ode` by reference and does NOT own it.

- [ ] **Step 5.2: Build**

```bash
cd build && ninja -j8 all
```

No integration test yet — the driver is tested when `Integrator` delegates to it (Task 8).

- [ ] **Step 5.3: Commit**

```bash
git add include/tycho/detail/integrators/adaptive_driver.h
git commit -m "refactor: extract AdaptiveDriver into adaptive_driver.h"
```

---

## Task 6: Extract `stm_driver.h`

**Purpose:** Extract Jacobian/Hessian chain computation into `STMDriver`.

**Files:**
- Create: `include/tycho/detail/integrators/stm_driver.h`

- [ ] **Step 6.1: Create `stm_driver.h`**

Move from `integrator.h`:
- `calculate_jacobian()` (~lines 1296-1380)
- `calculate_jacobians()` (~lines 1196-1294)
- `calculate_jacobian_hessian()` (~lines 1369-1580)
- `calculate_jacobians_hessians()` (if it exists as a separate method)

These methods take the transcription `stepper_` VF as a parameter (`const GenericFunction<-1,-1>&`) rather than accessing it as a member.

The SuperScalar batched Jacobian computation (`calculate_jacobians`) uses `DefaultSuperScalar` — this code moves as-is.

- [ ] **Step 6.2: Build**

```bash
cd build && ninja -j8 all
```

- [ ] **Step 6.3: Commit**

```bash
git add include/tycho/detail/integrators/stm_driver.h
git commit -m "refactor: extract STMDriver into stm_driver.h"
```

---

## Task 7: Extract `parallel_driver.h`

**Purpose:** Extract the SuperScalar SIMD batch integration path into `ParallelDriver`.

**Files:**
- Create: `include/tycho/detail/integrators/parallel_driver.h`

- [ ] **Step 7.1: Create `parallel_driver.h`**

Move `integrate_impl_vectorized` (~lines 749-1035) from `integrator.h`. This is the SuperScalar batch loop that packs N trajectories into SIMD lanes.

The `ParallelDriver` orchestrates:
- Packing initial states into `DefaultSuperScalar` vectors
- Running the adaptive loop with `Scalar = DefaultSuperScalar`
- Per-lane done-flag management (tracking which trajectories finished, repacking)
- Unpacking results back to scalar vectors

It internally uses `AdaptiveDriver<Alg, Controller, DODE, DefaultSuperScalar>` or, for SP1, directly contains the SuperScalar loop body from `integrate_impl_vectorized` (since the driver also has per-lane bookkeeping that doesn't exist in the scalar driver). The full unification of scalar and SuperScalar loops into a single `AdaptiveDriver` parameterized on `Scalar` may be deferred to SP2 if the per-lane bookkeeping complicates the extraction.

For SP1, the pragmatic approach: move the vectorized loop body as-is into `ParallelDriver`, with the same Stepper/Controller calls but keeping the SIMD-specific bookkeeping inline.

- [ ] **Step 7.2: Build**

```bash
cd build && ninja -j8 all
```

- [ ] **Step 7.3: Commit**

```bash
git add include/tycho/detail/integrators/parallel_driver.h
git commit -m "refactor: extract ParallelDriver into parallel_driver.h"
```

---

## Task 8: Refactor `integrator.h` to Delegate

**Purpose:** This is the switchover. Modify `Integrator<DODE>` to delegate to the extracted modules instead of using inline code. This is the highest-risk task — regression tests are critical.

**Risk note:** `AdaptiveDriver` (Task 5), `STMDriver` (Task 6), and
`ParallelDriver` (Task 7) were created without unit tests — they are first
exercised here. To manage debugging complexity, each substep below
integrates ONE module at a time and runs regression tests before proceeding
to the next. If a regression fails, the bug is localized to the module just
integrated.

**Note:** `make_table`, `midpoints_removed`, and `LGLInterpTable`
construction remain in `Integrator<DODE>` — they are NOT extracted. The
`EventHandler::resolve_events` receives a pre-built table from `Integrator`.

**Files:**
- Modify: `include/tycho/detail/integrators/integrator.h`
- Modify: `include/tycho/integrators.h` (add new includes)

- [ ] **Step 8.1: Add includes for new modules**

At the top of `integrator.h`, add:
```cpp
#include "tycho/detail/integrators/step_controller.h"
#include "tycho/detail/integrators/stepper.h"
#include "tycho/detail/integrators/adaptive_driver.h"
#include "tycho/detail/integrators/event_handler.h"
#include "tycho/detail/integrators/stm_driver.h"
#include "tycho/detail/integrators/parallel_driver.h"
```

Update `include/tycho/integrators.h` to include all new headers.

- [ ] **Step 8.2: Replace `stepper_compute_impl` with `Stepper` delegation**

Replace `stepper_compute_impl<IVPAlg, Scalar>(...)` calls at lines 547 and 551 with calls to a `Stepper<Alg, DODE, Scalar>` instance. The `stepper_compute` dispatcher (~lines 540-560) switches on `rk_method_` and delegates to the appropriate `Stepper` specialization.

For SP1, keep the `if/else` dispatch on `rk_method_` rather than introducing the variant — the variant switchover is a separate step to minimize risk.

Build and run regression tests after this step:
```bash
cd build && ninja -j8 all && ctest -R regression --output-on-failure
```

- [ ] **Step 8.3: Replace `calc_hnext` with `IController`**

Replace the inline `calc_hnext` call at lines 669 and 944 with `IController::propose_step`. Create an `IController` member on `Integrator<DODE>` initialized with `safety = step_frac_`, `exponent_bias = err_pow_fac_`.

Build and run regression tests.

- [ ] **Step 8.4: Replace inline event detection with `EventHandler`**

Replace the event sign-change detection blocks (~lines 694-740 in scalar loop, ~lines 920-970 in vectorized loop) with calls to `EventHandler::check_crossings`. Replace `find_events` with `EventHandler::resolve_events`.

Build and run regression tests.

- [ ] **Step 8.5: Replace `calculate_jacobian*` with `STMDriver`**

Replace inline `calculate_jacobian`, `calculate_jacobians`, `calculate_jacobian_hessian` methods with delegation to `STMDriver`. Pass `this->stepper_` (the transcription VF) as a parameter.

Build and run regression tests.

- [ ] **Step 8.6: Replace `integrate_impl_vectorized` with `ParallelDriver`**

Replace the inline vectorized loop with delegation to `ParallelDriver`.

Build and run regression tests.

- [ ] **Step 8.7: Remove dead inline code**

After all delegations are in place, delete the old inline code from `integrator.h`:
- `stepper_compute_impl` (entire method)
- `stepper_compute` (dispatcher)
- `calc_hnext`
- The inline event detection blocks
- `calculate_jacobian`, `calculate_jacobians`, `calculate_jacobian_hessian`, `calculate_jacobians_hessians`
- `integrate_impl_vectorized` (body, keep the method as a thin wrapper)
- `find_events`

Keep: `integrate_impl` (now a thin wrapper calling `AdaptiveDriver`), all public `integrate_*` methods (now delegating), `update_control`, `init_stepper_and_controller`, `make_table`, `midpoints_removed`, all VectorFunction `compute_*_impl` methods.

- [ ] **Step 8.8: Full verification**

```bash
cd build && ninja -j8 all && ctest --output-on-failure
```

Run ALL tests (not just regression) and the Python examples:
```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py --timeout 120"
```

- [ ] **Step 8.9: Commit**

```bash
git add include/tycho/detail/integrators/integrator.h \
        include/tycho/integrators.h
git commit -m "refactor: Integrator delegates to extracted Stepper/Controller/Driver/EventHandler/STM modules"
```

---

## Task 9: Clean Up `rk_steppers.h`

**Purpose:** Generalize transcription stepper to handle arbitrary tableaux, remove dead code.

**Files:**
- Modify: `include/tycho/detail/integrators/rk_steppers.h`

- [ ] **Step 9.1: Delete dead code**

Remove `RKStepper_Impl_NEW` (~lines 545-607), `RKStepper_NEW` (~lines 745-782), and `RKStepper2` (~lines 784-931). These are unreferenced outside this file.

Build to verify nothing breaks:
```bash
cd build && ninja -j8 all
```

- [ ] **Step 9.2: Generalize `final_state_sum` in `RKStepper_Impl`**

Replace the per-algorithm `if constexpr` branches in `RKStepper_Impl::final_state_sum` (~lines 503-542) with the generic recursive fallback. The generic code already handles arbitrary tableaux via compile-time zero-skipping on `B[Elem] == 0.0`.

Delete the `RK4Classic`, `DOPRI5`, and `DOPRI87` specialized branches. Keep only:
```cpp
template <int Elem, class Args, class KF>
static auto final_state_sum(const DODE& ode, const Args& args, const KF& kf) {
    if constexpr (Elem == RKCoeffs<RKOp>::Stages - 1) {
        if constexpr (RKCoeffs<RKOp>::B[Elem] == 0.0)
            return args.template head<DODE::XV>();
        else
            return kf * BCoeff<Elem>() + args.template head<DODE::XV>();
    } else {
        if constexpr (RKCoeffs<RKOp>::B[Elem] == 0.0)
            return final_state_sum<Elem + 1, Args, KF>(ode, args, kf);
        else
            return args.template tail<DODE::XV * (RKCoeffs<RKOp>::Stages - 1)>()
                       .template segment<DODE::XV, DODE::XV * Elem>() *
                   BCoeff<Elem>() +
                   final_state_sum<Elem + 1, Args, KF>(ode, args, kf);
    }
}
```

Note: use the renamed `B` field (not `BCoeffs`).

- [ ] **Step 9.3: Verify field references already updated**

Task 1 Step 1.3 already updated all `RKStepper` field references. Verify
with a quick grep that no old names remain:
```bash
rg "ACoeffs|BCoeffs|CCoeffs|Times|MidCoeffs|is_diag_|EmbeddedCorrector" include/tycho/detail/integrators/rk_steppers.h
```
Expected: no matches. If any remain, fix them.

- [ ] **Step 9.4: Verify regression**

```bash
cd build && ninja -j8 all && ctest -R regression --output-on-failure
```

The transcription stepper regression tests (8-12) may show FP-noise differences for DOPRI87 due to the expression-tree shape change. If so, loosen those specific assertions to `1e-14` tolerance and document why.

- [ ] **Step 9.5: Commit**

```bash
git add include/tycho/detail/integrators/rk_steppers.h
git commit -m "refactor: clean up rk_steppers.h — remove dead code, generalize final_state_sum"
```

---

## Task 10: Introduce Variant Dispatch

**Purpose:** Replace the `if/else` on `rk_method_` with `std::variant`-based dispatch in `Integrator<DODE>`.

**Files:**
- Modify: `include/tycho/detail/integrators/integrator.h`

- [ ] **Step 10.1: Add variant types and member**

Add `DriverVariant` and `SIMDDriverVariant` type aliases (as specified in the design spec). Add `driver_` and `simd_driver_` members.

```cpp
using DriverVariant = std::variant<
    AdaptiveDriver<IVPAlg::DOPRI54, IController, DODE, double>,
    AdaptiveDriver<IVPAlg::DOPRI87, IController, DODE, double>
>;
```

- [ ] **Step 10.2: Modify `set_method` to construct variant**

When `set_method()` is called, `emplace` the appropriate variant alternative instead of setting `rk_method_`. Forward current settings (tolerances, step sizes) to the new driver.

- [ ] **Step 10.3: Replace if/else dispatch with `std::visit`**

Change each `integrate_*` method to use `std::visit` on the driver variant instead of switching on `rk_method_`.

- [ ] **Step 10.4: Group settings into `Settings` struct**

Consolidate the scattered settings members (`error_order_`, `min_step_size_`, `def_step_size_`, `max_step_size_`, `max_step_change_`, `adaptive_`, `step_frac_`, `err_pow_fac_`, `abs_tols_`, `rel_tols_`, etc.) into a `Settings` struct. `Integrator` owns the settings and forwards changes to the active driver.

- [ ] **Step 10.5: Full verification**

```bash
cd build && ninja -j8 all && ctest --output-on-failure
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py --timeout 120"
```

- [ ] **Step 10.6: Commit**

```bash
git add include/tycho/detail/integrators/integrator.h
git commit -m "refactor: introduce std::variant dispatch for IVP driver in Integrator"
```

---

## Task 11: Final Verification and Cleanup

**Purpose:** Run the full pre-merge verification sequence, clean up any remaining issues.

**Files:**
- Possibly modify: various files for minor fixes
- Modify: `include/tycho/integrators.h` (ensure all includes are present)

- [ ] **Step 11.0: Verify Python bindings compile and reference no stale names**

```bash
rg "ACoeffs|BCoeffs|CCoeffs|Times|MidCoeffs|EmbeddedCorrector|is_diag_" src/bindings/
```

If any old names are referenced, update them. The bindings in
`src/bindings/integrators/rk_coeffs_bind.cpp` expose `IVPAlg` enum values
(DOPRI54, DOPRI87) — these don't change. Verify `integrator_bind.h` still
compiles (it references `Integrator<DODE>` public API which is unchanged).

```bash
cd build && ninja -j8 _tychopy
```

- [ ] **Step 11.1: Run clang-format**

```bash
cd build && ninja clang-format
```

- [ ] **Step 11.2: Run full C++ test suite**

```bash
cd build && ninja -j8 tycho_tests tycho_tests_light && ctest --output-on-failure
```

All tests must pass.

- [ ] **Step 11.3: Run all 38 Python examples**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

All 38 must exit 0.

- [ ] **Step 11.4: Run C++ brachistochrone example**

```bash
cmake --preset linux-clang-release -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j8 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
```

Must print "Optimal Solution Found", objective ~ 1.8013 s.

- [ ] **Step 11.5: Run benchmarks**

```bash
cmake --preset linux-clang-release -DBUILD_CPP_BENCHMARKS=ON
cd build && ninja -j2 bench_all
bench/bench_track.sh baseline
bench/bench_track.sh record
bench/bench_track.sh compare
```

Document any regressions. The refactor should not change performance; if it does, investigate before proceeding.

- [ ] **Step 11.6: Measure compile-time impact**

Compare full-build time before and after the refactor:
```bash
ninja -t clean && time ninja -j8 all 2>&1 | tail -5
```

Check peak RSS of heavy TUs (binding files, test files) to verify the
variant dispatch hasn't blown up memory usage beyond the `heavy_compile`
pool's capacity.

- [ ] **Step 11.7: Verify line counts**

Check that `integrator.h` has shrunk significantly (target: ~400-500 lines, down from 2207). Check that no extracted module exceeds ~500 lines.

```bash
wc -l include/tycho/detail/integrators/*.h
```

- [ ] **Step 11.8: Final commit if any cleanup was needed**

```bash
git add -A && git commit -m "chore: final cleanup for SP1 integrator refactor"
```
