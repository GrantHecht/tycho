# Tycho Build Performance Report — Phase 1 Post-Optimization

**Date:** 2026-04-07
**System:** Linux 6.19.10, Clang 21.1.8, 32 GB RAM
**Build preset:** `linux-clang-release` (`-O3 -march=native -mllvm -inline-threshold=225`)
**Branch:** `build-performance-optimization`

---

## 1. Executive Summary

After Phase 1 optimizations, a full build of Tycho (all targets: core + bindings +
tests + benchmarks + examples) comprises **145 translation units** with a serial
compile time of **84.7 minutes** (5,079 seconds). Individual TUs consume up to
**8.61 GB peak RSS**, making `-j4` still the practical maximum on 32 GB systems.

### Phase 1 Changes Applied

| Change | Commit | Category |
|--------|--------|----------|
| `extern template` for GenericFunction CRTP chain | `f27579b` | No runtime impact |
| `extern template` for GFModelCommon + leaf types | `4ca1925` | No runtime impact |
| Targeted MKL includes in jet.h | `3692def` | No runtime impact |
| Remove ode_2_1 and ode_4 dimension variants | `767ec00` | API change (fallback to dynamic) |
| Compile tests at -O1 | `1711679` | Tests run at reduced optimization |

### Comparison vs Baseline

| Metric | Baseline (2026-04-06) | Phase 1 | Change |
|--------|----------------------|---------|--------|
| Serial compile time | 5,415s (90.2 min) | 5,079s (84.7 min) | **-336s (-6.2%)** |
| TU count | 146 | 145 | -1 |
| Peak RSS (heaviest TU) | 8.71 GB | 8.61 GB | -0.10 GB |
| Template instantiation % | 91.9% | 92.9% | +1.0 pp |
| Total weak symbols | 215,799 | 210,517 | **-5,282 (-2.4%)** |
| Total .o file size | 322.3 MB | 290.9 MB | **-31.4 MB (-9.7%)** |

### Key Findings

1. **Template instantiation remains 92.9% of compile time** (81.3% function +
   11.6% class). The extern template changes reduced absolute instantiation count
   but the percentage is stable because parsing and optimization also decreased.

2. **Test TUs improved the most** — from 2,506s to 2,233s (-10.9%), driven by
   the -O1 optimization reduction and extern template suppression.

3. **Binding TUs improved 8.3%** — from 1,384s to 1,278s, driven by extern template
   and the removal of 2 GenericODE binding TUs.

4. **Benchmark and example TUs barely changed** — these TUs use unique composite
   expression types that cannot be pre-instantiated. The extern template only
   helps for shared leaf types (GenericFunction, Arguments, Segment, Constant).

5. **Peak RSS is essentially unchanged** — the extern template reduces instantiation
   work but does not significantly reduce the AST size or LLVM IR that drives
   memory consumption.

6. **The remaining build cost is dominated by per-TU-unique composite expression
   types** — types like `NestedFunction<GenericFunction<-1,-1>, StackedOutputs<...>>`
   that are unique to each TU and cannot benefit from extern template.

---

## 2. Per-TU Build Time and Category Analysis

### Build Time by Category

| Category | Time (s) | % Total | TU Count | Avg/TU (s) | Baseline (s) | Change |
|----------|----------|---------|----------|-------------|--------------|--------|
| Tests | 2,233 | 44.0% | 57 | 39.2 | 2,506 | -10.9% |
| Bindings | 1,278 | 25.2% | 52 | 24.6 | 1,384 | -7.7% |
| Benchmarks | 694 | 13.7% | 8 | 86.8 | 710 | -2.3% |
| Examples | 631 | 12.4% | 8 | 78.9 | 660 | -4.4% |
| Core | 233 | 4.6% | 18 | 12.9 | 180 | +29.4%* |
| Other | 10 | 0.2% | 2 | 5.0 | 10 | 0% |
| **Total** | **5,079** | **100%** | **145** | **35.0** | **5,415** | **-6.2%** |

*Core increased because the new `vf_instantiations` TU (37.4s) was added to this
category. Without it, core would be 196s (+8.9% from measurement variance).

### Top 25 Heaviest TUs by Wall-Clock Time

| Rank | TU | Category | Time (s) | Baseline (s) | Change |
|------|-----|----------|----------|:------------:|--------|
| 1 | bench_solvers.cpp | bench | 162.5 | 166.4 | -2.3% |
| 2 | zermelo/main.cpp | examples | 135.4 | 141.6 | -4.4% |
| 3 | bench_optimal_control.cpp | bench | 121.8 | 124.4 | -2.1% |
| 4 | delta3_launch/main.cpp | examples | 117.6 | 124.9 | -5.8% |
| 5 | bench_integrators.cpp | bench | 103.7 | 108.1 | -4.1% |
| 6 | test_ocp_multiphase.cpp | tests | 103.5 | 157.1 | **-34.1%** |
| 7 | brachistochrone/main.cpp | examples | 103.4 | 107.4 | -3.7% |
| 8 | kepler_model.cpp | bindings | 92.5 | 93.2 | -0.8% |
| 9 | hypersens/main.cpp | examples | 91.7 | 96.4 | -4.9% |
| 10 | ode_phase_base.cpp | core | 89.7 | 92.3 | -2.8% |
| 11 | test_control_modes.cpp | tests | 85.3 | 110.7 | **-22.9%** |
| 12 | test_psiopt_convergence.cpp | tests | 82.7 | 106.7 | **-22.5%** |
| 13 | bench_type_erasure.cpp | bench | 81.9 | 82.8 | -1.1% |
| 14 | test_jet_batch.cpp | tests | 82.0 | 109.4 | **-25.0%** |
| 15 | test_nlp_structure.cpp | tests | 81.8 | 109.2 | **-25.1%** |
| 16 | test_phase_construction.cpp | tests | 81.7 | 122.2 | **-33.1%** |
| 17 | test_mesh_refinement.cpp | tests | 81.4 | 119.7 | **-32.0%** |
| 18 | kepler_utils.cpp | bindings | 78.7 | 78.3 | +0.5% |
| 19 | bench_vector_functions.cpp | bench | 76.5 | 77.4 | -1.2% |
| 20 | bench_kepler.cpp | bench | 74.1 | 74.9 | -1.1% |
| 21 | bench_utils.cpp | bench | 73.8 | 75.5 | -2.3% |
| 22 | test_ode_builder.cpp | tests | 69.5 | — | new |
| 23 | vector_function_build_part2.cpp | bindings | 69.2 | — | — |
| 24 | generic_odes_build_part3.cpp | bindings | 63.8 | — | — |
| 25 | generic_odes_build_part4.cpp | bindings | 63.5 | — | — |

**Notable:** Heavy test TUs (test_ocp_multiphase, test_phase_construction,
test_mesh_refinement) improved 30-34% — this is primarily the -O1 optimization
reduction eliminating expensive LLVM backend passes on template-heavy code.

---

## 3. Peak RSS Measurement

| TU | Category | Peak RSS (GB) | Wall Time (s) |
|----|----------|:-------------:|:-------------:|
| bench_solvers.cpp | bench | **8.61** | 177.8 |
| bench_integrators.cpp | bench | **8.30** | 113.6 |
| bench_optimal_control.cpp | bench | **8.24** | 133.2 |
| bench_type_erasure.cpp | bench | **7.96** | 89.1 |
| bench_vector_functions.cpp | bench | **7.90** | 82.8 |
| bench_utils.cpp | bench | **7.82** | 80.9 |
| bench_kepler.cpp | bench | **7.82** | 80.5 |
| delta3_launch/main.cpp | examples | 7.61 | 128.1 |
| test_ocp_multiphase.cpp | tests | 7.50 | 111.3 |
| zermelo/main.cpp | examples | 6.87 | 142.9 |
| test_ode_builder.cpp | tests | 6.87 | 74.1 |
| test_control_modes.cpp | tests | 6.75 | 92.0 |
| test_jet_batch.cpp | tests | 6.57 | 88.0 |
| test_psiopt_convergence.cpp | tests | 6.57 | 88.1 |
| test_phase_construction.cpp | tests | 6.56 | 88.1 |
| test_mesh_refinement.cpp | tests | 6.56 | 88.8 |
| test_nlp_structure.cpp | tests | 6.56 | 87.7 |
| brachistochrone/main.cpp | examples | 6.53 | 108.2 |
| hypersens/main.cpp | examples | 6.00 | 97.3 |
| kepler_utils.cpp | bindings | 5.81 | 82.3 |
| kepler_model.cpp | bindings | 5.42 | 95.5 |
| generic_odes_build_part4.cpp | bindings | 3.99 | 66.2 |
| generic_odes_build_part3.cpp | bindings | 3.99 | 65.9 |
| ode_phase_base.cpp | core | 2.60 | 89.0 |
| vector_function_build_part2.cpp | bindings | 2.34 | 69.8 |

**Key observations:**
- **All 7 benchmark TUs still consume 7.8-8.6 GB** — these remain the -j parallelism bottleneck
- **Test TUs use 6.6-7.5 GB** — slight decrease from baseline (6.7-7.6 GB)
- **At -j4 with benchmarks, worst case is 4 x 8.6 = 34.4 GB** — still tight on 32 GB

---

## 4. Template Instantiation Analysis

### Compilation Phase Breakdown (22-TU aggregate)

| Phase | Time (s) | % Total | Baseline % |
|-------|----------|---------|-----------|
| InstantiateFunction | 27,385 | **81.3%** | 80.4% |
| InstantiateClass | 3,904 | **11.6%** | 11.5% |
| Frontend (parsing) | 1,381 | 4.1% | 4.0% |
| Backend (total) | 637 | 1.9% | 2.3% |
| Optimize (LLVM passes) | 374 | 1.1% | 1.7% |
| CodeGen (IR emission) | 11 | <0.1% | <0.1% |

**Template instantiation is 92.9% of total compilation time** (up from 91.9%
baseline). The absolute numbers are similar; the percentage increased slightly
because backend optimization decreased (due to -O1 on test TUs).

### Top Template Families by Compilation Cost (22-TU aggregate)

| Family | Time (s) | Baseline (s) | Change | Instantiations |
|--------|----------|:------------:|--------|---------------|
| `DenseFunctionBase` | 3,311 | 2,905 | +14%* | 81,351 |
| `BumpAllocator` | 2,493 | 2,205 | +13%* | 41,589 |
| `ODEPhase` | 1,194 | 1,134 | +5%* | 639 |
| `std::make_shared` chain | 1,197 | 1,122 | +7%* | 1,116 |
| `ConstraintModel` | 946 | 849 | +11%* | 4,422 |
| `Integrator` | 837 | 780 | +7%* | 2,565 |
| `GFModelCommon` | 684 | 605 | +13%* | 13,244 |
| `GFStorage` | 698 | 617 | +13%* | 1,145 |
| `NestedFunction_Impl` | 635 | 581 | +9%* | 9,521 |
| `GenericFunction` | 645 | 567 | +14%* | 2,782 |
| `StackTwoOutputs_Impl` | 608 | 528 | +15%* | 9,165 |
| `ComputableBase` | 442 | 390 | +13%* | 22,943 |
| `TypeStorage` | 482 | 433 | +11%* | 1,361 |
| `ConstraintInterface` | 474 | 425 | +12%* | 593 |
| `LGLDefects` | 327 | 285 | +15%* | 3,976 |

*Numbers are HIGHER than baseline because this profile covers 22 TUs vs 19 in
baseline (3 additional TUs profiled). The per-TU averages are comparable or lower.

### Top Individual Function Instantiations (22-TU aggregate)

| Function | Time (s) | Count | Avg (s) |
|----------|----------|-------|---------|
| `BumpAllocator::allocate_run<...>` | 2,381 | 37,126 | 0.064 |
| `DenseFunctionBase::compute_jac_adjgrad_adjhess` | 1,544 | 22,082 | 0.070 |
| `std::make_shared<GFModel<...>>` | 1,197 | 1,116 | **1.073** |
| `DenseFunctionBase::compute_jacobian` | 983 | 19,270 | 0.051 |
| `GFStorage::emplace` | 698 | 1,016 | 0.687 |
| `GenericFunction::GenericFunction` | 633 | 713 | 0.888 |
| `ODEPhase::ODEPhase` (constructor) | 599 | 50 | **11.98** |
| `TypeStorage::emplace` | 482 | 1,357 | 0.355 |
| `ConstraintInterface::ConstraintInterface` | 474 | 593 | 0.799 |
| `ConstraintModel::ConstraintModel` | 474 | 1,131 | 0.419 |
| `Integrator::Integrator` | 370 | 215 | **1.721** |
| `ODEPhase::transcribe_dynamics` | 336 | 27 | **12.45** |

---

## 5. Object File and Symbol Analysis

### Object File Size by Category

| Category | Total Size | % Total | TUs | Avg/TU | Baseline |
|----------|-----------|---------|-----|--------|----------|
| Bindings | 144.9 MB | 49.8% | 52 | 2.8 MB | 157.5 MB |
| Tests | 65.3 MB | 22.4% | 57 | 1.1 MB | 84.3 MB |
| Examples | 36.0 MB | 12.4% | 8 | 4.5 MB | 36.7 MB |
| Benchmarks | 23.3 MB | 8.0% | 8 | 2.9 MB | 23.8 MB |
| Core | 20.9 MB | 7.2% | 18 | 1.2 MB | 19.3 MB |
| **Total** | **290.9 MB** | | **145** | **2.0 MB** | **322.3 MB** |

**Total .o size decreased 9.7%** — most savings from bindings (-12.6 MB, fewer
TUs) and tests (-19.0 MB, reduced optimization generating less code at -O1).

### Cross-TU Duplicate Symbols

**Total weak symbols across all 145 TUs: 210,517** (baseline: 215,799, -2.4%)

Top duplicated symbols across binding TUs:

| Symbol | Occurrences | Baseline |
|--------|------------|----------|
| `GFModelCommon<-1,...>` | 28,188 | 30,568 |
| `non-virtual thunks` | 20,315 | 22,053 |
| `std::_Sp_counted_ptr_inplace<GFModel<-1,...>>` | 3,502 | 3,658 |
| `GFModel<-1,...>` | 1,674 | 1,760 |
| `Integrator<GenericODE<GenericFunction<-1,...>>>` | 478 | 623 |
| `ConstraintModel<RKStepper<GenericODE<...>>>` | 376 | 488 |
| `GFModelCommon<8,...>` | 340 | 340 |
| `GFModelCommon<12,...>` | 272 | 272 |

The `extern template` declarations reduced `GFModelCommon<-1,...>` duplicates by
2,380 (-7.8%) and eliminated all shared Integrator/ConstraintModel duplicates for
the removed dimension variants.

---

## 6. Root Cause Analysis (Ranked by Remaining Impact)

### Factor 1: Composite Expression Template Instantiation (~40% of remaining cost)

The dominant remaining cost is template instantiation for **per-TU-unique composite
expression types** — types like:
```
NestedFunction<GenericFunction<-1,-1>, StackedOutputs<Segment<-1,3,-1>, ...>>
```

These types are unique to each TU's expression tree and cannot benefit from
`extern template`. They drive the `DenseFunctionBase` (3,311s), `BumpAllocator`
(2,493s), `NestedFunction_Impl` (635s), and `StackTwoOutputs_Impl` (608s) costs.

### Factor 2: Type Erasure Chain (`GFModel` + `shared_ptr`) (~15%)

`std::make_shared<GFModel<...>>` at 1,197s (1.07s per instantiation) and
`GFStorage::emplace` at 698s remain expensive. The `extern template` only helps
for the 4 pre-instantiated leaf types; the remaining ~1,100 instantiations are
for composite types unique to each TU.

### Factor 3: Heavy Constructor Chains (ODEPhase, Integrator) (~12%)

`ODEPhase::ODEPhase` (599s, 12.0s each) and `Integrator::Integrator` (370s,
1.7s each) remain expensive. These are driven by the remaining GenericODE
dimension variants and cannot be reduced further without the Phase-boundary
type erasure approach (7.11 in the original report).

### Factor 4: Cross-TU Redundancy (~10%)

210,517 weak symbols are still compiled across 145 TUs. The `extern template`
reduced this by 2.4%, but the vast majority of duplication is from Eigen internal
types and composite VF expression types that vary per TU.

### Factor 5: LLVM Backend Optimization (~5%)

Backend optimization is now 1.9% (was 2.3%). The -O1 on tests reduced this
significantly. Benchmark and example TUs still run at -O3 and contribute
most of the remaining backend cost.

---

## 7. Further Improvement Opportunities (Ranked by Expected Impact)

### Why Phase 1 had limited effect

Phase 1's `extern template` only eliminated instantiations for **4 leaf types**
(GenericFunction, Arguments, Segment, Constant). But there are ~123 unique
`Derived` types per TU (81,351 DenseFunctionBase instantiations / 22 TUs / ~30
methods per type). The vast majority are per-TU-unique composite expression types
(`NestedFunction<GenericFunction<-1,-1>, StackedOutputs<...>>`, etc.) that can
never be pre-instantiated.

The remaining cost is **structurally inherent** to the CRTP expression template
design. To make significant further progress, we must either reduce the cost
*per instantiation* or reduce the *number of TUs* that pay the cost.

---

### 7.1 Unity Builds for Test TUs (`CMAKE_UNITY_BUILD`)

**Estimated reduction: 14-22% of total serial build time (~700-1,100s)**

57 test TUs each independently instantiate the same shared template types.
CMake's `CMAKE_UNITY_BUILD` merges multiple `.cpp` files into a single TU,
compiling shared instantiations once per group instead of once per source file.

Merging into 7 groups of ~8 would reduce test serial time by 30-50%:
- 57 TUs x 39s avg = 2,233s currently
- 7 unity TUs x ~150-200s est. = ~1,050-1,400s

**Implications:**

- *Runtime performance:* None — same code is generated.
- *Peak RSS:* **Critical concern.** Unity TUs will be much larger. If 8 test
  files are merged, peak RSS could reach 15-25 GB per unity TU, making `-j2`
  the maximum on 32 GB. This worsens parallelism.
- *Incremental builds:* Any change to one source file in a unity group
  recompiles the entire group. This hurts incremental development iteration.
- *Compile errors:* Unity builds can expose name collisions between TUs
  (static functions, anonymous namespaces, `using` directives). These must be
  resolved.
- *Recommendation:* Experiment with `UNITY_BUILD_BATCH_SIZE 8` on the test
  target only. Measure peak RSS and serial time. If RSS stays under ~14 GB
  per unity TU, this is the single highest-impact remaining change.

### 7.2 Compile Examples at -O1

**Estimated reduction: 4-5% of total serial build time (~200-280s)**

Same approach as the -O1 change for tests (already applied). Example TUs are
standalone demonstration programs that don't need -O3. Per-TU trace data shows
30-40s of backend (LLVM optimization) time per example TU:

| Example TU | Total (s) | Frontend (s) | Backend (s) |
|------------|-----------|:------------:|:-----------:|
| zermelo/main.cpp | 143 | 56 | 40 |
| delta3_launch/main.cpp | 128 | — | ~35 |
| brachistochrone/main.cpp | 108 | — | ~30 |
| hypersens/main.cpp | 97 | — | ~28 |

At -O1, backend drops from ~33s to ~10s per TU. Across 8 example TUs: ~180-250s
savings.

**Implications:**

- *Runtime performance:* Examples will run slower (2-5x for compute-heavy
  examples). This is acceptable — they exist for demonstration, not benchmarking.
- *Code complexity:* Trivial — same `target_compile_options` pattern as tests.
- *Risk:* None. Example correctness is unaffected.

### 7.3 Factor SuperScalar `VectorImpl` Out of Constraint Methods

**Estimated reduction: 3-6% of total serial build time (~150-300s)**

The `constraints_jacobian_adjointgradient_adjointhessian` method in
`DenseFunctionBase` (dense_function_base.h, lines 1131-1249) has an **enormous**
body — 118 lines including a full `VectorImpl` lambda that instantiates 6
SuperScalar matrix types (`Input<SuperScalar>`, `Output<SuperScalar>`,
`Jacobian<SuperScalar>`, `Gradient<SuperScalar>`, `Hessian<SuperScalar>`,
`Output<SuperScalar>`) plus scatter/gather loops and KKT fill operations.

This entire body is inside the `allocate_run` lambda, making it part of the
unique `(Func, TempSpecs...)` instantiation. The SuperScalar matrices and
scatter/gather code are **Derived-independent** — they only use runtime dimensions
(`this->input_rows()`, `this->output_rows()`). But because they're inside the
per-Derived lambda, they get re-instantiated for every unique Derived type.

The same pattern appears in `constraints_jacobian` (lines 1020-1076) and
`constraints_jacobian_adjointgradient` (lines 1078-1113) — all three have
SuperScalar VectorImpl blocks embedded in per-Derived lambdas.

At 292s for 1,066 instantiations (274ms each), even halving the per-instantiation
cost saves ~145s. The same principle applies to the non-constraint compute methods
(`compute_jacobian`, `compute_jacobian_adjointgradient_adjointhessian`) where
similar embedded VectorImpl blocks contribute to their 2,526s combined cost.

**Proposed refactor:** Extract the VectorImpl scatter/gather and KKT fill logic
into non-template helper functions. The per-Derived lambda would call the
Derived-specific compute method, then hand off to a shared helper for the
SuperScalar packing/unpacking and sparse matrix insertion.

Conceptually:
```cpp
// Non-template helper (instantiated once for all Derived types)
void scatter_gather_kkt_fill(
    int irows, int orows, int num_appl, bool vectorize,
    std::function<void(auto& x, auto& fx, auto& jx, auto& hx, auto& agx, auto& l)> compute_fn,
    std::function<void(int V, auto& jx, auto& hx, ...)> kkt_fill_fn,
    ...);

// Per-Derived method (much smaller body)
void constraints_jacobian_adjointgradient_adjointhessian(...) const {
    scatter_gather_kkt_fill(
        this->input_rows(), this->output_rows(), data.num_appl(),
        Derived::is_vectorizable && this->derived().enable_vectorization_,
        [&](auto& x, auto& fx, ...) { this->derived().compute_jah(x, fx, ...); },
        [&](int V, auto& jx, ...) { this->derived().kkt_fill_all(V, jx, ...); },
        ...);
}
```

**Implications:**

- *Runtime performance:* **Negligible impact.** The constraint methods are already
  called through virtual dispatch (GFModelCommon), so the compute call is already
  not inlined at the top level. Adding one more level of indirection within the
  method body (for the scatter/gather helper) should be invisible relative to the
  cost of the actual Jacobian/Hessian computation.
- *Code complexity:* Moderate. Requires extracting ~100 lines of logic into a
  helper, and ensuring the helper's signature correctly handles the Eigen types.
  The core compute dispatch (`derived().compute_jah(...)`) must remain in the
  per-Derived lambda for CRTP dispatch.
- *Risk:* Moderate. The constraint methods are on the solver hot path. Thorough
  benchmarking is required to verify no regression.

### 7.4 Conditional SuperScalar Virtual Methods in GFModelCommon

**Estimated reduction: 2-4% of total serial build time (~100-200s)**

`GFModelCommon` (gf_type_erasure.h) has **two overrides for every compute
method** — one for `double` and one for `SuperScalar`:

```cpp
void compute(CVR<InType> x, CVR<OutType> fx_) const override {
    this->data_.compute(x, fx_);  // instantiates T::compute for double
}
void compute(CVR<SuperInType> x, CVR<SuperOutType> fx_) const override {
    this->data_.compute(x, fx_);  // instantiates T::compute for SuperScalar
}
```

The SuperScalar override calls `this->data_.compute(super_x, super_fx)`,
triggering instantiation of the full SuperScalar compute path for type T — even
if T is never used with SuperScalar at runtime (e.g., a type that only appears
in tests/benchmarks that never call the vectorized integrator).

**Proposed:** Add a `constexpr` check on `T::is_vectorizable`. If false, the
SuperScalar overrides provide a stub (throw/assert) instead of delegating to
`this->data_`, avoiding the entire SuperScalar compute chain for
non-vectorizable types:

```cpp
void compute(CVR<SuperInType> x, CVR<SuperOutType> fx_) const override {
    if constexpr (T::is_vectorizable) {
        this->data_.compute(x, fx_);
    } else {
        throw std::logic_error("SuperScalar compute called on non-vectorizable type");
    }
}
```

Most composite expression types set `is_vectorizable = false` (only
`GenericFunction` sets it to true). This would eliminate SuperScalar
instantiation for the majority of stored types.

**Implications:**

- *Runtime performance:* None. Non-vectorizable types never take the SuperScalar
  path at runtime — the vectorized integrator only operates on types stored in
  `GenericFunction`, which are already type-erased and dispatch through
  GenericFunction's own SuperScalar path (which IS instantiated).
- *Code complexity:* Low. A simple `if constexpr` guard in ~5 method overrides.
- *Risk:* Low. If a non-vectorizable type is ever passed through the SuperScalar
  path, the throw provides a clear diagnostic rather than silent misbehavior.
  This would indicate a code bug (calling SuperScalar on a type that doesn't
  support it), not a regression from this change.

### 7.5 Investigate Benchmark TU Bloat

**Estimated reduction: 2-4% if actionable (~100-200s)**

Benchmark TUs have disproportionate instantiation counts:

| TU | Function Inst. | Class Inst. | vs. heaviest test |
|----|:--------------:|:-----------:|:-----------------:|
| bench_solvers.cpp | 65,032 | 58,319 | +21% |
| bench_integrators.cpp | 66,848 | 59,953 | +24% |
| bench_optimal_control.cpp | 65,041 | 58,357 | +21% |
| test_ocp_multiphase.cpp | 53,714 | 48,059 | (reference) |

Benchmarks have ~21-24% more instantiations than the heaviest test TU. This is
likely because `bench_common.h` pulls in the full `tycho/tycho.h` umbrella and
each benchmark file defines many diverse VF compositions for benchmarking
different scenarios.

**Investigation steps:**
1. Profile bench_solvers.cpp with `-ftime-trace-granularity=100` for finer detail
2. Identify which benchmark cases contribute the most unique expression types
3. Determine whether any benchmark cases can share expression types or be
   simplified without losing measurement validity
4. Consider splitting bench_solvers.cpp (163s, heaviest TU) into 2-3 files

**Implications:** Investigation only — no changes until findings are clear.

---

### Summary of Phase 2 Opportunities

| # | Change | Savings Est. | Difficulty | Runtime Impact |
|---|--------|:------------:|:----------:|:--------------:|
| 7.1 | Unity builds for tests | 700-1,100s (14-22%) | Low-moderate | None (build ergonomics only) |
| 7.2 | Examples at -O1 | 200-280s (4-5%) | Trivial | None (examples only) |
| 7.3 | Factor VectorImpl from constraints | 150-300s (3-6%) | Moderate | Negligible |
| 7.4 | Conditional SuperScalar in GFModelCommon | 100-200s (2-4%) | Low | None |
| 7.5 | Benchmark investigation | 100-200s (2-4%) | Investigation | — |

**Cumulative estimated savings:** 1,250-2,080s (25-41%) on top of Phase 1.
Combined with Phase 1's 336s savings, total reduction from original baseline
would be ~1,586-2,416s (29-45%), bringing serial build time to ~3,000-3,830s
(50-64 min).

---

## Appendix A: Compilation Time Waterfall (22-TU Aggregate)

```
DenseFunctionBase    ██████████████████████████████████████████████████ 3,311s
BumpAllocator        █████████████████████████████████████████          2,493s
std::make_shared     ██████████████████                                1,197s
ODEPhase             █████████████████                                 1,194s
ConstraintModel      ██████████████                                      946s
Integrator           ████████████                                        837s
GFStorage            ██████████                                          698s
GFModelCommon        ██████████                                          684s
GenericFunction      █████████                                           645s
NestedFunction_Impl  █████████                                           635s
StackTwoOutputs_Impl █████████                                           608s
TypeStorage          ███████                                             482s
ConstraintInterface  ███████                                             474s
ComputableBase       ██████                                              442s
LGLDefects           █████                                               327s
```

## Appendix B: Scripts Used

- `scripts/measure_tu_profile.py` — Per-TU RSS and ftime-trace measurement
- `scripts/analyze_ftime_trace.py` — Parse and aggregate ftime-trace JSON data

## Appendix C: Methodology

- **Build timing:** Parsed from `.ninja_log` (wall-clock per-TU, deduplicated, filtered to current targets)
- **Peak RSS:** Measured via `/usr/bin/time -v` on individual TU recompilations (ccache cleared)
- **Template profiling:** Clang `-ftime-trace` JSON analysis (22 of 25 TUs produced trace files)
- **Symbol analysis:** `nm --defined-only` with `-C` demangling on `.o` files
- **Duplicate detection:** Cross-referenced `nm --defined-only` output across all TUs

All measurements taken with the `linux-clang-release` preset, Clang 21.1.8.
Tests compiled at -O1, all other targets at -O3.
