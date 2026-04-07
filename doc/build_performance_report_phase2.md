# Tycho Build Performance Report — Phase 2 (Unity Builds)

**Date:** 2026-04-07
**System:** Linux 6.19.10, Clang 21.1.8, 32 GB RAM
**Build preset:** `linux-clang-release`
**Branch:** `build-performance-optimization`

---

## 1. Executive Summary

Phase 2 enabled CMake unity builds for the `tycho_tests` target (batch size 8),
merging 46 heavy test sources into 6 unity TUs. This eliminated cross-TU template
duplication within each batch, producing a **dramatic reduction in test compile time**.

### Phase 2 Result

| Metric | Baseline | Phase 1 | Phase 2 | Total Change |
|--------|----------|---------|---------|:------------:|
| **Serial compile time** | 5,415s (90 min) | 5,079s (85 min) | **3,286s (55 min)** | **-39.3%** |
| TU count | 146 | 145 | 105 | -41 |
| Test compile time | 2,506s | 2,233s | **484s** | **-80.7%** |
| Peak RSS (heaviest TU) | 8.71 GB | 8.61 GB | 8.62 GB | -1.0% |
| Peak RSS (heaviest unity TU) | — | — | **8.88 GB** | — |
| Total weak symbols | 215,799 | 210,517 | **186,742** | **-13.5%** |
| Total .o file size | 322.3 MB | 290.9 MB | **259.8 MB** | **-19.4%** |

### Key Findings

1. **Test compile time reduced 78%** — from 2,233s (Phase 1) to 484s, a saving
   of 1,749s. The 6 unity TUs average 78s each vs the previous 57 individual TUs
   averaging 39s each. Shared template instantiations (GenericFunction, Eigen,
   DenseFunctionBase for common types) are now compiled once per batch of 8
   instead of 8 times.

2. **Unity TU RSS is manageable** — the heaviest unity TU (unity_4) peaks at
   8.88 GB, comparable to benchmark TUs (~8.6 GB). At -j4 on 32 GB, the worst
   case is 4 x 8.88 = 35.5 GB, which is tight but within the job pool's
   throttling capability. No batch size reduction was needed.

3. **Weak symbols reduced 13.5%** — from 210,517 to 186,742. The unity batching
   eliminated ~23,775 duplicate weak symbols that were previously compiled in
   individual test TUs. The remaining 186K symbols are primarily from binding,
   benchmark, and example TUs.

4. **Object file size reduced 19.4%** — test .o files went from 65.3 MB (57 TUs)
   to 34.1 MB (17 TUs, including 6 unity + 11 light). Template deduplication
   within unity groups produces smaller combined output.

5. **Non-test TUs are unchanged** — benchmarks (686s), examples (630s), bindings
   (1,262s), and core (224s) are identical to Phase 1. These categories are the
   remaining optimization targets.

---

## 2. Per-TU Build Time and Category Analysis

### Build Time by Category

| Category | Time (s) | % Total | Phase 1 (s) | Change |
|----------|----------|---------|:-----------:|--------|
| Bindings | 1,262 | 38.4% | 1,278 | -1.3% |
| Benchmarks | 686 | 20.9% | 694 | -1.2% |
| Examples | 630 | 19.2% | 631 | -0.2% |
| Tests | 484 | 14.7% | 2,233 | **-78.3%** |
| Core | 224 | 6.8% | 233 | -3.9% |
| **Total** | **3,286** | **100%** | **5,079** | **-35.3%** |

Tests dropped from the #1 category (44% of build time) to #4 (15%). Bindings
are now the dominant cost at 38%.

### Unity TU Timing

| Unity TU | Time (s) | Contents (batch of 8) |
|----------|----------|----------------------|
| unity_4 | 126.2 | OC tests (multiphase, construction, mesh, etc.) |
| unity_5 | 90.6 | OC tests (control modes, builder) + solver tests |
| unity_2 | 74.3 | VF tests (hessian, composition chains, etc.) |
| unity_3 | 71.9 | VF tests (generic function, cwise stress, etc.) |
| unity_0 | 55.7 | Utils + VF tests (args, constant, scaled, etc.) |
| unity_1 | 47.9 | VF tests (norms, stacked, composition, etc.) |
| **Total** | **466.6** | **46 sources in 6 unity TUs** |

Plus 11 light test TUs (tycho_tests_light) at ~17s total.

### Top 15 Heaviest TUs

| Rank | TU | Category | Time (s) |
|------|-----|----------|----------|
| 1 | bench_solvers.cpp | bench | 161.4 |
| 2 | zermelo/main.cpp | examples | 135.7 |
| 3 | **unity_4_cxx.cxx** | **tests** | **126.2** |
| 4 | bench_optimal_control.cpp | bench | 120.0 |
| 5 | delta3_launch/main.cpp | examples | 117.4 |
| 6 | brachistochrone/main.cpp | examples | 103.2 |
| 7 | bench_integrators.cpp | bench | 102.6 |
| 8 | hypersens/main.cpp | examples | 90.7 |
| 9 | **unity_5_cxx.cxx** | **tests** | **90.6** |
| 10 | kepler_model.cpp | bindings | 86.6 |
| 11 | ode_phase_base.cpp | core | 85.4 |
| 12 | bench_type_erasure.cpp | bench | 80.6 |
| 13 | bench_vector_functions.cpp | bench | 75.1 |
| 14 | kepler_utils.cpp | bindings | 74.7 |
| 15 | **unity_2_cxx.cxx** | **tests** | **74.3** |

---

## 3. Peak RSS Measurement

Profiled top 15 TUs (all non-test, since unity TUs require separate measurement):

| TU | Category | Peak RSS (GB) | Wall Time (s) |
|----|----------|:-------------:|:-------------:|
| bench_solvers.cpp | bench | **8.62** | 177.0 |
| bench_integrators.cpp | bench | **8.29** | 113.1 |
| bench_optimal_control.cpp | bench | **8.24** | 132.8 |
| bench_type_erasure.cpp | bench | 7.96 | 89.2 |
| bench_vector_functions.cpp | bench | 7.90 | 82.6 |
| bench_utils.cpp | bench | 7.82 | 81.3 |
| bench_kepler.cpp | bench | 7.82 | 80.6 |
| delta3_launch/main.cpp | examples | 7.61 | 127.5 |
| zermelo/main.cpp | examples | 6.87 | 142.1 |
| brachistochrone/main.cpp | examples | 6.53 | 108.0 |
| hypersens/main.cpp | examples | 6.00 | 97.4 |
| kepler_utils.cpp | bindings | 5.81 | 81.9 |
| kepler_model.cpp | bindings | 5.42 | 95.8 |
| ode_phase_base.cpp | core | 2.60 | 89.2 |
| vector_function_build_part2.cpp | bindings | 2.34 | 69.7 |

**Heaviest unity TU (unity_4): 8.88 GB** (measured separately via `/usr/bin/time -v`).

**Parallelism assessment:** At -j4, worst case is 4 heavy TUs at ~8.9 GB = 35.6 GB.
The `heavy_compile` ninja job pool limits concurrent heavy TUs, so this is
manageable on 32 GB with the existing pool configuration.

---

## 4. Template Instantiation Analysis (12-TU Aggregate)

Note: Only 12 of 15 profiled TUs produced trace files (3 examples share the
`main.cpp` filename, causing trace file overwrites). Unity TUs are not included
in this aggregate (they were not profiled with ftime-trace).

### Compilation Phase Breakdown

| Phase | Time (s) | % Total |
|-------|----------|---------|
| InstantiateFunction | 16,059 | **81.2%** |
| InstantiateClass | 2,288 | **11.6%** |
| Frontend (parsing) | 792 | 4.0% |
| Backend (total) | 385 | 1.9% |
| Optimize (LLVM passes) | 252 | 1.3% |
| CodeGen (IR emission) | 4 | <0.1% |

Template instantiation remains 92.8% of compile time for non-test TUs.
This is expected — the Phase 2 change only affected test TUs, which are not
included in this profile.

---

## 5. Object File and Symbol Analysis

### Object File Size by Category

| Category | Total Size | % Total | Phase 1 | Change |
|----------|-----------|---------|---------|--------|
| Bindings | 144.9 MB | 55.8% | 144.9 MB | 0% |
| Examples | 36.0 MB | 13.8% | 36.0 MB | 0% |
| Tests | 34.1 MB | 13.1% | 65.3 MB | **-47.8%** |
| Benchmarks | 23.3 MB | 9.0% | 23.3 MB | 0% |
| Core | 20.9 MB | 8.0% | 20.9 MB | 0% |
| **Total** | **259.8 MB** | | **290.9 MB** | **-10.7%** |

### Cross-TU Duplicate Symbols

**Total weak symbols: 186,742** (Phase 1: 210,517, **-11.3%**)

The 23,775 eliminated weak symbols were duplicates across test TUs that are
now compiled once per unity batch instead of once per source file.

---

## 6. Remaining Improvement Opportunities

With tests now at 15% of build time (down from 44%), the remaining cost centers are:

| Category | Time (s) | % of Total | Status |
|----------|----------|-----------|--------|
| Bindings | 1,262 | 38.4% | Unchanged — extern template helps minimally |
| Benchmarks | 686 | 20.9% | Unchanged — unique expression types per TU |
| Examples | 630 | 19.2% | **7.2: Compile at -O1** would save ~200-280s |
| Core | 224 | 6.8% | Dominated by ode_phase_base (86s) + vf_instantiations (37s) |

### Actionable next steps (from Phase 1 report Section 7)

| # | Change | Expected Savings | Status |
|---|--------|:----------------:|--------|
| 7.2 | Compile examples at -O1 | 200-280s (6-9%) | Ready to implement |
| 7.3 | Factor VectorImpl from constraints | 150-300s (5-9%) | Needs investigation |
| 7.4 | Conditional SuperScalar in GFModelCommon | 100-200s (3-6%) | Needs investigation |
| 7.5 | Benchmark TU investigation | 100-200s (3-6%) | Needs investigation |

If all remaining opportunities are realized, serial time could reach ~2,400-2,800s
(40-47 min), a **48-56% total reduction** from the original 5,415s baseline.

---

## Appendix A: Collision Fixes for Unity Build

Three name collisions were resolved:

1. `make_brach_guess()` — renamed to `make_control_modes_brach_guess()` in
   `test_control_modes.cpp` (collision with `test_phase_wrapper.cpp`)
2. `DragBrach_Impl` / `DragBrach` — renamed to `SyntaxDragBrach_Impl` /
   `SyntaxDragBrach` in `test_ode_syntax_prototypes.cpp` (collision with
   `test_multi_param_ode.cpp`)
3. `DragBrach_Impl` / `ThreeParam_Impl` — wrapped in anonymous namespace in
   `test_multi_param_ode.cpp` (note: anonymous namespaces don't prevent
   collisions within unity TUs — the rename in file #2 was the actual fix)

## Appendix B: Methodology

- **Build timing:** ccache cleared, `ninja -t clean`, full `-j4` build, parsed `.ninja_log`
- **Peak RSS:** `/usr/bin/time -v` on individual TU recompilations (ccache cleared)
- **Unity TU RSS:** Measured separately for unity_4 (heaviest by ninja log time)
- **Template profiling:** Clang `-ftime-trace` on 12 non-test TUs
- **Symbol analysis:** `nm --defined-only` across all 105 current targets

All measurements with `linux-clang-release` preset, Clang 21.1.8.
Tests at -O1, all other targets at -O3.
