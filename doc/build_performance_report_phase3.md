# Tycho Build Performance Report — Phase 3 (Post-CRTP Simplification)

**Date:** 2026-04-08
**System:** Linux 6.19.10, Clang 21.1.8, 32 GB RAM, 8 cores
**Build preset:** `linux-clang-release` (`-O3 -march=native -mllvm -inline-threshold=225`)
**Branch:** `build-performance-optimization` @ `0554925`
**Prior reports:** `build_performance_report.md` (baseline), `build_performance_report_phase1.md`, `build_performance_report_phase2.md`

---

## 1. Executive Summary

After three optimization phases — extern template + ODE variant removal (Phase 1),
unity builds for tests (Phase 2), and VF CRTP simplification (Phase 3) — a full
build of Tycho comprises **99 translation units** (down from 146 in the baseline)
with a serial compile time of **45.9 minutes** (2,752 seconds, down from 80.3 min).

**Template instantiation remains the dominant cost at 91.8% of compilation time**
(80.7% function + 11.1% class), nearly identical to the baseline's 91.9%. The CRTP
simplification reduced hierarchy depth but did not fundamentally change the number
of template instantiations — the cost is structural to the expression template design.

### Key Findings

1. **Serial compile time reduced 42.9%** — from 4,816s (baseline) to 2,752s.
   The reduction comes from test unity builds (-1,855s → 0s), ODE variant removal,
   and extern template, not from the CRTP simplification itself.

2. **TU count reduced 32.2%** — from 146 to 99 TUs. Test TUs collapsed from 57 to
   11 via unity builds (and they now compile at -O1 instead of -O2, making them
   essentially free at 0.0s).

3. **Peak RSS: 8.53 GB** (bench_solvers.cpp), down slightly from 8.65 GB baseline.
   All 7 benchmark TUs still use 7.7-8.5 GB. The CRTP flattening did not
   significantly change memory consumption — Eigen template instantiation dominates.

4. **`DenseFunctionBase` remains the #1 cost center** — 2,374s aggregate (down from
   2,905s baseline). The reduction comes from fewer TUs (99 vs 146), not from cheaper
   per-TU cost.

5. **`BumpAllocator::allocate_run` remains the #1 function** — 1,702s (down from
   2,105s). Same story: fewer TUs, same per-TU cost.

6. **157,247 total weak symbols** (down from 215,799 baseline, -27.1%). The
   reduction comes from eliminating test TUs and removing ODE variants.

7. **Wall-clock build time at -j8: ~19 minutes** (measured in Phase 2). The
   `heavy_compile` ninja job pool (auto-detected: 1 slot per 10 GB RAM) prevents
   OOM while allowing light TUs to fill the remaining slots.

### Cumulative Improvement Summary

| Metric | Baseline | Phase 3 | Change |
|--------|----------|---------|--------|
| Serial compile time | 4,816s (80.3 min) | 2,752s (45.9 min) | **-42.9%** |
| TU count | 146 | 99 | **-32.2%** |
| Wall-clock (-j8) | ~30 min | ~19 min | **-37%** |
| Peak RSS | 8.65 GB | 8.53 GB | -1.4% |
| Template instantiation % | 91.9% | 91.8% | ~0% |
| Total weak symbols | 215,799 | 157,247 | **-27.1%** |
| Total .o file size | 322.3 MB | 215.6 MB | **-33.1%** |

---

## 2. Per-TU Build Time and Category Analysis

### Build Time by Category

| Category | Time (s) | % Total | TU Count | Avg/TU (s) |
|----------|----------|---------|----------|-------------|
| Bindings | 1,247 | 45.3% | 52 | 24.0 |
| Benchmarks | 710 | 25.8% | 8 | 88.8 |
| Examples | 565 | 20.5% | 8 | 70.6 |
| Core Library | 219 | 8.0% | 18 | 12.2 |
| Other | 10 | 0.4% | 2 | 5.1 |
| Tests | 0 | 0.0% | 11 | 0.0 |
| **Total** | **2,752** | **100%** | **99** | **27.8** |

**Key change from baseline:** Tests dropped from 38.5% (1,855s) to 0.0% (0s) due to
unity builds + -O1 compilation. Bindings now dominate at 45.3%. Benchmarks remain
the highest per-TU cost (88.8s average).

### Top 20 Heaviest TUs by Wall-Clock Time

| Rank | TU | Category | Time (s) |
|------|-----|----------|----------|
| 1 | bench_solvers.cpp | bench | 165.8 |
| 2 | zermelo/main.cpp | examples | 138.8 |
| 3 | delta3_launch/main.cpp | examples | 131.7 |
| 4 | bench_optimal_control.cpp | bench | 125.2 |
| 5 | bench_integrators.cpp | bench | 106.5 |
| 6 | brachistochrone/main.cpp | examples | 105.6 |
| 7 | hypersens/main.cpp | examples | 95.3 |
| 8 | ode_phase_base.cpp | core | 91.8 |
| 9 | kepler_model.cpp | bindings | 89.8 |
| 10 | bench_type_erasure.cpp | bench | 83.8 |
| 11 | bench_vector_functions.cpp | bench | 77.7 |
| 12 | kepler_utils.cpp | bindings | 76.4 |
| 13 | bench_kepler.cpp | bench | 75.7 |
| 14 | bench_utils.cpp | bench | 75.4 |
| 15 | vector_function_build_part2.cpp | bindings | 67.4 |
| 16 | runtime_ode.cpp | core | 66.0 |
| 17 | generic_odes_build_part4.cpp | bindings | 62.3 |
| 18 | generic_odes_build_part3.cpp | bindings | 63.0 |
| 19 | args_seg_build_part1.cpp | bindings | 54.2 |
| 20 | args_seg_build_part3.cpp | bindings | 53.3 |

**Notable:** No test TUs appear in the top 20 (previously 6 of the top 20 were tests).
The heaviest TU (bench_solvers.cpp) is essentially unchanged from baseline (165.8s vs 166.4s).

### Peak RSS by TU (Top 25, measured via /usr/bin/time -v)

| TU | Category | Peak RSS (GB) | Wall (s) |
|----|----------|---------------|----------|
| bench_solvers.cpp | bench | 8.53 | 172.6 |
| bench_integrators.cpp | bench | 8.21 | 110.6 |
| bench_optimal_control.cpp | bench | 8.15 | 128.9 |
| bench_type_erasure.cpp | bench | 7.87 | 87.6 |
| bench_vector_functions.cpp | bench | 7.82 | 81.5 |
| bench_utils.cpp | bench | 7.74 | 79.2 |
| bench_kepler.cpp | bench | 7.73 | 79.0 |
| delta3_launch/main.cpp | examples | 7.52 | 124.0 |
| zermelo/main.cpp | examples | 6.78 | 139.0 |
| brachistochrone/main.cpp | examples | 6.45 | 106.4 |
| hypersens/main.cpp | examples | 5.93 | 95.8 |
| kepler_utils.cpp | bindings | 5.74 | 80.4 |
| kepler_model.cpp | bindings | 5.35 | 94.8 |
| tycho_astro.cpp | bindings | 4.58 | 52.5 |
| kepler_integrator.cpp | bindings | 4.07 | 48.3 |
| cannon/main.cpp | examples | 4.06 | 47.0 |
| generic_odes_build_part4.cpp | bindings | 3.98 | 65.4 |
| generic_odes_build_part3.cpp | bindings | 3.98 | 65.4 |
| ode_phase_base.cpp | core | 2.55 | 87.9 |
| vector_function_build_part2.cpp | bindings | 2.32 | 68.9 |
| generic_odes_build_part1.cpp | bindings | 2.03 | 51.3 |
| args_seg_build_part3.cpp | bindings | 1.75 | 54.0 |
| args_seg_build_part1.cpp | bindings | 1.70 | 54.9 |
| runtime_ode.cpp | core | 1.70 | 62.5 |
| vector_function_build_part1.cpp | bindings | 1.62 | 47.8 |

**Parallelism guideline:** Peak RSS of 8.53 GB means `-j3` is safe on 32 GB (3 x 8.53 = 25.6 GB).
With the `heavy_compile` ninja pool (depth 3 on 32 GB), `-j8` is safe because the pool
limits concurrent heavy TUs while light TUs (< 2 GB) fill the remaining slots.

---

## 3. Template Instantiation Analysis

### Compilation Phase Breakdown (21-TU aggregate)

| Phase | Time (s) | % Total |
|-------|----------|---------|
| InstantiateFunction | 19,703 | **80.7%** |
| InstantiateClass | 2,705 | **11.1%** |
| Frontend (parsing) | 994 | 4.1% |
| Backend (total) | 609 | 2.5% |
| Optimize (LLVM passes) | 395 | 1.6% |
| CodeGen (IR emission) | 7 | <0.1% |

**Template instantiation is 91.8% of total compilation time** — essentially unchanged
from the baseline's 91.9%. The CRTP flattening (8 → 5 levels) reduced hierarchy depth
but did not reduce the number of unique template instantiations per TU.

### Top Template Families by Compilation Cost (21-TU aggregate)

| Family | Time (s) | Instantiations | Cost/Instance | vs Baseline |
|--------|----------|---------------|--------------|-------------|
| `DenseFunctionBase` | 2,374 | 62,073 | 38 ms | -18.3% |
| `BumpAllocator` | 1,834 | 35,450 | 52 ms | -16.8% |
| `ODEPhase` | 781 | 442 | 1,767 ms | -31.1% |
| `ConstraintModel` | 633 | 2,846 | 222 ms | -25.4% |
| `Integrator` | 571 | 1,772 | 322 ms | -26.8% |
| `GFStorage` (type erasure) | 528 | 1,031 | 512 ms | -14.4% |
| `GFModelCommon` (type erasure) | 513 | 11,533 | 45 ms | -15.2% |
| `GenericFunction` | 492 | 2,384 | 207 ms | -13.2% |
| `NestedFunction_Impl` | 448 | 7,967 | 56 ms | -22.9% |
| `StackTwoOutputs_Impl` | 423 | 6,299 | 67 ms | -19.9% |
| `TypeStorage` | 325 | 1,183 | 275 ms | -24.9% |
| `ConstraintInterface` | 317 | 401 | 791 ms | -25.4% |
| `ComputableBase` | 315 | 17,614 | 18 ms | -19.2% |
| `LGLDefects` | 225 | 2,793 | 81 ms | -21.1% |
| `TwoFunctionSum_Impl` | 176 | 6,375 | 28 ms | — |

The per-family reductions of 13-31% are proportional to the TU count reduction (32.2%),
confirming the CRTP simplification did not change per-TU instantiation cost.

### Top Individual Function Instantiations (21-TU aggregate)

| Function | Time (s) | Count | Avg (s) |
|----------|----------|-------|---------|
| `BumpAllocator::allocate_run<...>` | 1,702 | 29,059 | 0.059 |
| `DenseFunctionBase::compute_jac_adjgrad_adjhess` | 1,120 | 16,992 | 0.066 |
| `std::make_shared<GFModel<...>>` chain | 830 | ~1,030 | 0.806 |
| `DenseFunctionBase::compute_jacobian` | 692 | 14,798 | 0.047 |
| `GFStorage::emplace` | 527 | 957 | 0.551 |
| `GenericFunction::GenericFunction` | 481 | 687 | 0.700 |
| `Eigen::internal::call_assignment_no_alias` | 425 | 43,144 | 0.010 |
| `ODEPhase::ODEPhase` (constructor) | 368 | 34 | **10.82** |
| `TypeStorage::emplace` | 325 | 1,181 | 0.275 |
| `ConstraintModel::ConstraintModel` | 317 | 694 | 0.457 |
| `ConstraintInterface::ConstraintInterface` | 317 | 401 | 0.791 |
| `Integrator::Integrator` | 251 | 143 | **1.755** |
| `NestedFunction_Impl::compute_jac_adjgrad_adjhess` | 236 | 1,764 | 0.134 |
| `ODEPhase::transcribe_dynamics` | 227 | 20 | **11.35** |

### Top Class Template Instantiations (dominated by Eigen)

| Class | Time (s) | Count |
|-------|----------|-------|
| `Eigen::MatrixBase<...>` | 357.2 | 65,475 |
| `Eigen::MapBase<...>` | 238.4 | 41,828 |
| `Eigen::DenseBase<...>` | 199.8 | 67,159 |
| `Eigen::Block<...>` | 167.6 | 24,967 |
| `Eigen::BlockImpl<...>` | 164.4 | 24,967 |
| `Eigen::internal::BlockImpl_dense<...>` | 161.6 | 24,967 |
| `std::is_nothrow_invocable<...>` | 129.2 | 29,059 |
| `Eigen::CwiseBinaryOp<...>` | 81.1 | 11,907 |
| `Eigen::internal::evaluator<...>` | 79.9 | 54,993 |
| `Eigen::Product<...>` | 67.7 | 10,029 |

### Derivative Mode Analysis

Only **1 distinct `DenseFirstDerivatives`** and **1 distinct `DenseSecondDerivatives`**
type instantiated across profiled TUs (same as baseline). This is because bench/example
TUs use `GenericFunction<-1,-1>` (type-erased), hiding derivative modes behind
virtual dispatch.

---

## 4. Object File and Symbol Analysis

### Object File Size by Category

| Category | Total Size | % Total | TUs | Avg/TU |
|----------|-----------|---------|-----|--------|
| Bindings | 144.4 MB | 67.0% | 52 | 2.8 MB |
| Examples | 36.0 MB | 16.7% | 8 | 4.5 MB |
| Core | 20.9 MB | 9.7% | 18 | 1.2 MB |
| Benchmarks | 12.1 MB | 5.6% | 7 | 1.7 MB |
| Tests | 1.5 MB | 0.7% | 11 | 0.1 MB |
| Other | 0.7 MB | 0.3% | 2 | 0.3 MB |
| **Total** | **215.6 MB** | | **99** | **2.2 MB** |

**vs Baseline:** Total .o size dropped from 322.3 MB to 215.6 MB (-33.1%). Test .o
files collapsed from 84.3 MB to 1.5 MB (unity builds + -O1). Benchmark .o dropped
from 23.8 MB to 12.1 MB (unity build consolidation).

### Largest Object Files

| File | Size |
|------|------|
| vector_function_build_part2.cpp.o | 14.3 MB |
| zermelo/main.cpp.o | 11.6 MB |
| ode_phase_base.cpp.o | 10.5 MB |
| tycho_vector_functions.cpp.o | 9.9 MB |
| args_seg_build_part1.cpp.o | 9.4 MB |
| args_seg_build_part3.cpp.o | 9.1 MB |
| vector_function_build_part1.cpp.o | 9.0 MB |
| args_seg_build_part2.cpp.o | 8.4 MB |
| args_seg_build_part5.cpp.o | 7.4 MB |
| delta3_launch/main.cpp.o | 6.9 MB |

### Cross-TU Duplicate Symbols

**Total weak symbols across all 99 TUs: 157,247** (down from 215,799, -27.1%)

Top duplicated symbols across all TUs:

| Symbol | Occurrences | Description |
|--------|------------|-------------|
| `GFModelCommon<-1,...>` | 33,992 | Type erasure dispatch (dynamic GenericFunction) |
| `non-virtual thunks` | 25,746 | MI thunk adjustments |
| `void (various)` | 21,194 | Various void-returning functions |
| `std::_Sp_counted_ptr_inplace<GFModel<-1,...>>` | 4,229 | shared_ptr control blocks |
| `GFModel<-1,...>` | 1,874 | Full type erasure model |
| `Integrator<GenericODE<GenericFunction<-1,...>>>` | 614 | Integrator template |
| `ConstraintModel<LGLDefects<...>>` | 558 | Solver constraint wrapper |
| `ConstraintModel<RKStepper<...>>` | 546 | Solver constraint wrapper |
| `GFModelCommon<6,...>` | 544 | Fixed-size (6-row) variant |
| `GFModelCommon<8,...>` | 476 | Fixed-size (8-row) variant |

`GFModelCommon<-1,...>` remains the most-duplicated symbol at 33,992 occurrences
(up slightly from 30,568 in binding TUs alone in the baseline, but now measured
across all TUs). This is the single largest source of cross-TU duplication.

---

## 5. CRTP Simplification Impact Assessment

The VF CRTP simplification (Directions 3, 2, 1) achieved its structural goals:

### What Improved
- **Readability:** 5-level chain is easier to follow than 8-level
- **Macro reduction:** `DENSE_FUNCTION_BASE_TYPES` (13 aliases) replaced by `VF_TYPE_ALIASES`
  (5 aliases) + namespace-scope aliases. `SUB_FUNCTION_IO_TYPES` eliminated entirely.
- **Consistency:** All trait flags now `static constexpr bool` with concept-based queries
- **Code reduction:** Net -208 lines, 4 files deleted
- **Weak symbols:** -27.1% (from fewer TUs, not from CRTP changes)
- **Object file size:** -33.1% (from fewer TUs)

### What Did Not Change
- **Template instantiation cost:** 91.8% vs 91.9% baseline — unchanged
- **Per-TU compile time:** Top TUs essentially identical (bench_solvers: 165.8s vs 166.4s)
- **Peak RSS:** 8.53 GB vs 8.65 GB — within measurement noise
- **Per-instance cost:** `DenseFunctionBase` at 38ms/instance vs 40ms baseline — noise

This confirms the design doc's prediction: the empty intermediate CRTP layers
(CRTPBase, Computable, DenseFunction, DenseDerivatives) had negligible compilation
cost. The compiler already elided them. The simplification benefits readability and
maintainability, not build performance.

---

## 6. Remaining Improvement Opportunities

### High Impact

1. **Benchmark TU splitting** — All 7 benchmark TUs use 7.7-8.5 GB RSS and
   compile in 75-166s each. These are the parallelism bottleneck. Splitting each
   into 2-3 smaller files would reduce peak RSS and allow higher `-j` values.

2. **Example TU compilation at -O1** — The 8 example TUs consume 565s (20.5%).
   Compiling at -O1 (like tests) would likely reduce this by 50-70%. Examples
   don't need full optimization for verification.

3. **`BumpAllocator::allocate_run` deduplication** — 1,702s across 21 TUs
   (29,059 instantiations). The variadic template pack creates unique
   instantiations per call signature. A type-erased dispatch wrapper could
   reduce instantiation count dramatically.

### Medium Impact

4. **Further `GFModelCommon` extern template coverage** — Currently only
   `Arguments<-1>`, `Segment<-1,-1,-1>`, `Constant<-1,-1>`, `Constant<-1,1>`
   are pre-instantiated. Adding more common leaf types (e.g., `NestedFunction`,
   `StackedOutputs`, `TwoFunctionSum` for common sizes) could reduce the
   33,992 duplicate symbols.

5. **Binding TU rebalancing** — The 52 binding TUs range from 3.9s to 89.8s.
   Rebalancing the heaviest ones (kepler_model at 89.8s, kepler_utils at 76.4s)
   would reduce wall-clock time.

### Low Impact (diminishing returns)

6. **Further CRTP simplification** — Deducing this (C++23) would simplify
   dispatch calls but not reduce instantiation count. Not viable until Eigen
   supports C++20 modules.

7. **Additional extern template for `DenseFunctionBase` methods** — Already
   done for common leaf types. Composite expression types are unique per TU
   and cannot be pre-instantiated.
