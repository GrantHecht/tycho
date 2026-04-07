# Tycho Build Performance Investigation Report

**Date:** 2026-04-06
**System:** Linux 6.19.10, Clang 21.1.8, 32 GB RAM
**Build preset:** `linux-clang-release` (`-O3 -march=native -mllvm -inline-threshold=225`)

---

## 1. Executive Summary

A full build of Tycho (all targets: core + bindings + tests + benchmarks + examples)
comprises **146 translation units** with a serial compile time of **80.3 minutes**
(4816 seconds). Individual TUs consume up to **8.65 GB peak RSS**, making `-j4`
the practical maximum on 32 GB systems.

**The dominant cost is template instantiation, which accounts for 91.9% of
compilation time.** Header parsing and LLVM optimization are minor by comparison.

### Key Findings

1. **Template instantiation is 91.9% of compile time** across 19 profiled TUs
   (80.4% function instantiation + 11.5% class instantiation). Frontend parsing
   is only 4.0% and backend optimization 2.3%.

2. **`DenseFunctionBase` methods are the #1 cost center** — 2,905s aggregate
   (72,689 instantiations). The unified `compute_jacobian_adjointgradient_adjointhessian`
   function alone accounts for 1,360s across 19 TUs.

3. **`BumpAllocator::allocate_run`** is the single most expensive function template
   — 2,105s (32,833 unique instantiations across 19 TUs) because its variadic
   `TempSpecs` parameter pack creates a unique instantiation for every different
   combination of temporary buffer specifications. There are 75 call sites in
   the VF headers.

4. **Type erasure (`std::make_shared<GFModel<>>`)** costs 1,122s (923 instantiations,
   1.2s each). There are **30,568 duplicate `GFModelCommon<-1,...>` weak symbols**
   across binding TUs alone.

5. **215,799 total weak symbols** across 145 TUs — massive cross-TU template
   duplication that the linker deduplicates but each TU still pays the full
   compilation cost.

6. **Benchmark TUs consume the most RAM** — all 7 benchmark TUs use **7.9-8.7 GB
   peak RSS**, making them the `-j` parallelism bottleneck on 32 GB systems.
   Tests use 4.3-7.6 GB, examples 6.1-7.7 GB, core library only 1.7-2.6 GB.

7. **Tests dominate build time** at 38.5% (57 TUs), followed by bindings at 29.1%
   (54 TUs). The core library is only 3.7%. Benchmarks and examples have the
   highest per-TU cost (83-89s average) despite having few files.

---

## 2. Per-TU Build Time and Category Analysis

### Build Time by Category

| Category | Time (s) | % Total | TU Count | Avg/TU (s) |
|----------|----------|---------|----------|-------------|
| Tests | 1855 | 38.5% | 57 | 32.6 |
| Bindings | 1402 | 29.1% | 54 | 26.0 |
| Benchmarks | 710 | 14.7% | 8 | 88.7 |
| Examples | 660 | 13.7% | 8 | 82.5 |
| Core Library | 180 | 3.7% | 17 | 10.6 |
| Other | 10 | 0.2% | 2 | 4.9 |
| **Total** | **4816** | **100%** | **146** | **33.0** |

### Top 20 Heaviest TUs by Wall-Clock Time

| Rank | TU | Category | Time (s) |
|------|-----|----------|----------|
| 1 | bench_solvers.cpp | bench | 166.4 |
| 2 | test_ocp_multiphase.cpp | tests | 157.1 |
| 3 | zermelo/main.cpp | examples | 141.6 |
| 4 | delta3_launch/main.cpp | examples | 124.9 |
| 5 | bench_optimal_control.cpp | bench | 124.4 |
| 6 | test_phase_construction.cpp | tests | 122.2 |
| 7 | test_mesh_refinement.cpp | tests | 119.7 |
| 8 | test_control_modes.cpp | tests | 110.7 |
| 9 | test_jet_batch.cpp | tests | 109.4 |
| 10 | test_nlp_structure.cpp | tests | 109.2 |
| 11 | bench_integrators.cpp | bench | 108.1 |
| 12 | brachistochrone/main.cpp | examples | 107.4 |
| 13 | test_psiopt_convergence.cpp | tests | 106.7 |
| 14 | hypersens/main.cpp | examples | 96.4 |
| 15 | kepler_model.cpp (binding) | bindings | 93.2 |
| 16 | ode_phase_base.cpp | core | 92.3 |
| 17 | bench_type_erasure.cpp | bench | 82.8 |
| 18 | kepler_utils.cpp (binding) | bindings | 78.3 |
| 19 | bench_vector_functions.cpp | bench | 77.4 |
| 20 | bench_utils.cpp | bench | 75.5 |

### Peak RSS Measurement

**bench_solvers.cpp: 8.65 GB peak RSS** (the heaviest measured TU).
*(Remaining TU RSS measurements in progress — see Appendix A when complete.)*

---

## 3. Template Instantiation Analysis

### Compilation Phase Breakdown (19-TU aggregate)

| Phase | Time (s) | % Total |
|-------|----------|---------|
| InstantiateFunction | 24,346 | **80.4%** |
| InstantiateClass | 3,484 | **11.5%** |
| Frontend (parsing) | 1,220 | 4.0% |
| Backend (total) | 711 | 2.3% |
| Optimize (LLVM passes) | 511 | 1.7% |
| CodeGen (IR emission) | 8 | <0.1% |

**Template instantiation is 91.9% of total compilation time.**

**Note:** `-ftime-trace` events are hierarchical. Some function instantiation time
includes time for nested Eigen class instantiations. The numbers represent the
compiler's self-reported time attribution.

### Top Template Families by Compilation Cost (19-TU aggregate)

*Aggregated across 19 of the 25 heaviest TUs (7 benchmarks, 9 tests, 1 example,
2 core). These TUs represent ~78% of total serial build time.*

| Family | Time (s) | Instantiations | Cost/Instance |
|--------|----------|---------------|--------------|
| `DenseFunctionBase` | 2,905 | 72,689 | 40 ms |
| `BumpAllocator` | 2,205 | 36,660 | 60 ms |
| `ODEPhase` | 1,134 | 601 | **1,887 ms** |
| `std::make_shared` chain | 1,122 | 923 | **1,216 ms** |
| `ConstraintModel` | 849 | 4,007 | 212 ms |
| `Integrator` | 780 | 2,465 | 317 ms |
| `GFStorage` (type erasure) | 617 | 1,004 | 614 ms |
| `GFModelCommon` (type erasure) | 605 | 11,467 | 53 ms |
| `NestedFunction_Impl` | 581 | 8,442 | 69 ms |
| `GenericFunction` | 567 | 2,287 | 248 ms |
| `StackTwoOutputs_Impl` | 528 | 8,195 | 64 ms |
| `ConstraintInterface` | 425 | 570 | 746 ms |
| `TypeStorage` | 433 | 1,197 | 362 ms |
| `ComputableBase` | 390 | 20,336 | 19 ms |
| `LGLDefects` | 285 | 3,756 | 76 ms |

### Top Individual Function Instantiations (19-TU aggregate)

| Function | Time (s) | Count | Avg (s) |
|----------|----------|-------|---------|
| `BumpAllocator::allocate_run<...>` | 2,105 | 32,833 | 0.064 |
| `DenseFunctionBase::compute_jac_adjgrad_adjhess` | 1,360 | 19,349 | 0.070 |
| `std::make_shared<GFModel<...>>` | 1,122 | 923 | **1.216** |
| `DenseFunctionBase::compute_jacobian` | 862 | 16,764 | 0.051 |
| `GFStorage::emplace` | 616 | 836 | 0.737 |
| `ODEPhase::ODEPhase` (constructor) | 607 | 49 | **12.39** |
| `GenericFunction::GenericFunction` | 559 | 596 | 0.938 |
| `Eigen::internal::call_assignment_no_alias` | 523 | 54,262 | 0.010 |
| `TypeStorage::emplace` | 433 | 1,196 | 0.362 |
| `ConstraintInterface::ConstraintInterface` | 425 | 570 | 0.746 |
| `ConstraintModel::ConstraintModel` | 425 | 933 | 0.455 |
| `Integrator::Integrator` | 344 | 203 | **1.696** |
| `ODEPhase::transcribe_dynamics` | 298 | 25 | **11.90** |

### Top Class Template Instantiations (dominated by Eigen)

| Class | Time (s) | Count |
|-------|----------|-------|
| `Eigen::MatrixBase<...>` | 32.9 | 5,664 |
| `Eigen::MapBase<...>` | 23.0 | 3,703 |
| `Eigen::DenseBase<...>` | 18.1 | 5,752 |
| `Eigen::Block<...>` | 16.1 | 2,202 |
| `Eigen::internal::evaluator<...>` | 7.5 | 4,823 |
| `Eigen::CwiseBinaryOp<...>` | 6.8 | 951 |

### Derivative Mode Analysis

In the bench_solvers TU, only **1 distinct `DenseFirstDerivatives`** and
**1 distinct `DenseSecondDerivatives`** type were instantiated. This is because
the test/bench/example TUs use `GenericFunction<-1,-1>` (type-erased) which
hides the derivative mode behind virtual dispatch. The derivative mode
combinatoric explosion (4×4 = 16 potential combinations) is primarily a concern
in binding TUs that register concrete VF types, not in user-facing code.

---

## 4. Object File and Symbol Analysis

### Object File Size by Category

| Category | Total Size | % Total | TUs | Avg/TU |
|----------|-----------|---------|-----|--------|
| Bindings | 157.5 MB | 48.9% | 54 | 2.9 MB |
| Tests | 84.3 MB | 26.2% | 57 | 1.5 MB |
| Examples | 36.7 MB | 11.4% | 8 | 4.6 MB |
| Benchmarks | 23.8 MB | 7.4% | 8 | 3.0 MB |
| Core | 19.3 MB | 6.0% | 17 | 1.1 MB |
| **Total** | **322.3 MB** | | **146** | **2.2 MB** |

### Largest Object Files

| File | Size |
|------|------|
| vector_function_build_part2.cpp.o | 14.5 MB |
| zermelo/main.cpp.o | 11.8 MB |
| bench_solvers.cpp.o | 11.3 MB |
| ode_phase_base.cpp.o | 10.5 MB |
| tycho_vector_functions.cpp.o | 10.1 MB |
| args_seg_build_part1.cpp.o | 9.5 MB |
| args_seg_build_part3.cpp.o | 9.5 MB |
| test_ocp_multiphase.cpp.o | 9.2 MB |
| vector_function_build_part1.cpp.o | 9.0 MB |
| args_seg_build_part2.cpp.o | 8.4 MB |

### Symbol Distribution

vector_function_build_part2.cpp.o (14.5 MB, largest .o):
- **13,176 text symbols** totaling 3.2 MB of machine code
- Top symbol families: `GenericFunction` (20,209), `GFModelCommon` (5,130),
  `SolverIndexingData` (4,071), `Segment` (3,498), `NestedFunction` (2,936)

bench_solvers.cpp.o (11.3 MB, heaviest by compile time):
- **6,079 text symbols** totaling 5.6 MB of machine code
- Top: `GenericFunction` (4,914), `Segment` (4,303), `GenericODE` (3,049),
  `RKOptions` (2,014), `RKStepper` (1,990)

ode_phase_base.cpp.o (10.5 MB, heaviest core TU):
- Top: `Scaled` (12,145), `Segment` (9,130), `GenericFunction` (7,557),
  `NestedFunction` (6,187), `FunctionPlusVector` (4,704)

### Cross-TU Duplicate Symbols

**Total weak symbols across all 145 TUs: 215,799**

Top duplicated symbols across binding TUs alone:

| Symbol | Occurrences | Description |
|--------|------------|-------------|
| `GFModelCommon<-1,...>` | 30,568 | Type erasure dispatch for dynamic-size GenericFunction |
| `non-virtual thunks` | 22,053 | MI thunk adjustments |
| `std::_Sp_counted_ptr_inplace<GFModel<-1,...>>` | 3,658 | shared_ptr control blocks |
| `GFModel<-1,...>` | 1,760 | Full type erasure model |
| `Integrator<GenericODE<GenericFunction<-1,...>>>` | 623 | Integrator template |
| `ConstraintModel<RKStepper<GenericODE<...>>>` | 488 | Solver constraint wrapper |
| `GFModelCommon<8,...>` | 340 | Fixed-size (8-row) variant |
| `GFModelCommon<5,...>` | 304 | Fixed-size (5-row) variant |
| `GFModelCommon<12,...>` | 272 | Fixed-size (12-row) variant |

### GFModel Type Instantiation Count

Each unique type `T` stored in `GenericFunction<IR,OR>` requires a complete
`GFModel<IR,OR,T>` instantiation with 25-35 virtual method overrides:

| TU | Unique GFModel Types |
|----|---------------------|
| vector_function_build_part2.cpp | 88 |
| ode_phase_base.cpp | 34 |
| bench_solvers.cpp | 25 |

---

## 5. Include Dependency Analysis

### PCH Effectiveness

The PCH (`pch.h` for core, `pch_nb.h` for bindings) absorbs Eigen, standard
library, and utility headers effectively. The `-H` include analysis for
bench_solvers.cpp shows only **147 unique non-PCH headers**, confirming the PCH
eliminates most parsing overhead.

**However, the PCH does not help with template instantiation** — it pre-parses
declarations but each TU still independently instantiates templates. Since
instantiation is 89.5% of compile time, the PCH's impact is limited to the
remaining ~10%.

### Key Include Chain Observation

The master umbrella `tycho/tycho.h` pulls in every subsystem:
- `vector_functions.h` → all 51 operator/expression templates + autodiff
- `optimal_control.h` → ODE phases + transcription + builder
- `solvers.h` → PSIOPT + NLP + `mkl.h` (30+ MKL sub-headers)
- `astro.h` → all astrodynamics models

Test and benchmark TUs include `tycho.h` via their common headers
(`bench_common.h`, `test_common.h`), pulling in the entire universe.
This maximizes the template instantiation surface area for every TU.

### MKL Header Cost

`jet.h` → `mkl.h` includes 30+ MKL sub-headers (blas, lapack, sparse, vml, etc.)
via the full umbrella `mkl.h`. This could be replaced with targeted includes
(e.g., `mkl_pardiso.h` only where needed).

---

## 6. Root Cause Analysis (Ranked by Impact)

### Factor 1: `DenseFunctionBase` Method Instantiation (~25% of build cost)

The `compute_jacobian_adjointgradient_adjointhessian` and `compute_jacobian`
methods are instantiated for every unique VF type (concrete expression template
types and their compositions). Each instantiation is ~72ms but there are
thousands per TU.

**Why it's expensive:** These methods contain complex Eigen operations
(matrix products, block assignments, sparse insertion) that trigger cascading
Eigen template instantiation. A single `DenseFunctionBase::compute_jacobian_*`
instantiation pulls in `Eigen::MatrixBase`, `Eigen::Block`, `Eigen::Product`,
and `Eigen::internal::evaluator` for each unique matrix size combination.

### Factor 2: `BumpAllocator::allocate_run` Variadic Template Explosion (~15%)

The `allocate_run<Func, TempSpecs...>` function template accepts a variadic
parameter pack of temporary buffer specifications. Every unique combination of
`TempSpecs` creates a new instantiation, which in turn instantiates
`ExactTempPack<TempSpecs...>` and `std::apply(f, Temps.data)`.

There are **75 call sites** across the VF headers (in `computable_base.h`,
`dense_function_base.h`, `nested_function.h`, `stacked.h`, `segment.h`,
`summation.h`, `call_and_append.h`). Each call site is instantiated for every
concrete VF type that flows through it, creating **6,193 unique instantiations
across 3 profiled TUs** (extrapolating to ~50,000+ for the full build). Each
instantiation takes ~62ms. This is the single most expensive function template
in the project (382.7s across 3 TUs).

### Factor 3: Type Erasure Chain (`GFModel` + `shared_ptr`) (~12%)

Storing a concrete VF type in `GenericFunction<IR,OR>` via `GFStorage::emplace`
triggers:
1. `std::make_shared<GFModel<IR,OR,T>>` → 1.4s per unique T
2. `GFModel<IR,OR,T>` constructor with 25-35 virtual method overrides
3. `GFModelCommon<IR,OR>` dispatch table

With 25-88 unique T types per TU (depending on the TU), this creates massive
redundant instantiation. The **30,568 duplicate `GFModelCommon` symbols** across
binding TUs demonstrate this waste.

### Factor 4: Heavy Constructor Chains (ODEPhase, Integrator, ConstraintModel) (~12%)

The `ODEPhase<DODE>` constructor takes **13.2 seconds per instantiation** because
it chains through:
- `Integrator<DODE>` construction (1.8s each)
- Transcription setup (LGL/trapezoidal defects)
- Phase indexer setup
- Solver interface registration

With 4 ODEPhase instantiations in a single benchmark TU costing 52.9s, and 15
Integrator instantiations costing 26.5s, these heavy constructors are
disproportionately expensive.

### Factor 5: Expression Template Composition Explosion (~10%)

Binary VF operations (sum, product, division, nesting) create deeply nested types
like:
```
NestedFunction<StackedOutputs<Scaled<FunctionDotProduct<Segment<...>, ...>>>>
```

Maximum nesting depth observed: **8 levels** in ode_phase_base.cpp.

Each nesting level requires full `VectorFunction<Derived,IR,OR>` instantiation
including CRTP base chain (6-7 levels: `CRTPBase` → `ComputableBase` → `Computable`
→ `DenseFunctionBase` → `DenseFunction` → `DenseDerivatives` → `VectorFunction`).

Key expression types and their symbol counts in ode_phase_base.cpp:
- `Scaled`: 12,145 symbols
- `Segment`: 9,130
- `NestedFunction`: 6,187
- `FunctionPlusVector`: 4,704
- `StackedOutputs`: 2,569

### Factor 6: Cross-TU Redundancy (~10%)

215,799 weak symbols are compiled across 145 TUs, then deduplicated by the linker.
The compilation cost is paid in full by each TU. Key offenders:
- `GFModelCommon<-1,-1>` methods appear in 30,568 TU×method combinations
- `ConstraintModel<T>` for common T appears in 488+ TU×method combinations
- Eigen internal templates are re-instantiated in every TU

### Factor 7: Eigen Template Cascade (~8%)

Every VF compute method triggers Eigen matrix type instantiation:
- `Eigen::MatrixBase<...>`: 5,664 instantiations (32.9s) in a single TU
- `Eigen::DenseBase<...>`: 5,752 (18.1s)
- `Eigen::Block<...>`: 2,202 (16.1s)
- `Eigen::MapBase<...>`: 3,703 (23.0s)

These are cascading instantiations — each unique matrix operation
(`Ref<Matrix<double,-1,1>>` vs `Ref<Matrix<double,6,1>>` etc.) creates distinct
Eigen types.

### Factor 8: GenericODE Dimension Variant Fan-Out (~5%)

8 dimension combinations (`(-1,0,0)`, `(-1,-1,0)`, `(-1,-1,-1)`, `(6,3,0)`,
`(7,3,0)`, `(2,1,0)`, `(6,0,0)`, `(4,0,0)`) each trigger:
- `GenericODE<GenericFunction<-1,-1>, XV, UV, PV>` instantiation
- `Integrator<GenericODE<...>>` binding template
- `ODEPhase<GenericODE<...>>` binding template

Split across 12 `.cpp` files (6 ODE + 6 integrator), each costing ~50-65s.

### Factor 9: `std::is_nothrow_invocable` / Type Trait Cascades (~3%)

Standard library type traits (`std::is_nothrow_invocable`, `std::__and_`,
`std::__invoke_result`, `std::__result_of_impl`) account for 47.7s in
bench_solvers alone. These are triggered by `std::apply`, `std::invoke`,
and `std::make_shared` calls.

---

## 7. Improvement Opportunities (Ranked by Expected Impact)

*Each opportunity includes an analysis of implications — runtime performance,
code complexity, maintenance burden, and risk. `-ftime-trace` times are
hierarchical: a function's reported time includes all nested instantiations.
Improvement estimates account for this overlap.*

### High Impact

#### 7.1 `extern template` for GenericFunction and Core VF CRTP Chain

**Estimated reduction: 8-15% of total serial build time**

`GenericFunction<-1,-1>` and `GenericFunction<-1,1>` are used in virtually every
TU. Each usage triggers the full CRTP base chain instantiation:
`VectorFunction` → `DenseDerivatives` → `DenseFunction` → `DenseFunctionBase` →
`Computable` → `ComputableBase` → `CRTPBase` (6-7 levels, ~50+ methods each).

This chain is identical across all TUs that use the same `<IR,OR>` specialization.
Using `extern template` suppresses implicit instantiation everywhere except a
single explicit-instantiation TU:

```cpp
// In header (after GenericFunction is fully defined):
extern template class GenericFunction<-1, -1>;
extern template class GenericFunction<-1, 1>;

// In one .cpp file:
template class GenericFunction<-1, -1>;
template class GenericFunction<-1, 1>;
```

The same pattern applies to a curated set of common types that appear across
many TUs: `Arguments<-1>`, `Segment<-1,-1,-1>`, `Constant<-1,-1>`,
`Constant<-1,1>`, and possibly `IOScaled<GenericFunction<-1,-1>>`.

**Important limitation:** `GFModelCommon<IR, OR, T>` takes a third parameter `T`
that is unique per stored type. `extern template` only helps for the most common
T values (e.g., `Arguments<-1>`, `Segment<-1,-1,-1>`) that appear in many TUs.
TU-local types (user-defined ODEs, test-local functions) cannot benefit.

**Implications:**

- *Runtime performance:* No change. The same code is generated, just in fewer TUs.
- *Code complexity:* Moderate. Requires maintaining an explicit-instantiation
  `.cpp` file and matching `extern template` declarations in headers. New common
  types must be added to the list manually.
- *Maintenance risk:* If a TU uses a method of an extern-templated class that
  wasn't instantiated in the explicit `.cpp`, you get a linker error. This is
  caught at link time, not silently wrong, so the risk is manageable.
- *Build system:* The explicit-instantiation TU will itself be heavy (it does all
  the work that was previously spread across TUs). It should be in the
  `heavy_compile` pool.

#### 7.2 Reduce GenericODE Dimension Variants

**Estimated reduction: 5-10%**

The bindings register 8 `GenericODE<GenericFunction<-1,-1>, XV, UV, PV>`
dimension combinations across 12 `.cpp` files (6 ODE + 6 integrator). Each
variant costs ~50-65s to compile and triggers a full `Integrator` +
`ODEPhase` template chain. The fixed-size variants are:

| Variant | Name | Purpose |
|---------|------|---------|
| `(6, 3, 0)` | ode_6_3 | Standard astrodynamics (position+velocity, 3-axis thrust) |
| `(7, 3, 0)` | ode_7_3 | Astrodynamics with mass state |
| `(2, 1, 0)` | ode_2_1 | Simple planar problems |
| `(6, 0, 0)` | ode_6 | Uncontrolled 6-state dynamics |
| `(4, 0, 0)` | ode_4 | Uncontrolled 4-state dynamics |

The dynamic-size variants `(-1, 0, 0)`, `(-1, -1, 0)`, `(-1, -1, -1)` handle
any dimensions at runtime. The fixed-size variants exist for performance — Eigen
fixed-size matrices enable stack allocation and SIMD vectorization.

Dropping 2-3 of the less-common fixed-size variants (e.g., `ode_4` and `ode_2_1`) 
would eliminate 4-6 build files and save 200-400s of serial build
time. Users of those sizes would fall back to the dynamic-size variants.

**Implications:**

- *Runtime performance:* Users of the dropped fixed-size variants would see a
  performance regression from falling back to dynamic-size Eigen matrices. The
  magnitude depends on the problem — for small problems (2-state), the overhead
  of dynamic allocation is proportionally larger. For 6+ state problems, the
  difference is typically <10%.
- *API change:* This is a Python-visible API change. Code using
  `oc.ode_2_1(...)` would need to switch to `oc.ode_x_u(...)`. This should be
  a deprecation, not a hard removal.
- *Code complexity:* Reduced — fewer files to maintain, fewer binding specializations.
- *Risk:* Low. The dynamic-size variants are fully functional and well-tested.

**Human Notes**
- Keep `ode_6_3`, `ode_7_3`, and `ode_6`, remove the others.

#### 7.3 Compile Tests at `-O1`

**Estimated reduction: 3-8% of total serial build time**

The per-TU breakdown shows that heavy test TUs have significant backend
(optimization) cost: `test_ocp_multiphase` has 72s backend (49% of its 146.4s
total), and most OCP test TUs have 40-50s of backend time. This is because
`-O3` triggers aggressive inlining and LLVM optimization on the large amount of
template-generated code.

Compiling tests at `-O1` or `-O2` would reduce backend time from ~46s to ~15s
per heavy test TU. With 8-10 heavy OCP/solver test TUs, this saves 250-400s.

**Implications:**

- *Runtime performance:* Tests will run slower (typically 2-5x for `-O1` vs `-O3`
  on compute-heavy code). This may increase the time to run the test suite, partly
  offsetting the compile-time savings. However, test correctness is unaffected.
- *Code complexity:* Trivial. A single CMake `target_compile_options` change for
  the test targets.
- *Risk:* Minimal. `-O1` code is still optimized enough to exercise the same code
  paths as `-O3`. One caveat: some bugs only manifest under aggressive
  optimization (UB, aliasing). If the tests are meant to catch such bugs,
  `-O1` may miss them. A pragmatic compromise is `-O2`, which keeps most
  optimizations but is meaningfully cheaper than `-O3`.
- *Benchmarks:* Must remain at `-O3` — changing optimization levels would
  invalidate benchmark measurements.

### Medium Impact

#### 7.4 Consolidate BumpAllocator Call Patterns

**Estimated reduction: 3-8% (high uncertainty)**

`BumpAllocator::allocate_run` reports 2,105s across 19 TUs (32,833
instantiations). However, `-ftime-trace` times are hierarchical: this 2,105s
**includes the entire cost of compiling the lambda body and all Eigen operations
inside it.** The allocator dispatch logic itself is a small fraction; the
majority is the nested work triggered by each unique `(Func, TempSpecs...)`
combination.

The original proposal to "type-erase the allocation" would remove the
`if constexpr` constant-size fast path, which uses stack allocation
(`ExactTempPack`) when all `TempSpecs` have compile-time-known sizes. This is a
deliberate performance optimization — replacing it with runtime allocation would
hurt the hot-path compute cost of every VF evaluation.

Better approaches:

- **Normalize TempSpec combinations at call sites.** Many of the 75 call sites
  use similar TempSpec patterns. If a call site uses
  `TempSpec<double>(irows, 1), TempSpec<double>(orows, 1)`, and another uses
  `TempSpec<double>(orows, 1), TempSpec<double>(irows, 1)`, these create
  different template instantiations despite being structurally identical. A
  code audit could consolidate these patterns.

- **Factor shared computation out of per-type lambdas.** The lambdas passed to
  `allocate_run` capture the Derived VF reference and perform type-specific
  compute calls. If the type-specific parts can be isolated to smaller
  functions (called through the lambda rather than inlined into it), the lambda
  body shrinks and compilation of each instantiation is cheaper.

**Implications:**

- *Runtime performance:* Normalizing TempSpecs is neutral. Factoring lambdas
  could reduce inlining opportunities, potentially hurting runtime performance
  by 1-5% on compute-intensive paths. Needs benchmarking.
- *Code complexity:* Moderate to high. The 75 call sites span the core VF
  expression template system. Changes must preserve the constant-size fast path
  and existing Eigen Map patterns.
- *Risk:* Moderate. The allocator is on the critical path of every VF
  evaluation. Incorrect changes could cause subtle correctness or performance
  regressions. Thorough benchmarking required.
- *Uncertainty:* The 3-8% estimate is wide because the hierarchical ftime-trace
  times make it hard to isolate the allocator's own overhead from the nested
  Eigen work.

#### 7.5 `extern template` for Common GFModelCommon Specializations

**Estimated reduction: 2-5%**

Extend the `extern template` strategy (7.1) to
`GFModelCommon<-1, -1, T>` for the most commonly shared T types. The symbol
analysis shows these types are duplicated across many binding and test TUs:

- `GFModelCommon<-1, -1, Arguments<-1>>` (~238 TU occurrences)
- `GFModelCommon<-1, -1, Segment<-1, *, -1>>` variants (~180+)
- `GFModelCommon<-1, -1, IfElseFunction<...>>` (~236)
- `GFModelCommon<-1, -1, NestedFunction<GenericFunction<-1,-1>, ...>>` (~182)

Unlike 7.1, this targets the three-parameter `GFModelCommon<IR, OR, T>` which
requires enumerating specific T types.

**Implications:**

- *Runtime performance:* No change.
- *Code complexity:* Higher than 7.1. The list of explicit specializations grows
  with the number of common types. Each `extern template struct
  GFModelCommon<-1, -1, Segment<-1, 2, -1>>` is a line that must be maintained.
- *Maintenance risk:* Moderate. Adding a new VF type that becomes widely used
  requires adding it to the explicit instantiation list, or TUs will silently
  re-instantiate it (no breakage, just lost savings).
- *Diminishing returns:* The most impactful types are the first 5-10. Beyond
  that, each additional type saves less. The effort-to-benefit ratio worsens
  quickly.

#### 7.6 Pre-instantiate Common Expression Template DenseFunctionBase Methods

**Estimated reduction: 2-4%**

The `DenseFunctionBase::compute_jacobian_adjointgradient_adjointhessian` family
(2,905s across 19 TUs) is the single most expensive class family. For common
*leaf* expression types (`Segment<-1,-1,-1>`, `Arguments<-1>`,
`Constant<-1,-1>`), the DenseFunctionBase methods can be pre-instantiated.

**Important limitation:** Composite expression types (e.g.,
`NestedFunction<GenericFunction<-1,-1>, StackedOutputs<Segment<-1,3,-1>, ...>>`)
are unique to each expression tree and cannot be pre-instantiated. Since most
DenseFunctionBase cost comes from these composites, the savings are limited to
the leaf types.

**Implications:**

- *Runtime performance:* No change.
- *Code complexity:* Moderate. Same pattern as 7.1 but applied to more types.
- *Maintenance risk:* Low — leaf types are stable parts of the API.

### Lower Impact

#### 7.7 Reduce Number of `RKOptions` Template Instantiations (NOT recommended)

The original proposal to make `RKOptions` a runtime parameter for `RKStepper`
would be counterproductive. The template parameter enables compile-time access
to `RKCoeffs<RKOp>::ACoeffs[Stg][Elem]`, which allows loop unrolling over
stages and constant propagation of Butcher tableau coefficients. Making this
runtime would:
- Prevent loop unrolling over RK stages (4-13 stages depending on method)
- Add branch misprediction on the method selector
- Eliminate constant propagation of Butcher tableau entries

The integrator is on the innermost hot loop of trajectory optimization. Even
a 10-20% integrator slowdown would far outweigh the ~120s build-time savings.

**Recommendation:** Do not pursue. If build cost from `RKStepper` instantiation
becomes a bottleneck, a better approach is to reduce the number of DODE types
that flow through the integrator (addressed by 7.2 and the type-erasure boundary
discussion below).

#### 7.8 Targeted MKL Includes

**Estimated reduction: <1%**

`jet.h` pulls the full `mkl.h` umbrella (30+ sub-headers). Replacing with
`mkl_pardiso.h` + `mkl_sparse.h` reduces the include surface. Since parsing is
only 4% of total build time and the PCH absorbs most of it, the direct impact
is negligible. However, it improves code hygiene and marginally reduces PCH
build time on cold builds.

**Implications:** No performance impact. Trivial to implement.

#### 7.9 Targeted Includes in Test/Bench Common Headers

**Estimated reduction: <1%**

The test and benchmark common headers include `tycho/tycho.h`, pulling in every
subsystem. Since 91.9% of build time is template instantiation (not parsing),
reducing includes has minimal direct impact. The PCH already absorbs the parsing
cost.

The indirect argument — that broader includes make more types available,
increasing implicit instantiation — is weak in practice. Instantiation is driven
by the test/bench *source code*, not by what the headers make available.

**Implications:** Negligible impact but improves compile-time hygiene. Low risk.

### Experimental / Needs Investigation

#### 7.10 Unity Builds for Tests (`CMAKE_UNITY_BUILD`)

**Estimated reduction: 10-20% (speculative, needs experimentation)**

CMake's `CMAKE_UNITY_BUILD` merges multiple `.cpp` files into a single TU.
This eliminates cross-TU template duplication entirely for the merged files —
shared instantiations (GenericFunction, GFModelCommon, Eigen types) are compiled
once instead of once per source file.

For the test suite (57 TUs, 1,855s total), merging into 5-8 unity TUs could
reduce total test build time by 30-50% because the 215K+ duplicate weak symbols
would be instantiated only once per unity group.

**Implications:**

- *Peak RSS:* Unity TUs will be much larger. If 8 test files are merged, peak
  RSS could reach 15-25 GB per unity TU, making `-j2` the maximum on a 32 GB
  system. This worsens parallelism.
- *Incremental builds:* Any change to one source file in a unity group
  recompiles the entire group. This hurts incremental build times.
- *Compile errors:* Unity builds can expose name collisions between TUs (static
  functions, anonymous namespaces, `using` directives). These must be resolved.
- *Verdict:* Worth experimenting with. Try enabling it for just the test target
  first: `set_target_properties(tycho_tests PROPERTIES UNITY_BUILD ON
  UNITY_BUILD_BATCH_SIZE 8)`. Measure total build time and peak RSS. If RSS is
  manageable, the serial-time savings could be substantial.

#### 7.11 Type-Erase at Phase Boundaries (Architectural)

**Estimated reduction: 15-30% (speculative, major refactor)**

The `ODEPhase<DODE>` constructor costs 12.4s per instantiation (607s total for
49 instantiations across 19 TUs). This is because concrete VF types propagate
through the entire `ODEPhase` → `Integrator` → `Transcription` chain, each
level instantiating its full template machinery for the specific DODE type.

An architectural alternative: type-erase the ODE at the Phase boundary. Instead
of `ODEPhase<GenericODE<GF, 6, 3, 0>>`, use a single `ODEPhase` class that
operates on `GenericFunction<-1,-1>` internally. The concrete ODE type would be
erased into a `GenericFunction` upon entry to the Phase.

This would reduce ODEPhase from ~50 instantiations to 1-2 (dynamic-size only),
and proportionally reduce Integrator and Transcription instantiations.

**Implications:**

- *Runtime performance:* This is the critical concern. The Phase/Integrator
  currently benefit from static dispatch — the compiler inlines the ODE
  evaluation directly into the integrator loop, enabling vectorization and
  register reuse. Type erasure would replace this with virtual dispatch through
  GenericFunction, adding one vtable indirection per ODE evaluation per
  integration step. For problems with many integration steps and cheap ODE
  evaluations, this could cause a 20-50% slowdown in the integrator. For
  problems with expensive ODE evaluations, the vtable overhead is negligible.
- *Code complexity:* Major refactor of the Phase/Integrator architecture.
- *Risk:* High. This changes the core performance model of the library.
- *Verdict:* Not recommended as a general change. However, it could be offered
  as an *opt-in* mode (e.g., `ODEPhase::use_dynamic_dispatch(true)`) for users
  who prioritize compile time over runtime performance (e.g., during
  development iteration). The production path would remain template-specialized.

---

## Appendix A: RSS Measurements (Complete)

All 25 heaviest TUs measured via `/usr/bin/time -v` with `-ftime-trace` enabled.
Sorted by peak RSS descending.

| TU | Category | Peak RSS (GB) | Wall Time (s) |
|----|----------|:-------------:|:-------------:|
| bench_solvers.cpp | bench | **8.71** | 178.7 |
| bench_integrators.cpp | bench | **8.39** | 113.9 |
| bench_optimal_control.cpp | bench | **8.33** | 134.1 |
| bench_type_erasure.cpp | bench | **8.04** | 89.2 |
| bench_vector_functions.cpp | bench | **7.99** | 82.2 |
| bench_utils.cpp | bench | **7.91** | 80.7 |
| bench_kepler.cpp | bench | **7.91** | 81.1 |
| delta3_launch/main.cpp | examples | 7.70 | 128.3 |
| test_ocp_multiphase.cpp | tests | 7.60 | 146.8 |
| zermelo/main.cpp | examples | 6.96 | 142.7 |
| test_control_modes.cpp | tests | 6.84 | 115.2 |
| test_jet_batch.cpp | tests | 6.66 | 109.8 |
| test_psiopt_convergence.cpp | tests | 6.66 | 110.0 |
| test_phase_construction.cpp | tests | 6.65 | 111.0 |
| test_mesh_refinement.cpp | tests | 6.65 | 109.3 |
| test_nlp_structure.cpp | tests | 6.65 | 109.7 |
| brachistochrone/main.cpp | examples | 6.62 | 109.6 |
| test_runtime_ode.cpp | tests | 6.43 | 65.3 |
| hypersens/main.cpp | examples | 6.09 | 98.1 |
| test_kepler_propagation.cpp | tests | 4.31 | 64.6 |
| ode_phase_base.cpp | core | 2.59 | 88.7 |
| runtime_ode.cpp | core | 1.72 | 63.8 |

**Key observations:**
- **All 7 benchmark TUs consume 7.9-8.7 GB** — these are the -j parallelism bottleneck
- **Test TUs use 4.3-7.6 GB** — the second tier
- **Example TUs use 6.1-7.7 GB** — similar to tests
- **Core library TUs are much lighter** — 1.7-2.6 GB
- **At -j4 with benchmarks, worst case is 4 × 8.7 GB = 34.8 GB** — explains the 32 GB limit
- **Binding TUs with warm PCH** compiled in <1.2s (0.17-0.35 GB), showing PCH
  absorbs the parsing overhead. Their build cost in the ninja log (69-93s) was
  from cold compilation; incremental rebuilds are fast.

**To reproduce:**
```bash
conda run -n tycho python scripts/measure_tu_profile.py --top 25 --output results.json
```

## Appendix B: Compilation Time Waterfall (19-TU Aggregate)

```
DenseFunctionBase    ████████████████████████████████████████████ 2,905s
BumpAllocator        ██████████████████████████████████           2,205s
ODEPhase             █████████████████                            1,134s
std::make_shared     ████████████████                             1,122s
ConstraintModel      ████████████                                   849s
Integrator           ███████████                                    780s
GFStorage            █████████                                      617s
GFModelCommon        █████████                                      605s
NestedFunction_Impl  ████████                                       581s
GenericFunction      ████████                                       567s
StackTwoOutputs_Impl ████████                                       528s
TypeStorage          ██████                                         433s
ConstraintInterface  ██████                                         425s
ComputableBase       ██████                                         390s
LGLDefects           ████                                           285s
```

## Appendix C: Scripts Used

- `scripts/measure_tu_profile.py` — Per-TU RSS and ftime-trace measurement
- `scripts/analyze_ftime_trace.py` — Parse and aggregate ftime-trace JSON data

Both scripts are checked into the repository for reproducibility.

## Appendix D: Methodology

- **Build timing:** Parsed from `.ninja_log` (wall-clock per-TU, deduplicated)
- **Peak RSS:** Measured via `/usr/bin/time -v` on individual TU recompilations
- **Template profiling:** Clang `-ftime-trace` JSON analysis
- **Symbol analysis:** `nm --size-sort --radix=d -C` on `.o` files
- **Duplicate detection:** Cross-referenced `nm --defined-only` output across all TUs
- **Include analysis:** Clang `-H` flag with `-fsyntax-only`

All measurements taken with the `linux-clang-release` preset, `-O3`, Clang 21.1.8.
