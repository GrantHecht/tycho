# Build Performance Report — 2026-04-18

**Branch:** `ivp-solver-modernization` (post-SP3 completion)
**Commit:** `a3a8133`
**System:** Linux, 16 cores, 31 GB RAM, clang preset `linux-clang-release`
**Procedure:** `doc/build_performance_analysis.md`

## Top-line numbers

| Metric | Value |
|---|---|
| Total serial compile time (`all` target) | **5 027 s ≈ 83.8 min** |
| TU count | **142** |
| Peak RSS (heaviest TU) | **8.19 GB** (`zermelo_cpp` main.cpp) |
| Max safe parallelism on 31 GB | **⌊31 / 8.19⌋ = 3**, so `-j2` (1 heavy + 1 light) |
| Template instantiation share | **87.8 %** (77.4 % function + 10.4 % class) |
| Parsing share | **4.3 %** (PCH is working) |
| Backend + optimize + codegen | **7.8 %** |
| Weak symbols (pre-linker-dedup) | **253 601** |

---

## Where is build time going?

### Category breakdown

```
 2131 s   42.4 %   31 TUs, avg 68.7 s   examples       ← biggest single bucket
 1358 s   27.0 %   52 TUs, avg 26.1 s   bindings
  698 s   13.9 %   29 TUs, avg 24.1 s   tests
  543 s   10.8 %   11 TUs, avg 49.4 s   benchmarks
  297 s    5.9 %   19 TUs, avg 15.6 s   core_lib
```

**Observation:** the library itself is cheap. 89.5 % of serial build time is spent on TUs that *consume* the library (examples, bindings, tests, benchmarks), not building the library.

### Top 10 TUs by wall time

```
178.7 s   bench/cpp/.../bench_solvers.cpp.o
176.2 s   examples/.../zermelo/main.cpp.o
143.0 s   examples/.../delta3_launch/main.cpp.o
124.0 s   examples/.../brachistochrone/main.cpp.o
123.8 s   bench/cpp/.../bench_optimal_control.cpp.o
114.8 s   examples/.../hypersens/main.cpp.o
112.7 s   examples/.../betts_low_thrust/main.cpp.o
 98.3 s   src/bindings/.../_tychopy_heavy.dir/astro/kepler_model.cpp.o
 97.1 s   examples/.../optimal_docking/main.cpp.o
 85.5 s   src/.../optimal_control/ode_phase_base.cpp.o
```

Fourteen TUs take > 70 s each. Nine of them are one-file example programs.

### Peak RSS — top 10

```
 8.19 GB   main.cpp (zermelo)          151 s wall
 7.67 GB   main.cpp (delta3_launch)    184 s wall
 7.19 GB   main.cpp (betts_low_thrust) 120 s wall
 7.12 GB   bench_solvers.cpp           194 s wall
 7.11 GB   main.cpp (brachistochrone)  134 s wall
 6.67 GB   bench_optimal_control.cpp   133 s wall
 6.56 GB   main.cpp (hypersens)        121 s wall
 6.44 GB   main.cpp (optimal_docking)  104 s wall
 5.90 GB   kepler_utils.cpp (bindings)  85 s wall
 5.71 GB   main.cpp (parallel_parking)  77 s wall
```

At 8 GB peak, two parallel heavy TUs already saturate a 16 GB system; three saturate 32 GB. The `heavy_compile` ninja pool (1 slot per 10 GB, min 1) is currently what keeps `-j8` safe on a 32 GB box, but on this 31 GB system it resolves to depth 3, and in practice any pair of heavy example TUs pushes us into swap.

---

## What specific templates are expensive?

### Phase split (aggregate)

```
InstantiateFunction  11 431 s   77.4 %
InstantiateClass      1 542 s   10.4 %
Backend                 670 s    4.5 %
Frontend                638 s    4.3 %
Optimize                477 s    3.2 %
CodeGen                   8 s    0.1 %
```

Total (per trace aggregation): ~14 768 s. That is larger than the wall-clock total (5 027 s) because per-TU phase time counts parallel clang internal phases, not wall time.

**PCH is effective** — the "TOP 20 HEADERS BY PARSE TIME" section is empty, meaning no significant parse cost survives the PCH. The entire build is template instantiation.

### Top function templates (by aggregate instantiation time)

```
 903 s  18 817x  14/14 TUs   tycho::utils::BumpAllocator::allocate_run
 605 s  10 004x  14/14 TUs   tycho::vf::DenseFunctionBase::compute_jacobian_adjointgradient_adjointhessian
 456 s   1 022x  14/14 TUs   std::make_shared
 456 s   1 050x  14/14 TUs   std::shared_ptr::shared_ptr
 455 s   1 037x  14/14 TUs   std::__shared_ptr::__shared_ptr
 455 s   1 035x  14/14 TUs   std::__shared_count::__shared_count
 453 s   1 022x  14/14 TUs   std::_Sp_counted_ptr_inplace::ctor
 371 s   9 058x  14/14 TUs   tycho::vf::DenseFunctionBase::compute_jacobian
 338 s     962x  14/14 TUs   tycho::vf::GFStorage::emplace
 302 s     617x  14/14 TUs   tycho::vf::GenericFunction::GenericFunction
 253 s  23 491x  14/14 TUs   Eigen::internal::call_assignment_no_alias
 238 s      76x  12/14 TUs   tycho::integrators::Integrator::Integrator
 158 s      14x   7/14 TUs   tycho::oc::ODEPhase::ODEPhase
 157 s     809x  14/14 TUs   tycho::utils::TypeStorage::emplace
 149 s     166x  12/14 TUs   tycho::solvers::ConstraintInterface::ConstraintInterface
 149 s     568x  14/14 TUs   tycho::solvers::ConstraintModel::ConstraintModel
 134 s      30x  12/14 TUs   tycho::integrators::Integrator::set_method
 134 s     240x  12/14 TUs   tycho::integrators::Integrator::init_stepper_and_controller
```

### Tycho class-family roll-up

```
 1 268 s   34 627x   tycho::vf::DenseFunctionBase   ← single biggest family
 1 006 s   23 597x   tycho::utils::BumpAllocator
   566 s    1 661x   tycho::integrators::Integrator
   374 s      202x   tycho::oc::ODEPhase
   338 s    1 008x   tycho::vf::GFStorage
   325 s    9 804x   tycho::vf::GFModelCommon
   309 s    1 676x   tycho::vf::GenericFunction
   297 s    1 479x   tycho::solvers::ConstraintModel
   228 s    3 168x   tycho::vf::StackTwoOutputs_Impl
   225 s    5 088x   tycho::vf::NestedFunction_Impl
   172 s   10 364x   tycho::vf::ComputableBase
   157 s      814x   tycho::utils::TypeStorage
   149 s      166x   tycho::solvers::ConstraintInterface
   106 s    7 656x   tycho::integrators::RKStepper
```

**The DSL is the cost.** `DenseFunctionBase` + `BumpAllocator` + `Integrator` + `GenericFunction` + `GFModelCommon` + `GFStorage` together account for ~3 800 s of the 13 000 s InstFn+InstCls budget. Every TU that constructs a VectorFunction pays to re-instantiate the same 3-derivative pipeline.

### Eigen class templates (top instantiated)

```
 195 s  35 884x  Eigen::MatrixBase
 134 s  22 497x  Eigen::MapBase
 111 s  36 695x  Eigen::DenseBase
  91 s  13 283x  Eigen::Block + BlockImpl + BlockImpl_dense (3 closely-related classes)
  78 s  18 817x  std::is_nothrow_invocable   ← one per BumpAllocator::allocate_run
  74 s  21 516x  std::__and_
  46 s  29 703x  Eigen::internal::evaluator
  42 s   6 331x  Eigen::CwiseBinaryOp (+ Impl variant)
  36 s   5 414x  Eigen::Product (+ 3 close variants, ~100 s combined)
```

Eigen contributes ~800 s of class instantiation spread across its expression template machinery. This is mostly intrinsic to Eigen — limited improvement room without changing the DSL.

---

## Where is memory and symbol duplication going?

### Object file sizes by category

```
 169.3 MB   46.4 %   52 TUs   bindings
 107.7 MB   29.5 %   31 TUs   examples
  32.5 MB    8.9 %   29 TUs   tests
  27.8 MB    7.6 %   19 TUs   core
  27.8 MB    7.6 %   10 TUs   bench
 --------
 365.1 MB total
```

Top 5 largest .o files: one example (zermelo, 19.9 MB) and four bindings/bench TUs (10-14 MB each). All bindings TUs together are **bigger than the rest of the codebase combined**.

### Duplicate symbols — linker burns, compiler already paid

**253 601 weak symbols total** across 142 object files. Each one is a template instantiation that every containing TU processed fully before the linker chose one winner.

Top Tycho-family duplicates across all TUs (demangled):

```
  82   tycho::vf::GenericFunction<-1,-1>::~GenericFunction()
  80   tycho::vf::GFStorage<-1,-1>::~GFStorage()
  77   tycho::vf::DomainHolder<-1>::set_input_domain(...)
  72   non-virtual thunk to GFModel<-1,-1, InterpFunction<-1> >::~GFModel()
  67   tycho::utils::detail::BumpStack<double>::allocate(unsigned long)
  67   tycho::utils::detail::BumpStack<double>::restore(...)
  66   GenericFunction<-1,-1>::operator=(const&)
  63   tycho::vf::Segment_Impl<Arguments<-1>,-1,-1,0>::Segment_Impl(int,int,int)
  60 × 8   non-virtual thunks to GFModel<8,7, RKStepper<Kepler, IVPAlg::{0..15}> >::~GFModel()
  60 × 8   same for GFModel<-1,-1, RKStepper<Kepler, IVPAlg::{0..15}> >::~GFModel()
  60   non-virtual thunk to GFModel<-1,-1, IOScaled<GenericFunction<-1,-1>> >::~GFModel()
  55   tycho::utils::detail::BumpStack<Eigen::Array<double,4,1,…>>::allocate/restore
  54   tycho::utils::FlatMap<std::string, Eigen::Matrix<int,-1,1,…>>::…
  51   tycho::vf::scale_matrix_dynamic_domain_impl<…>
```

Interpretation: 16 `GFModel<…RKStepper<Kepler, IVPAlg::X>>::~GFModel` specialisations (one per integrator algorithm enum value × 2 input dimensions) are each emitted into ~60 TUs. That is 960 TU-level instantiations of destructors the linker will throw away.

### Non-tycho duplicates

```
  114   __clang_call_terminate    ← exception personality boilerplate
   97   std::_Sp_counted_base::~_Sp_counted_base()
   91 × ~30  fmt::v12::detail::write<...> overloads
```

91 out of 142 TUs (64 %) instantiate the full fmt `write<…>` overload set. Linker dedups; compiler front-end pays every time.

---

## Recommendations (ranked by effort vs. payoff)

### High leverage, low effort

1. **`extern template` for `tycho::vf::GFModel<…, RKStepper<Kepler, IVPAlg::X>>` and the 15 other RKStepper specialisations.** Pattern:
   - Declare `extern template class GFModel<…>;` in `integrators.h`.
   - Emit the explicit instantiation in `src/integrators/tycho_integrators.cpp` (currently header-only per CLAUDE.md; would need a new .cpp).
   - Expected: ~ (60 × 16 × destructor-instantiation-cost) removed from non-integrator-implementation TUs. Touches every binding, every example, every bench.

2. **`extern template` for `tycho::utils::BumpAllocator::allocate_run`.** 903 s aggregate, 18 817 instantiations. Many of these are trivial wrappers over the same underlying allocation — if the generic pathway can be reduced to a non-template core plus thin instantiated shims, the win is substantial.

3. **Reduce `DenseFunctionBase::compute_jacobian_adjointgradient_adjointhessian` instantiation cost.** 605 s aggregate across 10 004 instantiations ⇒ 60 ms per instantiation. Inspect whether the three-derivatives body can be split into a non-template helper that takes Eigen refs, leaving only a thin forwarding template. Same technique for `compute_jacobian` (371 s, 9 058x).

### Medium leverage, medium effort

4. **Shrink the examples.** Each example's `main.cpp` builds in 60-180 s because it instantiates `ODEPhase`, `Integrator`, `PSIOPT`, and a problem-specific VectorFunction all in one TU. Split example source into `problem_definition.cpp` + `main.cpp` with an `extern template` boundary for the phase/integrator specialisations. Given examples are 42.4 % of build time, even a 30 % per-example improvement saves ~10 min wall.

5. **Unity build batch size for bindings.** Bindings already use unity; each unit at 26 s average is reasonable, but the top-RSS bindings TUs (`kepler_model`, `kepler_utils` at 5.7-5.9 GB) are at the ragged edge. Check `generic_odes_build_part*` (4.4 GB, ~80 s): current batch size appears fine; the heavy outliers are single files, not batches.

6. **`tycho::oc::ODEPhase::transcribe_dynamics` + `ODEPhase::ODEPhase`.** 158 s for 14 instantiations, 100 s for 10 transcribes — 11 s and 10 s *per instantiation* respectively. These are the most expensive individual instantiation events in the codebase. Candidate for the same header/source split + `extern template` treatment as DenseFunctionBase.

### Low leverage, structural

7. **fmt::v12::detail::write overload set.** 91 TUs instantiate the full set. Two options: a) add `fmt` headers to the PCH only for TUs that actually print (unlikely — fmt is small); b) replace direct fmt use in headers with a thin non-template logging façade, keeping fmt instantiations in one TU. Low effort, low payoff (~few seconds), but reduces `.o` sizes.

8. **Consolidate `std::make_shared<GFModel<...>>` call sites.** 456 s aggregate. Most make_shared calls happen inside `GFStorage::emplace`. If `emplace` is refactored to take a raw pointer from a per-type factory, the shared_ptr machinery only instantiates once per GFModel type globally rather than per TU.

### Not worth pursuing

- **Reducing Eigen instantiation.** ~800 s across the Eigen class tree. Only way to cut this is to move away from the Eigen expression template DSL — not a reasonable trade.
- **Parsing cost optimisation.** 4.3 % of total. PCH is already doing its job; diminishing returns.

---

## Parallelism and memory ceiling

Current guidance in `CLAUDE.md` says `-j8` is safe on 32 GB because the `heavy_compile` ninja pool auto-throttles to 3 slots. The measurements above corroborate this:

- Heavy TUs (≥ 4 GB RSS): 25 of 142 TUs (18 %).
- Average heavy-TU RSS: 5.7 GB.
- 31 GB / 5.7 GB ≈ 5 concurrent heavy TUs is the theoretical ceiling, but worst-case 8 GB × 3 = 24 GB leaves ≤ 7 GB for light TUs and the OS — tight but workable.

**Observed on this machine:** `-j8` OOMs (see `feedback_build_j8_too_much_ram.md`); `-j6` works; `-j2` is the safe choice under memory pressure or with other agents active. The pool depth is already correct; the issue is that many example `main.cpp` TUs are *each* heavy, so the pool still admits up to 3 at once, times ~8 GB, while the full `-j` fan-out adds light TUs concurrently.

**Suggested tuning:** lower the `heavy_compile` threshold (currently 1 slot per 10 GB RAM). Options:

- **1 slot per 12 GB** → depth 2 on 32 GB. Safer default; protects against two 8 GB heavy TUs co-residing.
- **Tag example `main.cpp` TUs as heavy.** Currently the pool only applies to bindings, tests, and benchmarks per the CMake helper. Extending it to examples would cap parallel `main.cpp` builds even when `ninja -j8` is passed.

---

## Baseline saved for future comparison

```
/tmp/tu_timing.tsv                    (per-TU timing, 142 rows)
/tmp/tycho_tu_profile.json            (RSS + wall time for top 25)
/tmp/tycho_traces/                    (14 successful .json ftime-traces — 11 overwritten main.cpp collisions as documented)
/tmp/tycho_trace_analysis.txt         (template instantiation aggregate)
/tmp/obj_category.txt                 (object sizes by category)
/tmp/obj_top25.txt                    (top 25 largest .o files)
/tmp/dup_tycho.txt                    (most-duplicated tycho symbols)
/tmp/dup_symbols_all.txt              (most-duplicated symbols across all TUs)
/tmp/dup_symbols_bindings.txt         (most-duplicated within bindings)
```

Copy to `/tmp/tycho_baseline/` before making any build-perf changes, then re-run the procedure and diff.
