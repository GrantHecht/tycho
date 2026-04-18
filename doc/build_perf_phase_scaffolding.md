# Design Spec: Reducing Per-TU Build Cost of the Phase-Registration Scaffolding

**Status:** Design-only. No implementation planned at the time of writing.
**Driver:** Follow-on to the Phase A / Phase D / Phase E build-performance
work captured in `doc/build_performance_report_2026-04-18.md`,
`doc/user_guide_example_tu_split.md`, and the artefacts under
`bench/build_perf/2026-04-18-*/`.

## Motivation

Phase A (static examples) proved out a TU-split recipe that roughly halved
per-TU peak RSS and unlocked higher build parallelism on RAM-constrained
hosts (details in `bench/build_perf/2026-04-18-phaseA{1,2,3}/`). When we
piloted the same pattern on a builder-API example
(`parallel_parking_builder`, Phase D.0 survey at
`bench/build_perf/2026-04-18-phaseD0-builder-survey/README.md`), the gate
failed:

- `main.cpp` dropped only from 5.71 GB → 4.14 GB. That 4.14 GB is the
  **phase-registration scaffolding floor**: the set of templates pulled in
  by `phase.add_boundary_value`, `add_inequal_con(GenericFunction<-1,-1>(…))`,
  `add_lu_var_bound`, `add_delta_time_objective`, `solve_optimize`, and
  friends. It does not depend on the number of constraint factories the
  example uses; it is imposed by the act of constructing a Phase and
  registering any constraint at all.
- The companion `<name>_defs.cpp` ended up at 5.66 GB — 99 % of baseline —
  because it had to re-emit the full expression-tree set for every
  constraint plus the ODEBuilder lambda, while the main.cpp continued to
  pay the scaffolding floor.

Mechanism: builder examples have **no second erasure boundary inside the
example**. `ODEBuilder::build()` already erased the ODE at the library
boundary, so moving constraint expressions around the example's own source
tree just relocates the same working set.

The scaffolding floor is what prevents Phase D from buying anything. This
spec enumerates the three mechanisms that could move that floor, with
their respective costs and blockers, so a future iteration can pick one
without re-deriving the analysis.

## What's in the 4.14 GB floor

From the Phase D.0 clang diagnostics and the top-30 tables in
`doc/build_performance_report_2026-04-18.md`:

| Contributor | Aggregate serial cost | How it enters main.cpp |
|---|---:|---|
| `std::make_shared<GFModel<-1,-1,T>>` + internal `shared_ptr` / count-block ctors | ~456 s / 1022 TUs | via `phase.add_inequal_con(GenericFunction<-1,-1>(expr))` et al. |
| `tycho::vf::GFStorage<-1,-1>::emplace<T>` | ~338 s / 962 TUs | same call path |
| `tycho::vf::GenericFunction<-1,-1>::GenericFunction<T>(const T&)` | ~302 s / 617 TUs | same call path |
| `tycho::vf::TwoFunctionSum_Impl<…>`, `NestedFunction_Impl<…>`, `FunctionDifference<Segment<-1,-1,0>, Segment<-1,-1,-1>>` | ~228 + 225 + … s / thousands | phase transcription (`set_traj`, `transcribe_dynamics`, `add_link_equal_con`) emits these internally whenever the underlying ODE is `GenericODE<GenericFunction<-1,-1>, -1, -1, -1>`. |
| `tycho::solvers::ConstraintModel<NestedFunction<StackedOutputs<Segment…>>>` and friends | ~297 s aggregate; 448 duplicates × ~660 µs each for the largest specialisation | emitted per phase per constraint; type is fixed for the erased-ODE path, but each TU re-instantiates. |

The first three rows are **per-GenericFunction-construction** cost; the
last two are **per-phase-construction** cost. Both families fire in
main.cpp (phase construction + `add_*` calls) and in any companion
`defs.cpp` (constraint factories calling `GenericFunction<-1,-1>(expr)`).

## Three options

### Option 1 — Move the GenericFunction erasure chain behind a non-template entry point

**Target**: the first three rows above (~1100 s aggregate).

**Mechanism**: today, `GFStorage<-1,-1>::emplace<T>` is a template whose
body invokes `std::make_shared<GFModel<-1,-1,T>>(std::move(obj))`
(`include/tycho/detail/vf/type_erasure/gf_type_erasure.h:363`). The
template body therefore drags the full `shared_ptr` count-block
construction into every TU that wraps a new `T` into a GenericFunction.

Replace the template body with a thin allocator that hands a concrete
`GFModel<T>*` plus a non-template deleter pointer off to a non-template
factory that lives in a new `src/vf/generic_function_storage.cpp`:

```cpp
// include/tycho/detail/vf/type_erasure/gf_type_erasure.h (template, small)
template <GFStorable<IR, OR> T>
void emplace(T obj) {
    using Model = GFModel<IR, OR, std::decay_t<T>>;
    Model *raw = new Model(std::move(obj));                // template-local
    auto deleter = +[](void *p) { delete static_cast<Model *>(p); };
    ptr_ = tycho::vf::detail::make_gf_storage_ptr(raw, deleter); // non-template
}

// src/vf/generic_function_storage.cpp (non-template, compiled once)
namespace tycho::vf::detail {
std::shared_ptr<GFModelBase<-1, -1>>
make_gf_storage_ptr(GFModelBase<-1, -1> *raw, void (*del)(void *)) {
    return {raw, [del](auto *p) { del(p); }};
}
// … plus any other IR/OR pairs actually needed; most places use <-1,-1>.
}
```

The template body shrinks to: one `new`, one function-pointer
initialization, one non-template call. The `shared_ptr` machinery (count
block allocation, deleter storage, refcount manipulation) emits exactly
once per {IR, OR} pair, centrally.

**Expected savings**: 200–300 s aggregate serial (roughly 25 % of the
combined `make_shared` + `emplace` + GenericFunction-ctor budget). Most
of the wins manifest as peak-RSS reductions in TUs that construct many
`GenericFunction<-1,-1>(expr)` at `phase.add_*` call sites — i.e., the
builder examples this spec targets, plus every regression-test and
bindings TU that erases concrete expressions. Main.cpp's floor drops by
an unmeasured amount; empirical spike needed to quantify.

**Runtime risk**: **low, confined to construction time.** The extra
indirection is:

1. One function-pointer call through the deleter at destruction (heap
   dealloc path, not hot-loop).
2. One extra pointer hop in `make_gf_storage_ptr`'s body (amortised per
   construction, not per evaluation).

VectorFunction evaluation (`compute_impl`, `compute_jacobian_impl`,
`compute_jacobian_adjointgradient_adjointhessian_impl`) is unchanged —
those methods go through the already-materialised `GFModel::ptr_` with no
additional indirection. The bench suite's short-duration micros would not
be expected to regress on construction-time changes.

**Runtime verification plan**: the existing `bench/bench_track.sh compare`
suite would catch any regression ≥ 1 %, which is well above the expected
construction-cost delta. No changes to `tol=0` regression tests.

**Prerequisites**: None. Self-contained refactor.

### Option 2 — Extern-template the transcription-side ConstraintModel specialisations

**Target**: the fourth and fifth rows above (~250–400 s aggregate across
phase-side constraint/defect machinery).

**Mechanism**: when a Phase's underlying ODE is
`GenericODE<GenericFunction<-1,-1>, -1, -1, -1>` (the type
`ODEBuilder::build()` always produces — see
`include/tycho/detail/optimal_control/builder/runtime_ode.h:39`), the
transcription code emits a fixed set of internal expression types:
`NestedFunction<StackedOutputs<Segment<-1,-1,0>, Segment<-1,-1,-1>>>`,
`FunctionDifference<Segment<-1,-1,0>, Segment<-1,-1,-1>>`,
`LGLDefects<GenericODE<GenericFunction<-1,-1>,-1,-1,-1>>`, and the
`ConstraintModel<…>` wrappers around each. Because the outer ODE type is
fixed (all erased phases share it), the set of inner types is also fixed
and enumerable. That makes them valid `extern template` candidates.

The pattern is the existing one in
`include/tycho/detail/integrators/extern_templates.h` and
`src/integrators/extern_template_instantiations.cpp`, expanded to the
transcription family:

```cpp
// include/tycho/detail/optimal_control/extern_templates.h (new)
extern template class tycho::vf::NestedFunction_Impl<…>;
extern template class tycho::vf::TwoFunctionSum_Impl<…>;
extern template class tycho::solvers::ConstraintModel<…>;
// … per enumerated specialisation, IR/OR=-1/-1, erased ODE type

// src/optimal_control/extern_template_instantiations.cpp (new)
template class tycho::vf::NestedFunction_Impl<…>;
// etc.
```

Include the extern header from
`include/tycho/detail/optimal_control/builder/phase_wrapper.h` so every
builder-API consumer sees the declarations transitively. Compile the
explicit instantiations into a new `optimal_control_instantiations`
static library and link from libtycho / _tychopy.

**Expected savings**: on the order of 100–200 s aggregate serial across
bindings + examples, plus a floor reduction in main.cpp for any
builder-API program. Exact numbers need a spike.

**Runtime risk**: **medium.** These types evaluate in the solver inner
loop (transcription defect computation, constraint Jacobian assembly).
Moving their bodies from per-TU implicit instantiations to a central
explicit-instantiation TU changes codegen visibility, which Phase C
demonstrated can shift FP rounding at the 1e-12 level
(`bench/build_perf/2026-04-18-phaseC-deferred/README.md`).

**Blocker**: the `tol=0` golden regression compares in
`tests/cpp/integrators/regression/test_regression_{ivp,transcription}.cpp`
(`RegressionIVPTest`, `RegressionTranscriptionTest`). The IVP suite
doesn't use `Phase::add_*` so may be unaffected by Option 2 — but the
transcription suite (Case08–10) does exercise the phase construction
path.

**Prerequisite**: the same test-policy decision that blocks Phase C and
most of Phase E — either relax the golden tolerances to
`1e-10 · max(|actual|, |golden|)`, or accept golden regeneration as
part of landing this option.

**Synergy**: Options 1 and 2 are orthogonal. If the golden-policy decision
lands, both can be applied; the net floor reduction should be larger than
either alone.

### Option 3 — Runtime ODE builder

**Not recommended, documented for completeness.**

An API that accepts a plain `std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>`
would bypass the VectorFunction DSL entirely and eliminate the
per-example lambda instantiation cost. It would, however, forfeit
analytical Jacobians — the DSL's central runtime-perf feature — in
exchange for a modest build-time win. The tradeoff is backwards for
tycho's optimisation-focused use case. Mentioned only so future readers
can rule it out quickly.

## Recommendation

**Spike Option 1 first.** It has the largest projected payoff of the
three, the lowest runtime risk, and no prerequisites. The spike should:

1. On a scratch branch, implement the non-template `make_gf_storage_ptr`
   entry point exactly as sketched above.
2. Rebuild the four Phase A statics and the `parallel_parking_builder`
   pilot; measure peak RSS and wall-clock per TU via
   `scripts/measure_tu_profile.py`.
3. Compare against `bench/build_perf/2026-04-18-baseline/`:
   - Phase A examples' `main.cpp` + `*_ode.cpp` — expect each to drop
     some.
   - `parallel_parking_builder`'s `main.cpp` — expect the 4.14 GB floor
     to drop below 3.5 GB. If so, retry the Phase D pilot with Option 1
     in place and see whether the 0.80× RSS gate passes.
4. Run `bench/bench_track.sh compare` to confirm runtime neutrality.
5. Write results into `bench/build_perf/<date>-option1-spike/README.md`,
   alongside this spec.

If the spike delivers a meaningful main.cpp-floor reduction, land Option 1
as a proper refactor, then consider reviving Phase D's tier rollout. If
the spike disappoints, this doc stays as-is — no code changes — and the
floor is a genuine property of the DSL, not an incidental cost.

**Defer Option 2 until the golden-test tolerance policy is decided.** It
is a follow-on to Option 1, not a substitute.

**Rule out Option 3.**

## Cross-references

- `doc/build_performance_report_2026-04-18.md` — full baseline analysis,
  top-30 tables, duplicate-symbol counts. The numbers quoted in this spec
  come from there.
- `doc/user_guide_example_tu_split.md` — user-facing recipe for the Phase
  A pattern, with the "When NOT to split" entry describing why the
  unmodified builder-API path doesn't benefit from a companion `.cpp`.
- `bench/build_perf/2026-04-18-baseline/` — baseline artefacts.
- `bench/build_perf/2026-04-18-phaseA{1,2,3}/` — per-phase measurements
  that define "what a Phase D-class win looks like".
- `bench/build_perf/2026-04-18-phaseC-deferred/README.md` — the FP-drift
  failure mode that Option 2 would need to clear first.
- `bench/build_perf/2026-04-18-phaseD0-builder-survey/README.md` — the
  empirical measurement that motivated this spec.
- `bench/build_perf/2026-04-18-phaseE-survey/README.md` — test / bench
  RSS survey; overlaps with Option 2's blocker analysis.
- `include/tycho/detail/vf/type_erasure/gf_type_erasure.h:363` —
  `GFStorage::emplace<T>` body targeted by Option 1.
- `include/tycho/detail/integrators/extern_templates.h` /
  `src/integrators/extern_template_instantiations.cpp` — the existing
  extern-template + instantiation library pattern Option 2 would extend.
