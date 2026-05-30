# Phase D — Builder-API example TU split: empirical non-goal

The plan `kind-prancing-crown.md` Phase D proposed extending the static-API
TU-split pattern (Phase A) to the builder-API examples, with
`parallel_parking_builder` as the pilot. The pilot **did not meet the gate**;
the pattern does not translate from static to builder examples. This document
records the measurement, the mechanism, and the decision.

Phase A (statics) delivered peak RSS reductions of 54–64 % of baseline per
split example, and a 69 % wall-clock reduction at `-j2` across the four
statics. Phase D aimed for a similar result on the builder examples.

## What was measured on `parallel_parking_builder`

Baseline (single `main.cpp`, 278 LOC):

| | wall (s) | peak RSS (GB) |
|---|---:|---:|
| `main.cpp` | 71.5 | 5.71 |

After splitting into `main.cpp` + `parallel_parking_defs.cpp`, with the ODE
factory and all four constraint factories (`make_slot_constraint`,
`make_corner_constraint`, `make_final_y_constraint`, `make_curv_rate_fn`) in
the new `.cpp`:

| | wall (s) | peak RSS (GB) |
|---|---:|---:|
| `main.cpp` | 37.7 | 4.14 |
| `parallel_parking_defs.cpp` | 68.1 | **5.66** |

Plan gate: peak RSS on the heavier new TU ≤ 0.80 × baseline = 4.57 GB.
Result: **5.66 GB > 4.57 GB — fail**.

Additional observations:

- Parallel wall-clock at `-j2` dropped only 71.5 s → 68.1 s (−5 %), compared
  with Phase A's −69 % on the static examples.
- Serial CPU-time *rose* from 71.5 s to 37.7 + 68.1 = 105.8 s (+48 %).
- Numerical parity held: both runs produced `maneuver time = 18.265 s` at
  identical precision.

## Why the pattern doesn't translate

The static-example win came from the fact that the concrete `ODEPhase<CustomODE>`
specialisation lived entirely in `main.cpp`. Moving the concrete ODE into a
factory that returns `GenericFunction<-1,-1>` collapsed the phase onto the
already-compiled `ODEPhase<GenericODE<GenericFunction<-1,-1>,-1,-1,-1>>` path
inside libtycho — so the new `main.cpp` paid for almost nothing, and the
new `*_ode.cpp` emitted only the concrete ODE.

The builder-API examples don't have that lever. `ODEBuilder::build()` already
returned a type-erased `tycho::ODE` from day one, so:

- **main.cpp** never pays for a concrete `ODEPhase<T>` specialisation. What it
  *does* pay for is the phase-registration scaffolding (the concrete-type
  machinery inside `phase.add_boundary_value`, `add_lu_var_bound`,
  `add_inequal_con(GenericFunction<-1,-1>(…))`, `add_delta_time_objective`,
  `solve_optimize`, and so on). That scaffolding instantiates TwoFunctionSum,
  NestedFunction, GFStorage::emplace, make_shared<GFModel<…>>, and roughly 15
  other supporting templates. After the split, `main.cpp` dropped from
  5.71 GB → 4.14 GB — the only thing that moved out was the local
  expression-tree builders.
- **defs.cpp**, on the other hand, has to emit both
  - the ODEBuilder lambda's expression tree (the same one the old `main.cpp`
    used to emit), AND
  - the four constraint expression trees.

  So `defs.cpp` pays for essentially the full expression-tree instantiation set
  that baseline `main.cpp` used to, plus the GenericFunction wrappers to erase
  each one. That's why its RSS (5.66 GB) lands within 1 % of baseline.

Moreover, because the phase-registration scaffolding in `main.cpp` also pulls
in `GFStorage::emplace<…>` / `make_shared<GFModel<…>>` / the FunctionDifference
and TwoFunctionSum machinery (clang-diagnosed during the build), there is real
duplication between `main.cpp` and `defs.cpp`. That duplication is what pushes
serial CPU-time up 48 %.

In short: the builder API has **already** amortised the ODE instantiation at
the library boundary. There is no second boundary inside the example to
amortise. The constraint expression trees are the only heavy thing, and lumping
them together in a single `defs.cpp` reproduces most of the original working
set.

## What sub-splits *could* buy — and why we're not chasing it

A follow-on could fragment `defs.cpp` further: one file per constraint factory
(so `parallel_parking_slot.cpp`, `parallel_parking_corner.cpp`, etc.). The
`corner_constraint` (8 `tri_area` calls over 6 inputs each) is plausibly the
dominant expression. Splitting it off alone might drop each sub-TU below
4.5 GB at the cost of three or four files per example.

Even in the optimistic case, the ceiling is set by `main.cpp`'s 4.14 GB — the
phase-registration scaffolding. No amount of sub-splitting brings the example
below that. And every added TU multiplies the file-system cost per example
without lowering the parallel-wall ceiling.

Quantitatively: a three-way split that evenly distributes the expression trees
would give three TUs at roughly `4.1 + 1.8 + 1.8 = 7.7 GB` aggregate serial
(+8 % vs baseline) with parallel wall at `-j3` bounded by
`max(main=4.1, 1.8, 1.8) = ~main.cpp ≈ 40 s`. Compared with baseline 71.5 s,
that's about −44 % wall at `-j3`, but requires three extra files per example
and a `-j3`-capable host. The static-example split gave the same magnitude of
win at `-j2` with only one extra file.

## Decision

Do not extend Phase D beyond the pilot. Builder-API examples stay monolithic.
The `docs/dev/notes/user_guide_example_tu_split.md` user guide is updated with a candid
"When not to split" entry calling this out so downstream users don't sink time
trying the same pattern in their own builder-API programs.

The `parallel_parking` files have been reverted to their pre-pilot state. No
code changes are merged for Phase D. This artefact preserves the measurement
so the decision doesn't need to be re-derived by a future reader.

## If builder examples ever become the bottleneck again

The full design-space analysis lives in
`docs/dev/notes/build_perf_phase_scaffolding.md`. Summary: the 4.14 GB floor in
main.cpp after the D.1 split decomposes into two template families —
the `GenericFunction<-1,-1>` erasure chain (`make_shared<GFModel<T>>` +
`GFStorage::emplace<T>` + GF ctor; ~1100 s aggregate serial in the perf
report), and the transcription-side `ConstraintModel<NestedFunction<…>>`
et al. that the phase builder emits internally. Three options enumerated
there:

1. **Option 1 — non-template `make_gf_storage_ptr` entry point** (highest
   payoff, low runtime risk, no prerequisites). Move the `shared_ptr`
   count-block construction out of `GFStorage::emplace<T>`'s template
   body into a single non-template factory in a new
   `src/vf/generic_function_storage.cpp`. Recommended next action if the
   builder-example RSS becomes worth chasing.
2. **Option 2 — extern-template the transcription-side types** for the
   erased `GenericODE<GenericFunction<-1,-1>, -1, -1, -1>` phase path.
   Medium payoff; gated on the same `tol=0` golden-policy decision that
   blocks Phase C.
3. **Option 3 — runtime ODE builder**. Rejected: forfeits analytical
   Jacobians for modest build-time win.

A measurement-only spike of Option 1 is the low-risk first step.
