# Phase C (Integrator<Kepler> nested-template extern) — deferred

The plan `kind-prancing-crown.md` Phase C proposed extending the existing
`extern template class Integrator<Kepler>` pattern
(`include/tycho/detail/integrators/extern_templates.h`) to cover the 8
algorithm-specific nested member-template specialisations of
`Integrator<Kepler>::init_stepper_and_controller<IVPAlg::X>`.

## What was tried

1. Added `TYCHO_EXTERN_INIT_STEPPER(IVPAlg::X)` declarations to
   `extern_templates.h` and matching `TYCHO_INSTANTIATE_INIT_STEPPER(IVPAlg::X)`
   definitions to `src/integrators/extern_template_instantiations.cpp`, one
   per `{DOPRI5, DOPRI87, Tsit5Trans, BS3Trans, BS5Trans, Vern7Trans,
   Vern8Trans, Vern9Trans}`.
2. Added `#include <tycho/detail/integrators/extern_templates.h>` to
   `src/bindings/astro/kepler_model.cpp` so the declarations applied there.
3. (Abandoned) Tried to add the same include to
   `tests/cpp/integrators/integrator_test_utils.h`. That shared header reaches
   many test TUs; the blanket suppression pulled too many dependent template
   symbols (`GFModel<...>` vtables, `GenericFunction<-1,-1>::cachedata`) out of
   the implicit-instantiation chain and the `_tychopy` / `tycho_tests` links
   failed with undefined references.

## Why deferred

With the include limited to `kepler_model.cpp` alone, the measured gain was
about **4 s wall / 0.04 GB RSS** on that single TU (98 s → 94 s, 5.71 GB → 5.67
GB) — well below the plan's 40 s / 120-symbol gate.

Worse: moving the `init_stepper_and_controller<X>` bodies out of each
Kepler-using TU and into `libintegrators_instantiations.a` shifted FP
codegen slightly enough that the golden regression tests
`RegressionIVPTest.Case02_TwoBody_DOPRI87` and
`RegressionIVPTest.Case06_Batch_DOPRI87` (both `EXPECT_NEAR(..., 0.0)`
exact-bitwise compares) developed `1e-12`-level drift and started failing.
These tests already show similar rebuild-flaky behaviour (see the Phase A.1
commit note on `Case11/12`), but Phase C made the drift deterministic.

## Requirements for retry

Before landing Phase C, we would need either:

- **Relax the golden regression tolerances** from `tol = 0.0` to something
  like `tol = 1e-10 * max(|actual|, |golden|)` so FP reordering from any
  valid codegen change doesn't fail the tests.  OR
- **Regenerate the golden files** under the Phase-C link topology and accept
  that any future codegen change (optimiser updates, etc.) needs a fresh
  regeneration.

Both approaches affect far more than Phase C; they should be decided as a
test-policy change, not bundled into this plan.

## Plan status

Phase A (example TU split across brachistochrone, hypersens, delta3_launch,
zermelo + user guide) remains in place and is the entirety of the refactor
merged on `build-perf-refactor`. See the phase artefacts in
`bench/build_perf/2026-04-18-phaseA{1,2,3}/` and the report at
`doc/build_performance_report_2026-04-18.md` for the delivered numbers.
