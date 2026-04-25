# PR #44 — Follow-up Cleanup Items

## Context

PR #44 (`ivp-solver-modernization` → `main`,
[github.com/GrantHecht/tycho/pull/44](https://github.com/GrantHecht/tycho/pull/44))
delivered the IVP-stack modernization (SP1–SP3) plus a build-perf refactor
and the step-sizing API overhaul. The branch went through a five-agent
review sweep (`code-reviewer`, `silent-failure-hunter`, `comment-analyzer`,
`pr-test-analyzer`, `type-design-analyzer`) on 2026-04-25 followed by
multiple in-branch hardening rounds. This document captures the residual
items that the review identified as **deferred to a follow-up PR**:
correctness hardening that doesn't block merge, encapsulation passes that
are larger than a hardening commit, diagnostic improvements, doc-rot
cleanups, and a couple of test-coverage gaps.

It also reserves space (§6) for **STM / higher-order-derivative correctness
work** that the user has flagged but not yet specified — that section
should be filled in via a follow-up commit after the design discussion. The
boundary is reserved here so the cleanup doc doesn't fragment across
multiple files.

### Conventions used in this doc

- **Code anchors are symbol-name only**, not line numbers. Every reference
  uses `File::Symbol` or function/struct names. This is itself §4.1 of the
  cleanup — the segmented-STM plan doc has already drifted on its line
  anchors, so we don't repeat the mistake here.
- **Each item is self-contained**: Problem → Current code → Proposed fix →
  Acceptance. A follow-up agent can land any one item without reading
  any other item in this doc.
- **Items are independent**: no item gates another. Each can land as its
  own commit (or its own micro-PR) in any order.

### Pointers

- The PR itself: `gh pr view 44`
- The deferred segmented-STM perf plan:
  `doc/plans/2026-04-24-segmented-stm-main-thread-optimization.md`
- The build-perf TU-split user guide:
  `doc/user_guide_example_tu_split.md`
- Build-perf phase notes:
  `bench/build_perf/2026-04-18-baseline/`,
  `bench/build_perf/2026-04-18-phase{A1,A2,A3,C-deferred,D0-builder-survey,E-survey}/`
- The `integrate_stm2` segfault fix that closes a precondition for §6:
  commit `e121ed2`

---

## Section 1 — Correctness Hardening (Important)

### 1.1 — Unguarded `controller_spec_` optional dereference in `copy_settings_from`

**Symbol:** `Integrator::copy_settings_from` in
`include/tycho/detail/integrators/integrator.h` (controller-spec branch).

**Problem.** The optional dereference is gated on `target_uses_controller`,
which is itself derived from `this->use_controller_`. Both
`use_controller_` and `controller_spec_` are mutable members on the same
object; the invariant linking them is upheld only by `set_method`.
A future code path that flips `use_controller_` without rebuilding
`controller_spec_` produces undefined behavior on this `->first` /
`->second` deref instead of a defined `bad_optional_access`.

**Current code.**
```cpp
template <class OtherDODE> void copy_settings_from(const Integrator<OtherDODE> &src) {
    const bool target_uses_controller = this->use_controller_;
    // ...
    ControllerType target_controller_source;
    Eigen::VectorXi target_controller_varlocs;
    if (target_uses_controller) {
        target_controller_source = this->controller_spec_->first;   // unguarded
        target_controller_varlocs = this->controller_spec_->second; // unguarded
    }
    // ...
}
```

**Proposed fix.** Replace `->first/->second` with `.value().first/.second`,
or assert `controller_spec_.has_value()` once at the top of the branch.
Either gives a defined, debuggable failure mode if the invariant is ever
broken in the future. Per `feedback_no_disable_toggles_for_safety.md`,
this is a cheap unconditional safety check that should ship as-is — no
gate needed.

```cpp
if (target_uses_controller) {
    const auto &spec = this->controller_spec_.value();
    target_controller_source = spec.first;
    target_controller_varlocs = spec.second;
}
```

**Acceptance.** Existing tests
(`tests/cpp/integrators/test_copy_settings.cpp` and the
`copy_settings_from no-controller copy` case in `f36ae5f`) continue to
pass. No new test required — the fix removes a UB path; the existing
"with-controller copy" test exercises the new code path.

---

### 1.2 — `int` ↔ `int64_t` width drift in event-failure counters

**Symbols:**
- `Integrator::EventCounterWriteback::nfailed` (int reference) and the
  store into `Integrator::n_failed_event_refinements_` (int64_t member),
  in `include/tycho/detail/integrators/integrator.h`.
- `Integrator::find_events_counted` out-param
  `n_failed_event_refinements`, same file.

**Problem.** Commit `8afe63c` widened the **member** counters
(`naccept_`, `nreject_`, `n_failed_event_refinements_`) to `int64_t`, but
the per-call writeback reference and `find_events_counted`'s out-param
both remain `int`. The writeback narrows on store. Per-call locals are
bounded by `max_steps_` (int) so single-call overflow is impossible —
but the asymmetry is confusing and the `BatchEventCounterWriteback` *does*
sum correctly in `int64_t`, which makes the inconsistency stand out.

**Current code.**
```cpp
struct EventCounterWriteback {
    const Integrator &integ;
    int &nfailed;                                    // ← int ref
    ~EventCounterWriteback() noexcept {
        if (nfailed > 0) {
            std::fprintf(stderr,
                         "tycho: %d event-refinement failure(s); ...\n",
                         nfailed);
        }
        integ.n_failed_event_refinements_ = nfailed; // ← int → int64_t
    }
};
```
```cpp
EventLocsType find_events_counted(std::shared_ptr<LGLInterpTable> tab,
                                  const std::vector<EventPack> &events,
                                  const std::vector<std::vector<Eigen::Vector2d>> &eventtimes,
                                  int &n_failed_event_refinements) const {
    validate_events(events);
    n_failed_event_refinements = 0;
    return EventHandler::refine_events<ODEState<double>>(
        tab, events, eventtimes, this->ode_.input_rows(), this->max_event_iters_,
        this->event_tol_, n_failed_event_refinements);
}
```

**Proposed fix.** Pick **one** of the two:

1. **Widen both to `int64_t`** to match the member type. This is the
   cleanest fix and aligns with the principled-fix preference
   (`feedback_full_fixes_over_minimal.md`). Update `EventHandler::refine_events`
   signature accordingly. Update the `fprintf` format string from `%d` to
   `%lld` (with explicit `static_cast<long long>` like
   `BatchEventCounterWriteback` already does).

2. **Document the narrowing as safe-by-construction**: add a comment that
   single-call counts are bounded by `max_steps_` (int) by construction,
   so the writeback narrows safely. Less invasive but leaves the type
   asymmetry in place.

Recommend option 1.

**Acceptance.** Existing `tests/cpp/integrators/test_event_refinement_coverage.cpp`
and `test_max_steps.cpp` continue to pass. The widening is type-only; no
behavior change at counts < INT_MAX. No new test needed unless we want a
contrived high-count test (deemed not worth the build cost).

---

### 1.3 — `integrate_stm_parallel` threadpool branch silently zeros member counters

**Symbol:** `Integrator::integrate_stm_parallel` (threadpool branch and
non-threadpool fallback) in
`include/tycho/detail/integrators/integrator.h`.

**Problem.** The non-threadpool fallback uses a `CounterWriteback` RAII
to publish `total_na`/`total_nr` into `naccept_`/`nreject_`. The
threadpool branch keeps `main_na`/`main_nr` strictly local (no member
writes) to avoid cross-thread races. **Result:** after
`integrate_stm_parallel`, `get_naccept()` returns the segment-summed
counts on single-threaded systems and **stale or zero** on
threadpool-enabled systems. The behavior splits on a runtime backend
choice.

**Current code (threadpool branch).**
```cpp
if (n_parts > 1 && tycho::utils::use_thread_pool()) {
    // Main-thread propagation runs concurrently with workers;
    // keep its state strictly local (no member writes).
    ControllerVariant main_ctrl;
    int main_na = 0, main_nr = 0;
    int submitted = 0;
    try {
        for (int i = 0; i < n_parts; i++) {
            results[i] =
                tycho::utils::thread_pool().submit_task([&stm_op, i] { return stm_op(i); });
            submitted = i + 1;
            if (i < (n_parts - 1)) {
                main_ctrl = this->make_worker_controller();
                main_na = 0;  // ← reset per segment, never published
                main_nr = 0;
                auto stm_result =
                    this->integrate_stm_core(xs[i], ts[i + 1], main_ctrl, main_na, main_nr);
                xs[i + 1] = std::get<0>(stm_result);
            }
        }
    } catch (...) { /* aggregate worker failures */ }
    // worker result aggregation — also doesn't publish counters
}
```

**Current code (fallback).**
```cpp
} else {
    // Non-threadpool fallback is single-threaded, so it can safely
    // accumulate counters into the members (no cross-thread writes).
    int total_na = 0, total_nr = 0;
    CounterWriteback _writeback{*this, total_na, total_nr};   // ← publishes
    for (int i = 0; i < n_parts; i++) {
        ControllerVariant seg_ctrl = this->make_worker_controller();
        int seg_na = 0, seg_nr = 0;
        auto [xf, jx] = this->integrate_stm_core(xs[i], ts[i + 1], seg_ctrl, seg_na, seg_nr);
        total_na += seg_na;
        total_nr += seg_nr;
        // ...
    }
}
```

**Proposed fix.** Two viable paths:

1. **Race-free post-loop aggregation.** Have the threadpool worker task
   `stm_op(i)` return its segment's `(na, nr)` alongside the STM (extend
   `STMRet`). After the post-loop `.get()` aggregation, sum the
   per-segment counts plus the main-thread `main_na/main_nr` deltas, then
   publish to members under a `CounterWriteback` like the fallback. No
   cross-thread member writes during the integrate.

2. **Sentinel-flag the unmeasured state.** Set `naccept_ = nreject_ = -1`
   on threadpool entry, with `get_naccept()` documented to return -1 when
   the most recent integrate ran on the threadpool path. Cheaper but
   transfers the surprise to the consumer rather than fixing it.

Recommend option 1 — it makes the behavior uniform across backends,
which is the semantic the user model expects.

**Acceptance.** New test:
`tests/cpp/integrators/test_integrator_parallel.cpp::IntegrateStmParallel_PublishesCounters`
that runs `integrate_stm_parallel` on a small CR3BP segmented case with
the threadpool both enabled and disabled, asserts `get_naccept() > 0`
and `get_naccept()` matches the single-thread baseline. Pin both
`naccept_` and `nreject_`.

---

### 1.4 — Case 11/12 single-step Jacobian regression: switch to relative tolerance

**Symbol:**
`RegressionTranscriptionTest::Case11_SingleStepJacobian` and
`RegressionTranscriptionTest::Case12_SingleStepJacobianHessian` in
`tests/cpp/integrators/regression/test_regression_transcription.cpp`,
plus `kTranscriptionAbsTol`.

**Problem.** Both tests use `kTranscriptionAbsTol = 1e-13` against
goldens generated on a different toolchain / FP mode. Single-step
non-adaptive `integrate_stm`/`integrate_stm2` paths produce values that
agree with their goldens to **a few ULPs** — but for matrix entries with
magnitude > ~10 (Hessian off-diagonals on Kepler near-circular LEO), a
few ULPs at scale ~10^4 exceeds 1e-13 absolute. The bit-exact `tol=0`
cases (8/9/10) and the relative-tolerance helpers
(`expect_matrix_match_rel`, `expect_vector_match_rel`) already exist in
the same TU. The PR description claims `tol=0` for these cases — that's
a misread; the actual failure is the absolute-tolerance pinning.

The underlying behavior **is** covered by
`tests/cpp/integrators/test_stm_all_methods.cpp` (analytical-STM pin
across every method) and
`tests/cpp/integrators/test_segmented_stm_serial.cpp` (segmented-STM
endpoint contract). Case 11/12 are redundant policy pins, not
load-bearing correctness gates.

**Current code.**
```cpp
static constexpr double kTranscriptionAbsTol = 1e-13;

TEST_F(RegressionTranscriptionTest, Case11_SingleStepJacobian) {
    // ...
    expect_matrix_match(jx, golden_jx, kTranscriptionAbsTol, "Case11_Jacobian");
}

TEST_F(RegressionTranscriptionTest, Case12_SingleStepJacobianHessian) {
    // ...
    expect_matrix_match(jx, golden_jx, kTranscriptionAbsTol, "Case12_Jacobian");
    expect_matrix_match(hx, golden_hx, kTranscriptionAbsTol, "Case12_Hessian");
}
```

**Proposed fix.** Switch to `expect_matrix_match_rel` /
`expect_vector_match_rel` (already defined in the same file) at
`tol=1e-12`:

```cpp
static constexpr double kTranscriptionRelTol = 1e-12;

TEST_F(RegressionTranscriptionTest, Case11_SingleStepJacobian) {
    // ...
    expect_vector_match_rel(xf_dyn, golden_xf, kTranscriptionRelTol, "Case11_xf");
    expect_matrix_match_rel(jx, golden_jx, kTranscriptionRelTol, "Case11_Jacobian");
}
```

Or — alternative — regenerate the goldens on the merging machine using
`tests/cpp/integrators/regression/generate_golden.cpp`, in which case
`kTranscriptionAbsTol = 1e-13` is preserved but the golden becomes
host-FP-coupled (a documented coupling, not silent).

Recommend the relative-tolerance switch — it's host-independent and
matches what `expect_matrix_match_rel`'s docstring already says about
spanning-orders-of-magnitude single-step Jacobians.

**Acceptance.** `ctest --output-on-failure -R Regression` passes 790/790
(currently 788/790).

---

## Section 2 — Encapsulation & Type Design

### 2.1 — `EventPack::vf` is still public; `direction_` should be `enum class`

**Symbols:** `EventPack::vf`, `EventPack::direction_` in
`include/tycho/detail/integrators/event_handler.h`.

**Problem.** Commit `bb830e9` made `direction_` and `stop_count_`
private with validating setters — closing the post-construction-mutation
hole for those two fields. But `vf` is still a public field. `vf` is
the **most invariant-rich** field of the three: it must be a
`GenericFunction<-1, 1>` whose input rows match the integrator's ODE
state, with exactly one output. The type-system constraint is partial
(`<-1, 1>`); the input-rows constraint is enforced nowhere on the
EventPack itself.

Separately: `direction_` is an `int` constrained at runtime to
`{-1, 0, +1}`. An `enum class Direction { Falling=-1, Any=0, Rising=1 }`
moves the constraint into the type system, eliminating
`set_direction`'s runtime range check entirely.

**Current code.**
```cpp
class EventPack {
  public:
    GenericFunction<-1, 1> vf;   // ← public, no validation

    EventPack() = default;
    EventPack(GenericFunction<-1, 1> vf_, int direction_arg, int stop_count_arg)
        : vf(std::move(vf_)) {
        this->set_direction(direction_arg);
        this->set_stop_count(stop_count_arg);
    }
    int direction() const { return direction_; }
    int stop_count() const { return stop_count_; }
    void set_direction(int d) {
        if (!(d == -1 || d == 0 || d == 1)) { /* throw */ }
        direction_ = d;
    }
    // ...
  private:
    int direction_ = 0;          // ← runtime-constrained int
    int stop_count_ = 0;
};
```

**Proposed fix.**

1. **Make `vf` private** with construction-only access (constructor-only
   immutability is best; if mutability is required, add a `set_vf()` that
   asserts `output_rows() == 1` and matches the integrator's input rows).

2. **Convert `direction_` to `enum class Direction`** and update
   `event_direction::Falling/Any/Rising` consumers in the bindings and in
   `EventHandler::check_crossings`.

```cpp
enum class Direction : int { Falling = -1, Any = 0, Rising = 1 };

class EventPack {
  public:
    EventPack(GenericFunction<-1, 1> vf, Direction direction, int stop_count);
    const GenericFunction<-1, 1> &vf() const { return vf_; }
    Direction direction() const { return direction_; }
    int stop_count() const { return stop_count_; }
    void set_stop_count(int s);     // still validates s >= 0
  private:
    GenericFunction<-1, 1> vf_;
    Direction direction_ = Direction::Any;
    int stop_count_ = 0;
};
```

**Acceptance.** `tests/cpp/integrators/test_event_directions.cpp` (in
particular the
`PostConstructionMutationCaughtAtIntegrate` /
`PostConstructionMutationRejectedBySetter` cases) pass after migrating to
the `Direction` enum. Python bindings in
`src/bindings/optimal_control/tycho_optimal_control.cpp` continue to
expose the same constructor signature (the enum is convertible to int at
the binding boundary).

---

### 2.2 — `EventPack::validate()` is now an empty no-op — delete it

**Symbols:** `EventPack::validate()` and the free function
`validate_events()`, both in
`include/tycho/detail/integrators/event_handler.h`.

**Problem.** After the setter encapsulation in `bb830e9`, `validate()`
is structurally a no-op. The retained docstring rationalizes it as
"defense-in-depth at the integrate-entry boundary" — but the body is
literally `{}`, so there is **zero defense**. The free function
`validate_events()` walks the vector and calls the no-op on each entry,
plus a `try/catch` that can never trigger.

This is documentation theater. Per
`feedback_full_fixes_over_minimal.md`, the principled fix removes the
no-op rather than retaining a misleadingly-named method. A future
maintainer could put non-trivial checks back into `validate()` thinking
callers will invoke it on every integrate, not realizing the call sites
have been treating it as a no-op for an unknown number of releases.

**Current code.**
```cpp
class EventPack {
  // ...
    /// No-op now that the only mutation paths validate at the setter.
    /// Kept so existing integrate-entry `validate_events` call sites
    /// retain their defense-in-depth wrapper signature without churn.
    void validate() const {}
  // ...
};

inline void validate_events(const std::vector<EventPack> &events) {
    for (std::size_t i = 0; i < events.size(); ++i) {
        try {
            events[i].validate();   // calls the no-op
        } catch (const std::exception &e) {
            throw std::invalid_argument("events[" + std::to_string(i) + "]: " + e.what());
        }
    }
}
```

**Proposed fix.** Delete both `EventPack::validate()` and the free
function `validate_events()`. Update every call site (search for
`validate_events(`) to remove the call. This is structurally safe
because the EventPack ctor + setters are the only mutation paths.

Alternative: rename `validate()` to a name that reflects "no-op kept for
ABI" — but per the user's stated preferences, deletion is preferred.

**Acceptance.** `ctest` passes; no behavior change. The PR diff is
deletions only.

---

### 2.3 — Controller fields are fully public; split tunables (`Params`) from per-call state

**Symbols:** `IController`, `PIController`, `PIDController` in
`include/tycho/detail/integrators/step_controller.h`.

**Problem.** All three controllers are `struct`s with **every field
public**: tunables (`gamma_`, `qmin_`, `qmax_`, `qmax_first_step_`,
`qsteady_min_`, `qsteady_max_`, `beta1_`, `beta2_`, `beta3_`,
`accept_safety_`, `qoldinit_`) and per-call state (`qold_`, `errold_`,
`q11_`, `err_[]`) sit in the same access regime. Validation is post-hoc
(`validate()` method called explicitly by the driver entry, not by any
mutation). This means a caller can mutate `gamma_` to a nonsense value
and only learn about it on the next `integrate()` call.

The user already noted (`feedback_break_api_freely.md`) that there are
no users yet, so an API break here is fine if the new shape is
meaningfully better. The structural answer is to **split tunables from
state**:

- `IController::Params { gamma, qmin, qmax, qmax_first_step, qsteady_min, qsteady_max }`
  — invariant-by-construction; ctor validates.
- `IController { Params params; double qold; }` — `params` is `const`,
  `qold` is the only mutable per-call state.

The same split applies to `PIController` and `PIDController`.

**Current code (representative — `IController`).**
```cpp
struct IController {
    double gamma_ = 0.9;
    double qmin_ = 1.0 / 5.0;
    double qmax_ = 10.0;
    double qmax_first_step_ = 10000.0;
    double qsteady_min_ = 1.0;
    double qsteady_max_ = 1.0;

    // Internal state
    double qold_ = 1.0;

    void validate() const { /* runtime checks */ }
    ControllerOutput update(double h, double err_norm, int order, int naccept) { /* ... */ }
    void reset() { qold_ = 1.0; }
};
```

**Proposed fix.**
```cpp
struct IController {
    struct Params {
        double gamma            = 0.9;
        double qmin             = 1.0 / 5.0;
        double qmax             = 10.0;
        double qmax_first_step  = 10000.0;
        double qsteady_min      = 1.0;
        double qsteady_max      = 1.0;

        Params() = default;
        Params(double gamma_, double qmin_, double qmax_,
               double qmax_first_step_, double qsteady_min_, double qsteady_max_);
        // ctor validates; no separate validate() method needed
    };

    explicit IController(Params p = {}) : params(std::move(p)) {}
    ControllerOutput update(double h, double err_norm, int order, int naccept);
    void reset() { qold_ = 1.0; }

    const Params &get_params() const { return params; }
    // No setter on params: replace the controller, don't mutate fields.

  private:
    Params params;       // const-after-construction
    double qold_ = 1.0;  // per-call state
};
```

**Migration.** Update `Integrator::set_controller`, every
`make_worker_controller()` callsite, `reset_controller()`, the variant
visitor in `step_controller.h`, the Python bindings, and tests that
poke fields directly (`test_integrator_parallel.cpp` lines flagged in
the review touch `gamma_`, `qmin_` directly — they migrate to
`IController(Params{...})`).

**Acceptance.**

- `tests/cpp/integrators/test_step_controller.cpp`,
  `test_controller_pi.cpp`, `test_controller_pid.cpp` migrate to the new
  `Params` ctor and pass.
- Validation moves from runtime `validate()` (called by driver entry) to
  ctor-time. The existing
  `tests/cpp/integrators/test_step_controller.cpp` validation-rejection
  cases still throw `std::invalid_argument` — same exception type, same
  message format.
- `tychopy/test/test_OptimalControl/test_Controllers.py` migrates to the
  new constructor binding.

This is the largest item in the doc — likely its own PR rather than a
single commit. Worth landing as a stand-alone "Controller Params split".

---

### 2.4 — Other Integrator public-mutable config fields — survey + decide

**Symbols, in `include/tycho/detail/integrators/integrator.h`:**
- `Integrator::adaptive_` (bool)
- `Integrator::abs_tols_` (Eigen::VectorXd)
- `Integrator::rel_tols_` (Eigen::VectorXd)
- `Integrator::max_steps_` (int)
- `Integrator::vectorize_batch_calls_` (bool)
- `Integrator::enable_vectorization_` (bool, if present)
- `Integrator::event_tol_` (double)
- `Integrator::max_event_iters_` (int)
- `Integrator::max_step_change_` (double)
- `Integrator::def_step_size_` (double)

**Problem.** All of the above are `public:` mutable members on
`Integrator`. Some have setters (`set_abs_tol`, `set_rel_tol`,
`set_max_steps`); some don't. Tests routinely write directly
(`integ.adaptive_ = false`, `integ.vectorize_batch_calls_ = false` —
both observed in the regression corpus). The setter-validation pass that
landed for `EventPack` (commit `bb830e9`) and is proposed for the
controllers (§2.3) hasn't been extended to these fields.

**Survey decisions needed.** For each field, decide:

- **Keep public** — fine for fields with no nontrivial invariant
  (e.g., `adaptive_` is a binary flag; no validation possible).
- **Make private with validating setter** — for fields with bounds or
  cross-field invariants:
  - `abs_tols_` / `rel_tols_`: must be elementwise > 0; size must match
    `ode_.output_rows()`.
  - `max_steps_`: must be > 0.
  - `event_tol_`: must be > 0.
  - `max_event_iters_`: must be > 0.
  - `max_step_change_`: must be > 1.
  - `def_step_size_`: documented invariants — check.
- **Make private with `set_method` as the only mutator** — for fields
  that participate in a cross-field invariant (`use_controller_` /
  `controller_spec_`).

The decision is structural; this item is a **survey + design**, not a
pre-decided patch. Recommended deliverable: a small follow-up PR with a
table in the PR description listing each field and the decision, plus
the implementation.

**Proposed fix sketch.**
```cpp
class Integrator {
  public:
    void set_abs_tol(double tol);                // already exists
    void set_abs_tol(const Eigen::VectorXd &t);  // already exists
    void set_rel_tol(double tol);                // already exists
    void set_rel_tol(const Eigen::VectorXd &t);  // already exists
    void set_max_steps(int n);                   // already exists, makes private
    void set_event_tol(double t);                // new
    void set_max_event_iters(int n);             // new
    void set_max_step_change(double f);          // new
    void set_adaptive(bool a) { adaptive_ = a; } // trivial
    void set_vectorize_batch_calls(bool v);      // trivial
    bool adaptive() const { return adaptive_; }
    // ... getters ...
  private:
    bool adaptive_ = true;
    Eigen::VectorXd abs_tols_;
    Eigen::VectorXd rel_tols_;
    int max_steps_ = 1'000'000;
    bool vectorize_batch_calls_ = true;
    double event_tol_ = 1.0e-6;
    int max_event_iters_ = 10;
    double max_step_change_ = 10.0;
    double def_step_size_ = 0.1;
};
```

**Acceptance.** All tests that wrote fields directly (e.g.,
`test_regression_transcription.cpp` Case11/12, `test_parallel_stm_errors.cpp`
flagged in the review) migrate to setters. `ctest` passes. Python
bindings continue to expose only the setter form (already snake_case
per CLAUDE.md naming convention).

---

### 2.5 — Driver `IO` aggregate "caller must zero counters" precondition is prose-only

**Symbols:** `AdaptiveDriver::IO`, `ParallelDriver::IO`, `STMDriver`
(static methods), in
`include/tycho/detail/integrators/{adaptive_driver,parallel_driver,stm_driver}.h`.

**Problem.** Phase 8c (`c875614`) introduced an `IO` aggregate to cut
the driver `integrate()` parameter bloat. The aggregate bundles output
references (counters, eventtimes, eventstates, etc.) that the caller
holds. The contract that **counters must be zeroed before passing the
IO** is documented in prose comments at each driver entry but not
expressed in the type. A caller who reuses an `IO` across calls — the
exact pattern that motivates the aggregate — silently accumulates
counter values from prior calls.

**Proposed fix.** Add a `make_io(...)` factory per driver that
constructs the `IO` aggregate from output buffers, zeros all counters,
and asserts/sizes the output vectors as a side effect. The precondition
moves from "must be true when you call integrate()" to "is true by
construction of the IO".

```cpp
struct AdaptiveDriver {
    static IO make_io(/* output references */) {
        IO io = {/* aggregate init */};
        io.naccept = 0;
        io.nreject = 0;
        io.n_failed_event_refinements = 0;
        // resize/clear output vectors
        return io;
    }
    static void integrate(/* state inputs */, IO &io);
};
```

**Acceptance.** Existing tests pass with no behavior change. This is a
structural improvement, not a bug fix. New test
`tests/cpp/integrators/test_adaptive_driver_compile.cpp` (or similar)
gains a `make_io_zeros_counters` case.

---

## Section 3 — Diagnostics & Observability

### 3.1 — Counter writebacks use raw `fprintf(stderr)`

**Symbols:** `Integrator::EventCounterWriteback::~EventCounterWriteback`
and `Integrator::BatchEventCounterWriteback::~BatchEventCounterWriteback`
in `include/tycho/detail/integrators/integrator.h`.

**Problem.** Both writebacks emit refinement-failure summaries via raw
`std::fprintf(stderr, ...)`. A user redirecting or closing stderr (e.g.,
embedded Python in a headless service, GUI app) loses the diagnostic
silently. The destructor's `noexcept` constraint forecloses anything
that throws — but the project's `logForDebugging` / `logError` /
`logEvent` helpers exist for exactly this case, and a `noexcept` logger
would satisfy the constraint.

**Current code.**
```cpp
struct EventCounterWriteback {
    const Integrator &integ;
    int &nfailed;
    ~EventCounterWriteback() noexcept {
        if (nfailed > 0) {
            std::fprintf(stderr,
                         "tycho: %d event-refinement failure(s); see "
                         "Integrator::get_failed_event_count() for diagnostics\n",
                         nfailed);
        }
        integ.n_failed_event_refinements_ = nfailed;
    }
};

struct BatchEventCounterWriteback {
    const Integrator &integ;
    const std::vector<int> &nfailed;
    ~BatchEventCounterWriteback() noexcept {
        int64_t nfailed_sum = std::accumulate(nfailed.begin(), nfailed.end(), int64_t{0});
        if (nfailed_sum > 0) {
            std::fprintf(stderr,
                         "tycho: %lld event-refinement failure(s) across batch; ...\n",
                         static_cast<long long>(nfailed_sum));
        }
        integ.n_failed_event_refinements_ = nfailed_sum;
    }
};
```

**Proposed fix.** Route through a `noexcept` project logger (or add one
if absent). Verify the logger is `noexcept`-compatible — if not, gate
with `try { logError(...); } catch (...) {}` to preserve destructor
contract.

**Acceptance.** No behavior change visible in tests. Manual smoke:
write a short C++ program that closes `stderr` and runs an
event-refinement workload; confirm the destructor still publishes the
counter and doesn't crash.

---

### 3.2 — Per-event-failure diagnostics lost when `refine_events` returns nullopt

**Symbol:** `EventHandler::refine_events` (Newton/bisect NaN-return
paths) in `include/tycho/detail/integrators/event_handler.h`.

**Problem.** When Newton or bisect produces a non-finite value during
event refinement, the routine returns a `std::nullopt` slot in
`eventstates[i]` and increments the failure counter. The user learns
**which** events failed only via the failure-count summary at writeback
(stderr) and the count of `std::nullopt` slots. There's no
per-failure breadcrumb: which event index, which t-bracket, what the
non-finite value was. For event-rich problems ("5 events failed, good
luck finding which") this is a real diagnosis cost.

**Proposed fix.** Add an optional per-failure diagnostic vector,
mirroring `eventtimes`, populated only when the user opts in via
`set_event_diagnostics(true)`:

```cpp
struct EventFailureDiagnostic {
    int event_index;
    double t_lo, t_hi;
    double v_lo, v_hi;          // values at bracket endpoints
    std::string reason;         // "newton_nan", "bisect_nan", "max_iters"
};

// New accessor
std::vector<EventFailureDiagnostic> get_event_failure_diagnostics() const;
```

Default-off (zero overhead when not opted in); off by default to
preserve the noexcept-and-cheap counter writeback path.

**Acceptance.** New test
`tests/cpp/integrators/test_event_refinement_coverage.cpp::Diagnostics_PopulatedOnFailure`
that constructs a deliberately-failing event (returns NaN at known
bracket), enables diagnostics, runs `find_events`, and asserts the
diagnostic vector contains the expected entry with correct event index,
bracket, and reason string.

---

## Section 4 — Documentation & Anchors

### 4.1 — Sweep `2026-04-24-segmented-stm-main-thread-optimization.md` for stale line anchors

**File:** `doc/plans/2026-04-24-segmented-stm-main-thread-optimization.md`.

**Problem.** The plan doc has 13+ hard-coded `integrator.h:NNN` line
anchors. Several have already drifted against `HEAD` of
`ivp-solver-modernization` (e.g., the doc's "1924" is now 1935; the
"1935-1944" main-propagation block is now 1943-1955; "1947-2006" exception
aggregation is now 1958-2017). Commit `76daef1` *removed* a hard-coded
line-number ref from `parallel_driver.h` for exactly this reason — the
plan doc reintroduces the smell at scale.

**Proposed fix.** Replace every line-range anchor with a symbol-name
reference:
- `integrator.h:1896-2031` → `Integrator::integrate_stm_parallel` (in `integrator.h`)
- `integrator.h:1167-1172` → `Integrator::integrate_stm_core` (in `integrator.h`)
- `integrator.h:1054-1067` → `Integrator::integrate_dense_core` (in `integrator.h`)
- `integrator.h:1276-1281` → `STMDriver::calculate_jacobian` (in `stm_driver.h`)
- ...and so on.

Keep the section-internal markdown anchors (`## §1.2`, etc.) — those
are stable. Where the doc needs to point at a code block, use a fenced
quote of the actual code rather than a line-range reference.

**Acceptance.** A mechanical check: `grep -nE 'integrator\.h:[0-9]'`
returns no hits in the doc. A reviewer skimming the doc can resolve
every anchor without opening `integrator.h`.

---

### 4.2 — Tighten commit message wording for `0616a7a`

**Commit:** `0616a7a feat(integrator): type-encode Stepper::peek_fsal freshness`.

**Problem.** The implementation is a runtime check (`if (!peek_fresh_)
throw std::logic_error`), not a compile-time type encoding. The runtime
check is good — unconditional, debuggable, helpful error message — but
the message overstates the mechanism. A future reader expecting a
phantom-typed `FreshDeriv` or a `std::optional`-returning accessor will
be confused.

**Proposed fix.** No code change. If the Stepper API gets refactored
later (§2.3-style restructuring), revisit and either:

- Actually type-encode (return a `StepResult` with a `fresh()` member
  that exposes `fsal()`), or
- Note in the cleanup commit that the runtime check is the chosen
  mechanism and the commit-message wording is loose shorthand.

**Acceptance.** None — this is a documentation note for whoever next
touches the Stepper API.

**Current code (for reference).**
```cpp
const ODEDeriv &peek_fsal() const {
    if (!peek_fresh_) {
        throw std::logic_error(
            "Stepper::peek_fsal: k_fsal_ does not hold f(xf). The most recent step() did "
            "not populate the FSAL endpoint cache for this method × ComputeMidpoint "
            "combination. Methods without LastStageIsFxf require step<true>() to refresh "
            "the cache.");
    }
    return k_fsal_;
}
```

---

### 4.3 — Trim restating-the-obvious comments

**Symbols:**
- `EventPack` class doc in `event_handler.h` (the "the typed fields
  `direction_` and `stop_count_` are private; mutate via the
  validating ... the only way to enter a non-validated state is to bypass
  the setters, which the language now blocks" paragraph).
- `Stepper::peek_fsal` reset comment in `stepper.h` ("previous step's
  peek freshness is invalidated as soon as we begin a new step" — the
  next line literally writes `peek_fresh_ = false;`).

**Problem.** Per CLAUDE.md: "Default to writing no comments. Only add
one when the WHY is non-obvious." Both comments restate what the access
modifier or the next code line already says. They are well-intentioned
but redundant.

**Proposed fix.** Trim. For `EventPack`: keep the
"out-of-range values would silently coerce inside `check_crossings`
(direction sign-product check) and `find_events` (size==stop comparator)"
sentence — that's the WHY. Drop the "the language now blocks" tail.

For `Stepper::peek_fsal`: collapse to one line — the multi-line block
at lines 104-118 already pins the WHY.

**Acceptance.** Code reads cleaner; no behavior change.

---

## Section 5 — Test Coverage Gaps

### 5.1 — Tsit5 / BS3 / BS5 / Vern7-9 not covered on the zero-interval *dense* path

**Symbol:** `tests/cpp/integrators/test_zero_interval.cpp`. The
`Integrate` test is parameterized across all 8 RK methods; `IntegrateDense`,
`IntegrateSTM`, and the batch variants are fixed-fixture on DOPRI54/87
only.

**Problem.** Zero-interval dense output is a legitimate edge case
(empty trajectory, single-point dense table); the new methods could
fail it differently from DOPRI5/8 because their interpolant stage
counts differ.

**Proposed fix.** Promote the dense / STM / batch zero-interval cases
to `TYPED_TEST_P` parameterized over the full method list. Reuse the
method-typing pattern from `test_method_*.cpp`.

**Acceptance.** New parameterizations pass on all 8 methods.

---

### 5.2 — `find_events` max-iters nullopt path conflated with residual path

**Symbol:** `tests/cpp/integrators/test_event_refinement_coverage.cpp`,
specifically `ResetPerCall_SecondCallIndependent`.

**Problem.** That test exercises a tight-tol path that *can* hit either
the residual-check nullopt or the max-iters nullopt depending on
arithmetic. The comment at the top of the file admits "find_events has
other nullopt paths not covered here". A dedicated max-iters case would
harden against future refactors that drop one branch.

**Proposed fix.** Add `MaxItersExhausted_ReturnsNullopt`: a deliberately
un-converging event function (returns `t * cos(1e6 * t)` or similar,
many sign changes within bracket) with `max_event_iters_=1` to force
the max-iters path specifically.

**Acceptance.** New test passes; coverage tool shows the max-iters
branch hit independently of residual branch.

---

### 5.3 — `integrate_dense_parallel` worker-throw aggregation only implicitly tested

**Symbol:** `tests/cpp/integrators/test_parallel_stm_errors.cpp` —
covers `integrate_stm_parallel` thoroughly (3 cases on the unwind-drain
path) but no equivalent for `integrate_dense_parallel`. The
`ParallelIntegrateThrowsUnderConcurrency` case in
`test_integrator_parallel.cpp` only checks `EXPECT_THROW`, not the
aggregation message structure.

**Proposed fix.** Add a `DenseParallelExceptionAggregation` test that
forces multiple workers to throw with distinct messages and asserts the
composite `runtime_error` contains "primary segment failure",
"additional failures", and the per-worker messages — mirroring the
shape of the existing `integrate_stm_parallel` aggregation tests.

**Acceptance.** New test passes; message-structure regression is now
pinned.

---

## Section 6 — Forward-looking: STM / Higher-Order Derivative Correctness

**Status:** Reserved. To be filled in after a separate design discussion
the user has flagged.

The user has correctness concerns related to how STMs and higher-order
derivatives are computed in the integrator stack. Concerns are not yet
specified in this doc. The boundary is reserved here so the cleanup
doc can be extended in place rather than fragmenting across multiple
plan docs.

### Already-on-disk related artifacts

- `doc/plans/2026-04-24-segmented-stm-main-thread-optimization.md` —
  deferred performance optimization on the segmented-STM threadpool
  path. Note: this doc itself needs the line-anchor sweep in §4.1.
- Commit `e121ed2` — `integrate_stm2` binding-cast fix (segfault on
  statically-sized ODEs like Kepler whose Jacobian/Hessian resolve to
  fixed-size matrices instead of `MatrixXd`). The fix dropped the
  function-pointer cast so nanobind deduces the overload. This closes a
  precondition for any STM2 correctness work.
- The 8 new RK methods (Tsit5, BS3, BS5, Vern7/8/9) have STM coverage
  in `tests/cpp/integrators/test_stm_all_methods.cpp` — analytical-STM
  pin per method on a Kepler reference. Any correctness audit should
  start there.
- `integrate_stm_parallel` segmented chaining and bit-identity pins in
  `tests/cpp/integrators/test_segmented_stm_serial.cpp` and
  `test_integrator_parallel.cpp::SegmentedStmBitIdentity`.

### Suggested structure for §6 once filled in

1. **Concern statement** — what specifically is the user worried about
   (chain-rule numerical conditioning, autodiff vs. variational
   equations, second-order sensitivity at single-step vs. multi-step,
   frame-of-evaluation drift across segments, etc.).
2. **Reproducer** — a small failing test or failing analytical
   comparison.
3. **Affected paths** — which `integrate_stm`, `integrate_stm2`,
   `integrate_*_parallel` entry points are implicated.
4. **Proposed approach** — fix, redesign, or audit-only.
5. **Acceptance** — test/benchmark that demonstrates the concern is
   resolved.

---

## Acceptance for this doc

- Every code anchor is symbol-named (no `:NNN` line refs in body text).
- Every Section 1–5 item is independently mergeable; no item depends on
  another in this doc.
- §6 is filled in via a follow-up commit after the design discussion.
- PR #44's body links to this doc under a "Deferred follow-ups" header
  so future-self can find the deferred work post-merge.
- Each item's "Acceptance" subsection names a specific test (existing or
  new) that proves the fix works, or explicitly notes "no behavior
  change" / "compile-only smoke" where applicable.
