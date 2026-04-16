# IVP Solver Modernization — Design Spec

## Overview

Modernize Tycho's IVP solver and transcription stepper subsystems: add modern
RK algorithms (Tsit5, BS3, BS5, Vern7, Vern8, Vern9), improve adaptive step-size
control (PI/PID controllers), and restructure for proper encapsulation. Both the
IVP integration path and the optimal-control transcription stepper path benefit
from these changes.

Implementations are based on
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) as the
reference. Julia source locations below refer to installed package paths under
`~/.julia/packages/`; the corresponding GitHub repo is
`SciML/OrdinaryDiffEq.jl` at `master`.

## Sub-Project Decomposition

| SP  | Scope | Oracle |
|-----|-------|--------|
| SP1 | Pure refactor: extract Stepper/Controller/Driver/EventHandler, split `integrator.h`, clean up `rk_steppers.h`, normalize `rk_coeffs.h` | Bit-identical with FP-noise fallback |
| SP2 | Add Tsit5, BS3, BS5, Vern7, Vern8, Vern9 to both IVP and transcription paths; Julia comparison harness | Julia parity (final state, step count, runtime) |
| SP3 | PI/PID controllers, error norm options (RMS vs max), controller auto-selection per algorithm | Julia parity (step count, rejection rate) |
| SP4 | Shooting transcription performance investigation (deferred) | Profiling-guided |

Each sub-project gets its own spec-plan-implement cycle.

---

## SP1: Pure Refactor

### File Layout

New files under `include/tycho/detail/integrators/`:

| File | Responsibility | Depends on |
|------|---------------|------------|
| `rk_coeffs.h` | `IVPAlg` enum + `RKCoeffs<Alg>` tableaux. Shared by IVP and transcription. | `<array>` |
| `stepper.h` | `Stepper<Alg, DODE, Scalar>` — stateful IVP stepper | `rk_coeffs.h`, VF types |
| `step_controller.h` | `IController` (SP1); PI/PID added in SP3 | `<cmath>` |
| `adaptive_driver.h` | `AdaptiveDriver<Alg, Controller, DODE, Scalar>` — unified adaptive loop | `stepper.h`, `step_controller.h`, `event_handler.h` |
| `event_handler.h` | `EventHandler` — sign-change detection + bisection/Newton refinement | VF types |
| `stm_driver.h` | `STMDriver<DODE>` — Jacobian/Hessian chain computation via transcription stepper | VF types |
| `parallel_driver.h` | SuperScalar SIMD batch orchestration | `adaptive_driver.h` |
| `integrator.h` | `Integrator<DODE>` — public VectorFunction class, variant dispatch, Python API | all above + `rk_steppers.h` |
| `rk_steppers.h` | Transcription stepper VFs (cleaned up) | `rk_coeffs.h`, VF system |

Dependency direction is strictly downward. The umbrella `include/tycho/integrators.h`
continues to include everything.

### Tableau Interface (`rk_coeffs.h`)

Standardize `RKCoeffs<Alg>` to expose:

```cpp
static constexpr int Stages;
static constexpr int Order;           // NEW
static constexpr int ErrorOrder;      // NEW
static constexpr bool FSAL;
static constexpr bool HasEmbedded;    // renamed from EmbeddedCorrector

static constexpr std::array<std::array<double, Stages-1>, Stages-1> A;  // renamed from ACoeffs
static constexpr std::array<double, Stages-1> C;    // renamed from Times (abscissae)
static constexpr std::array<double, Stages> B;       // renamed from BCoeffs (weights)
static constexpr std::array<double, Stages> Bhat;    // renamed from CCoeffs (embedded weights)
static constexpr std::array<double, Stages> Bmid;    // renamed from MidCoeffs (optional)
```

Standard Butcher tableau notation per Hairer, Norsett & Wanner.

The `DOPRI5` internal tag is removed; `RK4Classic` stays as internal-only.
All coefficient values remain bit-identical; this is a rename + interface
normalization.

### IVP Stepper (`stepper.h`)

Replaces `stepper_compute_impl<IVPAlg>` in current `integrator.h` (~lines
416-560).

```cpp
template <IVPAlg Alg, class DODE, class Scalar = double>
struct Stepper {
    using Coeffs = RKCoeffs<Alg>;

    explicit Stepper(const DODE& ode);

    void step(const ODEState<Scalar>& x, Scalar tf,
              ODEState<Scalar>& xf, ODEState<Scalar>& xf_est,
              bool compute_midpoint, ODEState<Scalar>& xf_mid);

    void reset_fsal();

    static constexpr int stages()      { return Coeffs::Stages; }
    static constexpr int order()       { return Coeffs::Order; }
    static constexpr int error_order() { return Coeffs::ErrorOrder; }
    static constexpr bool has_fsal()   { return Coeffs::FSAL; }
};
```

State: `k_` stage buffer, `k_fsal_` cached first-stage derivative,
`fsal_valid_` flag. Allocated once at construction. FSAL is fully internal
to the stepper; the driver calls `step()` and `reset_fsal()` without knowing
whether the method uses FSAL.

`Scalar` template parameter allows `double` (scalar path) and
`DefaultSuperScalar` (SIMD batch path) using distinct stepper instances.

### Step-Size Controller (`step_controller.h`)

SP1 implements `IController` reproducing current `calc_hnext` exactly:

```cpp
struct IController {
    double safety;         // default 0.9
    double exponent_bias;  // default 0.0

    double propose_step(double h, double err_norm, int order) const;
    void accept(double err_norm);
    void reject(double err_norm);
    void reset();
};
```

The driver computes the scalar `err_norm` and calls `propose_step`. Clamping
(`max_step_change`, `min/max_step_size`) stays in the driver. The accept/reject
decision (`err_norm > 1.0`) stays in the driver.

The `accept`/`reject`/`reset` methods are no-ops for `IController` but exist
so PI/PID controllers in SP3 can cache error history without changing the
driver code.

**OrdinaryDiffEq.jl reference:**
`~/.julia/packages/OrdinaryDiffEqCore/bDZEh/src/integrators/controllers.jl`
lines 138-200 (`IController`), 280-366 (`PIController`), 469-637
(`PIDController`).

### Adaptive Driver (`adaptive_driver.h`)

Replaces both duplicated adaptive loops (scalar ~line 630, SuperScalar batch
~line 830 in current `integrator.h`).

```cpp
template <IVPAlg Alg, class Controller, class DODE, class Scalar = double>
struct AdaptiveDriver {
    Stepper<Alg, DODE, Scalar> stepper_;
    Controller controller_;
    // tolerances, step bounds, error buffers
};
```

Single unified loop: initialize -> step -> compute error norm -> propose new h
-> clamp -> accept/reject -> store states/derivs -> check events -> continue.

Per-component tolerances (`abs_tols_`, `rel_tols_` as `ODEDeriv<double>`
vectors) are preserved. Error norm is currently max-norm; SP3 may switch to
RMS norm for specific methods.

The driver does NOT own the ODE, events, or control-function composition.
These are passed in per call.

### Event Handler (`event_handler.h`)

Stateless utility with static methods:

```cpp
struct EventHandler {
    static bool check_crossings(
        const ODEState<double>& x_prev, const ODEState<double>& x_next,
        const std::vector<EventPack>& events,
        std::vector<Vector1<double>>& prev_vals,
        std::vector<Vector1<double>>& next_vals,
        std::vector<std::vector<Eigen::Vector2d>>& event_times);

    static std::vector<std::vector<ODEState<double>>> resolve_events(
        const std::vector<std::vector<Eigen::Vector2d>>& event_times,
        const std::shared_ptr<LGLInterpTable>& table,
        int max_iters);
};
```

Extracted from inline event logic in both adaptive loops.

### STM Driver (`stm_driver.h`)

Coordination layer for sensitivity computation. Contains
`calculate_jacobian`, `calculate_jacobians`, `calculate_jacobian_hessian`,
`calculate_jacobians_hessians` — all use the transcription `stepper_`
VectorFunction. The transcription stepper is passed by reference, not owned.

### Parallel Driver (`parallel_driver.h`)

SIMD batch orchestration. Packs N scalar trajectories into
`DefaultSuperScalar` SIMD lanes, runs
`AdaptiveDriver<Alg, Controller, DODE, DefaultSuperScalar>` with per-lane
done-flags, and unpacks results. Per-lane bookkeeping (which trajectories
are still running, repacking when a lane finishes) lives here.

### Integrator (`integrator.h`)

Public-facing class. Inherits `VectorFunction` (unchanged for OC pipeline).

```cpp
template <class DODE>
struct Integrator : VectorFunction<Integrator<DODE>, ...> {
    // IVP system — variant dispatch, fully inlined via std::visit
    using DriverVariant = std::variant<
        AdaptiveDriver<IVPAlg::DOPRI54, IController, DODE, double>,
        AdaptiveDriver<IVPAlg::DOPRI87, IController, DODE, double>
    >;
    DriverVariant driver_;
    SIMDDriverVariant simd_driver_;

    // Transcription system — preserved from current
    StepperWrapperType stepper_;   // type-erased RKStepper VF
    STMDriver<DODE> stm_driver_;

    // Settings struct (tolerances, step bounds, etc.)
    Settings settings_;
};
```

`set_method()` switches both the IVP driver variant and rebuilds the
transcription stepper. All public `integrate_*` methods delegate to the active
driver via `std::visit`. Thread-level parallel methods
(`integrate_parallel`, `integrate_stm_parallel`) remain in this file.

Python binding surface is unchanged.

### Transcription Stepper Cleanup (`rk_steppers.h`)

**Dead code removal:** `RKStepper_Impl_NEW`, `RKStepper_NEW`, `RKStepper2`
are unreferenced outside `rk_steppers.h`. Deleted.

**Generalize `final_state_sum`:** Remove per-algorithm `if constexpr`
branches (`RK4Classic`, `DOPRI5`, `DOPRI87`). The generic recursive fallback
already handles arbitrary tableaux with compile-time zero elimination. After
cleanup, any new `RKCoeffs<Alg>` specialization automatically gets a working
`RKStepper<DODE, Alg>` with full derivative chain (Jacobian, adjoint gradient,
adjoint Hessian).

**Regression risk:** Expression-tree shape changes for DOPRI87 may affect FP
evaluation order at the ULP level. These cases downgrade from bit-identical
to FP-noise tolerant in the regression oracle, with documented justification.

### Regression Testing Strategy

Golden corpus generated from `main` before SP1 starts. Serialized to
`tests/cpp/integrators/regression/`.

**IVP path (bit-identical):**

| # | ODE | Algorithm | Exercises |
|---|-----|-----------|-----------|
| 1 | Two-body | DOPRI54 | FSAL, standard adaptive |
| 2 | Two-body | DOPRI87 | Non-FSAL, higher-order |
| 3 | CR3BP (L1 halo) | DOPRI87 | Sensitive dynamics, many steps |
| 4 | Forced control | DOPRI54 | `update_control` path |
| 5 | Altitude crossing event | DOPRI54 | Event bisection/Newton |
| 6 | Batch (N trajectories) | DOPRI87 | SuperScalar path |
| 7 | Backward integration | DOPRI54 | Negative step direction |

**Transcription/OC path (bit-identical or FP-noise):**

| # | ODE | What's checked |
|---|-----|----------------|
| 8 | Two-body | `calculate_jacobian` chain |
| 9 | Two-body | `calculate_jacobian_hessian` + adjoint gradient |
| 10 | CR3BP | `calculate_jacobians` (SuperScalar-batched) |
| 11 | Two-body | `RKStepper<DOPRI54>::compute_jacobian_impl` isolated |
| 12 | Two-body | `RKStepper<DOPRI87>::compute_jacobian_adjointgradient_adjointhessian_impl` isolated |

**End-to-end OC (convergence-equivalent):**

| # | Problem | Gate |
|---|---------|------|
| 13 | Brachistochrone (C++) | "Optimal Solution Found", obj ~ 1.8013 s |
| 14 | Shooting problem (Python) | Convergence + objective matches golden |

---

## SP2: New Algorithms (Sketch)

### Algorithms to Add

| Algorithm | Order | Embedded | FSAL | Stages | Reference source |
|-----------|-------|----------|------|--------|-----------------|
| Tsit5 | 5(4) | Yes | Yes | 7 | `OrdinaryDiffEqTsit5/.../src/tsit_tableaus.jl` |
| BS3 | 3(2) | Yes | Yes | 4 | `OrdinaryDiffEqLowOrderRK/.../src/low_order_rk_tableaus.jl` lines 1-61 |
| BS5 | 5(4) | Yes | No | 8 | `OrdinaryDiffEqLowOrderRK/.../src/low_order_rk_tableaus.jl` lines 513+ |
| Vern7 | 7(6) | Yes | No | 10 | `OrdinaryDiffEqVerner/.../src/verner_tableaus.jl` |
| Vern8 | 8(7) | Yes | No | 13 | `OrdinaryDiffEqVerner/.../src/verner_tableaus.jl` |
| Vern9 | 9(8) | Yes | No | 16 | `OrdinaryDiffEqVerner/.../src/verner_tableaus.jl` |

Full Julia package paths (installed via `Pkg.add("OrdinaryDiffEq")` into
`/tmp/tycho_julia_ref/`):

- `~/.julia/packages/OrdinaryDiffEqTsit5/8E8fO/src/tsit_tableaus.jl`
- `~/.julia/packages/OrdinaryDiffEqLowOrderRK/5x2GR/src/low_order_rk_tableaus.jl`
- `~/.julia/packages/OrdinaryDiffEqVerner/GlLDJ/src/verner_tableaus.jl`
- `~/.julia/packages/OrdinaryDiffEqHighOrderRK/VZ3Fs/src/high_order_rk_tableaus.jl` (DP8 reference)

### Coefficient Convention

OrdinaryDiffEq.jl stores `btilde = b - bhat` (the error weight difference)
rather than `bhat` directly. Tycho stores `B` (weights) and `Bhat` (embedded
weights) separately. When transcribing coefficients, compute
`Bhat[i] = B[i] - btilde[i]` (or store `btilde` directly and adjust the
error computation).

For Tsit5 specifically: the `a71...a76` values are the `b` weights (FSAL —
the 7th row of the A matrix is the b vector). The `btilde1...btilde7` are
`b - bhat`.

### Perform-Step References

- Tsit5: `~/.julia/packages/OrdinaryDiffEqTsit5/8E8fO/src/tsit_perform_step.jl`
- BS3: `~/.julia/packages/OrdinaryDiffEqLowOrderRK/5x2GR/src/low_order_rk_perform_step.jl`
- BS5: same file as BS3
- Vern7/8/9: `~/.julia/packages/OrdinaryDiffEqVerner/GlLDJ/src/verner_rk_perform_step.jl`

### Dense Output / Interpolation References

- Tsit5: `~/.julia/packages/OrdinaryDiffEqTsit5/8E8fO/src/tsit_tableaus.jl`
  (function `Tsit5Interp` — 4th-order polynomial coefficients `r11..r74`)
- Vern7/8/9: `~/.julia/packages/OrdinaryDiffEqVerner/GlLDJ/src/verner_addsteps.jl`

### Julia Comparison Harness

Lives in `/tmp/tycho_julia_ref/` (Julia env already initialized with
`OrdinaryDiffEq` installed). For each algorithm, a Julia script solves
reference problems (two-body, CR3BP) and outputs: final state, step count,
wall time, trajectory. A C++ test solves the same problems with Tycho.
Comparison gates:

- Final state: match within method-order tolerance
- Step count: similar (large divergence indicates coefficient or controller mismatch)
- Runtime: informational, not a gate

### SP1 Interface Validation

SP2 touches only: `RKCoeffs` (add specialization), `IVPAlg` (add enum values),
variant type lists (add alternatives), Python bindings (expose new enum values).
No changes to `Stepper`, `AdaptiveDriver`, `EventHandler`, `RKStepper`, or
`Integrator` method bodies.

---

## SP3: Controllers + Adaptive Stepping (Sketch)

### Controllers to Add

| Controller | Formula | Reference |
|------------|---------|-----------|
| PI | `q = EEst^β₁ / errold^β₂`, `dt_new = dt / (q / γ)` | `controllers.jl` lines 280-366 |
| PID | `dt_factor = err₁^(β₁/k) · err₂^(β₂/k) · err₃^(β₃/k)` with `limiter(x) = 1 + atan(x - 1)` | `controllers.jl` lines 469-637 |

### Default Controller Parameters Per Algorithm

From `~/.julia/packages/OrdinaryDiffEqCore/bDZEh/src/alg_utils.jl` lines
519-532:

- Default `γ = 0.9`
- Default `β₂ = 2 / (5 · order)`
- Default `β₁ = 7 / (10 · order)`
- Default `qmin = 1/5`, `qmax = 10`

Per-algorithm overrides:
- DP5: `β₂ = 4/100`, `β₁ = 1/order - 3β₂/4` (PI controller)
- DP8: `β₂ = 0` (I-controller, not PI), `qmin = 1/3`, `qmax = 6`
- Tsit5: default PI (β₂ = 2/25, β₁ = 7/50)
- BS3: default PI (β₂ = 2/15, β₁ = 7/30)
- BS5: default PI (β₂ = 2/25, β₁ = 7/50)
- Vern7: default PI (β₂ = 2/35, β₁ = 1/10)
- Vern8: default PI (β₂ = 1/20, β₁ = 7/80)
- Vern9: default PI (β₂ = 2/45, β₁ = 7/90)

### Error Norm

From `~/.julia/packages/DiffEqBase/3Bxly/src/`:

- `calculate_residuals.jl`: element-wise `ũ / (α + max(|u₀|, |u₁|) · ρ)`
  where `α = atol`, `ρ = rtol`, `u₀ = state before step`, `u₁ = state after step`
- `common_defaults.jl` line 54: `ODE_DEFAULT_NORM` for arrays is RMS:
  `sqrt(sum(abs2(ui)) / length(u))`

Current Tycho uses max-norm of the element-wise scaled error. SP3 may add
RMS norm as an option, defaulting per algorithm to match OrdinaryDiffEq.jl
behavior.

Current Tycho residual: `(xnext - xnext_est) / (atol + |xnext| · rtol)`.
OrdinaryDiffEq.jl residual uses `max(|u₀|, |u₁|)` instead of just `|u₁|`.
SP3 aligns with the Julia convention.

### Adaptive Loop References

- Step accept/reject: `~/.julia/packages/OrdinaryDiffEqCore/bDZEh/src/integrators/integrator_utils.jl`
  `loopfooter!` (~line 481): calls `stepsize_controller!` then
  `accept_step_controller` then `step_accept_controller!` or
  `step_reject_controller!`
- Initial dt selection: `~/.julia/packages/OrdinaryDiffEqCore/bDZEh/src/initdt.jl`

### SP1 Interface Validation

The `Controller` template parameter on `AdaptiveDriver` is already in place.
The `propose_step`/`accept`/`reject`/`reset` interface is defined. SP3 adds
implementations without changing the driver.

---

## SP4: Shooting Performance (Future)

Deferred. SP1's decomposition makes investigation tractable:

- `STMDriver` isolates Jacobian chain computation for profiling
- `ParallelDriver` isolates SIMD batch path for benchmarking
- Transcription `stepper_` VF is passed explicitly for easy specialization
- Better algorithms + controllers from SP2/SP3 reduce step counts

Potential directions: cache stage values during IVP to avoid re-evaluation in
`calculate_jacobian`, reduce type-erasure overhead, parallelize Jacobian chain.

---

## Design Decisions Log

| # | Decision | Choice | Alternatives considered |
|---|----------|--------|------------------------|
| 1 | Refactor scope | C (aggressive) — both IVP and transcription | A (minimal), B (moderate) |
| 2 | Decomposition | Hybrid (SP1 refactor, SP2 methods, SP3 controllers) | Horizontal, Vertical |
| 3 | Regression oracle | Bit-identical with FP-noise fallback | FP-noise default, tolerance-based |
| 4 | Dispatch model | A (compile-time, std::variant) | B (type-erased), C (hybrid) |
| 5 | IVP Stepper | Stateful object (FSAL, k_stages internal) | Stateless policy, workspace pattern |
| 6 | Controller | Stateful with internal error history | Stateless with caller-provided history |
| 7 | File layout | 9-file split as above | Fewer files, different grouping |
