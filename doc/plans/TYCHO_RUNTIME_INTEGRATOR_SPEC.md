# Spec: Runtime ODE Integrator (Subsystem 2)

> Add `ODE::integrator()` to the builder API so users can integrate runtime
> ODEs for initial guess construction — with control laws, event detection,
> and interpolation table controls. Also rename `RKOptions` → `IVPAlg` and
> replace all string-based algorithm selection with the enum.

## Motivation

~8 C++ builder examples construct initial guesses via hand-rolled Euler
integration or linear interpolation because the builder `ODE` class has no
`integrator()` method. The underlying `Integrator<DODE>` template already
supports control laws, event detection, and interpolation table controls —
but only for typed ODEs, not the type-erased runtime `ODE`. Bridging this
gap enables faithful 1-to-1 ports of Python examples that use
`ode.integrator(dt, Ulaw(), [vars])` and event-based trajectory splitting.

## Scope

1. Rename `RKOptions` → `IVPAlg` in `tycho` namespace
2. Replace all `std::string` method parameters in `Integrator<DODE>` with
   `IVPAlg`
3. Add `ODE::integrator()` factory methods returning `Integrator<DynODE>`
4. Update Python bindings to accept both `IVPAlg` enum and string (string
   converted at binding layer)
5. Update C++ examples to use `ode.integrator()` for initial guess
   construction
6. Verify convergence parity with Python examples

This spec does NOT cover:
- VF expression additions (RowMatrix, quat_rotate) — Subsystem 3
- InterpTable VF composition — Subsystem 4
- Advanced OC features (LGLInterpTable builder access, STM, Jet) — Subsystem 5
- Domain models (MEE factories) — Subsystem 6

## Changes

### 1. Rename RKOptions → IVPAlg

Rename the `RKOptions` enum class to `IVPAlg` and move it to the `tycho`
namespace (alongside other public enums like `TranscriptionModes`,
`PhaseRegionFlags`, etc.):

```cpp
namespace tycho {

enum class IVPAlg {
    DOPRI54,    // Dormand-Prince 5(4) — 6 stages, adaptive
    DOPRI87,    // Dormand-Prince 8(7) — 13 stages, adaptive (default)
    RK4Classic, // Classic RK4 — 4 stages, fixed step only
    DOPRI5,     // Dormand-Prince 5 — 6 stages, fixed step only
};

} // namespace tycho
```

DOPRI54 and DOPRI87 are fully wired with adaptive step control. RK4Classic
and DOPRI5 have coefficient tables but do not support adaptive stepping —
using them with `adaptive_ = true` (the default) throws
`std::invalid_argument`. They are included in the enum because their
coefficients are implemented and they represent valid IVP algorithms; the
adaptive code path can be added later.

Dead values from the old `RKOptions` enum that had NO coefficient
specializations (`RK438`, `RK54`, `RK78`, `Ralston3`, `Ralston2`) are
removed entirely.

All internal references to `RKOptions` are updated. The old name is removed
entirely (no backwards-compat alias). The `rk_method_` member initializer
is cleaned up to default to `IVPAlg::DOPRI87` (matching the constructor
default — currently there's an inconsistency where the member defaults to
DOPRI54 but the constructor overrides to DOPRI87).

### 2. Replace String-Based Algorithm Selection in Integrator

All `Integrator<DODE>` constructors that take `std::string meth` change to
`IVPAlg meth`:

```cpp
// Before:
Integrator(const DODE &dode, std::string meth, double defstep);
Integrator(const DODE &dode, std::string meth, double defstep,
           const ControllerType &ucon, const ControlIndexType &varlocs_t);
// ... etc

// After:
Integrator(const DODE &dode, IVPAlg meth, double defstep);
Integrator(const DODE &dode, IVPAlg meth, double defstep,
           const ControllerType &ucon, const ControlIndexType &varlocs_t);
// ... etc
```

The internal `set_method()` helper changes from string parsing to direct
enum dispatch. No string comparison.

Default algorithm remains `IVPAlg::DOPRI87`.

### 3. ODE::integrator() Factory Methods

Add to the `ODE` class (`runtime_ode.h`):

```cpp
using DynIntegrator = Integrator<DynODE>;

// Basic (no control)
DynIntegrator integrator(double defstep) const;
DynIntegrator integrator(IVPAlg method, double defstep) const;

// With control law — index-based
DynIntegrator integrator(double defstep,
                         GenericFunction<-1, -1> ulaw,
                         const Eigen::VectorXi &varlocs) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         GenericFunction<-1, -1> ulaw,
                         const Eigen::VectorXi &varlocs) const;

// With control law — named (resolves via VarRegistry)
DynIntegrator integrator(double defstep,
                         GenericFunction<-1, -1> ulaw,
                         std::initializer_list<std::string> varlocs) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         GenericFunction<-1, -1> ulaw,
                         std::initializer_list<std::string> varlocs) const;

// With constant control vector
DynIntegrator integrator(double defstep,
                         const Eigen::VectorXd &u_const) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         const Eigen::VectorXd &u_const) const;

// With LGL interpolation table
DynIntegrator integrator(double defstep,
                         std::shared_ptr<LGLInterpTable> tab,
                         const Eigen::VectorXi &ulocs) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         std::shared_ptr<LGLInterpTable> tab,
                         const Eigen::VectorXi &ulocs) const;
DynIntegrator integrator(double defstep,
                         std::shared_ptr<LGLInterpTable> tab) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         std::shared_ptr<LGLInterpTable> tab) const;
```

Each factory method calls `generic_ode()` to build the `DynODE`, populates
the DynODE's internal FlatMap from the `VarRegistry` (matching the pattern
in `ODE::phase()`), then constructs `Integrator<DynODE>(dyn_ode, ...)`.
Named varlocs are resolved through the `VarRegistry` before passing to the
constructor.

The returned `DynIntegrator` has all existing `integrate_dense` overloads
(basic, with output count, with stop function, with events), plus tolerance
setters — no wrapping needed. Event detection (`EventPack` =
`tuple<GenericFunction<-1,1>, int, int>`) works directly on the
`DynIntegrator`.

### 4. Python Bindings

Update the Python binding layer to:
- Expose `IVPAlg` enum to Python (e.g., `typy.IVPAlg.DOPRI87`)
- Accept both `IVPAlg` enum and string in `ode.integrator()` calls
- String-to-enum conversion happens at the binding layer only (e.g.,
  `"DOPRI54"` / `"DP54"` → `IVPAlg::DOPRI54`)
- Remove string acceptance from C++ `Integrator` constructors entirely

### 5. Update C++ Examples

Update examples that currently use hand-rolled initial guesses to use
`ode.integrator()`:

- **GoddardRocket** — `ode.integrator(0.01, ulaw, {"m"})` with stop
  function for initial guess
- **MultiPhaseCannon** — `ode.integrator(0.01)` with event detection for
  ascent/descent trajectory splitting
- **TopputtoLowThrust** — integrated spiral initial guess
- **OrbitContinuation** — integrated periodic orbit initial guess

## Verification

**Per-example success criteria:**
1. Uses `ode.integrator()` matching the Python example's integrator usage
2. Convergence behavior matches Python (same solution, same iteration count
   within ~1-2 iterations)
3. If other open gaps (MEE models, InterpTable, etc.) prevent full parity,
   document which gap blocks it — the integrator portion itself must still
   match

**No regressions:**
- All existing C++ unit tests pass
- All C++ example CTests pass
- All Python examples pass (bindings updated)

## Files Modified

**Core integrator changes (RKOptions → IVPAlg rename):**
- `include/tycho/detail/integrators/rk_coeffs.h` — rename enum, move to
  `tycho` namespace, remove dead values
- `include/tycho/detail/integrators/rk_steppers.h` — update template
  parameters (10 occurrences)
- `include/tycho/detail/integrators/integrator.h` — replace `std::string`
  with `IVPAlg` in all constructors, fix member initializer

**Builder API:**
- `include/tycho/detail/optimal_control/builder/runtime_ode.h` —
  `DynIntegrator` alias and `integrator()` declarations
- `src/optimal_control/runtime_ode.cpp` — `integrator()` implementations

**Python bindings:**
- `src/bindings/integrators/integrator_bind.h` — string-to-enum conversion,
  expose `IVPAlg`
- `src/bindings/integrators/rk_coeffs_bind.cpp` — rename enum binding
- `tychopy/OptimalControl/__init__.py` — update `RKOptions` re-export to
  `IVPAlg`

**Examples:**
- `examples/cpp_examples/builder/goddard_rocket/main.cpp`
- `examples/cpp_examples/builder/multi_phase_cannon/main.cpp`
- `examples/cpp_examples/builder/topputto_low_thrust/main.cpp`
- `examples/cpp_examples/builder/orbit_continuation/main.cpp`

**Umbrella headers (if IVPAlg needs re-export):**
- `include/tycho/integrators.h` or equivalent
