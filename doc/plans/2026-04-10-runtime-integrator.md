# Runtime ODE Integrator — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ODE::integrator()` to the builder API with control laws, event detection, and interpolation table controls. Rename `RKOptions` → `IVPAlg` and replace all string-based algorithm selection with the enum.

**Architecture:** The `IVPAlg` enum replaces `RKOptions` in `rk_coeffs.h` and moves to the `tycho` namespace. All `Integrator<DODE>` constructors change from `std::string` to `IVPAlg`. `ODE::integrator()` factory methods construct `Integrator<DynODE>` by building a `DynODE` via `generic_ode()`, populating its FlatMap, and forwarding to the `Integrator` constructor. Python bindings convert strings to `IVPAlg` at the binding layer.

**Tech Stack:** C++20, Eigen, nanobind, existing Integrator/RK infrastructure

**Spec:** `doc/plans/TYCHO_RUNTIME_INTEGRATOR_SPEC.md`

---

## File Structure

**Modified files:**
- `include/tycho/detail/integrators/rk_coeffs.h` — `RKOptions` → `IVPAlg`, move to `tycho` ns, remove dead values
- `include/tycho/detail/integrators/rk_steppers.h` — update all `RKOptions` template params
- `include/tycho/detail/integrators/integrator.h` — `std::string` → `IVPAlg` in constructors and `set_method`
- `include/tycho/detail/optimal_control/builder/runtime_ode.h` — `DynIntegrator` alias, `integrator()` declarations
- `src/optimal_control/runtime_ode.cpp` — `integrator()` implementations
- `src/bindings/integrators/rk_coeffs_bind.cpp` — rename enum binding
- `src/bindings/integrators/integrator_bind.h` — string-to-enum at binding layer
- `tychopy/OptimalControl/__init__.py` — `RKOptions` → `IVPAlg` re-export
- 4 example `main.cpp` files (GoddardRocket, MultiPhaseCannon, TopputtoLowThrust, OrbitContinuation)

---

## Task 0: Rename RKOptions → IVPAlg

**Files:**
- Modify: `include/tycho/detail/integrators/rk_coeffs.h`
- Modify: `include/tycho/detail/integrators/rk_steppers.h`
- Modify: `include/tycho/detail/integrators/integrator.h`

This is a mechanical rename across 3 headers. No behavioral changes.

- [ ] **Step 1: Read all three files**

Read `rk_coeffs.h`, `rk_steppers.h`, and `integrator.h` to locate every `RKOptions` occurrence.

- [ ] **Step 2: Update rk_coeffs.h**

Replace the enum definition (lines 21-31):

```cpp
// Before (namespace tycho::integrators):
enum class RKOptions {
    RK4Classic, RK438, DOPRI54, DOPRI87, RK54, RK78, Ralston3, Ralston2, DOPRI5
};

// After (namespace tycho — move out of integrators sub-namespace):
enum class IVPAlg {
    DOPRI54,    // Dormand-Prince 5(4) — 6 stages, adaptive
    DOPRI87,    // Dormand-Prince 8(7) — 13 stages, adaptive (default)
    RK4Classic, // Classic RK4 — 4 stages, fixed step only
    DOPRI5,     // Dormand-Prince 5 — 6 stages, fixed step only
};
```

Update all `RKCoeffs<RKOptions::...>` specializations to `RKCoeffs<IVPAlg::...>`.
Remove the specializations for dead values (RK438, RK54, RK78, Ralston3, Ralston2) — these had no specializations anyway, so just the enum values go away.

The enum must be in `namespace tycho` (not `tycho::integrators`). The `RKCoeffs` struct template stays in `tycho::integrators` but its template parameter changes from `RKOptions` to `IVPAlg`.

- [ ] **Step 3: Update rk_steppers.h**

Replace all `RKOptions` references (~10 occurrences) with `IVPAlg`:
- Template parameters: `template <class DODE, RKOptions RKOp>` → `template <class DODE, IVPAlg RKOp>`
- Comparisons: `RKOptions::RK4Classic` → `IVPAlg::RK4Classic`, etc.

- [ ] **Step 4: Update integrator.h — member and template references**

- Member (line 131): `RKOptions rk_method_ = RKOptions::DOPRI54;` → `IVPAlg rk_method_ = IVPAlg::DOPRI87;` (fix the DOPRI54/87 inconsistency)
- Template type alias (line 104): `RKOptions` → `IVPAlg`
- `init_stepper_and_controller` template param: `RKOptions` → `IVPAlg`
- `stepper_compute` switch: `RKOptions::DOPRI54` → `IVPAlg::DOPRI54`, etc.
- All other `RKOptions` references

Do NOT change the constructor signatures yet (that's Task 1).

- [ ] **Step 5: Build to verify**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
./examples/cpp_examples/builder/brachistochrone/brachistochrone_builder
```
Expected: compiles and runs, prints "optimal solution found"

- [ ] **Step 6: Commit**

```bash
git add include/tycho/detail/integrators/rk_coeffs.h \
        include/tycho/detail/integrators/rk_steppers.h \
        include/tycho/detail/integrators/integrator.h
git commit -m "refactor: rename RKOptions to IVPAlg in tycho namespace

Move IVP algorithm enum from tycho::integrators::RKOptions to
tycho::IVPAlg. Remove dead enum values (RK438, RK54, RK78, Ralston3,
Ralston2). Keep DOPRI54, DOPRI87 (adaptive), RK4Classic, DOPRI5
(fixed-step). Fix member initializer to default to DOPRI87."
```

---

## Task 1: Replace String-Based Algorithm Selection

**Files:**
- Modify: `include/tycho/detail/integrators/integrator.h`

Change all constructor `std::string meth` parameters to `IVPAlg meth` and rewrite `set_method()` from string comparison to enum dispatch.

- [ ] **Step 1: Update set_method()**

Replace the string-based `set_method()` (lines 213-231):

```cpp
// Before:
void set_method(std::string str, const DODE &dode, double defstep, bool usecontrol,
                const GenericFunction<-1, -1> &ucon, const ControlIndexType &varlocs_t) {
    this->set_step_sizes(defstep, defstep / 10000, defstep * 10000);
    if (str == "DOPRI54" || str == "DP54") {
        ...
    } else if (str == "DOPRI87" || str == "DP87") {
        ...
    } else {
        throw std::invalid_argument("Invalid integration method");
    }
}

// After:
void set_method(IVPAlg alg, const DODE &dode, double defstep, bool usecontrol,
                const GenericFunction<-1, -1> &ucon, const ControlIndexType &varlocs_t) {
    this->set_step_sizes(defstep, defstep / 10000, defstep * 10000);
    switch (alg) {
    case IVPAlg::DOPRI54:
        this->rk_method_ = IVPAlg::DOPRI54;
        this->error_order_ = 4;
        this->init_stepper_and_controller<IVPAlg::DOPRI5>(dode, usecontrol, ucon, varlocs_t);
        break;
    case IVPAlg::DOPRI87:
        this->rk_method_ = IVPAlg::DOPRI87;
        this->error_order_ = 7;
        this->init_stepper_and_controller<IVPAlg::DOPRI87>(dode, usecontrol, ucon, varlocs_t);
        break;
    case IVPAlg::RK4Classic:
    case IVPAlg::DOPRI5:
        throw std::invalid_argument(
            "IVPAlg::RK4Classic and IVPAlg::DOPRI5 do not support adaptive step control. "
            "Set adaptive = false before using these methods.");
    }
}
```

- [ ] **Step 2: Update all constructor signatures**

Change every `std::string meth` parameter to `IVPAlg meth` across all 11 constructors. For constructors that default to `"DOPRI87"`, change to `IVPAlg::DOPRI87`.

For the `Integrator(DODE, double)` constructor that currently delegates to the string version, change to delegate to the `IVPAlg` version:
```cpp
Integrator(const DODE &dode, double defstep)
    : Integrator(dode, IVPAlg::DOPRI87, defstep) {}
```

- [ ] **Step 3: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder hypersens_builder
./examples/cpp_examples/builder/brachistochrone/brachistochrone_builder
./examples/cpp_examples/builder/hypersens/hypersens_builder
```

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/integrators/integrator.h
git commit -m "refactor: replace string-based IVP algorithm selection with IVPAlg enum

All Integrator<DODE> constructors now take IVPAlg instead of
std::string. set_method() uses switch instead of string comparison.
RK4Classic and DOPRI5 throw if adaptive stepping is enabled."
```

---

## Task 2: Update Python Bindings

**Files:**
- Modify: `src/bindings/integrators/rk_coeffs_bind.cpp`
- Modify: `src/bindings/integrators/integrator_bind.h`
- Modify: `tychopy/OptimalControl/__init__.py`

- [ ] **Step 1: Update enum binding in rk_coeffs_bind.cpp**

```cpp
// Before:
void RKFlagsBuild(nb::module_ &m) {
    nb::enum_<RKOptions>(m, "RKOptions")
        .value("RK4", RKOptions::RK4Classic)
        .value("DOPRI54", RKOptions::DOPRI54)
        .value("DOPRI87", RKOptions::DOPRI87);
}

// After:
void RKFlagsBuild(nb::module_ &m) {
    nb::enum_<IVPAlg>(m, "IVPAlg")
        .value("DOPRI54", IVPAlg::DOPRI54)
        .value("DOPRI87", IVPAlg::DOPRI87)
        .value("RK4Classic", IVPAlg::RK4Classic)
        .value("DOPRI5", IVPAlg::DOPRI5);
}
```

- [ ] **Step 2: Update integrator_bind.h — add string-to-enum conversion**

Add a helper function at the top of the bind namespace:

```cpp
inline IVPAlg parse_ivp_alg(const std::string &str) {
    if (str == "DOPRI54" || str == "DP54") return IVPAlg::DOPRI54;
    if (str == "DOPRI87" || str == "DP87") return IVPAlg::DOPRI87;
    if (str == "RK4" || str == "RK4Classic") return IVPAlg::RK4Classic;
    if (str == "DOPRI5" || str == "DP5") return IVPAlg::DOPRI5;
    throw std::invalid_argument(fmt::format("Unknown IVP algorithm: '{}'", str));
}
```

Update all `IntegratorBuildConstructors` lambda bindings that take `std::string meth` to accept either `IVPAlg` or `std::string` (via separate overloads or `nb::overload_cast`). The string overloads call `parse_ivp_alg()` then forward to the C++ constructor.

- [ ] **Step 3: Update Python re-export**

In `tychopy/OptimalControl/__init__.py`, change:
```python
# Before:
RKOptions = _tychopy.OptimalControl.RKOptions

# After:
IVPAlg = _tychopy.OptimalControl.IVPAlg
RKOptions = IVPAlg  # backwards compat alias
```

- [ ] **Step 4: Build full (bindings are heavy)**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 _tychopy
```

- [ ] **Step 5: Test Python examples**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py --filter Brachistochrone"
```

- [ ] **Step 6: Commit**

```bash
git add src/bindings/integrators/rk_coeffs_bind.cpp \
        src/bindings/integrators/integrator_bind.h \
        tychopy/OptimalControl/__init__.py
git commit -m "feat: expose IVPAlg enum to Python, accept string or enum

Python can use IVPAlg.DOPRI87 or 'DOPRI87' interchangeably.
String-to-enum conversion at binding layer only. RKOptions kept
as backwards-compat alias."
```

---

## Task 3: ODE::integrator() Factory Methods

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/runtime_ode.h`
- Modify: `src/optimal_control/runtime_ode.cpp`

- [ ] **Step 1: Add DynIntegrator alias and declarations to runtime_ode.h**

Add `#include "tycho/detail/integrators/integrator.h"` to runtime_ode.h.

Add to the `ODE` class public section:

```cpp
using DynIntegrator = integrators::Integrator<DynODE>;

// Basic (no control)
DynIntegrator integrator(double defstep) const;
DynIntegrator integrator(IVPAlg method, double defstep) const;

// With control law — index-based
DynIntegrator integrator(double defstep, GenericFunction<-1, -1> ulaw,
                         const Eigen::VectorXi &varlocs) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         GenericFunction<-1, -1> ulaw,
                         const Eigen::VectorXi &varlocs) const;

// With control law — named
DynIntegrator integrator(double defstep, GenericFunction<-1, -1> ulaw,
                         std::initializer_list<std::string> varlocs) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         GenericFunction<-1, -1> ulaw,
                         std::initializer_list<std::string> varlocs) const;

// With constant control vector
DynIntegrator integrator(double defstep, const Eigen::VectorXd &u_const) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         const Eigen::VectorXd &u_const) const;

// With LGL interpolation table
DynIntegrator integrator(double defstep, std::shared_ptr<oc::LGLInterpTable> tab,
                         const Eigen::VectorXi &ulocs) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         std::shared_ptr<oc::LGLInterpTable> tab,
                         const Eigen::VectorXi &ulocs) const;
DynIntegrator integrator(double defstep,
                         std::shared_ptr<oc::LGLInterpTable> tab) const;
DynIntegrator integrator(IVPAlg method, double defstep,
                         std::shared_ptr<oc::LGLInterpTable> tab) const;
```

- [ ] **Step 2: Add a private helper to build DynODE with FlatMap**

In `runtime_ode.cpp`, add a private helper (or reuse inline in each method):

```cpp
ODE::DynODE ODE::make_dyn_ode() const {
    DynODE ode = generic_ode();
    if (registry_) {
        for (const auto &[name, idxs] : registry_->entries())
            ode.add_idx(name, idxs);
    }
    return ode;
}
```

This matches the pattern from `ODE::phase()` (lines 35-44 of runtime_ode.cpp).

- [ ] **Step 3: Implement all integrator() overloads in runtime_ode.cpp**

Each overload calls `make_dyn_ode()` then constructs the integrator. For named varlocs, resolve through `VarRegistry` first:

```cpp
ODE::DynIntegrator ODE::integrator(double defstep) const {
    return DynIntegrator(make_dyn_ode(), defstep);
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep) const {
    return DynIntegrator(make_dyn_ode(), method, defstep);
}

ODE::DynIntegrator ODE::integrator(double defstep, GenericFunction<-1, -1> ulaw,
                                   const Eigen::VectorXi &varlocs) const {
    return DynIntegrator(make_dyn_ode(), defstep, ulaw, varlocs);
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep,
                                   GenericFunction<-1, -1> ulaw,
                                   const Eigen::VectorXi &varlocs) const {
    return DynIntegrator(make_dyn_ode(), method, defstep, ulaw, varlocs);
}

// Named varlocs — resolve then delegate
ODE::DynIntegrator ODE::integrator(double defstep, GenericFunction<-1, -1> ulaw,
                                   std::initializer_list<std::string> varlocs) const {
    check_registry();
    return DynIntegrator(make_dyn_ode(), defstep, ulaw, registry_->resolve(varlocs));
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep,
                                   GenericFunction<-1, -1> ulaw,
                                   std::initializer_list<std::string> varlocs) const {
    check_registry();
    return DynIntegrator(make_dyn_ode(), method, defstep, ulaw, registry_->resolve(varlocs));
}

// Constant control, LGLInterpTable — straightforward forwarding
// ... (same pattern: make_dyn_ode() then construct)
```

- [ ] **Step 4: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/runtime_ode.h \
        src/optimal_control/runtime_ode.cpp
git commit -m "feat: add ODE::integrator() factory methods

14 overloads covering basic, control-law (index + named), constant
control, and LGLInterpTable variants. Each builds a DynODE with
FlatMap populated from VarRegistry, then constructs Integrator."
```

---

## Task 4: Update C++ Examples

**Files:**
- Modify: `examples/cpp_examples/builder/goddard_rocket/main.cpp`
- Modify: `examples/cpp_examples/builder/multi_phase_cannon/main.cpp`
- Modify: `examples/cpp_examples/builder/topputto_low_thrust/main.cpp`
- Modify: `examples/cpp_examples/builder/orbit_continuation/main.cpp`

Update examples that use hand-rolled Euler integration to use `ode.integrator()`.

- [ ] **Step 1: Read each Python example and current C++ example**

For each example, read both the Python source and the current C++ implementation to understand exactly what the Python integrator does and what the C++ should match.

- [ ] **Step 2: Update GoddardRocket**

Python pattern:
```python
integ = ode.integrator(0.01, Ulaw(), [2])
TrajIG = integ.integrate_dense(X0, 60/Tstar, 1000, StopFunc)
```

Replace the hand-rolled Euler loop with:
```cpp
// Control law: u = 1 if m > mf, else 0
auto ulaw_args = Arguments<1>();
auto ulaw_expr = IfElseFunction(ulaw_args.coeff<0>() > mf, 1.0, 0.0);
auto ulaw = GenericFunction<-1, -1>(ulaw_expr);

auto integ = ode.integrator(0.01, ulaw, {"m"});
integ.set_abs_tol(1.0e-12);

auto stop_fn = [](const Eigen::VectorXd &x) { return x[1] < 0; };
auto traj_ig = integ.integrate_dense(X0, 60.0 / Tstar, 1000, stop_fn);
```

Build and run, verify convergence matches Python.

- [ ] **Step 3: Update MultiPhaseCannon**

Python pattern:
```python
integ = ode.integrator(0.01)
integ.set_abs_tol(1.0e-14)
AscentIG = integ.integrate_dense(IG, 60/Tstar, [(AscentEvent(), 0, 1)])[0]
DescentIG = integ.integrate_dense(AscentIG[-1], ..., [(ode_args[2], 0, 1)])[0]
```

Replace hand-rolled IG with integrated trajectory + event detection for ascent/descent split.

- [ ] **Step 4: Update TopputtoLowThrust**

Python uses `integ = ode.integrator(0.01)` with event detection to find when radius reaches target. Replace linear IG with integrated spiral.

- [ ] **Step 5: Update OrbitContinuation**

Python uses `integ = ode.integrator(dt)` to propagate periodic orbits. Replace manual orbit construction with integrated trajectory.

- [ ] **Step 6: Build all updated examples**

```bash
cd /home/ghecht/Projects/tycho/build
ninja -j8 goddard_rocket_builder
ninja -j8 multi_phase_cannon_builder
ninja -j8 topputto_low_thrust_builder
ninja -j8 orbit_continuation_builder
```

Run each and verify convergence.

- [ ] **Step 7: Commit**

```bash
git add examples/cpp_examples/builder/goddard_rocket/main.cpp \
        examples/cpp_examples/builder/multi_phase_cannon/main.cpp \
        examples/cpp_examples/builder/topputto_low_thrust/main.cpp \
        examples/cpp_examples/builder/orbit_continuation/main.cpp
git commit -m "feat: update C++ examples to use ODE::integrator() for initial guesses

Replace hand-rolled Euler integration with ode.integrator() in
GoddardRocket (control law + stop function), MultiPhaseCannon
(event detection), TopputtoLowThrust (spiral IG), and
OrbitContinuation (periodic orbit propagation)."
```

---

## Task 5: Full Verification

- [ ] **Step 1: Build all C++ examples**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j2 cpp_examples_all
```

- [ ] **Step 2: Run CTest**

```bash
ctest -R cpp_example --output-on-failure -j1
```
Expected: all pass.

- [ ] **Step 3: Run Python examples**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```
Expected: all 38 pass (bindings updated).

- [ ] **Step 4: Run C++ unit tests**

```bash
ctest --output-on-failure -j1
```

- [ ] **Step 5: Compare convergence with Python for updated examples**

For GoddardRocket, MultiPhaseCannon, TopputtoLowThrust, OrbitContinuation:
run both Python and C++ versions, compare iteration counts and objective values.
Document any discrepancies.

- [ ] **Step 6: Commit any remaining fixes**
