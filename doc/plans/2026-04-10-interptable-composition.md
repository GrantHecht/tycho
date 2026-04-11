# InterpTable VF Composition — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `interp()` free functions and `InterpType` enum so that `InterpTable1D`–`InterpTable4D` can be composed with VF expressions in the C++ builder API, matching the Python API.

**Architecture:** New `InterpType` enum in `tycho` namespace replaces per-class nested enums. New `interp()` free functions in `tycho::vf` wrap existing `InterpFunction*D` + `.eval()` composition. Tables passed as `shared_ptr` for zero-copy ownership sharing. Python bindings get `parse_interp_type()` helper for backward-compatible string acceptance.

**Tech Stack:** C++20, Eigen, existing VF expression template machinery, nanobind

**Spec:** `doc/plans/TYCHO_INTERPTABLE_COMPOSITION_SPEC.md`

---

## File Structure

**New files:**
- `include/tycho/detail/optimal_control/interp/interp_type.h` — `tycho::InterpType` enum
- `include/tycho/detail/vf/operators/interp_compose.h` — `interp()` free functions (1D–4D)

**Modified files:**
- `include/tycho/detail/optimal_control/interp/interp_table_1d.h` — enum migration + enum constructor
- `include/tycho/detail/optimal_control/interp/interp_table_2d.h` — same
- `include/tycho/detail/optimal_control/interp/interp_table_3d.h` — same
- `include/tycho/detail/optimal_control/interp/interp_table_4d.h` — same
- `include/tycho/optimal_control.h` — umbrella includes
- `src/bindings/vf/interp_table_bind.h` — `parse_interp_type()`, dual constructors, expose enum
- `examples/cpp_examples/builder/minimum_time_to_climb/main.cpp` — table-based aero
- `examples/cpp_examples/builder/min_time_climb_tables/main.cpp` — table-based aero

---

## Task 0: InterpType Enum + Enum Migration

**Files:**
- Create: `include/tycho/detail/optimal_control/interp/interp_type.h`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_1d.h`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_2d.h`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_3d.h`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_4d.h`

- [ ] **Step 1: Read all four interp_table headers**

Read the full contents of each to understand every `InterpType` reference:
- `include/tycho/detail/optimal_control/interp/interp_table_1d.h`
- `include/tycho/detail/optimal_control/interp/interp_table_2d.h`
- `include/tycho/detail/optimal_control/interp/interp_table_3d.h`
- `include/tycho/detail/optimal_control/interp/interp_table_4d.h`

- [ ] **Step 2: Create interp_type.h**

Create `include/tycho/detail/optimal_control/interp/interp_type.h`:

```cpp
#pragma once

namespace tycho {

/// Interpolation method for InterpTable1D through InterpTable4D.
enum class InterpType { Cubic, Linear };

} // namespace tycho
```

- [ ] **Step 3: Migrate InterpTable1D**

In `include/tycho/detail/optimal_control/interp/interp_table_1d.h`:

1. Add `#include "tycho/detail/optimal_control/interp/interp_type.h"` after the existing include
2. Add `using tycho::InterpType;` inside the `namespace tycho::oc {` block (after the existing using declarations)
3. Remove `enum class InterpType { cubic_interp, linear_interp };` from inside `struct InterpTable1D`
4. Change member default: `InterpType interp_kind_ = InterpType::cubic_interp;` → `InterpType interp_kind_ = InterpType::Cubic;`
5. Add enum-based constructor overloads that delegate to a new `set_data` overload:

```cpp
InterpTable1D(const Eigen::VectorXd &Ts, const MatType &Vs, int axis,
              InterpType kind) {
    set_data(Ts, Vs, axis, kind);
}
InterpTable1D(const Eigen::VectorXd &Ts, const Eigen::VectorXd &Vs, int axis,
              InterpType kind) {
    MatType Vstmp = Vs.transpose();
    set_data(Ts, Vstmp, 1, kind);
}
InterpTable1D(const std::vector<Eigen::VectorXd> &Vts, int tvar,
              InterpType kind) {
    // Copy lines 65-98 from the existing string constructor verbatim,
    // but the final call on line 98 becomes: set_data(Ts, Vs, 1, kind);
    // (the only change is passing `kind` as InterpType instead of string)
}
```

6. Add enum-based `set_data` overload:

```cpp
void set_data(const Eigen::VectorXd &Ts, const MatType &Vs, int axis,
              InterpType kind) {
    this->ts_ = Ts;
    if (axis == 1) { this->axis_ = 1; this->vs_ = Vs; }
    else if (axis == 0) { this->axis_ = 0; this->vs_ = Vs.transpose(); }
    else { throw std::invalid_argument("Interpolation axis must be 0 or 1"); }
    this->interp_kind_ = kind;
    // Copy lines 122-150 from the existing string set_data verbatim:
    // tsize_, vlen_, ttotal_ assignment, size checks, ascending order
    // check, even-spacing check, and the final calc_derivs() call.
    // The ONLY difference from the string set_data is that the
    // string-parsing if/else block (lines 114-120) is removed —
    // `kind` is already the correct enum value.
}
```

7. Make the string-based `set_data` delegate to the enum version:

```cpp
void set_data(const Eigen::VectorXd &Ts, const MatType &Vs, int axis,
              std::string kind) {
    InterpType k;
    if (kind == "cubic" || kind == "Cubic") k = InterpType::Cubic;
    else if (kind == "linear" || kind == "Linear") k = InterpType::Linear;
    else throw std::invalid_argument("Unrecognized interpolation type");
    set_data(Ts, Vs, axis, k);
}
```

8. Replace all `InterpType::cubic_interp` → `InterpType::Cubic` and `InterpType::linear_interp` → `InterpType::Linear` throughout the file (in `interp_impl` and elsewhere).

- [ ] **Step 4: Migrate InterpTable2D**

Same pattern as Step 3 for `interp_table_2d.h`:
1. Add include for `interp_type.h`
2. Add `using tycho::InterpType;` in namespace block
3. Remove nested `enum class InterpType`
4. Update member default to `InterpType::Cubic`
5. Add enum-based constructor: `InterpTable2D(VectorXd, VectorXd, MatType, InterpType)`
6. Add enum-based `set_data` that does the actual work
7. Make string `set_data` delegate to enum version
8. Replace all `cubic_interp`/`linear_interp` references

- [ ] **Step 5: Migrate InterpTable3D**

Same pattern for `interp_table_3d.h`. Note: default is `InterpType::linear_interp` (not cubic) — preserve as `InterpType::Linear`.

- [ ] **Step 6: Migrate InterpTable4D**

Same pattern for `interp_table_4d.h`. Note: default is also `InterpType::linear_interp` — preserve as `InterpType::Linear`.

- [ ] **Step 7: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

This verifies the enum migration compiles. If it fails, the error will point to missed `InterpType` references.

- [ ] **Step 8: Commit**

```bash
git add include/tycho/detail/optimal_control/interp/interp_type.h \
        include/tycho/detail/optimal_control/interp/interp_table_1d.h \
        include/tycho/detail/optimal_control/interp/interp_table_2d.h \
        include/tycho/detail/optimal_control/interp/interp_table_3d.h \
        include/tycho/detail/optimal_control/interp/interp_table_4d.h
git commit -m "refactor: replace string-based InterpType with enum class

Add tycho::InterpType { Cubic, Linear } in interp_type.h. Migrate all
four InterpTable headers to use the new enum. String-based constructors
remain for backward compatibility, delegating to enum-based set_data."
```

---

## Task 1: interp() Free Functions

**Files:**
- Create: `include/tycho/detail/vf/operators/interp_compose.h`
- Modify: `include/tycho/optimal_control.h`

- [ ] **Step 1: Read dependency headers**

Read these to confirm exact types and signatures:
- `include/tycho/detail/vf/type_erasure/generic_function.h` — confirm `GenericFunction<IR, OR>` signature
- `include/tycho/detail/vf/expressions/stacked.h` — confirm `stack()` signature

- [ ] **Step 2: Create interp_compose.h**

Create `include/tycho/detail/vf/operators/interp_compose.h`:

```cpp
#pragma once

#include <memory>

#include "tycho/detail/optimal_control/interp/interp_table_1d.h"
#include "tycho/detail/optimal_control/interp/interp_table_2d.h"
#include "tycho/detail/optimal_control/interp/interp_table_3d.h"
#include "tycho/detail/optimal_control/interp/interp_table_4d.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
#include "tycho/detail/vf/expressions/stacked.h"

namespace tycho::vf {

// ── 1D: single VF input ────────────────────────────────────────────
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable1D> &table,
            const DenseFunctionBase<Func, IR, OR> &input) {
    return GenericFunction<-1, -1>(
        oc::InterpFunction1D<-1>(table).eval(input.derived()));
}

// ── 2D: two scalar VF inputs ───────────────────────────────────────
template <class F1, int IR1, int OR1, class F2, int IR2, int OR2>
auto interp(const std::shared_ptr<oc::InterpTable2D> &table,
            const DenseFunctionBase<F1, IR1, OR1> &x,
            const DenseFunctionBase<F2, IR2, OR2> &y) {
    return GenericFunction<-1, 1>(
        oc::InterpFunction2D(table).eval(stack(x.derived(), y.derived())));
}

// ── 2D: single 2-element VF input ──────────────────────────────────
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable2D> &table,
            const DenseFunctionBase<Func, IR, OR> &xy) {
    return GenericFunction<-1, 1>(
        oc::InterpFunction2D(table).eval(xy.derived()));
}

// ── 3D: three scalar VF inputs ─────────────────────────────────────
template <class F1, int IR1, int OR1,
          class F2, int IR2, int OR2,
          class F3, int IR3, int OR3>
auto interp(const std::shared_ptr<oc::InterpTable3D> &table,
            const DenseFunctionBase<F1, IR1, OR1> &x,
            const DenseFunctionBase<F2, IR2, OR2> &y,
            const DenseFunctionBase<F3, IR3, OR3> &z) {
    return GenericFunction<-1, 1>(
        oc::InterpFunction3D(table).eval(
            stack(x.derived(), y.derived(), z.derived())));
}

// ── 3D: single 3-element VF input ──────────────────────────────────
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable3D> &table,
            const DenseFunctionBase<Func, IR, OR> &xyz) {
    return GenericFunction<-1, 1>(
        oc::InterpFunction3D(table).eval(xyz.derived()));
}

// ── 4D: four scalar VF inputs ──────────────────────────────────────
template <class F1, int IR1, int OR1,
          class F2, int IR2, int OR2,
          class F3, int IR3, int OR3,
          class F4, int IR4, int OR4>
auto interp(const std::shared_ptr<oc::InterpTable4D> &table,
            const DenseFunctionBase<F1, IR1, OR1> &x,
            const DenseFunctionBase<F2, IR2, OR2> &y,
            const DenseFunctionBase<F3, IR3, OR3> &z,
            const DenseFunctionBase<F4, IR4, OR4> &w) {
    return GenericFunction<-1, 1>(
        oc::InterpFunction4D(table).eval(
            stack(x.derived(), y.derived(), z.derived(), w.derived())));
}

// ── 4D: single 4-element VF input ──────────────────────────────────
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable4D> &table,
            const DenseFunctionBase<Func, IR, OR> &xyzw) {
    return GenericFunction<-1, 1>(
        oc::InterpFunction4D(table).eval(xyzw.derived()));
}

} // namespace tycho::vf
```

- [ ] **Step 3: Update umbrella header**

In `include/tycho/optimal_control.h`, add after the existing `lgl_interp_functions.h` include (line 18) and before the builder includes (line 19+):

```cpp
#include "tycho/detail/optimal_control/interp/interp_table_1d.h"
#include "tycho/detail/optimal_control/interp/interp_table_2d.h"
#include "tycho/detail/optimal_control/interp/interp_table_3d.h"
#include "tycho/detail/optimal_control/interp/interp_table_4d.h"
#include "tycho/detail/vf/operators/interp_compose.h"
```

- [ ] **Step 4: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/vf/operators/interp_compose.h \
        include/tycho/optimal_control.h
git commit -m "feat: add interp() free functions for InterpTable VF composition

interp(shared_ptr<InterpTable*D>, vf_exprs...) composes interpolation
tables with VF expressions. Supports 1D through 4D with both
multi-scalar-arg and single-vector-arg overloads. Zero-copy ownership
via shared_ptr sharing."
```

---

## Task 2: Update Python Bindings

**Files:**
- Modify: `src/bindings/vf/interp_table_bind.h`

- [ ] **Step 1: Read the current binding file**

Read `src/bindings/vf/interp_table_bind.h` to see all constructor bindings and `__call__` overloads.

- [ ] **Step 2: Add parse_interp_type() and expose InterpType enum**

Add at the top of the file (after the namespace opening), before `InterpTable1DBuild`:

```cpp
inline InterpType parse_interp_type(const std::string &str) {
    if (str == "cubic" || str == "Cubic")
        return InterpType::Cubic;
    if (str == "linear" || str == "Linear")
        return InterpType::Linear;
    throw std::invalid_argument(
        fmt::format("Unknown interpolation type: '{}'", str));
}
```

The `InterpType` enum binding should go in `src/bindings/vf/tycho_vector_functions.cpp` (which calls `InterpTable1DBuild` at line 105), placed before the `InterpTable*DBuild` calls:

```cpp
nb::enum_<InterpType>(mod, "InterpType")
    .value("Cubic", InterpType::Cubic)
    .value("Linear", InterpType::Linear);
```

Note: after Task 0, `using tycho::InterpType;` is in the `tycho::oc` namespace, and the binding file already has `using namespace tycho::oc;` (line 25), so unqualified `InterpType` resolves correctly.

- [ ] **Step 3: Update InterpTable1D bindings**

The existing constructors take `std::string kind`. Add overloads that take `InterpType` directly. For each existing `nb::init<..., std::string>`, add a matching `nb::init<..., InterpType>` overload. Keep the string overloads for backward compatibility.

- [ ] **Step 4: Update InterpTable2D/3D/4D bindings**

Same pattern for each remaining table type.

- [ ] **Step 5: Build Python module**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 _tychopy
```

- [ ] **Step 6: Test Python examples still work**

```bash
cd /home/ghecht/Projects/tycho && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py --filter MinimumTimeToClimb"
```

- [ ] **Step 7: Commit**

```bash
git add src/bindings/vf/interp_table_bind.h
git commit -m "feat: expose InterpType enum to Python, accept string or enum

Add parse_interp_type() helper and dual constructor overloads.
Python can use InterpType.Cubic or 'cubic' interchangeably."
```

---

## Task 3: Update MinTimeToClimb Example

**Files:**
- Modify: `examples/cpp_examples/builder/minimum_time_to_climb/main.cpp`

- [ ] **Step 1: Read both Python and C++ versions**

Read:
- `examples/python_examples/MinimumTimeToClimb.py` — the target pattern
- `examples/python_examples/MinimumTimeToClimbTables.py` — table data
- `examples/cpp_examples/builder/minimum_time_to_climb/main.cpp` — current workaround

- [ ] **Step 2: Replace simplified aero with table-based aero**

Replace the ODE definition to use InterpTable with `interp()`, matching the Python pattern exactly. The key changes:

1. Add the atmospheric and aerodynamic data arrays (from `MinimumTimeToClimbTables.py`) as `Eigen::VectorXd` / `Eigen::Matrix<double, -1, -1, Eigen::RowMajor>` before the `ODEBuilder` call
2. Create the table shared_ptrs before the ODEBuilder call:

```cpp
auto rhoTab = std::make_shared<InterpTable1D>(alts, rhos, 0, InterpType::Cubic);
auto sosTab = std::make_shared<InterpTable1D>(alts, soss, 0, InterpType::Cubic);
auto ClalphaTab = std::make_shared<InterpTable1D>(AeroMach, Clalpha, 0, InterpType::Cubic);
auto etaTab = std::make_shared<InterpTable1D>(AeroMach, eta, 0, InterpType::Cubic);
auto CD0Tab = std::make_shared<InterpTable1D>(AeroMach, CD0, 0, InterpType::Cubic);
auto ThrustTab = std::make_shared<InterpTable2D>(ThrustMach, ThrustAlt, ThrustData, InterpType::Cubic);
```

3. Capture tables in the lambda (by value — copies the shared_ptrs, not the data):

```cpp
auto ode = ODEBuilder(4, 1)
    .define([=, /* existing captures */](auto &args) {
        // tables captured by value (shared_ptr copy)
```

4. In the ODE lambda, use `interp(table, vf_expr)` instead of polynomial approximations.

The ODE body should look like:
```cpp
auto rho = interp(rhoTab, h * Lstar) / Rhostar;
auto sos = interp(sosTab, h * Lstar) / Vstar;
auto Mach = v / sos;
auto CD0 = interp(CD0Tab, Mach);
auto Clalpha = interp(ClalphaTab, Mach);
auto eta_val = interp(etaTab, Mach);
auto Thrust = interp(ThrustTab, Mach, h * Lstar) / Fstar;
auto CD = CD0 + eta_val * Clalpha * (alpha * alpha);
auto CL = Clalpha * alpha;
auto q = 0.5 * rho * v * v;
auto D = q * S_nd * CD;
auto L = q * S_nd * CL;
auto r = h + Re_nd;
auto hdot = v * sin(fpa);
auto vdot = (Thrust * cos(alpha) - D) / mass - mu_nd * sin(fpa) / (r * r);
auto fpadot = (Thrust * sin(alpha) + L) / (mass * v)
    + cos(fpa) * (v / r - mu_nd / (v * (r * r)));
auto mdot = (-1.0) * Thrust / vexhaust_nd;
return stack(hdot, vdot, fpadot, mdot);
```

Match the Python phase setup: LGL5 transcription, 50 segments, HighestOrderSpline control, adaptive mesh with tol 1e-7, 8 partitions.

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 minimum_time_to_climb_builder
./examples/cpp_examples/builder/minimum_time_to_climb/minimum_time_to_climb_builder
```

Expected: "PASSED", climb time ~321 s (matching Python reference).

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/minimum_time_to_climb/main.cpp
git commit -m "refactor: MinTimeToClimb uses InterpTable for aero/atmosphere data

Replace simplified polynomial atmosphere model with table-based
aero matching Python exactly. Uses interp() with InterpTable1D
for rho, sos, CLalpha, CD0, eta and InterpTable2D for Thrust."
```

---

## Task 4: Update MinTimeClimbTables Example

**Files:**
- Modify: `examples/cpp_examples/builder/min_time_climb_tables/main.cpp`

- [ ] **Step 1: Read current example**

Read `examples/cpp_examples/builder/min_time_climb_tables/main.cpp` — currently a placeholder documenting the gap.

- [ ] **Step 2: Replace with full table-based implementation**

This example should be a faithful port of `MinimumTimeToClimb.py` — same ODE, same tables, same solver setup. The ODE body is identical to Task 3. The main difference is this example is specifically about demonstrating table usage.

Update the example to:
1. Construct all 6 tables (5 x 1D + 1 x 2D) with exact Python data
2. Define ODE using `interp()`
3. Set up phase matching Python exactly
4. Verify convergence

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 min_time_climb_tables_builder
./examples/cpp_examples/builder/min_time_climb_tables/min_time_climb_tables_builder
```

Expected: "PASSED", climb time ~321 s.

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/min_time_climb_tables/main.cpp
git commit -m "refactor: MinTimeClimbTables uses real InterpTable1D/2D data

Replace polynomial workaround with full table-based aero matching
Python exactly. Demonstrates interp() with shared_ptr<InterpTable>
for atmospheric density, speed of sound, aero coefficients, and
2D thrust table."
```

---

## Task 5: Full Verification

- [ ] **Step 0: Triage BettsLowThrust and OptimalDocking**

Read these two examples and check if `InterpTable` (as opposed to `LGLInterpTable` from Subsystem 5) enables more faithful porting:
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp`
- `examples/cpp_examples/builder/optimal_docking/main.cpp`

Also check the Python versions:
- `examples/python_examples/BettsLowThrust.py` — does NOT use `InterpTable` (confirmed by grep)
- `examples/python_examples/OptimalDocking.py` — uses `LGLInterpTable` only (Subsystem 5)

If neither uses plain `InterpTable`, document this finding and skip. If either does, update accordingly.

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
cd /home/ghecht/Projects/tycho && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Expected: all 38 pass.

- [ ] **Step 4: Commit any remaining fixes**
