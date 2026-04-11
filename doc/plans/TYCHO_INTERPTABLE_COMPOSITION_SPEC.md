# Spec: InterpTable VF Composition (Subsystem 4)

> Expose `InterpTable1D` through `InterpTable4D` for composition with
> VectorFunction expressions in the C++ builder API, and modernize the
> interpolation type selection to use an enum.

## Motivation

`InterpTable1D/2D` (and 3D/4D) exist as C++ classes in `tycho::oc` with
full analytic derivative support via `InterpFunction1D/2D/3D/4D`. The
Python bindings expose call-syntax composition (`table(vf_expr)`) through
`__call__` lambdas that internally create `InterpFunction` and call
`.eval()`. However, this composition is not accessible from the C++
builder API — users cannot use interpolation tables inside ODE definitions.

This is a High-severity gap affecting four examples:
- **MinTimeToClimb** — atmospheric and aerodynamic data as functions of
  altitude and Mach number (5 x 1D tables + 1 x 2D table)
- **MinTimeClimbTables** — same tables, dedicated table-visualization example
- **BettsLowThrust** — potential use for RTN rotation or atmospheric data
- **OptimalDocking** — Form2 target trajectory (uses `LGLInterpTable` +
  `InterpFunction`, which is Subsystem 5; plain `InterpTable` may also
  be needed)

Additionally, the interpolation type is currently specified via strings
(`"cubic"`, `"linear"`) in C++ constructors. This should use an enum
class, with string parsing only in the Python binding layer (same pattern
as `IVPAlg` in Subsystem 2).

## Scope

This spec covers:
- `InterpType` enum class (replacing string-based kind selection)
- `interp()` free functions for VF composition (1D through 4D)
- Umbrella header additions
- Example updates for MinTimeToClimb and MinTimeClimbTables

This spec does NOT cover:
- `LGLInterpTable` / `InterpFunction` for trajectory interpolation (Subsystem 5)
- `make_periodic()` or STM integration (Subsystem 5)
- MEE astro ODE factories (Subsystem 6)

## Changes

### 1. InterpType Enum

New header: `include/tycho/detail/optimal_control/interp/interp_type.h`

Add a new enum class to the `tycho` namespace:

```cpp
namespace tycho {
enum class InterpType { Cubic, Linear };
}
```

This replaces the per-class `InterpType` enums currently nested inside
each `InterpTable` struct (e.g., `InterpTable1D::InterpType`). The nested
enums are removed, and all internal references are migrated. Every site
in `interp_table_1d.h` through `interp_table_4d.h` that uses
`InterpType::cubic_interp` or `InterpType::linear_interp` must be
rewritten to use `tycho::InterpType::Cubic` / `tycho::InterpType::Linear`
(or unqualified `InterpType::Cubic` within the `tycho` or `tycho::oc`
namespace). Member variables typed as the old nested enum
(`interp_kind_`) change to `tycho::InterpType`.

Each `InterpTable` header includes `interp_type.h`. Each constructor
gains an overload taking `InterpType` instead of `std::string`. The
string-based constructors remain for internal backward compatibility but
are no longer the primary C++ API.

**Python bindings:** Add `parse_interp_type()` helper (like
`parse_ivp_alg()` from Subsystem 2) to convert strings to the enum.
Binding `__init__` methods accept both `InterpType` and `str`. The
`InterpType` enum is also exposed to Python directly.

### 2. `interp()` Free Functions

New header: `include/tycho/detail/vf/operators/interp_compose.h`

Namespace: `tycho::vf`

Includes the four `interp_table_*.h` headers and provides `interp()`
overloads:

**Include dependencies:** `interp_compose.h` includes
`tycho/detail/optimal_control/interp/interp_table_1d.h` through
`interp_table_4d.h` (which provide `InterpTable*D` and
`InterpFunction*D` types). It also includes
`tycho/detail/vf/type_erasure/generic_function.h` and
`tycho/detail/vf/expressions/stacked.h` (for `stack()`).

**1D** — single VF input:
```cpp
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable1D>& table,
            const DenseFunctionBase<Func, IR, OR>& input)
    -> GenericFunction<-1, -1>;
```

Internally: `GenericFunction<-1, -1>(oc::InterpFunction1D<-1>(table).eval(input.derived()))`.

Note: the return type is always `GenericFunction<-1, -1>` (dynamic output
rows). The Python bindings branch on `table->vlen_` to return
`GenericFunction<-1, 1>` for scalar tables, but this is a Python-specific
optimization. In C++, `GenericFunction<-1, -1>` with runtime output
rows = 1 works identically for all downstream VF arithmetic — no
functional difference.

The table is passed as `shared_ptr` — the resulting expression tree
shares ownership. No copies of table data occur.

**2D** — two scalar VF inputs or one 2-element VF input:
```cpp
// Two separate scalar VF arguments
template <class F1, int IR1, int OR1, class F2, int IR2, int OR2>
auto interp(const std::shared_ptr<oc::InterpTable2D>& table,
            const DenseFunctionBase<F1, IR1, OR1>& x,
            const DenseFunctionBase<F2, IR2, OR2>& y)
    -> GenericFunction<-1, 1>;

// Single 2-element VF argument
template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable2D>& table,
            const DenseFunctionBase<Func, IR, OR>& xy)
    -> GenericFunction<-1, 1>;
```

Internally: `GenericFunction<-1, 1>(InterpFunction2D(table).eval(stack(x, y)))` or
`InterpFunction2D(table).eval(xy)`.

**3D** — three scalar VF inputs or one 3-element VF input:
```cpp
template <class F1, int IR1, int OR1,
          class F2, int IR2, int OR2,
          class F3, int IR3, int OR3>
auto interp(const std::shared_ptr<oc::InterpTable3D>& table,
            const DenseFunctionBase<F1, IR1, OR1>& x,
            const DenseFunctionBase<F2, IR2, OR2>& y,
            const DenseFunctionBase<F3, IR3, OR3>& z)
    -> GenericFunction<-1, 1>;

template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable3D>& table,
            const DenseFunctionBase<Func, IR, OR>& xyz)
    -> GenericFunction<-1, 1>;
```

**4D** — four scalar VF inputs or one 4-element VF input:
```cpp
template <class F1, int IR1, int OR1,
          class F2, int IR2, int OR2,
          class F3, int IR3, int OR3,
          class F4, int IR4, int OR4>
auto interp(const std::shared_ptr<oc::InterpTable4D>& table,
            const DenseFunctionBase<F1, IR1, OR1>& x,
            const DenseFunctionBase<F2, IR2, OR2>& y,
            const DenseFunctionBase<F3, IR3, OR3>& z,
            const DenseFunctionBase<F4, IR4, OR4>& w)
    -> GenericFunction<-1, 1>;

template <class Func, int IR, int OR>
auto interp(const std::shared_ptr<oc::InterpTable4D>& table,
            const DenseFunctionBase<Func, IR, OR>& xyzw)
    -> GenericFunction<-1, 1>;
```

All overloads follow the same pattern: create the corresponding
`InterpFunction` wrapper, `stack()` the scalar inputs if multiple, call
`.eval()`, wrap in `GenericFunction`. Note that `InterpFunction1D<ORR>`
is a class template (requires template argument `-1` for dynamic output),
while `InterpFunction2D`, `InterpFunction3D`, and `InterpFunction4D` are
non-template classes.

**Ownership:** The `shared_ptr` is shared (not copied) into the expression
tree via `InterpFunction`. The table data lives as long as any expression
referencing it. No copies of table data ever occur.

### 3. Umbrella Header Changes

Add to `include/tycho/optimal_control.h` (after the existing interp
includes but before the builder includes):
```cpp
#include "tycho/detail/optimal_control/interp/interp_table_1d.h"
#include "tycho/detail/optimal_control/interp/interp_table_2d.h"
#include "tycho/detail/optimal_control/interp/interp_table_3d.h"
#include "tycho/detail/optimal_control/interp/interp_table_4d.h"
#include "tycho/detail/vf/operators/interp_compose.h"
```

The `interp_compose.h` header goes in `optimal_control.h` (NOT
`vector_functions.h`) because it depends on the `oc::InterpTable*D` and
`oc::InterpFunction*D` types. Placing it in `vector_functions.h` would
create a layering violation (VF layer depending on OC layer).
`optimal_control.h` already includes `vector_functions.h`, so
`interp_compose.h` has access to all VF types it needs.

**Compile-time note:** `interp_table_3d.h` (~800 lines) and
`interp_table_4d.h` (~2950 lines) are large headers. Since they are
included unconditionally via the umbrella, this adds to compilation of
every TU using `tycho.h`. This is acceptable because: (a) the headers
are data-structure definitions, not heavy template instantiations; (b)
they are already included in the Python binding TUs; (c) the
`heavy_compile` ninja pool already manages the truly heavy TUs.

## Verification

Update C++ examples that currently use workarounds:

- **MinTimeClimbTables** — replace polynomial approximations with actual
  `InterpTable1D` (5 tables: rho, sos, CLalpha, CD0, eta) and
  `InterpTable2D` (1 table: Thrust vs Mach/altitude). Uses exact same
  data as Python. Should produce convergence matching Python.
- **MinTimeToClimb** — same ODE update as MinTimeClimbTables (they share
  the same aircraft model).
- **BettsLowThrust** — check if `InterpTable` enables more faithful
  porting beyond what was achieved with Subsystem 3. Update if applicable,
  skip if remaining gaps are Subsystem 5/6.
- **OptimalDocking** — check if plain `InterpTable` is needed beyond the
  `LGLInterpTable` usage (Subsystem 5). Update if applicable.

**Success criteria per example:**
1. Uses `interp()` with real table data matching the Python version
2. Builds and runs, prints "PASSED"
3. Convergence behavior matches Python (same objective, similar iteration
   count)

**No regressions:**
- All existing C++ unit tests pass
- All C++ example CTests pass
- All Python examples pass

## Files Modified

**New:**
- `include/tycho/detail/optimal_control/interp/interp_type.h` —
  `tycho::InterpType` enum class (`Cubic`, `Linear`)
- `include/tycho/detail/vf/operators/interp_compose.h` — `interp()` free
  functions for 1D-4D in `tycho::vf` namespace

**Modified (enum migration + constructor overloads):**
- `include/tycho/detail/optimal_control/interp/interp_table_1d.h` —
  remove nested `InterpType`, include `interp_type.h`, migrate all
  `InterpType::cubic_interp`/`linear_interp` to `InterpType::Cubic`/
  `Linear`, add enum-based constructor overload
- `include/tycho/detail/optimal_control/interp/interp_table_2d.h` — same
- `include/tycho/detail/optimal_control/interp/interp_table_3d.h` — same
- `include/tycho/detail/optimal_control/interp/interp_table_4d.h` — same

**Modified (umbrella):**
- `include/tycho/optimal_control.h` — add interp_table and
  interp_compose includes

**Modified (Python bindings):**
- `src/bindings/vf/interp_table_bind.h` — `parse_interp_type()` helper,
  dual overloads (enum + string) for all constructors, expose
  `InterpType` enum to Python

**Modified (examples):**
- `examples/cpp_examples/builder/min_time_climb_tables/main.cpp` — full
  table-based aero using `interp()` with `InterpTable1D` and
  `InterpTable2D`
- `examples/cpp_examples/builder/min_time_to_climb/main.cpp` — same ODE
  update (shares aircraft model with MinTimeClimbTables)
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp` — if applicable
- `examples/cpp_examples/builder/optimal_docking/main.cpp` — if applicable

**C++ table construction pattern** (for reference in examples):
```cpp
// Python: rhoTab = vf.InterpTable1D(alts, rhos, kind="cubic")
// C++:
auto rhoTab = std::make_shared<InterpTable1D>(
    alts_vec, rhos_vec, 0, InterpType::Cubic);
//  ~~~~~~~~  ~~~~~~~~  ~  ~~~~~~~~~~~~~~~~
//  Eigen::VectorXd      axis  enum (not string)

// Python: ThrustTab = vf.InterpTable2D(ThrustMach, ThrustAlt, ThrustData, kind="cubic")
// C++:
auto thrustTab = std::make_shared<InterpTable2D>(
    mach_vec, alt_vec, data_mat, InterpType::Cubic);
//  ~~~~~~~~  ~~~~~~~  ~~~~~~~~  ~~~~~~~~~~~~~~~~
//  VectorXd  VectorXd  RowMajor MatrixXd  enum
```
