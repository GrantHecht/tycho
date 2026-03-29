# Binding Layer Developer Guide

This document provides a comprehensive, bottom-up explanation of Tycho's Python binding layer -- the system that maps every C++ `tycho::` type into the `_tychopy` Python extension module. After reading this guide, you should be able to understand, use, and extend the binding layer at every level: from the nanobind module entry point, through the `TychoBind<T>` trait dispatch, to the type casters that translate Python objects into Eigen vectors, and the pure-Python wrapper layer that smooths over remaining ergonomic gaps.

This guide assumes familiarity with the [VectorFunction Developer Guide](VectorFunction.md). The binding layer is the bridge between that C++ system and the Python API users interact with.

## Table of Contents

1. [Overview -- What the Binding Layer Does](#1-overview----what-the-binding-layer-does)
2. [Module Structure](#2-module-structure)
3. [The `TychoBind<T>` Trait Pattern](#3-the-tychobindt-trait-pattern)
4. [`bind::` Free-Function Helpers](#4-bind-free-function-helpers)
5. [Type Casters (`type_casters.h`)](#5-type-casters-type_castersh)
6. [Precompiled Header Setup](#6-precompiled-header-setup)
7. [File Organization](#7-file-organization)
8. [Build System Integration](#8-build-system-integration)
9. [Python-Side Wrappers (`tychopy/`)](#9-python-side-wrappers-tychopy)
10. [Key Patterns and Conventions](#10-key-patterns-and-conventions)
11. [Adding a New Binding (Step-by-Step)](#11-adding-a-new-binding-step-by-step)
12. [Reference: Complete File Listing](#12-reference-complete-file-listing)

---

## 1. Overview -- What the Binding Layer Does

The binding layer is the translation layer between Tycho's C++ core and its Python API. It has three responsibilities:

1. **Map C++ types to Python classes** -- every `VectorFunction`, ODE, `Phase`, `Integrator`, and solver type visible in Python is bound in `src/bindings/`.
2. **Convert Python objects to C++ types** -- custom type casters handle `numpy.ndarray <-> Eigen::VectorXd`, `list/tuple -> VectorXd`, `None -> "none"` sentinel, and variant dispatch.
3. **Provide ergonomic wrappers** -- the pure-Python `tychopy/` package re-exports the C++ bindings and adds thin wrappers to coerce Python types that nanobind cannot handle natively (e.g., `list` to `numpy.ndarray` for fixed-size Eigen types).

### nanobind, not pybind11

Tycho uses [nanobind](https://github.com/wjakob/nanobind), not pybind11. The migration (PR 3) was motivated by:

- **Smaller binary size** -- nanobind produces significantly smaller `.so` files.
- **Faster compile times** -- less template bloat.
- **Better ndarray support** -- first-class `nb::ndarray` with dtype/device/contiguity constraints.
- **Active development** -- nanobind is the successor project by the same author.

Key API differences from pybind11 that affect the binding code:

| pybind11 | nanobind |
|---|---|
| `PYBIND11_MODULE(name, m)` | `NB_MODULE(name, m)` |
| `py::class_<T>` | `nb::class_<T>` |
| `py::init<>()` | `nb::init<>()` |
| `py::arg(...)` | `nb::arg(...)` (no implicit None; use `.none()` explicitly) |
| `py::return_value_policy::reference_internal` | `nb::rv_policy::reference_internal` (rejects ndarrays without owner) |
| Template partial specialization for type casters | Must use explicit full specializations (nanobind's Eigen partial spec conflicts) |

### Two-Layer Architecture

```
+-----------------------------+
|  Python user code           |
+-----------------------------+
|  tychopy/  (pure Python)    |  <-- Wrappers, re-exports, ODEBase class
+-----------------------------+
|  _tychopy  (nanobind C++)   |  <-- All C++ bindings live here
+-----------------------------+
|  tycho:: C++ core           |  <-- No nanobind code
+-----------------------------+
```

**Strict separation rule:** Core C++ source under `src/vf/`, `src/optimal_control/`, `src/integrators/`, `src/astro/`, `src/solvers/`, and `src/utils/` contains **zero** nanobind code. All binding code lives exclusively in `src/bindings/`.

---

## 2. Module Structure

### Entry Point

The module is defined in `src/bindings/tycho_module.cpp`:

```cpp
NB_MODULE(_tychopy, m) {
    m.doc() = "Tycho";
    m.def("PyMain", &main);
    m.def("SoftwareInfo", &SoftwareInfo);

    FunctionRegistry reg(m);     // Must be built first
    VectorFunctionBuild(reg, m); // Must be built second
    SolversBuild(reg, m);        // Third: PSIOPT shows up better in autocomplete
    OptimalControlBuild(reg, m);
    UtilsBuild(m);
    AstroBuild(reg, m);

    ExtensionsBuild(reg, reg.extmod);
}
```

**Build order matters.** The `FunctionRegistry` must be constructed first because it creates the `VectorFunction` and `ScalarFunction` type-erased base classes that all subsequent registrations reference. `VectorFunctionBuild` must come second because it registers the fundamental types (`Arguments`, `Segment`, `Element`, etc.) that ODE and solver bindings depend on.

### Submodules

The `FunctionRegistry` constructor creates four submodules:

| Python path | C++ variable | Purpose |
|---|---|---|
| `_tychopy.VectorFunctions` | `reg.vfmod` | Vector/scalar function types and free functions |
| `_tychopy.OptimalControl` | `reg.ocmod` | ODEs, phases, transcription modes, link functions |
| `_tychopy.Solvers` | `reg.solmod` | PSIOPT, NLP, solver flags |
| `_tychopy.Extensions` | `reg.extmod` | User-defined extension types |

The `Astro` submodule is created inside `AstroBuild()` rather than in the registry constructor.

### The `FunctionRegistry` Struct

Defined in `src/bindings/function_registry.h`, `FunctionRegistry` is the central state object passed through the build pipeline. It holds:

- **Module references** -- `mod` (root), `vfmod`, `ocmod`, `solmod`, `extmod`.
- **Base class objects** -- `vfuncx` (`nb::class_<VectorFunctionalX>`) and `sfuncx` (`nb::class_<ScalarFunctionalX>`), the type-erased `GenericFunction<-1,-1>` and `GenericFunction<-1,1>` wrappers.
- **Registration methods** -- `Build_Register<T>()` which calls `TychoBind<T>::Build()` then registers implicit conversions.

```cpp
struct FunctionRegistry {
    using VectorFunctionalX = GenericFunction<-1, -1>;
    using ScalarFunctionalX = GenericFunction<-1, 1>;

    nb::module_ &mod;
    nb::module_ vfmod, ocmod, solmod, extmod;
    nb::class_<VectorFunctionalX> vfuncx;
    nb::class_<ScalarFunctionalX> sfuncx;

    template <class Derived> void Build_Register(nb::module_ &m, const char *name);
    template <class Derived> void Register();
    // ...
};
```

---

## 3. The `TychoBind<T>` Trait Pattern

The binding layer uses a **compile-time trait pattern** to dispatch binding construction for each C++ type. This is the central mechanism that connects the registration pipeline to per-type binding definitions.

### Primary Template

Declared in `src/bindings/function_registry.h`:

```cpp
namespace tycho {
// Primary template -- undefined; specializations in *_bind.h files
template <class T> struct TychoBind;
}
```

The primary template is deliberately left undefined. Any attempt to bind a type without a specialization produces a compile error -- this is intentional, as it catches missing bindings at build time.

**Note:** The primary template is NOT guarded by `#ifdef TYCHO_PYTHON_BINDINGS`. It is a pure C++ trait with no nanobind dependency. Only the `FunctionRegistry` struct and `FuncPack` helper (which use `nb::` types) are guarded.

### Specializations

Full and partial specializations are defined in `*_bind.h` headers throughout `src/bindings/`:

```cpp
// Full specialization (src/bindings/optimal_control/ode_phase_bind.h)
template <> struct TychoBind<ODEPhaseBase> {
    static void Build(nb::module_ &m);
};

// Partial specialization (same file)
template <class DODE> struct TychoBind<ODEPhase<DODE>> {
    static void Build(nb::module_ &m) {
        auto phase = nb::class_<ODEPhase<DODE>, ODEPhaseBase>(m, "phase");
        bind::ODEPhaseBuildImpl<DODE>(phase);
        // ...
    }
};

// Partial specialization for VectorFunction types (CommonFunctionsBind.h)
template <int IR, int OR, int ST> struct TychoBind<Segment<IR, OR, ST>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Segment<IR, OR, ST>>(m, name);
        bind::DenseBaseBuild<Segment<IR, OR, ST>>(obj);
        bind::SegBuild<Segment<IR, OR, ST>>(obj);
    }
};
```

### `Build()` Signatures

Two signatures are used depending on whether the type's Python name is fixed or caller-determined:

```cpp
static void Build(nb::module_ &m);                    // Name is hardcoded inside
static void Build(nb::module_ &m, const char *name);  // Name provided by caller
```

### `FunctionRegistry::Build_Register<T>()`

This is the primary entry point for registering VectorFunction types. It:

1. Calls `TychoBind<T>::Build(m, name)` to create the `nb::class_<T>` and define its methods.
2. Calls `RegSelector<T::IRC, T::ORC>::Register<T>(this)` to register implicit conversions to the type-erased base classes.

```cpp
template <class Derived> void Build_Register(nb::module_ &m, const char *name) {
    TychoBind<Derived>::Build(m, name);
    RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
}
```

### `RegSelector<IR, OR>` -- Implicit Conversion Registration

The `RegSelector` partial specialization on `OR` controls which type-erased base classes a type can convert to:

- **`OR != 1`** (vector output): registers implicit conversion to `VectorFunctionalX` only.
- **`OR == 1`** (scalar output): registers implicit conversion to **both** `VectorFunctionalX` and `ScalarFunctionalX`.

```cpp
template <int IR, int OR> struct RegSelector {
    template <class Derived> static void Register(FunctionRegistry *reg) {
        nb::implicitly_convertible<Derived, VectorFunctionalX>();
        reg->vfuncx.def(nb::init<Derived>());
    }
};

template <int IR> struct RegSelector<IR, 1> {
    template <class Derived> static void Register(FunctionRegistry *reg) {
        nb::implicitly_convertible<Derived, VectorFunctionalX>();
        reg->vfuncx.def(nb::init<Derived>());
        nb::implicitly_convertible<Derived, ScalarFunctionalX>();
        reg->sfuncx.def(nb::init<Derived>());
    }
};
```

This is how Python code like `phase.add_equal_con(my_segment_func)` works -- the `Segment` is implicitly converted to a `VectorFunction` (the type-erased `GenericFunction<-1,-1>`).

---

## 4. `bind::` Free-Function Helpers

The `bind::` namespace (inside `tycho::bind`) contains reusable free-function templates that register common method sets on `nb::class_` objects. These avoid duplicating binding code across the many VectorFunction types.

### `DenseBaseBuild<Derived>(obj)` -- Core Methods

Defined in `src/bindings/vf/dense_function_base_bind.h`. Registers the fundamental methods that every VectorFunction type exposes to Python:

| Method | Python signature | Description |
|---|---|---|
| `compute(x)` | `f(numpy.ndarray) -> numpy.ndarray` | Evaluate f(x) |
| `__call__(x)` | `f(x)` | Alias for compute |
| `jacobian(x)` | `f.jacobian(x) -> numpy.ndarray` | Jacobian matrix J(x) |
| `compute_jacobian(x)` | `-> (fx, Jx)` | Value and Jacobian together |
| `adjointgradient(x, lm)` | `-> numpy.ndarray` | J^T * lambda |
| `adjointhessian(x, lm)` | `-> numpy.ndarray` | Adjoint Hessian |
| `computeall(x, lm)` | `-> (fx, Jx, gx, Hx)` | All derivatives at once |
| `IRows()` | `-> int` | Input dimension |
| `ORows()` | `-> int` | Output dimension |
| `name()` | `-> str` | Type name string |
| `input_domain()` | `-> ...` | Domain information |
| `is_linear()` | `-> bool` | Linearity flag |
| `rpt(...)` | | Repeat/tile |
| `vf()` | `-> VectorFunction` | Convert to type-erased VectorFunction |
| `sf()` | `-> ScalarFunction` | Convert to ScalarFunction (OR=1 only) |

**Dual overload pattern:** Each method accepting vector input is registered twice:

```cpp
// Overload 1: zero-copy numpy fast path
obj.def("compute",
    [=](const Derived &f, ConstEigenRef<VectorXd> x) { return compute_body(f, x); });

// Overload 2: Python sequence fallback (list, tuple, range)
obj.def("compute",
    [=](const Derived &f, VectorXd x) { return compute_body(f, x); });
```

The `ConstEigenRef<VectorXd>` overload uses nanobind's built-in Eigen caster for zero-copy numpy access. The `VectorXd` (by-value) overload triggers our custom type caster (see [Section 5](#5-type-casters-type_castersh)) which accepts lists, tuples, and other sequences. Without both overloads, either numpy calls would copy or list/tuple calls would fail.

### Math Builder Hierarchy

These helpers build the Python operator overloads and math methods for VectorFunction types. They are designed to compose via `SegBuild`:

| Helper | What it registers |
|---|---|
| `DoubleMathBuild<T>(obj)` | `__add__`/`__sub__` with VectorXd, `__mul__`/`__truediv__` with double, `__neg__`, `__array_ufunc__ = None` |
| `UnaryMathBuild<T>(obj)` | `norm`, `squared_norm`, `normalized`, `sin`, `cos`, `tan`, `sqrt`, `exp`, `log`, `squared`, `pow`, `arcsin`/`arccos`/`arctan`, `sinh`/`cosh`/`tanh`, `__abs__`, `sign` |
| `BinaryMathBuild<T>(obj)` | `cross`, `dot`, `cwiseProduct`, `cwiseQuotient` (with other VF types and VectorXd constants) |
| `BinaryOperatorsBuild<T>(obj)` | `eval`, `apply`, `__add__`/`__sub__`/`__mul__`/`__truediv__` with other VF types |
| `FunctionIndexingBuild<T>(obj)` | `segment`, `head`, `tail`, `coeff`, `__getitem__` (int and slice), `padded_lower`/`padded_upper`/`padded`, fixed-size `segment_2`/`segment_3` aliases |
| `ConditionalOperatorsBuild<T>(obj)` | `__lt__`, `__gt__`, `__le__`, `__ge__` (returns `GenericConditional`) |

### `SegBuild<Derived>(obj)` -- The All-in-One Combiner

Defined in `src/bindings/vf/common_functions_bind.h`, `SegBuild` calls all the math builders in sequence:

```cpp
template <class Derived, class PyClass> void SegBuild(PyClass &obj) {
    bind::DoubleMathBuild<Derived>(obj);
    bind::UnaryMathBuild<Derived>(obj);
    bind::BinaryMathBuild<Derived>(obj);
    bind::BinaryOperatorsBuild<Derived>(obj);
    bind::FunctionIndexingBuild<Derived>(obj);
    bind::ConditionalOperatorsBuild<Derived>(obj);

    obj.def("tolist", ...);  // Convert to list of Element/Segment objects
}
```

Types that need the full VectorFunction Python API (like `Arguments`, `Segment`, `Element`) call `SegBuild`. More specialized types may call only `DenseBaseBuild`.

### `GenericBuild<Derived>(obj)`

For `GenericFunction<IR,OR>` types (the type-erased wrappers). Registers init-from-self, `SuperTest`, `SpeedTest`, and then calls `DenseBaseBuild`.

### ODE and Phase Builders

| Helper | Purpose |
|---|---|
| `ODEPhaseBuildImpl<DODE>(phase)` | Registers all `ODEPhase` constructors (TranscriptionModes, string mode, trajectory data) |
| `IntegratorBuildConstructors<DODE>(obj)` | Adds `.integrator(...)` factory methods to an ODE class |
| `ODESizeBuild<XV,UV,PV,Derived>(obj)` | Registers ODE dimension accessors: `XVars()`, `UVars()`, `PVars()`, `TVar()`, index helpers |
| `BuildGenODEModule<BaseType,XV,UV,PV>(name, mod, reg)` | Creates a complete ODE submodule with `ode`, `integrator`, `phase` classes |
| `BuildODEModule<BaseType,Derived,XV,UV,PV>(name, mod, reg)` | Same but for concrete (non-generic) ODE types |
| `ODE_ExpressionBuild<Derived,ExprImpl,Ts...>(m, name)` | Registers an ODE expression type with DenseBaseBuild + phase factory + integrator constructors |

### Conditional and Comparative Builders

| Helper | Purpose |
|---|---|
| `ComparativeBuild(m)` | Registers `Comparative` class with `compute` and `MinMaxBuild` methods |
| `MinMaxBuild(obj)` | Registers `max`/`min` free functions for scalar and vector generic functions |
| `ConditionalBuild(m)` | Registers `Conditional` class with `compute`, `__and__`, `__or__`, and `IfElseBuild` |
| `IfElseBuild(obj)` | Registers `ifelse(test, true_func, false_func)` overloads |

---

## 5. Type Casters (`type_casters.h`)

Custom type casters in `src/bindings/type_casters.h` handle the automatic conversion between Python objects and C++ types that nanobind cannot handle natively.

### Why Explicit Full Specializations?

nanobind ships with a partial specialization for all `Eigen::Matrix<Scalar, Rows, Cols>` types. A partial specialization for `Eigen::VectorXd` (i.e., `Eigen::Matrix<double, -1, 1>`) would be ambiguous with nanobind's own. The solution is explicit **full** specializations, which take priority over any partial specialization:

```cpp
template <> struct type_caster<Eigen::VectorXd> { ... };   // Full spec -- wins
template <> struct type_caster<Eigen::VectorXi> { ... };   // Full spec -- wins
```

### Include Ordering Requirement

`type_casters.h` **must** be included **before** any nanobind STL caster headers (`stl/vector.h`, `stl/variant.h`, etc.). Explicit specializations must be visible before the compiler implicitly instantiates the corresponding partial specializations from the STL casters. Violating this rule is ill-formed per C++14 [temp.expl.spec]/6.

This ordering is enforced in `pch_nb.h`:
```cpp
#include "type_casters.h"       // BEFORE stl/ casters
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>
// ...
```

### Caster Reference

#### `Eigen::Tensor<Scalar, N>` -- numpy ndarray

Converts between C-contiguous numpy arrays and `Eigen::Tensor`. **Always copies** in both directions (Eigen::Tensor owns its buffer).

```
Python numpy.ndarray (C-contiguous, CPU) <-> Eigen::Tensor<Scalar, N>
```

#### `Eigen::VectorXd` -- numpy array + sequence fallback

The most important caster. Two-stage `from_python`:

1. **numpy fast path** -- uses `nb::ndarray<double, c_contig, cpu>`. Zero-copy reference to the numpy buffer (but our caster copies into a VectorXd for ownership safety).
2. **Sequence fallback** -- if not a numpy array, tries `PySequence_Fast()` to handle `list`, `tuple`, `range`, and other sequence types. Converts each element via `PyNumber_Float()`.

`from_cpp` **always copies** -- nanobind cannot safely share an Eigen vector's internal buffer with Python's GC (no owner object to prevent deallocation).

```
Python numpy.ndarray[float64] -> VectorXd  (memcpy)
Python list/tuple of numbers  -> VectorXd  (element-wise conversion)
VectorXd -> numpy.ndarray[float64]         (always copies)
```

#### `Eigen::VectorXi` -- numpy int32 array + sequence fallback

Same pattern as VectorXd but for `int32_t`:

```
Python numpy.ndarray[int32] -> VectorXi
Python list/tuple of ints   -> VectorXi
VectorXi -> numpy.ndarray[int32]
```

#### `VarIndexType` -- `std::variant<int, VectorXi, string, vector<string>>`

Used for indexing operations. Priority order in `from_python`:

1. **`int`** -- `PyLong_Check` -> `int`
2. **`str`** -- `PyUnicode_Check` -> `std::string`
3. **Sequence** -- `PySequence_Fast`, then check first element:
   - If first element is `str` -> `std::vector<std::string>`
   - Otherwise try as ints -> `Eigen::VectorXi`

```
Python int                     -> int
Python str                     -> std::string
Python list[str]               -> std::vector<std::string>
Python list[int] / range / ... -> Eigen::VectorXi
```

#### `std::vector<Eigen::VectorXd>` -- list of arrays

Converts a Python list of numpy arrays (or lists of numbers) to a C++ vector of VectorXd. Each element goes through the VectorXd caster individually.

```
Python list[numpy.ndarray]     -> std::vector<VectorXd>
Python list[list[float]]       -> std::vector<VectorXd>
```

#### `std::variant<double, Eigen::VectorXd>` -- scalar or vector

Used for boundary value parameters. Priority order:

1. **`float`** -- `PyFloat_Check` -> `double`
2. **`int`** -- `PyLong_Check` -> `double` (cast)
3. **numpy array or sequence** -> `Eigen::VectorXd`

```
Python float/int               -> double
Python numpy.ndarray           -> Eigen::VectorXd
Python list/tuple of numbers   -> Eigen::VectorXd
```

#### `ScaleType` -- `std::variant<double, VectorXd, string>`

Used for auto-scaling parameters. Has special `None` handling:

```
Python None                    -> std::string("none")    # disables auto-scaling
Python str                     -> std::string
Python float/int               -> double
Python numpy.ndarray           -> Eigen::VectorXd
```

The `None -> "none"` sentinel conversion is why all `AutoScale` parameters use `nb::arg(...).none()` -- without `.none()`, nanobind rejects `None` before the caster sees it.

---

## 6. Precompiled Header Setup

The binding layer uses a dedicated precompiled header (PCH) chain to avoid recompiling nanobind, Eigen, and the VectorFunction templates in every `.cpp` file.

### PCH Chain

The `pch_bindings` static library defines the PCH order:

```
src/pch.h                              # STL headers, Eigen, fmt, autodiff
  -> src/bindings/pch_nb.h             # nanobind headers, type_casters.h, function_registry.h, jet_bind.h
    -> src/vf/tycho_vector_functions.h # All VF type definitions
      -> src/bindings/vector_functions_bind.h        # VF binding helpers
```

**Why this order?**

- `pch.h` provides all standard library and Eigen types.
- `pch_nb.h` brings in nanobind and the type casters. `type_casters.h` must come before STL casters (see [Section 5](#5-type-casters-type_castersh)).
- `tycho_vector_functions.h` defines all VectorFunction types (GenericFunction, Segment, etc.) that the binding headers need.
- `vector_functions_bind.h` aggregates the VF binding headers that depend on all of the above.

### `pch_nb.h` Contents

```cpp
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;

#include "type_casters.h"             // BEFORE stl/ casters

#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "function_registry.h"        // TychoBind<T> primary template
#include "solvers/jet_bind.h"         // JetInvoker partial spec for nb::args
```

### `TYCHO_PYTHON_BINDINGS` Macro Scope

The `TYCHO_PYTHON_BINDINGS` preprocessor macro is defined **only** for:

- `pch_bindings` static library
- `_tychopy` nanobind module
- `tycho_extensions` module

It is **not** defined globally. This ensures that core C++ headers never see nanobind types. All `*_bind.h` files are guarded with `#ifdef TYCHO_PYTHON_BINDINGS`, making them safe to include from aggregate headers -- the guarded content is simply empty when compiled without the macro.

### `_tychopy` Reuses `pch_bindings`

The `_tychopy` target uses `REUSE_FROM pch_bindings` to share the precompiled header:

```cmake
target_precompile_headers(_tychopy REUSE_FROM pch_bindings)
```

This means `_tychopy` TUs must use **exactly the same compile flags** as `pch_bindings` to avoid PCH mismatches. Both use `BINDING_COMPILE_FLAGS`.

---

## 7. File Organization

### Directory Structure

`src/bindings/` mirrors the structure of `src/` with matching subdirectories:

```
src/bindings/
  CMakeLists.txt                  Build rules for pch_bindings + _tychopy
  tycho_module.cpp                NB_MODULE entry point
  function_registry.h             TychoBind<T> primary template, FunctionRegistry struct
  type_casters.h                  Custom nanobind type casters (Eigen, variants)
  pch_nb.h                        Binding-specific PCH (nanobind + type casters)
  vector_functions_bind.h         Aggregate: includes all VF binding headers
  utils_bind.cpp                  Thread pool, timer, memory utilities
  bump_allocator_bind.cpp/.h      Bump allocator / memory manager binding
  vf/
    dense_function_base_bind.h    Core binding helpers: DenseBaseBuild, math builders, SegBuild
    common_functions_bind.h       TychoBind<> for Segment, Arguments, Constant, etc.
    generic_function_bind.h       GenericBuild, ComparativeBuild, ConditionalBuild, IfElseBuild
    vector_function_bind.h        TychoBind<> for VectorFunction base types
    interp_table_bind.h           InterpTable1D/2D/3D/4D build functions
    python_functions.h            PyVectorFunction/PyScalarFunction (Python-callable VFs)
    python_arg_parsing.h          ParsePythonArgs, ParsePythonArgsScalar declarations
    python_arg_parsing.cpp        ParsePythonArgs/ParsePythonArgsScalar implementations
    tycho_vector_functions.cpp    VectorFunctionBuild() orchestrator
    vector_function_build_part1.cpp  VF type registrations (part 1)
    vector_function_build_part2.cpp  VF type registrations (part 2)
    args_seg_build_part1..5.cpp   Arguments/Segment instantiations (split for compile time)
    bulk_operations_build.cpp     Stack, Sum, SumElems, etc.
    free_functions_build.cpp      Module-level free functions (cross, dot, etc.)
    matrix_function_build.cpp     ColMatrix, RowMatrix types
  optimal_control/
    ode_phase_bind.h              TychoBind<> for ODEPhase, StateFunction, LinkFunction, etc.
    ode_bind.h                    ODE_ExpressionBuild, BuildODEModule, BuildGenODEModule
    ode_sizes_bind.h              ODESizeBuild<XV,UV,PV,Derived> helper
    ode_arguments_bind.h          ODEArguments binding
    mesh_iterate_info_bind.h/.cpp MeshIterateInfo binding
    tycho_optimal_control.cpp     OptimalControlBuild() orchestrator
    generic_odes_build_part1..6.cpp  GenericODE instantiations (split for compile time)
    ode_phase_base_bind.cpp       Out-of-line TychoBind<ODEPhaseBase>::Build
    optimal_control_problem_bind.cpp Out-of-line TychoBind<OptimalControlProblem>::Build
    lgl_interp_table_bind.cpp     Out-of-line TychoBind<LGLInterpTable>::Build
  solvers/
    jet_bind.h/.cpp               TychoBind<Jet>, JetInvoker partial spec for nb::args
    psiopt_bind.h/.cpp            TychoBind<PSIOPT>
    optimization_problem_bind.h/.cpp TychoBind<OptimizationProblem>
    optimization_problem_base.cpp OptimizationProblemBase binding
    tycho_solvers.cpp             SolversBuild() orchestrator
  astro/
    tycho_astro.cpp               AstroBuild() orchestrator
    kepler_utils.cpp              Kepler utility function bindings
    kepler_model.cpp              Kepler ODE model bindings
    lambert_solvers_bind.cpp      Lambert solver bindings
  integrators/
    integrator_bind.h             TychoBind<Integrator<DODE>>, IntegratorBuildConstructors
    rk_coeffs_bind.cpp            Runge-Kutta coefficient table bindings
```

### `*_bind.h` vs `*_bind.cpp`

- **`*_bind.h` headers** -- contain template code (TychoBind partial specializations, bind:: helpers). Included via the PCH chain so they are available in every binding TU without explicit `#include`.
- **`*_bind.cpp` files** -- contain out-of-line implementations for non-template specializations and the heavy registration functions that would bloat the PCH if included as headers.

The split is driven by compile-time performance: template code must be in headers, but large non-template binding functions should be in `.cpp` files.

---

## 8. Build System Integration

### `src/bindings/CMakeLists.txt`

This file defines two targets:

1. **`pch_bindings`** -- a static library that serves as the PCH donor:

```cmake
add_library(pch_bindings STATIC ${CMAKE_SOURCE_DIR}/src/pch.cpp)
target_precompile_headers(pch_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pch_nb.h
    ${CMAKE_SOURCE_DIR}/src/vf/tycho_vector_functions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_functions_bind.h
)
target_compile_definitions(pch_bindings PRIVATE TYCHO_PYTHON_BINDINGS)
target_compile_options(pch_bindings PUBLIC ${BINDING_COMPILE_FLAGS})
target_link_libraries(pch_bindings PRIVATE nanobind-static)
```

2. **`_tychopy`** -- the nanobind module listing all `.cpp` source files:

```cmake
nanobind_add_module(_tychopy
    tycho_module.cpp
    utils_bind.cpp
    bump_allocator_bind.cpp
    vf/tycho_vector_functions.cpp
    # ... all .cpp files listed explicitly
)
target_compile_options(_tychopy PRIVATE ${BINDING_COMPILE_FLAGS})
target_compile_definitions(_tychopy PUBLIC TYCHO_PYTHON_BINDINGS)
target_precompile_headers(_tychopy REUSE_FROM pch_bindings)
```

### `BINDING_COMPILE_FLAGS`

By default, binding code compiles with `-Os` (optimize for size) instead of `-O3`. This follows nanobind best practice -- binding glue code benefits more from smaller binary size than aggressive optimization.

```cmake
if(OPTIMIZE_BINDINGS)
    set(BINDING_COMPILE_FLAGS ${COMPILE_FLAGS})           # Keep -O3
else()
    string(REPLACE "-O3" "-Os" BINDING_COMPILE_FLAGS "${COMPILE_FLAGS}")  # -Os
endif()
```

Set `OPTIMIZE_BINDINGS=ON` if you need `-O3` for profiling or if binding dispatch overhead is measurable.

### Build Performance Features

| Feature | CMake option | Effect |
|---|---|---|
| Size-optimized bindings | `OPTIMIZE_BINDINGS=OFF` (default) | `-Os` for binding code, `-O3` for core |
| Build profiling | `COMPILE_TIME_TRACE=ON` | Enables Clang `-ftime-trace` |
| Compiler cache | Automatic | `ccache` auto-detected via `find_program` |
| Split TUs | N/A (hardcoded) | Heavy templates split across Part1..N files |

The `ArgsSegBuildPart1..5.cpp` and `GenericODESBuildPart1..6.cpp` files exist purely to split template-heavy instantiations across multiple translation units, reducing peak memory usage and enabling parallel compilation.

### Install Targets

The build installs two things into the Python site-packages directory:

1. **`_tychopy.cpython-<ver>-darwin.so`** -- the compiled extension module
2. **`tychopy/`** -- the entire pure-Python package directory (copied fresh each build)

```cmake
add_custom_command(TARGET _tychopy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_tychopy> ${PYTHON_LOCAL_INSTALL_DIR}/
)
add_custom_command(TARGET pytychosrc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/tychopy
                                                ${PYTHON_LOCAL_INSTALL_DIR}/tychopy
)
```

`PYTHON_LOCAL_INSTALL_DIR` is resolved from `python -m site --user-site` or from `Python_SITEARCH` (Windows). The `config_and_build.sh` script sets this automatically from the `tycho` conda environment.

---

## 9. Python-Side Wrappers (`tychopy/`)

The `tychopy/` package provides a pure-Python layer over the `_tychopy` extension module. Its main roles are:

1. **Re-export** -- make C++ types available as `tychopy.VectorFunctions.Arguments` instead of `_tychopy.VectorFunctions.Arguments`.
2. **Coerce types** -- wrap functions that accept fixed-size Eigen types (e.g., `Vector6<double>`) to also accept Python lists and tuples.
3. **Add pure-Python extensions** -- like `ODEBase`, `CR3BPFrame`, and mesh error plotting.

### `tychopy/__init__.py`

Imports `_tychopy` and each subpackage:

```python
import _tychopy as _tychopy
import tychopy.Astro
import tychopy.OptimalControl
import tychopy.Solvers
import tychopy.Utils
import tychopy.VectorFunctions
```

### `tychopy/VectorFunctions/__init__.py` -- `cross` wrapper

The C++ `cross(a, b)` function expects VectorFunction types. When the first argument is a Python list or tuple (a constant vector), nanobind's fixed-size Eigen caster may reject it. The wrapper coerces it:

```python
_cross_cpp = _tychopy.VectorFunctions.cross

def cross(a, b):
    if isinstance(a, (list, tuple)):
        a = np.asarray(a, dtype=np.float64)
    return _cross_cpp(a, b)
```

### `tychopy/Astro/__init__.py` -- `_vec6_wrap`

Many astrodynamics functions take `Vector6<double>` as their first argument. The `_vec6_wrap` decorator coerces lists/tuples to numpy arrays while passing VectorFunction objects through unchanged:

```python
def _vec6_wrap(fn):
    def wrapper(arr_or_func, *args):
        if hasattr(arr_or_func, "eval"):  # VectorFunction -- pass through
            return fn(arr_or_func, *args)
        return fn(np.asarray(arr_or_func, dtype=np.float64), *args)
    wrapper.__name__ = fn.__name__
    return wrapper

cartesian_to_classic = _vec6_wrap(_tychopy.Astro.cartesian_to_classic)
classic_to_cartesian = _vec6_wrap(_tychopy.Astro.classic_to_cartesian)
# ... etc.
```

### `tychopy/OptimalControl/__init__.py` -- `ODEBase`

Re-exports all C++ optimal control types and adds the pure-Python `ODEBase` class (from `ODEBaseClass.py`), which provides a Python-friendly base for defining ODEs with overridable `compute` methods.

---

## 10. Key Patterns and Conventions

### `#ifdef TYCHO_PYTHON_BINDINGS` Guard

Every `*_bind.h` file is wrapped in this guard:

```cpp
#pragma once
#ifdef TYCHO_PYTHON_BINDINGS
// ... binding code ...
#endif // TYCHO_PYTHON_BINDINGS
```

This makes them safe to include from aggregate headers that are used by both binding and non-binding TUs. When `TYCHO_PYTHON_BINDINGS` is not defined, the file is empty.

### Header Shadowing Rule

When including a `bindings/` header from another `bindings/` file, **always use the explicit subdirectory path**:

```cpp
// CORRECT:
#include "vf/python_arg_parsing.h"
#include "solvers/jet_bind.h"

// WRONG (may resolve to a file in another src/ subdirectory):
#include "python_arg_parsing.h"
```

Because `src/bindings/` subdirectories are added to the include path alongside the core `src/` subdirectories, flat names can be shadowed by files in the core directories.

### Out-of-Line `TychoBind` Implementations

When a `TychoBind` specialization's `Build()` body is too large for a header, declare it in the header and define it in a `.cpp` file:

```cpp
// ODEPhaseBind.h
template <> struct TychoBind<ODEPhaseBase> {
    static void Build(nb::module_ &m);  // declaration only
};

// ODEPhaseBaseBind.cpp
void TychoBind<ODEPhaseBase>::Build(nb::module_ &m) {
    // ... large implementation ...
}
```

In the `.cpp` file, use `using namespace tycho;` and explicit type aliases (`using VectorXd = Eigen::VectorXd;` etc.) since class-scope aliases from the header are not available in the out-of-line definition context.

### `nb::arg(...).none()` on AutoScale Parameters

All parameters typed as `ScaleType` must use `.none()`:

```cpp
obj.def("setAutoScale", &T::setAutoScale, nb::arg("scale").none());
```

Without `.none()`, nanobind rejects Python `None` before the `ScaleType` type caster can convert it to the `"none"` sentinel string.

### `nb::call_guard<nb::gil_scoped_release>()` on Parallel Methods

Any method that runs C++ code in parallel (e.g., `integrate_parallel`, `integrate_dense_parallel`, batch `integrate`) must release the GIL:

```cpp
obj.def("integrate_parallel", ...,
    nb::arg("Xt0UPs"), nb::arg("tfs"), nb::arg("threads"),
    nb::call_guard<nb::gil_scoped_release>());
```

This allows Python threads to run while C++ threads do integration work.

### `nb::is_operator()` on Python Dunder Methods

All operator overloads use `nb::is_operator()`:

```cpp
obj.def("__add__", [](const Derived &a, const Derived &b) { ... }, nb::is_operator());
```

This tells nanobind the function is an operator, affecting error handling behavior (returns `NotImplemented` instead of raising `TypeError` for unsupported operand types).

### `__array_ufunc__ = None`

Set on all VectorFunction types in `DoubleMathBuild`:

```cpp
obj.attr("__array_ufunc__") = nb::none();
```

This prevents numpy from hijacking `__radd__` and `__rmul__` when a VectorFunction appears on the right side of an operation with a numpy array. Without this, `np.array([1,2,3]) + vf` would try to broadcast instead of calling `vf.__radd__`.

---

## 11. Adding a New Binding (Step-by-Step)

### Adding a New VectorFunction Type

1. **Implement the C++ type** in `include/tycho/detail/vf/` as a `VectorFunction<Derived, IR, OR, Jm, Hm>` subclass (see [VectorFunction Developer Guide](VectorFunction.md)), then include it from the appropriate aggregate header in `src/vf/tycho_vector_functions.h`.

2. **Add a `TychoBind<T>` specialization** in an appropriate `*_bind.h` file (or create a new one in `src/bindings/vf/`):

```cpp
template <> struct TychoBind<MyNewFunction> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<MyNewFunction>(m, name);
        obj.def(nb::init</* constructor args */>());
        bind::DenseBaseBuild<MyNewFunction>(obj);
        // Optional: bind::SegBuild if it needs full math/indexing API
    }
};
```

3. **Register it** in the appropriate build function (e.g., `VectorFunctionBuild`):

```cpp
reg.Build_Register<MyNewFunction>(reg.vfmod, "MyNewFunction");
```

4. **Update CMakeLists.txt** if you created a new `.cpp` file -- add it to the `nanobind_add_module(_tychopy ...)` list in `src/bindings/CMakeLists.txt`.

5. **Re-export in Python** -- add to `tychopy/VectorFunctions/__init__.py`:

```python
MyNewFunction = _tychopy.VectorFunctions.MyNewFunction
```

### Adding a New ODE

Use `BuildGenODEModule` for a generic ODE (accepts any VectorFunction as dynamics) or `BuildODEModule` for a concrete ODE type:

```cpp
// In the appropriate build function:
tycho::bind::BuildGenODEModule<BaseType, XV, UV, PV>("my_ode", reg.ocmod, reg);
```

This creates a submodule `_tychopy.OptimalControl.my_ode` with `ode`, `integrator`, and `phase` classes.

### Adding a New Type Caster

1. Add the specialization to `type_casters.h`:

```cpp
template <> struct type_caster<MyType> {
    NB_TYPE_CASTER(MyType, const_name("MyType"))
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept { ... }
    static handle from_cpp(const MyType &src, rv_policy policy, cleanup_list *cleanup) { ... }
};
```

2. Place it **before** the STL caster includes in `type_casters.h` if any STL caster might reference it.

3. Rebuild -- the PCH will pick it up automatically.

### Checklist for Any New Binding

- [ ] File is in `src/bindings/` (not in core `src/` directories)
- [ ] Header is guarded with `#ifdef TYCHO_PYTHON_BINDINGS`
- [ ] Include paths use explicit subdirectory names (no shadowing)
- [ ] New `.cpp` files added to `nanobind_add_module(_tychopy ...)` in `src/bindings/CMakeLists.txt`
- [ ] `nb::arg(...).none()` used for any parameter that accepts Python `None`
- [ ] `nb::is_operator()` used for all dunder methods
- [ ] `nb::call_guard<nb::gil_scoped_release>()` used for parallel methods
- [ ] Re-exported in the corresponding `tychopy/` Python subpackage
- [ ] Build and test: `cd build && ninja -j6 all && python scripts/run_examples.py`

---

## 12. Reference: Complete File Listing

Every file in `src/bindings/` with a one-line description:

| File | Purpose |
|---|---|
| **Top-level** | |
| `CMakeLists.txt` | Build rules for `pch_bindings` and `_tychopy` targets |
| `tycho_module.cpp` | `NB_MODULE(_tychopy, m)` entry point; orchestrates all build functions |
| `function_registry.h` | `TychoBind<T>` primary template; `FunctionRegistry` struct with submodule refs |
| `type_casters.h` | Custom type casters: VectorXd, VectorXi, Tensor, VarIndexType, ScaleType, variants |
| `pch_nb.h` | Binding PCH: nanobind headers, type_casters, function_registry, jet_bind |
| `vector_functions_bind.h` | Aggregate include for VF binding headers (common_functions, dense_function_base, generic, etc.) |
| `utils_bind.cpp` | Thread pool, timers, color console, memory utilities |
| `bump_allocator_bind.cpp` | Bump allocator / memory manager binding implementation |
| `bump_allocator_bind.h` | Bump allocator / memory manager binding declaration |
| **vf/** | |
| `dense_function_base_bind.h` | `DenseBaseBuild`, `DoubleMathBuild`, `UnaryMathBuild`, `BinaryMathBuild`, `BinaryOperatorsBuild`, `FunctionIndexingBuild`, `ConditionalOperatorsBuild` |
| `common_functions_bind.h` | `SegBuild`; `TychoBind` for Segment, Arguments, Constant, NestedFunction, ParsedInput, TwoFunctionSum, NormalizedPower, IntegralNorm |
| `generic_function_bind.h` | `GenericBuild`, `MinMaxBuild`, `ComparativeBuild`, `ConditionalBuild`, `IfElseBuild` |
| `vector_function_bind.h` | `TychoBind` specializations for VectorFunction wrapper types |
| `interp_table_bind.h` | `InterpTable1DBuild`, `InterpTable2DBuild`, `InterpTable3DBuild`, `InterpTable4DBuild` |
| `python_functions.h` | `PyVectorFunction` / `PyScalarFunction` -- Python-callable VF wrappers using FD derivatives |
| `python_arg_parsing.h` | `ParsePythonArgs()`, `ParsePythonArgsScalar()` declarations |
| `python_arg_parsing.cpp` | Implementations of ParsePythonArgs/ParsePythonArgsScalar |
| `tycho_vector_functions.cpp` | `VectorFunctionBuild()` orchestrator |
| `vector_function_build_part1.cpp` | VF type registrations (first half) |
| `vector_function_build_part2.cpp` | VF type registrations (second half) |
| `args_seg_build_part1.cpp` | Arguments/Segment template instantiations (part 1 of 5) |
| `args_seg_build_part2.cpp` | Arguments/Segment template instantiations (part 2 of 5) |
| `args_seg_build_part3.cpp` | Arguments/Segment template instantiations (part 3 of 5) |
| `args_seg_build_part4.cpp` | Arguments/Segment template instantiations (part 4 of 5) |
| `args_seg_build_part5.cpp` | Arguments/Segment template instantiations (part 5 of 5) |
| `bulk_operations_build.cpp` | Stack, Sum, SumElems, StackScalar, SumScalar bindings |
| `free_functions_build.cpp` | Module-level free functions (cross, dot, arctan2, etc.) |
| `matrix_function_build.cpp` | ColMatrix, RowMatrix type bindings |
| **optimal_control/** | |
| `ode_phase_bind.h` | `TychoBind` for ODEPhase, StateFunction, LinkFunction, FDDerivArbitrary; `ODEPhaseBuildImpl` helper; forward declarations for ODEPhaseBase, LGLInterpTable, OptimalControlProblem |
| `ode_bind.h` | `ODE_ExpressionBuild`, `BuildODEModule`, `BuildGenODEModule`, `TychoBind<InterpFunction>` |
| `ode_sizes_bind.h` | `ODESizeBuild<XV,UV,PV,Derived>` helper |
| `ode_arguments_bind.h` | ODEArguments binding |
| `mesh_iterate_info_bind.h` | MeshIterateInfo binding declaration |
| `mesh_iterate_info_bind.cpp` | MeshIterateInfo binding implementation |
| `tycho_optimal_control.cpp` | `OptimalControlBuild()` orchestrator |
| `generic_odes_build_part1.cpp` | GenericODE instantiations (part 1 of 6) |
| `generic_odes_build_part2.cpp` | GenericODE instantiations (part 2 of 6) |
| `generic_odes_build_part3.cpp` | GenericODE instantiations (part 3 of 6) |
| `generic_odes_build_part4.cpp` | GenericODE instantiations (part 4 of 6) |
| `generic_odes_build_part5.cpp` | GenericODE instantiations (part 5 of 6) |
| `generic_odes_build_part6.cpp` | GenericODE instantiations (part 6 of 6) |
| `ode_phase_base_bind.cpp` | Out-of-line `TychoBind<ODEPhaseBase>::Build` implementation |
| `optimal_control_problem_bind.cpp` | Out-of-line `TychoBind<OptimalControlProblem>::Build` implementation |
| `lgl_interp_table_bind.cpp` | Out-of-line `TychoBind<LGLInterpTable>::Build` implementation |
| **solvers/** | |
| `jet_bind.h` | `TychoBind<Jet>` declaration; `JetInvoker` partial spec for `nb::args` |
| `jet_bind.cpp` | `TychoBind<Jet>::Build` implementation |
| `psiopt_bind.h` | `TychoBind<PSIOPT>` declaration |
| `psiopt_bind.cpp` | `TychoBind<PSIOPT>::Build` implementation |
| `optimization_problem_bind.h` | `TychoBind<OptimizationProblem>` declaration |
| `optimization_problem_bind.cpp` | `TychoBind<OptimizationProblem>::Build` implementation |
| `optimization_problem_base.cpp` | OptimizationProblemBase binding |
| `tycho_solvers.cpp` | `SolversBuild()` orchestrator |
| **astro/** | |
| `tycho_astro.cpp` | `AstroBuild()` orchestrator |
| `kepler_utils.cpp` | Kepler utility function bindings (cartesian/classic/modified conversions, propagation) |
| `kepler_model.cpp` | Kepler ODE model bindings (Kepler.ode, Kepler.phase, etc.) |
| `lambert_solvers_bind.cpp` | Lambert solver bindings (includes `astro/lambert_solvers.h` directly, not the full aggregate) |
| **integrators/** | |
| `integrator_bind.h` | `TychoBind<Integrator<DODE>>` with all integrate/integrate_dense/integrate_stm methods; `IntegratorBuildConstructors` |
| `rk_coeffs_bind.cpp` | Runge-Kutta coefficient table and method selection bindings |
