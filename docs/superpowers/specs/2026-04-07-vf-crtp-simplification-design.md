# VectorFunction CRTP Simplification Design Spec

**Goal:** Reduce the VectorFunction CRTP hierarchy from 8 levels to 5, clean up type alias macros, and modernize trait flags to C++20 concepts. No behavioral changes — purely structural.

**Architecture:** Three independent directions executed sequentially: concepts first (zero-risk consistency win), macro cleanup second (mechanical refactor), chain flattening last (benefits from cleanup in 1 and 2). One commit per direction. C++ unit tests after each direction; full verification (Python examples, brachistochrone, benchmarks) after all three.

**Tech Stack:** C++20 (concepts, requires clauses), Clang 21, CMake/ninja build system.

---

## Current State

### 8-Level CRTP Hierarchy

```
CRTPBase<Derived>                                          // derived(), name() + 6 dead methods
  -> ComputableBase<Derived, IR, OR>                       // IO sizes, static flags, compute(), constraints()
    -> Computable<Derived, IR, OR>                         // forwards 3 type aliases -- nothing else
      -> DenseFunctionBase<Derived, IR, OR>                // Jacobian/Hessian, expression ops, domain products
        -> DenseFunction<Derived, IR, OR>                  // routes OR==1 -> DenseScalarFunctionBase
          -> DenseFirstDerivatives<Derived, IR, OR, JMode> // specialization injection point
            -> DenseSecondDerivatives<..., JMode, HMode>   // specialization injection point
              -> DenseDerivatives<..., JMode, HMode>       // VF_TYPE_ALIASES only
                -> VectorFunction<..., JMode, HMode>       // VF_TYPE_ALIASES only
```

### What Each Level Contributes

| Level | Genuine content? | Eliminated? |
|-------|-----------------|-------------|
| `CRTPBase` | 2 used methods (`derived()`, `name()`), 6 dead | Yes -- inline 2, drop 6 |
| `ComputableBase` | **Core** -- IO, flags, compute, constraints | No |
| `Computable` | **Nothing** -- 3 forwarded type aliases | Yes |
| `DenseFunctionBase` | **Core** -- Jacobian/Hessian, expression DSL | No |
| `DenseFunction` | Routes OR==1 -> DenseScalarFunctionBase | Yes -- use `requires` |
| `DenseScalarFunctionBase` | objective(), objective_gradient(), objective_gradient_hessian(), kkt_fill_hess() | Yes -- merge into DenseFunctionBase |
| `DenseFirstDerivatives` | Specialization point for Jacobian mode | No |
| `DenseSecondDerivatives` | Specialization point for Hessian mode | No |
| `DenseDerivatives` | Only `DENSE_FUNCTION_BASE_TYPES(Base)` | Yes -- merge into VectorFunction |
| `VectorFunction` | Only `DENSE_FUNCTION_BASE_TYPES(Base)` | No -- this is the public name |

---

## Target State

### 5-Level CRTP Hierarchy

```
ComputableBase<Derived, IR, OR>                              // absorbs CRTPBase (derived, name)
  -> DenseFunctionBase<Derived, IR, OR>                      // absorbs DenseScalarFunctionBase via requires
    -> DenseFirstDerivatives<Derived, IR, OR, JMode>         // unchanged
      -> DenseSecondDerivatives<Derived, IR, OR, JMode, HMode> // unchanged
        -> VectorFunction<Derived, IR, OR, JMode, HMode>      // absorbs DenseDerivatives
```

### End-State User Code (unchanged API)

```cpp
struct MyODE : VectorFunction<MyODE, 7, 6, DenseDerivativeMode::AutodiffFwd> {
    static constexpr bool is_vectorizable = true;

    void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        // ...
    }
};
```

---

## Direction 3: Concepts for Leaf Type Constraints

**Commit scope:** One commit covering all of 3a-3c.
**Risk:** None. Same codegen, better consistency and error messages.
**Verification:** C++ unit tests.

### 3a. `static const bool` -> `static constexpr bool`

Modernize ~74 trait flag declarations across `ComputableBase` and all leaf types (~37 files).

```cpp
// Before (computable_base.h)
static const bool is_vectorizable = false;
static const bool is_linear_function = false;

// After
static constexpr bool is_vectorizable = false;
static constexpr bool is_linear_function = false;
```

Files affected: `computable_base.h` + all leaf types that override these flags.

### 3b. Add Missing Concepts, Unify Usage

Add three missing concepts to `vector_function_concepts.h`:

```cpp
template <typename T>
concept ConditionalVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_conditional); };

template <typename T>
concept CwiseVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_cwise_operator); };

template <typename T>
concept HasDiagonalHess =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::has_diagonal_hessian); };
```

Unify all `if constexpr` branches in `dense_function_base.h` and elsewhere to use concept names:

```cpp
// Before
if constexpr (!Derived::is_vectorizable) {
if constexpr (Derived::is_linear_function) {

// After
if constexpr (!Vectorizable<Derived>) {
if constexpr (LinearVF<Derived>) {
```

### 3c. Add `DenseVectorFunction` Concept

```cpp
template <typename T>
concept DenseVectorFunction = requires(const T& t) {
    { T::IRC } -> std::convertible_to<int>;
    { T::ORC } -> std::convertible_to<int>;
    { t.input_rows() } -> std::same_as<int>;
    { t.output_rows() } -> std::same_as<int>;
    { T::is_vectorizable } -> std::convertible_to<bool>;
    { T::is_linear_function } -> std::convertible_to<bool>;
    typename T::template Output<double>;
    typename T::template Input<double>;
    typename T::template Jacobian<double>;
    typename T::INPUT_DOMAIN;
};
```

Constrain the type-erasure entry points:

```cpp
// gf_type_erasure.h
template <int IR, int OR, DenseVectorFunction T>
    requires (T::IRC == IR || IR == -1) && (T::ORC == OR || OR == -1)
struct GFModel final : GFModelCommon<IR, OR, T> { /* ... */ };

// generic_function.h — preserve existing Eigen exclusion alongside the new concept
template <int IR, int OR>
struct GenericFunction : VectorFunction<GenericFunction<IR, OR>, IR, OR> {
    template <DenseVectorFunction T>
        requires (!std::derived_from<std::decay_t<T>, Eigen::EigenBase<std::decay_t<T>>>)
              && (T::IRC == IR || IR == -1) && (T::ORC == OR || OR == -1)
    GenericFunction(T func) { /* ... */ }
};
```

Note: The existing `GenericFunction` constructor uses `enable_if` to exclude Eigen types
from implicit conversion. The `DenseVectorFunction` concept alone does not exclude Eigen
types, so the Eigen exclusion must be preserved as an explicit `requires` clause. The
existing `GFStorable` concept in `gf_type_erasure.h` partially overlaps with
`DenseVectorFunction` — during implementation, evaluate whether `GFStorable` should be
redefined in terms of `DenseVectorFunction` or kept separate.

---

## Direction 2: Replace `DENSE_FUNCTION_BASE_TYPES` Macro

**Commit scope:** One commit covering all of 2a-2d.
**Risk:** Low. Purely mechanical rename/move. Large diff but no semantic change.
**Verification:** C++ unit tests.

### 2a. Move Fixed Aliases to Namespace Scope

Create `include/tycho/detail/vf/core/eigen_ref_aliases.h`:

```cpp
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace tycho::vf {

template <class T> using CVecRef  = const Eigen::MatrixBase<T>&;
template <class T> using VecRef   = Eigen::MatrixBase<T>&;
template <class T> using CMatRef  = const Eigen::MatrixBase<T>&;
template <class T> using MatRef   = Eigen::MatrixBase<T>&;
template <class T> using CEigRef  = const Eigen::EigenBase<T>&;
template <class T> using EigRef   = Eigen::EigenBase<T>&;
template <class T> using CDiagRef = const Eigen::DiagonalBase<T>&;
template <class T> using DiagRef  = Eigen::DiagonalBase<T>&;

} // namespace tycho::vf
```

### 2b. Reduce the Macro to 5 Dependent Aliases

Rename `DENSE_FUNCTION_BASE_TYPES` -> `VF_TYPE_ALIASES` in `function_type_def_macros.h`:

```cpp
#define VF_TYPE_ALIASES(Base)                                                      \
    template <class Scalar> using Output   = typename Base::template Output<Scalar>; \
    template <class Scalar> using Input    = typename Base::template Input<Scalar>;  \
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>; \
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>; \
    template <class Scalar> using Hessian  = typename Base::template Hessian<Scalar>;
```

### 2c. Update All Method Signatures

Replace per-class Eigen ref aliases with namespace-scope versions. This is a large
mechanical rename affecting ~750+ occurrences across ~55 files spanning VF headers,
optimal control, integrators, astro, extensions, and tests.

```cpp
// Before
template <class InType, class OutType>
inline void compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {

// After
template <class InType, class OutType>
inline void compute(CVecRef<InType> x, CVecRef<OutType> fx_) const {
```

The rename covers all 8 alias pairs:
- `ConstVectorBaseRef` -> `CVecRef`, `VectorBaseRef` -> `VecRef`
- `ConstMatrixBaseRef` -> `CMatRef`, `MatrixBaseRef` -> `MatRef`
- `ConstEigenBaseRef` -> `CEigRef`, `EigenBaseRef` -> `EigRef`
- `ConstDiagonalBaseRef` -> `CDiagRef`, `DiagonalBaseRef` -> `DiagRef`

### 2d. Eliminate `SUB_FUNCTION_IO_TYPES` Entirely

Replace macro-generated aliases with direct qualified access:

```cpp
// Before
SUB_FUNCTION_IO_TYPES(OuterFunc)
SUB_FUNCTION_IO_TYPES(InnerFunc)
OuterFunc_Output<Scalar> fx_outer;

// After
typename OuterFunc::template Output<Scalar> fx_outer;
// Or local aliases inside method bodies:
using OuterOut = typename OuterFunc::template Output<Scalar>;
```

---

## Direction 1: Flatten the CRTP Chain (8 -> 5 levels)

**Commit scope:** One commit covering all of 1a-1e.
**Risk:** Low-medium. Most complex direction, but all changes are mechanical and verified by investigation.
**Verification:** C++ unit tests.

### Investigation Findings

The following was confirmed by deep codebase investigation:

1. **CRTPBase**: Only `ComputableBase` inherits from it. Only `derived()` and `name()` are called anywhere. `cast()`, `copy()`, `deep_copy_into()`, `find()`, `copy_if_same()`, `forward_if_same()` have zero call sites. `CRTPMixin` is also unused (DomainHolder does not actually use it).
2. **Computable**: Both templates are pure forwarding. Only `DenseFunctionBase` references it (2 lines). `SparseFunctionBase` already inherits from `ComputableBase` directly.
3. **DenseFunction**: Empty routing class. 4 inheritance sites (DenseFirstDerivatives primary + 3 specializations).
4. **DenseScalarFunctionBase**: 4 methods to move. 2 extern template declarations to remove.
5. **DenseDerivatives**: Zero content beyond macro. Only VectorFunction inherits from it. No specializations.
6. **No SFINAE/enable_if dependencies** on the partial specialization routing.

### 1a. Inline CRTPBase into ComputableBase, Delete crtp_base.h

Move `derived()` (both overloads) and `name()` into `ComputableBase`. Drop 6 dead
methods (`cast()`, `copy()`, `deep_copy_into()`, `find()`, `copy_if_same()`,
`forward_if_same()` -- all have zero call sites). Delete `CRTPMixin` (also unused --
`DomainHolder` does not actually use it despite the original design doc's claim).

Note: The original design doc (`VectorFunctionSimplification.md`) says to keep `cast()`
and `copy()`. Investigation confirmed zero call sites for both, so we drop them.

Delete `crtp_base.h` and remove all stale includes (~22 files). None of these files
use `CRTPBase` or `CRTPMixin` directly -- the includes are purely transitive.

Files:
- Modify: `computable_base.h` (add 3 methods, remove CRTPBase inheritance)
- Delete: `crtp_base.h`
- Remove stale `#include "crtp_base.h"` from ~22 files:
  - VF: `dense_function_specs.h`, `dense_function_operations.h`, `function_domains.h`,
    `expression_fwd_declarations.h`, `vector_function_concepts.h`, `detect_diagonal.h`,
    `auto_scaling_utils.h`, `conditional_base.h`, `generic_conditional.h`,
    `generic_comparative.h`
  - Solvers: `solver_interface_specs.h`, `sizing_specs.h`, `indexing_data.h`
  - Optimal control: `ode_phase.h`, `ode_phase_base.h`, `optimal_control_problem.h`,
    `lgl_interp_table.h`, `blocked_ode_wrapper.h`, `phase_indexer.h`
  - Integrators: `integrator.h`
  - Other: `shooting_defects.h`, `include/tycho/utils.h`
  - Bindings: `src/bindings/vf/python_arg_parsing.h`

### 1b. Eliminate Computable

Change `DenseFunctionBase` inheritance from `Computable<Derived, IR, OR>` to `ComputableBase<Derived, IR, OR>`.

Files:
- Modify: `dense_function_base.h` (2 lines: inheritance + Base alias)
- Delete: `computable.h`

### 1c. Eliminate DenseFunction + DenseScalarFunctionBase

Move `objective()`, `objective_gradient()`, `objective_gradient_hessian()`, and `kkt_fill_hess()` into `DenseFunctionBase` with `requires (OR == 1)`:

```cpp
// In dense_function_base.h:
void objective(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double& Val,
               const tycho::solvers::SolverIndexingData& data) const
    requires (OR == 1)
{ /* body from DenseScalarFunctionBase */ }

void objective_gradient(/* ... */) const requires (OR == 1) { /* ... */ }
void objective_gradient_hessian(/* ... */) const requires (OR == 1) { /* ... */ }

protected:
void kkt_fill_hess(/* ... */) const requires (OR == 1) { /* ... */ }
```

Change DenseFirstDerivatives (4 files) to inherit from `DenseFunctionBase` directly.

Update extern template files: remove 2 `DenseScalarFunctionBase` declarations + 2 instantiations.

Files:
- Modify: `dense_function_base.h` (add 4 methods with `requires`)
- Modify: `dense_derivatives.h`, `dense_fdiff_fwd.h`, `dense_fdiff_cent_array.h`, `dense_autodiff_fwd.h` (re-parent)
- Modify: `extern_templates.h`, `extern_template_instantiations.cpp` (remove DenseScalarFunctionBase entries)
- Delete: `dense_function.h`, `dense_scalar_function_base.h`

### 1d. Eliminate DenseDerivatives

VectorFunction inherits directly from `DenseSecondDerivatives`. Remove `DenseDerivatives` class from `dense_derivatives.h`.

Files:
- Modify: `vector_function.h` (re-parent)
- Modify: `dense_derivatives.h` (remove DenseDerivatives class, ~5 lines)

### 1e. Clean Up Includes

Remove `#include` directives for deleted files from umbrella headers and any transitive includes.

---

## Performance Guarantees

### Runtime Performance: No Impact

- All dispatch is CRTP (static). No virtual methods affected.
- Removing empty intermediate layers has zero runtime effect -- compiler already elides them.
- `requires (OR == 1)` produces identical codegen to the current partial specialization.
- Method inlining behavior unchanged (all methods remain in headers).

### Build Performance: Neutral-to-Positive

- Fewer template classes to instantiate (3 empty classes removed from every VF type's chain).
- 2 fewer extern template instantiations (DenseScalarFunctionBase removed).
- No new TUs, no new template instantiation sites.
- `requires` doesn't change instantiation patterns vs partial specialization.

---

## Verification Plan

| After | Verification |
|-------|-------------|
| Direction 3 (concepts) | C++ unit tests (`ctest --output-on-failure`) |
| Direction 2 (macro cleanup) | C++ unit tests |
| Direction 1 (flatten chain) | C++ unit tests |
| All three complete | Full: C++ tests + Python examples + brachistochrone + benchmark regression check |
