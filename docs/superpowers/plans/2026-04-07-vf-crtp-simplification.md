# VectorFunction CRTP Simplification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the VectorFunction CRTP hierarchy from 8 levels to 5, clean up macros, and modernize trait flags to C++20 concepts.

**Architecture:** Three directions executed sequentially (3 -> 2 -> 1). Each direction is one commit. C++ unit tests verify after each; full verification after all three.

**Tech Stack:** C++20 (concepts, requires), Clang 21, CMake/ninja, Google Test

**Spec:** `docs/superpowers/specs/2026-04-07-vf-crtp-simplification-design.md`

---

## File Structure

### Files Created
- `include/tycho/detail/vf/core/eigen_ref_aliases.h` — namespace-scope Eigen ref type aliases (Direction 2)

### Files Deleted
- `include/tycho/detail/utils/crtp_base.h` — CRTPBase + CRTPMixin (Direction 1)
- `include/tycho/detail/vf/core/computable.h` — Computable forwarding layer (Direction 1)
- `include/tycho/detail/vf/core/dense_function.h` — DenseFunction routing (Direction 1)
- `include/tycho/detail/vf/core/dense_scalar_function_base.h` — DenseScalarFunctionBase (Direction 1)

### Files Modified (by direction)

**Direction 3 (concepts):**
- `include/tycho/detail/vf/core/computable_base.h` — `static const` -> `static constexpr`
- ~32 VF leaf type headers — same change
- `include/tycho/detail/vf/core/vector_function_concepts.h` — add 4 new concepts
- `include/tycho/detail/vf/core/dense_function_base.h` — unify `if constexpr` to use concepts
- `include/tycho/detail/vf/type_erasure/generic_function.h` — constrain constructor with concept
- `include/tycho/detail/vf/type_erasure/gf_type_erasure.h` — keep `GFStorable` as-is (see note below)

**Direction 2 (macro cleanup):**
- `include/tycho/detail/vf/core/function_type_def_macros.h` — rename macro, reduce to 5 aliases
- ~54 files using `DENSE_FUNCTION_BASE_TYPES` -> `VF_TYPE_ALIASES`
- ~55 files using `ConstVectorBaseRef` etc. -> `CVecRef` etc. (~750 occurrences)
- ~15 files using `SUB_FUNCTION_IO_TYPES` -> direct qualified access

**Direction 1 (flatten chain):**
- `include/tycho/detail/vf/core/computable_base.h` — absorb CRTPBase methods
- `include/tycho/detail/vf/core/dense_function_base.h` — absorb DenseScalarFunctionBase methods, re-parent
- `include/tycho/detail/vf/derivatives/dense_derivatives.h` — remove DenseDerivatives class, re-parent DenseFirstDerivatives
- `include/tycho/detail/vf/derivatives/dense_fdiff_fwd.h` — re-parent
- `include/tycho/detail/vf/derivatives/dense_fdiff_cent_array.h` — re-parent
- `include/tycho/detail/vf/derivatives/dense_autodiff_fwd.h` — re-parent
- `include/tycho/detail/vf/core/vector_function.h` — re-parent to DenseSecondDerivatives
- `include/tycho/detail/vf/extern_templates.h` — remove DenseScalarFunctionBase entries
- `src/vf/extern_template_instantiations.cpp` — remove DenseScalarFunctionBase entries
- ~22 files — remove stale `#include "crtp_base.h"`

---

## Task 0: Direction 3 — Concepts for Leaf Type Constraints

**Files:**
- Modify: `include/tycho/detail/vf/core/computable_base.h:92-107`
- Modify: `include/tycho/detail/vf/core/vector_function_concepts.h`
- Modify: `include/tycho/detail/vf/core/dense_function_base.h` (8 `if constexpr` branches)
- Modify: `include/tycho/detail/vf/type_erasure/generic_function.h:48-54`
- Modify: ~37 VF leaf type headers (static const -> static constexpr) — includes files in
  `include/tycho/detail/vf/`, `include/tycho/detail/optimal_control/transcription/`,
  `include/tycho/detail/integrators/`, `include/tycho/detail/astro/`

**Note on GFStorable vs DenseVectorFunction:** `GFStorable<T, IR, OR>` in `gf_type_erasure.h`
constrains what can be stored in `GFStorage` (checking runtime interface: `input_rows()`,
`output_rows()`, `is_linear()`, `name()`, `thread_safe()`). `DenseVectorFunction` constrains
type structure (compile-time: `IRC`, `ORC`, `Output<double>`, etc.). These serve different
purposes. Keep `GFStorable` on `GFModelCommon`/`GFModel`/`GFStorage`; use `DenseVectorFunction`
only on the `GenericFunction` constructor (the user-facing entry point).

### Step 0a: `static const bool` -> `static constexpr bool`

- [ ] **Step 1: Replace `static const bool` with `static constexpr bool` in computable_base.h**

In `include/tycho/detail/vf/core/computable_base.h`, lines 96-107, change:
```cpp
    static const bool InputIsDynamic = (IR < 0);
    static const bool OutputIsDynamic = (OR < 0);
    static const bool JacobianIsDynamic = (IR < 0 || OR < 0);
    static const bool FullyDynamic = (IR < 0 && OR < 0);

    static const bool is_vectorizable = false;
    static const bool is_linear_function = false;
    static const bool has_diagonal_jacobian = false;
    static const bool has_diagonal_hessian = false;
    static const bool is_cwise_operator = false;
    static const bool is_generic_function = false;
    static const bool is_conditional = false;
```
to:
```cpp
    static constexpr bool InputIsDynamic = (IR < 0);
    static constexpr bool OutputIsDynamic = (OR < 0);
    static constexpr bool JacobianIsDynamic = (IR < 0 || OR < 0);
    static constexpr bool FullyDynamic = (IR < 0 && OR < 0);

    static constexpr bool is_vectorizable = false;
    static constexpr bool is_linear_function = false;
    static constexpr bool has_diagonal_jacobian = false;
    static constexpr bool has_diagonal_hessian = false;
    static constexpr bool is_cwise_operator = false;
    static constexpr bool is_generic_function = false;
    static constexpr bool is_conditional = false;
```

Also change `static const int IRC = IR;` and `static const int ORC = OR;` (lines 92-94) to `static constexpr int`.

- [ ] **Step 2: Replace `static const bool` in all VF leaf type headers**

Use `grep -rn "static const bool" include/tycho/detail/` to find all ~37 files. This includes
files outside `vf/` — leaf VF types in `optimal_control/transcription/` (trapezoidal_defects,
shooting_defects, lgl_defects), `integrators/` (rk_steppers), and `astro/` (mee_dynamics,
kepler_propagator) also override these flags. In each, replace `static const bool` with
`static constexpr bool`. These are all override declarations like:
```cpp
static constexpr bool is_vectorizable = true;
static constexpr bool is_linear_function = true;
static constexpr bool has_diagonal_jacobian = true;
```

The files are spread across `expressions/`, `operators/`, `scaling/`, `common/`, and `type_erasure/` subdirectories.

- [ ] **Step 3: Build and run C++ tests**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light && ctest --output-on-failure
```

Expected: All tests pass. This is a zero-risk mechanical change.

### Step 0b: Add missing concepts and unify usage

- [ ] **Step 4: Add 4 new concepts to vector_function_concepts.h**

In `include/tycho/detail/vf/core/vector_function_concepts.h`, before the closing `} // namespace tycho::vf`, add:

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

- [ ] **Step 5: Unify `if constexpr` branches in dense_function_base.h to use concepts**

In `include/tycho/detail/vf/core/dense_function_base.h`, replace all direct trait access with concept checks. The 8 locations (found by grep):

| Line | Before | After |
|------|--------|-------|
| 308 | `if constexpr (!Derived::is_vectorizable)` | `if constexpr (!Vectorizable<Derived>)` |
| 420 | `if constexpr (!Derived::is_vectorizable)` | `if constexpr (!Vectorizable<Derived>)` |
| 681 | `if constexpr (Derived::is_linear_function)` | `if constexpr (LinearVF<Derived>)` |
| 853 | `if constexpr (Derived::is_linear_function)` | `if constexpr (LinearVF<Derived>)` |
| 1062 | `if constexpr (Derived::is_vectorizable)` | `if constexpr (Vectorizable<Derived>)` |
| 1233 | `if constexpr (Derived::is_vectorizable)` | `if constexpr (Vectorizable<Derived>)` |
| 1256 | `!Derived::is_linear_function` | **KEEP AS-IS** — this is a `return` expression, not `if constexpr`; concepts cannot be used as runtime return values |
| 1260 | `if constexpr (!Derived::is_linear_function)` | `if constexpr (!LinearVF<Derived>)` |

Also check `computable_base.h` lines 160, 306 for any `Derived::is_vectorizable` / `Vectorizable<Derived>` — these already use `Vectorizable<Derived>` so no change needed there.

- [ ] **Step 6: Constrain GenericFunction constructor with DenseVectorFunction concept**

In `include/tycho/detail/vf/type_erasure/generic_function.h`, replace lines 48-51:
```cpp
    template <class T, std::enable_if_t<
                           !std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>>,
                           bool> = true>
    GenericFunction(const T &t) {
```
with:
```cpp
    template <DenseVectorFunction T>
        requires (!std::derived_from<std::decay_t<T>, Eigen::EigenBase<std::decay_t<T>>>)
    GenericFunction(const T &t) {
```

This adds `<concepts>` header if not already included (it's already available via `vector_function_concepts.h` which is included transitively).

- [ ] **Step 7: Build and run C++ tests**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light && ctest --output-on-failure
```

Expected: All tests pass.

- [ ] **Step 8: Commit Direction 3**

```bash
git add -u && git commit -m "refactor(vf): modernize trait flags to concepts and constexpr"
```

---

## Task 1: Direction 2 — Replace DENSE_FUNCTION_BASE_TYPES Macro

**Files:**
- Create: `include/tycho/detail/vf/core/eigen_ref_aliases.h`
- Modify: `include/tycho/detail/vf/core/function_type_def_macros.h`
- Modify: ~54 files using `DENSE_FUNCTION_BASE_TYPES` -> `VF_TYPE_ALIASES`
- Modify: ~55 files using old Eigen ref aliases -> new namespace-scope aliases (~750 occurrences)
- Modify: ~15 files using `SUB_FUNCTION_IO_TYPES`

### Step 1a: Create namespace-scope Eigen ref aliases

- [ ] **Step 1: Create `eigen_ref_aliases.h`**

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

- [ ] **Step 2: Include `eigen_ref_aliases.h` early in the VF include chain**

Add `#include "tycho/detail/vf/core/eigen_ref_aliases.h"` to `computable_base.h` (near the top includes), so all downstream VF headers have access. This file is at the root of the VF include chain.

### Step 1b: Rename macro and reduce to 5 aliases

- [ ] **Step 3: Replace the macro definition in function_type_def_macros.h**

Replace the entire `DENSE_FUNCTION_BASE_TYPES` macro (lines 19-32) with:
```cpp
#define VF_TYPE_ALIASES(Base)                                                            \
    template <class Scalar> using Output   = typename Base::template Output<Scalar>;     \
    template <class Scalar> using Input    = typename Base::template Input<Scalar>;      \
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;   \
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>;   \
    template <class Scalar> using Hessian  = typename Base::template Hessian<Scalar>;
```

Keep `SUB_FUNCTION_IO_TYPES` for now — it's eliminated in step 1d.

- [ ] **Step 4: Rename all `DENSE_FUNCTION_BASE_TYPES(Base)` -> `VF_TYPE_ALIASES(Base)` across ~54 files**

Use project-wide search-and-replace. The files span:
- `include/tycho/detail/vf/` (~42 files)
- `include/tycho/detail/optimal_control/` (~8 files)
- `include/tycho/detail/integrators/` (~2 files)
- `include/tycho/detail/astro/` (~2 files)
- `extensions/Tycho_Extensions.h`

The replacement is exact: `DENSE_FUNCTION_BASE_TYPES(Base)` -> `VF_TYPE_ALIASES(Base)`. Some files use `DENSE_FUNCTION_BASE_TYPES(Base);` with a trailing semicolon — keep the semicolon.

- [ ] **Step 5: Build to verify macro rename compiles**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light
```

### Step 1c: Replace per-class Eigen ref aliases with namespace-scope versions

- [ ] **Step 6: Replace all 8 old alias names with new names across ~55 files**

The 8 replacements (in order — do longer names first to avoid partial matches):

| Old name | New name | Approx count |
|----------|----------|:------------:|
| `ConstDiagonalBaseRef` | `CDiagRef` | ~5 |
| `DiagonalBaseRef` | `DiagRef` | ~5 |
| `ConstEigenBaseRef` | `CEigRef` | ~20 |
| `EigenBaseRef` | `EigRef` | ~10 |
| `ConstMatrixBaseRef` | `CMatRef` | ~30 |
| `MatrixBaseRef` | `MatRef` | ~20 |
| `ConstVectorBaseRef` | `CVecRef` | ~450 |
| `VectorBaseRef` | `VecRef` | ~210 |

Do longer names first (`ConstDiagonalBaseRef` before `ConstVectorBaseRef`) to avoid partial
matches. When using search-and-replace tools, process each pair sequentially — never in
parallel, as `VectorBaseRef` is a substring of `ConstVectorBaseRef`.

These span `include/tycho/detail/vf/`, `include/tycho/detail/optimal_control/`,
`include/tycho/detail/integrators/`, `include/tycho/detail/astro/`, `extensions/`, and
`tests/cpp/vector_functions/test_ode_syntax_prototypes.cpp`.

**CRITICAL: Exclude `sparse_function_base.h` from `ConstMatrixBaseRef`/`MatrixBaseRef` rename.**
`include/tycho/detail/vf/core/sparse_function_base.h` defines `ConstMatrixBaseRef` as
`const Eigen::SparseMatrixBase<T>&` — DIFFERENT from the dense alias (`Eigen::MatrixBase`).
A naive rename to `CMatRef` would silently break the sparse path. Either skip this file
entirely for ConstMatrixBaseRef/MatrixBaseRef, or create separate sparse aliases.

Also update `include/tycho/detail/vf/core/dense_function_specs.h` — it defines its own local
ref-type aliases (lines 62-68) for the virtual interface (`ConstVectorBaseRef`, `VectorBaseRef`,
`ConstMatrixBaseRef`, `ConstEigenBaseRef`, `ConstDiagonalBaseRef`, `MatrixBaseRef`). Replace
these definitions with `using` declarations importing the namespace-scope aliases, and update
the ~69 usages in that file's virtual method signatures.

Remove the 2 per-class ref-type aliases from `ComputableBase` (lines 88-89 of
`computable_base.h`). Do NOT modify `Computable` (lines 27-28, 40-41 of `computable.h`) —
it will be deleted in Direction 1.

- [ ] **Step 7: Build to verify ref-type rename compiles**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light
```

### Step 1d: Eliminate SUB_FUNCTION_IO_TYPES macro

- [ ] **Step 8: Replace SUB_FUNCTION_IO_TYPES usage in 15 files**

For each file using `SUB_FUNCTION_IO_TYPES(Func)`, remove the macro invocation and replace `Func_Output<Scalar>` etc. with `typename Func::template Output<Scalar>` (or a local `using` alias inside method bodies where it appears multiple times).

The 15 files (from grep results, excluding `function_type_def_macros.h` definition and doc):
- `expressions/`: `summation.h`, `stacked.h`, `parsed_input.h`, `nested_function.h`, `call_and_append.h`
- `operators/`: `vector_scalar_function_product.h`, `vector_scalar_function_division.h`, `vector_products.h`, `matrix_product.h`, `dot_product.h`, `cwise_sum.h`, `cwise_product.h`, `cross_product.h`
- `scaling/`: `scaled.h`

For each, the pattern is:
```cpp
// Before:
SUB_FUNCTION_IO_TYPES(Func1)
// ... later:
Func1_Output<Scalar> fx1(/* ... */);

// After (local alias in method):
using Func1Out = typename Func1::template Output<Scalar>;
Func1Out fx1(/* ... */);
```

Use `typename Func::template Output<Scalar>` directly for single-use, local `using` alias for repeated use within a method.

- [ ] **Step 9: Delete SUB_FUNCTION_IO_TYPES from function_type_def_macros.h**

Remove lines 34-39 from `function_type_def_macros.h`.

- [ ] **Step 10: Build and run C++ tests**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light && ctest --output-on-failure
```

Expected: All tests pass.

- [ ] **Step 11: Commit Direction 2**

```bash
git add -u && git commit -m "refactor(vf): replace DENSE_FUNCTION_BASE_TYPES macro with namespace-scope aliases"
```

---

## Task 2: Direction 1 — Flatten the CRTP Chain (8 -> 5)

**Files:**
- Modify: `include/tycho/detail/vf/core/computable_base.h`
- Delete: `include/tycho/detail/utils/crtp_base.h`
- Modify: `include/tycho/detail/vf/core/dense_function_base.h`
- Delete: `include/tycho/detail/vf/core/computable.h`
- Delete: `include/tycho/detail/vf/core/dense_function.h`
- Delete: `include/tycho/detail/vf/core/dense_scalar_function_base.h`
- Modify: `include/tycho/detail/vf/derivatives/dense_derivatives.h`
- Modify: `include/tycho/detail/vf/derivatives/dense_fdiff_fwd.h`
- Modify: `include/tycho/detail/vf/derivatives/dense_fdiff_cent_array.h`
- Modify: `include/tycho/detail/vf/derivatives/dense_autodiff_fwd.h`
- Modify: `include/tycho/detail/vf/core/vector_function.h`
- Modify: `include/tycho/detail/vf/extern_templates.h`
- Modify: `src/vf/extern_template_instantiations.cpp`
- Modify: ~22 files to remove stale `#include "crtp_base.h"`

### Step 2a: Inline CRTPBase into ComputableBase

- [ ] **Step 1: Add `derived()` and `name()` to ComputableBase**

In `include/tycho/detail/vf/core/computable_base.h`, replace the inheritance:
```cpp
// Before (line 82):
struct ComputableBase : tycho::utils::CRTPBase<Derived>, InputOutputSize<IR, OR> {
```
with:
```cpp
struct ComputableBase : InputOutputSize<IR, OR> {
```

Then add these 3 methods at the top of the class body (after the opening brace):
```cpp
    Derived &derived() { return static_cast<Derived &>(*this); }
    const Derived &derived() const { return static_cast<const Derived &>(*this); }
    std::string name() const { return type_name<Derived>(); }
```

Keep the `#include "tycho/detail/utils/type_name.h"` include (already present transitively; verify it's available).

- [ ] **Step 2: Remove `#include "tycho/detail/utils/crtp_base.h"` from computable_base.h**

Remove line 59: `#include "tycho/detail/utils/crtp_base.h"`

- [ ] **Step 3: Delete crtp_base.h**

```bash
rm include/tycho/detail/utils/crtp_base.h
```

- [ ] **Step 4: Remove stale `#include "crtp_base.h"` from ~22 files**

Remove the include line from each of these files:
- `include/tycho/detail/vf/core/dense_function_specs.h`
- `include/tycho/detail/vf/core/dense_function_operations.h`
- `include/tycho/detail/vf/core/function_domains.h`
- `include/tycho/detail/vf/core/expression_fwd_declarations.h`
- `include/tycho/detail/vf/core/vector_function_concepts.h`
- `include/tycho/detail/vf/derivatives/detect_diagonal.h`
- `include/tycho/detail/vf/scaling/auto_scaling_utils.h`
- `include/tycho/detail/vf/type_erasure/conditional_base.h`
- `include/tycho/detail/vf/type_erasure/generic_conditional.h`
- `include/tycho/detail/vf/type_erasure/generic_comparative.h`
- `include/tycho/detail/solvers/solver_interface_specs.h`
- `include/tycho/detail/solvers/sizing_specs.h`
- `include/tycho/detail/solvers/indexing_data.h`
- `include/tycho/detail/optimal_control/phase/ode_phase.h`
- `include/tycho/detail/optimal_control/phase/ode_phase_base.h`
- `include/tycho/detail/optimal_control/phase/optimal_control_problem.h`
- `include/tycho/detail/optimal_control/transcription/lgl_interp_table.h`
- `include/tycho/detail/optimal_control/transcription/blocked_ode_wrapper.h`
- `include/tycho/detail/optimal_control/phase/phase_indexer.h`
- `include/tycho/detail/integrators/integrator.h`
- `include/tycho/detail/optimal_control/transcription/shooting_defects.h`
- `include/tycho/utils.h`
- `src/bindings/vf/python_arg_parsing.h`

Search with `grep -rn "crtp_base.h"` to catch any missed files.

- [ ] **Step 5: Build to verify CRTPBase inlining compiles**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light
```

### Step 2b: Eliminate Computable

- [ ] **Step 6: Re-parent DenseFunctionBase from ComputableBase**

In `include/tycho/detail/vf/core/dense_function_base.h`, change:
```cpp
// Before (line 47):
struct DenseFunctionBase : Computable<Derived, IR, OR>, DomainHolder<IR> {
    using Base = Computable<Derived, IR, OR>;
```
to:
```cpp
struct DenseFunctionBase : ComputableBase<Derived, IR, OR>, DomainHolder<IR> {
    using Base = ComputableBase<Derived, IR, OR>;
```

- [ ] **Step 7: Remove `#include "computable.h"` and delete computable.h**

In `dense_function_base.h`, if there's an explicit `#include "computable.h"`, remove it. Then:
```bash
rm include/tycho/detail/vf/core/computable.h
```

Note: `dense_function_base.h` includes `computable_base.h` already (transitively through other headers). Verify the include chain still works.

- [ ] **Step 8: Build to verify Computable elimination compiles**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light
```

### Step 2c: Eliminate DenseFunction + DenseScalarFunctionBase

- [ ] **Step 9: Move scalar objective methods into DenseFunctionBase with `requires (OR == 1)`**

In `include/tycho/detail/vf/core/dense_function_base.h`, add these 4 methods near the end of the class (before the `protected:` section or as a new section). Copy the exact bodies from `dense_scalar_function_base.h`:

```cpp
    // ---- Scalar objective interface (OR == 1 only) ----
    void objective(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                   const tycho::solvers::SolverIndexingData &data) const
        requires (OR == 1)
    {
        Input<double> x(this->input_rows());
        Output<double> fx(1);

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            fx.setZero();
            this->derived().compute(x, fx);
            Val += fx[0] * ObjScale;
        }
    }

    void objective_gradient(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                            EigenRef<Eigen::VectorXd> GX,
                            const tycho::solvers::SolverIndexingData &data) const
        requires (OR == 1)
    {
        Input<double> x(this->input_rows());
        Output<double> fx(1);
        Jacobian<double> jx(1, this->input_rows());
        Eigen::Map<Input<double>> gx(NULL, this->input_rows());

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            new (&gx) Eigen::Map<Input<double>>(GX.data() + data.inner_gradient_starts_[V],
                                                this->input_rows());
            fx.setZero();
            jx.setZero();
            gx.setZero();

            this->derived().compute_jacobian(x, fx, jx);
            Val += fx[0] * ObjScale;
            gx = jx.transpose() * ObjScale;
        }
    }

    void objective_gradient_hessian(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                                    EigenRef<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    EigenRef<Eigen::VectorXi> KKTLocations,
                                    EigenRef<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks,
                                    const tycho::solvers::SolverIndexingData &data) const
        requires (OR == 1)
    {
        Input<double> x(this->input_rows());
        Output<double> fx(1);
        Jacobian<double> jx(1, this->input_rows());
        Eigen::Map<Input<double>> gx(NULL, this->input_rows());
        Hessian<double> hx(this->input_rows(), this->input_rows());
        Output<double> lm(1);
        lm[0] = ObjScale;

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            new (&gx) Eigen::Map<Input<double>>(GX.data() + data.inner_gradient_starts_[V],
                                                this->input_rows());

            fx.setZero();
            jx.setZero();
            gx.setZero();
            hx.setZero();

            this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

            Val += fx[0] * ObjScale;
            this->kkt_fill_hess(V, hx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
    }
```

And add the protected helper:
```cpp
  protected:
    void kkt_fill_hess(int Apl, const Hessian<double> &hx,
                       Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                       EigenRef<Eigen::VectorXi> KKTLocs, EigenRef<Eigen::VectorXi> VarClashes,
                       std::vector<std::mutex> &ClashLocks,
                       const tycho::solvers::SolverIndexingData &data) const
        requires (OR == 1)
    {
        int freeloc = data.inner_kkt_starts_[Apl];
        double *mpt = KKTmat.valuePtr();
        const int *lpt = KKTLocs.data();
        int ActiveVar;

        auto Lock = [&](int var) {
            if (VarClashes[var] == -1) { }
            else { ClashLocks[VarClashes[var]].lock(); }
        };
        auto UnLock = [&](int var) {
            if (VarClashes[var] == -1) { }
            else { ClashLocks[VarClashes[var]].unlock(); }
        };

        const int IRR = (IRC > 0) ? IRC : this->input_rows();

        for (int i = 0; i < IRR; i++) {
            ActiveVar = data.v_loc(i, Apl);
            Lock(ActiveVar);
            for (int j = i; j < IRR; j++) {
                this->derived().add_hessian_elem(hx(j, i), j, i, mpt, lpt, freeloc);
            }
            UnLock(ActiveVar);
        }
    }
```

Note: The `kkt_fill_hess` body above is adapted from `dense_scalar_function_base.h` line 122.
The original uses `Base::IRC` — since we're now inside `DenseFunctionBase` itself, `IRC` is
directly accessible. Both `IRC` and `Base::IRC` would work; the snippet above uses `IRC` for
clarity.

- [ ] **Step 10: Re-parent DenseFirstDerivatives from DenseFunctionBase (4 files)**

In `include/tycho/detail/vf/derivatives/dense_derivatives.h`, change:
```cpp
// Before (line 29):
struct DenseFirstDerivatives : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
```
to:
```cpp
struct DenseFirstDerivatives : DenseFunctionBase<Derived, IR, OR> {
    using Base = DenseFunctionBase<Derived, IR, OR>;
```

Also change the `#include` on line 17 from:
```cpp
#include "tycho/detail/vf/core/dense_function.h"
```
to:
```cpp
#include "tycho/detail/vf/core/dense_function_base.h"
```

In `include/tycho/detail/vf/derivatives/dense_fdiff_fwd.h`, change:
```cpp
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::FDiffFwd>
    : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
```
to:
```cpp
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::FDiffFwd>
    : DenseFunctionBase<Derived, IR, OR> {
    using Base = DenseFunctionBase<Derived, IR, OR>;
```

Same pattern in `dense_fdiff_cent_array.h` and `dense_autodiff_fwd.h`.

- [ ] **Step 11: Update extern template files**

In `include/tycho/detail/vf/extern_templates.h`:
- Remove line 16: `#include "tycho/detail/vf/core/dense_scalar_function_base.h"`
- Remove line 33: `extern template struct DenseScalarFunctionBase<GenericFunction<-1, 1>, -1>;`
- Remove line 54: `extern template struct DenseScalarFunctionBase<Constant<-1, 1>, -1>;`
- Update the comment on lines 29-30 to remove DenseScalarFunctionBase reference.

In `src/vf/extern_template_instantiations.cpp`:
- Remove the matching `template struct DenseScalarFunctionBase<...>` lines (lines 24 and 41).
- Remove `#include "tycho/detail/vf/core/dense_scalar_function_base.h"` if present.

- [ ] **Step 12: Delete dense_function.h and dense_scalar_function_base.h**

```bash
rm include/tycho/detail/vf/core/dense_function.h
rm include/tycho/detail/vf/core/dense_scalar_function_base.h
```

- [ ] **Step 13: Build to verify DenseFunction/ScalarBase elimination compiles**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light
```

### Step 2d: Eliminate DenseDerivatives

- [ ] **Step 14: Re-parent VectorFunction from DenseSecondDerivatives**

In `include/tycho/detail/vf/core/vector_function.h`, change:
```cpp
// Before (lines 27-29):
struct VectorFunction : DenseDerivatives<Derived, IR, OR, Jm, Hm> {
    using Base = DenseDerivatives<Derived, IR, OR, Jm, Hm>;
    DENSE_FUNCTION_BASE_TYPES(Base)
```
to:
```cpp
struct VectorFunction : DenseSecondDerivatives<Derived, IR, OR, Jm, Hm> {
    using Base = DenseSecondDerivatives<Derived, IR, OR, Jm, Hm>;
    VF_TYPE_ALIASES(Base)
```

(Note: `DENSE_FUNCTION_BASE_TYPES` was already renamed to `VF_TYPE_ALIASES` in Direction 2.)

- [ ] **Step 15: Remove DenseDerivatives class from dense_derivatives.h**

Remove lines 38-42 from `include/tycho/detail/vf/derivatives/dense_derivatives.h`:
```cpp
template <class Derived, int IR, int OR, DenseDerivativeMode Jmode, DenseDerivativeMode Hmode>
struct DenseDerivatives : DenseSecondDerivatives<Derived, IR, OR, Jmode, Hmode> {
    using Base = DenseSecondDerivatives<Derived, IR, OR, Jmode, Hmode>;
    DENSE_FUNCTION_BASE_TYPES(Base)
};
```

Keep the enum and DenseFirstDerivatives/DenseSecondDerivatives primary templates.

### Step 2e: Clean up includes for deleted files

- [ ] **Step 16: Remove includes of deleted files from umbrella headers**

Search for any remaining `#include` of the 4 deleted files:
```bash
grep -rn "dense_function\.h\|dense_scalar_function_base\.h\|computable\.h\|crtp_base\.h" include/ src/ extensions/ tests/
```

Remove any stale includes found. Key locations:
- Umbrella headers in `src/vf/`, `src/optimal_control/`, etc.
- Any VF header that included `dense_function.h` directly

- [ ] **Step 17: Build and run C++ tests**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests tycho_tests_light && ctest --output-on-failure
```

Expected: All tests pass.

- [ ] **Step 18: Commit Direction 1**

```bash
git add -u && git commit -m "refactor(vf): flatten CRTP chain from 8 to 5 levels"
```

---

## Task 3: Full Verification

- [ ] **Step 1: Run C++ unit tests**

```bash
cd /home/ghecht/Projects/tycho/build && ctest --output-on-failure
```

- [ ] **Step 2: Run Python examples**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

- [ ] **Step 3: Run C++ brachistochrone example**

```bash
./build/examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
```

Expected: "Optimal Solution Found", objective ~ 1.8013 s.

- [ ] **Step 4: Run benchmark regression check**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 bench_all && ../bench/bench_track.sh compare
```

- [ ] **Step 5: Address any failures**

If any verification step fails, fix the issue and amend the relevant commit.
