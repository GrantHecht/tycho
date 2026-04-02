# Enzyme Forward-Mode AD — Experimental VectorFunction Implementation

**Status:** Draft
**Date:** 2026-04-01
**Scope:** Add an optional `EnzymeFwd` derivative mode alongside the existing `AutodiffFwd`, `FDiffFwd`, and `FDiffCentArray` modes. Disabled by default (`ENABLE_ENZYME_AD=OFF`). No changes to existing derivative paths.

---

## Motivation

The autodiff library computes derivatives via operator-overloading on dual-number types.
Enzyme operates at the LLVM IR level — it differentiates *optimized* machine code rather
than C++ abstractions. This has two potential advantages:

1. **Faster derivative code.** Enzyme differentiates after the compiler has inlined Eigen
   expression templates, eliminated dead code, and simplified control flow. The generated
   derivative code is then further optimized by LLVM.

2. **No scalar-type template requirement.** Enzyme differentiates concrete `double` code
   directly. User ODE models could (optionally) be written without the
   `template <class InType, class OutType>` scaffolding. (Note: in Phase 1 this
   simplification is available but not required — existing templated `compute_impl`
   functions work unchanged.)

The goal is an experimental, benchmarkable side-by-side comparison with autodiff on
real Tycho workloads (brachistochrone, CR3BP, modified equinoctial elements) before
committing to any migration.

---

## Background: Existing Derivative Architecture

The VectorFunction CRTP hierarchy dispatches derivatives via two template parameters:

```
VectorFunction<Derived, IR, OR, Jm, Hm>
    └─ DenseDerivatives<Derived, IR, OR, Jm, Hm>
        └─ DenseSecondDerivatives<Derived, IR, OR, Jm, Hm>   ← Hessian (mode Hm)
            └─ DenseFirstDerivatives<Derived, IR, OR, Jm>     ← Jacobian (mode Jm)
                └─ DenseFunction<Derived, IR, OR>
```

Each derivative mode is a partial specialization of `DenseFirstDerivatives` (Jacobian)
and `DenseSecondDerivatives` (Hessian). The mode is selected at compile time via the
`DenseDerivativeMode` enum:

```cpp
enum class DenseDerivativeMode { Analytic, FDiffFwd, FDiffCentArray, AutodiffFwd };
```

**Key files:**

| File                                                           | Role                                    |
| -------------------------------------------------------------- | --------------------------------------- |
| `include/tycho/detail/vf/derivatives/dense_derivatives.h`      | Enum + primary templates                |
| `include/tycho/detail/vf/derivatives/dense_autodiff_fwd.h`     | AutodiffFwd specializations (131 lines) |
| `include/tycho/detail/vf/derivatives/dense_fdiff_fwd.h`        | FDiffFwd specializations                |
| `include/tycho/detail/vf/derivatives/dense_fdiff_cent_array.h` | FDiffCentArray specializations          |
| `include/tycho/detail/vf/core/vector_function.h`               | Includes all derivative headers         |

Adding a new mode requires:
1. A new enum value in `DenseDerivativeMode`
2. A new header with `DenseFirstDerivatives` and `DenseSecondDerivatives` specializations
3. An `#include` in `vector_function.h`

No changes to any other part of the VF hierarchy, type erasure layer, optimizer, or
phase infrastructure.

---

## Implementation Plan

### Phase 1: Enzyme Jacobians (First Derivatives)

**Goal:** `DenseFirstDerivatives<Derived, IR, OR, EnzymeFwd>` computes Jacobians via
Enzyme forward-mode AD. Hessians fall back to existing modes (AutodiffFwd or FDiffFwd)
via the independent `Hm` template parameter.

#### 1a. CMake: `ENABLE_ENZYME_AD` Option

Add to `CMakeLists.txt` after the compiler detection section:

```cmake
option(ENABLE_ENZYME_AD
    "Enable experimental Enzyme automatic differentiation (requires Clang + Enzyme plugin)"
    OFF)
```

When `ON`:
- **Verify Clang.** `FATAL_ERROR` if `CMAKE_CXX_COMPILER_ID` is not `Clang`.
- **Detect LLVM major version** from the compiler (`clang++ --version`).
- **Find Enzyme.** Try `find_package(Enzyme CONFIG)` first (provides `ClangEnzymeFlags`
  imported target). Fall back to `find_library(ClangEnzyme-${LLVM_MAJOR})` with Homebrew
  hints (`/opt/homebrew/opt/enzyme/lib`, `/usr/local/opt/enzyme/lib`).
- **Version match.** If `find_package` succeeds, verify `Enzyme_LLVM_VERSION_MAJOR`
  matches the compiler's LLVM major. If using `find_library`, the library name
  (`ClangEnzyme-22`) encodes the version implicitly.
- **`try_compile` smoke test.** Compile a minimal program calling `__enzyme_autodiff`
  with `-fplugin=${ENZYME_PLUGIN} -O1` to verify the plugin loads and works.
  `FATAL_ERROR` on failure with a diagnostic message.
- **Set variables:**
  - `TYCHO_ENZYME_PLUGIN` — absolute path to `.dylib`/`.so`
  - `TYCHO_ENZYME_COMPILE_FLAGS` — `-fplugin=${TYCHO_ENZYME_PLUGIN}`
  - `add_compile_definitions(TYCHO_HAS_ENZYME_AD)` — globally visible macro

No new external dependency is pulled in (Enzyme is a pre-installed system tool, like
clang-format). The `dep/` directory is unmodified.

**Installation (not part of this plan, but documented):**
- macOS: `brew install enzyme`
- Linux: build from source against system LLVM (no apt package)

#### 1b. Enum Extension

In `dense_derivatives.h`, add `EnzymeFwd` to the enum:

```cpp
enum class DenseDerivativeMode {
    Analytic,
    FDiffFwd,
    FDiffCentArray,
    AutodiffFwd,
    EnzymeFwd,       // Enzyme LLVM-based forward-mode AD (requires ENABLE_ENZYME_AD)
};
```

The enum value is always present (no `#ifdef`). This keeps the enum ABI stable and
avoids conditional compilation in user code that references the enum. If someone uses
`EnzymeFwd` without `ENABLE_ENZYME_AD=ON`, they get a clear compile error (see 1c).

#### 1c. New Header: `dense_enzyme_fwd.h`

**Path:** `include/tycho/detail/vf/derivatives/dense_enzyme_fwd.h`

Structure:

```cpp
#pragma once
#include "tycho/detail/vf/derivatives/dense_derivatives.h"

namespace tycho::vf {

#ifndef TYCHO_HAS_ENZYME_AD

// Stub: compile error if EnzymeFwd is used without Enzyme enabled
template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::EnzymeFwd>
    : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
    template <class> struct dependent_false : std::false_type {};
    static_assert(dependent_false<Derived>::value,
        "DenseDerivativeMode::EnzymeFwd requires CMake option ENABLE_ENZYME_AD=ON "
        "and the Enzyme Clang plugin installed.");
};

// Same stub for DenseSecondDerivatives<..., EnzymeFwd>
// ...

#else // TYCHO_HAS_ENZYME_AD defined

// ── Enzyme marker declarations ──────────────────────────────────────────────
// These are NOT library functions. Enzyme intercepts calls to these symbols
// at the LLVM IR level and replaces them with synthesised derivative code.
// No Enzyme headers are included — only these forward declarations are needed.

extern "C" {
extern int enzyme_dup;
extern int enzyme_const;
extern int enzyme_dupnoneed;
extern int enzyme_width;
}

template <typename RT, typename... Args>
RT __enzyme_fwddiff(Args...);

// ── Jacobian via Enzyme forward mode ────────────────────────────────────────
template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::EnzymeFwd>
    : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_) const;
};

// ── Hessian via Enzyme forward mode ─────────────────────────────────────────
template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseSecondDerivatives<Derived, IR, OR, JMode, DenseDerivativeMode::EnzymeFwd>
    : DenseFirstDerivatives<Derived, IR, OR, JMode> {
    using Base = DenseFirstDerivatives<Derived, IR, OR, JMode>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template <class InType, class OutType, class JacType, class AdjGradType,
              class AdjHessType, class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_,
        ConstVectorBaseRef<AdjVarType> adjvars) const;
};

#endif // TYCHO_HAS_ENZYME_AD
} // namespace tycho::vf
```

**Jacobian algorithm (`compute_jacobian_impl`):**

The approach mirrors the existing autodiff column-by-column structure but replaces dual
numbers with Enzyme's `__enzyme_fwddiff`:

1. Define a static wrapper function that calls `this->derived().compute_impl()` with
   `double` (via `Eigen::Map`) — a plain function pointer Enzyme can differentiate.
2. For each input dimension `i`, create a unit tangent vector `dx` with `dx[i] = 1.0`.
3. Call `__enzyme_fwddiff` with:
   - `enzyme_const` on the `Derived*` self pointer (object data members are constants)
   - `enzyme_dup` on the input data pointer (primal `x` + tangent `dx`)
   - `enzyme_dup` on the output data pointer (primal `fx` + tangent `dfx`)
   - `enzyme_const` on dimension arguments
4. Read `dfx[j]` → `jx(j, i)` for each output `j`.
5. On the first iteration, copy primal `fx` to the output.

**Why this works:**
- `compute_impl` is defined in headers (CRTP), so the compiler inlines it into the
  wrapper before Enzyme runs — Enzyme sees the full function body.
- `enzyme_const` on `self` means Enzyme doesn't differentiate through stored parameters
  (e.g., `mu` in CR3BPAD) — correct, since we want ∂f/∂x not ∂f/∂params.
- The wrapper uses raw `double*` pointers, avoiding any Eigen abstraction in the
  Enzyme-visible signature. Eigen::Map construction inside the wrapper is inlined away.

**Include in `vector_function.h`:**

```cpp
// In vector_function.h, after the other derivative includes:
#include "tycho/detail/vf/derivatives/dense_enzyme_fwd.h"
```

This is unconditional — the header handles `#ifdef TYCHO_HAS_ENZYME_AD` internally.
When Enzyme is disabled, the header provides only the `static_assert` stub.

#### 1d. Enzyme Compile Flags on Targets

Enzyme's `-fplugin=` flag must be on any translation unit that **instantiates**
`EnzymeFwd`-mode VectorFunctions. The PCH does not need the flag (it only parses
template definitions, not instantiations).

For the test target (the only consumer in Phase 1):

```cmake
if(ENABLE_ENZYME_AD)
    add_executable(tycho_enzyme_tests
        tests/cpp/enzyme/test_enzyme_jacobian.cpp
        tests/cpp/enzyme/test_enzyme_hessian.cpp    # Phase 2
    )
    target_compile_options(tycho_enzyme_tests PRIVATE
        ${TYCHO_ENZYME_COMPILE_FLAGS}
    )
    target_link_libraries(tycho_enzyme_tests PRIVATE
        tycho_core GTest::gtest_main
    )
    # PCH: reuse the main pch (no Enzyme-specific PCH needed).
    # Enzyme operates on IR after PCH merge — template instantiations in
    # the consumer TU are visible to the plugin.
    target_precompile_headers(tycho_enzyme_tests REUSE_FROM pch)
    gtest_discover_tests(tycho_enzyme_tests)
endif()
```

**PCH interaction:** The Enzyme plugin injects preprocessor macros
(`ENZYME_VERSION_MAJOR`, etc.) and a virtual filesystem overlay at AST level. Since
we don't use these macros and the PCH was compiled without them, there's a theoretical
mismatch. In practice, this is harmless because:
- The injected macros are never referenced in our code
- The virtual filesystem only matters for `#include <enzyme/enzyme>` (which we don't use)
- Template bodies from the PCH are instantiated in the consumer TU and lowered to LLVM IR
  normally — the Enzyme pass then operates on that IR

If PCH reuse causes issues in practice, the fallback is
`set_target_properties(tycho_enzyme_tests PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)`.
The test target has few TUs, so PCH savings are minimal anyway.

#### 1e. Tests

**Path:** `tests/cpp/enzyme/test_enzyme_jacobian.cpp`

Tests mirror the existing VF test structure:

| Test                        | What it validates                                                                                                    |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `EnzymeJacobianVsCentralFD` | Enzyme Jacobian matches central FD (tol 1e-7) on BrachODE-like function using `EnzymeFwd`                            |
| `EnzymeJacobianVsAutodiff`  | Enzyme Jacobian matches autodiff Jacobian (tol 1e-14) on same function                                               |
| `EnzymeJacobianCR3BP`       | Enzyme on CR3BP dynamics (7→6, transcendental + norm)                                                                |
| `EnzymeJacobianMEE`         | Enzyme on modified equinoctial elements (9→6)                                                                        |
| `EnzymeMixedMode`           | `VectorFunction<D, IR, OR, EnzymeFwd, AutodiffFwd>` — Enzyme Jacobian + autodiff Hessian, verify Hessian consistency |
| `EnzymeGenericFunction`     | Type-erase an EnzymeFwd function into `GenericFunction<-1,-1>`, verify compute + Jacobian still work                 |

Test ODE models: reuse the existing `BrachODE` from `test_utils.h` redeclared with
`EnzymeFwd`, plus `CR3BPAD` and `ModifiedDynamicsAD` from `Tycho_Extensions.h`
redeclared with `EnzymeFwd`. (The redeclarations are trivial — change the mode template
arguments.)

```cpp
// Example: Brachistochrone ODE with Enzyme Jacobians
struct BrachEnzyme : VectorFunction<BrachEnzyme, 5, 3,
                                     DenseDerivativeMode::EnzymeFwd,
                                     DenseDerivativeMode::AutodiffFwd> {
    // ... same compute_impl as BrachODE (templated on InType/OutType)
};
```

#### 1f. Benchmark Scaffold

**Path:** `bench/cpp/bench_enzyme.cpp`

Google Benchmark comparing Jacobian computation time:
- `BM_Jacobian_Autodiff/BrachODE` vs `BM_Jacobian_Enzyme/BrachEnzyme`
- `BM_Jacobian_Autodiff/CR3BPAD` vs `BM_Jacobian_Enzyme/CR3BPEnzyme`
- Same input vectors, same number of iterations, `-O2` for both

This benchmark is the primary decision criterion for whether to invest further in
Enzyme. Expected outcome: Enzyme should be competitive or faster for Jacobians due to
post-optimization differentiation.

Build (only when both `BUILD_CPP_BENCHMARKS` and `ENABLE_ENZYME_AD` are ON):

```cmake
if(ENABLE_ENZYME_AD AND BUILD_CPP_BENCHMARKS)
    add_executable(bench_enzyme bench/cpp/bench_enzyme.cpp)
    target_compile_options(bench_enzyme PRIVATE ${TYCHO_ENZYME_COMPILE_FLAGS})
    target_link_libraries(bench_enzyme PRIVATE tycho_core benchmark::benchmark_main)
    target_precompile_headers(bench_enzyme REUSE_FROM pch)
endif()
```

---

### Phase 2: Enzyme Hessians (Second Derivatives)

**Goal:** `DenseSecondDerivatives<Derived, IR, OR, JMode, EnzymeFwd>` computes adjoint
Hessians via Enzyme. Two candidate strategies; benchmark both:

#### Strategy A: Forward-over-Reverse (preferred)

Compute the adjoint gradient `g(x) = J(x)^T * lambda` using Enzyme reverse mode
(`__enzyme_autodiff`), then forward-differentiate `g(x)` to obtain Hessian columns.

- **Cost:** O(IR) forward-mode sweeps of the reverse-mode gradient function.
  Each sweep costs ~4× the primal (reverse) + 2× (forward of reverse).
  Total: ~O(IR × primal), comparable to the autodiff nested-dual approach.

- **Advantage:** Enzyme reverse mode can be more efficient than forward mode when
  OR << IR (common for objective functions where OR=1). The forward-over-reverse
  composition is well-supported in Enzyme.

- **Risk:** Nested Enzyme calls (forward wrapping reverse) have known fragility in some
  edge cases. The implementation should include a `try_compile` test for forward-over-reverse
  at CMake configure time.

#### Strategy B: Forward-over-Forward (fallback)

Nest `__enzyme_fwddiff` calls. Outer loop seeds input `i`, inner function computes
Jacobian column `j` via Enzyme. The outer Enzyme call differentiates the inner Enzyme
call to extract `∂²f/∂x_i ∂x_j`.

- **Cost:** O(IR²) forward sweeps — same asymptotic cost as autodiff nested duals.
- **Advantage:** Only uses forward mode (no reverse mode needed), simpler implementation.
- **Risk:** Forward-over-forward is less exercised in Enzyme's C++ test suite.

#### Implementation approach

1. Implement Strategy A (forward-over-reverse) first.
2. Add a `try_compile` test for nested Enzyme calls at configure time.
3. If Strategy A fails on a supported platform, fall back to Strategy B.
4. Benchmark both against the autodiff Hessian (the primary comparison target).

#### Marker declarations

Add to `dense_enzyme_fwd.h`:

```cpp
template <typename RT, typename... Args>
RT __enzyme_autodiff(Args...);    // reverse mode marker
```

#### Hessian tests

**Path:** `tests/cpp/enzyme/test_enzyme_hessian.cpp`

| Test                              | What it validates                                                   |
| --------------------------------- | ------------------------------------------------------------------- |
| `EnzymeHessianSymmetry`           | H == H^T                                                            |
| `EnzymeHessianVsAutodiff`         | Enzyme Hessian matches autodiff Hessian (tol 1e-12)                 |
| `EnzymeHessianAdjointConsistency` | g == J^T * lambda                                                   |
| `EnzymeHessianCR3BP`              | Hessian on CR3BP dynamics                                           |
| `EnzymeFullPipeline`              | `VectorFunction<D, IR, OR, EnzymeFwd, EnzymeFwd>` — all-Enzyme path |

#### Hessian benchmarks

Extend `bench_enzyme.cpp`:
- `BM_Hessian_Autodiff/CR3BPAD` vs `BM_Hessian_Enzyme/CR3BPEnzyme`
- `BM_Hessian_Autodiff/ModifiedDynamicsAD` vs `BM_Hessian_Enzyme/ModifiedDynamicsEnzyme`

---

### Phase 3: Batch Forward Mode Optimization

**Goal:** Use Enzyme's `enzyme_width` to compute multiple Jacobian columns
simultaneously, reducing the per-column overhead.

Instead of IR separate `__enzyme_fwddiff` calls (each seeding one tangent direction),
batch W tangent directions per call:

```cpp
// Pseudocode: batch forward mode
constexpr int W = 4;  // batch width (tunable)
for (int i = 0; i < ir; i += W) {
    int w = std::min(W, ir - i);
    // Set up W tangent vectors: identity columns i..i+w-1
    // Single __enzyme_fwddiff call with enzyme_width = w
    // Enzyme generates one function that propagates W tangents simultaneously
    // Extract W columns of the Jacobian from the result
}
```

**Implementation considerations:**
- `enzyme_width` is a compile-time constant in the generated function. For dynamic IR,
  use a fixed chunk size (e.g., 4 or 8) and handle the remainder.
- The return type must be a struct of W doubles per output. Define a helper struct.
- Benchmark different chunk sizes to find the sweet spot for typical problem dimensions
  (IR 5–20 for most astrodynamics ODEs).

**Decision gate:** Only pursue if Phase 1 benchmarks show Enzyme is competitive.
Batch mode adds implementation complexity but could yield significant speedups for
larger IR by enabling SIMD-width derivative propagation.

---

### Phase 4: Non-Templated `compute_impl` Support

**Goal:** Allow users to write ODE models with plain `double` types instead of
the `template <class InType, class OutType>` pattern. This is the user-facing payoff
of Enzyme — simpler C++ ODE definitions.

```cpp
// Current (autodiff-compatible): must be templated
struct MyODE : VectorFunction<MyODE, 7, 6, AutodiffFwd, AutodiffFwd> {
    template <class InType, class OutType>
    void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        // ... Scalar arithmetic ...
    }
};

// New (Enzyme-only): plain double, no template
struct MyODE : VectorFunction<MyODE, 7, 6, EnzymeFwd, EnzymeFwd> {
    void compute_impl(ConstVectorBaseRef<Eigen::VectorXd> x,
                      ConstVectorBaseRef<Eigen::VectorXd> fx_) const {
        // ... double arithmetic, can use any library ...
    }
};
```

**Implementation:**
- The `compute_wrapper` in `dense_enzyme_fwd.h` already calls `compute_impl` with
  `Eigen::Map<VectorXd>` — a `double`-typed Eigen type. If `compute_impl` is not
  a template but a concrete function accepting `VectorXd`-compatible refs, the wrapper
  still compiles.
- The only issue is SuperScalar: `ComputableBase::compute()` tries to instantiate
  `compute_impl<SuperScalarType>`. A non-templated `compute_impl` would fail this
  instantiation.
- **Solution:** Add a concept/SFINAE guard in `ComputableBase::compute()` that skips
  SuperScalar dispatch when `compute_impl` is not a template or not instantiable with
  SuperScalar types. `if constexpr (requires { derived().compute_impl(super_x, super_fx); })`

**Note:** This phase is independent of Phases 2–3 and can be done in any order.
It's lower priority because the templated pattern works fine with Enzyme — the
simplification is ergonomic, not functional.

---

## Risks and Mitigations

| Risk                           | Likelihood | Impact                                                                 | Mitigation                                                                                                                                                                                                           |
| ------------------------------ | ---------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Eigen SIMD incompatibility     | Medium     | Enzyme can't differentiate Eigen's SIMD intrinsics for some operations | The wrapper function uses `Eigen::Map` on raw `double*`, and typical dynamics code (trig, arithmetic, norm) doesn't generate problematic SIMD patterns. If hit, add `-DEIGEN_DONT_VECTORIZE` to Enzyme targets only. |
| PCH + Enzyme plugin conflict   | Low        | Macro injection mismatch between PCH and consumer TU                   | Fallback: `DISABLE_PRECOMPILE_HEADERS` on Enzyme targets (minimal build time impact for a small number of TUs).                                                                                                      |
| Forward-over-reverse fragility | Medium     | Nested Enzyme calls produce incorrect results or crash                 | `try_compile` at configure time; fall back to forward-over-forward or autodiff Hessians.                                                                                                                             |
| Enzyme version drift           | Low        | Homebrew updates Enzyme past tested version, breaks plugin loading     | Pin to tested LLVM major version in CMake; `try_compile` catches mismatches at configure time.                                                                                                                       |
| Compile-time regression        | Low–Medium | Enzyme plugin adds overhead to every TU it loads on                    | Plugin flag is applied ONLY to targets that use Enzyme (tests, benchmarks). Core library builds are unaffected.                                                                                                      |

---

## Files Modified / Created

### New files

| File                                                     | Purpose                                                                            |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `include/tycho/detail/vf/derivatives/dense_enzyme_fwd.h` | `DenseFirstDerivatives` + `DenseSecondDerivatives` specializations for `EnzymeFwd` |
| `tests/cpp/enzyme/test_enzyme_jacobian.cpp`              | Phase 1 Jacobian tests                                                             |
| `tests/cpp/enzyme/test_enzyme_hessian.cpp`               | Phase 2 Hessian tests                                                              |
| `bench/cpp/bench_enzyme.cpp`                             | Enzyme vs autodiff benchmarks                                                      |

### Modified files

| File                                                      | Change                                                         |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| `include/tycho/detail/vf/derivatives/dense_derivatives.h` | Add `EnzymeFwd` to `DenseDerivativeMode` enum                  |
| `include/tycho/detail/vf/core/vector_function.h`          | Add `#include "dense_enzyme_fwd.h"`                            |
| `CMakeLists.txt`                                          | `ENABLE_ENZYME_AD` option, find Enzyme, test/benchmark targets |

### Files NOT modified

- `dense_autodiff_fwd.h` — unchanged, autodiff path is untouched
- `pch.h` — no Enzyme headers in the PCH
- `src/bindings/` — no Python bindings for EnzymeFwd in Phase 1
- `dep/` — no new submodule (Enzyme is a system-installed tool)

---

## Sequencing and Decision Gates

```
Phase 1: Enzyme Jacobians
  ├─ 1a. CMake integration
  ├─ 1b. Enum extension
  ├─ 1c. dense_enzyme_fwd.h (Jacobian impl)
  ├─ 1d. Test target setup
  ├─ 1e. Jacobian tests (must pass)
  └─ 1f. Jacobian benchmarks
         │
         ▼
    ┌─────────────────────────────────────────┐
    │ DECISION GATE: Is Enzyme Jacobian       │
    │ performance competitive with autodiff?  │
    │ (within 2× on brachistochrone/CR3BP)    │
    └─────────────────────────────────────────┘
         │ Yes                          │ No
         ▼                              ▼
Phase 2: Enzyme Hessians          STOP. Keep as
  ├─ Hessian impl (FoR or FoF)    experimental
  ├─ Hessian tests                 reference only.
  └─ Hessian benchmarks
         │
         ▼
    ┌─────────────────────────────────────────┐
    │ DECISION GATE: Is the full Enzyme path  │
    │ (Jac + Hess) faster end-to-end?         │
    └─────────────────────────────────────────┘
         │ Yes                          │ No
         ▼                              ▼
Phase 3: Batch optimization       Phase 4 only
Phase 4: Non-templated API        (ergonomic benefit
                                   without perf win)
```

---

## Build & Test Commands

```bash
# Install Enzyme (macOS)
brew install enzyme

# Configure with Enzyme enabled
cmake --preset macos-llvm-release -DENABLE_ENZYME_AD=ON -DBUILD_CPP_TESTS=ON

# Build core + Enzyme tests (Enzyme flag only on test target)
cd build && ninja -j2 tycho_enzyme_tests

# Run Enzyme tests
ctest --output-on-failure -R enzyme

# Benchmarks
cmake --preset macos-llvm-release -DENABLE_ENZYME_AD=ON -DBUILD_CPP_BENCHMARKS=ON
cd build && ninja -j2 bench_enzyme
./bench/cpp/bench_enzyme --benchmark_repetitions=5
```
