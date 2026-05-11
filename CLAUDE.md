# Tycho — Claude Code Memory

## Project Overview

Tycho is an independently maintained fork of [ASSET (AlabamaASRL/asset_asrl)](https://github.com/AlabamaASRL/asset_asrl),
a high-performance C++/Python library for trajectory design and optimal control.
The core use cases are general optimal control problems (solved via direct collocation)
and space trajectory optimization. The built-in optimizer is called **PSIOPT**
(a high-performance interior-point solver).

The Python-facing module is `_tychopy` (nanobind extension) imported via the `tychopy` package.
The top-level C++ namespace is `tycho`.

## A Word of Caution for Multi-Agent Workloads

Compiling tycho is very computationally expensive, and you MUST therefore be careful about
how many parallel jobs are being used in any given build, and how many agents are building
simultaneously.

As a rule of thumb:
- macOS (Apple Silicon): ALWAYS use -j6 for builds
- Linux / Windows (32 GB+): ALWAYS use -j6 for builds
- DO NOT PERFORM MORE THAN 2 SIMULTANEOUS BUILDS AT ONCE
- **NEVER launch a second build if one is already running** — not even in the background.
  If you started a build with `run_in_background`, WAIT for the completion notification
  before running any other `ninja` or `cmake --build` command. Two concurrent builds WILL
  OOM the system. This applies within a single conversation, not just across agents.

## Repository Structure

Top-level files of note: `CMakeLists.txt` (root build), `CMakePresets.json`, `CMakeSettings.json` (MSVC),
`setup.py`, `requirements.txt`, `Dockerfile`, `Dockerfile-dev`.

```
include/                Public C++ API headers
  tycho/
    tycho.h             Master umbrella — includes all public modules
    vector_functions.h  VectorFunction subsystem umbrella
    typedefs.h          Eigen type aliases umbrella
    utils.h             Utilities umbrella
    integrators.h       Integrators umbrella
    optimal_control.h   Optimal control umbrella
    solvers.h           Solvers umbrella
    astro.h             Astrodynamics umbrella
    detail/             Template implementation bodies (included automatically)
      utils/            Threading, math helpers, type utilities, CRTP base
      typedefs/         Eigen type aliases
      vf/               VectorFunction implementation (core/, expressions/, operators/,
                          derivatives/, scaling/, type_erasure/, common/)
      integrators/      RK steppers and coefficients
      optimal_control/  Phase/ODE transcription (core/, phase/, transcription/,
                          interp/, builder/)
      solvers/          PSIOPT and NLP layer (linear/ for sparse solver interfaces)
      astro/            Astrodynamics models

src/                    C++ source code (private implementation)
  tycho_internal.h      Internal aggregate (forwards to include/tycho/tycho.h)
  pch.h / pch.cpp       Precompiled header
  bindings/             ALL Python binding code — nanobind .cpp files, *_bind.h
                          headers, function_registry.h (TychoBind<T> trait),
                          type_casters.h, tycho_module.cpp
  vf/                   tycho_vector_functions.h aggregate + function_domains.cpp
  optimal_control/      tycho_optimal_control.h aggregate + snake_case .cpp files
  solvers/              tycho_solvers.h aggregate + snake_case .cpp files
  astro/                tycho_astro.h aggregate + snake_case .cpp files
  integrators/          tycho_integrators.h aggregate (header-only)
  utils/                tycho_utils.h aggregate + private utility .cpp files
  typedefs/             tycho_typedefs.h aggregate (header-only)

tychopy/                Python package (pure-Python layer over _tychopy extension)
  VectorFunctions/      Python-side VectorFunction utilities
  OptimalControl/       Pure-Python ODE base class, mesh-error plotting
  Solvers/              Python solver helpers
  Astro/                Astrodynamics helpers — frames, models, constants,
                          SPICE read, date utilities, Lambert wrappers
  Utils/                Python utility helpers
  Extensions/           Python extension helpers
  test/                 Test suite (pytest-style, organised by subsystem)

dep/                    Vendored submodule dependencies
  eigen/                Eigen (MPL-2.0)
  fmt/                  {fmt} (MIT)
  nanobind/             nanobind (BSD)

tests/                  C++ unit tests (Google Test, organised by subsystem)
bench/                  Benchmark suite and tracking
  MACBENCH.md           macOS benchmarking procedure
  WINBENCH.md           Windows benchmarking procedure
  bench_track.sh        Benchmark tracking script (record, compare, list)
  cpp/                  Google Benchmark source files

extensions/             Optional extension module (Tycho_Extensions.cpp/.h)
examples/               Example programs
  cpp_examples/         C++ example programs
  python_examples/      Python example scripts (38 examples)
docs/                   Documentation source tree
  source/               Public Sphinx site source (Markdown + RST)
  dev/                  Internal engineering notes (raw markdown, not rendered)
    notes/              EnzymeAD work, build perf, design rationales
    analysis/           Performance analyses
    plans/              Engineering work plans
  superpowers/specs/    Brainstorm specs (gitignored, local-only)
doc-legacy/             Archived asset_asrl carryover (frozen, not rendered)
notices/                Third-party license notices — DO NOT modify or delete
```

## Key Concepts

- **VectorFunction** — the core DSL for defining dynamics, constraints, and objectives.
  Everything in the problem definition layer is expressed as a VectorFunction.
  This is the central abstraction of the library; treat it with care.
- **Phase** — the core optimal control object. Multiple phases can be linked together
  for multi-phase trajectory problems.
- **PSIOPT** — the bundled interior-point nonlinear optimizer.
- **Collocation** — the transcription method used to convert continuous optimal control
  problems into finite-dimensional NLPs.
- **Astrodynamics** — the primary application domain, though the library is general.

## Technical Details

- **VectorFunction implementation:** `docs/dev/notes/VectorFunction.md`
- **Python bindings (nanobind):** `docs/dev/notes/Bindings.md`

## Build System

This is a CMake + nanobind project. The output is a nanobind shared library
(`_tychopy.cpython-<ver>-<platform>.so`) plus the pure-Python `tychopy/` package, both
installed directly into the active Python environment's site-packages by the build step.

**Always use the `tycho` conda environment** for all Python and build operations.
Activate it before configuring — CMake auto-detects Python from `$PATH`:

```bash
conda activate tycho
```

Each platform has a corresponding CMake preset in `CMakePresets.json`. **Always use the
preset for your platform** — do not configure manually.

| Platform         | Configure preset        | Build parallelism |
| ---------------- | ----------------------- | ----------------- |
| macOS (Apple Si) | `macos-llvm-release`    | `-j6`             |
| Linux / WSL2     | `linux-clang-release`   | `-j6`             |
| Windows x64      | `x64-Clang-Release`     | `-j6`             |

**Note:** The `-j` values above are upper bounds — use lower values on
RAM-constrained systems. The `heavy_compile` ninja job pool auto-throttles
heavy TUs (1 slot per 10 GB RAM), so higher `-j` values are safe: light TUs
fill extra slots while heavy TUs are limited. On 32 GB, `-j6` is safe
(pool depth 3); on 16 GB, use `-j4` (pool depth 1).

**Build memory and ninja job pools:** Many binding, test, benchmark, and example
TUs consume 3-8 GB RAM each due to heavy template instantiation. A ninja job
pool named `heavy_compile` limits concurrent compilation of these TUs so they
cannot OOM the system. The pool depth is **auto-detected** from system RAM
(1 slot per 10 GB, minimum 1) and can be overridden:

```bash
cmake --preset <preset> -DTYCHO_HEAVY_COMPILE_JOBS=3   # e.g. for a 36 GB machine
```

Light TUs (~200 MB - 1 GB) run at the full `-j` level. This means you can
safely use higher `-j` values — ninja automatically throttles the heavy TUs
while keeping light TUs parallel.

```bash
cd build && ninja -j6 all      # safe — pool limits heavy TUs automatically
                               # use -j6 on 32 GB systems, -j4 on 16 GB
```

The `dep/` submodules (eigen, fmt, nanobind) must be initialised before the
first build. The cmake helpers in `cmake/git-submodule-*.cmake` do this automatically.

**Python environment (all platforms) — conda env named `tycho`:**
```bash
conda create -n tycho python=3.13
conda activate tycho
pip install numpy scipy matplotlib spiceypy
# Linux only: keep conda's libstdc++ aligned with the system compiler.
# Required when the build toolchain emits GLIBCXX symbols newer than what
# conda's default libstdcxx ships (otherwise importing _tychopy fails with
# 'GLIBCXX_X.Y.Z not found').
conda install -c conda-forge "libstdcxx-ng>=15"
```

**Linux libstdc++ note (bleeding-edge distros):** On systems whose system
GCC is newer than what conda-forge currently ships (e.g. Fedora 44 with
GCC 16, where conda-forge tops out at GCC 15 / GLIBCXX_3.4.34), the
`_tychopy` extension may require a `GLIBCXX_3.4.X` symbol the conda env
can't provide. Two workarounds until conda-forge catches up:

1. `LD_PRELOAD=/usr/lib64/libstdc++.so.6 python -c 'import _tychopy'`
   (force the loader to use system libstdc++ before conda's)
2. Build wheels with `auditwheel repair` (production path; bundles
   libstdc++ into the wheel for manylinux compatibility) — not yet wired
   into the project's CI.

This is a runtime concern only; `cmake --build build --target _tychopy`
and `pip wheel .` both succeed regardless. See PR #51 discussion for
detail.

**Local wheel builds** (`pip wheel .`) restrict the CMake build to the
`_tychopy` target via `[tool.scikit-build].cmake.targets` in
`pyproject.toml`. Stub generation targets (added to `ALL` by
`nanobind_add_stub`) import the freshly built `.so` and would fail under
the libstdc++ mismatch above; the wheel ships the committed source-tree
snapshot at `tychopy/_stubs/` instead. Refresh that snapshot via
`ninja tychopy_stubs_snapshot` after editing any binding.

### macOS (Apple Silicon)

**System tools:** `brew install llvm ninja ccache jq`

**Sparse solver:** Apple Accelerate (ships with macOS, detected automatically).

**First-time build:**
```bash
mkdir build && conda activate tycho
cmake --preset macos-llvm-release
cd build && ninja -j6 all
```

### Linux / WSL2 (Ubuntu)

**System tools:** `sudo apt install clang llvm llvm-dev libomp-dev lld ninja-build ccache`

**Sparse solver — Intel MKL (via oneAPI):**
```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update && sudo apt install intel-oneapi-mkl-devel
source /opt/intel/oneapi/setvars.sh   # add to ~/.bashrc
```

**First-time build:**
```bash
mkdir build && conda activate tycho
cmake --preset linux-clang-release
cd build && ninja -j6 all
```

### Windows x64

Uses LLVM/Clang with clang-cl frontend. Preset: `x64-Clang-Release`.
See `CMakePresets.json` for compiler paths. Sparse solver: Intel MKL.

### Subsequent builds (all platforms)

```bash
cd build && ninja -j<N> all    # N = 6 on all platforms (4 on RAM-constrained 16 GB)
```

### Enzyme AD (experimental)

`EnzymeAD` is an opt-in derivative mode for `VectorFunction`, gated by the
CMake option `ENABLE_ENZYME_AD` (default `OFF`). Enabling it requires:

1. A Clang installation matching the LLVM major version that Enzyme was built
   against.
2. The Enzyme Clang plugin available at a path discoverable via
   `CMAKE_PREFIX_PATH` (Enzyme ships an `EnzymeConfig.cmake`).

Default-`OFF` builds, the precompiled header, the Python bindings, and all
existing examples/tests are unaffected by this option.

**Override invocation (current dev machine — Linux):**

```bash
cmake --preset linux-clang-release \
  -DCMAKE_CXX_COMPILER=$HOME/Software/llvm-project/install/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$HOME/Software/Enzyme/install \
  -DENABLE_ENZYME_AD=ON \
  -DBUILD_CPP_TESTS=ON
# Add -DBUILD_CPP_BENCHMARKS=ON for benchmarks.
cd build && ninja -j6 tycho_enzyme_tests
ctest --test-dir tests/cpp/enzyme --output-on-failure
```

**Mode constraints (Phase 1):**

- `<EnzymeAD, FDiffFwd>` is the supported mixed pairing (Enzyme Jacobian + finite-difference Hessian).

**Mode constraints (Phase 2):**

- `<EnzymeAD, EnzymeAD>` is supported. The Hessian path uses Enzyme
  Forward-over-Reverse (`__enzyme_fwddiff(__enzyme_autodiff(...))`).
  FoR was selected over an early FoF prototype (4–9× faster on
  Hessian micro-cases); FoF is no longer cmake-selectable — see the
  Phase 7/7+ archive block below for revival conditions.

**Batched forward-mode Jacobian (Phase 3):**

The EnzymeAD Jacobian path uses `__enzyme_fwddiff` with `enzyme_width=W`
to propagate `W` tangent vectors per call (`enzyme_dupv` with a byte
stride for the shadow buffers, SoA-by-lane layout via column-major
`Eigen::Matrix<double, IR, W>`).  `TYCHO_ENZYME_BATCH_WIDTH` selects
`W` at configure time:

- `W=1` (Phase 1 fallback) — one tangent per call.
- `W=4` (default) — wins on every test case; Brach 1.17×, CR3BP 16×,
  MEE 1.97× faster than scalar.
- `W=8` — best for IR ≥ 8 workloads (MEE: **60× faster than the FDiff reference**);
  silently falls back to `W=1` tail loop when IR < 8.

The Hessian path is unaffected (it still uses scalar `__enzyme_fwddiff`
calls for the FoR outer pass; the inner pass is a separate `__enzyme_autodiff`
that batching does not apply to).  Hessian wins on CR3BP/MEE come from the
nested-AD strategy, not from `enzyme_width`.

**Non-templated `compute_impl` (Phase 4):**

When paired with `<EnzymeAD, EnzymeAD>` (or `<EnzymeAD, FDiffFwd>`), a
user dynamics struct may declare `compute_impl` with concrete-typed Eigen
`Map` arguments instead of the templated `<class InType, class OutType>`
form. The signature must match what the EnzymeAD wrapper passes:

```cpp
inline void compute_impl(
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> x,
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> fx) const {
    // ordinary double-typed arithmetic only
}
```

The user's body then uses ordinary `double` arithmetic — no need for the
dual-number-friendly templated form that templated derivative modes require.

Constraints:
- A non-templated `compute_impl` is supported only under the `<EnzymeAD, EnzymeAD>`
  pairing. FD modes (`FDiffFwd`, `FDiffCentArray`) require the templated form because
  their `compute_jacobian_impl` calls `compute()` with an `Eigen::Array`-typed Scalar
  for SuperScalar dispatch, which a concrete-`double` signature cannot accept.
- A non-templated `compute_impl` cannot be marked
  `Vectorizable<Derived>=true`. SuperScalar dispatch's `VectorImpl()`
  path requires templated `compute_impl`; this is a fundamental dispatch
  constraint, not Enzyme-specific.
- The non-templated form supports only the EnzymeAD-routed methods
  (`compute_jacobian`, `compute_jacobian_adjointgradient_adjointhessian`).
  Calling `.compute()` directly on a non-templated VF is a separate
  signature mismatch — use the templated form if `.compute()` is needed.

**Vectorizable VFs with EnzymeAD (Phase 5a + 5b):**

A VectorFunction with `static constexpr bool is_vectorizable = true;` may
be paired with `EnzymeAD`.  When invoked with `Eigen::Array<double, W, 1>`
SuperScalar Scalar:

- **Jacobian (Phase 5b — direct SIMD):** the templated `compute_impl`
  body is differentiated by Enzyme directly under the SuperScalar
  arithmetic.  One `__enzyme_fwddiff` call per input dim produces W
  per-lane tangents simultaneously via SIMD ops.  Per-lane speedup
  ~1.78× on Brach (small body, scalar arithmetic) and ~1.58× on CR3BP
  (`sqrt`-only, vectorises via `vsqrtpd`).
- **Hessian (Phase 6 — direct SIMD FoR, opt-out per VF):** the outer
  `__enzyme_fwddiff` over `enzyme_for_outer_wrapper_simd` propagates W
  lane-local Hessian columns per call; the inner `__enzyme_autodiff`
  runs reverse mode on SuperScalar arithmetic.  Gated at configure time
  by `TYCHO_ENZYME_SIMD_HESSIAN` (default ON); `OFF` falls back to the
  Phase 5a scalarize-per-lane Hessian.  A per-VF opt-out is available
  by inheriting from `tycho::vf::EnzymeSimdHessianUnsupported` (see
  `enzyme_simd_hessian_supported_v` for the predicate); use it for VFs
  whose body trips Enzyme's SS reverse-mode IR analysis (composite
  trig+sqrt+division — currently MEE-class bodies).
- **Hessian (Phase 7 / 7+ — FoF strategy, archived research path):**
  Forward-over-Forward direct-SIMD Hessian (with combined J+H output and
  doubly-batched variant) is **retained as a working but un-tested,
  un-benched reference** in `dense_enzyme.h` and is **no longer
  cmake-selectable** — CMakeLists.txt unconditionally defines
  `TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse` and never defines
  the FoF strategy macro; there is no cache variable or `-D` flag a user
  can pass to switch.  Bench results
  at the time of archival (BW=4): FoR-SIMD wins ~2× on Brach (1038 vs
  2102 ns) and ~5× on CR3BP (291 vs 1545 ns).  *(Both FoR-SIMD and
  FoF-SIMD lose to Phase 5a on Brach — see the rule-of-thumb section
  below for the regression context.)*  FoF's only qualitative
  advantage was on MEE-class composites (cos/sin/sqrt/division) where
  FoR-SIMD's inner reverse pass fails Enzyme TypeAnalysis — but no
  real PSIOPT workload has surfaced that case.  Doubly-batched variant
  proves nested `__enzyme_fwddiff(enzyme_width)` composition works but
  yields only a ~5% speedup over singly-batched (per-call overhead is
  not the dominant cost; body work dominates).  See the heavy archive
  docstring at the top of the FoF block in `dense_enzyme.h` for
  revival conditions.

**Trig-bearing bodies under Phase 5b (Eigen 5 + opt-in `tycho::math`):**

Eigen 5 ships vectorised `pcos<Packet4d>` / `psin<Packet4d>` (and
friends), but the lowered IR uses x86 round intrinsics + packed bitwise
sign-bit tricks that Enzyme cannot currently differentiate (upstream
issues [#689](https://github.com/EnzymeAD/Enzyme/issues/689),
[#2679](https://github.com/EnzymeAD/Enzyme/issues/2679),
[#2794](https://github.com/EnzymeAD/Enzyme/issues/2794)).  To use
Phase 5b SIMD primal evaluation on a VF that calls `sin`/`cos` under
EnzymeAD, route those calls through the wrapper API:

```cpp
#include <tycho/math.h>

template <class InType, class OutType>
inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
    using tycho::math::cos;   // ADL hits double (-> std::cos) and
    using tycho::math::sin;   // Eigen::Array<double,W,1> (-> per-lane libm).
    // ...
}
```

The SuperScalar overloads (`Eigen::Array<double, W, 1>` for W ∈ {2, 4, 8})
manually unroll into `W` scalar `std::cos(double)` / `std::sin(double)`
calls.  Each lowers to `@llvm.cos.f64` / `@llvm.sin.f64`, which Enzyme's
built-in handler differentiates cleanly under any `enzyme_width`.  Phase
3 batched tangents and Phase 5b SIMD primal compose naturally — the
surrounding arithmetic in `compute_impl` still vectorises across the W
lanes; only the trig itself is per-lane scalar.

Code paths that are NOT differentiated by Enzyme should keep using Eigen
directly — the analytic-integration path (e.g.
`include/tycho/detail/astro/mee_dynamics.h`) benefits from Eigen 5 SIMD
trig (`pcos<Packet4d>`) without going through this layer.

```cpp
struct MyVF : tycho::vf::VectorFunction<MyVF, IR, OR, EnzymeAD, ...> {
    static constexpr bool is_vectorizable = true;   // no other flags needed
    // ...
};
```

**Rule of thumb (post Eigen 5 + Phase 6):**

Bench numbers below: 2026-04-28, BW=4, AVX2; reproducible via
`bench/cpp/bench_enzyme.cpp` + `bench/bench_track.sh`.

- **Arithmetic / `sqrt`-only VFs, IR ≥ 6** (e.g. CR3BP-class): leave
  `is_vectorizable = true`.  Gets full Phase 3 + 5b Jacobian SIMD AND
  Phase 6 Hessian SIMD.  Hessian wins ~12× on CR3BP (291 ns Phase 6 vs
  3672 ns Phase 5a per SuperScalar call) — the per-call Enzyme overhead
  is amortised across 4 lanes via the SIMD reverse pass.
- **Tiny trig-bearing VFs, IR ≤ 5** (e.g. Brach-class): `is_vectorizable
  = true`; route trig through `tycho::math::cos/sin`.  Phase 5b Jacobian
  wins per-lane; **Phase 6 Hessian regresses** vs Phase 5a (~12× slower
  on Brach: 1038 ns vs 87 ns) because the SS reverse-mode tape overhead
  dwarfs the body cost.  Acceptable — PSIOPT vectorizable workloads run
  CR3BP/MEE-class bodies, not toy Brach.
- **Composite trig+sqrt+division VFs** (e.g. MEE-class):
  `is_vectorizable = true` AND inherit from
  `tycho::vf::EnzymeSimdHessianUnsupported`.  Phase 5b handles the
  Jacobian; Phase 6 SIMD Hessian falls back to Phase 5a (the marker
  forces it).  An archived FoF SIMD path exists in
  `dense_enzyme.h` and could in principle handle MEE-class Hessians at
  SIMD; not currently tested or benched, see the FoF archive block in
  the header for revival.
- **Trig-bearing VFs evaluated one trajectory at a time:** either
  approach works; `tycho::math::*` with `is_vectorizable = true` is
  consistent across single- and multi-trajectory call sites.

A Vectorizable VF invoked with plain `double` Scalar still goes through
the scalar Jacobian path (Phase 1 / Phase 3 batched), unchanged — the
double-Scalar `tycho::math::cos/sin` overloads forward to `std::cos` /
`std::sin` directly.

Phase 3 (enzyme_width batched tangents) and Phase 5b (SIMD primal) are
orthogonal axes.  In current bench numbers the two don't compose
materially — per-call Enzyme overhead is small once cached, so reducing
call count via Phase 3 doesn't help when Phase 5b is already SIMD-
saturated.

For the design rationale, see
`docs/superpowers/specs/2026-04-25-claude-enzyme-ad-support-design.md`.

**Upstream canary — when can we drop the scalar-libm wrapper?**

The two upstream gaps that forced the per-lane scalar libm design are
tracked by a pair of compile-time canaries under
`scripts/upstream_canary/`:

  * Test A — `Eigen::Array<>::cos()` differentiable by Enzyme at
    `enzyme_width > 1`. If this passes, drop `simd_math.h`'s manual
    unrolling and call `x.cos() / x.sin()` directly.
  * Test B — Enzyme custom-registered derivative composes with
    `enzyme_width > 1`. If this passes, an alternative architecture
    (custom derivative + SIMD packet trig + Phase 3 batching) becomes
    viable.

Run before bumping the Enzyme submodule, the Eigen submodule, or system
clang/LLVM:

```bash
scripts/upstream_canary/check.sh
```

Exit 0 = status quo (both still fail). Non-zero = upstream changed,
revisit `simd_math.h` and the rule-of-thumb above.

| Last run | Toolchain | Result |
|---|---|---|
| 2026-04-27 | clang 21.1.8 / Enzyme tip (`cecf3492`) / Eigen 5.0.1 | both fail (status quo) |
| 2026-04-30 | clang 21.1.8 / Enzyme plugin (`ClangEnzyme-21.so` mtime 2026-04-27) / Eigen 5.0.1 | both fail (status quo) |

### Key CMake variables

| Variable                      | Purpose                                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `Python_EXECUTABLE`           | Path to Python interpreter to build against                                                  |
| `PYTHON_LOCAL_INSTALL_DIR`    | Site-packages directory to install into; defaults to `python -m site --user-site` if not set |
| `TYCHO_FP_MODE`               | Floating-point mode: `STRICT`, `SAFER_FAST`, or `FAST` (default `SAFER_FAST`)               |
| `BUILD_SPHINX_DOCS`           | `ON` to also build documentation (requires sphinx, breathe, furo, exhale)                    |
| `BUILD_CPP_EXAMPLES`          | `ON` to build C++ example programs under `examples/cpp_examples/` (default OFF)              |
| `BUILD_CPP_TESTS`             | `ON` to build C++ unit tests via Google Test (fetched via FetchContent)                       |
| `BUILD_CPP_BENCHMARKS`        | `ON` to build C++ benchmarks via Google Benchmark (fetched via FetchContent)                  |
| `BUILD_CPP_BENCHMARKS_LEGACY` | `ON` to build legacy hand-rolled benchmark executables                                        |
| `TYCHO_HEAVY_COMPILE_JOBS`    | Max concurrent heavy TU compiles; auto-detected (1 per 10 GB RAM). Override for your system.  |

## Code Style

All source code is formatted by automated tools. **Always run the formatter before
committing.** Do not introduce new external dependencies without discussion.
Prefer modifying existing files over creating new ones unless the feature is
clearly self-contained.

### Naming Conventions

- Types and classes: `PascalCase` (e.g., `DenseFunction`, `ODEPhase`)
- Member functions: `snake_case` (e.g., `set_io_rows()`, `add_equal_con()`)
- Member variables: `snake_case_` with trailing underscore (e.g., `num_defects_`)
- Free functions: `snake_case`

**Python API:** All method and property names exposed via nanobind use `snake_case`,
matching the C++ member function names (e.g., `phase.add_boundary_value()`,
`ocp.return_traj()`). Grandfathered exceptions: `"adjointgradient"`, `"adjointhessian"`,
`"computeall"`.

Do **not** bulk find-and-replace remaining `asset_asrl` or `ASSET` identifiers without
explicit instruction — residual uses may be load-bearing (CMake targets, internal type
names) and require coordinated changes.

### C++ — clang-format (LLVM style)

Config: `.clang-format` at the repo root (LLVM base, 4-space indent, 100-column limit).
The CMake `clang-format` target applies it to `src/`, `extensions/`, and `include/tycho/detail/`.

```bash
cd build && ninja clang-format
```

Do **not** add or modify `.clang-format` files inside `src/` — the root config is
authoritative. Do not reformat `dep/` vendored dependencies.

### Python — ruff (Black-compatible format + isort)

Config: `pyproject.toml` at the repo root.

```bash
conda run -n tycho ruff format .
conda run -n tycho ruff check --select I --fix .
```

Line length: 88 (Black default). `dep/` and `build/` are excluded automatically.

### Commit Conventions

Use descriptive commit messages with these prefixes:

| Prefix | Use for |
| ------ | ------- |
| `feat:` | New features or capabilities |
| `fix:` | Bug fixes |
| `refactor:` | Code restructuring without behavior change |
| `docs:` | Documentation-only changes |
| `chore:` | Build system, CI, dependency, or tooling changes |

## License and Notices

The project is licensed under **Apache 2.0**. The `notices/` directory contains
copyright and license notices for all bundled third-party libraries.

**Never delete or modify files in `notices/`.** If a new dependency is added, its
license notice must be added to `notices/` as well. Preserve copyright headers in
original source files.

Key obligations:
- **Eigen** (MPL-2.0) — any Eigen *source files* directly modified must remain MPL-2.0
- **Intel MKL** (Intel Simplified Software License) — redistribution has specific terms;
  flag any changes touching MKL integration for manual review
- **Nanobind** (BSD), **fmt** (MIT) — all permissive, just preserve notices
- **boost-threads** (Boost Software License 1.0), **rubber_types** (MIT), **kepler propagator** (MIT),
  **lambert** (MIT), **ctpl** (Apache 2.0) — all permissive; see `notices/` for full list

## Things to Flag for Human Review

Changes in the following areas require explicit human review before merging:

- **PSIOPT optimizer internals** — algorithmic changes can silently degrade convergence
- **Intel MKL / Apple Accelerate integration** — redistribution terms and initialization are sensitive
- **Public API changes** — any Python-facing rename or removal breaks downstream user code
- **New third-party dependencies** — must add license notice to `notices/` and get approval
- **PyPI / packaging** — changes to `setup.py`, `pyproject.toml`, or wheel metadata

## Testing

### C++ unit tests (Google Test)

```bash
cmake --preset <preset> -DBUILD_CPP_TESTS=ON   # one-time reconfigure
cd build && ninja -j<N> tycho_tests tycho_tests_light
ctest --output-on-failure
```

### Python examples (integration tests)

The 32 Python example scripts under `examples/python_examples/` serve as the **integration
test suite** and acceptance gate for all changes merged into `main`.

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Options: `--timeout SECONDS`, `--filter SUBSTRING`.

Required packages for all 32 examples to run (none skipped):
```bash
conda run -n tycho pip install numpy scipy matplotlib seaborn spiceypy
conda install -n tycho -c conda-forge basemap
```

### C++ brachistochrone example

C++ examples are not built by default (`BUILD_CPP_EXAMPLES=OFF`). To build and run:

```bash
cmake --preset <preset> -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j<N> brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
# Builder-API variant (equivalent convergence check):
# ninja -j<N> brachistochrone_builder
# ./examples/cpp_examples/builder/brachistochrone/brachistochrone_builder
```

Expected: "Optimal Solution Found", objective ≈ 1.8013 s.

### Pre-Merge Verification Sequence

Run all four steps in order before opening or merging any PR into `main`:

1. **C++ unit tests** — `ctest --output-on-failure` — all must pass
2. **Python examples** — `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"` — all 32 must exit 0
3. **C++ brachistochrone** — `cmake --preset <preset> -DBUILD_CPP_EXAMPLES=ON && cd build && ninja -j<N> brachistochrone_cpp && ./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp` — must print "Optimal Solution Found", obj ≈ 1.8013 s
4. **Benchmarks** — `bench/bench_track.sh compare` — justify any regressions in the PR description

### Merge policy

**All C++ unit tests must pass, all 32 Python examples must pass, the C++
brachistochrone example must converge, and benchmarks must show no unexplained
regressions before any pull request can be merged into `main`.** Fix broken examples
or justify benchmark regressions in the same PR.

## Benchmarking

```bash
cmake --preset <preset> -DBUILD_CPP_BENCHMARKS=ON   # one-time reconfigure
cd build && ninja -j6 bench_all                      # heavy_compile pool auto-limits concurrent heavy TUs
./bench/cpp/bench_all

bench/bench_track.sh baseline   # record baseline
bench/bench_track.sh record     # record after changes
bench/bench_track.sh compare    # compare HEAD vs baseline
```

See `bench/MACBENCH.md` (macOS) or `bench/WINBENCH.md` (Windows) for the full procedure.
