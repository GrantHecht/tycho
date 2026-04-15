# Cartesian & CRTBP Codegen Dynamics Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CartesianDynamics` and `CRTBPDynamics` via the Tycho SymPy codegen pipeline, delete `cr3bp_model.h`, convert four C++ builder examples and associated tests/benchmarks, and unbuild `extensions/` as a forced side-effect.

**Architecture:** Two new scripts under `utils/` emit generated headers into `include/tycho/detail/astro/`, mirroring the existing `codegen_mee_dynamics.py` / `mee_dynamics.h` pattern. The new classes take `[state, ax, ay, az]` inputs (9-in, 6-out) like `MEEDynamics`, enabling uniform composition for both controlled and uncontrolled examples. Old hand-written `CR3BP` (a `StaticODE_Expression`) is retained through a correctness-parity checkpoint, then deleted. Extensions subdirectory is unbuilt (not deleted) to unblock header removal.

**Tech Stack:** C++17, Eigen, SymPy, nanobind, CMake, Ninja, Google Test, Google Benchmark.

**Spec:** `doc/plans/2026-04-14-cartesian-crtbp-dynamics-design.md`

**Branch:** `feat/cartesian-crtbp-dynamics` (already created with spec commits)

---

## File Structure

**New files:**
- `utils/codegen_cartesian_dynamics.py` — SymPy script, generates `cartesian_dynamics.h`
- `utils/codegen_crtbp_dynamics.py` — SymPy script, generates `crtbp_dynamics.h`
- `include/tycho/detail/astro/cartesian_dynamics.h` — **generated**, do not edit
- `include/tycho/detail/astro/crtbp_dynamics.h` — **generated**, do not edit

**Deleted files:**
- `include/tycho/detail/astro/cr3bp_model.h`

**Modified files:**
- `include/tycho/astro.h` — swap include list
- `src/bindings/astro/tycho_astro.cpp` — drop orphan include, add two lambda bindings
- `tests/cpp/astro/test_cr3bp.cpp` — port to `CRTBPDynamics` (consider renaming `test_crtbp_dynamics.cpp`)
- `tests/cpp/vector_functions/test_vf_ode_expressions.cpp` — port `CR3BPODEAdjointConsistency`
- `bench/cpp/vector_functions/bench_vector_functions.cpp` — port `BM_VF_CR3BP_*`, drop `cr3bp_model.h` include
- `examples/cpp_examples/builder/simple_low_thrust/main.cpp` — replace inline Kepler+thrust
- `examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp` — replace `make_lt_ode` body
- `examples/cpp_examples/builder/orbit_continuation/main.cpp` — replace `make_cr3bp_ode` body
- `examples/cpp_examples/builder/heteroclinic/main.cpp` — replace `make_cr3bp_ode` body
- `CMakeLists.txt` (root) — comment out `add_subdirectory(extensions)` at line 688

**Reference files (read-only, do not edit):**
- `utils/codegen_mee_dynamics.py` — template for new codegen scripts
- `utils/CodeGen.py` — `TychoHeaderGen` implementation, for debugging only
- `include/tycho/detail/astro/mee_dynamics.h` — generated output reference
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp:105-107` — `MEEDynamics` usage pattern

---

## Conventions and Gotchas

- **Conda env:** every Python invocation below assumes `conda run -n tycho ...` or an active `tycho` environment.
- **Build parallelism:** Linux/WSL `-j8`, macOS `-j6`. **Never run two concurrent builds.** Wait for completion notifications before starting the next build.
- **Formatters:** run `ninja clang-format` after C++ edits and `conda run -n tycho ruff format utils/codegen_*.py` after Python edits before committing.
- **Generated files:** `cartesian_dynamics.h` and `crtbp_dynamics.h` are generated. Do not hand-edit after generation. The current `mee_dynamics.h` contains a manually-preserved ASSET attribution block — but since neither new file replaces ASSET code directly (`cr3bp_model.h` is deleted, not regenerated), no ASSET attribution is required in the new headers.
- **Physics parity:** the correctness of `CRTBPDynamics` depends on matching `cr3bp_model.h` bit-for-bit up to floating-point rearrangement. Task 4 is the mandatory parity checkpoint. Do not convert downstream examples until Task 4 passes.
- **Snake-case filename:** `TychoHeaderGen` converts the class name to snake_case automatically — `CartesianDynamics` → `cartesian_dynamics.h`, `CRTBPDynamics` → `crtbp_dynamics.h`. Confirm the output filename matches.
- **Commit granularity:** one commit per task unless noted. Use the conventional prefixes from CLAUDE.md (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`).

---

## Task 1: Codegen script and generated header for `CartesianDynamics`

**Files:**
- Create: `utils/codegen_cartesian_dynamics.py`
- Create (via codegen): `include/tycho/detail/astro/cartesian_dynamics.h`

- [ ] **Step 1.1: Copy `codegen_mee_dynamics.py` as the starting template**

```bash
cp utils/codegen_mee_dynamics.py utils/codegen_cartesian_dynamics.py
```

- [ ] **Step 1.2: Replace the `MEEDynamics()` function body**

Edit `utils/codegen_cartesian_dynamics.py`. Rename the top-level function to `CartesianDynamics()` and replace its body with:

```python
def CartesianDynamics():
    """Generate Cartesian two-body dynamics VectorFunction.

    Input:  [x, y, z, vx, vy, vz, ax, ay, az] — 6 state + 3 extra accelerations
    Output: [xdot, ydot, zdot, vxdot, vydot, vzdot]
    Parameter: mu — gravitational parameter

    Dynamics: point-mass two-body gravity plus user-supplied perturbation.
    """
    Xs = sp.symbols("x:9")
    x, y, z, vx, vy, vz, ax, ay, az = Xs
    mu = sp.symbols("mu")

    r2 = x**2 + y**2 + z**2
    r = sp.sqrt(r2)
    r3 = r**3
    inv_r3 = 1 / r3

    xdot = vx
    ydot = vy
    zdot = vz
    vxdot = -mu * x * inv_r3 + ax
    vydot = -mu * y * inv_r3 + ay
    vzdot = -mu * z * inv_r3 + az

    Eq = sp.Matrix([xdot, ydot, zdot, vxdot, vydot, vzdot])

    header = TychoHeaderGen(
        "CartesianDynamics",
        Eq,
        sp.Matrix(Xs),
        [(mu, "Gravitational Parameter")],
        docstr="Cartesian two-body dynamics with extra acceleration",
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )
    header.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_cartesian_dynamics.py",
    )
    print(f"Generated cartesian_dynamics.h in {output_dir}")


if __name__ == "__main__":
    CartesianDynamics()
```

Also update the module docstring at the top of the file.

- [ ] **Step 1.3: Format and run**

```bash
conda run -n tycho ruff format utils/codegen_cartesian_dynamics.py
cd utils && conda run -n tycho python codegen_cartesian_dynamics.py
```

Expected: `Generated cartesian_dynamics.h in .../include/tycho/detail/astro`.

- [ ] **Step 1.4: Inspect the generated header**

```bash
wc -l /home/ghecht/Projects/tycho/include/tycho/detail/astro/cartesian_dynamics.h
head -60 /home/ghecht/Projects/tycho/include/tycho/detail/astro/cartesian_dynamics.h
```

Expected:
- File exists and compiles (verified in next task group).
- Struct signature: `struct CartesianDynamics : VectorFunction<CartesianDynamics, 9, 6, DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic>`.
- `double mu_;` present.
- Likely **zero** `pc0_`-style precomputed members (no `mu`-only subexpressions in Cartesian two-body). If CSE surfaces some, that's fine.
- File size ~100-250 lines.

- [ ] **Step 1.5: Commit the script and generated header together**

```bash
git add utils/codegen_cartesian_dynamics.py \
        include/tycho/detail/astro/cartesian_dynamics.h
git commit -m "feat: add CartesianDynamics codegen and generated header

Cartesian two-body dynamics with extra-acceleration inputs, generated via
the same TychoHeaderGen pipeline used for MEEDynamics. Mirrors the 9-in,
6-out VectorFunction pattern.
"
```

---

## Task 2: Codegen script and generated header for `CRTBPDynamics`

**Files:**
- Create: `utils/codegen_crtbp_dynamics.py`
- Create (via codegen): `include/tycho/detail/astro/crtbp_dynamics.h`

- [ ] **Step 2.1: Copy `codegen_cartesian_dynamics.py` as the starting template**

```bash
cp utils/codegen_cartesian_dynamics.py utils/codegen_crtbp_dynamics.py
```

- [ ] **Step 2.2: Replace the function body**

Edit `utils/codegen_crtbp_dynamics.py`. Rename the function to `CRTBPDynamics()` and replace the body:

```python
def CRTBPDynamics():
    """Generate CR3BP rotating-frame dynamics VectorFunction.

    Input:  [x, y, z, vx, vy, vz, ax, ay, az] — 6 rotating-frame state + 3 extra accelerations
    Output: [xdot, ydot, zdot, vxdot, vydot, vzdot]
    Parameter: mu — CR3BP mass ratio m2 / (m1 + m2)

    Frame: rotating frame with primaries at (-mu, 0, 0) and (1 - mu, 0, 0).
    Dynamics: standard Coriolis + centrifugal + two-body attractions, plus
    user-supplied perturbation.
    """
    Xs = sp.symbols("x:9")
    x, y, z, vx, vy, vz, ax, ay, az = Xs
    mu = sp.symbols("mu")

    one_minus_mu = 1 - mu

    # Distance to primary (at -mu, 0, 0) and secondary (at 1 - mu, 0, 0)
    dx1 = x + mu
    dx2 = x - one_minus_mu
    d1 = sp.sqrt(dx1**2 + y**2 + z**2)
    d2 = sp.sqrt(dx2**2 + y**2 + z**2)
    inv_d1_3 = 1 / d1**3
    inv_d2_3 = 1 / d2**3

    # Gravitational accelerations (rotating frame: gravity only, Coriolis and
    # centrifugal added below)
    g1x = -one_minus_mu * dx1 * inv_d1_3
    g1y = -one_minus_mu * y * inv_d1_3
    g1z = -one_minus_mu * z * inv_d1_3

    g2x = -mu * dx2 * inv_d2_3
    g2y = -mu * y * inv_d2_3
    g2z = -mu * z * inv_d2_3

    xdot = vx
    ydot = vy
    zdot = vz
    vxdot = 2 * vy + x + g1x + g2x + ax
    vydot = -2 * vx + y + g1y + g2y + ay
    vzdot = g1z + g2z + az

    Eq = sp.Matrix([xdot, ydot, zdot, vxdot, vydot, vzdot])

    header = TychoHeaderGen(
        "CRTBPDynamics",
        Eq,
        sp.Matrix(Xs),
        [(mu, "CR3BP Mass Ratio")],
        docstr="CR3BP rotating-frame dynamics with extra acceleration",
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "include", "tycho", "detail", "astro"
    )
    header.make_header(
        output_dir=output_dir,
        script_name="utils/codegen_crtbp_dynamics.py",
    )
    print(f"Generated crtbp_dynamics.h in {output_dir}")


if __name__ == "__main__":
    CRTBPDynamics()
```

- [ ] **Step 2.3: Format and run**

```bash
conda run -n tycho ruff format utils/codegen_crtbp_dynamics.py
cd utils && conda run -n tycho python codegen_crtbp_dynamics.py
```

Expected: `Generated crtbp_dynamics.h in .../include/tycho/detail/astro`.

- [ ] **Step 2.4: Inspect the generated header**

```bash
wc -l /home/ghecht/Projects/tycho/include/tycho/detail/astro/crtbp_dynamics.h
head -60 /home/ghecht/Projects/tycho/include/tycho/detail/astro/crtbp_dynamics.h
```

Expected:
- Struct signature: `struct CRTBPDynamics : VectorFunction<CRTBPDynamics, 9, 6, ...>`.
- `double mu_;` present.
- At least one `pc0_`-style precomputed member (from CSE of `1 - mu`).
- File size ~300-600 lines (CRTBP has more subexpressions than Cartesian).

- [ ] **Step 2.5: Commit**

```bash
git add utils/codegen_crtbp_dynamics.py \
        include/tycho/detail/astro/crtbp_dynamics.h
git commit -m "feat: add CRTBPDynamics codegen and generated header

CR3BP rotating-frame dynamics with extra-acceleration inputs, generated
via TychoHeaderGen. Same 9-in, 6-out VectorFunction pattern as MEEDynamics
and CartesianDynamics.
"
```

---

## Task 3: Register headers in umbrella and add Python bindings

**Files:**
- Modify: `include/tycho/astro.h`
- Modify: `src/bindings/astro/tycho_astro.cpp`

**Important:** Do **not** delete `cr3bp_model.h` or its `#include` in the umbrella yet. The parity test in Task 4 needs both classes simultaneously. Only Task 12 deletes the old file.

- [ ] **Step 3.1: Add new includes to the umbrella**

Edit `include/tycho/astro.h`. Add the following two lines near the other `detail/astro/` includes, leaving `cr3bp_model.h` untouched for now:

```cpp
#include "tycho/detail/astro/cartesian_dynamics.h"
#include "tycho/detail/astro/crtbp_dynamics.h"
```

- [ ] **Step 3.2: Add the two Python bindings**

Edit `src/bindings/astro/tycho_astro.cpp`. Inside `AstroBuild`, after the existing `mod.def("modified_dynamics", ...)` lambda (~line 43-44), add:

```cpp
mod.def("cartesian_dynamics",
        [](double mu) { return GenericFunction<-1, -1>(CartesianDynamics(mu)); });

mod.def("crtbp_dynamics",
        [](double mu) { return GenericFunction<-1, -1>(CRTBPDynamics(mu)); });
```

Leave the `#include "tycho/detail/astro/cr3bp_model.h"` at line 17 in place — Task 12 removes it.

- [ ] **Step 3.3: Build and verify the Python module**

```bash
cd build && ninja -j8 _tychopy   # Linux/WSL; use -j6 on macOS
```

Expected: clean build. The new headers compile into the nanobind module.

Quick Python sanity check:

```bash
conda run -n tycho python -c "
import tychopy._tychopy as t
dyn = t.Astro.cartesian_dynamics(1.0)
print('cartesian_dynamics:', dyn.input_rows(), '->', dyn.output_rows())
crtbp = t.Astro.crtbp_dynamics(0.01215)
print('crtbp_dynamics:', crtbp.input_rows(), '->', crtbp.output_rows())
"
```

Expected output:
```
cartesian_dynamics: 9 -> 6
crtbp_dynamics: 9 -> 6
```

- [ ] **Step 3.4: Commit**

```bash
git add include/tycho/astro.h src/bindings/astro/tycho_astro.cpp
git commit -m "feat: wire CartesianDynamics and CRTBPDynamics into bindings"
```

---

## Task 4: Physics parity checkpoint (temporary scaffolding)

**Files:**
- Create (temporary): `tests/cpp/astro/test_crtbp_parity.cpp`
- Modify: `tests/cpp/astro/CMakeLists.txt`

**Purpose:** Verify that `CRTBPDynamics(mu)` with zero extra acceleration produces bit-compatible (to tolerance) outputs and Jacobians against the hand-written `CR3BP(mu)`. This test is scaffolding — it is deleted in Task 12 along with `cr3bp_model.h`.

A similar parity check is not strictly needed for `CartesianDynamics` because its physics is so simple, but do one value-only spot check in the same file.

- [ ] **Step 4.1: Write the parity test**

Create `tests/cpp/astro/test_crtbp_parity.cpp`:

```cpp
// TEMPORARY: deleted together with cr3bp_model.h in the cr3bp-removal commit.
// Verifies numerical parity between the new codegen CRTBPDynamics and the
// legacy hand-written CR3BP struct.

#include <tycho/tycho.h>
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;

class CRTBPParityTest : public TychoTest::VectorFunctionFixture {};

TEST_F(CRTBPParityTest, ValueParityAgainstLegacyCR3BP) {
    const double mu = 0.012150585; // Earth-Moon
    astro::CR3BP legacy(mu);
    astro::CRTBPDynamics dyn(mu);

    // Random state, zero extra acceleration.
    Eigen::VectorXd x_legacy(7);
    x_legacy << 0.5, 0.1, 0.05, 0.02, 0.5, 0.03, 0.0;

    Eigen::VectorXd x_new(9);
    x_new.head<6>() = x_legacy.head<6>();
    x_new.tail<3>().setZero();

    Eigen::VectorXd out_legacy(6), out_new(6);
    out_legacy.setZero();
    out_new.setZero();
    legacy.compute(x_legacy, out_legacy);
    dyn.compute(x_new, out_new);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(out_legacy[i], out_new[i], 1e-12)
            << "Output mismatch at index " << i;
    }
}

TEST_F(CRTBPParityTest, AdjointConsistencyMatchesLegacy) {
    const double mu = 0.012150585;
    astro::CRTBPDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 0.5, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 300, -1.0, 1.0);
    TychoTest::verify_adjoint_consistency(dyn, x, lm, 1e-10);
}

TEST_F(CRTBPParityTest, CartesianDynamicsValueSpotCheck) {
    const double mu = 1.0;
    astro::CartesianDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd out(6);
    out.setZero();
    dyn.compute(x, out);

    EXPECT_NEAR(out[0], 0.0, 1e-15);
    EXPECT_NEAR(out[1], 1.0, 1e-15);
    EXPECT_NEAR(out[2], 0.0, 1e-15);
    EXPECT_NEAR(out[3], -1.0, 1e-15); // -mu * x / r^3 with x=r=1
    EXPECT_NEAR(out[4], 0.0, 1e-15);
    EXPECT_NEAR(out[5], 0.0, 1e-15);
}
```

- [ ] **Step 4.2: Wire the test file into CMake**

Grep `tests/cpp/astro/CMakeLists.txt` for how `test_cr3bp.cpp` is registered and add `test_crtbp_parity.cpp` alongside it:

```bash
grep -n test_cr3bp tests/cpp/astro/CMakeLists.txt
```

Add `test_crtbp_parity.cpp` to the same `add_executable` / `target_sources` list (or equivalent) that `test_cr3bp.cpp` uses. Format matches the existing entries.

- [ ] **Step 4.3: Build and run just this test**

```bash
cd build && ninja -j8 tycho_tests
./tests/cpp/tycho_tests --gtest_filter='CRTBPParityTest.*'
```

Expected: 3 tests, all PASS. If any fail, **stop and debug** — the physics has been misexpressed in the codegen script and must be fixed before proceeding. Do not convert downstream examples with a broken dynamics class.

- [ ] **Step 4.4: Commit**

```bash
git add tests/cpp/astro/test_crtbp_parity.cpp tests/cpp/astro/CMakeLists.txt
git commit -m "test: add temporary CRTBPDynamics parity test vs legacy CR3BP

Scaffolding test deleted together with cr3bp_model.h once all downstream
consumers are migrated.
"
```

---

## Task 5: Port `tests/cpp/astro/test_cr3bp.cpp` to `CRTBPDynamics`

**Files:**
- Modify: `tests/cpp/astro/test_cr3bp.cpp` (or rename to `test_crtbp_dynamics.cpp`)

**Decision point:** rename the file `test_crtbp_dynamics.cpp` and update the class/test fixture to `CRTBPDynamicsTest` for consistency with the new class name. If the rename proves disruptive to the CMake list, keep the original filename.

- [ ] **Step 5.1: Rename the file**

```bash
git mv tests/cpp/astro/test_cr3bp.cpp tests/cpp/astro/test_crtbp_dynamics.cpp
```

And update the corresponding entry in `tests/cpp/astro/CMakeLists.txt`.

- [ ] **Step 5.2: Edit the test body**

- Change `CR3BP cr3bp(mu);` to `astro::CRTBPDynamics dyn(mu);`.
- Change input vector from size 7 to size 9; append 3 zeros for extra acceleration.
- `EXPECT_EQ(cr3bp.input_rows(), 7)` → `EXPECT_EQ(dyn.input_rows(), 9)`.
- `EXPECT_EQ(cr3bp.output_rows(), 6)` stays.
- The `L1ApproximateLocation` test: state becomes 9-D with trailing zeros; the bisection lambda constructs the 9-D input; `deriv[3]` indexing is unchanged.
- Rename `class CR3BPTest` to `class CRTBPDynamicsTest` and update `TEST_F` fixtures.

Reference (pre-edit file is in git history: `git show HEAD:tests/cpp/astro/test_cr3bp.cpp`).

- [ ] **Step 5.3: Build and run**

```bash
cd build && ninja -j8 tycho_tests
./tests/cpp/tycho_tests --gtest_filter='CRTBPDynamicsTest.*'
```

Expected: both tests PASS. The L1 location should match (same physics, same root-finding tolerance).

- [ ] **Step 5.4: Commit**

```bash
git add tests/cpp/astro/ tests/cpp/astro/CMakeLists.txt
git commit -m "test: port test_cr3bp to CRTBPDynamics"
```

---

## Task 6: Port `CR3BPODEAdjointConsistency` in `test_vf_ode_expressions.cpp`

**Files:**
- Modify: `tests/cpp/vector_functions/test_vf_ode_expressions.cpp`

- [ ] **Step 6.1: Edit the test**

Find the test at line 61:

```cpp
TEST_F(VFCompositionTest, CR3BPODEAdjointConsistency) {
    ...
    CR3BP cr3bp(mu);
    EXPECT_EQ(cr3bp.input_rows(), 7);
    ...
}
```

Update to use `astro::CRTBPDynamics`. Change input shape from 7 to 9 (append 3 zeros). Rename the test to `CRTBPDynamicsAdjointConsistency`.

- [ ] **Step 6.2: Build and run**

```bash
cd build && ninja -j8 tycho_tests
./tests/cpp/tycho_tests --gtest_filter='*CRTBPDynamicsAdjointConsistency*'
```

Expected: PASS.

- [ ] **Step 6.3: Commit**

```bash
git add tests/cpp/vector_functions/test_vf_ode_expressions.cpp
git commit -m "test: port CR3BP ODE adjoint consistency test to CRTBPDynamics"
```

---

## Task 7: Port `BM_VF_CR3BP_*` benchmarks

**Files:**
- Modify: `bench/cpp/vector_functions/bench_vector_functions.cpp`

- [ ] **Step 7.1: Edit the benchmark file**

- Remove the `#include <tycho/detail/astro/cr3bp_model.h>` at line 6.
- In `BM_VF_CR3BP_Compute` (line 134) and `BM_VF_CR3BP_ComputeJacobian` (line 147):
  - Change `CR3BP ode(0.01215);` to `astro::CRTBPDynamics dyn(0.01215);`.
  - Adjust any local input vectors from 7 to 9 (append 3 trailing zeros).
- Rename the two benchmarks: `BM_VF_CR3BP_Compute` → `BM_VF_CRTBPDynamics_Compute`, and similarly for the Jacobian variant. Update both `BENCHMARK(...)` registration lines.

- [ ] **Step 7.2: Build and run**

```bash
cd build && ninja -j8 bench_all   # uses heavy_compile pool
./bench/cpp/bench_all --benchmark_filter='BM_VF_CRTBPDynamics_.*'
```

Expected: both benchmarks run to completion with sensible timings (order of microseconds or less for evaluation).

- [ ] **Step 7.3: Commit**

```bash
git add bench/cpp/vector_functions/bench_vector_functions.cpp
git commit -m "bench: port CR3BP VF benchmarks to CRTBPDynamics"
```

---

## Task 8: Convert `simple_low_thrust` builder example

**Files:**
- Modify: `examples/cpp_examples/builder/simple_low_thrust/main.cpp`

- [ ] **Step 8.1: Replace the inline ODE expression**

Current (main.cpp:31-44):

```cpp
auto args = ODEArguments(6, 3, 0);
auto R = args.head<3>();
auto V = args.segment<3>(3);
auto u = args.tail<3>();

auto grav = (-mu) * R.normalized_power<3>();
auto thrust = acc * u;
auto ode_expr = StackedOutputs{V, grav + thrust};

auto ode = ODE(ode_expr, 6, 3)
               .var_group("R", 0, 3)
               .var_group("V", 3, 3)
               .var_names({{"t", 6}})
               .var_group("u", 7, 3);
```

New:

```cpp
auto args = ODEArguments(6, 3, 0);
auto state = args.head<6>();
auto u = args.tail<3>();
auto thrust_acc = acc * u;

auto dyn = astro::CartesianDynamics(mu);
auto ode_expr = GenericFunction<-1, -1>(dyn.eval(stack(state, thrust_acc)));

auto ode = ODE(ode_expr, 6, 3)
               .var_group("R", 0, 3)
               .var_group("V", 3, 3)
               .var_names({{"t", 6}})
               .var_group("u", 7, 3);
```

- [ ] **Step 8.2: Build and run**

```bash
cd build && ninja -j8 simple_low_thrust
./examples/cpp_examples/builder/simple_low_thrust/simple_low_thrust
```

Expected output: all three phases (Time/Power/Mass) converge, `SimpleLowThrust: PASSED` at the end, final boundary errors < 1e-4.

If final boundary errors degrade noticeably compared to the pre-conversion baseline, investigate before committing — the codegen derivatives should be at least as good as the inline expression.

- [ ] **Step 8.3: Commit**

```bash
git add examples/cpp_examples/builder/simple_low_thrust/main.cpp
git commit -m "refactor: simple_low_thrust uses CartesianDynamics"
```

---

## Task 9: Convert `multi_spacecraft_opt` builder example

**Files:**
- Modify: `examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp`

- [ ] **Step 9.1: Replace `make_lt_ode` body**

Current (main.cpp:11-26):

```cpp
ODE make_lt_ode(double mu_val, double ltacc) {
    auto args = ODEArguments(6, 3, 0);
    auto r = args.head<3>();
    auto v = args.segment<3>(3);
    auto u = args.tail<3>();

    auto grav = (-mu_val) * r.normalized_power<3>();
    auto thrust = ltacc * u;
    auto ode_expr = StackedOutputs{v, grav + thrust};

    return ODE(ode_expr, 6, 3)
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_names({{"t", 6}})
        .var_group("u", 7, 3);
}
```

New:

```cpp
ODE make_lt_ode(double mu_val, double ltacc) {
    auto args = ODEArguments(6, 3, 0);
    auto state = args.head<6>();
    auto u = args.tail<3>();
    auto thrust_acc = ltacc * u;

    auto dyn = astro::CartesianDynamics(mu_val);
    auto ode_expr = GenericFunction<-1, -1>(dyn.eval(stack(state, thrust_acc)));

    return ODE(ode_expr, 6, 3)
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_names({{"t", 6}})
        .var_group("u", 7, 3);
}
```

- [ ] **Step 9.2: Build and run**

```bash
cd build && ninja -j8 multi_spacecraft_opt
./examples/cpp_examples/builder/multi_spacecraft_opt/multi_spacecraft_opt
```

Expected: both continuation passes (`ltacc = 0.015` and `ltacc = 0.005`) complete, `MultiSpacecraftOpt: PASSED` at the end.

- [ ] **Step 9.3: Commit**

```bash
git add examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp
git commit -m "refactor: multi_spacecraft_opt uses CartesianDynamics"
```

---

## Task 10: Convert `orbit_continuation` builder example

**Files:**
- Modify: `examples/cpp_examples/builder/orbit_continuation/main.cpp`

- [ ] **Step 10.1: Replace `make_cr3bp_ode` body**

Current (main.cpp:17-52): ~35 lines of inline expression-template construction of the CR3BP rotating-frame acceleration, ending with an `ODE(...)` call configured with `var_group("R", 0, 3)`, `var_group("V", 3, 3)`, and `var_names({...})`.

New body: uncontrolled ODE with 6 state, 0 controls. We feed `CRTBPDynamics` a 9-D input built by stacking the 6-D state with a constant 3-vector of zeros.

```cpp
ODE make_cr3bp_ode(double mu_val) {
    auto args = ODEArguments(6, 0, 0);
    auto state = args.head<6>();

    Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
    auto zero_accel = Constant<7, 3>(7, zero3);

    auto dyn = astro::CRTBPDynamics(mu_val);
    auto ode_expr = GenericFunction<-1, -1>(dyn.eval(stack(state, zero_accel)));

    return ODE(ode_expr, 6, 0)
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_names({{"x", 0}, {"y", 1}, {"z", 2}, {"vx", 3}, {"vy", 4}, {"vz", 5}, {"t", 6}});
}
```

**Note on the `Constant<7, 3>` template parameters:** the first template argument is the outer input dimension (6 state + 1 time = 7, matching `ODEArguments(6, 0, 0)`). The runtime size argument is also 7. Verify this matches existing usage by grepping nearby examples — `heteroclinic/main.cpp:116-117` uses `Constant<7, 3>(7, p1loc)` for the same input shape.

- [ ] **Step 10.2: Build and run**

```bash
cd build && ninja -j8 orbit_continuation
./examples/cpp_examples/builder/orbit_continuation/orbit_continuation
```

Expected: the example prints Lyapunov and Halo family continuation progress and exits 0.

- [ ] **Step 10.3: Commit**

```bash
git add examples/cpp_examples/builder/orbit_continuation/main.cpp
git commit -m "refactor: orbit_continuation uses CRTBPDynamics"
```

---

## Task 11: Convert `heteroclinic` builder example

**Files:**
- Modify: `examples/cpp_examples/builder/heteroclinic/main.cpp`

- [ ] **Step 11.1: Replace `make_cr3bp_ode` body**

Current (main.cpp:106-136): inline CR3BP rotating-frame construction, same pattern as `orbit_continuation`. Note this version has no `mu_val` argument — `mu` is a file-scope constant. Keep that signature.

New body:

```cpp
ODE make_cr3bp_ode() {
    auto args = ODEArguments(6, 0, 0);
    auto state = args.head<6>();

    Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
    auto zero_accel = Constant<7, 3>(7, zero3);

    auto dyn = astro::CRTBPDynamics(mu);
    auto ode_expr = GenericFunction<-1, -1>(dyn.eval(stack(state, zero_accel)));

    return ODE(ode_expr, 6, 0)
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_names({{"t", 6}});
}
```

Preserve any other `var_group` / `var_names` calls that the original version had by inspecting the pre-edit file carefully.

- [ ] **Step 11.2: Build and run**

```bash
cd build && ninja -j8 heteroclinic
./examples/cpp_examples/builder/heteroclinic/heteroclinic
```

Expected: the example runs to completion. Heteroclinic is a long-running example (~470 lines, does continuation and connection finding); allow several minutes.

- [ ] **Step 11.3: Commit**

```bash
git add examples/cpp_examples/builder/heteroclinic/main.cpp
git commit -m "refactor: heteroclinic uses CRTBPDynamics"
```

---

## Task 12: Delete `cr3bp_model.h` and remove scaffolding

**Files:**
- Delete: `include/tycho/detail/astro/cr3bp_model.h`
- Delete: `tests/cpp/astro/test_crtbp_parity.cpp`
- Modify: `include/tycho/astro.h`
- Modify: `src/bindings/astro/tycho_astro.cpp`
- Modify: `tests/cpp/astro/CMakeLists.txt`

- [ ] **Step 12.1: Confirm nothing else references the old header**

```bash
grep -rn 'cr3bp_model\.h\|\\bCR3BP\\b' \
    --include='*.cpp' --include='*.h' \
    src/ include/ tests/ bench/ examples/ extensions/
```

Expected matches after Tasks 5-11:
- `extensions/Tycho_Extensions.h` (will be unbuilt in Task 13 — do NOT fix here).
- `src/bindings/astro/tycho_astro.cpp` orphan `#include` (delete in this task).
- `include/tycho/astro.h` include line (delete in this task).

If any other hits remain (tests, benches, examples), fix them first — this task assumes the conversion work is complete.

- [ ] **Step 12.2: Delete the header and scaffolding test**

```bash
git rm include/tycho/detail/astro/cr3bp_model.h
git rm tests/cpp/astro/test_crtbp_parity.cpp
```

Also remove the `test_crtbp_parity.cpp` entry from `tests/cpp/astro/CMakeLists.txt`.

- [ ] **Step 12.3: Remove orphan includes**

- `include/tycho/astro.h`: delete the `#include "tycho/detail/astro/cr3bp_model.h"` line.
- `src/bindings/astro/tycho_astro.cpp`: delete the `#include "tycho/detail/astro/cr3bp_model.h"` line near the top.

- [ ] **Step 12.4: Build and run tests**

```bash
cd build && ninja -j8 _tychopy tycho_tests
./tests/cpp/tycho_tests
```

Expected: clean build, all tests pass (including `CRTBPDynamicsTest`). No "CR3BP not declared" errors anywhere.

- [ ] **Step 12.5: Commit**

```bash
git add -A
git commit -m "refactor: delete cr3bp_model.h and parity scaffolding

All consumers migrated to CRTBPDynamics. The temporary parity test is
removed together with the legacy header.
"
```

---

## Task 13: Unbuild `extensions/`

**Files:**
- Modify: `CMakeLists.txt` (root)

- [ ] **Step 13.1: Comment out the `add_subdirectory` line**

Edit root `CMakeLists.txt`, line 688. Replace:

```cmake
add_subdirectory(extensions)
```

with:

```cmake
# add_subdirectory(extensions)
# Unbuilt as a side-effect of cr3bp_model.h removal. The extension mechanism
# itself is under review; see doc/plans/2026-04-14-cartesian-crtbp-dynamics-design.md.
```

Leave all files under `extensions/` untouched on disk.

- [ ] **Step 13.2: Clean rebuild and verify**

```bash
cd build && ninja -j8 all
```

Expected: full build completes without touching `extensions/`. The `Tycho_Extensions` target disappears from the build graph; no other target depends on it.

- [ ] **Step 13.3: Commit**

```bash
git add CMakeLists.txt
git commit -m "chore: unbuild extensions/ subdirectory

Required by the cr3bp_model.h removal (Tycho_Extensions.h includes the
deleted header). Leaves files on disk; the extension mechanism itself is
deferred to a follow-up PR.
"
```

---

## Task 14: Pre-merge verification sequence

Follow CLAUDE.md §Testing verbatim. **Do not open a PR until all four gates pass.**

- [ ] **Step 14.1: Regenerate headers (idempotency check)**

```bash
cd utils && conda run -n tycho python codegen_cartesian_dynamics.py
cd utils && conda run -n tycho python codegen_crtbp_dynamics.py
cd /home/ghecht/Projects/tycho && git status
```

Expected: `git status` is clean (the regenerated files are byte-identical to what's in git). If not, commit the delta and document why in the commit message.

- [ ] **Step 14.2: Full build with tests, benchmarks, examples**

```bash
cmake --preset linux-clang-release \
    -DBUILD_CPP_TESTS=ON \
    -DBUILD_CPP_BENCHMARKS=ON \
    -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j8 all
```

(macOS: `macos-llvm-release` preset, `-j6`.)

Expected: clean build.

- [ ] **Step 14.3: C++ unit tests**

```bash
cd build && ctest --output-on-failure
```

Expected: 100% pass rate.

- [ ] **Step 14.4: Python integration suite (all 38 examples)**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Expected: all 38 examples exit 0. In particular, `OrbitContinuation.py` and `Heteroclinic.py` — which use the Python-side `AstroModels.CR3BP` — must still pass unchanged (pure-Python dependency, unaffected by this refactor).

- [ ] **Step 14.5: C++ brachistochrone example**

```bash
cd build && ninja -j8 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
```

Expected: "Optimal Solution Found", objective ≈ 1.8013 s.

- [ ] **Step 14.6: Benchmarks**

```bash
bench/bench_track.sh baseline     # if not already baselined on main
bench/bench_track.sh record       # on this branch
bench/bench_track.sh compare
```

Expected: the old `BM_VF_CR3BP_*` benchmarks show as removed, the new `BM_VF_CRTBPDynamics_*` benchmarks show as added. No unexplained regressions on the MEE / Kepler / phase-level benchmarks (which should be entirely unaffected). Capture the before/after numbers for the PR description.

- [ ] **Step 14.7: Run all four converted builder examples end-to-end**

```bash
./examples/cpp_examples/builder/simple_low_thrust/simple_low_thrust
./examples/cpp_examples/builder/multi_spacecraft_opt/multi_spacecraft_opt
./examples/cpp_examples/builder/orbit_continuation/orbit_continuation
./examples/cpp_examples/builder/heteroclinic/heteroclinic
```

Expected: all four converge / complete with their usual `PASSED` exit strings.

- [ ] **Step 14.8: Open PR**

Draft a PR with:
- Summary: one paragraph pointing at the design doc.
- Test plan: checklist mirroring Steps 14.2-14.7.
- Benchmark section: before/after numbers for `BM_VF_CRTBPDynamics_*` vs. the old `BM_VF_CR3BP_*`.
- Explicit note about the `extensions/` unbuild: "files on disk, not building, mechanism decision deferred."
- Link to the spec: `doc/plans/2026-04-14-cartesian-crtbp-dynamics-design.md`.

```bash
gh pr create --title "feat: CartesianDynamics and CRTBPDynamics via codegen, delete cr3bp_model.h" \
             --body-file /tmp/pr-body.md
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Physics mismatch between `CRTBPDynamics` and legacy `CR3BP` | Task 4 parity test runs before any downstream conversion. If it fails, fix the codegen script before touching examples. |
| Codegen script desync (generated file hand-edited) | Task 14.1 regenerates and checks for `git status` cleanliness. |
| `Constant<7, 3>` template argument confusion | Task 10 explicitly references `heteroclinic/main.cpp:116-117` as the precedent for the same 7-input shape. |
| `extensions/` silently broken | Task 13 only unbuilds, does not delete. Files remain in tree and can be re-enabled by uncommenting one line. |
| Simultaneous builds OOMing the machine | Every build step runs one at a time. Never launch a second `ninja` invocation before the first completes. |
| Benchmarks showing a regression we can't explain | Capture numbers at Task 14.6 and investigate any MEE / Kepler / phase-level regression before opening the PR. Regressions on the removed/renamed CR3BP benchmarks are expected and should be called out. |

---

## Definition of Done

- [ ] `CartesianDynamics` and `CRTBPDynamics` generated headers are committed.
- [ ] `cr3bp_model.h` is deleted from the repo.
- [ ] All four target C++ builder examples (`simple_low_thrust`, `multi_spacecraft_opt`, `orbit_continuation`, `heteroclinic`) converge using the new dynamics classes.
- [ ] All C++ unit tests pass via `ctest`.
- [ ] All 38 Python integration examples pass.
- [ ] C++ brachistochrone example converges with objective ≈ 1.8013 s.
- [ ] Benchmarks recorded and any deltas explained in PR body.
- [ ] `extensions/` is unbuilt (not deleted); files remain on disk.
- [ ] PR opened, referencing the design spec.
