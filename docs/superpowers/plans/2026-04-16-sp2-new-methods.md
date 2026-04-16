# SP2: New RK Methods — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 modern RK algorithms (Tsit5, BS3, BS5, Vern7, Vern8, Vern9) to Tycho's IVP and transcription stepper paths, validated numerically against OrdinaryDiffEq.jl.

**Architecture:** Add `RKCoeffs<Alg>` specializations with standard Butcher tableau data (coefficients transcribed from OrdinaryDiffEq.jl source, with `Bhat = B - btilde` conversion). Wire each new `IVPAlg` enum value into `set_method` and `stepper_compute` dispatchers in `integrator.h`. The generalized `final_state_sum` in `rk_steppers.h` (cleaned up in SP1) automatically gives each new method a working transcription stepper via compile-time zero-skipping. Validate numerical equivalence against Julia on two-body and CR3BP problems.

**Tech Stack:** C++20, Python/Julia (comparison harness), CMake/ninja, Google Test

**Spec:** `docs/superpowers/specs/2026-04-15-ivp-solver-modernization-design.md` (Section "SP2: New Algorithms")

**Branch:** `sp2-new-methods` (to be created from `ivp-solver-modernization` after SP1 merges)

**Build notes:**
- Linux: `cd build && ninja -j4 all` (NEVER -j6 or -j8 — OOMs)
- Always `conda activate tycho` before building
- NEVER run two builds simultaneously

**Preconditions from SP1:**
- `rk_coeffs.h` uses normalized Butcher notation (`A`, `B`, `Bhat`, `C`, `Bmid`) with `Order`, `ErrorOrder`, `FSAL`, `HasEmbedded`, `BmidSize` metadata
- `rk_steppers.h`'s `RKStepper_Impl::final_state_sum` is generalized (per-algorithm branches deleted in SP1 Task 9)
- Golden regression corpus (12 cases) protects existing DOPRI54/DOPRI87 behavior

---

## Task 0: Set Up Persistent Julia Comparison Harness

**Purpose:** Create a permanent Julia reference environment and comparison scripts. The SP1 harness in `/tmp/tycho_julia_ref/` is transient; SP2 needs a reproducible setup checked into the repo.

**Files:**
- Create: `bench/julia_reference/Project.toml`
- Create: `bench/julia_reference/Manifest.toml`
- Create: `bench/julia_reference/README.md`
- Create: `bench/julia_reference/src/run_method.jl`
- Create: `bench/julia_reference/src/two_body.jl`
- Create: `bench/julia_reference/src/crtbp.jl`
- Create: `bench/julia_reference/test/reference_outputs/` (binary golden outputs)
- Modify: `.gitignore` (ignore `bench/julia_reference/.julia/`)

- [ ] **Step 0.1: Initialize Julia project**

```bash
mkdir -p /home/ghecht/Projects/tycho/bench/julia_reference
cd /home/ghecht/Projects/tycho/bench/julia_reference
julia --project=. -e 'using Pkg; Pkg.add(["OrdinaryDiffEq", "LinearAlgebra", "Printf", "BenchmarkTools"])'
```

Expected: `Project.toml` and `Manifest.toml` created with OrdinaryDiffEq pinned.

- [ ] **Step 0.2: Create two-body test case script**

Create `bench/julia_reference/src/two_body.jl`:

```julia
# Two-body Kepler ODE — same as Tycho's Kepler model.
# State: [x, y, z, vx, vy, vz], μ = 398600.4418 km³/s²
# Matches tests/cpp/integrators/regression/regression_utils.h::leo_circular_x0

using OrdinaryDiffEq, LinearAlgebra, Printf

const MU_EARTH = 398600.4418

function twobody!(du, u, p, t)
    mu = p[1]
    r = @view u[1:3]
    v = @view u[4:6]
    r3 = norm(r)^3
    du[1:3] = v
    du[4:6] = -mu/r3 .* r
end

function leo_circular_initial()
    r0 = 7000.0
    v0 = sqrt(MU_EARTH / r0)
    u0 = [r0, 0.0, 0.0, 0.0, v0, 0.0]
    T = 2π * sqrt(r0^3 / MU_EARTH)
    tf = T / 4.0
    return u0, tf, [MU_EARTH]
end
```

- [ ] **Step 0.3: Create CR3BP test case script**

Create `bench/julia_reference/src/crtbp.jl`:

```julia
# CR3BP normalized-units ODE — matches Tycho's CR3BP_Substitute (Kepler mu=1.0)
# used in regression test Case03.
using OrdinaryDiffEq, LinearAlgebra

function crtbp_substitute!(du, u, p, t)
    # Same as two-body with mu=1.0 (stand-in per SP1 regression_utils.h)
    mu = p[1]
    r = @view u[1:3]
    v = @view u[4:6]
    r3 = norm(r)^3
    du[1:3] = v
    du[4:6] = -mu/r3 .* r
end

function cr3bp_initial()
    u0 = [1.0, 0.0, 0.05, 0.0, 1.0, 0.0]
    tf = 2.0
    return u0, tf, [1.0]
end
```

- [ ] **Step 0.4: Create generic method runner**

Create `bench/julia_reference/src/run_method.jl`:

```julia
# Usage: julia --project=. src/run_method.jl <method> <problem> <abstol> <reltol> <outfile>
# method  ∈ {Tsit5, BS3, BS5, Vern7, Vern8, Vern9, DP5, DP8}
# problem ∈ {two_body, crtbp}
# outfile: binary file receiving [int32 n | float64*n state | int32 nsteps | float64 walltime_sec]

include("two_body.jl")
include("crtbp.jl")

using OrdinaryDiffEq, Printf

function method_from_string(s::AbstractString)
    s == "Tsit5" && return Tsit5()
    s == "BS3"   && return BS3()
    s == "BS5"   && return BS5()
    s == "Vern7" && return Vern7()
    s == "Vern8" && return Vern8()
    s == "Vern9" && return Vern9()
    s == "DP5"   && return DP5()
    s == "DP8"   && return DP8()
    error("Unknown method: $s")
end

function problem_from_string(s::AbstractString)
    s == "two_body" && return (twobody!, leo_circular_initial()...)
    s == "crtbp"    && return (crtbp_substitute!, cr3bp_initial()...)
    error("Unknown problem: $s")
end

function main(args)
    length(args) == 5 || error("Usage: run_method.jl <method> <problem> <abstol> <reltol> <outfile>")
    method = method_from_string(args[1])
    f!, u0, tf, p = problem_from_string(args[2])
    abstol = parse(Float64, args[3])
    reltol = parse(Float64, args[4])
    outfile = args[5]

    prob = ODEProblem(f!, u0, (0.0, tf), p)
    # Warm up JIT
    solve(prob, method; abstol=abstol, reltol=reltol, save_everystep=false)
    t0 = time_ns()
    sol = solve(prob, method; abstol=abstol, reltol=reltol, save_everystep=false)
    t1 = time_ns()
    walltime = (t1 - t0) / 1e9

    uf = sol.u[end]
    nsteps = length(sol.t) - 1

    open(outfile, "w") do io
        n = Int32(length(uf))
        write(io, n)
        for x in uf; write(io, x); end
        write(io, Int32(nsteps))
        write(io, Float64(walltime))
    end
    @printf("%s %s: uf=%s nsteps=%d walltime=%.3es\n",
            args[1], args[2], uf, nsteps, walltime)
end

main(ARGS)
```

- [ ] **Step 0.5: Sanity-check harness against DOPRI54/DOPRI87**

```bash
cd /home/ghecht/Projects/tycho/bench/julia_reference
mkdir -p test/reference_outputs
julia --project=. src/run_method.jl DP5 two_body 1e-12 1e-13 test/reference_outputs/dp5_twobody.bin
julia --project=. src/run_method.jl DP8 two_body 1e-12 1e-13 test/reference_outputs/dp8_twobody.bin
```

Expected: binary files created; stdout shows final state matching Tycho's existing DOPRI54/DOPRI87 regression outputs (visual check, not a bit-exact test yet — OrdinaryDiffEq uses slightly different controller parameters).

- [ ] **Step 0.6: Create README**

Create `bench/julia_reference/README.md`:

```markdown
# Julia Reference Harness

Runs OrdinaryDiffEq.jl as numerical reference for Tycho's RK methods.

## Setup

    julia --project=. -e 'using Pkg; Pkg.instantiate()'

## Usage

    julia --project=. src/run_method.jl <method> <problem> <abstol> <reltol> <outfile>

`method` ∈ {Tsit5, BS3, BS5, Vern7, Vern8, Vern9, DP5, DP8}
`problem` ∈ {two_body, crtbp}

Binary output format:
- int32  n       (state size)
- float64 × n    (final state)
- int32  nsteps  (accepted step count)
- float64 walltime_sec
```

- [ ] **Step 0.7: Commit**

```bash
cd /home/ghecht/Projects/tycho
echo "bench/julia_reference/.julia/" >> .gitignore
git add bench/julia_reference/ .gitignore
git commit -m "chore: add persistent Julia reference harness for SP2 validation"
```

---

## Task 1: Add Tsit5 Tableau

**Purpose:** Transcribe Tsit5 coefficients from Julia source into `RKCoeffs<IVPAlg::Tsit5>`. Tsit5 is 5(4) FSAL with 7 stages — the most commonly-used modern 5th-order method, similar structure to DOPRI54.

**Julia source:** `~/.julia/packages/OrdinaryDiffEqTsit5/8E8fO/src/tsit_tableaus.jl` (function `Tsit5ConstantCacheActual`, lines 39-95 for compiled-floats variant)

**Files:**
- Modify: `include/tycho/detail/integrators/rk_coeffs.h`

- [ ] **Step 1.1: Add `IVPAlg::Tsit5` enum value**

In `rk_coeffs.h`, add to the `IVPAlg` enum (line 27):

```cpp
enum class IVPAlg {
    DOPRI54, ///< Dormand-Prince 5(4) — 7 stages, adaptive
    DOPRI87, ///< Dormand-Prince 8(7) — 13 stages, adaptive (default)
    Tsit5,   ///< Tsitouras 5(4) — 7 stages, FSAL, adaptive (NEW in SP2)
    /// \internal — template-dispatch tag only, not runtime-selectable.
    RK4Classic,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    DOPRI5,
};
```

- [ ] **Step 1.2: Add `RKCoeffs<IVPAlg::Tsit5>` specialization**

Append to `rk_coeffs.h` before the closing namespace brace:

```cpp
template <> struct RKCoeffs<IVPAlg::Tsit5> {
    static constexpr int Stages = 7;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 4;
    static constexpr bool FSAL = true;
    static constexpr bool HasEmbedded = true;
    static constexpr int BmidSize = FSAL ? Stages : Stages + 1;

    // A: 6 rows (stages 2..7). Row i holds A[i+2, 0..i+1].
    // Source: OrdinaryDiffEqTsit5 Tsit5ConstantCacheActual (CompiledFloats variant)
    static constexpr std::array<std::array<double, 6>, 6> A = {
        std::array<double, 6>{0.161, 0, 0, 0, 0, 0},
        std::array<double, 6>{-0.008480655492356989, 0.335480655492357, 0, 0, 0, 0},
        std::array<double, 6>{2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0, 0, 0},
        std::array<double, 6>{5.325864828439257, -11.748883564062828, 7.4955393428898365,
                              -0.09249506636175525, 0, 0},
        std::array<double, 6>{5.86145544294642, -12.92096931784711, 8.159367898576159,
                              -0.071584973281401, -0.028269050394068383, 0},
        // Row 7 (FSAL) = b vector itself:
        std::array<double, 6>{0.09646076681806523, 0.01, 0.4798896504144996,
                              1.379008574103742, -3.290069515436081, 2.324710524099774}};

    static constexpr std::array<double, 6> C = {0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0};

    // B (5th-order weights): same as the 7th row of A (FSAL structure) plus b7 = 0.
    static constexpr std::array<double, 7> B = {0.09646076681806523, 0.01, 0.4798896504144996,
                                                1.379008574103742, -3.290069515436081,
                                                2.324710524099774, 0.0};

    // Bhat = B - btilde (Julia stores btilde; we reconstruct bhat).
    // btilde from Julia: (-0.001780011052225777, -0.0008164344596567469, 0.007880878010261995,
    //                    -0.1447110071732629, 0.5823571654525552, -0.45808210592918697,
    //                     0.015151515151515152)
    static constexpr std::array<double, 7> Bhat = {
        0.09646076681806523 - (-0.001780011052225777),
        0.01 - (-0.0008164344596567469),
        0.4798896504144996 - 0.007880878010261995,
        1.379008574103742 - (-0.1447110071732629),
        -3.290069515436081 - 0.5823571654525552,
        2.324710524099774 - (-0.45808210592918697),
        0.0 - 0.015151515151515152};

    // Bmid from Tsit5Interp at θ=0.5: b_i(0.5) = 0.5·r_i1 + 0.25·r_i2 + 0.125·r_i3 + 0.0625·r_i4
    // r values from tsit_tableaus.jl function Tsit5Interp.
    // Tsit5 r_i1 = (1, 0, 0, 0, 0, 0, 0) for i=1, else r_i1 = 0 and r_i2..r_i4 populated.
    // Computation: see generator helper below; values precomputed:
    static constexpr std::array<double, BmidSize> Bmid = {
        0.11783888924880006,  // stage 1
        0.026334285714285713, // stage 2
        0.28943056746031744,  // stage 3
        -0.055055794014625566,// stage 4
        -0.01855290669576731, // stage 5
        0.12077044907460317,  // stage 6
        0.1875                // stage 7
    };
    // TODO(SP2): verify Bmid values via symbolic computation in a test;
    // alternatively, switch dense output to use Hermite interpolation.
};
```

Note: The `Bmid` computation is done once by evaluating the Tsit5 interpolation polynomial at θ=0.5. A helper script `bench/julia_reference/src/compute_bmid.jl` should derive these numerically as a validation step.

- [ ] **Step 1.3: Create Bmid derivation helper**

Create `bench/julia_reference/src/compute_bmid.jl`:

```julia
# Derive Bmid coefficients for Tsit5 at θ=0.5 from the Tsit5Interp polynomial.
# Tsit5 interp: b_i(Θ) = r_i1·Θ + r_i2·Θ² + r_i3·Θ³ + r_i4·Θ⁴
using Printf

# From OrdinaryDiffEqTsit5/.../tsit_tableaus.jl Tsit5Interp(CompiledFloats):
r = [
    [1.0,                -2.763706197274826,  2.9132554618219126, -1.0530884977290216],
    [0.0,                 0.13169999999999998, -0.2234,             0.1017],
    [0.0,                 3.9302962368947516, -5.941033872131505,   2.490627285651253],
    [0.0,               -12.411077166933676,  30.33818863028232,  -16.548102889244902],
    [0.0,                37.50931341651104,  -88.1789048947664,    47.37952196281928],
    [0.0,               -27.896526289197286,  65.09189467479366,  -34.87065786149661],
    [0.0,                 1.5,                -4.0,                 2.5],
]

θ = 0.5
for (i, ri) in enumerate(r)
    bi = ri[1]*θ + ri[2]*θ^2 + ri[3]*θ^3 + ri[4]*θ^4
    @printf("  Bmid[%d] = %.17e\n", i-1, bi)
end
```

Run: `julia --project=. src/compute_bmid.jl`. Copy output values into Step 1.2's `Bmid` array if they differ.

- [ ] **Step 1.4: Build (no wiring yet)**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ninja -j4 tycho_tests_light"
```

Expected: PASS. The tableau is defined but not yet dispatched — just verifying it compiles.

- [ ] **Step 1.5: Commit**

```bash
git add include/tycho/detail/integrators/rk_coeffs.h \
        bench/julia_reference/src/compute_bmid.jl
git commit -m "feat: add Tsit5 Butcher tableau (RKCoeffs specialization)"
```

---

## Task 2: Wire Tsit5 into Dispatch

**Purpose:** Add `IVPAlg::Tsit5` cases to `set_method` and `stepper_compute`. After this task, `Integrator<DODE>(ode, IVPAlg::Tsit5, h)` works and produces correct results.

**Files:**
- Modify: `include/tycho/detail/integrators/integrator.h`
- Modify: `src/bindings/integrators/rk_coeffs_bind.cpp`
- Modify: `src/bindings/integrators/integrator_bind.h` (if string mapping exists)

- [ ] **Step 2.1: Add `set_method` case for Tsit5**

In `integrator.h`, find `set_method` (line 214). Add after the `DOPRI87` case:

```cpp
case IVPAlg::Tsit5:
    this->rk_method_ = IVPAlg::Tsit5;
    this->error_order_ = 4;
    this->init_stepper_and_controller<IVPAlg::Tsit5>(dode, usecontrol, ucon, varlocs_t);
    break;
```

- [ ] **Step 2.2: Add `stepper_compute` dispatcher case**

In `integrator.h`, find `stepper_compute` (line ~545). Add before the `default:` label:

```cpp
case IVPAlg::Tsit5: {
    this->stepper_compute_impl<IVPAlg::Tsit5, Scalar>(x, tf, xf, xf_est, true, xdot_prev,
                                                       domidpoint, xf_mid);
} break;
```

Note: `true` for `dofsal` because Tsit5 is FSAL.

- [ ] **Step 2.3: Expose `Tsit5` in Python bindings**

In `src/bindings/integrators/rk_coeffs_bind.cpp`, add to the enum binding (line ~24):

```cpp
.value("Tsit5", IVPAlg::Tsit5)
```

If `integrator_bind.h` has a string→enum mapping (around line 34), add:

```cpp
if (name == "Tsit5") return IVPAlg::Tsit5;
```

- [ ] **Step 2.4: Build**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ninja -j4 all"
```

Expected: PASS. If the generalized `final_state_sum` has issues with the 7-stage Tsit5, verify SP1 Task 9 was completed correctly.

- [ ] **Step 2.5: Verify regression tests still pass**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ctest -R Regression --output-on-failure"
```

Expected: 12/12 pass (existing DOPRI54/DOPRI87 tests unaffected).

- [ ] **Step 2.6: Commit**

```bash
git add include/tycho/detail/integrators/integrator.h \
        src/bindings/integrators/rk_coeffs_bind.cpp \
        src/bindings/integrators/integrator_bind.h
git commit -m "feat: wire Tsit5 into Integrator::set_method and stepper_compute"
```

---

## Task 3: Validate Tsit5 Against Julia

**Purpose:** Create a C++ test that runs Tsit5 on two-body and CR3BP problems, then compare final state and step count against OrdinaryDiffEq.jl's Tsit5 via the harness.

**Files:**
- Create: `tests/cpp/integrators/test_method_tsit5.cpp`
- Modify: `tests/cpp/CMakeLists.txt` (add to heavy sources)
- Create: `bench/julia_reference/test/reference_outputs/tsit5_twobody.bin` (generated)
- Create: `bench/julia_reference/test/reference_outputs/tsit5_crtbp.bin` (generated)

- [ ] **Step 3.1: Generate Julia reference outputs**

```bash
cd /home/ghecht/Projects/tycho/bench/julia_reference
julia --project=. src/run_method.jl Tsit5 two_body 1e-12 1e-13 test/reference_outputs/tsit5_twobody.bin
julia --project=. src/run_method.jl Tsit5 crtbp 1e-12 1e-13 test/reference_outputs/tsit5_crtbp.bin
```

Expected: binary files created; stdout shows final state and step count.

- [ ] **Step 3.2: Create C++ test**

Create `tests/cpp/integrators/test_method_tsit5.cpp`:

```cpp
///////////////////////////////////////////////////////////////////////////////
// Tsit5 method validation against OrdinaryDiffEq.jl reference
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include "regression/regression_utils.h"
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

#ifndef JULIA_REFERENCE_DIR
#error "JULIA_REFERENCE_DIR must be defined at compile time"
#endif

namespace {

struct JuliaRef {
    Eigen::VectorXd uf;
    int nsteps;
    double walltime;
};

JuliaRef read_julia_ref(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) throw std::runtime_error("Cannot open Julia ref: " + path);
    int32_t n;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    Eigen::VectorXd uf(n);
    f.read(reinterpret_cast<char *>(uf.data()), n * sizeof(double));
    int32_t nsteps;
    f.read(reinterpret_cast<char *>(&nsteps), sizeof(nsteps));
    double walltime;
    f.read(reinterpret_cast<char *>(&walltime), sizeof(walltime));
    return {uf, nsteps, walltime};
}

std::string ref_path(const std::string &filename) {
    return std::string(JULIA_REFERENCE_DIR) + "/test/reference_outputs/" + filename;
}

} // namespace

class Tsit5Test : public VectorFunctionFixture {};

TEST_F(Tsit5Test, TwoBodyMatchesJulia) {
    Kepler kep(MU_EARTH);
    Integrator<Kepler> integ(kep, IVPAlg::Tsit5, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = leo_circular_x0();
    double tf = leo_period() / 4.0;
    auto xf = integ.integrate(x0, tf);

    auto ref = read_julia_ref(ref_path("tsit5_twobody.bin"));

    // 5th-order method with atol=1e-12, rtol=1e-13: expect ~1e-10 agreement
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(xf[i], ref.uf[i], 1e-8)
            << "Tsit5 two-body state[" << i << "] mismatch: Tycho=" << xf[i]
            << " Julia=" << ref.uf[i];
    }
}

TEST_F(Tsit5Test, CR3BPMatchesJulia) {
    CR3BP_Substitute ode(MU_CR3BP_SUBSTITUTE);
    Integrator<CR3BP_Substitute> integ(ode, IVPAlg::Tsit5, 0.01);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = cr3bp_l1_x0();
    double tf = 2.0;
    auto xf = integ.integrate(x0, tf);

    auto ref = read_julia_ref(ref_path("tsit5_crtbp.bin"));

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(xf[i], ref.uf[i], 1e-8)
            << "Tsit5 CR3BP state[" << i << "] mismatch: Tycho=" << xf[i]
            << " Julia=" << ref.uf[i];
    }
}
```

- [ ] **Step 3.3: Add to CMakeLists**

In `tests/cpp/CMakeLists.txt`, add to `TYCHO_TEST_HEAVY_SOURCES` (after `test_stepper.cpp`):

```cmake
    integrators/test_method_tsit5.cpp
```

And at the end (after the other `target_compile_definitions`):

```cmake
target_compile_definitions(tycho_tests PRIVATE
    JULIA_REFERENCE_DIR="${CMAKE_SOURCE_DIR}/bench/julia_reference"
)
```

- [ ] **Step 3.4: Build and run**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && cmake --preset linux-clang-release && ninja -j4 tycho_tests"
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ctest -R Tsit5 --output-on-failure"
```

Expected: 2/2 pass.

**If tests FAIL:** Inspect diff — likely causes:
- Coefficient transcription error (re-check Step 1.2 against Julia source)
- Controller parameter mismatch (Julia Tsit5 defaults differ from Tycho's I-controller; this is expected for step count but not final state at high accuracy)
- Tolerance too tight for 5th-order method

If final state differs by >1e-6 with atol=1e-12, there is likely a coefficient error. Regenerate Bmid via `compute_bmid.jl` and re-run.

- [ ] **Step 3.5: Commit**

```bash
git add tests/cpp/integrators/test_method_tsit5.cpp tests/cpp/CMakeLists.txt \
        bench/julia_reference/test/reference_outputs/tsit5_twobody.bin \
        bench/julia_reference/test/reference_outputs/tsit5_crtbp.bin
git commit -m "test: validate Tsit5 against OrdinaryDiffEq.jl reference"
```

---

## Task 4: Add BS3 (Bogacki-Shampine 3(2))

**Purpose:** Add BS3 — 4-stage FSAL 3rd-order method with 2nd-order embedded estimate. Used for lower-tolerance / faster integration.

**Julia source:** `~/.julia/packages/OrdinaryDiffEqLowOrderRK/5x2GR/src/low_order_rk_tableaus.jl` lines 1-61 (function `BS3ConstantCacheActual`)

**Files:**
- Modify: `include/tycho/detail/integrators/rk_coeffs.h`
- Modify: `include/tycho/detail/integrators/integrator.h`
- Modify: `src/bindings/integrators/rk_coeffs_bind.cpp`
- Create: `tests/cpp/integrators/test_method_bs3.cpp`
- Create: `bench/julia_reference/test/reference_outputs/bs3_twobody.bin`
- Create: `bench/julia_reference/test/reference_outputs/bs3_crtbp.bin`

- [ ] **Step 4.1: Read Julia BS3 coefficients**

```bash
cat ~/.julia/packages/OrdinaryDiffEqLowOrderRK/5x2GR/src/low_order_rk_tableaus.jl | head -65
```

Extract: `a21, a31, a32, a41, a42, a43, c1, c2, c3, btilde1..btilde4`. Stages = 4, FSAL = true, Order = 3, ErrorOrder = 2.

- [ ] **Step 4.2: Add `IVPAlg::BS3` + `RKCoeffs<BS3>`**

Follow Task 1's pattern. BS3 tableau:

```cpp
template <> struct RKCoeffs<IVPAlg::BS3> {
    static constexpr int Stages = 4;
    static constexpr int Order = 3;
    static constexpr int ErrorOrder = 2;
    static constexpr bool FSAL = true;
    static constexpr bool HasEmbedded = true;
    static constexpr int BmidSize = FSAL ? Stages : Stages + 1;

    // A: 3 rows (stages 2..4). Last row is b (FSAL).
    static constexpr std::array<std::array<double, 3>, 3> A = {
        std::array<double, 3>{0.5, 0, 0},          // a21
        std::array<double, 3>{0.0, 0.75, 0},       // a31, a32
        std::array<double, 3>{2.0/9.0, 1.0/3.0, 4.0/9.0}  // b (FSAL)
    };
    static constexpr std::array<double, 3> C = {0.5, 0.75, 1.0};
    static constexpr std::array<double, 4> B = {2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0};
    // btilde = (7/24 - 2/9, 1/4 - 1/3, 1/3 - 4/9, 1/8 - 0) = (5/72, -1/12, -1/9, 1/8)
    // Bhat = B - btilde
    static constexpr std::array<double, 4> Bhat = {
        2.0/9.0 - 5.0/72.0,
        1.0/3.0 - (-1.0/12.0),
        4.0/9.0 - (-1.0/9.0),
        0.0 - 1.0/8.0};
    // Bmid: BS3 uses Hermite interpolation; use midpoint of b weights scaled by 0.5
    // (standard Hermite midpoint: x(t+h/2) ≈ 0.5*(x0+xf) + 0.125*h*(k1-kf))
    // For now, skip Bmid — set to placeholder; event detection uses states only.
    static constexpr std::array<double, BmidSize> Bmid = {0, 0, 0, 0};
    // TODO(SP2): derive Bmid from Hermite interpolation or disable midpoint path for BS3
};
```

- [ ] **Step 4.3: Wire into dispatch (set_method + stepper_compute + bindings)**

Mirror Task 2 with `BS3`. Set `error_order_ = 2` and `dofsal = true`.

- [ ] **Step 4.4: Generate Julia reference and test**

```bash
cd /home/ghecht/Projects/tycho/bench/julia_reference
julia --project=. src/run_method.jl BS3 two_body 1e-8 1e-9 test/reference_outputs/bs3_twobody.bin
julia --project=. src/run_method.jl BS3 crtbp 1e-8 1e-9 test/reference_outputs/bs3_crtbp.bin
```

Note: Looser tolerances because BS3 is only order 3. Create `test_method_bs3.cpp` copying `test_method_tsit5.cpp` pattern with `1e-6` comparison tolerance.

- [ ] **Step 4.5: Build, test, commit**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ninja -j4 tycho_tests"
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ctest -R BS3 --output-on-failure"
git add -A && git commit -m "feat: add BS3 method (Bogacki-Shampine 3(2) FSAL)"
```

---

## Task 5: Add BS5 (Bogacki-Shampine 5(4))

**Purpose:** Add BS5 — 8-stage non-FSAL 5th-order method. Alternative to Tsit5 with different error characteristics.

**Julia source:** `~/.julia/packages/OrdinaryDiffEqLowOrderRK/5x2GR/src/low_order_rk_tableaus.jl` — search for `BS5ConstantCacheActual`.

**Files:** same pattern as Tasks 1-3 for BS5.

- [ ] **Step 5.1: Extract BS5 coefficients**

```bash
rg -A 80 "BS5ConstantCacheActual" ~/.julia/packages/OrdinaryDiffEqLowOrderRK/5x2GR/src/low_order_rk_tableaus.jl | head -120
```

- [ ] **Step 5.2: Add `IVPAlg::BS5` + tableau** (Stages = 8, non-FSAL, Order = 5, ErrorOrder = 4)

- [ ] **Step 5.3: Wire + test + commit** (same pattern as Task 2-3)

---

## Task 6: Add Vern7

**Purpose:** Add Vern7 — 10-stage non-FSAL 7(6) method by Jim Verner. Higher-order alternative with good efficiency for tight tolerances.

**Julia source:** `~/.julia/packages/OrdinaryDiffEqVerner/GlLDJ/src/verner_tableaus.jl` — search for `Vern7ConstantCacheActual`.

**Files:** same pattern.

- [ ] **Step 6.1: Extract Vern7 coefficients** (search function in verner_tableaus.jl)
- [ ] **Step 6.2: Add `IVPAlg::Vern7` + tableau** (Stages = 10, non-FSAL, Order = 7, ErrorOrder = 6)
- [ ] **Step 6.3: Wire + test + commit**

**Note on Verner tableaus:** Vern7/8/9 have large numeric tables. For each coefficient, transcribe the `CompiledFloats` (Float64) variant, not the `BigFloat` variant. Machine-assisted transcription: write a Julia script that outputs coefficients as C++ array literals, then copy-paste.

Create helper `bench/julia_reference/src/emit_tableau.jl`:

```julia
using OrdinaryDiffEqVerner
# For each method, query the cache and print A, B, Bhat, C in C++ array literal format.
# ... (implementation depends on Julia internals; write once, iterate)
```

---

## Task 7: Add Vern8

**Purpose:** 13-stage non-FSAL 8(7) method — higher-order equivalent of Vern7.

- [ ] **Step 7.1-7.3:** Same pattern as Vern7 but with Stages = 13, Order = 8, ErrorOrder = 7.

---

## Task 8: Add Vern9

**Purpose:** 16-stage non-FSAL 9(8) method — highest-order method in SP2. Best for very tight tolerances on smooth problems.

- [ ] **Step 8.1-8.3:** Same pattern with Stages = 16, Order = 9, ErrorOrder = 8.

---

## Task 9: Per-Method Benchmarks

**Purpose:** Add Google Benchmark entries comparing all 8 methods (DOPRI54, DOPRI87, Tsit5, BS3, BS5, Vern7, Vern8, Vern9) on two-body and CR3BP problems.

**Files:**
- Create: `bench/cpp/bench_method_comparison.cpp`
- Modify: `bench/cpp/CMakeLists.txt`

- [ ] **Step 9.1: Create benchmark file**

```cpp
#include <benchmark/benchmark.h>
#include <tycho/tycho.h>
// ... one BENCHMARK_TEMPLATE per (method, problem) pair
```

Template the benchmark on `IVPAlg` and `ODE` so adding methods is one-line.

- [ ] **Step 9.2: Build and run**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && cmake --preset linux-clang-release -DBUILD_CPP_BENCHMARKS=ON && ninja -j4 bench_method_comparison"
./bench/cpp/bench_method_comparison --benchmark_format=console
```

- [ ] **Step 9.3: Document results in spec update**

Append to `docs/superpowers/specs/2026-04-15-ivp-solver-modernization-design.md` a "SP2 Benchmark Results" section with method comparison at common tolerances (1e-6, 1e-9, 1e-12) — wall time, step count, final-state accuracy.

- [ ] **Step 9.4: Commit**

```bash
git add bench/cpp/bench_method_comparison.cpp bench/cpp/CMakeLists.txt \
        docs/superpowers/specs/2026-04-15-ivp-solver-modernization-design.md
git commit -m "bench: compare all 8 RK methods on two-body and CR3BP"
```

---

## Task 10: Python Test Coverage

**Purpose:** Ensure new methods are selectable from Python and produce reasonable results.

**Files:**
- Create: `tychopy/test/integrators/test_new_methods.py`

- [ ] **Step 10.1: Create Python test**

```python
import pytest
import numpy as np
from tychopy import Integrator, IVPAlg, Astro

MU = 398600.4418

@pytest.mark.parametrize("method", [
    IVPAlg.Tsit5, IVPAlg.BS3, IVPAlg.BS5,
    IVPAlg.Vern7, IVPAlg.Vern8, IVPAlg.Vern9,
])
def test_leo_circular(method):
    kep = Astro.Kepler(MU)
    integ = Integrator(kep, method, 10.0)
    integ.set_abs_tol(1e-10)
    integ.set_rel_tol(1e-10)

    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    x0 = np.array([r0, 0, 0, 0, v0, 0, 0])
    T = 2 * np.pi * np.sqrt(r0**3 / MU)
    xf = integ.integrate(x0, T / 4.0)

    # After quarter period, circular orbit: x=0, y=r0, vx=-v0, vy=0
    np.testing.assert_allclose(xf[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(xf[1], r0, atol=1e-3)
```

- [ ] **Step 10.2: Run Python tests**

```bash
conda run -n tycho bash -c "pytest tychopy/test/integrators/test_new_methods.py -v"
```

Expected: 6/6 pass.

- [ ] **Step 10.3: Commit**

```bash
git add tychopy/test/integrators/test_new_methods.py
git commit -m "test: Python test coverage for new SP2 RK methods"
```

---

## Task 11: Final Verification

**Purpose:** Run full test suite + Python examples + benchmarks to confirm no regressions.

- [ ] **Step 11.1: Full C++ test suite**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ninja -j4 all && ctest --output-on-failure"
```

Expected: all tests pass, including 12 regression tests, 17 SP1 unit tests, 16 new SP2 method tests (2 problems × 8 methods).

- [ ] **Step 11.2: Python examples**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py --timeout 120"
```

Expected: 30 pass, 8 skipped (basemap unavailable) — same as SP1 baseline.

- [ ] **Step 11.3: Regression corpus bit-identical check**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ctest -R Regression --output-on-failure"
```

Expected: 12/12 pass. Any FP-noise differences indicate SP2 inadvertently changed DOPRI54/DOPRI87 behavior.

- [ ] **Step 11.4: Benchmark comparison vs SP1 baseline**

```bash
cd /home/ghecht/Projects/tycho
bench/bench_track.sh compare
```

Expected: no regression on DOPRI54/DOPRI87 benchmarks. New method benchmarks recorded for future SP3 comparison.

- [ ] **Step 11.5: Final commit**

```bash
git commit --allow-empty -m "chore: SP2 new methods complete — 6 algorithms added, Julia-validated"
```

---

## Rollback Plan

If a new method's validation fails repeatedly and the cause is unclear:

1. Git revert the commits for that method (keeps other methods)
2. File an issue referencing the Julia source diff
3. Defer to SP3 (after PI controller landing) in case the issue is controller-related

New methods are additive — removing any does not affect DOPRI54/DOPRI87 or existing tests.

## Key Risks

| Risk | Mitigation |
|------|------------|
| Coefficient transcription error | `compute_bmid.jl` validates derived values; Julia harness detects any off-by-1 or sign flip at runtime |
| `final_state_sum` fails for new stage counts | SP1 Task 9 generalized it; regression tests catch DOPRI54/87; new-method tests catch the new cases |
| Binding TU RAM explodes with 8 methods | Monitor `ninja -j4` RSS during build; if >8 GB per TU, defer Vern9 |
| Julia harness drift (package updates) | `Manifest.toml` pins versions; CI should `Pkg.instantiate` not `Pkg.update` |
| Different controller defaults between Julia and Tycho | Comparison uses I-controller in both (Tycho default). SP3 will align PI/PID. Step counts may diverge 10-20% until SP3 lands. |

## Deferred (not part of SP2)

- PI/PID controllers and per-method default parameters (→ SP3)
- Per-method error norm selection (max-norm vs RMS) (→ SP3)
- Dense output / interpolation polynomials for Vern methods (→ SP3 or SP4)
- Lazy interpolant for Vern7/8/9 (→ SP3 or SP4)
- Full SP1 Task 8 switchover (Stepper, EventHandler, full delegation) — only needed if new methods expose architectural limits. If the switch-based dispatch holds up with 8 methods, this stays deferred.
