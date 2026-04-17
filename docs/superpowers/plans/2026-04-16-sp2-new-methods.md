# SP2: New RK Methods — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 modern RK algorithms (Tsit5, BS3, BS5, Vern7, Vern8, Vern9) to Tycho's IVP and transcription stepper paths, validated numerically against OrdinaryDiffEq.jl.

**Architecture:** Dispatch in SP2 uses the existing **enum-switch path** (`stepper_compute` in `integrator.h` switches on `rk_method_`). The extracted `Stepper<Alg,DODE,Scalar>` variant/visit pipeline from SP1 Task 10 is DEFERRED; SP2 adds methods without changing the dispatch mechanism. For each new `IVPAlg` enum value, add an `RKCoeffs<Alg>` specialization with standard Butcher tableau data (coefficients transcribed from OrdinaryDiffEq.jl, with `Bhat = B - btilde` conversion). Wire each enum into `set_method` and `stepper_compute` cases. The generalized `final_state_sum` in `rk_steppers.h` (cleaned up in SP1 Task 9) makes each new method automatically work in the transcription stepper via compile-time zero-skipping on `B[i]==0`. Validate numerical equivalence against Julia on two-body and CR3BP.

**Midpoint / dense output:** The `xf_mid` from each accepted step feeds Tycho's LGL5 interpolation table, which is the backbone of event detection (`find_events` via `interpolate_ref`) and dense resampling (`integrate_dense(n, ...)`). LGL5 needs `(x_n, f(x_n), x_mid, x_{n+1}, f(x_{n+1}))` per segment to hit its 5th-order accuracy. To preserve end-to-end accuracy consistent with the integrator order, SP2 transcribes the **exact** Julia interpolation polynomial for each new method, evaluated at Θ=0.5. For methods whose polynomial requires additional "lazy" f-evaluations beyond the main integration stages (BS5: 3 extra, Vern7: 6, Vern8: 8, Vern9: 10), the stepper computes those extras only when `domidpoint=true` (zero cost when midpoint isn't requested). Tsit5 needs no extras (7-stage interp uses only main stages). BS3 uses Hermite (Julia's default for BS3). Task 3.5 extends the schema and step loop to support extras.

**Bmid storage convention:** `Bmid[i] = 2·b_i(Θ=0.5)` in Julia's formulation `sol(Θ) = x + h·Σ b_i(Θ)·f_i`. The factor of 2 cancels the `/2.0` applied inside `stepper_compute_impl`'s midpoint sum. Sanity check: Σ Bmid = 1.0 (consistent with the existing DOPRI54/DOPRI87 Bmid arrays).

**Coefficient storage convention (rational form):** Tycho transcribes coefficients from Julia's generic (`T::Type`) tableau variant. A `constexpr double rat(long long, long long)` helper in `rk_coeffs.h` mirrors Julia's `N // D` rational notation. Use `rat(N, D)` for rational coefficients (bit-identical to Julia's `convert(Float64, N//D)` via IEEE 754 round-to-nearest). Use Float64 literals for irrational coefficients, copied from Julia's `CompiledFloats` variant (bit-identical to Julia's `convert(Float64, big"...")`). All existing methods (DOPRI54, DOPRI87, RK4Classic, DOPRI5) have been migrated to `rat()` form — new methods follow suit.

**FSAL transcription tag:** Analogous to how DOPRI54 (FSAL, 7 stages) uses an internal `IVPAlg::DOPRI5` tag (non-FSAL, 6 stages) for the transcription stepper, FSAL methods in SP2 (Tsit5, BS3) get companion internal tags (`Tsit5Trans`, `BS3Trans`) with the FSAL stage 7/4 dropped. Non-FSAL methods (BS5, Vern7, Vern8, Vern9) reuse their own enum in both paths.

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

**New SP2 preconditions:**
1. **`HasMidpoint` flag** on existing specializations (DOPRI54 / DOPRI87 / RK4Classic / DOPRI5 / Tsit5 / Tsit5Trans) — done in Task 1. Set `true` for methods that support midpoint (DOPRI54, DOPRI87, Tsit5), `false` for transcription-only tags (RK4Classic, DOPRI5, Tsit5Trans).
2. **Dense-output schema extension** — Task 3.5 adds `InterpStages`, `ExtraA`, `ExtraC`, `BmidStages`, `LastStageIsFxf` fields to `RKCoeffs` and extends `stepper_compute_impl` to compute extra stages when `domidpoint=true`. Existing methods default to `InterpStages=0` with no behavior change.
3. **Tsit5 Bmid correction** — committed separately: original Tsit5 Bmid stored `b_i(0.5)` instead of `2·b_i(0.5)`, half the correct value. Fixed before Task 2.

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

**Important:** Always use `Pkg.instantiate()`, NEVER `Pkg.update()`. Manifest.toml
pins package versions exactly; updating breaks reproducibility across machines
and invalidates regression-tied golden outputs.

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

**Julia source:** `~/.julia/packages/OrdinaryDiffEqTsit5/<hash>/src/tsit_tableaus.jl` — function `Tsit5ConstantCacheActual` (the `CompiledFloats` variant, not the `BigFloat` variant). The `<hash>` depends on the Manifest.toml pin from Task 0.

**Files:**
- Modify: `include/tycho/detail/integrators/rk_coeffs.h`

- [ ] **Step 1.1: Add `IVPAlg::Tsit5` and `IVPAlg::Tsit5Trans` enum values**

In `rk_coeffs.h`, add to the `IVPAlg` enum (line 27):

```cpp
enum class IVPAlg {
    DOPRI54,    ///< Dormand-Prince 5(4) — 7 stages, FSAL, adaptive
    DOPRI87,    ///< Dormand-Prince 8(7) — 13 stages, adaptive (default)
    Tsit5,      ///< Tsitouras 5(4) — 7 stages, FSAL, adaptive (NEW in SP2)
    /// \internal — template-dispatch tag only, not runtime-selectable.
    RK4Classic,
    /// \internal — template-dispatch tag only, not runtime-selectable.
    DOPRI5,     ///< Internal: 6-stage non-FSAL variant for DOPRI54 transcription
    /// \internal — template-dispatch tag only, not runtime-selectable.
    Tsit5Trans, ///< Internal: 6-stage non-FSAL variant for Tsit5 transcription
};
```

**Rationale for Tsit5Trans:** The transcription stepper (`RKStepper` in `rk_steppers.h`) always does `Stages` full ODE evaluations; it does NOT use FSAL. For Tsit5 (7-stage FSAL) the 7th stage's k-value would be wasted (its B[7]=0 after the standard transformation), so we use a 6-stage non-FSAL tableau for the transcription path — matching the existing DOPRI5→DOPRI54 pattern.

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

Note: The `Bmid` computation is done once by evaluating the Tsit5 interpolation polynomial at θ=0.5. A helper script `bench/julia_reference/src/compute_bmid.jl` (Step 1.3 below) derives these numerically as a validation step. **Bmid is load-bearing** — setting it to zeros would silently corrupt midpoint interpolation (used by event detection and dense output). If derivation is uncertain, disable the midpoint code path for the method instead (see Task 4 for how).

After the `RKCoeffs<IVPAlg::Tsit5>` specialization, add the `Tsit5Trans` companion tableau (6 stages, non-FSAL, same A-matrix rows 1-5, `B = row 6 of Tsit5's A`):

```cpp
template <> struct RKCoeffs<IVPAlg::Tsit5Trans> {
    static constexpr int Stages = 6;
    static constexpr int Order = 5;
    static constexpr int ErrorOrder = 0;   // no embedded in transcription
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = false;
    // A: 5 rows (stages 2..6), same as RKCoeffs<IVPAlg::Tsit5> rows 0..4
    static constexpr std::array<std::array<double, 5>, 5> A = {
        std::array<double, 5>{0.161, 0, 0, 0, 0},
        std::array<double, 5>{-0.008480655492356989, 0.335480655492357, 0, 0, 0},
        std::array<double, 5>{2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0, 0},
        std::array<double, 5>{5.325864828439257, -11.748883564062828, 7.4955393428898365,
                              -0.09249506636175525, 0},
        std::array<double, 5>{5.86145544294642, -12.92096931784711, 8.159367898576159,
                              -0.071584973281401, -0.028269050394068383}};
    static constexpr std::array<double, 5> C = {0.161, 0.327, 0.9, 0.9800255409045097, 1.0};
    // B = Tsit5's row 7 of A (the b-vector via FSAL structure):
    static constexpr std::array<double, 6> B = {0.09646076681806523, 0.01, 0.4798896504144996,
                                                1.379008574103742, -3.290069515436081,
                                                2.324710524099774};
    static constexpr std::array<double, 6> Bhat = {0, 0, 0, 0, 0, 0}; // unused
    // No Bmid — transcription doesn't need midpoint
};
```

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
    // Use the non-FSAL Tsit5Trans tag for transcription (same pattern as
    // DOPRI54 using DOPRI5). The IVP path still dispatches on IVPAlg::Tsit5
    // (full 7-stage FSAL) via stepper_compute.
    this->init_stepper_and_controller<IVPAlg::Tsit5Trans>(dode, usecontrol, ucon, varlocs_t);
    break;
```

Also add to the "internal tag" rejection list:

```cpp
case IVPAlg::RK4Classic:
case IVPAlg::DOPRI5:
case IVPAlg::Tsit5Trans:  // NEW
    throw std::invalid_argument(
        "IVPAlg::RK4Classic, IVPAlg::DOPRI5, IVPAlg::Tsit5Trans are internal "
        "template-dispatch tags. Use IVPAlg::DOPRI54, IVPAlg::DOPRI87, or IVPAlg::Tsit5.");
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

Do NOT expose `Tsit5Trans`, `DOPRI5`, or `RK4Classic` — they are internal tags.

Inspect `src/bindings/integrators/integrator_bind.h` lines 30-40 for the string→enum mapping (it exists per SP1 audit). Add:

```cpp
if (name == "Tsit5") return IVPAlg::Tsit5;
```

If the mapping uses `strcmp`/`std::string_view`, adapt accordingly.

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

**Test tolerance rationale (1e-8 for Tsit5 two-body):** Tycho uses I-controller; Julia Tsit5 uses PI-controller. Step counts will differ 10-20% (SP3 will align via PI-controller implementation). However, both methods integrate to `atol=1e-12, rtol=1e-13`, so the *final state accuracy* is bounded by accumulated local error at the tolerance floor, not the step controller. For Tsit5 (5th-order) on a smooth two-body problem, final states should match to ~1e-10 (tolerance floor × problem scale). `1e-8` gives 2 orders of margin for FP-noise and controller-induced step-placement differences.

**If tests FAIL:** Inspect diff — likely causes:
- Coefficient transcription error (re-check Step 1.2 against Julia source)
- `Bhat` reconstruction error (`Bhat = B - btilde`, check signs)
- Bmid values differ from `compute_bmid.jl` output → Tsit5 interp polynomial typo

If final state differs by >1e-6 with atol=1e-12, there is almost certainly a coefficient error. Re-run `compute_bmid.jl` and compare against Step 1.2's hardcoded values.

- [ ] **Step 3.6: Commit**

```bash
git add tests/cpp/integrators/test_method_tsit5.cpp tests/cpp/CMakeLists.txt \
        bench/julia_reference/test/reference_outputs/tsit5_twobody.bin \
        bench/julia_reference/test/reference_outputs/tsit5_crtbp.bin
git commit -m "test: validate Tsit5 against OrdinaryDiffEq.jl reference"
```

---

## Task 3.5: Extend Stepper Infrastructure for Extra Interpolation Stages

**Purpose:** Add schema and step-loop support for interpolation "lazy stages" beyond the main integration stages. This is the foundation BS5, Vern7, Vern8, and Vern9 will rely on. Tsit5 and BS3 need no extras, so existing methods remain untouched.

**Files:**
- Modify: `include/tycho/detail/integrators/rk_coeffs.h` (add new schema fields; update existing specializations)
- Modify: `include/tycho/detail/integrators/integrator.h` (extend `stepper_compute_impl`'s `domidpoint` branch)

- [ ] **Step 3.5.1: Document the extended schema in a comment in `rk_coeffs.h`**

Add at top of file, above `enum class IVPAlg`:

```cpp
// RKCoeffs dense-output schema (SP2):
//
//   Main integration tableau:
//     Stages         — main integration f-evals per step
//     A, B, C        — standard Butcher tableau
//     Bhat           — embedded (error-estimator) weights (zeros if !HasEmbedded)
//     FSAL           — last main stage equals next step's first stage
//     HasMidpoint    — does this method support midpoint dense output?
//
//   Dense-output extras (SP2):
//     InterpStages   — # of *additional* f-evals needed for the method's
//                      interpolation polynomial, beyond the main Stages.
//                      0 means "no extras" (Tsit5, BS3, DOPRI54, DOPRI87).
//     LastStageIsFxf — does k_Stages (the last main stage) equal f(xf)?
//                      True for FSAL methods (by construction). Also true for
//                      BS5/Vern7/Vern8/Vern9 because their perform_step evaluates
//                      the last main stage at (t+h, xf) with a-row equal to B.
//                      False only for DOPRI87 (which needs an explicit f(xf)).
//     BmidStages     — total k-values referenced by the midpoint sum:
//                      Stages + InterpStages + (LastStageIsFxf ? 0 : 1).
//     ExtraA         — InterpStages × (Stages + InterpStages) a-matrix; row e
//                      builds the argument to f for extra stage e:
//                        x_e = x + h · Σ_{j<Stages+e} ExtraA[e][j] · k_j
//                      (k_0..k_{Stages-1} are main; k_Stages..k_{Stages+e-1} are
//                       prior extras.)
//     ExtraC         — c-value for each extra stage (t_e = t0 + ExtraC[e]·h).
//     Bmid           — weights for the midpoint sum, stored as 2·b_i(Θ=0.5)
//                      where b_i(Θ) is Julia's interpolation polynomial.
//                      Factor of 2 cancels the /2 applied inside
//                      stepper_compute_impl. Σ Bmid[i] = 1.0 sanity-check.
//                      Layout:
//                        Bmid[0..Stages-1]                 — main stages
//                        Bmid[Stages]                      — f(xf) term (only if !LastStageIsFxf)
//                        Bmid[Stages + offset .. BmidStages-1]  — extra stages
//                        (offset = LastStageIsFxf ? 0 : 1)
```

- [ ] **Step 3.5.2: Add defaults to existing specializations**

For each existing specialization (DOPRI54, DOPRI87, Tsit5, RK4Classic, DOPRI5, Tsit5Trans), add:

```cpp
static constexpr int InterpStages = 0;
static constexpr bool LastStageIsFxf = FSAL;  // overridden to false only for DOPRI87
static constexpr int BmidStages = Stages + InterpStages + (LastStageIsFxf ? 0 : 1);
```

Explicit override for DOPRI87:
```cpp
static constexpr bool LastStageIsFxf = false;  // k_13 is not f(xf); needs extra eval
```

Retire the old `BmidSize` alias in favor of `BmidStages` (they compute the same value for `InterpStages=0`). Use `replace_all` rename via sed for consistency, then verify no other references.

- [ ] **Step 3.5.3: Extend `stepper_compute_impl` `domidpoint` branch**

Current code in `integrator.h` lines 517–538:
```cpp
if (domidpoint) {
    xtup = x;
    xtup[t_var] = t0 + h/2.0;
    for (int i = 0; i < Stages; i++) {
        xtup.segment(...) += (Bmid[i]/2) * k_vals[i];
    }
    if constexpr (!RKData::FSAL) {
        k_vals.back().setZero();
        ode_.compute(xf, k_vals.back());           // f(xf)
        xtup.segment(...) += (Bmid.back()/2) * k_vals.back() * h;
        xdot_prev = k_vals.back();
    }
    xf_mid = xtup;
}
```

Replace with (full extension):

```cpp
if (domidpoint) {
    // Step A: ensure k-value for f(xf) is available.
    // For FSAL methods, k_vals[Stages-1] already == f(xf)·h.
    // For LastStageIsFxf=true non-FSAL methods (BS5/Vern*), also k_vals[Stages-1] == f(xf)·h.
    // For LastStageIsFxf=false (DOPRI87 only), compute f(xf) and store into k_vals_extra[0].
    //
    // k_vals_extra layout:
    //   k_vals_extra[0]         — f(xf)·h if !LastStageIsFxf, else first interp extra
    //   k_vals_extra[0..InterpStages-1 + offset] — interp extras
    //   where offset = LastStageIsFxf ? 0 : 1
    constexpr int offset = RKData::LastStageIsFxf ? 0 : 1;
    if constexpr (!RKData::LastStageIsFxf) {
        ode_.compute(xf, k_vals_extra[0]);
        k_vals_extra[0] *= h;
    }

    // Step B: compute extra interpolation stages (if any).
    if constexpr (RKData::InterpStages > 0) {
        for (int e = 0; e < RKData::InterpStages; e++) {
            xtup = x;
            xtup[t_var] = t0 + RKData::ExtraC[e] * h;
            // main stages
            for (int j = 0; j < RKData::Stages; j++) {
                if constexpr (RKData::ExtraA[e][j] != 0.0) {   // skip zeros at compile time if desired
                    xtup.segment(...) += RKData::ExtraA[e][j] * k_vals[j];
                }
            }
            // prior extra stages
            for (int j = 0; j < e; j++) {
                xtup.segment(...) += RKData::ExtraA[e][RKData::Stages + j] * k_vals_extra[offset + j];
            }
            ode_.compute(xtup, k_vals_extra[offset + e]);
            k_vals_extra[offset + e] *= h;
        }
    }

    // Step C: build xf_mid via Bmid sum.
    xtup = x;
    xtup[t_var] = t0 + h/2.0;
    for (int i = 0; i < RKData::Stages; i++) {
        xtup.segment(...) += (RKData::Bmid[i]/2.0) * k_vals[i];
    }
    if constexpr (!RKData::LastStageIsFxf) {
        // Bmid[Stages] weights f(xf)·h, which is k_vals_extra[0].
        xtup.segment(...) += (RKData::Bmid[RKData::Stages]/2.0) * k_vals_extra[0];
    }
    if constexpr (RKData::InterpStages > 0) {
        for (int e = 0; e < RKData::InterpStages; e++) {
            xtup.segment(...) += (RKData::Bmid[RKData::Stages + offset + e] / 2.0) *
                                  k_vals_extra[offset + e];
        }
    }

    // Step D: xdot_prev update for LastStageIsFxf=false methods
    //         (FSAL and BS5/Vern* get xdot_prev via existing FSAL path earlier).
    if constexpr (!RKData::LastStageIsFxf) {
        xdot_prev = k_vals_extra[0] * (1.0 / h);   // unscaled f(xf)
    }

    xf_mid = xtup;
}
```

Notes on implementation:
- `k_vals_extra` is a new `ArrayOfTempSpecs<ODEDeriv<Scalar>, InterpStages + (LastStageIsFxf ? 0 : 1)>` allocated by the existing `BumpAllocator::allocate_run` call. Size is a compile-time constant per `RKOp` specialization.
- When both `InterpStages == 0` and `LastStageIsFxf == true` (Tsit5, BS3, DOPRI54), the extras array has size 0 — confirm `ArrayOfTempSpecs<T, 0>` is well-defined, or add a constexpr-if guard that omits the allocation.
- The `if constexpr (RKData::ExtraA[e][j] != 0.0)` comment is optional compile-time zero-skipping — plain runtime loop is also fine and simpler.
- The FSAL path for `xdot_prev` (line 502 in existing code) is unchanged. It runs when `dofsal || domidpoint` and handles FSAL methods. The new Step D only fires for non-FSAL, non-LastStageIsFxf methods (i.e., DOPRI87).

- [ ] **Step 3.5.4: Update the BumpAllocator allocation**

Current:
```cpp
BumpAllocator::allocate_run(
    Impl, ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(output_rows, 1),
    TempSpec<ODEState<Scalar>>(input_rows, 1));
```

Extend to:
```cpp
constexpr int ExtraCount = RKData::InterpStages + (RKData::LastStageIsFxf ? 0 : 1);
if constexpr (ExtraCount == 0) {
    BumpAllocator::allocate_run(
        Impl, ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(output_rows, 1),
        TempSpec<ODEState<Scalar>>(input_rows, 1));
} else {
    BumpAllocator::allocate_run(
        Impl, ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(output_rows, 1),
        TempSpec<ODEState<Scalar>>(input_rows, 1),
        ArrayOfTempSpecs<ODEDeriv<Scalar>, ExtraCount>(output_rows, 1));
}
```

Update `Impl` lambda signature to accept the third temp argument. The lambda's capture of `k_vals_extra` comes through the allocate_run callback protocol.

- [ ] **Step 3.5.5: Build and run existing regression tests**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ninja -j4 tycho_tests tycho_tests_light"
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ctest --output-on-failure"
```

Expected: all 12 existing DOPRI54/DOPRI87 regression cases pass — no behavior change because `InterpStages=0` for existing methods and the new paths are compile-time-skipped.

- [ ] **Step 3.5.6: Commit**

```bash
git add include/tycho/detail/integrators/rk_coeffs.h \
        include/tycho/detail/integrators/integrator.h
git commit -m "feat: extend RK dense-output schema with InterpStages/ExtraA/ExtraC (SP2)"
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
    static constexpr bool HasMidpoint = true;

    // SP2 dense-output schema fields (all defaults; no extra stages needed)
    static constexpr int InterpStages = 0;
    static constexpr bool LastStageIsFxf = true;  // FSAL
    static constexpr int BmidStages = Stages;     // = 4

    // A: 3 rows (stages 2..4). Last row is b (FSAL).
    // Julia source: BS3ConstantCache (generic T::Type variant, rational coefficients).
    static constexpr std::array<std::array<double, 3>, 3> A = {
        std::array<double, 3>{rat(1, 2), 0, 0},           // a21 = 1 // 2
        std::array<double, 3>{0, rat(3, 4), 0},           // a32 = 3 // 4
        std::array<double, 3>{rat(2, 9), rat(1, 3), rat(4, 9)}   // b (FSAL): 2/9, 1/3, 4/9
    };
    static constexpr std::array<double, 3> C = {rat(1, 2), rat(3, 4), 1.0};
    static constexpr std::array<double, 4> B = {rat(2, 9), rat(1, 3), rat(4, 9), 0.0};
    // Bhat = 2nd-order embedded weights from the Bogacki-Shampine 1989 paper.
    // Equivalent to Julia's b + btilde, where btilde = bhat - b = (5/72, -1/12, -1/9, 1/8).
    static constexpr std::array<double, 4> Bhat = {rat(7, 24), rat(1, 4), rat(1, 3), rat(1, 8)};

    // Bmid: BS3 falls back to cubic Hermite in Julia (no BS3-specific *Interp
    // table in OrdinaryDiffEqLowOrderRK; confirmed by grep for BS3 in interpolants.jl,
    // interp_func.jl, low_order_rk_addsteps.jl — zero matches). Tycho therefore
    // derives Bmid from the Hermite cubic at Θ=0.5 using the method's B vector:
    //
    //   Hermite cubic: y(Θ) = (1-3Θ²+2Θ³)·y₀ + (3Θ²-2Θ³)·y₁
    //                         + h·(Θ-2Θ²+Θ³)·f₀ + h·(-Θ²+Θ³)·f₁
    //   At Θ=0.5:      y_mid = 0.5·(y₀+y₁) + (h/8)·(f₀-f₁)
    //   For FSAL methods (f₁=k_s): b_i(0.5) = 0.5·B_i, with corrections:
    //     b_1(0.5) += +1/8  (f₀ contribution)
    //     b_s(0.5) += -1/8  (f₁ contribution, on the FSAL-last stage)
    //
    // BS3 values: b_i(0.5) = (17/72, 1/6, 2/9, -1/8). Sum = 36/72 = 0.5 ✓.
    // Tycho stores 2·b_i(0.5): (17/36, 1/3, 4/9, -1/4). Sum = 1.0 ✓.
    static constexpr std::array<double, BmidStages> Bmid = {
        17.0 / 36.0,  1.0 / 3.0,  4.0 / 9.0,  -1.0 / 4.0};
};
```

- [ ] **Step 4.3: Add `BS3Trans` internal tag**

Like Tsit5/Tsit5Trans, BS3 is FSAL, so the transcription stepper needs a companion non-FSAL 3-stage tag. Add `IVPAlg::BS3Trans` to the enum and `RKCoeffs<IVPAlg::BS3Trans>` with Stages = 3, Order = 3, ErrorOrder = 0, FSAL = false, HasEmbedded = false. The A matrix drops BS3's last FSAL row; C and B are derived identically to the Tsit5Trans pattern.

- [ ] **Step 4.4: Wire into dispatch (set_method + stepper_compute + bindings)**

Mirror Task 2 with `BS3`. `init_stepper_and_controller<IVPAlg::BS3Trans>` (not `BS3`) for the transcription path. Set `error_order_ = 2` and `dofsal = true` in `stepper_compute`. Add `BS3Trans` to the internal-tag rejection list in `set_method`.

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

## Midpoint Policy (SP2 revision — supersedes prior Option A/B/C discussion)

**All new methods match Julia's exact interpolant at Θ=0.5.** The dense-output schema introduced in Task 3.5 carries the additional f-evaluations ("extra stages") each method's interpolant polynomial needs beyond the main integration stages.

Per-method summary:

| Method   | Stages | InterpStages | `LastStageIsFxf` | `BmidStages` | Julia source |
|----------|--------|--------------|-----------------|--------------|-------------|
| DOPRI54  | 7      | 0            | true (FSAL)     | 7            | existing (unchanged) |
| DOPRI87  | 13     | 0            | false           | 14           | existing (unchanged) |
| Tsit5    | 7      | 0            | true (FSAL)     | 7            | `tsit_tableaus.jl` `Tsit5Interp` |
| BS3      | 4      | 0            | true (FSAL)     | 4            | generic Hermite (Julia default for BS3) |
| BS5      | 8      | 3            | true            | 11           | `low_order_rk_tableaus.jl` `BS5Interp` + `_ode_addsteps!(BS5ConstantCache)` |
| Vern7    | 10     | 6            | true            | 16           | `verner_addsteps.jl` + `verner_tableaus.jl` `Vern7ExtraStages` + `interpolants.jl` |
| Vern8    | 13     | 8            | true            | 21           | `verner_addsteps.jl` + `Vern8ExtraStages` + `interpolants.jl` |
| Vern9    | 16     | 10           | true            | 26           | `verner_addsteps.jl` + `Vern9ExtraStages` + `interpolants.jl` |

`LastStageIsFxf` = "the last main integration stage's k-value equals `f(xf)`." True for FSAL methods trivially. Also true for BS5/Vern7/Vern8/Vern9 because their last main stage is evaluated at `(t+h, xf)` with a-row equal to the b-vector — a standard property of those methods' Butcher tableaus. False only for DOPRI87 (which requires an explicit extra `f(xf)` eval when midpoint is needed — existing behavior, unchanged). This property determines whether `BmidStages` needs a `+1` slot for `f(xf)`.

**Vern9 caveat:** Vern9's perform_step computes 16 internal f-evaluations, but Julia's interpolation polynomial uses only a subset (k_1 and k_8..k_16). In Tycho's model we store all 16 main stages anyway; the `Bmid` weights for the unused internal stages (k_2..k_7) are 0. No special-casing in the stepper — just zero weights.

---

## Task 5: Add BS5 (Bogacki-Shampine 5(4))

**Purpose:** 8-stage non-FSAL 5(4) method with 3-stage lazy interpolant (Julia's `BS5Interp`). First SP2 method to exercise the Task 3.5 extra-stage infrastructure.

**Julia sources:**
- Main tableau: `~/.julia/packages/OrdinaryDiffEqLowOrderRK/5x2GR/src/low_order_rk_tableaus.jl` — function `BS5ConstantCache`
- Extra stages: `low_order_rk_addsteps.jl` — `_ode_addsteps!(...cache::BS5ConstantCache...)` — 3 extra stages k_9, k_10, k_11
- Interpolant polynomial: `low_order_rk_tableaus.jl` — function `BS5Interp_polyweights` — provides `r_ij` coefficients for stages {1, 3, 4, 5, 6, 7, 8, 9, 10, 11}. (Stage 2 has zero interp contribution — BS5's b_2 = 0.)
- Interp evaluation: `interpolants.jl` — lines around `@bs5unpack` / `@bs5pre0`

**Files:**
- Modify: `include/tycho/detail/integrators/rk_coeffs.h`
- Modify: `include/tycho/detail/integrators/integrator.h` (set_method + stepper_compute cases)
- Modify: `src/bindings/integrators/rk_coeffs_bind.cpp`, `integrator_bind.h`
- Create: `tests/cpp/integrators/test_method_bs5.cpp`
- Create: `bench/julia_reference/src/compute_bmid_bs5.jl`
- Create: `bench/julia_reference/test/reference_outputs/bs5_{twobody,crtbp}.bin`

- [ ] **Step 5.1: Transcribe BS5 main tableau**

`Stages = 8, Order = 5, ErrorOrder = 4, FSAL = false, HasEmbedded = true, HasMidpoint = true, InterpStages = 3, LastStageIsFxf = true` (BS5's k_8 is evaluated at (t+h, xf) with a_{8,j} = b_j, so k_8 = f(xf)), `BmidStages = 8 + 3 = 11`.

Transcribe the 7 A rows (a21; a31, a32; a41, a42, a43; a51, a52, a53, a54; a61, a62, a63, a64, a65; a71, a72, a73, a74, a75, a76; a81, a83, a84, a85, a86, a87) and 8-long C, B, Bhat vectors. Source all numeric values from `BS5ConstantCache` (`T::Type{<:CompiledFloats}` variant for bit-identical Float64 values).

- [ ] **Step 5.2: Transcribe BS5 extra-stage tableau**

From `_ode_addsteps!(...BS5ConstantCache...)`, rows for k_9, k_10, k_11:

```cpp
// BS5 extra-stage tableau — computes k_9, k_10, k_11 when midpoint is requested.
// Each row uses k_1..k_8 (main) plus any prior extras.
static constexpr std::array<std::array<double, Stages + InterpStages>, InterpStages> ExtraA = {
    // k_9:  coefficients a91..a98 (and 0 for k_9..k_11 self/prior)
    {a91, a92, a93, a94, a95, a96, a97, a98, 0, 0, 0},
    // k_10: coefficients a101..a108 (for main) + a109 (for k_9)
    {a101, a102, a103, a104, a105, a106, a107, a108, a109, 0, 0},
    // k_11: coefficients a111..a119 (main+k9) + a1110 (for k_10)
    {a111, a112, a113, a114, a115, a116, a117, a118, a119, a1110, 0}
};
static constexpr std::array<double, InterpStages> ExtraC = {c6, c7, c8};
// ^ c6/c7/c8 come from BS5Interp function's return values (not the main c-array).
```

- [ ] **Step 5.3: Compute Bmid via `compute_bmid_bs5.jl`**

Create the helper:

```julia
# Evaluate Julia's BS5Interp polynomial at Θ=0.5 and emit Bmid = 2·b_i(0.5).
# BS5 interp polynomial from low_order_rk_tableaus.jl BS5Interp_polyweights:
#   b_i(Θ) = r_i2·Θ² + r_i3·Θ³ + r_i4·Θ⁴ + r_i5·Θ⁵ + r_i6·Θ⁶
# (minimum order is Θ²; no Θ¹ term because k_1 has separate handling.)
# Exception: b_1 has an extra Θ¹ term (it's the "linear" baseline).
using OrdinaryDiffEqLowOrderRK: BS5Interp_polyweights
using Printf

r = BS5Interp_polyweights(Float64)
# r is a tuple of (r_i6, r_i5, r_i4, r_i3, r_i2) for i ∈ {1, 3, 4, 5, 6, 7, 8, 9, 10, 11}
# (no r_11 in BS5 — only 10 stages contribute to the interp.)
# Indexing convention — need to match interpolants.jl @bs5pre0 macro.

# Bmid layout (11 slots):
#   Bmid[0]  → k_1   (a.k.a. stage 1)
#   Bmid[1]  → k_2   (unused by BS5 interp — weight = 0)
#   Bmid[2..7] → k_3..k_8 (main stages 3..8)
#   Bmid[8..10] → k_9..k_11 (extras)

θ = 0.5
# ... evaluate each b_i(0.5), output 2·b_i, verify Σ = 1.0
```

Run: `julia --project=. src/compute_bmid_bs5.jl`. Paste output into rk_coeffs.h.

- [ ] **Step 5.4: Wire into dispatch (set_method, stepper_compute)**

Mirror Task 2. `dofsal = false` for BS5 in stepper_compute. Set `error_order_ = 4`.

- [ ] **Step 5.5: Expose in Python bindings**

- [ ] **Step 5.6: Generate Julia reference**

```bash
julia --project=. src/run_method.jl BS5 two_body 1e-10 1e-11 test/reference_outputs/bs5_twobody.bin
julia --project=. src/run_method.jl BS5 crtbp 1e-10 1e-11 test/reference_outputs/bs5_crtbp.bin
```

- [ ] **Step 5.7: Validate against Julia — compare final state and midpoint**

Create `tests/cpp/integrators/test_method_bs5.cpp`:
- Two tests per problem: `final_state_matches_julia` and `midpoint_matches_julia_at_first_accepted_step`.
- The midpoint test integrates 1 accepted step (by choosing h to match Julia's first dt), computes `xf_mid` via Tycho, and compares against Julia's `sol(dt/2)` queried via a helper script extension.

Expected tolerance: 1e-7 for final state, 1e-9 for midpoint (5th-order local error at tight abstol).

- [ ] **Step 5.8: Build, run tests, commit**

```bash
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ninja -j4 tycho_tests"
conda run -n tycho bash -c "cd /home/ghecht/Projects/tycho/build && ctest -R BS5 --output-on-failure"
git add -A && git commit -m "feat: add BS5 method with exact Julia interpolant (3 extra stages)"
```

---

## Task 6: Add Vern7 (Verner 7(6))

**Purpose:** 10-stage non-FSAL 7(6) method by Jim Verner with 6-stage lazy interpolant (Julia's `Vern7Interp`).

**Julia sources:**
- Main tableau: `~/.julia/packages/OrdinaryDiffEqVerner/GlLDJ/src/verner_tableaus.jl` — `Vern7Tableau`
- Extra stages: `verner_addsteps.jl` — `_ode_addsteps!(...cache::Vern7ConstantCache...)` block for `length(k) < 16` — k_11..k_16 (6 stages) using `Vern7ExtraStages` tableau
- Interpolation polynomial: `interpolants.jl` — search for Vern7 b_i(Θ) formula; polynomial order 7.

- [ ] **Step 6.1: Write `emit_tableau.jl` helper**

Vern7 has ~50 non-zero A entries + 6 extra-stage rows. Machine-extraction preferred over hand-transcription.

```julia
# bench/julia_reference/src/emit_tableau.jl
using OrdinaryDiffEqVerner: Vern7Tableau, Vern7ExtraStages
using Printf

function emit_cache(name::String, cache)
    @printf("// %s: fields emitted from OrdinaryDiffEqVerner\n", name)
    for field in fieldnames(typeof(cache))
        val = getfield(cache, field)
        if val isa Number
            @printf("    static constexpr double %s = %.17e;\n", field, val)
        end
    end
end

tab = Vern7Tableau(Float64, Float64)
emit_cache("Vern7Tableau", tab)
extra = Vern7ExtraStages(Float64, Float64)
emit_cache("Vern7ExtraStages", extra)
```

Run: `julia --project=. src/emit_tableau.jl > /tmp/vern7_coeffs.txt`.

- [ ] **Step 6.2: Hand-format into `RKCoeffs<IVPAlg::Vern7>`**

```cpp
template <> struct RKCoeffs<IVPAlg::Vern7> {
    static constexpr int Stages = 10;
    static constexpr int Order = 7;
    static constexpr int ErrorOrder = 6;
    static constexpr bool FSAL = false;
    static constexpr bool HasEmbedded = true;
    static constexpr bool HasMidpoint = true;

    static constexpr int InterpStages = 6;
    static constexpr bool LastStageIsFxf = true;  // k_10 is evaluated at (t+h, xf)
    static constexpr int BmidStages = Stages + InterpStages;  // 16

    // A (9 rows × 9 cols), C (9 entries), B, Bhat (10 entries) — from Vern7Tableau
    // ...

    // ExtraA: 6 rows × 16 cols (Stages + InterpStages). Row e uses columns
    // 0..(Stages+e-1); later columns = 0.
    static constexpr std::array<std::array<double, 16>, 6> ExtraA = { /* ... */ };
    static constexpr std::array<double, 6> ExtraC = { /* c11..c16 from Vern7ExtraStages */ };

    // Bmid[0..15] = 2·b_i(Θ=0.5) from Vern7 interpolation polynomial.
    static constexpr std::array<double, 16> Bmid = { /* compute_bmid_vern7.jl output */ };
};
```

- [ ] **Step 6.3: Compute Bmid via `compute_bmid_vern7.jl`**

Vern7's b_i(Θ) is order-7 in Θ. Transcribe the polynomial coefficients from `interpolants.jl` (search for the Vern7 interp function or `@vern7unpack`/`@vern7pre0` macros; the polynomial is explicit in the expression `@muladd Θ*(... + Θ*(...))`). Evaluate at Θ=0.5 for each of 16 stages, output `2·b_i(0.5)`, verify Σ = 1.0.

- [ ] **Step 6.4: Wire into dispatch + Python bindings**

Mirror Task 2. `dofsal = false` for Vern7.

- [ ] **Step 6.5: Generate Julia reference + test**

```bash
julia --project=. src/run_method.jl Vern7 two_body 1e-12 1e-13 test/reference_outputs/vern7_twobody.bin
julia --project=. src/run_method.jl Vern7 crtbp 1e-12 1e-13 test/reference_outputs/vern7_crtbp.bin
```

Tolerance: 1e-10 for final state, 1e-11 for midpoint.

- [ ] **Step 6.6: Build, test, commit**

---

## Task 7: Add Vern8 (Verner 8(7))

**Purpose:** 13-stage non-FSAL 8(7) method with 8-stage lazy interpolant.

**Julia sources:** `verner_tableaus.jl` `Vern8Tableau`; `verner_addsteps.jl` `Vern8ConstantCache` block (`length(k) < 21`) — k_14..k_21 (8 stages); `interpolants.jl` Vern8 polynomial.

- [ ] **Step 7.1: Extract coefficients via `emit_tableau.jl`** (extend helper to Vern8)
- [ ] **Step 7.2: Add `IVPAlg::Vern8` + tableau**

`Stages = 13, Order = 8, ErrorOrder = 7, FSAL = false, HasEmbedded = true, HasMidpoint = true, InterpStages = 8, LastStageIsFxf = true, BmidStages = 21`.

- [ ] **Step 7.3: Transcribe ExtraA (8 rows × 21 cols), ExtraC (8), Bmid (21)**

Same pattern as Vern7. Bmid via `compute_bmid_vern8.jl`.

- [ ] **Step 7.4: Wire, bindings, test, commit**

Tolerance: 1e-11 for final state at atol=1e-13; 1e-12 for midpoint.

---

## Task 8: Add Vern9 (Verner 9(8))

**Purpose:** 16-stage non-FSAL 9(8) method — highest-order in SP2 — with 10-stage lazy interpolant. Special Julia storage convention (some internal stages unused by interp polynomial).

**Julia sources:** `verner_tableaus.jl` `Vern9Tableau`; `verner_addsteps.jl` `Vern9ConstantCache` block (`length(k) < 20`) — k_11..k_20 (10 extras); `interpolants.jl` Vern9 polynomial.

- [ ] **Step 8.1: Extract coefficients via `emit_tableau.jl`** (extend helper to Vern9)

- [ ] **Step 8.2: Map Julia's k-indexing to Tycho's**

Julia stores only 10 of the 16 internal stages in the k array: `k[1]=k_1, k[2]=k_8, k[3]=k_9, ..., k[10]=k_16`. When transcribing ExtraA rows from `verner_addsteps.jl`, each `a_IJ * k[p]` reference maps to Tycho's `ExtraA[I-11][?]`:
- `k[1]`  → Tycho column 0  (main stage 0 = k_1)
- `k[2]`  → Tycho column 7  (main stage 7 = k_8)
- `k[3..10]` → Tycho columns 8..15  (main stages 8..15 = k_9..k_16)
- `k[11..20]` → Tycho columns 16..25  (extras)

Document this mapping in a comment block above `ExtraA` in the specialization.

- [ ] **Step 8.3: Add `IVPAlg::Vern9` + tableau**

`Stages = 16, Order = 9, ErrorOrder = 8, FSAL = false, HasEmbedded = true, HasMidpoint = true, InterpStages = 10, LastStageIsFxf = true, BmidStages = 26`.

Bmid values: set `Bmid[1..6] = 0` (weights for unused internal k_2..k_7). All other weights from Vern9 interpolation polynomial at Θ=0.5.

- [ ] **Step 8.4: Wire, bindings, test, commit**

Tolerance: 1e-12 for final state at atol=1e-14; 1e-13 for midpoint.

Monitor binding TU RAM: after Vern9, run `/usr/bin/time -v ninja -j4 _tychopy` and check max_resident. If >8 GB, defer Vern9 by commenting out its `emplace` in the binding enum.

---

## Task 8.5: Extern Template Declarations (binding TU mitigation)

**Purpose:** SP2 adds 6 methods, so the `Integrator<DODE>` instantiation cost grows 4× in binding TUs. Add `extern template` declarations for built-in ODEs to move instantiation to dedicated .cpp files.

**Files:**
- Modify: `include/tycho/detail/integrators/integrator.h` (extern decls)
- Create: `src/integrators/integrator_instantiations.cpp` (explicit instantiations)
- Modify: `src/integrators/tycho_integrators.h` (include new .cpp)

- [ ] **Step 8.5.1: Add extern declarations at the bottom of `integrator.h`**

```cpp
// Extern template declarations for common ODEs to reduce binding TU RAM.
namespace tycho {
extern template struct Integrator<tycho::astro::Kepler>;
extern template struct Integrator<tycho::astro::CartesianDynamics>;
extern template struct Integrator<tycho::astro::CRTBPDynamics>;
extern template struct Integrator<tycho::astro::MEEDynamics>;
}
```

- [ ] **Step 8.5.2: Create explicit instantiation TU**

`src/integrators/integrator_instantiations.cpp`:

```cpp
#include "tycho/tycho.h"
namespace tycho {
template struct Integrator<tycho::astro::Kepler>;
template struct Integrator<tycho::astro::CartesianDynamics>;
template struct Integrator<tycho::astro::CRTBPDynamics>;
template struct Integrator<tycho::astro::MEEDynamics>;
}
```

- [ ] **Step 8.5.3: Measure binding TU RAM before/after**

```bash
/usr/bin/time -v ninja -j1 _tychopy 2>&1 | grep "Maximum resident"
```

Record baseline and post-change numbers in the commit message.

- [ ] **Step 8.5.4: Verify all tests still pass, commit**

```bash
ninja -j4 all && ctest --output-on-failure
git add -A && git commit -m "refactor: extern template declarations for built-in ODEs"
```

---

## Task 9: Per-Method Benchmarks

**Purpose:** Add Google Benchmark entries comparing all 8 methods (DOPRI54, DOPRI87, Tsit5, BS3, BS5, Vern7, Vern8, Vern9) on two-body and CR3BP problems.

**Files:**
- Create: `bench/cpp/bench_method_comparison.cpp`
- Modify: `bench/cpp/CMakeLists.txt`

- [ ] **Step 9.1: Create benchmark file**

`bench/cpp/bench_method_comparison.cpp`:

```cpp
#include <benchmark/benchmark.h>
#include <tycho/tycho.h>

using namespace tycho;

static void BM_Method(benchmark::State &state, IVPAlg alg, double atol) {
    astro::Kepler kep(398600.4418);
    double r0 = 7000.0, v0 = std::sqrt(398600.4418 / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0, 0, 0, v0, 0, 0;
    double T = 2 * M_PI * std::sqrt(r0 * r0 * r0 / 398600.4418);
    for (auto _ : state) {
        Integrator<astro::Kepler> integ(kep, alg, 10.0);
        integ.set_abs_tol(atol);
        integ.set_rel_tol(atol);
        auto xf = integ.integrate(x0, T / 4.0);
        benchmark::DoNotOptimize(xf);
    }
}

#define REGISTER(alg, tol_str, tol_val) \
    BENCHMARK_CAPTURE(BM_Method, alg##_##tol_str, IVPAlg::alg, tol_val)

REGISTER(DOPRI54, tol1e6,  1e-6);  REGISTER(DOPRI87, tol1e6,  1e-6);
REGISTER(Tsit5,   tol1e6,  1e-6);  REGISTER(BS3,     tol1e6,  1e-6);
REGISTER(BS5,     tol1e6,  1e-6);  REGISTER(Vern7,   tol1e6,  1e-6);
REGISTER(Vern8,   tol1e6,  1e-6);  REGISTER(Vern9,   tol1e6,  1e-6);

REGISTER(DOPRI54, tol1e9,  1e-9);  REGISTER(DOPRI87, tol1e9,  1e-9);
REGISTER(Tsit5,   tol1e9,  1e-9);  REGISTER(BS5,     tol1e9,  1e-9);
REGISTER(Vern7,   tol1e9,  1e-9);  REGISTER(Vern8,   tol1e9,  1e-9);
REGISTER(Vern9,   tol1e9,  1e-9);

REGISTER(DOPRI54, tol1e12, 1e-12); REGISTER(DOPRI87, tol1e12, 1e-12);
REGISTER(Tsit5,   tol1e12, 1e-12); REGISTER(Vern7,   tol1e12, 1e-12);
REGISTER(Vern8,   tol1e12, 1e-12); REGISTER(Vern9,   tol1e12, 1e-12);

BENCHMARK_MAIN();
```

- [ ] **Step 9.2: Add to `bench/cpp/CMakeLists.txt`**

```cmake
add_executable(bench_method_comparison bench_method_comparison.cpp)
target_compile_options(bench_method_comparison PRIVATE ${COMPILE_FLAGS})
target_include_directories(bench_method_comparison PRIVATE ${INCLUDE_DIRS} ${TYCHO_INCLUDES})
target_link_libraries(bench_method_comparison PRIVATE tycho benchmark::benchmark)
set_property(TARGET bench_method_comparison PROPERTY JOB_POOL_COMPILE heavy_compile)
add_dependencies(bench_all bench_method_comparison)
```

- [ ] **Step 9.3: Build and run**

```bash
conda run -n tycho bash -c "cmake --preset linux-clang-release -DBUILD_CPP_BENCHMARKS=ON && cd build && ninja -j4 bench_method_comparison"
./build/bench/cpp/bench_method_comparison --benchmark_format=console > bench/sp2_method_comparison.txt
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

## Task 10.5: SuperScalar Path Coverage

**Purpose:** Verify each new method works through the SIMD batch integration path (`integrate_impl_vectorized`), not just the scalar path.

**Files:**
- Create: `tests/cpp/integrators/test_method_superscalar.cpp`

- [ ] **Step 10.5.1: Create SuperScalar test for new methods**

```cpp
#include "integrator_test_utils.h"
#include <gtest/gtest.h>
using namespace tycho;
using namespace TychoTest;

class SuperScalarMethodTest : public VectorFunctionFixture {};

template <IVPAlg Alg>
static void run_batch_test(double atol, double rtol, double tol) {
    Kepler kep(398600.4418);
    Integrator<Kepler> integ(kep, Alg, 10.0);
    integ.set_abs_tol(atol);
    integ.set_rel_tol(rtol);
    integ.vectorize_batch_calls_ = true;
    double r0 = 7000.0, v0 = std::sqrt(398600.4418 / r0);
    std::vector<Eigen::VectorXd> x0s;
    Eigen::VectorXd tfs(4);
    for (int i = 0; i < 4; ++i) {
        Eigen::VectorXd x0(7);
        x0 << r0 + 100.0 * i, 0, 0, 0, v0, 0, 0;
        x0s.push_back(x0);
        tfs[i] = 2000.0 + 100.0 * i;
    }
    auto xfs = integ.integrate(x0s, tfs);
    // Compare against scalar
    integ.vectorize_batch_calls_ = false;
    for (size_t i = 0; i < x0s.size(); ++i) {
        auto xf_scalar = integ.integrate(x0s[i], tfs[i]);
        for (int j = 0; j < xf_scalar.size(); ++j) {
            EXPECT_NEAR(xfs[i][j], xf_scalar[j], tol)
                << "SIMD vs scalar mismatch: method=" << int(Alg) << " traj=" << i << " j=" << j;
        }
    }
}

TEST_F(SuperScalarMethodTest, Tsit5)  { run_batch_test<IVPAlg::Tsit5>(1e-10, 1e-11, 1e-7); }
TEST_F(SuperScalarMethodTest, BS3)    { run_batch_test<IVPAlg::BS3>(1e-6, 1e-7, 1e-4); }
TEST_F(SuperScalarMethodTest, BS5)    { run_batch_test<IVPAlg::BS5>(1e-10, 1e-11, 1e-7); }
TEST_F(SuperScalarMethodTest, Vern7)  { run_batch_test<IVPAlg::Vern7>(1e-11, 1e-12, 1e-8); }
TEST_F(SuperScalarMethodTest, Vern8)  { run_batch_test<IVPAlg::Vern8>(1e-12, 1e-13, 1e-9); }
TEST_F(SuperScalarMethodTest, Vern9)  { run_batch_test<IVPAlg::Vern9>(1e-13, 1e-14, 1e-10); }
```

- [ ] **Step 10.5.2: Add to CMakeLists, build, test, commit**

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

Expected: 30 pass, 8 skipped (basemap unavailable) — same as SP1 baseline on this machine. **For pre-merge gating per CLAUDE.md:** install `mpl_toolkits.basemap` in the reviewer's environment so all 38 examples run. A skipped example is not a failure for SP2 validation (the 8 skipped examples don't exercise SP2 additions), but merging to `main` requires all 38 to pass per project policy.

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
