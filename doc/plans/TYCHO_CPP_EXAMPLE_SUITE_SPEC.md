# Spec: C++ Builder API Example Suite

> Port all unique Python examples to C++ using the builder API.
> Serves as both a C++ integration test suite and an API deficiency catalog.

## Motivation

The Python example suite (38 scripts) is the primary acceptance gate for Tycho.
There is no equivalent C++ coverage — only 4 builder examples exist. Porting all
unique Python examples to C++ will:

1. Validate the builder API against real problems at scale
2. Expose remaining API deficiencies (missing features, ergonomic issues)
3. Create a C++ integration test suite with convergence assertions
4. Provide reference code for C++ users

## Prerequisites

- **Builder type renames** (see `TYCHO_BUILDER_TYPE_RENAMES_SPEC.md`) must land first.
  All new examples use the renamed types (`ODE`, `OptimalControlProblem`, etc.).

## Example Inventory

23 examples after deduplication (removing UpdatedInterface duplicates and 4 already
ported). Organized into tiers by complexity and expected API gaps.

### Tier 1: Simple Single-Phase (no expected gaps)

| # | Example | States | Ctrls | Dynamics | Objective | Key Features |
|---|---------|--------|-------|----------|-----------|--------------|
| 1 | VanDerPol | 2 | 1 | Oscillator | min integral(x0^2+x1^2+u^2) | BlockConstant control |
| 2 | BrysonDenham | 2 | 1 | Point mass | min integral(u^2/2) | State upper bound (x<=1/9) |
| 3 | MountainCar | 2 | 1 | Terrain oscillator | min time | State bounds, scaling |
| 4 | AnalyticExample | 1 | 1 | Linear | min custom integral | Custom objective function |
| 5 | BikeObstacle | 4 | 2 | Bicycle kinematics | min time | Obstacle avoidance inequality |
| 6 | FreeFlyingRobot | 6 | 4 | Planar 6-DOF | min integral(u_sum) | Large control vector |

### Tier 2: Single-Phase with Advanced Features

| # | Example | States | Ctrls | Dynamics | Objective | Key Features | Notes |
|---|---------|--------|-------|----------|-----------|--------------|-------|
| 7 | CartPole | 4 | 1 | Swingup, matrix inverse | min control effort | `RowMatrix`, `.inverse()` | **Validate matrix VF ops work in builder lambda** |
| 8 | ParallelParking | 6 | 2+1p | Bicycle kinematics | min time | Path constraints, static params, `refine_traj_manual`, BlockConstant | May need `phase.base()` for manual mesh refinement |
| 9 | HangingChain | 1 | 1+1p | Chain statics | min potential energy | Integral parameter constraint | **Jet parallel solve gap** — Phase wrapper has no Jet integration |
| 10 | Reentry | 5 | 2 | Shuttle reentry | max cross-range | Heating constraint, adaptive mesh, two-step solve | |
| 11 | MinimumTimeToClimb | 4 | 1 | Aircraft climb | min time | Inline atmospheric model, adaptive mesh | |

### Tier 3: Multi-Phase

| # | Example | States | Ctrls | Phases | Dynamics | Objective | Key Features |
|---|---------|--------|-------|--------|----------|-----------|--------------|
| 12 | GoddardRocket | 3 | 1 | 3 | Rocket ascent | max altitude | Singular arc, stage separation |
| 13 | MultiPhaseCannon | 4 | 0 | 2 | Ballistic trajectory | max range | ODE parameter opt, phase-dependent controls |
| 14 | MultiPhaseZermelo | 2 | 1 | N | Point navigation | min time | Region-dependent wind, continuation |

### Tier 4: Low-Thrust Astrodynamics

| # | Example | States | Ctrls | Dynamics | Objective | Expected Gaps |
|---|---------|--------|-------|----------|-----------|---------------|
| 15 | SimpleLowThrust | 6 | 3 | Two-body + thrust | min time/power/mass | Likely none (Cartesian dynamics) |
| 16 | TopputtoLowThrust | 4 | 2 | Polar orbit transfer | min time, min mass | Control-law integration for initial guess |
| 17 | DionysusLowThrust | 7 | 3 | MEE + CSI thruster | max final mass | **MEE astro models from C++** |
| 18 | BettsLowThrust | 7 | 3+1p | MEE + J2/J3/J4 | max final mass | **Interp tables, MEE, control-law integration** |
| 19 | OrbitContinuation | 7 | 0 | CR3BP | Periodic orbits | **CR3BP utils, STM integration, continuation** |

### Tier 5: Advanced Astrodynamics

| # | Example | States | Ctrls | Phases | Dynamics | Expected Gaps |
|---|---------|--------|-------|--------|----------|---------------|
| 20 | Heteroclinic | 7 | 0 | 2 | CR3BP manifolds | **CR3BP, manifold generation, event detection** |
| 21 | OptimalDocking | 13-20 | 6 | 1 | Relative quaternion | **Quaternion ops, interp table for target traj** |
| 22 | MultiSpacecraftOpt | 6*N | 3*N | N | Two-body multi-agent | **Link parameters, continuation loop** |

### Tier 6: Interpolation Tables

| # | Example | States | Ctrls | Dynamics | Expected Gaps |
|---|---------|--------|-------|----------|---------------|
| 23 | MinTimeTClimbTables | 4 | 1 | Aircraft + table aero | **InterpTable1D/2D in builder API** |

## Directory Structure

All new examples under `examples/cpp_examples/builder/<snake_case_name>/`:

```
examples/cpp_examples/builder/
  brachistochrone/          # (existing)
  zermelo/                  # (existing)
  hypersens/                # (existing)
  delta3_launch/            # (existing)
  van_der_pol/              # new
  bryson_denham/            # new
  mountain_car/             # new
  analytic_example/         # new
  bike_obstacle/            # new
  free_flying_robot/        # new
  cart_pole/                # new
  parallel_parking/         # new
  hanging_chain/            # new
  reentry/                  # new
  minimum_time_to_climb/    # new
  goddard_rocket/           # new
  multi_phase_cannon/       # new
  multi_phase_zermelo/      # new
  simple_low_thrust/        # new
  topputto_low_thrust/      # new
  dionysus_low_thrust/      # new
  betts_low_thrust/         # new
  orbit_continuation/       # new
  heteroclinic/             # new
  optimal_docking/          # new
  multi_spacecraft_opt/     # new
  min_time_climb_tables/    # new
```

Each directory contains `CMakeLists.txt` + `main.cpp`.

## Example Structure (template)

Based on the existing builder examples (which use `iostream`, `PSIOPT::ConvergenceFlags`,
and `phase.return_traj()`):

```cpp
#include "tycho/tycho.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace tycho;

int main() {
    // 1. Define ODE
    auto ode = oc::ODEBuilder(nstates, ncontrols)
        .var_names({...})
        .define([](auto& args) { ... })
        .build();

    // 2. Configure phase
    auto phase = ode.phase(transcription, segments);
    phase.add_boundary_value(...);
    phase.add_delta_time_objective(1.0);
    // ... constraints, bounds, objectives

    // 3. Solve
    phase.optimizer().set_...();
    auto flag = phase.solve_optimize();

    // 4. Convergence check
    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "FAILED: solver did not converge (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    // 5. Extract and verify results
    auto traj = phase.return_traj();
    // Extract objective from trajectory (e.g., final time for min-time problems)
    double tf = traj.back()[time_index];
    std::cout << std::fixed << std::setprecision(6)
              << "Objective: " << tf << "\n";

    if (std::abs(tf - EXPECTED_TF) > TOLERANCE) {
        std::cerr << "FAILED: objective " << tf
                  << " != expected " << EXPECTED_TF << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASSED\n";
    return EXIT_SUCCESS;
}
```

## Build System

- Each example gets a `CMakeLists.txt` that adds an executable and links `tycho`
- `examples/cpp_examples/CMakeLists.txt` conditionally adds all subdirectories
  when `BUILD_CPP_EXAMPLES=ON`
- All example executables placed in `heavy_compile` job pool (via a foreach loop
  over all builder example targets in the parent CMakeLists.txt)
- Add a custom target `cpp_examples_all` that depends on all example executables
  (must be created in `examples/cpp_examples/CMakeLists.txt`)
- Each example registered as a CTest test via `add_test(NAME cpp_example_<name>
  COMMAND <target>)` in `examples/cpp_examples/CMakeLists.txt`, enabling
  `ctest -R cpp_example` to run all C++ examples

## Gap Tracking

A new file `doc/cpp_example_api_gaps.md` will track every feature gap discovered:

```markdown
# C++ Builder API Gaps (discovered during example porting)

| Example | Feature | Severity | Description | Suggested Fix |
|---------|---------|----------|-------------|---------------|
| BettsLT | InterpTable | High | No InterpTable1D/2D in builder API | Expose via ODE lambda or add builder method |
| ... | ... | ... | ... | ... |
```

Examples blocked by gaps will compile but include `// TODO: <description>` markers
at the blocked sections, with the gap tracked in the above document.

## Verification

1. `cmake --preset <preset> -DBUILD_CPP_EXAMPLES=ON`
2. `ninja -j8 cpp_examples_all` — all examples compile
3. Each example exits 0 and prints `PASSED` (or documents gap with TODO)
4. `ctest -R cpp_example` — all registered examples pass
5. No regressions in existing C++ tests or Python examples
6. Integration: C++ examples become step 5 of the pre-merge verification
   sequence in CLAUDE.md (after benchmarks)

## Implementation Order

1. Tier 1 (6 examples) — validates core builder API
2. Tier 2 (5 examples) — mesh refinement, special constraints, matrix ops, Jet
3. Tier 3 (3 examples) — multi-phase, OCP linking
4. Tier 4 (5 examples) — low-thrust, astro features (gap discovery)
5. Tier 5 (3 examples) — advanced astro (most gaps expected)
6. Tier 6 (1 example) — interpolation tables

Each tier is independently valuable and can be merged separately.
