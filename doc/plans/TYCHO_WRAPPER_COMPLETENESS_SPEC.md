# Spec: Phase/OCP Wrapper Completeness (Subsystem 1)

> Close the API gaps between the builder wrappers (Phase, OptimalControlProblem)
> and their underlying base classes, so users never need `phase.base()` in normal
> usage. All new methods provide both index-based and named-variable overloads.

## Motivation

During the C++ example porting effort, 10+ examples required `phase.base()` or
`ocp.base()` fallbacks to access methods that exist on `ODEPhaseBase` /
`OptimalControlProblemBase` but aren't forwarded through the builder wrappers.
This defeats the purpose of the builder API — users should have the *choice* of
named variables or indices, and both should be first-class.

## Scope

This spec covers only wrapper forwarding and name-resolution fixes. It does NOT
cover:
- Runtime ODE integrator (`ODE::integrator()`) — Subsystem 2
- VF expression additions (RowMatrix, quat_rotate) — Subsystem 3
- InterpTable VF composition — Subsystem 4
- Advanced OC features (LGLInterpTable, STM, Jet) — Subsystem 5
- Domain models (MEE factories) — Subsystem 6

## Changes

### 1. Phase Wrapper — Method Forwarding

Add the following methods to `Phase` (in `phase_wrapper.h`), forwarding to
`ODEPhaseBase` via `base()`:

#### Trajectory refinement

```cpp
void refine_traj_manual(int ndef);
void refine_traj_manual(const Eigen::VectorXd& dbs, const Eigen::VectorXi& dpb);
```

No named-variable variant needed — these don't reference variable names.

#### Variable substitution

```cpp
// Index-based
void sub_variable(PhaseRegionFlags region, int var, double val);
void sub_variables(PhaseRegionFlags region, const Eigen::VectorXi& vars,
                   const Eigen::VectorXd& vals);

// Named
void sub_variable(PhaseRegionFlags region, const std::string& var, double val);
void sub_variables(PhaseRegionFlags region,
                   const std::vector<std::string>& vars,
                   const Eigen::VectorXd& vals);
```

Named overloads resolve names via the ODE's `VarRegistry` for XtUP regions
(`Front`, `Back`, `Path`, `ODEParams`), and via the Phase's static param
registry for the `StaticParams` region. The region flag determines which
registry is consulted.

### 2. Phase Wrapper — Mixed Variable Source Constraints

Add overloads to constraint methods that accept separate XtUP, ODE param, and
static param variable lists. Affected methods:

- `add_equal_con`
- `add_inequal_con`
- `add_upper_func_bound` / `add_lower_func_bound` / `add_lu_func_bound`

Each gets two new overloads (preserving existing simple overloads):

```cpp
// Index-based with ODE params + static params
void add_inequal_con(PhaseRegionFlags region, GenFunc func,
                     const Eigen::VectorXi& xtup_vars,
                     const Eigen::VectorXi& ode_param_vars,
                     const Eigen::VectorXi& static_param_vars,
                     double scale = 1.0);

// Named with ODE params + static params
void add_inequal_con(PhaseRegionFlags region, GenFunc func,
                     const std::vector<std::string>& xtup_vars,
                     const std::vector<std::string>& ode_param_vars,
                     const std::vector<std::string>& static_param_vars,
                     double scale = 1.0);
```

Same pattern for `add_equal_con` and all func bound methods. Named overloads
resolve XtUP names via the ODE's `VarRegistry`, ODE param names via the ODE's
`VarRegistry` (with index translation from XtUP-space to P-relative), and
static param names via the Phase's static param registry.

### 3. Phase Wrapper — Static Param Naming

Add a lightweight naming mechanism for static params on Phase, separate from the
ODE's `VarRegistry`:

```cpp
void set_static_param_names(
    std::initializer_list<std::pair<std::string, int>> names);
void add_static_param_name(const std::string& name, int sp_index);
void add_static_param_group(const std::string& name, int start, int count);
```

Internally stored as `std::unordered_map<std::string, Eigen::VectorXi>` with a
thin `resolve_sp(const std::string& name)` method. Validation: bounds checking
against the static param vector size, duplicate name detection. No reuse of
`VarRegistry` — static params are a Phase-level concept, not an ODE-level
concept.

### 4. Phase Wrapper — ODEParams Name Resolution Fix

**Bug:** When a method receives `PhaseRegionFlags::ODEParams` and resolves a
name like `"rad"` via the ODE's `VarRegistry`, it gets the XtUP-space index
(e.g., 5). But `ODEPhaseBase` expects the P-relative index (e.g., 0).

**Fix:** Add a single helper method on Phase:

```cpp
int to_region_index(PhaseRegionFlags region, int xtup_index) const;
```

Logic:
- `ODEParams` → subtract P-block offset (`xvars + 1 + uvars`)
- `Front` / `Back` / `Path` → return as-is (XtUP indices are what base expects)
- `StaticParams` → not applicable (resolved via SP registry, not XtUP)

Called at the resolution point in every method that translates names to indices.
One function, clear logic, used everywhere.

### 5. ODE::phase() — LerpIG Flag

Add a fourth parameter to `ODE::phase()` for sparse waypoint interpolation:

```cpp
Phase phase(TranscriptionModes mode,
            const std::vector<Eigen::VectorXd>& traj,
            int nsegs,
            bool lerp_ig = false);
```

Default `false` preserves backward compatibility. When `true`, passes through to
the underlying phase construction to interpolate sparse waypoints onto the
collocation mesh.

### 6. OCP Wrapper — add_direct_link_equal_con

Add `add_direct_link_equal_con` to the `OptimalControlProblem` wrapper:

```cpp
// Index-based
void add_direct_link_equal_con(int phase_a, PhaseRegionFlags region_a,
                               const Eigen::VectorXi& vars_a,
                               int phase_b, PhaseRegionFlags region_b,
                               const Eigen::VectorXi& vars_b);

// Named
void add_direct_link_equal_con(int phase_a, PhaseRegionFlags region_a,
                               const std::vector<std::string>& vars_a,
                               int phase_b, PhaseRegionFlags region_b,
                               const std::vector<std::string>& vars_b);
```

For the named overload, the OCP wrapper looks up each phase's registries
(already stored via `add_phase()`) to resolve names. Name resolution uses the
same region-aware logic from Section 4 — `ODEParams` names get translated to
P-relative indices, `StaticParams` names use the SP registry, etc.

## Verification

Update C++ examples that currently use `phase.base()` or `ocp.base()`
workarounds to use the new APIs:

- **ParallelParking** — `refine_traj_manual`, `sub_variable`, mixed-var
  `add_inequal_con`
- **Reentry** — `refine_traj_manual`
- **MultiPhaseCannon** — `add_direct_link_equal_con`, mixed-var constraints
- **MultiSpacecraftOpt** — `sub_variable` / `sub_variables`, OCP link calls
- Any other example that fell back to `.base()`

**Success criteria per example:**
1. No calls to `phase.base()` or `ocp.base()` remain
2. Compiles and passes CTest
3. Same convergence flag as before (optimal or acceptable)
4. Objective value within tolerance of previous result

## Files Modified

- `include/tycho/detail/optimal_control/builder/phase_wrapper.h` — new methods,
  SP registry, `to_region_index` helper
- `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` — new
  `add_direct_link_equal_con` overloads
- `include/tycho/detail/optimal_control/builder/runtime_ode.h` — `lerp_ig`
  parameter on `ODE::phase()`
- `src/optimal_control/runtime_ode.cpp` — `lerp_ig` implementation (if not
  header-only)
- `examples/cpp_examples/builder/parallel_parking/main.cpp` — update to new API
- `examples/cpp_examples/builder/reentry/main.cpp` — update to new API
- `examples/cpp_examples/builder/multi_phase_cannon/main.cpp` — update to new API
- `examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp` — update to new API
- Other examples as needed
