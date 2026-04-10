# Phase/OCP Wrapper Completeness — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all API gaps between the builder wrappers (`Phase`, `OptimalControlProblem`) and their base classes so users never need `phase.base()` in normal usage.

**Architecture:** All changes are in 3 header files (`phase_wrapper.h`, `ocp_wrapper.h`, `runtime_ode.h`) plus one `.cpp` file. New methods follow the existing pattern: named overloads resolve via `VarRegistry` then delegate to `phase_->`. A `to_region_index` helper fixes the ODEParams index translation bug across all existing and new methods. A lightweight SP name registry on Phase enables named static param references.

**Tech Stack:** C++20, Eigen, fmt, existing Phase/OCP wrapper headers

**Spec:** `doc/plans/TYCHO_WRAPPER_COMPLETENESS_SPEC.md`

---

## File Structure

**Modified files:**
- `include/tycho/detail/optimal_control/builder/phase_wrapper.h` — to_region_index helper, SP registry, new methods, fix existing methods
- `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` — add_direct_link_equal_con overloads, store Phase references
- `include/tycho/detail/optimal_control/builder/runtime_ode.h` — lerp_ig parameter declaration
- `src/optimal_control/runtime_ode.cpp` — lerp_ig implementation

**Example files updated:**
- `examples/cpp_examples/builder/parallel_parking/main.cpp`
- `examples/cpp_examples/builder/reentry/main.cpp`
- `examples/cpp_examples/builder/multi_phase_cannon/main.cpp`
- `examples/cpp_examples/builder/simple_low_thrust/main.cpp`
- `examples/cpp_examples/builder/topputto_low_thrust/main.cpp`
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp`
- `examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp`

---

## Task 0: to_region_index Helper + Fix Existing Methods

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/phase_wrapper.h`

This is the foundational fix — a single helper that translates XtUP-space indices to region-relative indices. Then retrofit all existing named-variable methods to use it.

- [ ] **Step 1: Read the current phase_wrapper.h**

Read `include/tycho/detail/optimal_control/builder/phase_wrapper.h` to understand the current `resolve_single()` usage pattern.

- [ ] **Step 2: Add to_region_index helper to Phase private section**

Add after the existing `resolve_single` method (around line 483):

```cpp
/// Translate a XtUP-space index to the region-relative index that
/// ODEPhaseBase expects.  For ODEParams, subtract the P-block offset.
/// For StaticParams, pass through as-is (already SP-relative).
/// For all other regions, return as-is (XtUP is what base expects).
int to_region_index(PhaseRegionFlags region, int xtup_index) const {
    if (region == PhaseRegionFlags::ODEParams) {
        int p_offset = registry_.xvars() + 1 + registry_.uvars();
        int result = xtup_index - p_offset;
        if (result < 0 || result >= registry_.pvars()) {
            throw std::invalid_argument(fmt::format(
                "Phase::to_region_index: XtUP index {} does not fall in the P-block "
                "[{}, {}) for ODEParams region",
                xtup_index, p_offset, p_offset + registry_.pvars()));
        }
        return result;
    }
    // StaticParams indices are already SP-relative (resolved via SP registry)
    // All other regions use XtUP indices directly
    return xtup_index;
}

/// Resolve a name and translate to region-relative index (single var).
int resolve_for_region(PhaseRegionFlags region, const std::string &var_name,
                       const char *method) const {
    int xtup_idx = resolve_single(var_name, method);
    return to_region_index(region, xtup_idx);
}

/// Resolve names and translate to region-relative indices (multi var).
Eigen::VectorXi resolve_for_region(PhaseRegionFlags region,
                                   std::initializer_list<std::string> var_names,
                                   const char *method) const {
    auto idx = resolve(var_names);
    if (region == PhaseRegionFlags::ODEParams) {
        for (int i = 0; i < idx.size(); ++i)
            idx[i] = to_region_index(region, idx[i]);
    }
    return idx;
}
```

- [ ] **Step 3: Update existing named-variable methods to use resolve_for_region**

Replace every `resolve_single(var_name, ...)` call with `resolve_for_region(flag, var_name, ...)` in methods that pass a `PhaseRegionFlags flag`:

- `add_boundary_value` (both named overloads)
- `add_lu_var_bound`
- `add_lower_var_bound`
- `add_upper_var_bound`
- `add_value_objective`
- `add_delta_var_objective`
- `add_delta_var_equal_con`
- `add_lower_delta_var_bound`
- `add_upper_delta_var_bound`

Also update every `initializer_list<string>` method that takes a region flag — replace `resolve(var_names)` with `resolve_for_region(flag, var_names, "method_name")`:

- `add_boundary_value` (the `initializer_list<string>` overload)
- `add_value_lock`
- `add_equal_con`
- `add_inequal_con`
- `add_state_objective`
- `add_lu_func_bound`
- `add_lower_func_bound`
- `add_upper_func_bound`
- `add_lu_norm_bound`
- `add_lower_norm_bound`
- `add_upper_norm_bound`
- `add_lu_squared_norm_bound`
- `add_lower_squared_norm_bound`
- `add_upper_squared_norm_bound`

(`add_integral_objective`, `add_integral_param_function`, and `add_periodicity_con` don't take a region flag — leave them unchanged.)

**Example change (add_lu_var_bound, line 78-81):**
```cpp
// Before:
int add_lu_var_bound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                     double upper, double scale = 1.0) {
    return phase_->add_lu_var_bound(flag, resolve_single(var_name, "add_lu_var_bound"), lower,
                                    upper, scale);
}

// After:
int add_lu_var_bound(PhaseRegionFlags flag, const std::string &var_name, double lower,
                     double upper, double scale = 1.0) {
    return phase_->add_lu_var_bound(flag, resolve_for_region(flag, var_name, "add_lu_var_bound"),
                                    lower, upper, scale);
}
```

- [ ] **Step 4: Build to verify no regressions**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
./examples/cpp_examples/builder/brachistochrone/brachistochrone_builder
```
Expected: builds and prints "optimal solution found"

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/phase_wrapper.h
git commit -m "fix: add to_region_index helper and fix ODEParams name resolution

All existing named-variable methods on Phase now translate XtUP-space
indices to region-relative indices via to_region_index(). Fixes bug
where ODEParams names resolved to wrong indices."
```

---

## Task 1: Static Param Naming Infrastructure

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/phase_wrapper.h`

- [ ] **Step 1: Add SP registry member and methods to Phase**

Add to Phase's private section:
```cpp
std::unordered_map<std::string, Eigen::VectorXi> sp_names_;
```

Add to Phase's public section (after `set_static_params`):
```cpp
// ── Static param naming ────────────────────────────────────────────

void set_static_param_names(
    std::initializer_list<std::pair<std::string, int>> names) {
    sp_names_.clear();
    for (const auto &[name, idx] : names)
        add_static_param_name(name, idx);
}

void add_static_param_name(const std::string &name, int sp_index) {
    if (name.empty())
        throw std::invalid_argument("Phase::add_static_param_name: name must not be empty");
    if (sp_names_.count(name) > 0)
        throw std::invalid_argument(fmt::format(
            "Phase::add_static_param_name: '{}' already registered", name));
    Eigen::VectorXi idx(1);
    idx[0] = sp_index;
    sp_names_.emplace(name, std::move(idx));
}

void add_static_param_group(const std::string &name, int start, int count) {
    if (name.empty())
        throw std::invalid_argument("Phase::add_static_param_group: name must not be empty");
    if (sp_names_.count(name) > 0)
        throw std::invalid_argument(fmt::format(
            "Phase::add_static_param_group: '{}' already registered", name));
    if (count <= 0)
        throw std::invalid_argument(fmt::format(
            "Phase::add_static_param_group: count must be positive (got {})", count));
    Eigen::VectorXi idx(count);
    for (int i = 0; i < count; ++i)
        idx[i] = start + i;
    sp_names_.emplace(name, std::move(idx));
}
```

- [ ] **Step 2: Add SP resolution helper to Phase private section**

```cpp
Eigen::VectorXi resolve_sp(const std::string &name) const {
    auto it = sp_names_.find(name);
    if (it == sp_names_.end())
        throw std::invalid_argument(fmt::format(
            "Phase::resolve_sp: unknown static param name '{}'", name));
    return it->second;
}

int resolve_sp_single(const std::string &name, const char *method) const {
    auto idx = resolve_sp(name);
    if (idx.size() != 1)
        throw std::invalid_argument(fmt::format(
            "Phase::{}: static param '{}' maps to {} indices, expected 1",
            method, name, idx.size()));
    return idx[0];
}
```

- [ ] **Step 3: Build to verify**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/phase_wrapper.h
git commit -m "feat: add static param naming infrastructure to Phase wrapper

Adds set_static_param_names(), add_static_param_name(), and
add_static_param_group() with resolve_sp()/resolve_sp_single()
helpers. Stored as lightweight unordered_map, separate from the
ODE's VarRegistry."
```

---

## Task 2: Simple Method Passthroughs

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/phase_wrapper.h`

Add all the simple forwarding methods that don't involve complex name resolution.

- [ ] **Step 1: Add refine_traj_manual, sub_variable/sub_variables, remove_*, solve_optimize_solve**

Add to Phase's public section:

```cpp
// ── Trajectory refinement ──────────────────────────────────────────

void refine_traj_manual(int ndef) { phase_->refine_traj_manual(ndef); }
void refine_traj_manual(const Eigen::VectorXd &dbs, const Eigen::VectorXi &dpb) {
    phase_->refine_traj_manual(dbs, dpb);
}

// ── Variable substitution ──────────────────────────────────────────

void sub_variable(PhaseRegionFlags region, int var, double val) {
    phase_->sub_variable(region, var, val);
}
void sub_variables(PhaseRegionFlags region, const Eigen::VectorXi &vars,
                   const Eigen::VectorXd &vals) {
    phase_->sub_variables(region, vars, vals);
}

// Named overloads
void sub_variable(PhaseRegionFlags region, const std::string &var, double val) {
    if (region == PhaseRegionFlags::StaticParams) {
        phase_->sub_variable(region, resolve_sp_single(var, "sub_variable"), val);
    } else {
        phase_->sub_variable(region, resolve_for_region(region, var, "sub_variable"), val);
    }
}
void sub_variables(PhaseRegionFlags region, const std::vector<std::string> &vars,
                   const Eigen::VectorXd &vals) {
    Eigen::VectorXi idx(static_cast<int>(vars.size()));
    if (region == PhaseRegionFlags::StaticParams) {
        for (int i = 0; i < static_cast<int>(vars.size()); ++i)
            idx[i] = resolve_sp_single(vars[i], "sub_variables");
    } else {
        for (int i = 0; i < static_cast<int>(vars.size()); ++i)
            idx[i] = resolve_for_region(region, vars[i], "sub_variables");
    }
    phase_->sub_variables(region, idx, vals);
}

// ── Constraint/objective removal ───────────────────────────────────

void remove_state_objective(int idx) { phase_->remove_state_objective(idx); }
void remove_integral_objective(int idx) { phase_->remove_integral_objective(idx); }
void remove_equal_con(int idx) { phase_->remove_equal_con(idx); }
void remove_inequal_con(int idx) { phase_->remove_inequal_con(idx); }

// ── Additional solve sequence ──────────────────────────────────────

PSIOPT::ConvergenceFlags solve_optimize_solve() { return phase_->solve_optimize_solve(); }
```

- [ ] **Step 2: Build and test with an existing example**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 van_der_pol_builder
./examples/cpp_examples/builder/van_der_pol/van_der_pol_builder
```

- [ ] **Step 3: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/phase_wrapper.h
git commit -m "feat: add refine_traj_manual, sub_variable, remove_*, solve_optimize_solve to Phase

Forward trajectory refinement, variable substitution (index + named),
constraint/objective removal, and solve_optimize_solve through the
Phase wrapper. Named sub_variable uses SP registry for StaticParams
and resolve_for_region for all other regions."
```

---

## Task 3: Mixed Variable Source Constraints

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/phase_wrapper.h`

Add overloads for `add_equal_con`, `add_inequal_con`, and func bound methods that accept separate XtUP, ODE param, and static param variable lists.

- [ ] **Step 1: Add index-based mixed-var overloads**

```cpp
// ── Mixed variable source constraints (index-based) ────────────────

int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                  const Eigen::VectorXi &xtup_vars,
                  const Eigen::VectorXi &ode_param_vars,
                  const Eigen::VectorXi &static_param_vars,
                  ScaleType scale = ScaleModes::AUTO) {
    return phase_->add_equal_con(flag, func, VarIndexType(xtup_vars),
                                 VarIndexType(ode_param_vars),
                                 VarIndexType(static_param_vars), scale);
}

int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                    const Eigen::VectorXi &xtup_vars,
                    const Eigen::VectorXi &ode_param_vars,
                    const Eigen::VectorXi &static_param_vars,
                    ScaleType scale = ScaleModes::AUTO) {
    return phase_->add_inequal_con(flag, func, VarIndexType(xtup_vars),
                                   VarIndexType(ode_param_vars),
                                   VarIndexType(static_param_vars), scale);
}
```

For func bound methods, use `GenericFunction<-1, 1>` (scalar output) and follow
the existing wrapper's simplified scale convention:

```cpp
int add_lu_func_bound(PhaseRegionFlags flag, GenericFunction<-1, 1> func,
                      const Eigen::VectorXi &xtup_vars,
                      const Eigen::VectorXi &ode_param_vars,
                      const Eigen::VectorXi &static_param_vars,
                      double lower, double upper,
                      double scale = 1.0, ScaleType scale_t = ScaleModes::AUTO);
```

Same for `add_lower_func_bound` (no `upper`) and `add_upper_func_bound` (no `lower`).

- [ ] **Step 2: Add named mixed-var overloads**

```cpp
// ── Mixed variable source constraints (named) ──────────────────────

int add_equal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                  const std::vector<std::string> &xtup_names,
                  const std::vector<std::string> &ode_param_names,
                  const std::vector<std::string> &static_param_names,
                  ScaleType scale = ScaleModes::AUTO) {
    auto xtup_idx = resolve_names_to_indices(xtup_names, "add_equal_con");
    auto op_idx = resolve_ode_param_names(ode_param_names, "add_equal_con");
    auto sp_idx = resolve_sp_names(static_param_names, "add_equal_con");
    return phase_->add_equal_con(flag, func, VarIndexType(xtup_idx),
                                 VarIndexType(op_idx), VarIndexType(sp_idx), scale);
}

int add_inequal_con(PhaseRegionFlags flag, GenericFunction<-1, -1> func,
                    const std::vector<std::string> &xtup_names,
                    const std::vector<std::string> &ode_param_names,
                    const std::vector<std::string> &static_param_names,
                    ScaleType scale = ScaleModes::AUTO) {
    auto xtup_idx = resolve_names_to_indices(xtup_names, "add_inequal_con");
    auto op_idx = resolve_ode_param_names(ode_param_names, "add_inequal_con");
    auto sp_idx = resolve_sp_names(static_param_names, "add_inequal_con");
    return phase_->add_inequal_con(flag, func, VarIndexType(xtup_idx),
                                   VarIndexType(op_idx), VarIndexType(sp_idx), scale);
}
```

Same pattern for func bound methods — use `GenericFunction<-1, 1>` (not `-1, -1`)
and match the index-based signatures from Step 1.

- [ ] **Step 3: Add private helpers for mixed-var name resolution**

```cpp
/// Resolve XtUP var names to indices (no region translation — these are
/// always XtUP-relative for mixed-var constraints).
Eigen::VectorXi resolve_names_to_indices(const std::vector<std::string> &names,
                                         const char *method) const {
    check_registry(method);
    Eigen::VectorXi idx(static_cast<int>(names.size()));
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
        auto resolved = registry_.resolve(names[i]);
        if (resolved.size() != 1)
            throw std::invalid_argument(fmt::format(
                "Phase::{}: XtUP var '{}' maps to {} indices, expected 1",
                method, names[i], resolved.size()));
        idx[i] = resolved[0];
    }
    return idx;
}

/// Resolve ODE param names to P-relative indices.
Eigen::VectorXi resolve_ode_param_names(const std::vector<std::string> &names,
                                        const char *method) const {
    check_registry(method);
    int p_offset = registry_.xvars() + 1 + registry_.uvars();
    Eigen::VectorXi idx(static_cast<int>(names.size()));
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
        auto resolved = registry_.resolve(names[i]);
        if (resolved.size() != 1)
            throw std::invalid_argument(fmt::format(
                "Phase::{}: ODE param '{}' maps to {} indices, expected 1",
                method, names[i], resolved.size()));
        idx[i] = resolved[0] - p_offset;
        if (idx[i] < 0 || idx[i] >= registry_.pvars())
            throw std::invalid_argument(fmt::format(
                "Phase::{}: '{}' (XtUP index {}) is not in the P-block",
                method, names[i], resolved[0]));
    }
    return idx;
}

/// Resolve static param names to SP-relative indices.
Eigen::VectorXi resolve_sp_names(const std::vector<std::string> &names,
                                 const char *method) const {
    Eigen::VectorXi idx(static_cast<int>(names.size()));
    for (int i = 0; i < static_cast<int>(names.size()); ++i)
        idx[i] = resolve_sp_single(names[i], method);
    return idx;
}
```

- [ ] **Step 4: Build to verify**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/phase_wrapper.h
git commit -m "feat: add mixed variable source constraint overloads to Phase

Add index-based and named overloads for add_equal_con, add_inequal_con,
add_lu_func_bound, add_lower_func_bound, add_upper_func_bound that
accept separate XtUP, ODE param, and static param variable lists."
```

---

## Task 4: LerpIG Flag on ODE::phase() and Phase::set_traj()

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/runtime_ode.h`
- Modify: `src/optimal_control/runtime_ode.cpp`
- Modify: `include/tycho/detail/optimal_control/builder/phase_wrapper.h`

Two changes: (a) add `lerp_ig` parameter to `ODE::phase()` for one-shot
construction, and (b) add a `set_traj` overload with `lerp_ig` to the Phase
wrapper for cases where the trajectory is re-set after construction (e.g.,
ParallelParking constructs with a dummy IG then calls `set_traj` with real
sparse waypoints).

- [ ] **Step 1: Read current runtime_ode.cpp implementation**

Read `src/optimal_control/runtime_ode.cpp` to understand the current `ODE::phase()` implementation.

- [ ] **Step 2: Add lerp_ig parameter to ODE::phase() declaration**

In `runtime_ode.h`, change the `phase()` declaration:

```cpp
// Before:
Phase phase(TranscriptionModes mode, const std::vector<Eigen::VectorXd> &traj,
            int num_segments) const;

// After:
Phase phase(TranscriptionModes mode, const std::vector<Eigen::VectorXd> &traj,
            int num_segments, bool lerp_ig = false) const;
```

- [ ] **Step 3: Update ODE::phase() implementation to pass lerp_ig**

In `runtime_ode.cpp`, update the `ODE::phase()` implementation to use the
`set_traj` overload with `LerpTraj` when `lerp_ig` is true. Read the file
first to understand the current impl, then modify.

The key change: when `lerp_ig` is true, call
`phase_base->set_traj(traj, num_segments, true)` instead of the default path.

- [ ] **Step 4: Add set_traj lerp overload to Phase wrapper**

In `phase_wrapper.h`, add alongside existing `set_traj` overloads:

```cpp
void set_traj(const std::vector<Eigen::VectorXd> &traj, int ndef, bool lerp_ig) {
    phase_->set_traj(traj, ndef, lerp_ig);
}
```

- [ ] **Step 5: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 6: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/runtime_ode.h \
        src/optimal_control/runtime_ode.cpp \
        include/tycho/detail/optimal_control/builder/phase_wrapper.h
git commit -m "feat: add lerp_ig to ODE::phase() and Phase::set_traj()

ODE::phase(mode, traj, nsegs, lerp_ig=true) interpolates sparse
waypoints during construction. Phase::set_traj(traj, ndef, lerp_ig)
does the same for post-construction trajectory replacement."
```

---

## Task 5: OCP add_direct_link_equal_con

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/ocp_wrapper.h`

- [ ] **Step 1: Add Phase reference storage to OCP wrapper**

The wrapper needs to store Phase references to resolve names for direct link constraints. Add a private member:

```cpp
std::vector<Phase *> phases_;
```

Update `add_phase()` to store:
```cpp
int add_phase(Phase &p) {
    phases_.push_back(&p);
    return ocp_.add_phase(p.base_ptr());
}
int add_phase(Phase &p, const std::string &name) {
    phases_.push_back(&p);
    return ocp_.add_phase(p.base_ptr(), name);
}
```

- [ ] **Step 2: Add helper to resolve phase index from Phase reference**

```cpp
private:
    int find_phase_index(Phase &p) const {
        for (int i = 0; i < static_cast<int>(phases_.size()); ++i) {
            if (phases_[i] == &p)
                return i;
        }
        throw std::invalid_argument(
            "OptimalControlProblem::add_direct_link_equal_con: phase not found "
            "— was it added via add_phase()?");
    }
```

- [ ] **Step 3: Add add_direct_link_equal_con overloads**

```cpp
// ── Direct link constraints ────────────────────────────────────────

// Index-based (phase indices)
int add_direct_link_equal_con(int phase_a, PhaseRegionFlags region_a,
                              const Eigen::VectorXi &vars_a,
                              int phase_b, PhaseRegionFlags region_b,
                              const Eigen::VectorXi &vars_b) {
    return ocp_.add_direct_link_equal_con(phase_a, region_a, vars_a,
                                          phase_b, region_b, vars_b);
}

// Index-based (Phase references)
int add_direct_link_equal_con(Phase &phase_a, PhaseRegionFlags region_a,
                              const Eigen::VectorXi &vars_a,
                              Phase &phase_b, PhaseRegionFlags region_b,
                              const Eigen::VectorXi &vars_b) {
    return ocp_.add_direct_link_equal_con(
        phase_a.base_ptr(), region_a, vars_a,
        phase_b.base_ptr(), region_b, vars_b);
}

// Named (phase indices) — resolves names via stored Phase references
int add_direct_link_equal_con(int phase_a, PhaseRegionFlags region_a,
                              const std::vector<std::string> &vars_a,
                              int phase_b, PhaseRegionFlags region_b,
                              const std::vector<std::string> &vars_b) {
    auto idx_a = resolve_link_vars(*phases_.at(phase_a), region_a, vars_a);
    auto idx_b = resolve_link_vars(*phases_.at(phase_b), region_b, vars_b);
    return ocp_.add_direct_link_equal_con(phase_a, region_a, idx_a,
                                          phase_b, region_b, idx_b);
}

// Named (Phase references)
int add_direct_link_equal_con(Phase &phase_a, PhaseRegionFlags region_a,
                              const std::vector<std::string> &vars_a,
                              Phase &phase_b, PhaseRegionFlags region_b,
                              const std::vector<std::string> &vars_b) {
    auto idx_a = resolve_link_vars(phase_a, region_a, vars_a);
    auto idx_b = resolve_link_vars(phase_b, region_b, vars_b);
    return ocp_.add_direct_link_equal_con(
        phase_a.base_ptr(), region_a, idx_a,
        phase_b.base_ptr(), region_b, idx_b);
}
```

- [ ] **Step 4: Add private resolve_link_vars helper**

```cpp
/// Resolve variable names for a link constraint, respecting region.
/// ODEParams names get P-relative translation, StaticParams use SP registry.
Eigen::VectorXi resolve_link_vars(Phase &p, PhaseRegionFlags region,
                                  const std::vector<std::string> &names) const {
    Eigen::VectorXi idx(static_cast<int>(names.size()));
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
        if (region == PhaseRegionFlags::StaticParams) {
            idx[i] = p.resolve_sp_single(names[i], "add_direct_link_equal_con");
        } else {
            idx[i] = p.resolve_for_region(region, names[i], "add_direct_link_equal_con");
        }
    }
    return idx;
}
```

**Note:** This requires `resolve_sp_single` and `resolve_for_region` to be accessible from the OCP wrapper. Since the OCP wrapper is a `friend` or in the same namespace, and these are currently private on Phase, they either need to become public or the OCP wrapper needs to be declared a friend. Making them public is fine — they're utility methods, not implementation secrets.

- [ ] **Step 5: Make resolve_for_region and resolve_sp_single public on Phase**

Move `resolve_for_region` and `resolve_sp_single` from Phase's private section to a new public section labeled `// ── Name resolution (public for OCP wrapper) ──`.

- [ ] **Step 6: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 delta3_launch_builder
./examples/cpp_examples/builder/delta3_launch/delta3_launch_builder
```

- [ ] **Step 7: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/ocp_wrapper.h \
        include/tycho/detail/optimal_control/builder/phase_wrapper.h
git commit -m "feat: add add_direct_link_equal_con to OCP wrapper

Four overloads: index/named × int-phase/Phase-ref. OCP wrapper
stores Phase pointers for name resolution. Phase exposes
resolve_for_region and resolve_sp_single publicly for OCP use."
```

---

## Task 6: Update Examples to Use New API

**Files:**
- Modify: 7 example main.cpp files

Update all examples that currently use `phase.base()` or `ocp.base()` to use the new wrapper methods. Verify each compiles and passes.

- [ ] **Step 1: Audit all examples for base() calls**

```bash
grep -rn "\.base()\." examples/cpp_examples/builder/*/main.cpp
```

This will find every `phase.base().method()` and `ocp.base().method()` call.

- [ ] **Step 2: Update ParallelParking**

Replace `phase.base().refine_traj_manual()`, `phase.base().sub_variable()`, and `phase.base().add_inequal_con(...)` with wrapper methods. Add `phase.set_static_param_names(...)` for the smoothing constant `k`.

Build and run:
```bash
ninja -j8 parallel_parking_builder
./examples/cpp_examples/builder/parallel_parking/parallel_parking_builder
```

- [ ] **Step 3: Update Reentry**

Replace `phase.base().refine_traj_manual()` with `phase.refine_traj_manual()`.

Build and run:
```bash
ninja -j8 reentry_builder
./examples/cpp_examples/builder/reentry/reentry_builder
```

- [ ] **Step 4: Update MultiPhaseCannon**

Replace `ocp.base().add_direct_link_equal_con()` with wrapper method. Replace mixed-var `phase.base().add_inequal_con()` calls.

Build and run:
```bash
ninja -j8 multi_phase_cannon_builder
./examples/cpp_examples/builder/multi_phase_cannon/multi_phase_cannon_builder
```

- [ ] **Step 5: Update SimpleLowThrust**

Replace `phase.base().remove_state_objective()` and `phase.base().remove_integral_objective()` with wrapper methods.

Build and run:
```bash
ninja -j8 simple_low_thrust_builder
./examples/cpp_examples/builder/simple_low_thrust/simple_low_thrust_builder
```

- [ ] **Step 6: Update TopputtoLowThrust**

Replace `phase.base().solve_optimize_solve()`, `phase.base().remove_state_objective()`, `phase.base().refine_traj_manual()` with wrapper methods.

Build and run:
```bash
ninja -j8 topputto_low_thrust_builder
./examples/cpp_examples/builder/topputto_low_thrust/topputto_low_thrust_builder
```

- [ ] **Step 7: Update BettsLowThrust**

The ODEParams name resolution fix (Task 0) should now make named `add_lu_var_bound` work correctly with `ODEParams` region. Update if the example currently uses index workaround.

Build and run:
```bash
ninja -j8 betts_low_thrust_builder
./examples/cpp_examples/builder/betts_low_thrust/betts_low_thrust_builder
```

- [ ] **Step 8: Update MultiSpacecraftOpt**

Replace `phase.base().sub_variable()` / `sub_variables()` with wrapper methods. Note: OCP link param calls (`set_link_params`, `add_link_equal_con` PhasePack form) are out of scope — those `ocp.base()` calls remain.

Build and run:
```bash
ninja -j8 multi_spacecraft_opt_builder
./examples/cpp_examples/builder/multi_spacecraft_opt/multi_spacecraft_opt_builder
```

- [ ] **Step 9: Final audit — verify no unexpected base() calls remain**

```bash
grep -rn "\.base()\." examples/cpp_examples/builder/*/main.cpp
```

Expected: only MultiSpacecraftOpt OCP link param calls remain (explicitly out of scope).

- [ ] **Step 10: Commit all example updates**

```bash
git add examples/cpp_examples/builder/*/main.cpp
git commit -m "refactor: update examples to use new Phase/OCP wrapper methods

Remove phase.base() and ocp.base() workarounds from ParallelParking,
Reentry, MultiPhaseCannon, SimpleLowThrust, TopputtoLowThrust,
BettsLowThrust, and MultiSpacecraftOpt. Only OCP link param calls
in MultiSpacecraftOpt remain (out of scope for this spec)."
```

---

## Task 7: Full CTest Verification

- [ ] **Step 1: Build all examples**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j2 cpp_examples_all
```

- [ ] **Step 2: Run CTest**

```bash
ctest -R cpp_example --output-on-failure -j1
```

Expected: 31/31 pass.

- [ ] **Step 3: Run existing C++ tests if configured**

```bash
ctest --output-on-failure -j1
```

- [ ] **Step 4: Commit any remaining fixes**

If any tests failed, fix and commit.
