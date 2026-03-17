# Phase 9: Future Work Backlog

> This is a **living document** — updated as phases 1-8 progress.

**Purpose:** Track follow-up work identified during implementation. Items are
prioritized into tiers but not scheduled.

---

## Tier 1: Near-Term (natural follow-ups from phases 1-8)

### BettsLowThrust C++ Example
- The "boss fight" integration test from TODO.md
- 7 states, 3 controls, 1 parameter — MEE dynamics with J2/J3/J4
- Requires control-law integration (see below) for initial guess generation
- Candidate for both static DSL (with firebreaks) and builder API versions
- Reference: `examples/UpdatedInterface/BettsLowThrust.py`

### Control-Law Integration in C++
- Python: `ode.integrator(step, control_law, var_group)`
- C++: extend `Integrator<ODE>` to accept a control law `GenericFunction`
- Needed for BettsLT and any problem where linear interpolation won't converge
- Builder API should wrap this with named variable support

### C++ OptimalControlProblem Builder
- Named-phase builder for multi-phase problems
- `tycho::OCPBuilder` that manages named phases and link constraints
- Simplifies the 4-phase Delta3Launch pattern significantly

### Additional C++ Examples
- CR3BP / DeltaV_CR3BP — tests astro module from C++
- MultiPhaseZermelo — simpler multi-phase problem
- SimpleLowThrust — mid-complexity astro problem

---

## Tier 2: Medium-Term

### Documentation
- Sphinx/Doxygen for the C++ API
- Public header separation (Phase 2) makes this straightforward
- Tutorial-style docs for both static DSL and builder API

### CI/CD
- Automated testing of C++ examples on PR
- Benchmark regression detection
- Cross-platform builds (Linux, Windows)

### PyPI Packaging for tychopy
- Publish the Python package
- scikit-build-core based wheels
- Update `pyproject.toml` with proper metadata

### Benchmarking Builder vs. Static DSL
- Quantify runtime performance gap
- Measure: function evaluation, Jacobian computation, full phase solve
- Help users make informed API choice

---

## Tier 3: Longer-Term

### Package Manager Distribution
- vcpkg port
- Conan recipe
- Depends on stable API and versioning

### Julia Bindings (Tycho.jl)
- CxxWrap.jl or direct C API
- Separate repository (Tycho.jl)

### Expression Tree Simplification
- Beyond `Scaled<Scaled>` fix: general compile-time optimization passes
- Constant folding, common subexpression elimination
- Could dramatically reduce template depth for complex ODEs

### Explicit Jacobian ODEs
- Allow hand-coded Jacobians and Hessians
- Bypasses automatic differentiation entirely
- Common practice for production astrodynamics codes
- Big ask for users since Tycho requires second derivatives

### Static DSL Further Improvements
- Better error messages for template failures
- Compile-time expression validation
- Integration of C++20 concepts for clearer type constraints

---

## Pain Points Discovered During Implementation

_(This section populated as phases 4-8 progress)_

<!--
Template:
### Pain Point: [Description]
- **Discovered in:** Phase X, Task Y
- **Severity:** High/Medium/Low
- **Workaround:** [if any]
- **Proposed fix:** [if known]
-->
