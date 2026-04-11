# Spec: Domain Models (Subsystem 6)

> Modernize the SymPy codegen infrastructure, generate a
> `MEEToCartesian` VectorFunction with analytic derivatives, add
> `set_jet_job_mode()` to Phase wrapper, and update HangingChain and
> BettsLowThrust examples for full Python parity.

## Motivation

Three remaining gaps prevent 1-to-1 Python parity:

1. **Jet.map() parallel solve** (Medium) — HangingChain uses
   `Jet.map()` for a 100-point parameter sweep. The C++ example solves
   one configuration. The `Jet` class is already in C++ headers but the
   Phase wrapper lacks `set_jet_job_mode()`.

2. **BettsLowThrust J2/J3/J4** (High) — The Python version uses full
   zonal gravity via MEE→Cartesian→gravity→RTN composition. The C++
   version is two-body only. Subsystem 3 added `RowMatrix().inverse()`,
   enabling the RTN rotation. The remaining piece is a performant
   MEE-to-Cartesian VectorFunction with analytic derivatives.

3. **Codegen infrastructure** — The existing `utils/CodeGen.py` SymPy
   framework generates C++ VectorFunctions with CSE-optimized analytic
   derivatives (compute_impl, jacobian, adjoint hessian). It produced
   `MEEDynamics` in `mee_dynamics.h`. However, it needs modernization:
   the output still uses the old `ASSET` namespace, pybind11 binding
   syntax, and a stale include path. Updating it now ensures future
   orbital conversions (Cartesian→MEE, etc.) can be generated easily.

## Scope

This spec covers:
- Modernize `utils/CodeGen.py` for tycho namespace and includes
- New SymPy script for MEE→Cartesian conversion
- Generated `MEEToCartesian` VF header with analytic derivatives
- `set_jet_job_mode()` on Phase wrapper
- HangingChain example with `Jet::map()` parameter sweep
- BettsLowThrust with full J2/J3/J4 zonal gravity using the generated
  `MEEToCartesian` VF

This spec does NOT cover:
- Other orbital conversions (Cartesian→MEE, Classical↔Cartesian) — future
  work using the same codegen infrastructure
- MEE ODE factory classes (YAGNI — examples work with inline dynamics)
- Full astro model library redesign

## Changes

### 1. Modernize CodeGen.py

Update `utils/CodeGen.py` to generate code compatible with the current
tycho codebase:

**Namespace:** Change from `namespace ASSET {` to configurable namespace
(default `tycho::astro`).

**Include:** Change from `#include "ASSET_VectorFunctions.h"` to
`#include "tycho/detail/vf/core/vector_function.h"` (or a configurable
include path).

**Header guard:** Add `#pragma once` to generated output.

**Build method:** The generated `Build()` static method uses pybind11
syntax (`py::module`). Either update to nanobind or make generation of
the build method optional (since C++ VFs don't need Python bindings
in-header — those are added separately in `src/bindings/`).

**Output path:** `make_header()` currently writes to CWD. Add an optional
output path parameter.

**Core generation logic is unchanged:** The Jacobian, adjoint gradient,
and adjoint Hessian computation via SymPy is correct and battle-tested
(produced `MEEDynamics`). CSE optimization and the `fixup()` method for
scalar parameter wrapping work correctly.

### 2. MEE→Cartesian SymPy Script

New file: `utils/codegen_mee_to_cartesian.py`

Defines the MEE→Cartesian conversion symbolically:

**Input:** 6 symbols `[p, f, g, h, k, L]` (Modified Equinoctial Elements)
**Parameter:** `mu` (gravitational parameter)
**Output:** 6 expressions `[x, y, z, vx, vy, vz]` (Cartesian state)

The formulas (matching `kepler_utils.h` lines 183-214 and
`BettsLowThrust.py` `MEECartFunc`):

```python
sinL, cosL = sp.sin(L), sp.cos(L)
w = 1 + f*cosL + g*sinL
s2 = 1 + h**2 + k**2
alpha2 = h**2 - k**2
r = p / w

# Position
x = (r / s2) * (cosL + alpha2*cosL + 2*h*k*sinL)
y = (r / s2) * (sinL - alpha2*sinL + 2*h*k*cosL)
z = (2*r / s2) * (h*sinL - k*cosL)

# Velocity
sqpm = sp.sqrt(mu / p)
vx = -(sqpm / s2) * (sinL + alpha2*sinL - 2*h*k*cosL + g - 2*f*h*k + alpha2*g)
vy = -(sqpm / s2) * (-cosL + alpha2*cosL + 2*h*k*sinL - f + 2*g*h*k + alpha2*f)
vz = (2*sqpm / s2) * (h*cosL + k*sinL + f*h + g*k)
```

Run through `AssetHeaderGen` (updated) to produce the C++ header with
analytic Jacobian and adjoint Hessian.

### 3. Generated MEEToCartesian VF Header

Output: `include/tycho/detail/astro/mee_to_cartesian.h`

Generated struct:
```cpp
struct MEEToCartesian : VectorFunction<MEEToCartesian, 6, 6,
    DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic> {
    double mu;
    MEEToCartesian(double mu) : mu(mu) { this->set_io_rows(6, 6); }

    // SymPy-generated, CSE-optimized:
    compute_impl(...)           // function evaluation
    compute_jacobian_impl(...)  // function + 6×6 Jacobian
    compute_jacobian_adjointgradient_adjointhessian_impl(...)  // full adjoint
};
```

This replaces the expression-tree-based `ModifiedToCartesian` (in
`kepler_utils.h`) for cases where performance matters (e.g., inside
an ODE that's evaluated millions of times during optimization). The
expression-tree version remains available for convenience.

### 4. Phase Wrapper: set_jet_job_mode()

Add to `include/tycho/detail/optimal_control/builder/phase_wrapper.h`:

```cpp
void set_jet_job_mode(JetJobModes mode) {
    phase_->jet_job_mode_ = mode;
}
```

One-liner forwarding. `JetJobModes` is defined in
`optimization_problem_base.h`.

### 5. HangingChain: Jet::map() Parameter Sweep

Update to sweep over 100 chain lengths (L = 2.25 to 8.0):

```cpp
std::vector<std::shared_ptr<OptimizationProblemBase>> jobs;
for (double L : Ls) {
    auto phase = make_chain_phase(ode, a, b, L, n_segs);
    phase.set_jet_job_mode(JetJobModes::SolveOptimize);
    jobs.push_back(phase.base_ptr());
}
auto results = solvers::Jet::map(jobs, true);
```

### 6. BettsLowThrust: Full J2/J3/J4 Zonal Gravity

Update BettsLowThrust ODE with the full MEE + zonal gravity dynamics
chain matching Python's `LTModel`:

1. **MEE state** → **Cartesian** via the generated `MEEToCartesian` VF
   (composed with state extraction via `.eval()`)
2. **Zonal gravity** (J2/J3/J4) computed from Cartesian position
   (inline VF expressions for gravitational potential derivatives)
3. **Cartesian→RTN rotation** via `RowMatrix(basis, 3, 3).inverse()`
   to transform gravity accelerations to RTN frame
4. **MEE rate equations** with combined thrust + gravity RTN
   accelerations (using the existing `MEEDynamics` VF or inline Gauss
   variational equations)

The composition chain:
```cpp
auto mee_to_cart = MEEToCartesian(mu);
auto cart_state = GenericFunction<-1, -1>(mee_to_cart.eval(mee_elements));
auto r_vec = cart_state.head<3>();
auto v_vec = cart_state.tail<3>();
auto grav_accel = zonal_gravity(r_vec, mu, Re, J2, J3, J4);  // inline VF
auto rtn_basis = build_rtn_basis(r_vec, v_vec);               // inline VF
auto rtn_matrix = RowMatrix(rtn_basis, 3, 3);
auto grav_rtn = rtn_matrix * grav_accel;  // matrix-vector product
```

Success criteria: converges with at least J2, trajectory closer to
Python reference than current two-body version.

## Verification

- **HangingChain:** 100 configurations solved via Jet::map(), PASSED
- **BettsLowThrust:** Converges with zonal gravity, PASSED
- All CTests pass (no regressions)

## Files Modified

**New:**
- `utils/codegen_mee_to_cartesian.py` — SymPy script for MEE→Cartesian
- `include/tycho/detail/astro/mee_to_cartesian.h` — generated VF

**Modified:**
- `utils/CodeGen.py` — modernized for tycho namespace/includes
- `include/tycho/detail/optimal_control/builder/phase_wrapper.h` — add
  `set_jet_job_mode()`
- `examples/cpp_examples/builder/hanging_chain/main.cpp` — Jet::map()
  parameter sweep
- `examples/cpp_examples/builder/betts_low_thrust/main.cpp` — full
  J2/J3/J4 zonal gravity with generated MEEToCartesian VF
