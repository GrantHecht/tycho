# Cartesian & CRTBP Dynamics via Codegen — Design

**Date:** 2026-04-14
**Status:** Approved for implementation
**Author:** brainstorming session

## Summary

Add two new codegen-generated astrodynamics `VectorFunction`s — `CartesianDynamics`
and `CRTBPDynamics` — modeled after the existing `MEEDynamics` / `MEEToCartesian`
pattern (SymPy → `TychoHeaderGen` → analytic-derivative C++ header). Both accept
extra acceleration terms as inputs (matching `MEEDynamics`'s convention), so the
same dynamics class can be used for uncontrolled propagation, low-thrust optimal
control, or arbitrary perturbation models.

The existing hand-written `cr3bp_model.h` (a `StaticODE_Expression`-based ODE) is
removed entirely. Four C++ builder examples are converted to the new dynamics,
three of which currently build inline expression-template ODEs. Tests and
benchmarks that referenced the old `CR3BP` type are ported. The `extensions/`
subdirectory — whose `Tycho_Extensions.h` includes `cr3bp_model.h` — is
unbuilt (not deleted) to unblock header removal; a proper decision about
the extension mechanism is deferred to a future PR.

Python examples (`OrbitContinuation.py`, `Heteroclinic.py`) are unaffected:
`tychopy.Astro.AstroModels.CR3BP` is a pure-Python class that builds its own
EOMs via `CR3BPFrame.CR3BPEOMs` and does not depend on `cr3bp_model.h`.

## Motivation

- **Speed.** The codegen path produces analytic Jacobians and adjoint
  Hessians with CSE applied offline. Measurement on `MEEDynamics` vs. the
  equivalent expression-template ODE suggests this is meaningfully faster
  for hot-loop phase evaluation. Converting `multi_spacecraft_opt` — which
  has no control but uses inline Kepler dynamics — is motivated primarily
  by this speedup.
- **Uniformity.** The existing codebase has three distinct styles for defining
  astro dynamics (hand-written `StaticODE_Expression` like `Kepler` / `CR3BP`,
  inline expression-template builders in each example, and codegen VFs like
  `MEEDynamics`). Consolidating the Cartesian and CRTBP cases onto the codegen
  style matches MEE and reduces per-example duplication.
- **Extra-acceleration composability.** Following the `MEEDynamics`
  `[state, accel]` input convention makes low-thrust / perturbation models
  trivially expressible as `dyn.eval(stack(state, accel))`.
- **Dead code removal.** `cr3bp_model.h` has a single nontrivial consumer
  (the tests + benchmark), a second consumer that's about to be unbuilt
  (extensions), and an orphan `#include` in the astro bindings. Replacing
  it with a generated VF that all four builder examples can share is a
  straightforward simplification.

## Non-Goals

- Replacing `Kepler` / `kepler_model.h`. Kepler solves a different problem
  (Kepler propagator integration, analytic two-body phase construction with
  the `use_kepler_propagator_` fast path) and is retained as-is.
- Touching the Python `tychopy.Astro.AstroModels.CR3BP` family. That class
  hierarchy is pure Python and independent of the C++ type.
- Redesigning the extension mechanism. This PR only unbuilds `extensions/`
  as a side-effect of header removal; the broader question of whether and
  how extensions should exist is explicitly deferred.
- Adding new Python bindings beyond the existing `modified_dynamics` lambda
  pattern. Two analogous `mod.def` entries are sufficient.

## Scope

### In scope

1. Two new codegen scripts under `utils/`.
2. Two new generated headers under `include/tycho/detail/astro/`.
3. Deletion of `include/tycho/detail/astro/cr3bp_model.h`.
4. Umbrella header update (`include/tycho/astro.h`).
5. Two new Python bindings in `src/bindings/astro/tycho_astro.cpp`.
6. Conversion of four C++ builder examples.
7. Port of `tests/cpp/astro/test_cr3bp.cpp`.
8. Port of CR3BP cases in `tests/cpp/vector_functions/test_vf_ode_expressions.cpp`.
9. Port of `BM_VF_CR3BP_*` benchmarks in `bench/cpp/vector_functions/bench_vector_functions.cpp`.
10. Comment out / remove `add_subdirectory(extensions)` in root `CMakeLists.txt`.

### Out of scope / deferred

- Rework or deletion of the `extensions/` subdirectory contents.
- Rework of the Python `tychopy.Astro.AstroModels.CR3BP` family.
- Removal of `kepler_model.h` or replacement of `Kepler`.
- Any PyPI / packaging / Sphinx documentation changes beyond source-level
  docstrings in the new generated headers.

## Design

### 1. Codegen scripts

Both scripts follow the exact structure of `utils/codegen_mee_dynamics.py`:
import SymPy, define input symbols, write SymPy expressions for the output
vector, construct a `TychoHeaderGen` with the parameter list, and call
`make_header`. Output is a single `.h` file in
`include/tycho/detail/astro/`.

#### `utils/codegen_cartesian_dynamics.py`

**Generates:** `include/tycho/detail/astro/cartesian_dynamics.h`, containing
```cpp
struct CartesianDynamics
    : VectorFunction<CartesianDynamics, 9, 6,
                     DenseDerivativeMode::Analytic,
                     DenseDerivativeMode::Analytic>;
```

**Inputs:** `[x, y, z, vx, vy, vz, ax, ay, az]` (Cartesian state + 3
extra acceleration components in the inertial frame).

**Outputs:** `[xdot, ydot, zdot, vxdot, vydot, vzdot]` where
- `xdot = vx`, `ydot = vy`, `zdot = vz`
- `vxdot = -mu * x / r^3 + ax`
- `vydot = -mu * y / r^3 + ay`
- `vzdot = -mu * z / r^3 + az`
- `r = sqrt(x^2 + y^2 + z^2)`

**Parameter:** `mu` (gravitational parameter).

**Notes:** CSE will hoist `r^3` and any `mu`-only subexpressions to precomputed
members where applicable. No `mu`-only subexpressions exist for Cartesian
two-body, but the codegen framework will still produce `pc0_`-style members
if CSE finds any. The header should compile to ~100-200 lines of generated code.

#### `utils/codegen_crtbp_dynamics.py`

**Generates:** `include/tycho/detail/astro/crtbp_dynamics.h`, containing
```cpp
struct CRTBPDynamics
    : VectorFunction<CRTBPDynamics, 9, 6,
                     DenseDerivativeMode::Analytic,
                     DenseDerivativeMode::Analytic>;
```

**Inputs:** `[x, y, z, vx, vy, vz, ax, ay, az]` (rotating-frame state + 3
extra acceleration components in the rotating frame).

**Outputs:** `[xdot, ydot, zdot, vxdot, vydot, vzdot]` where

- `xdot = vx`, `ydot = vy`, `zdot = vz`
- Let `d = ((x + mu)^2 + y^2 + z^2)^(1/2)` (distance to primary at `(-mu, 0, 0)`)
- Let `r = ((x - (1 - mu))^2 + y^2 + z^2)^(1/2)` (distance to secondary at `(1 - mu, 0, 0)`)
- `vxdot = 2*vy + x - (1 - mu) * (x + mu) / d^3 - mu * (x - (1 - mu)) / r^3 + ax`
- `vydot = -2*vx + y - (1 - mu) * y / d^3 - mu * y / r^3 + ay`
- `vzdot = -(1 - mu) * z / d^3 - mu * z / r^3 + az`

**Parameter:** `mu` (CR3BP mass ratio, `m2 / (m1 + m2)`).

**Notes:** CSE will hoist `(1 - mu)` and the two inverse-cube subexpressions to
precomputed / local scope respectively. The generated analytic Jacobian and
adjoint Hessian should be substantially faster than the expression-template
equivalent currently used by `orbit_continuation` and `heteroclinic` — this
is the primary motivation for the conversion.

**Physics parity check:** The expressions above match the hand-written
`CR3BP_Impl::Definition` in `cr3bp_model.h` and the inline builders in
`orbit_continuation/main.cpp` and `heteroclinic/main.cpp`. Unit test
parity with the pre-existing `cr3bp_accel_partials` harness in
`test_vf_ode_expressions.cpp` confirms correctness.

### 2. Umbrella header and bindings

**`include/tycho/astro.h`:** Remove `#include "tycho/detail/astro/cr3bp_model.h"`.
Add:
```cpp
#include "tycho/detail/astro/cartesian_dynamics.h"
#include "tycho/detail/astro/crtbp_dynamics.h"
```

**Delete** `include/tycho/detail/astro/cr3bp_model.h`.

**`src/bindings/astro/tycho_astro.cpp`:** Remove the orphan
`#include "tycho/detail/astro/cr3bp_model.h"` at line 17. Add two bindings
inside `AstroBuild` alongside the existing `modified_dynamics`:

```cpp
mod.def("cartesian_dynamics",
        [](double mu) {
            return GenericFunction<-1, -1>(CartesianDynamics(mu));
        });

mod.def("crtbp_dynamics",
        [](double mu) {
            return GenericFunction<-1, -1>(CRTBPDynamics(mu));
        });
```

No full `TychoBind<>` specialization is needed — the lambda pattern is
consistent with `modified_dynamics` and sufficient for the downstream
pure-Python consumers to stack accelerations and build ODEs.

### 3. C++ builder example conversions

All four examples adopt the pattern already used by `betts_low_thrust` and
`dionysus_low_thrust`:

```cpp
auto dyn = astro::XDynamics(mu);
auto Xdot = GenericFunction<-1, -1>(dyn.eval(stack(state_args, accel_expr)));
auto ode = ODE(Xdot, nstate, ncontrol, nparam).var_group(...);
```

#### `examples/cpp_examples/builder/simple_low_thrust/main.cpp`

Current inline form (main.cpp:36):
```cpp
auto grav = (-mu) * R.normalized_power<3>();
auto acc = grav + thrust;
```

New form:
```cpp
auto dyn = astro::CartesianDynamics(mu);
auto Xdot = GenericFunction<-1, -1>(dyn.eval(stack(state, thrust_acc)));
```

The existing `var_group` / `var_names` / boundary-condition scaffolding is
unchanged. This is the cleanest conversion of the four.

#### `examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp`

Current inline form (main.cpp:19): inline Kepler with no extra acceleration.

New form: stack a compile-time zero 3-vector onto the 6-D state argument
using `Constant<6, 3>`:

```cpp
auto args = Arguments<6>();
Eigen::Vector3d zero_accel = Eigen::Vector3d::Zero();
auto zero_const = Constant<6, 3>(6, zero_accel);
auto dyn = astro::CartesianDynamics(mu);
auto Xdot = GenericFunction<-1, -1>(dyn.eval(stack(args, zero_const)));
```

Slightly more verbose than the original inline form but consistent, and
the codegen'd analytic derivatives are the performance win.

#### `examples/cpp_examples/builder/orbit_continuation/main.cpp`

Delete the bespoke `make_cr3bp_ode(double mu_val)` function entirely
(lines 17-52). Replace with an inline `make_cr3bp_ode` that uses
`CRTBPDynamics` stacked with a zero accel, preserving the same `var_group` /
`var_names` configuration. The signature of the rest of the example
(`solve_periodic`, `contin`, `main`) is unchanged.

#### `examples/cpp_examples/builder/heteroclinic/main.cpp`

Same treatment as `orbit_continuation`. Delete the inline
`make_cr3bp_ode()` (lines 106-136) and replace with a `CRTBPDynamics`-based
version. The rest of the 470-line example is unchanged.

### 4. Tests, benchmarks, extensions

#### `tests/cpp/astro/test_cr3bp.cpp` (63 lines)

Currently tests `CR3BP(mu)` for construction, input/output row counts, and
adjoint consistency at a libration point. Port to `CRTBPDynamics`:

- Change construction to `CRTBPDynamics dyn(mu)`.
- Update `input_rows()` expectation from 7 to 9 (added 3 accel inputs).
- Update `output_rows()` stays 6.
- Input vector gains three trailing zeros (no extra accel) in the
  consistency check.
- Keep the `cr3bp_accel_partials` numerical harness — it's a reference
  implementation of the same physics and validates output values directly.
- Optionally rename to `test_crtbp_dynamics.cpp` and update the GTest
  fixture names accordingly.

#### `tests/cpp/vector_functions/test_vf_ode_expressions.cpp`

The `CR3BPODEAdjointConsistency` test (line 61) uses `CR3BP cr3bp(mu)`.
Port it to `CRTBPDynamics` with the same input-shape updates as above.
Rename to `CRTBPDynamicsAdjointConsistency`.

#### `bench/cpp/vector_functions/bench_vector_functions.cpp`

Two benchmarks at lines 131-161: `BM_VF_CR3BP_Compute` and
`BM_VF_CR3BP_ComputeJacobian`. Port to `CRTBPDynamics` and rename to
`BM_VF_CRTBPDynamics_Compute` / `BM_VF_CRTBPDynamics_ComputeJacobian`.
Record a fresh baseline in the PR description; numbers are not expected to
match the old expression-template benchmark point-for-point since the new
class has analytic derivatives from CSE'd SymPy.

Remove the now-unused `#include <tycho/detail/astro/cr3bp_model.h>` at line 6.

#### Extensions

Comment out or remove `add_subdirectory(extensions)` at root
`CMakeLists.txt:688`. Leave all files under `extensions/` untouched on disk
— the files remain version-controlled and can be re-enabled by uncommenting
the line. This is explicitly framed in the commit message and design doc
as "unbuilt to unblock `cr3bp_model.h` removal; decision on the extension
mechanism deferred to a follow-up PR."

### 5. Python examples

No changes. `tychopy.Astro.AstroModels.CR3BP` builds its own EOMs in Python
via `CR3BPFrame.CR3BPEOMs` and does not depend on the C++ `CR3BP` type or
`cr3bp_model.h`. `OrbitContinuation.py` and `Heteroclinic.py` continue to
work unchanged.

## Interfaces

### Generated C++ API

```cpp
namespace tycho::astro {

struct CartesianDynamics
    : VectorFunction<CartesianDynamics, 9, 6,
                     DenseDerivativeMode::Analytic,
                     DenseDerivativeMode::Analytic> {
    double mu_;
    CartesianDynamics();
    explicit CartesianDynamics(double mu);
    // compute_impl, compute_jacobian_impl,
    // compute_jacobian_adjointgradient_adjointhessian_impl (inherited)
};

struct CRTBPDynamics
    : VectorFunction<CRTBPDynamics, 9, 6,
                     DenseDerivativeMode::Analytic,
                     DenseDerivativeMode::Analytic> {
    double mu_;
    double pc0_; // 1 - mu (CSE'd out of the acceleration terms)
    CRTBPDynamics();
    explicit CRTBPDynamics(double mu);
    // compute_impl, compute_jacobian_impl,
    // compute_jacobian_adjointgradient_adjointhessian_impl (inherited)
};

} // namespace tycho::astro
```

### Python API additions

Under the `tychopy._tychopy.Astro` submodule:

```python
Astro.cartesian_dynamics(mu: float) -> GenericFunction
Astro.crtbp_dynamics(mu: float) -> GenericFunction
```

Usage pattern mirrors the existing `Astro.modified_dynamics`: returns a
`GenericFunction` that the caller composes with their own state/acceleration
stacking.

## Error Handling

The generated dynamics functions perform no runtime validation beyond what
Eigen / `VectorFunction` base classes already do. `r = 0` (singularity at
the primary for Cartesian, or at either primary for CRTBP) is
mathematically undefined and produces NaN/Inf, consistent with the existing
`Kepler` / `CR3BP` behavior. Callers are responsible for boundary
conditions that keep the state away from singularities.

## Testing

The pre-merge verification gate from `CLAUDE.md` is applied in full:

1. **Regenerate headers** before building:
   ```bash
   cd utils && conda run -n tycho python codegen_cartesian_dynamics.py
   cd utils && conda run -n tycho python codegen_crtbp_dynamics.py
   ```
2. **Reconfigure and full build** with tests, benchmarks, and examples:
   ```bash
   cmake --preset linux-clang-release \
     -DBUILD_CPP_TESTS=ON -DBUILD_CPP_BENCHMARKS=ON -DBUILD_CPP_EXAMPLES=ON
   cd build && ninja -j8 all
   ```
   (macOS uses `-j6`. Never run two builds simultaneously.)
3. **C++ unit tests:** `ctest --output-on-failure` — all must pass,
   including the ported `test_cr3bp.cpp` (or its renamed equivalent).
4. **C++ builder example smoke tests:** run each of the four converted
   examples and confirm convergence / expected output.
5. **Python integration suite:**
   ```bash
   conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
   ```
   All 38 examples must exit 0.
6. **C++ brachistochrone:** build and run to confirm "Optimal Solution
   Found", objective ≈ 1.8013 s.
7. **Benchmarks:** `bench/bench_track.sh compare`. Record new
   `BM_VF_CRTBPDynamics_*` baselines. The old `BM_VF_CR3BP_*` entries no
   longer exist, so the comparison will show them as removed and the new
   entries as added — justify in the PR body, do not treat as a
   regression.

## Implementation Order

A roughly safe build order (detailed task breakdown happens in the
implementation plan via `writing-plans`):

1. Write and run `codegen_cartesian_dynamics.py`; commit the generated
   header.
2. Write and run `codegen_crtbp_dynamics.py`; commit the generated header.
3. Add the two headers to `include/tycho/astro.h`; add the two lambda
   bindings; build once to verify compilation (without removing
   `cr3bp_model.h` yet).
4. Port `test_cr3bp.cpp`, the CR3BP case in `test_vf_ode_expressions.cpp`,
   and the benchmarks. Run `ctest` and benchmarks to verify physics parity
   between the old and new classes. **This is the correctness
   checkpoint — compare numeric outputs between `CR3BP(mu)` and
   `CRTBPDynamics(mu)` on matching inputs (extra accel zero).**
5. Convert the four builder examples one at a time, running each to
   convergence after conversion.
6. Delete `cr3bp_model.h`, its `#include` in `tycho_astro.cpp`, and the
   unused `#include` in `bench_vector_functions.cpp`.
7. Comment out `add_subdirectory(extensions)` in root `CMakeLists.txt`.
8. Full pre-merge verification sequence; commit; open PR.

## Risks and Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Physics mismatch between `CRTBPDynamics` and hand-written `CR3BP` | Direct numerical parity test at step 4 above; keep `cr3bp_model.h` in place until that passes |
| Performance regression on CRTBP hot loops | Dedicated benchmark (`BM_VF_CRTBPDynamics_*`) + adjoint consistency test give explicit coverage; if slower, revisit CSE strategy in the codegen script |
| Extensions silently broken for downstream users | Unbuild, not delete — files remain in tree, easy to re-enable. PR body explicitly calls this out |
| `multi_spacecraft_opt` zero-accel stacking loses the current inline optimization opportunity | Acceptable by user directive — analytic derivatives from codegen expected to more than offset |
| Generated header regeneration desync (someone edits generated file manually) | Standard practice in this repo; generated files include a `// Generated by ...` header banner already |

## Deferred Items

- **Extensions mechanism rework.** `extensions/` is left on disk but
  unbuilt in this PR. A separate design/brainstorm should decide whether
  the extension mechanism exists in some form, gets deleted, or gets
  reworked into a different pattern.
- **`kepler_model.h` future.** Retained untouched. The `Kepler` type plays
  a distinct role (Kepler propagator integration and two-body analytic
  phase construction). Any future unification with `CartesianDynamics`
  requires a separate design.
- **Full `TychoBind<>` specialization for the new dynamics.** The
  `mod.def` lambda pattern matches the existing `modified_dynamics`
  convention and is sufficient; a future cleanup pass could promote all
  three to full bindings with introspection.

## Open Questions Resolved During Brainstorm

- **Q:** Keep or delete `cr3bp_model.h`? **A:** Delete. (Option A in
  brainstorm.)
- **Q:** Keep or delete `kepler_model.h`? **A:** Keep — solves a different
  problem.
- **Q:** Zero-accel stacking style for `multi_spacecraft_opt`? **A:** Use
  `Constant<6, 3>` with `Eigen::Vector3d::Zero()`.
- **Q:** Struct name — `CRTBPDynamics` vs. `CR3BPDynamics`? **A:**
  `CRTBPDynamics` (matches the filename the user specified and is
  consistent with the `MEEDynamics` naming).
- **Q:** Scope for extensions? **A:** Unbuild in this PR, leave files on
  disk, defer mechanism decision.
- **Q:** Python example updates? **A:** None needed —
  `AstroModels.CR3BP` is pure Python and independent of `cr3bp_model.h`.
