# Spec: Advanced OC Features (Subsystem 5)

> Expose LGLInterpTable, InterpFunction, and STM integration for use in
> C++ builder API examples, and add a `lgl_interp()` convenience
> function. Update Heteroclinic and OptimalDocking examples to full
> Python parity.

## Motivation

Three Advanced OC features are used in the Python Heteroclinic and
OptimalDocking examples but not yet exercised from the C++ builder API:

1. **LGLInterpTable** — trajectory interpolation using LGL polynomial
   basis. Used to store periodic orbit data and target spacecraft
   trajectories.
2. **InterpFunction** — VF expression wrapping LGLInterpTable for use in
   constraints/objectives. Enables time-dependent boundary conditions.
3. **STM integration** — `integrate_stm()` and `integrate_stm_parallel()`
   for State Transition Matrix computation. Used for monodromy matrix
   eigendecomposition and manifold direction identification.

Unlike Subsystems 1-4, these features already exist as C++ classes in the
umbrella headers (`lgl_interp_table.h`, `lgl_interp_functions.h`,
`integrator.h`). The `LGLInterpTable` constructor accepts
`GenericFunction<-1, -1>` (the builder ODE's internal type), and
`DynIntegrator` inherits all STM methods from `Integrator<DynODE>`. The
gap is purely that nobody has verified the integration or provided
convenience API.

## Scope

This spec covers:
- `lgl_interp()` convenience free function for InterpFunction composition
- Full Heteroclinic example port (all 4 stages: orbit computation,
  manifold computation, connection search, heteroclinic optimization)
- OptimalDocking Form2 with torque-free target trajectory via
  LGLInterpTable

This spec does NOT cover:
- MEE astro ODE factories (Subsystem 6)
- `Jet.map()` for batch parameter sweeps (Subsystem 6)

## Changes

### 1. `lgl_interp()` Convenience Function

Add to `include/tycho/detail/vf/operators/interp_compose.h` (alongside
existing `interp()` functions), in the `tycho::vf` namespace:

**Two-argument form** — returns a `GenericFunction` wrapping the
InterpFunction (not yet composed with a time expression). The caller
composes via `.eval(t_expr)`:

```cpp
inline auto lgl_interp(const std::shared_ptr<oc::LGLInterpTable>& table,
                       const Eigen::VectorXi& vars) {
    return GenericFunction<-1, -1>(oc::InterpFunction<-1>(table, vars));
}
```

**Three-argument form** — composes with a time expression directly,
returning the fully composed VF:

```cpp
template <class Func, int IR, int OR>
auto lgl_interp(const std::shared_ptr<oc::LGLInterpTable>& table,
                const Eigen::VectorXi& vars,
                const DenseFunctionBase<Func, IR, OR>& t_expr) {
    return GenericFunction<-1, -1>(
        oc::InterpFunction<-1>(table, vars).eval(t_expr.derived()));
}
```

**Include dependency:** `interp_compose.h` already includes
`lgl_interp_functions.h` transitively via the umbrella ordering
(`optimal_control.h` includes `lgl_interp_functions.h` before
`interp_compose.h`). However, to be explicit and self-contained, add
`#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"`
to `interp_compose.h`.

Usage:
```cpp
// Python: PosFunc = oc.InterpFunction(OrbitTab, range(0,3)).vf()
//         pos_at_t = PosFunc(t)
// C++:
auto pos_func = lgl_interp(tab, pos_vars);          // GenericFunction
auto pos_at_t = lgl_interp(tab, pos_vars, t_expr);  // composed
```

### 2. Heteroclinic Example — Full Port

Replace the current simplified Heteroclinic example with a faithful port
of all four stages from `Heteroclinic.py`:

**Stage 1: `MakeOrbit()`** — Already implemented. Compute L1 and L2
Lyapunov orbits via periodic orbit optimization with Jacobi constant
constraint.

**Stage 2: `GetManifold()`** — New. Compute stable/unstable manifolds:
- Create `DynIntegrator` via `ode.integrator(IVPAlg::DOPRI87, 0.01)`
- Sample orbit at `nman` points via `integrate_dense()`
- Compute monodromy matrices via `integrate_stm_parallel()`
- Eigenvalue decomposition via `Eigen::EigenSolver<Eigen::MatrixXd>`
- Perturb initial conditions along stable/unstable eigenvectors
- Propagate manifold arms via `integrate_dense_parallel()` with event
  detection (Moon crossing, SOI cull)
- Filter results by event crossings

**Stage 3: `FindClosestConnection()`** — New. Brute-force search over
manifold endpoint pairs for minimum position+velocity distance. Pure
Eigen math.

**Stage 4: `MakeHeteroclinic()`** — New. Two-phase OCP with orbit-
matching constraints:
- `std::make_shared<LGLInterpTable>(orbit_data)` + `make_periodic()`
- Position constraint via `lgl_interp(tab, pos_vars, t)` in
  `add_equal_con()`
- Velocity departure objective via `lgl_interp(tab, vel_vars, t)` in
  `add_state_objective()`
- Two-phase OCP with forward link equal constraint
- Adaptive mesh, optimize

**Success criteria:** Problem converges, produces a reasonable total DV
(same order of magnitude as Python). Exact numerical match not expected
due to manifold selection sensitivity.

### 3. OptimalDocking Form2

Add Form2 to the OptimalDocking example, matching the Python pattern:

**TorqueFree ODE** — 7 states (quaternion + angular velocity), 0
controls. Torque-free rotational dynamics with moment-of-inertia
coupling:
```cpp
auto pdot = quat_product(p, phi.padded_lower(1)) / 2.0;
auto L2 = cwise_product(phi, I2_vec);
auto phidot = cwise_product(L2.cross(phi), I2_inv_vec);
```

**Target trajectory generation:**
- Build TorqueFree ODE via builder API
- Create integrator, propagate for `SimTime = 600/Tstar`
- Store in `LGLInterpTable` with the ODE's VF

**RelDynModel2 ODE** — 13 states (servicer only), 6 controls. Same CW
frame dynamics + quaternion kinematics as Form1, minus target states.

**RendCon2 constraint** — Uses `lgl_interp()` to look up target state
at time `t`, then composes with the existing `RendCon` rendezvous
function to compute position+velocity error.

**Phase setup:** LGL3, 384 segments, BlockConstant control, thrust/torque
bounds, safety sphere constraint, time-optimal objective.

**Success criteria:** Converges, final time matches Python (~similar
order). Form2 should produce similar results to existing Form1 but with
a smaller OCP (13 vs 20 states).

## Verification

- **Heteroclinic:** All 4 stages complete, problem converges with
  reasonable DV. PASSED output.
- **OptimalDocking:** Both Form1 (existing) and Form2 (new) converge.
  PASSED output for both.
- All 31+ C++ example CTests pass (no regressions).
- All Python examples pass.

## Files Modified

**Modified:**
- `include/tycho/detail/vf/operators/interp_compose.h` — add
  `lgl_interp()` convenience functions + LGLInterpTable include
- `examples/cpp_examples/builder/heteroclinic/main.cpp` — full 4-stage
  port
- `examples/cpp_examples/builder/optimal_docking/main.cpp` — add Form2

**No new files.** All infrastructure already exists in the codebase.
