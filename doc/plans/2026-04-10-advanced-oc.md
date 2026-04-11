# Advanced OC Features — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lgl_interp()` convenience function and update Heteroclinic (full 4-stage port) and OptimalDocking (Form2 with torque-free target) examples to use LGLInterpTable, InterpFunction, and STM integration.

**Architecture:** Most infrastructure already exists — LGLInterpTable, InterpFunction, and STM methods are in umbrella headers with compatible types. We add one thin convenience function (`lgl_interp()`), then port the examples. The Heteroclinic port is the most complex: STM eigendecomposition → manifold propagation with events → orbit-matching OCP via LGLInterpTable.

**Tech Stack:** C++20, Eigen (incl. EigenSolver for eigendecomposition), existing VF/OC/integrator machinery

**Spec:** `doc/plans/TYCHO_ADVANCED_OC_SPEC.md`

---

## File Structure

**Modified files:**
- `include/tycho/detail/vf/operators/interp_compose.h` — add `lgl_interp()` + LGLInterpTable include
- `examples/cpp_examples/builder/heteroclinic/main.cpp` — full 4-stage port
- `examples/cpp_examples/builder/optimal_docking/main.cpp` — add Form2

---

## Task 0: lgl_interp() Convenience Function

**Files:**
- Modify: `include/tycho/detail/vf/operators/interp_compose.h`

- [ ] **Step 1: Read interp_compose.h**

Read `include/tycho/detail/vf/operators/interp_compose.h` to see the existing `interp()` and `interp_scalar()` functions.

- [ ] **Step 2: Add lgl_interp_functions.h include**

Add after the existing interp_table includes:
```cpp
#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"
```

- [ ] **Step 3: Add lgl_interp() functions**

Add at the end of the `namespace tycho::vf` block, after the existing 4D interp functions:

```cpp
// ── LGL trajectory interpolation ───────────────────────────────────

/// Return a GenericFunction wrapping an LGLInterpTable (not yet composed
/// with a time expression). Call .eval(t_expr) on the result to compose.
inline auto lgl_interp(const std::shared_ptr<oc::LGLInterpTable> &table,
                       const Eigen::VectorXi &vars) {
    return GenericFunction<-1, -1>(oc::InterpFunction<-1>(table, vars));
}

/// Compose an LGLInterpTable with a VF time expression directly.
template <class Func, int IR, int OR>
auto lgl_interp(const std::shared_ptr<oc::LGLInterpTable> &table,
                const Eigen::VectorXi &vars,
                const DenseFunctionBase<Func, IR, OR> &t_expr) {
    return GenericFunction<-1, -1>(
        oc::InterpFunction<-1>(table, vars).eval(t_expr.derived()));
}
```

- [ ] **Step 4: Build and test**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_builder
```

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/vf/operators/interp_compose.h
git commit -m "feat: add lgl_interp() convenience function for LGLInterpTable

lgl_interp(table, vars) returns a GenericFunction wrapping InterpFunction.
lgl_interp(table, vars, t_expr) composes with a time expression directly."
```

---

## Task 1: Full Heteroclinic Port

**Files:**
- Modify: `examples/cpp_examples/builder/heteroclinic/main.cpp`

This is the largest task. It replaces the current simplified example with a faithful port of all 4 stages from `Heteroclinic.py`.

- [ ] **Step 1: Read the Python example and current C++ example**

Read:
- `examples/python_examples/Heteroclinic.py` (full file)
- `examples/cpp_examples/builder/heteroclinic/main.cpp` (current)
- `include/tycho/detail/integrators/integrator.h` (STM methods: lines 1985-2031, integrate_dense_parallel: lines 1903-1936, EventPack at line 96)
- `include/tycho/detail/optimal_control/transcription/lgl_interp_table.h` (constructors: lines 92-170, make_periodic: lines 193-207)

- [ ] **Step 2: Implement full heteroclinic example**

Replace the entire `main.cpp` with a full port. The structure should follow the Python closely:

**Keep existing infrastructure:**
- `make_cr3bp_ode()` — already works, keep as-is
- `make_jacobi_func()` — already works, keep as-is
- `make_orbit()` — already works, keep as-is (may need minor adjustments)

**Add `GetManifold()` function:**

```cpp
using DynIntegrator = ODE::DynIntegrator;

std::vector<std::vector<Eigen::VectorXd>> get_manifold(
    const ODE &ode,
    const std::vector<Eigen::VectorXd> &orbit_in,
    double dx, double dt, int nman = 50, bool stable = true)
{
    auto integ = ode.integrator(IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1.0e-13);

    double period = orbit_in.back()[6];
    auto orbit = integ.integrate_dense(orbit_in[0], period, nman);

    // STM integration for monodromy matrices
    Eigen::VectorXd times(orbit.size());
    for (int i = 0; i < (int)orbit.size(); ++i)
        times[i] = orbit[i][6] + period;

    int ncores = 8;
    auto stm_results = integ.integrate_stm_parallel(orbit, times, ncores);

    // Perturb along stable/unstable eigenvectors
    std::vector<Eigen::VectorXd> eig_igs;
    for (int i = 0; i < (int)stm_results.size(); ++i) {
        auto &[xf, jac] = stm_results[i];

        // Eigendecomposition of 6x6 state block
        Eigen::EigenSolver<Eigen::MatrixXd> es(jac.block(0, 0, 6, 6));
        auto vals = es.eigenvalues();
        auto vecs = es.eigenvectors();

        // Sort by magnitude
        std::vector<int> idxs(6);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(),
                  [&](int a, int b) { return std::abs(vals[a]) < std::abs(vals[b]); });

        // Pick stable (smallest) or unstable (largest) eigenvector
        Eigen::Vector3d dir = stable
            ? vecs.col(idxs[0]).real().head<3>()
            : vecs.col(idxs[5]).real().head<3>();
        dir.normalize();

        // Two perturbations per orbit point (+ and -)
        Eigen::VectorXd xp = orbit[i];
        xp.head<3>() += dir * dx;
        eig_igs.push_back(xp);

        xp = orbit[i];
        xp.head<3>() -= dir * dx;
        eig_igs.push_back(xp);
    }

    // Propagation times (negative for stable manifold)
    double prop_dt = stable ? -dt : dt;
    Eigen::VectorXd prop_times(eig_igs.size());
    for (int i = 0; i < (int)eig_igs.size(); ++i)
        prop_times[i] = eig_igs[i][6] + prop_dt;

    // Event functions
    // CrossMoon: x - (1 - mu) = 0
    auto cross_moon_args = Arguments<7>();
    auto cross_moon_func = GenericFunction<-1, 1>(
        cross_moon_args.coeff<0>() - (1.0 - mu));

    // Cull: (alt_from_moon - 0.015) * (y - 0.15) * (y + 0.15)
    auto cull_args = Arguments<7>();
    Eigen::Vector3d p2loc;
    p2loc << 1.0 - mu, 0.0, 0.0;
    auto p2c = Constant<7, 3>(7, p2loc);
    auto alt = (cull_args.head<3>() - p2c).norm() - 0.015;
    auto y_fac = (cull_args.coeff<1>() - 0.15) * (cull_args.coeff<1>() + 0.15);
    auto cull_func = GenericFunction<-1, 1>(alt * y_fac);

    using EventPack = DynIntegrator::EventPack;
    std::vector<EventPack> events = {
        {cross_moon_func, 0, 1},
        {cull_func, 0, 1}
    };

    auto results = integ.integrate_dense_parallel(eig_igs, prop_times, ncores, events);

    // Filter: accept those that crossed moon and weren't culled
    std::vector<std::vector<Eigen::VectorXd>> manifolds;
    for (auto &[traj, eventlocs] : results) {
        if (eventlocs[0].size() == 1 && eventlocs[1].empty()) {
            traj.pop_back();
            traj.push_back(eventlocs[0][0]);
            manifolds.push_back(std::move(traj));
        }
    }
    return manifolds;
}
```

**Add `FindClosestConnection()` function:**

```cpp
std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
find_closest_connection(
    const std::vector<std::vector<Eigen::VectorXd>> &orbs1,
    const std::vector<std::vector<Eigen::VectorXd>> &orbs2)
{
    double best_dist = std::numeric_limits<double>::max();
    int best_i = 0, best_j = 0;
    for (int i = 0; i < (int)orbs1.size(); ++i) {
        for (int j = 0; j < (int)orbs2.size(); ++j) {
            double dist = (orbs1[i].back().head<6>() - orbs2[j].back().head<6>()).norm();
            if (dist < best_dist) {
                best_dist = dist;
                best_i = i;
                best_j = j;
            }
        }
    }
    return {orbs1[best_i], orbs2[best_j]};
}
```

**Add `MakeHeteroclinic()` function:**

```cpp
void make_heteroclinic(const ODE &ode,
                       const std::vector<Eigen::VectorXd> &man1,
                       const std::vector<Eigen::VectorXd> &man2,
                       const std::vector<Eigen::VectorXd> &L1Orbit,
                       const std::vector<Eigen::VectorXd> &L2Orbit)
{
    // Create LGL interp tables for periodic orbits
    auto orbit_tab1 = std::make_shared<LGLInterpTable>(L1Orbit);
    orbit_tab1->make_periodic();
    auto orbit_tab2 = std::make_shared<LGLInterpTable>(L2Orbit);
    orbit_tab2->make_periodic();

    // Position constraint: R - PosFunc(t) = 0
    Eigen::VectorXi pos_vars(3);
    pos_vars << 0, 1, 2;
    Eigen::VectorXi vel_vars(3);
    vel_vars << 3, 4, 5;

    auto make_pos_con = [&](const std::shared_ptr<LGLInterpTable> &tab) {
        auto rt_args = Arguments<4>();
        auto R = rt_args.head<3>();
        auto t = rt_args.coeff<3>();
        return GenericFunction<-1, -1>(R - lgl_interp(tab, pos_vars, t));
    };

    auto make_dv_obj = [&](const std::shared_ptr<LGLInterpTable> &tab) {
        auto vt_args = Arguments<4>();
        auto V = vt_args.head<3>();
        auto t = vt_args.coeff<3>();
        return GenericFunction<-1, -1>(
            (V - lgl_interp(tab, vel_vars, t)).squared_norm());
    };

    // Phase 1: departing from L1 orbit (skip first point of manifold)
    std::vector<Eigen::VectorXd> man1_trimmed(man1.begin() + 1, man1.end());
    auto phase1 = ode.phase(TranscriptionModes::LGL7, man1_trimmed, 50);

    double L1_period = L1Orbit.back()[6];
    phase1.add_lower_var_bound(PhaseRegionFlags::Front, 6, -L1_period);
    phase1.add_upper_var_bound(PhaseRegionFlags::Front, 6, 2.0 * L1_period);

    // Orbit-matching constraints at departure
    phase1.add_equal_con(PhaseRegionFlags::Front, make_pos_con(orbit_tab1),
                         {0, 1, 2, 6});
    phase1.add_state_objective(PhaseRegionFlags::Front, make_dv_obj(orbit_tab1),
                               {3, 4, 5, 6});

    // Phase 2: arriving at L2 orbit (skip last point of manifold)
    std::vector<Eigen::VectorXd> man2_trimmed(man2.begin(), man2.end() - 1);
    auto phase2 = ode.phase(TranscriptionModes::LGL7, man2_trimmed, 50);

    double L2_period = L2Orbit.back()[6];
    phase2.add_equal_con(PhaseRegionFlags::Back, make_pos_con(orbit_tab2),
                         {0, 1, 2, 6});
    phase2.add_state_objective(PhaseRegionFlags::Back, make_dv_obj(orbit_tab2),
                               {3, 4, 5, 6});

    // Time bounds for phase 1 back (arrival near L2)
    phase1.add_lower_var_bound(PhaseRegionFlags::Back, 6, -L2_period);
    phase1.add_upper_var_bound(PhaseRegionFlags::Back, 6, 2.0 * L2_period);

    // Two-phase OCP
    auto ocp = OptimalControlProblem();
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);
    ocp.add_forward_link_equal_con(phase1, phase2, {0, 1, 2, 3, 4, 5});
    ocp.set_adaptive_mesh(true);

    ocp.optimizer().set_econ_tol(1.0e-9);
    ocp.optimizer().set_opt_ls_mode("L1");
    ocp.optimize();

    auto traj1 = phase1.return_traj();
    auto traj2 = phase2.return_traj();

    // Compute DV
    auto orb1_state = orbit_tab1->interpolate<double>(traj1[0][6]);
    auto orb2_state = orbit_tab2->interpolate<double>(traj2.back()[6]);

    double dv1 = (traj1[0].segment<3>(3) - orb1_state.segment<3>(3)).norm();
    double dv2 = (traj2.back().segment<3>(3) - orb2_state.segment<3>(3)).norm();

    // Non-dimensionalise: vstar = sqrt(mu_total / LD)
    double LD = 384400000.0; // m
    double mu_total = MuEarth + MuMoon;
    double vstar = std::sqrt(mu_total / LD);
    std::cout << "Total DV: " << (dv1 + dv2) * vstar << " m/s\n";
}
```

**Update `main()`:**
- After computing L1/L2 orbits (existing), add:
  - `GetManifold()` calls for unstable L1 and stable L2
  - `FindClosestConnection()` to select best pair
  - Reverse the stable manifold (Man2)
  - `MakeHeteroclinic()` for the final optimization
  - Print PASSED

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 heteroclinic_builder
./examples/cpp_examples/builder/heteroclinic/heteroclinic_builder
```

Expected: converges, prints total DV in m/s, "PASSED".

Note: this example may take 30-60 seconds due to manifold propagation. The CTest timeout should be generous (120s+). Check `examples/cpp_examples/builder/heteroclinic/CMakeLists.txt` for the timeout setting and increase if needed.

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/heteroclinic/main.cpp
git commit -m "feat: full Heteroclinic port with STM manifolds and orbit-matching

Complete 4-stage port: periodic orbits, STM-based manifold computation
with eigendecomposition, closest connection search, heteroclinic
optimization with LGLInterpTable + lgl_interp() orbit-matching
constraints. Uses integrate_stm_parallel, integrate_dense_parallel
with event detection."
```

---

## Task 2: OptimalDocking Form2

**Files:**
- Modify: `examples/cpp_examples/builder/optimal_docking/main.cpp`

- [ ] **Step 1: Read files**

Read:
- `examples/python_examples/OptimalDocking.py` (full file — focus on TorqueFree class at lines 161-179, RelDynModel2 at lines 123-157, RendCon2 at lines 205-211, Form2 at lines 369-429)
- `examples/cpp_examples/builder/optimal_docking/main.cpp` (current C++ Form1)

- [ ] **Step 2: Add Form2 implementation**

Add to the existing file (after the current Form1 code):

**TorqueFree ODE** — 7 states, 0 controls:
```cpp
ODE make_torquefree_ode(const Eigen::Vector3d &I_vec) {
    auto args = ODEArguments(7, 0, 0);
    auto p = args.head<4>().normalized();
    auto phi = args.segment<3>(4);

    auto pdot = quat_product(p, phi.padded_lower<1>()) / 2.0;
    auto L2 = cwise_product(phi, I_vec);
    Eigen::Vector3d I_inv;
    I_inv << 1.0 / I_vec[0], 1.0 / I_vec[1], 1.0 / I_vec[2];
    auto phidot = cwise_product(L2.cross(phi), I_inv);

    return ODE(StackedOutputs{pdot, phidot}, 7, 0);
}
```

**RelDynModel2 ODE** — 13 states, 6 controls (servicer only):
Same as existing Form1 dynamics but without target (p, phi) states.

**RendCon2 function** — time-dependent rendezvous constraint:
```cpp
auto make_rendcon2(const Eigen::Vector3d &ud_vec,
                   const std::shared_ptr<LGLInterpTable> &target_tab) {
    // Look up 7 target states (quat + omega) at time t
    Eigen::VectorXi target_vars(7);
    target_vars << 0, 1, 2, 3, 4, 5, 6;
    auto target_func = lgl_interp(target_tab, target_vars);

    // Full RendCon takes 20 inputs: [X,V,q,w, p,phi]
    // RendCon2 takes 14 inputs: [X,V,q,w, t]
    // Maps t → target state (p,phi) via InterpFunction, then calls RendCon
    auto args_14 = Arguments<14>();
    auto X = args_14.head<3>();
    auto V = args_14.segment<3>(3);
    auto q = args_14.segment<4>(6);
    auto w = args_14.segment<3>(10);
    auto t = args_14.coeff<13>();

    auto target_state = GenericFunction<-1, -1>(target_func.eval(t));
    // target_state gives [p(4), phi(3)] at time t

    // Build RendCon inline: rendezvous position/velocity error
    // using quat_rotate for body-to-global transformation
    auto ud = Constant<20, 3>(20, ud_vec);
    auto rendcon_args = Arguments<20>();
    // ... compose RendCon(ud)(X, V, q, w, target_state(t))
    // This matches RendCon2 in Python
    auto rendcon = make_rendcon(ud_vec); // existing RendCon function
    return GenericFunction<-1, -1>(
        rendcon.eval(stack(X, V, q, w, target_state)));
}
```

**Form2 function:**
1. Build TorqueFree ODE, integrate target trajectory
2. Create `LGLInterpTable` from target trajectory with ODE's VF
3. Build RelDynModel2 ODE (servicer only, 13 states)
4. Integrate initial guess
5. Set up phase with RendCon2 constraint at "Last"
6. Solve, print final time

**Update `main()`** to run both Form2 and Form1 (like Python's `__main__`).

- [ ] **Step 3: Build and run**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 optimal_docking_builder
./examples/cpp_examples/builder/optimal_docking/optimal_docking_builder
```

Expected: Both Form2 and Form1 converge, print final times, "PASSED".

- [ ] **Step 4: Commit**

```bash
git add examples/cpp_examples/builder/optimal_docking/main.cpp
git commit -m "feat: add OptimalDocking Form2 with LGLInterpTable target trajectory

Add torque-free ODE, integrate target trajectory, compose via
lgl_interp() in RendCon2 constraint. Form2 reduces state from 20
to 13 by parameterizing target via interpolation table."
```

---

## Task 3: Full Verification

- [ ] **Step 0: Triage remaining examples**

BettsLowThrust and OptimalDocking Form1 don't use LGLInterpTable/InterpFunction — confirmed in Subsystem 4 triage. No changes needed beyond Form2.

- [ ] **Step 1: Build all examples**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j2 cpp_examples_all
```

- [ ] **Step 2: Run CTest**

```bash
ctest -R cpp_example --output-on-failure -j1
```

Expected: all pass. Note: Heteroclinic may take 30-60s.

- [ ] **Step 3: Update gap doc**

Mark resolved:
- Heteroclinic `LGLInterpTable` + `make_periodic()` → RESOLVED
- Heteroclinic `InterpFunction` → RESOLVED
- Heteroclinic `integrate_stm_parallel` → RESOLVED
- OptimalDocking `InterpTable` for target trajectory → RESOLVED

- [ ] **Step 4: Commit any remaining fixes**
