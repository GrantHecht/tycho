(user-guide-examples)=

# Worked examples

This page collects complete, runnable optimal-control solves drawn straight
from Tycho's example suite. Each one is a real problem — dynamics, phase setup,
constraints, an objective, and a solve — and each is curated to surface a
different part of the library. The snippets below are faithful excerpts that
show the *core formulation* in both Python and C++; imports, plotting, and
post-processing are trimmed for clarity. The full sources (including the
plotting that produced the figures these problems are known for) live under
[`examples/`](https://github.com/GrantHecht/tycho/tree/main/examples) — the
Python scripts double as Tycho's integration-test suite.

Across the slate below you'll see single-phase and multi-phase problems, free
and fixed final time, path/state bounds and nonlinear path constraints, vector
(norm) control bounds, ODE parameters, event detection, table-interpolated
data, adaptive mesh refinement, and an astrodynamics low-thrust transfer.

## Brachistochrone

The canonical first optimal-control problem: a bead slides without friction
from a start point to an end point under gravity, steered only by the wire
angle `theta`. State is position and speed `(x, y, v)`; the control is
`theta`; the objective is to minimize the (free) final time.

- **Features:** single-phase basics, free final time
  (`add_delta_time_objective`), a control path bound, front/back boundary
  values.

::::{tab-set}
:::{tab-item} Python
```python
class Brachistochrone(oc.ODEBase):
    def __init__(self, g):
        XtU = oc.ODEArguments(3, 1)
        x, y, v = XtU.x_vec().tolist()
        theta = XtU.u_var(0)

        xdot = vf.sin(theta) * v
        ydot = -1.0 * vf.cos(theta) * v
        zdot = g * vf.cos(theta)
        ode = vf.stack([xdot, ydot, zdot])
        super().__init__(ode, 3, 1)


ode = Brachistochrone(9.81)

phase = ode.phase("LGL3", Xs, 32)
phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
phase.add_lu_var_bound("Path", 4, -0.1, 2.00)
phase.add_boundary_value("Back", [0, 1], [xf, yf])
phase.add_delta_time_objective(1.0)
phase.optimize()
```
:::
:::{tab-item} C++
```cpp
auto ode = ODEBuilder(3, 1)
               .define([g](auto &args) {
                   auto v = args.x_var(2);
                   auto theta = args.u_var(0);
                   return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
               })
               .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
               .build();

auto phase = ode.phase(TranscriptionModes::LGL3, traj, n_defects);
phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                         Eigen::Vector4d(x0, y0, v0, 0.0));
phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"}, Eigen::Vector2d(xf, yf));
phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);
phase.add_delta_time_objective(1.0);
const auto status = phase.solve_optimize();
```
:::
::::

**Result:** the optimal descent time is about **1.8013 s** — the known
brachistochrone minimum for this geometry.

Full source:
[`Brachistochrone.py`](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/Brachistochrone.py)
·
[`brachistochrone/main.cpp`](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/brachistochrone/main.cpp)

## Van der Pol oscillator

A classic nonlinear system: minimize the integral of `x0^2 + x1^2 + u^2` while
driving the Van der Pol oscillator from its limit cycle to the origin over a
*fixed* final time. The control is bounded and evaluated as a block-constant
profile.

- **Features:** integral objective (`add_integral_objective`), block-constant
  control mode, a control path bound, fixed final time, solver tolerance and
  partition tuning.

::::{tab-set}
:::{tab-item} Python
```python
class VanderPol(oc.ODEBase):
    def __init__(self):
        args = oc.ODEArguments(2, 1)
        x0 = args[0]
        x1 = args[1]
        u = args[3]

        x0dot = (1.0 - x1 * x1) * x0 - x1 + u
        x1dot = x0
        ode = vf.stack(x0dot, x1dot)
        super().__init__(ode, 2, 1)


ode = VanderPol()
phase = ode.phase("LGL3", TrajIG, 256)
phase.set_control_mode("BlockConstant")
phase.add_boundary_value("Front", range(0, 3), [0, 1, 0])
phase.add_lu_var_bound("Path", 3, -0.75, 1.0, 1.0)
phase.add_integral_objective(Args(3).squared_norm(), [0, 1, 3])
phase.add_boundary_value("Back", [0, 1, 2], [0.0, 0.0, tf])
phase.set_num_partitions(8, 8)
phase.optimizer.set_tols(1.0e-8, 1.0e-8, 1.0e-8)
phase.optimize()
```
:::
:::{tab-item} C++
```cpp
auto ode = ODEBuilder(2, 1)
               .define([](auto &args) {
                   auto x0 = args.x_var(0);
                   auto x1 = args.x_var(1);
                   auto u = args.u_var(0);
                   auto x0dot = (1.0 - x1 * x1) * x0 - x1 + u;
                   auto x1dot = x0;
                   return stack(x0dot, x1dot);
               })
               .var_names({{"x0", 0}, {"x1", 1}, {"t", 2}, {"u", 3}})
               .build();

auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);
phase.set_control_mode(ControlModes::BlockConstant);
phase.add_boundary_value(PhaseRegionFlags::Front, {"x0", "x1", "t"},
                         Eigen::Vector3d(0.0, 1.0, 0.0));
phase.add_boundary_value(PhaseRegionFlags::Back, {"x0", "x1", "t"},
                         Eigen::Vector3d(0.0, 0.0, tf));
phase.add_lu_var_bound(PhaseRegionFlags::Path, "u", -0.75, 1.0);

auto obj_args = Arguments<3>();
auto obj_expr = obj_args.squared_norm();
phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"x0", "x1", "u"});
phase.set_num_partitions(8, 8);
const auto flag = phase.optimize();
```
:::
::::

**Result:** the solver drives both states to the origin at `t = tf` while
minimizing the running cost; the optimal control rides its lower bound early
in the trajectory and relaxes as the states settle.

Full source:
[`VanDerPol.py`](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/VanDerPol.py)
·
[`van_der_pol/main.cpp`](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/van_der_pol/main.cpp)

## Goddard rocket (singular arc)

Maximize the final altitude of a vertically-launched rocket subject to drag and
a thrust budget. The classic single-phase formulation has a *singular arc*; the
multi-phase formulation below splits the trajectory into full-thrust, singular,
and coast phases, defining the singular-arc control with a nonlinear equality
*path constraint* and linking the phases for continuity.

- **Features:** multi-phase linkage (`add_forward_link_equal_con`), a nonlinear
  equality path constraint (`add_equal_con`) for the singular arc, a value
  objective, auto-scaling with physical units, and an integrator-with-control
  initial guess.

::::{tab-set}
:::{tab-item} Python
```python
class GoddardRocket(oc.ODEBase):
    def __init__(self, sigma, c, h_ref, Tmag, g):
        XtU = oc.ODEArguments(3, 1)
        h, v, m = XtU.x_vec().tolist()
        u = XtU.u_var(0)

        hdot = v
        vdot = (u * Tmag - sigma * (v**2) * vf.exp(-h / h_ref)) / m - g
        mdot = -u * Tmag / c
        ode = vf.stack(hdot, vdot, mdot)
        super().__init__(ode, 3, 1)


# Phase 1: full thrust.  Phase 2: singular arc.  Phase 3: coast.
phase2.set_control_mode("NoSpline")
phase2.add_lu_var_bound("Path", "u", 0.0, 1.0, 1.0)
phase2.add_equal_con("Path", PathCon(sigma, c, h_ref, Tmag, g), ["h", "v", "m", "u"])

phase3.add_value_objective("Back", "h", -1.0)   # maximize final altitude

ocp = oc.OptimalControlProblem()
ocp.add_phase(phase1)
ocp.add_phase(phase2)
ocp.add_phase(phase3)
ocp.add_forward_link_equal_con(phase1, phase3, ["h", "v", "m", "t"])
ocp.set_auto_scaling(True, True)
ocp.optimize()
```
:::
:::{tab-item} C++
```cpp
auto ode = ODEBuilder(3, 1)
    .define([](auto &args) {
        auto h = args.x_var(0);
        auto v = args.x_var(1);
        auto m = args.x_var(2);
        auto u = args.u_var(0);
        auto hdot = v;
        auto vdot = (u * Tmag - sigma * (v * v) * exp(h * (-1.0 / h_ref))) / m - g;
        auto mdot = u * (-Tmag / c);
        return stack(hdot, vdot, mdot);
    })
    .var_names({{"h", 0}, {"v", 1}, {"m", 2}, {"t", 3}, {"u", 4}})
    .build();

// Phase 2 carries the singular-arc control as an equality path constraint.
phase2.set_control_mode(ControlModes::NoSpline);
phase2.add_lu_var_bound(PhaseRegionFlags::Path, "u", 0.0, 1.0, 1.0);
phase2.add_equal_con(PhaseRegionFlags::Path, make_path_constraint(), {"h", "v", "m", "u"});
phase3.add_value_objective(PhaseRegionFlags::Back, "h", -1.0);  // maximize altitude

OptimalControlProblem ocp;
ocp.add_phase(phase1);
ocp.add_phase(phase2);
ocp.add_phase(phase3);
ocp.add_forward_link_equal_con(phase1, phase3, {"h", "v", "m", "t"});
ocp.set_auto_scaling(true, true);
const auto status = ocp.optimize();
```
:::
::::

**Result:** the solver recovers the bang–singular–coast structure — a
full-thrust burn, a smoothly varying singular arc that follows Betts'
constraint, and a ballistic coast to maximum altitude.

Full source:
[`GoddardRocket.py`](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/GoddardRocket.py)
·
[`goddard_rocket/main.cpp`](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/goddard_rocket/main.cpp)

## Multi-phase cannonball

Find the cannonball *radius* that maximizes downrange distance, given a fixed
launch energy and constant material density. An ascent phase and a descent
phase are linked for state continuity, and the shared radius is carried as an
ODE *parameter* linked across the two phases. The launch energy enters as an
inequality constraint and the apogee/ground-impact events seed the initial
guess.

- **Features:** multi-phase linkage, ODE parameters (`p_var`) linked with
  `add_param_link_equal_con`, an inequality constraint (`add_inequal_con`),
  event detection for the initial guess, and a value objective (max range).

::::{tab-set}
:::{tab-item} Python
```python
class Cannon(oc.ODEBase):
    def __init__(self, CD, RhoAir, RhoIron, h_scale, g):
        args = oc.ODEArguments(4, 0, 1)
        v = args.x_var(0)
        gamma = args.x_var(1)
        h = args.x_var(2)
        r = args.x_var(3)
        rad = args.p_var(0)          # ODE parameter: cannonball radius

        S = np.pi * (rad**2)                            # frontal area
        M = (4 / 3) * (np.pi * RhoIron) * (rad**3)      # mass (constant density)

        rho = RhoAir * vf.exp(-h / h_scale)
        D = (0.5 * CD) * rho * (v**2) * S
        vdot = -D / M - g * vf.sin(gamma)
        gammadot = -g * vf.cos(gamma) / v
        hdot = v * vf.sin(gamma)
        rdot = v * vf.cos(gamma)
        ode = vf.stack([vdot, gammadot, hdot, rdot])
        super().__init__(ode, 4, 0, 1)


aphase.add_lower_var_bound("ODEParams", "rad", 0.0)
aphase.add_inequal_con("Front", EFunc(), ["v"], ["rad"], [])
dphase.add_value_objective("Back", "r", -1.0)    # maximize range

ocp = oc.OptimalControlProblem()
ocp.add_phase(aphase)
ocp.add_phase(dphase)
ocp.add_forward_link_equal_con(aphase, dphase, range(0, 5))
ocp.add_param_link_equal_con(aphase, dphase, "ODEParams", "rad")
ocp.optimize()
```
:::
:::{tab-item} C++
```cpp
auto ode = ODEBuilder(4, 0, 1)
    .define([](auto &args) {
        auto v = args.x_var(0);
        auto gamma = args.x_var(1);
        auto h = args.x_var(2);
        auto rad = args.p_var(0);          // ODE parameter: cannonball radius
        auto M = (4.0 / 3.0) * (M_PI * RhoIron) * (rad * rad * rad);
        auto S = M_PI * (rad * rad);
        auto rho = RhoAir * exp(h * (-1.0 / h_scale));
        auto D = (0.5 * CD) * rho * (v * v) * S;
        auto vdot = D * (-1.0) / M - g * sin(gamma);
        auto gammadot = g * cos(gamma) * (-1.0) / v;
        auto hdot = v * sin(gamma);
        auto rdot = v * cos(gamma);
        return stack(vdot, gammadot, hdot, rdot);
    })
    .var_names({{"v", 0}, {"gamma", 1}, {"h", 2}, {"r", 3}, {"t", 4}, {"rad", 5}})
    .build();

aphase.add_inequal_con(PhaseRegionFlags::Front, make_energy_constraint(),
                       xvars, pvars, empty, ScaleModes::AUTO);
dphase.add_value_objective(PhaseRegionFlags::Back, "r", -1.0);   // maximize range

OptimalControlProblem ocp;
ocp.add_phase(aphase);
ocp.add_phase(dphase);
ocp.add_forward_link_equal_con(aphase, dphase, {"v", "gamma", "h", "r", "t"});
// Direct form of the ODE-parameter link (equivalent to add_param_link_equal_con).
ocp.add_direct_link_equal_con(0, PhaseRegionFlags::ODEParams, pvar,
                              1, PhaseRegionFlags::ODEParams, pvar);
const auto status = ocp.optimize();
```
:::
::::

**Result:** the optimizer settles on a launch angle of about **31.9°** and a
cannonball radius near **0.042 m**, giving a downrange distance of roughly
**3280 m**.

Full source:
[`MultiPhaseCannon.py`](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/MultiPhaseCannon.py)
·
[`multi_phase_cannon/main.cpp`](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/multi_phase_cannon/main.cpp)

## Simple low-thrust transfer

A continuous low-thrust orbit-raising problem in non-dimensional two-body
dynamics: spiral from a circular orbit at `r = 1` to a circular orbit at
`r = 2` using a bounded thrust-direction vector. The same phase is re-solved
against three different objectives — minimum time, minimum power, and minimum
fuel — reusing one transcription.

- **Features:** astrodynamics / continuous low-thrust control, a vector control
  *norm* bound (`add_lu_norm_bound`), and swapping objectives
  (time / power / fuel) on a single phase.

::::{tab-set}
:::{tab-item} Python
```python
class LTModel(oc.ODEBase):
    def __init__(self, mu, ltacc):
        args = oc.ODEArguments(6, 3)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        g = r.normalized_power3() * (-mu)
        thrust = u * ltacc
        ode = vf.stack([v, g + thrust])
        super().__init__(ode, 6, 3)


phase = ode.phase("LGL3", TrajIG, 256)
phase.add_boundary_value("Front", range(0, 7), X0)
phase.add_lu_norm_bound("Path", [7, 8, 9], 0.001, 1, 1.0)   # bound |u|
phase.add_boundary_value("Back", range(0, 6), Xf[0:6])

# Minimum-time solve ...
phase.add_delta_time_objective(1.0)
phase.optimize()

# ... then re-solve for minimum power on the same phase.
phase.remove_state_objective(-1)
phase.add_integral_objective(LTModel.powerobj(0.5), [7, 8, 9])
phase.optimize()

# (min-fuel solve omitted here — same re-solve pattern with LTModel.massobj)
```
:::
:::{tab-item} C++
```cpp
auto args = ODEArguments(6, 3, 0);
auto state = args.head<6>();
auto u = args.tail<3>();
auto thrust_acc = acc * u;

auto dyn = astro::CartesianDynamics(mu);
auto ode_expr = GenericFunction<-1, -1>(dyn.eval(stack(state, thrust_acc)));
auto ode = ODE(ode_expr, 6, 3)
               .var_group("R", 0, 3).var_group("V", 3, 3)
               .var_names({{"t", 6}}).var_group("u", 7, 3);

auto phase = ode.phase(TranscriptionModes::LGL3, trajIG, nSeg);
phase.add_boundary_value(PhaseRegionFlags::Front, {"R", "V", "t"}, X0);
phase.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.001, 1.0, 1.0);  // bound |u|
phase.add_boundary_value(PhaseRegionFlags::Back, {"R", "V"}, Xf);

// Minimum-time solve ...
phase.add_delta_time_objective(1.0);
auto flag_time = phase.solve_optimize();

// ... then re-solve for minimum power on the same phase.
phase.remove_state_objective(-1);
auto obj_expr = Arguments<3>().squared_norm() * 0.5;
phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"u"});
auto flag_power = phase.optimize();

// (min-fuel solve omitted here — same re-solve pattern with u.norm())
```
:::
::::

**Result:** all three objectives converge to feasible transfers that meet the
target orbit; the time-optimal solution thrusts at its bound throughout, while
the fuel-optimal solution exhibits a bang-off-bang thrust profile whose
switches line up with the zeros of the costate switching function.

Full source:
[`SimpleLowThrust.py`](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/SimpleLowThrust.py)
·
[`simple_low_thrust/main.cpp`](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/simple_low_thrust/main.cpp)

## Minimum time to climb (table data)

Minimum-time supersonic climb for a fighter aircraft, with aerodynamic
coefficients, atmospheric density/speed-of-sound, and engine thrust all
supplied as *interpolated tables*. The dynamics read those tables, the
trajectory is bounded by altitude/velocity/flight-path-angle path constraints,
and the mesh is refined adaptively to resolve the climb.

- **Features:** table-interpolated data (`InterpTable1D` / `InterpTable2D`),
  multiple path/state bounds, and adaptive mesh refinement with a configurable
  mesh-error estimator.

::::{tab-set}
:::{tab-item} Python
```python
rhoTab = vf.InterpTable1D(alts, rhos, kind="cubic")
sosTab = vf.InterpTable1D(alts, soss, kind="cubic")
CD0Tab = vf.InterpTable1D(AeroMach, CD0, kind="cubic")
ThrustTab = vf.InterpTable2D(ThrustMach, ThrustAlt, ThrustData, kind="cubic")


class AirPlane(oc.ODEBase):
    def __init__(self):
        XtU = oc.ODEArguments(4, 1)
        h, v, fpa, mass = XtU.x_vec().tolist()
        alpha = XtU.u_var(0)

        Mach = v / sosTab(h)
        Thrust = ThrustTab(Mach, h)
        CD = CD0Tab(Mach) + etaTab(Mach) * ClalphaTab(Mach) * (alpha**2)
        q = 0.5 * rhoTab(h) * (v**2)
        D, L = q * S * CD, q * S * ClalphaTab(Mach) * alpha
        r = h + Re

        hdot = v * vf.sin(fpa)
        vdot = (Thrust * vf.cos(alpha) - D) / mass - mu * vf.sin(fpa) / (r**2)
        fpadot = (Thrust * vf.sin(alpha) + L) / (mass * v) + vf.cos(fpa) * (
            v / r - mu / (v * (r**2)))
        mdot = -Thrust / vexhaust
        super().__init__(vf.stack([hdot, vdot, fpadot, mdot]), 4, 1, Vgroups=Vgroups)


phase = ode.phase("LGL3", Traj, 50)
phase.set_auto_scaling(True)
phase.add_lu_var_bound("Path", "h", hmin, hmax)
phase.add_lu_var_bound("Path", "alpha", alphamin, alphamax)
phase.add_delta_time_objective(1.0)
phase.set_adaptive_mesh(True)
phase.set_mesh_error_estimator(oc.MeshErrorEstimators.INTEGRATOR)
phase.set_mesh_tol(1.0e-7)
phase.optimize()
```
:::
:::{tab-item} C++
```cpp
auto rhoTab = std::make_shared<InterpTable1D>(alts, rhos, 0, InterpType::Cubic);
auto sosTab = std::make_shared<InterpTable1D>(alts, soss, 0, InterpType::Cubic);
auto CD0Tab = std::make_shared<InterpTable1D>(AeroMach, CD0_data, 0, InterpType::Cubic);
auto ThrustTab =
    std::make_shared<InterpTable2D>(ThrustMach, ThrustAlt, ThrustData, InterpType::Cubic);

auto ode = ODEBuilder(4, 1)
    .define([=](auto &args) {
        auto h = args.x_var(0);
        auto v = args.x_var(1);
        auto fpa = args.x_var(2);
        auto mass = args.x_var(3);
        auto alpha = args.u_var(0);
        auto rho = interp_scalar(rhoTab, h * Lstar) / Rhostar;
        auto Mach = v / (interp_scalar(sosTab, h * Lstar) / Vstar);
        auto Thrust = interp(ThrustTab, Mach, h * Lstar) / Fstar;
        auto CD = interp_scalar(CD0Tab, Mach) +
                  interp_scalar(etaTab, Mach) * interp_scalar(ClalphaTab, Mach) * (alpha * alpha);
        auto q = 0.5 * rho * v * v;
        auto D = q * S_nd * CD;
        // ... assemble hdot, vdot, fpadot, mdot ...
        return stack(hdot, vdot, fpadot, mdot);
    })
    .var_names({{"h", 0}, {"v", 1}, {"fpa", 2}, {"m", 3}, {"t", 4}, {"alpha", 5}})
    .build();

phase.add_lu_var_bound(PhaseRegionFlags::Path, "h", hmin_nd, hmax_nd);
phase.add_lu_var_bound(PhaseRegionFlags::Path, "alpha", alphamin, alphamax);
phase.add_delta_time_objective(1.0);
phase.set_adaptive_mesh(true);
phase.set_mesh_tol(1.0e-7);
phase.set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);
const auto flag = phase.solve_optimize();
```
:::
::::

**Result:** the aircraft reaches the target altitude in roughly **319 s**, with
the trajectory hugging the energy-height contours that characterize the classic
Bryson minimum-time climb.

The Python version splits the table definitions
([`MinimumTimeToClimbTables.py`](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/MinimumTimeToClimbTables.py))
from the solve
([`MinimumTimeToClimb.py`](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/MinimumTimeToClimb.py));
the C++ twin
([`min_time_climb_tables/main.cpp`](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/min_time_climb_tables/main.cpp))
keeps both in one file.

## More examples

These six are a curated cross-section. The
[`examples/`](https://github.com/GrantHecht/tycho/tree/main/examples)
directory holds the full set — over thirty Python scripts under
[`examples/python_examples/`](https://github.com/GrantHecht/tycho/tree/main/examples/python_examples)
and their C++ twins under
[`examples/cpp_examples/`](https://github.com/GrantHecht/tycho/tree/main/examples/cpp_examples) —
covering reentry, multi-spacecraft optimization, orbit continuation,
heteroclinic connections, obstacle avoidance, and more.
