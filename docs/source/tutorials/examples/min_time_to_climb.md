(tutorials-examples-min-time-to-climb)=
# Minimum time to climb

The classic supersonic-aircraft climb problem: drive an aircraft from low-speed,
near-sea-level flight to a high-altitude supersonic condition in minimum time.
The state is altitude, velocity, flight-path angle, and mass `(h, v, fpa, mass)`;
the control is the angle of attack `alpha`. The aerodynamics and propulsion are
supplied as *interpolated tables* (lift, drag, density, speed of sound, and a 2-D
thrust map), and the mesh is refined adaptively until the discretization error
falls below tolerance.

**Demonstrates:** interpolated table data (`InterpTable1D` / `InterpTable2D`)
embedded in a VectorFunction, and adaptive mesh refinement
(`set_adaptive_mesh`, integrator-based error estimator).

## Python

```{eval-rst}
.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   import tychopy as typy
   import tychopy.optimal_control as oc
   import tychopy.vector_functions as vf
   from tychopy.vector_functions import Arguments as Args

   # -------------------------------------------------------------------------
   # Aerodynamic / atmospheric table data (inlined from the original
   # MinimumTimeToClimbTables sibling module so this program is self-contained).
   # Data from ICLOCS' formulation of the Bryson minimum-time-to-climb problem.
   # -------------------------------------------------------------------------

   # Aerodynamic coefficients vs Mach number.
   AeroMach = [0, 0.4, 0.6, 0.75, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
   Clalpha = [3.44, 3.44, 3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44]
   CD0 = [0.013, 0.013, 0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035]
   eta = [0.54, 0.54, 0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93]

   # Density and speed-of-sound vs altitude (1976 US standard atmosphere).
   AtmosData = np.array(
       [
           [-2000, 1.478e00, 3.479e02], [0, 1.225e00, 3.403e02],
           [2000, 1.007e00, 3.325e02], [4000, 8.193e-01, 3.246e02],
           [6000, 6.601e-01, 3.165e02], [8000, 5.258e-01, 3.081e02],
           [10000, 4.135e-01, 2.995e02], [12000, 3.119e-01, 2.951e02],
           [14000, 2.279e-01, 2.951e02], [16000, 1.665e-01, 2.951e02],
           [18000, 1.216e-01, 2.951e02], [20000, 8.891e-02, 2.951e02],
           [22000, 6.451e-02, 2.964e02], [24000, 4.694e-02, 2.977e02],
           [26000, 3.426e-02, 2.991e02], [28000, 2.508e-02, 3.004e02],
           [30000, 1.841e-02, 3.017e02], [32000, 1.355e-02, 3.030e02],
           [34000, 9.887e-03, 3.065e02], [36000, 7.257e-03, 3.101e02],
           [38000, 5.366e-03, 3.137e02], [40000, 3.995e-03, 3.172e02],
           [42000, 2.995e-03, 3.207e02], [44000, 2.259e-03, 3.241e02],
           [46000, 1.714e-03, 3.275e02], [48000, 1.317e-03, 3.298e02],
           [50000, 1.027e-03, 3.298e02], [52000, 8.055e-04, 3.288e02],
           [54000, 6.389e-04, 3.254e02], [56000, 5.044e-04, 3.220e02],
           [58000, 3.962e-04, 3.186e02], [60000, 3.096e-04, 3.151e02],
           [62000, 2.407e-04, 3.115e02], [64000, 1.860e-04, 3.080e02],
           [66000, 1.429e-04, 3.044e02], [68000, 1.091e-04, 3.007e02],
           [70000, 8.281e-05, 2.971e02], [72000, 6.236e-05, 2.934e02],
           [74000, 4.637e-05, 2.907e02], [76000, 3.430e-05, 2.880e02],
           [78000, 2.523e-05, 2.853e02], [80000, 1.845e-05, 2.825e02],
           [82000, 1.341e-05, 2.797e02], [84000, 9.690e-06, 2.769e02],
           [86000, 6.955e-06, 2.741e02],
       ]
   ).T
   alts, rhos, soss = AtmosData[0], AtmosData[1], AtmosData[2]

   # Thrust vs Mach and altitude (padded with negative altitude data so the
   # table is never read outside its bounds during the solve).
   ThrustMach = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
   ThrustAlt = 304.8 * np.array([-0.5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 70])
   ThrustData = 4448.2 * np.array(
       [
           [24.2, 24.2, 24.0, 20.3, 17.3, 14.5, 12.2, 10.2, 5.7, 3.4, 0.1],
           [28.0, 28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, 6.5, 3.9, 0.2],
           [28.3, 28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, 0.4],
           [30.8, 30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, 0.8],
           [34.5, 34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1],
           [37.9, 37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4],
           [36.1, 36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7],
           [36.1, 36.1, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2],
           [36.1, 36.1, 35.2, 42.1, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9],
           [36.1, 36.1, 33.8, 45.7, 41.3, 39.8, 34.6, 31.1, 21.7, 13.3, 3.1],
       ]
   ).T

   # Build interpolation tables from the inlined data.
   rhoTab = vf.InterpTable1D(alts, rhos, kind="cubic")
   sosTab = vf.InterpTable1D(alts, soss, kind="cubic")
   ClalphaTab = vf.InterpTable1D(AeroMach, Clalpha, kind="cubic")
   etaTab = vf.InterpTable1D(AeroMach, eta, kind="cubic")
   CD0Tab = vf.InterpTable1D(AeroMach, CD0, kind="cubic")
   ThrustTab = vf.InterpTable2D(ThrustMach, ThrustAlt, ThrustData, kind="cubic")

   # -------------------------------------------------------------------------
   # Problem constants and dynamics.
   # -------------------------------------------------------------------------
   g0 = 9.80665
   Lstar = 10000.0
   Tstar = 250.0
   Mstar = 19050.864
   Vstar = Lstar / Tstar

   mu = 3.986012e14
   Re = 6378145.0
   S = 49.2386
   vexhaust = 1600 * g0


   class AirPlane(oc.ODEBase):
       def __init__(self):
           XtU = oc.ODEArguments(4, 1)
           # States: altitude, velocity, flight-path angle, mass.
           h, v, fpa, mass = XtU.x_vec().tolist()
           alpha = XtU.u_var(0)  # angle of attack (control)

           rho = rhoTab(h)
           sos = sosTab(h)
           Mach = v / sos
           CD0 = CD0Tab(Mach)
           Clalpha = ClalphaTab(Mach)
           eta = etaTab(Mach)
           Thrust = ThrustTab(Mach, h)

           CD = CD0 + eta * Clalpha * (alpha**2)
           CL = Clalpha * alpha
           q = 0.5 * rho * (v**2)
           D = q * S * CD
           L = q * S * CL
           r = h + Re

           hdot = v * vf.sin(fpa)
           vdot = (Thrust * vf.cos(alpha) - D) / mass - mu * vf.sin(fpa) / (r**2)
           fpadot = (Thrust * vf.sin(alpha) + L) / (mass * v) + vf.cos(fpa) * (
               v / r - mu / (v * (r**2))
           )
           mdot = -Thrust / vexhaust
           ode = vf.stack([hdot, vdot, fpadot, mdot])

           Vgroups = {}
           Vgroups[("h", "altitude")] = h
           Vgroups[("v", "velocity")] = v
           Vgroups["mass"] = mass
           Vgroups["fpa"] = fpa
           Vgroups[("alpha", "AoA")] = alpha
           Vgroups[("t", "time")] = XtU.t_var()
           super().__init__(ode, 4, 1, Vgroups=Vgroups)


   # Boundary conditions and bounds.
   ht0 = 0.010  # not exactly zero, to avoid conflict with the path constraint
   htf = 19994.88
   vt0 = 129.314
   vtf = 295.092
   fpat0 = 0
   fpatf = 0
   mass0 = 19050.864

   hmin, hmax = 0.0, 21000.0
   vmin, vmax = 5.0, 600.0
   fpamin, fpamax = -20 * np.pi / 180, 40 * np.pi / 180
   massmin = 16500.0
   alphamin, alphamax = -np.pi / 4, np.pi / 4
   tfig = 200

   ode = AirPlane()
   XtU0 = ode.make_input(h=ht0, v=vt0, fpa=fpat0, mass=mass0)
   XtUf = ode.make_input(h=htf, v=vtf, fpa=fpatf, mass=mass0, t=tfig)
   Traj = [XtU0 * (1 - t) + XtUf * t for t in np.linspace(0, 1, 100)]

   phase = ode.phase("LGL3", Traj, 50)
   phase.set_auto_scaling(True)
   phase.set_units(h=Lstar, v=Vstar, mass=Mstar, t=Tstar)
   phase.set_control_mode("HighestOrderSpline")
   phase.add_boundary_value("Front", range(0, 5), [ht0, vt0, fpat0, mass0, 0])
   phase.add_lu_var_bound("Path", "h", hmin, hmax)
   phase.add_lu_var_bound("Path", "v", vmin, vmax)
   phase.add_lu_var_bound("Path", "fpa", fpamin, fpamax)
   phase.add_lower_var_bound("Back", "mass", massmin)
   phase.add_lu_var_bound("Path", "alpha", alphamin, alphamax)
   phase.add_boundary_value("Back", ["h", "v", "fpa"], [htf, vtf, fpatf])
   phase.add_delta_time_objective(1.0)

   phase.optimizer.set_print_level(3)
   phase.set_num_partitions(8, 8)

   # Adaptive mesh refinement, driven by the integrator-based error estimator.
   phase.set_adaptive_mesh(True)
   phase.set_mesh_error_estimator(oc.MeshErrorEstimators.INTEGRATOR)
   phase.set_mesh_tol(1.0e-7)
   phase.optimize()

   Traj = phase.return_traj()
   print("Minimum time to climb: {0:.2f} s".format(Traj[-1][4]))

   T = np.array(Traj).T
   fig, axs = plt.subplots(2, 2, figsize=(11, 7.0))
   axs[0, 0].plot(T[4], T[0])
   axs[0, 0].set(xlabel="t (s)", ylabel="h (m)", title="Altitude")
   axs[0, 1].plot(T[4], T[1])
   axs[0, 1].set(xlabel="t (s)", ylabel="v (m/s)", title="Velocity")
   axs[1, 0].plot(T[4], T[2] * 180 / np.pi)
   axs[1, 0].set(xlabel="t (s)", ylabel="fpa (deg)", title="Flight-path angle")
   axs[1, 1].plot(T[1], T[0])
   axs[1, 1].set(xlabel="v (m/s)", ylabel="h (m)", title="Climb in v-h plane")
   for ax in axs.ravel():
       ax.grid(True)
   fig.suptitle("Minimum time to climb")
   fig.tight_layout()
```

**Result:** the optimizer finds a minimum-time-to-climb of about **319 s**,
with the adaptive mesh refining until the integrator-based error estimate meets
the `1e-7` tolerance — the classic energy-climb profile that dives to build
airspeed before the final supersonic zoom climb.

## C++

```{eval-rst}
.. literalinclude:: ../../../../examples/cpp_examples/builder/min_time_climb_tables/main.cpp
   :language: cpp
```

Full sources: [Python](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/MinimumTimeToClimb.py) · [C++](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/min_time_climb_tables/main.cpp)
