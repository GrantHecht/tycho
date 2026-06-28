(tutorials-examples-goddard-rocket)=
# Goddard rocket (singular arc)

Maximize the final altitude of a vertically launched rocket subject to drag and
a thrust budget. The state is altitude, velocity, and mass `(h, v, m)`; the
control is the throttle `u`. The classic single-phase formulation has a
*singular arc*, so this program uses the multi-phase formulation: a full-thrust
phase, a singular phase whose control is defined by a nonlinear equality *path
constraint*, and a coast phase — linked for state continuity.

**Demonstrates:** multi-phase linkage (`add_forward_link_equal_con`), a
nonlinear equality path constraint (`add_equal_con`) defining the singular arc,
a value objective (`add_value_objective`), auto-scaling with physical units, and
an integrator-with-control initial guess.

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

   # Physical constants (Betts, "Practical Methods for OC", Section 4.14).
   g0 = 32.2
   Lstar = 10000.0  # feet
   Tstar = 60.0     # sec
   Mstar = 1.0      # slugs
   Vstar = Lstar / Tstar

   h_ref = 23800.0
   g = g0
   Tmag = 200.0
   c = 1580.94
   sigma = 5.4915e-5

   m0 = 3.0
   mf = 1.0


   class GoddardRocket(oc.ODEBase):
       def __init__(self, sigma, c, h_ref, Tmag, g):
           XtU = oc.ODEArguments(3, 1)
           h, v, m = XtU.x_vec().tolist()
           u = XtU.u_var(0)

           hdot = v
           vdot = (u * Tmag - sigma * (v**2) * vf.exp(-h / h_ref)) / m - g
           mdot = -u * Tmag / c
           ode = vf.stack(hdot, vdot, mdot)

           Vgroups = {}
           Vgroups[("h", "altitude")] = h
           Vgroups[("v", "velocity")] = v
           Vgroups[("m", "mass")] = m
           Vgroups[("t", "time")] = XtU.t_var()
           Vgroups["u"] = u
           super().__init__(ode, 3, 1, Vgroups=Vgroups)


   def PathCon(sigma, c, h_ref, Tmag, g):
       # Betts eq. 4.188: the equality that defines the singular-arc control.
       h, v, m, u = Args(4).tolist()
       t1 = (u * Tmag - sigma * (v**2) * vf.exp(-h / h_ref)) - g * m
       t2 = (m * g / (1 + 4 * (c / v) + 2 * (c / v) ** 2)) * (
           c * c * (1 + v / c) / (h_ref * g) - 1.0 - 2.0 * c / v
       )
       return t1 - t2


   ode = GoddardRocket(sigma, c, h_ref, Tmag, g)
   units = ode.make_units(h=Lstar, v=Vstar, m=Mstar, t=Tstar)


   # Initial guess: integrate full thrust until burnout, coast until v < 0.
   def Ulaw():
       m = Args(1)[0]
       return vf.ifelse(m > mf, 1, 0)


   def StopFunc(x):
       return x[1] < 0


   integ = ode.integrator(0.01, Ulaw(), "m")
   X0 = ode.make_input(h=0, v=0, m=m0, u=1)
   TrajIG = integ.integrate_dense(X0, 60, 1000, StopFunc)

   # Split the guess into thrust / singular / coast arcs.
   n = int(len(TrajIG) / 3)
   TrajIG1 = TrajIG[0:n]
   TrajIG2 = TrajIG[n : 2 * n]
   TrajIG3 = TrajIG[2 * n : -1]

   # Phase 1 - full thrust (u = 1).
   phase1 = ode.phase("LGL3", TrajIG1, 32)
   phase1.add_boundary_value("Front", ["h", "v", "m", "t"], TrajIG[0][0:4])
   phase1.add_boundary_value("Path", "u", 1.0)

   # Phase 2 - singular arc: control defined by the equality path constraint.
   phase2 = ode.phase("LGL3", TrajIG2, 32)
   phase2.set_control_mode("NoSpline")  # PathCon makes control splines redundant
   phase2.add_lu_var_bound("Path", "u", 0.0, 1.0, 1.0)
   phase2.add_equal_con("Path", PathCon(sigma, c, h_ref, Tmag, g), ["h", "v", "m", "u"])

   # Phase 3 - coast (u = 0), maximize final altitude.
   phase3 = ode.phase("LGL3", TrajIG3, 32)
   phase3.add_boundary_value("Path", "u", 0)
   phase3.add_boundary_value("Back", ["v", "m"], [0, mf])
   phase3.add_value_objective("Back", "h", -1.0)

   ocp = oc.OptimalControlProblem()
   ocp.add_phase(phase1)
   ocp.add_phase(phase2)
   ocp.add_phase(phase3)
   # chains every consecutive pair (phase1->phase2->phase3) for [h, v, m, t] continuity
   ocp.add_forward_link_equal_con(phase1, phase3, ["h", "v", "m", "t"])

   phase1.add_lower_delta_time_bound(0)
   phase2.add_lower_delta_time_bound(0)
   phase3.add_lower_delta_time_bound(0)

   phase1.set_units(units)
   phase2.set_units(units)
   phase3.set_units(units)
   ocp.set_auto_scaling(True, True)
   ocp.set_num_partitions(8, 8)
   ocp.optimizer.set_print_level(3)
   ocp.optimize()

   Traj = phase1.return_traj() + phase2.return_traj() + phase3.return_traj()
   T = np.array(Traj).T

   fig, axs = plt.subplots(4, 1, figsize=(8.0, 9.0), sharex=True)
   axs[0].plot(T[3], T[0])
   axs[1].plot(T[3], T[1])
   axs[2].plot(T[3], T[2])
   axs[3].plot(T[3], T[4] * Tmag)
   axs[0].set_ylabel("h (ft)")
   axs[1].set_ylabel("v (ft/s)")
   axs[2].set_ylabel("m (slug)")
   axs[3].set_ylabel("T (lb)")
   axs[3].set_xlabel("t (s)")
   for ax in axs:
       ax.grid(True)
   axs[0].set_title("Goddard rocket: bang-singular-coast solution")

   fig.tight_layout()
```

**Result:** the solver recovers the bang–singular–coast structure — a
full-thrust burn, a smoothly varying singular arc that follows Betts'
constraint, and a ballistic coast to a maximum altitude of roughly
**18,700 ft** at about **43 s**.

## C++

```{eval-rst}
.. literalinclude:: ../../../../examples/cpp_examples/builder/goddard_rocket/main.cpp
   :language: cpp
```

Full sources: [Python](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/GoddardRocket.py) · [C++](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/goddard_rocket/main.cpp)
