(tutorials-examples-multi-phase-cannon)=
# Multi-phase cannonball

Find the cannonball radius that maximizes down range. Assuming a fixed launch
energy and constant material density, a larger ball carries more momentum but
also more drag — so an optimal radius exists. The state is speed, flight-path
angle, altitude, and range `(v, gamma, h, r)`; the radius enters as an *ODE
parameter*. The flight is split into an ascent phase (launch to apogee) and a
descent phase (apogee to ground), linked for state continuity and to share the
single radius parameter.

**Demonstrates:** ODE parameters (`p_var`), multi-phase parameter linking
(`add_param_link_equal_con`), and an event-detection initial guess built by
integrating to apogee and then to ground contact.

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

   # Physical constants (Dymos multi-phase cannonball, exponential atmosphere).
   g0 = 9.81
   Lstar = 1000.0  # m
   Tstar = 60.0    # s
   Mstar = 10.0    # kg
   Vstar = Lstar / Tstar

   CD = 0.5
   RhoAir = 1.225
   RhoIron = 7870.0
   h_scale = 8.44e3
   E0 = 400000.0
   g = g0


   def MFunc(rad, RhoIron):
       return (4 / 3) * (np.pi * RhoIron) * (rad**3)


   def SFunc(rad):
       return np.pi * (rad**2)


   class Cannon(oc.ODEBase):
       def __init__(self, CD, RhoAir, RhoIron, h_scale, g):
           # 4 states (v, gamma, h, r), 0 controls, 1 ODE parameter (rad).
           args = oc.ODEArguments(4, 0, 1)

           v = args.x_var(0)
           gamma = args.x_var(1)
           h = args.x_var(2)
           r = args.x_var(3)
           rad = args.p_var(0)

           S = SFunc(rad)
           M = MFunc(rad, RhoIron)
           rho = RhoAir * vf.exp(-h / h_scale)
           D = (0.5 * CD) * rho * (v**2) * S

           vdot = -D / M - g * vf.sin(gamma)
           gammadot = -g * vf.cos(gamma) / v
           hdot = v * vf.sin(gamma)
           rdot = v * vf.cos(gamma)
           ode = vf.stack([vdot, gammadot, hdot, rdot])

           Vgroups = {}
           Vgroups["v"] = v
           Vgroups["gamma"] = gamma
           Vgroups["h"] = h
           Vgroups[("r", "range")] = r
           Vgroups["rad"] = rad
           Vgroups["t"] = args.t_var()
           super().__init__(ode, 4, 0, 1, Vgroups=Vgroups)

           # Event functions for the integrator-based initial guess.
           self.apogee_event = v * vf.sin(gamma)
           self.ground_contact_event = h


   def EFunc():
       # Launch-energy upper bound: 1/2 M v^2 <= E0 (active at equality for max range).
       v, rad = Args(2).tolist()
       M = MFunc(rad, RhoIron)
       return 0.5 * M * (v**2) - E0


   ode = Cannon(CD, RhoAir, RhoIron, h_scale, g)

   # Initial guess: integrate from launch to apogee (ascent), then apogee to
   # ground contact (descent), using event detection to find each arc.
   rad0 = 0.1
   h0 = 100.0
   r0 = 0.0
   gamma0 = np.deg2rad(45)
   m0 = MFunc(rad0, RhoIron)
   v0 = np.sqrt(2 * E0 / m0) * 0.99

   integ = ode.integrator(0.01)
   integ.set_abs_tol(1.0e-12)
   XtP0 = ode.make_input(v=v0, gamma=gamma0, h=h0, r=r0, rad=rad0)
   AscentIG = integ.integrate_dense(XtP0, 60, [(ode.apogee_event, 0, 1)])[0]
   DescentIG = integ.integrate_dense(
       AscentIG[-1], AscentIG[-1][4] + 1000, [(ode.ground_contact_event, 0, 1)]
   )[0]

   tmode = "LGL5"
   nsegs = 128
   units = ode.make_units(v=Vstar, h=Lstar, r=Lstar, rad=Lstar)

   # Ascent phase: launch energy fixed, climb to apogee (gamma = 0 at back).
   aphase = ode.phase(tmode, AscentIG, nsegs)
   aphase.set_auto_scaling(True)
   aphase.set_units(units)
   aphase.add_lower_var_bound("ODEParams", "rad", 0.0)
   aphase.add_lower_var_bound("Front", "gamma", 0.0)
   aphase.add_boundary_value("Front", ["h", "r", "t"], [h0, r0, 0])
   aphase.add_inequal_con("Front", EFunc(), ["v"], ["rad"], [])
   aphase.add_boundary_value("Back", "gamma", 0.0)

   # Descent phase: fall back to the ground, maximizing down range.
   dphase = ode.phase(tmode, DescentIG, nsegs)
   dphase.set_auto_scaling(True)
   dphase.set_units(units)
   dphase.add_boundary_value("Back", "h", 0.0)
   dphase.add_value_objective("Back", "r", -1.0)

   ocp = oc.OptimalControlProblem()
   ocp.set_auto_scaling(True)
   ocp.add_phase(aphase)
   ocp.add_phase(dphase)
   # State continuity across the apogee, plus shared cannonball radius parameter.
   ocp.add_forward_link_equal_con(aphase, dphase, range(0, 5))
   ocp.add_param_link_equal_con(aphase, dphase, "ODEParams", "rad")
   ocp.optimizer.set_print_level(3)
   ocp.optimize()

   Ascent = aphase.return_traj()
   Descent = dphase.return_traj()
   ropt, radopt = ode.get_vars(["r", "rad"], Descent[-1])
   print("Optimized Range:", ropt, "m")
   print("Optimized Radius:", radopt, "m")

   AT = np.array(Ascent).T
   DT = np.array(Descent).T
   fig, ax = plt.subplots(figsize=(8.0, 5.0))
   ax.plot(AT[3], AT[2], color="r", label="Ascent")
   ax.plot(DT[3], DT[2], color="b", label="Descent")
   ax.set_xlabel("Down range (m)")
   ax.set_ylabel("Altitude (m)")
   ax.set_title("Multi-phase cannonball: max-range trajectory")
   ax.grid(True)
   ax.legend()
   fig.tight_layout()
```

**Result:** the optimizer settles on a cannonball radius of about **0.042 m**,
giving a maximum down range of roughly **3280 m** — the ascent and descent
phases meet continuously at the apogee, sharing the single optimized radius.

## C++

```{eval-rst}
.. literalinclude:: ../../../../examples/cpp_examples/builder/multi_phase_cannon/main.cpp
   :language: cpp
```

Full sources: [Python](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/MultiPhaseCannon.py) · [C++](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/multi_phase_cannon/main.cpp)
