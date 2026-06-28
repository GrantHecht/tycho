(tutorials-examples-simple-low-thrust)=
# Simple low-thrust transfer

A low-thrust spacecraft spirals from a circular orbit of radius 1 to a circular
orbit of radius 2 in minimum time, in normalized two-body dynamics. The state is position and
velocity `(r, v)` in three dimensions; the control is a thrust-direction vector
`u` whose *magnitude* is bounded by a vector-norm path constraint. This is the
canonical low-thrust orbit-raising problem.

**Demonstrates:** astrodynamics with a normalized two-body model
(`normalized_power3`), a vector-norm control bound (`add_lu_norm_bound`), and a
norm-bounded integrator initial guess.

## Python

```{eval-rst}
.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   import tychopy as typy

   vf = typy.vector_functions
   oc = typy.optimal_control
   Args = vf.Arguments


   class LTModel(oc.ODEBase):
       def __init__(self, mu, ltacc):
           Xvars = 6  # position (3) + velocity (3)
           Uvars = 3  # thrust-direction control
           args = oc.ODEArguments(Xvars, Uvars)
           r = args.head3()
           v = args.segment3(3)
           u = args.tail3()
           g = r.normalized_power3() * (-mu)  # two-body gravity
           thrust = u * ltacc
           acc = g + thrust
           ode = vf.stack([v, acc])
           super().__init__(ode, Xvars, Uvars)

       # massobj / powerobj: the min-mass and min-power cost functions used by
       # the full example; only the min-time solve is shown below.
       class massobj(vf.ScalarFunction):
           def __init__(self, scale):
               u = Args(3)
               super().__init__(u.norm() * scale)

       class powerobj(vf.ScalarFunction):
           def __init__(self, scale):
               u = Args(3)
               super().__init__(u.norm().squared() * scale)


   mu = 1.0
   acc = 0.02
   ode = LTModel(mu, acc)

   # Circular start orbit (r0 = 1) to circular target orbit (rf = 2), in-plane.
   r0 = 1.0
   v0 = np.sqrt(mu / r0)
   rf = 2.0
   vF = np.sqrt(mu / rf)

   X0 = np.zeros(7)
   X0[0] = r0
   X0[4] = v0

   Xf = np.zeros(6)
   Xf[0] = rf
   Xf[4] = vF

   # Initial guess: integrate forward with a fixed-magnitude tangential thrust.
   XIG = np.zeros(10)
   XIG[0:7] = X0
   integ = ode.integrator(0.01, Args(3).normalized() * 0.8, [3, 4, 5])
   TrajIG = integ.integrate_dense(XIG, 6.4 * np.pi, 100)

   phase = ode.phase("LGL3", TrajIG, 256)
   phase.add_boundary_value("Front", range(0, 7), X0)
   # Vector-norm bound on the control: 0.001 <= |u| <= 1.
   phase.add_lu_norm_bound("Path", [7, 8, 9], 0.001, 1, 1.0)
   phase.add_boundary_value("Back", range(0, 6), Xf[0:6])

   phase.optimizer.set_print_level(3)
   phase.optimizer.set_bound_fraction(0.995)
   phase.optimizer.set_opt_ls_mode("L1")
   phase.optimizer.set_max_ls_iters(2)
   phase.optimizer.set_delta_h(1.0e-6)

   # Minimum-time transfer. (The full example also runs minimum-power and
   # minimum-mass objectives on the same phase via powerobj / massobj; here we
   # show only the time-optimal solution.)
   phase.add_delta_time_objective(1.0)
   phase.optimize()

   TimeOptimal = phase.return_traj()
   print("Transfer time:", TimeOptimal[-1][6])

   T = np.array(TimeOptimal).T
   IG = np.array(TrajIG).T

   fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
   ax0.plot(IG[0], IG[1], color=_brand.STEEL, ls="--", label="initial guess")
   ax0.plot(T[0], T[1], color=_brand.AMBER, label="time-optimal")
   ax0.scatter([X0[0]], [X0[1]], color=_brand.STEEL_DARK, zorder=5, label="start")
   ax0.scatter([Xf[0]], [Xf[1]], color=_brand.AMBER_BRIGHT, zorder=5, label="target")
   ax0.set(xlabel="x", ylabel="y", title="Low-thrust spiral transfer")
   ax0.axis("equal")
   ax0.grid(True)
   ax0.legend()

   umag = (T[7] ** 2 + T[8] ** 2 + T[9] ** 2) ** 0.5
   ax1.plot(T[6], umag, color=_brand.AMBER)
   ax1.set(xlabel="t (ND)", ylabel="|u|", title="Control magnitude")
   ax1.grid(True)

   fig.tight_layout()
```

**Result:** the time-optimal spiral reaches the target orbit in about **18.26**
nondimensional time units, with the control riding its upper magnitude bound
(`|u| = 1`) for essentially the entire transfer — the expected bang structure
for minimum-time low-thrust.

## C++

```{eval-rst}
.. literalinclude:: ../../../../examples/cpp_examples/builder/simple_low_thrust/main.cpp
   :language: cpp
```

Full sources: [Python](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/SimpleLowThrust.py) · [C++](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/simple_low_thrust/main.cpp)
