(tutorials-examples-brachistochrone)=
# Brachistochrone

The canonical first optimal-control problem: a bead slides without friction from
a start point to an end point under gravity, steered only by the wire angle
`theta`. The state is position and speed `(x, y, v)`, the control is `theta`, and
the objective is to minimize the *free* final time. The known minimum for this
geometry is about 1.8013 s.

**Demonstrates:** single-phase basics, free final time
(`add_delta_time_objective`), a control path bound, and front/back boundary
values.

## Python

```{eval-rst}
.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   import tychopy as typy

   vf = typy.vector_functions
   oc = typy.optimal_control


   class Brachistochrone(oc.ODEBase):
       def __init__(self, g):
           XVars, UVars = 3, 1
           XtU = oc.ODEArguments(XVars, UVars)
           x, y, v = XtU.x_vec().tolist()
           theta = XtU.u_var(0)

           xdot = vf.sin(theta) * v
           ydot = -1.0 * vf.cos(theta) * v
           vdot = g * vf.cos(theta)
           ode = vf.stack([xdot, ydot, vdot])
           super().__init__(ode, XVars, UVars)


   g = 9.81
   ode = Brachistochrone(g)

   x0, y0, v0, theta0 = 0.0, 10.0, 0.0, 1.0
   xf, yf = 10.0, 5.0
   tf = 1.0

   # Straight-line initial guess; each node is [x, y, v, t, theta].
   ts = np.linspace(0.0, tf, 100)
   TrajIG = []
   for t in ts:
       X = np.zeros(5)
       X[0] = x0 + (xf - x0) * t / tf
       X[1] = y0 + (yf - y0) * t / tf
       X[2] = g * t * np.cos(theta0)
       X[3] = t
       X[4] = theta0
       TrajIG.append(X)

   phase = ode.phase("LGL3", TrajIG, 32)
   phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0.0])
   phase.add_lu_var_bound("Path", 4, -0.1, 2.00)
   phase.add_boundary_value("Back", [0, 1], [xf, yf])
   phase.add_delta_time_objective(1.0)
   phase.optimizer.set_print_level(3)
   phase.optimize()

   Traj = np.array(phase.return_traj()).T

   fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.0))
   ax0.plot(Traj[0], Traj[1], color="C0")
   ax0.set(xlabel="x", ylabel="y", title="Brachistochrone path")
   ax0.grid(True)

   ax1.plot(Traj[3], Traj[4], color="C1")
   ax1.set(xlabel="t [s]", ylabel="theta [rad]", title="Wire-angle control")
   ax1.grid(True)

   fig.tight_layout()
```

**Result:** the optimal descent time is about **1.8013 s** — the known
brachistochrone minimum for this geometry.

## C++

```{eval-rst}
.. literalinclude:: ../../../../examples/cpp_examples/builder/brachistochrone/main.cpp
   :language: cpp
```

Full sources: [Python](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/Brachistochrone.py) · [C++](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/brachistochrone/main.cpp)
