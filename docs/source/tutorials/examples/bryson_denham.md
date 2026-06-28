(tutorials-examples-bryson-denham)=
# Bryson–Denham problem

The classic state-path-constrained double integrator: minimize the control
effort $\tfrac12\int u^2\,dt$ that drives a unit mass from $(x, v) = (0, 1)$ to
$(0, -1)$ over a fixed unit time, subject to a hard upper bound on the
displacement, $x \le 1/9$. The constrained arc rides the bound exactly — the
hallmark of an active state path constraint.

**Demonstrates:** a state path constraint (`add_upper_var_bound` on `Path`)
together with an integral control-effort objective (`add_integral_objective`).

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


   class Model(oc.ODEBase):
       def __init__(self):
           Xvars, Uvars = 2, 1
           args = oc.ODEArguments(Xvars, Uvars)
           x = args.x_vec()[0]
           v = args.x_vec()[1]
           u = args.u_vec()[0]
           ode = vf.stack([v, u])
           super().__init__(ode, Xvars, Uvars)


   n = 100
   ts = np.linspace(0, 1, n)
   vs = np.linspace(1, -1, n)
   # Initial guess; each node is [x, v, t, u].
   IG = [[0.0, v, t, 0.0] for t, v in zip(ts, vs)]

   ode = Model()

   phase = ode.phase("LGL5", IG, 32)
   phase.add_boundary_value("Front", range(0, 3), [0, 1, 0])
   phase.add_upper_var_bound("Path", 0, 1 / 9)
   # minimize integral of u^2/2; [3] is the index of the control u in the node layout [x, v, t, u]
   phase.add_integral_objective((Args(1)[0] ** 2) / 2, [3])
   phase.add_boundary_value("Back", range(0, 3), [0, -1, 1])
   phase.optimizer.set_opt_ls_mode("L1")
   phase.optimizer.set_kkt_tol(1.0e-10)
   phase.optimizer.set_print_level(3)
   phase.set_num_partitions(1, 1)
   phase.optimize()

   TT = np.array(phase.return_traj()).T

   fig, axs = plt.subplots(3, 1, figsize=(7.5, 7.0), sharex=True)
   axs[0].plot(TT[2], TT[0])
   axs[1].plot(TT[2], TT[1])
   axs[2].plot(TT[2], TT[3])
   axs[0].axhline(1 / 9, color=_brand.STEEL_DARK, ls="--", lw=0.8, label="state bound")
   axs[0].set_ylabel("x")
   axs[1].set_ylabel("v")
   axs[2].set_ylabel("u")
   axs[2].set_xlabel("t")
   axs[0].legend()
   for i in range(3):
       axs[i].grid(True)
   axs[0].set_title("Bryson-Denham: state-constrained double integrator")

   fig.tight_layout()
```

**Result:** the optimizer finds the optimal solution — the displacement `x`
rises to touch its bound `1/9 ≈ 0.1111`, rides along it during the constrained
arc, then returns, while the velocity reverses from `+1` to `-1` exactly as
required at the final time.

## C++

```{eval-rst}
.. literalinclude:: ../../../../examples/cpp_examples/builder/bryson_denham/main.cpp
   :language: cpp
```

Full sources: [Python](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/BrysonDenham.py) · [C++](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/bryson_denham/main.cpp)
