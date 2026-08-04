(tutorials-examples-van-der-pol)=
# Van der Pol oscillator

A classic nonlinear system: minimize the integral of `x0^2 + x1^2 + u^2` while
driving the Van der Pol oscillator from its limit cycle to the origin over a
*fixed* final time. The control is bounded and evaluated as a block-constant
profile.

**Demonstrates:** an integral objective (`add_integral_objective`), the
block-constant control mode (`set_control_mode`), a control path bound
(`add_lu_var_bound`), a fixed final time, and solver tolerance and partition
tuning.

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


   tf = 10.0
   # Initial guess; each node is [x0, x1, t, u].
   TrajIG = [[0, 0, t, 0] for t in np.linspace(0, tf, 100)]

   ode = VanderPol()

   phase = ode.phase("LGL3", TrajIG, 256)
   phase.set_control_mode("BlockConstant")
   phase.add_boundary_value("Front", range(0, 3), [0, 1, 0])
   phase.add_lu_var_bound("Path", 3, -0.75, 1.0)
   phase.add_integral_objective(Args(3).squared_norm(), [0, 1, 3])
   phase.add_boundary_value("Back", [0, 1, 2], [0.0, 0.0, tf])
   phase.optimizer.set_print_level(3)
   phase.set_num_partitions(8, 8)
   phase.optimizer.set_tols(1.0e-8, 1.0e-8, 1.0e-8)
   phase.optimize()

   TT = np.array(phase.return_traj()).T

   fig, ax = plt.subplots(figsize=(7.5, 4.5))
   ax.plot(TT[2], TT[0], label=r"$x_0$")
   ax.plot(TT[2], TT[1], label=r"$x_1$")
   ax.plot(TT[2], TT[3], label=r"$u$")
   ax.set_xlabel("t")
   ax.legend()
   ax.grid(True)
   ax.set_title("Van der Pol oscillator: driving the limit cycle to the origin")

   fig.tight_layout()
```

**Result:** the optimizer drives both states to the origin at `t = tf` while
minimizing the running cost; the control briefly goes negative, then rides
its upper bound around `t ≈ 1.2` before relaxing to zero as the states
settle — "Optimal Solution Found" in 14 iterations, objective ≈ 2.8737.

## C++

```{eval-rst}
.. literalinclude:: ../../../../examples/cpp_examples/builder/van_der_pol/main.cpp
   :language: cpp
```

Full sources: [Python](https://github.com/GrantHecht/tycho/blob/main/examples/python_examples/VanDerPol.py) · [C++](https://github.com/GrantHecht/tycho/blob/main/examples/cpp_examples/builder/van_der_pol/main.cpp)
