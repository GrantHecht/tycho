"""Energy-drift figure for the "Propagating an ODE" tutorial.

Rendered by the matplotlib ``plot`` directive (file-based form, so the image is
named after this script and cannot collide with the page's other figure). This
is a build asset, not a documentation page.
"""

import matplotlib.pyplot as plt
import numpy as np

from tychopy import optimal_control as oc
from tychopy import vector_functions as vf


class Pendulum(oc.ODEBase):
    def __init__(self, g, L):
        args = oc.ODEArguments(2, 0)
        theta, omega = args.x_vec().tolist()
        super().__init__(vf.stack([omega, -(g / L) * vf.sin(theta)]), 2, 0)


g, L = 9.81, 1.0
ode = Pendulum(g, L)
integ = ode.integrator(0.01)
x0 = np.array([1.0, 0.0, 0.0])
dense = np.array(integ.integrate_dense(x0, 5.0))  # one row per accepted step

E = 0.5 * dense[:, 1] ** 2 + (g / L) * (1.0 - np.cos(dense[:, 0]))
drift = np.abs(E - E[0])

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.semilogy(dense[1:, 2], drift[1:])
ax.axhline(1e-9, color="C3", ls="--", lw=1, label=r"$10^{-9}$ reference")
ax.set(
    xlabel="time [s]",
    ylabel=r"energy error $|E(t) - E_0|$",
    title="Energy drift over the propagation",
)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
