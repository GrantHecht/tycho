"""Swing + phase-portrait figure for the "Propagating an ODE" tutorial.

Rendered by the matplotlib ``plot`` directive (file-based form, so the image is
named after this script and cannot collide with the page's other figure). This
is a build asset, not a documentation page.
"""

import matplotlib.pyplot as plt
import numpy as np

import _brand  # sibling module; the plot directive puts _plots/ on sys.path
from tychopy import optimal_control as oc
from tychopy import vector_functions as vf

_brand.apply()


class Pendulum(oc.ODEBase):
    def __init__(self, g, L):
        args = oc.ODEArguments(2, 0)
        theta, omega = args.x_vec().tolist()
        ode = vf.stack([omega, -(g / L) * vf.sin(theta)])
        super().__init__(ode, 2, 0)


g, L = 9.81, 1.0
ode = Pendulum(g, L)
integ = ode.integrator(0.01)
x0 = np.array([1.0, 0.0, 0.0])
T = np.array(integ.integrate_dense(x0, 5.0, 400))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
ax1.plot(T[:, 2], T[:, 0], color=_brand.AMBER, lw=2)
ax1.set(xlabel="time [s]", ylabel=r"angle $\theta$ [rad]", title="Pendulum swing")
ax1.grid(True)

ax2.plot(T[:, 0], T[:, 1], color=_brand.STEEL, lw=2)
ax2.set(xlabel=r"$\theta$ [rad]", ylabel=r"$\omega$ [rad/s]", title="Phase portrait")
ax2.grid(True)

fig.tight_layout()
