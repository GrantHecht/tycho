"""Low-thrust spiral-transfer figure for the documentation landing page.

Rendered by the matplotlib ``plot`` directive (file-based form, so the image is
named after this script and cannot collide with other figures). This is a build
asset, not a documentation page. It solves the same time-optimal orbit-raising
problem shown in the landing-page snippet and the worked example.
"""

import matplotlib.pyplot as plt
import numpy as np

import _brand  # sibling module; the plot directive puts _plots/ on sys.path
import tychopy as typy

_brand.apply()

vf = typy.vector_functions
oc = typy.optimal_control
Args = vf.Arguments


class LowThrust(oc.ODEBase):
    def __init__(self, mu=1.0, acc=0.02):
        args = oc.ODEArguments(6, 3)  # state [r, v] (6), control u (3)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        accel = r.normalized_power3() * (-mu) + u * acc  # gravity + thrust
        super().__init__(vf.stack([v, accel]), 6, 3)


mu = 1.0
ode = LowThrust(mu)

# Circular start orbit (r = 1) to circular target orbit (r = 2), in-plane.
X0 = np.zeros(7)
X0[0], X0[4] = 1.0, np.sqrt(mu / 1.0)
Xf = np.zeros(6)
Xf[0], Xf[4] = 2.0, np.sqrt(mu / 2.0)

# Initial guess: integrate forward under a fixed tangential thrust direction.
XIG = np.zeros(10)
XIG[:7] = X0
integ = ode.integrator(0.01, Args(3).normalized() * 0.8, [3, 4, 5])
guess = integ.integrate_dense(XIG, 6.4 * np.pi, 100)

phase = ode.phase("LGL3", guess, 256)
phase.add_boundary_value("Front", range(0, 7), X0)
phase.add_lu_norm_bound("Path", [7, 8, 9], 0.001, 1, 1.0)
phase.add_boundary_value("Back", range(0, 6), Xf)
phase.optimizer.set_print_level(3)
phase.add_delta_time_objective(1.0)
phase.optimize()

T = np.array(phase.return_traj()).T
G = np.array(guess).T

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.4))

# Reference orbits + the optimized spiral.
ang = np.linspace(0, 2 * np.pi, 200)
for rad in (1.0, 2.0):
    ax0.plot(
        rad * np.cos(ang), rad * np.sin(ang),
        color=_brand.FAINT, lw=1, alpha=0.6, zorder=1,
    )
ax0.plot(G[0], G[1], color=_brand.STEEL, ls="--", lw=1, label="initial guess", zorder=2)
ax0.plot(T[0], T[1], color=_brand.AMBER, lw=2.2, label="time-optimal", zorder=3)
ax0.scatter([X0[0]], [X0[1]], color=_brand.STEEL_DARK, zorder=5, label="start")
ax0.scatter([Xf[0]], [Xf[1]], color=_brand.AMBER_BRIGHT, edgecolor=_brand.AMBER,
            zorder=5, label="target")
ax0.set(xlabel="x", ylabel="y", title="Low-thrust spiral transfer")
ax0.axis("equal")
ax0.grid(True)
ax0.legend(loc="upper left", fontsize=9)

umag = np.sqrt(T[7] ** 2 + T[8] ** 2 + T[9] ** 2)
ax1.plot(T[6], umag, color=_brand.AMBER, lw=2.2)
ax1.set(
    xlabel="time (nondimensional)",
    ylabel="thrust magnitude  |u|",
    title="Time-optimal control",
    ylim=(0, 1.1),
)
ax1.grid(True, alpha=0.3)

fig.tight_layout()
