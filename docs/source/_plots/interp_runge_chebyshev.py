"""Runge function: equispaced polynomial vs Chebyshev interpolation.

Rendered by the matplotlib ``plot`` directive (file-based form, so the image is
named after this script and cannot collide with other figures on the page). This
is a build asset, not a documentation page.
"""

import matplotlib.pyplot as plt
import numpy as np

import _brand  # sibling module; the plot directive puts _plots/ on sys.path
from tychopy import vector_functions as vf

_brand.apply()

ORDER = 18


def runge(x):
    return 1.0 / (1.0 + 25.0 * x * x)


x_dense = np.linspace(-1.0, 1.0, 600)
y_true = runge(x_dense)

# Equispaced polynomial interpolant via numpy (the "bad" baseline).
x_equi = np.linspace(-1.0, 1.0, ORDER + 1)
y_equi = runge(x_equi)
coeffs = np.polyfit(x_equi, y_equi, ORDER)
y_poly = np.polyval(coeffs, x_dense)

# Chebyshev fit at the same order.
cheb_table = vf.cheb_from_function(runge, -1.0, 1.0, ORDER)
y_cheb = np.array([float(cheb_table.eval(x)[0]) for x in x_dense])

# Chebyshev nodes for this order (used as markers).
nodes = vf.ChebTable.cheb_points(ORDER, -1.0, 1.0)
y_nodes = runge(np.asarray(nodes))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_dense, y_true, color=_brand.FAINT, lw=2.0, label="Runge function")
ax.plot(
    x_dense,
    y_poly,
    color=_brand.STEEL,
    lw=1.5,
    ls="--",
    label=f"equispaced poly (order {ORDER})",
)
ax.plot(
    x_dense,
    y_cheb,
    color=_brand.AMBER,
    lw=1.5,
    ls="-.",
    label=f"ChebTable (order {ORDER})",
)
ax.scatter(np.asarray(nodes), y_nodes, color=_brand.AMBER_BRIGHT, zorder=5, s=25, label="Chebyshev nodes")
ax.set_ylim(-0.5, 1.5)
ax.set(
    xlabel="x",
    ylabel="f(x)",
    title="Runge function: equispaced polynomial vs Chebyshev interpolation",
)
ax.legend()
ax.grid(True)
fig.tight_layout()
