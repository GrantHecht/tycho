"""InterpTable2D filled-contour figure for the Interpolation tutorial.

Rendered by the matplotlib ``plot`` directive (file-based form, so the image is
named after this script and cannot collide with other figures on the page). This
is a build asset, not a documentation page.
"""

import matplotlib.pyplot as plt
import numpy as np

import _brand  # sibling module; the plot directive puts _plots/ on sys.path
from tychopy import vector_functions as vf

_brand.apply()

# Build the table on a coarse grid (matching the doctest above).
xs = np.linspace(0.0, 1.0, 11)
ys = np.linspace(0.0, 2.0, 21)
X_coarse, Y_coarse = np.meshgrid(xs, ys)  # shape (21, 11)
z_coarse = np.sin(X_coarse) * np.cos(Y_coarse)
field = vf.InterpTable2D(xs, ys, z_coarse)

# Evaluate on a dense grid.
x_fine = np.linspace(0.0, 1.0, 200)
y_fine = np.linspace(0.0, 2.0, 200)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
Z_fine = np.array([[float(field(x, y)) for x in x_fine] for y in y_fine])

fig, ax = plt.subplots(figsize=(5, 5))
cf = ax.contourf(X_fine, Y_fine, Z_fine, levels=20, cmap="YlOrBr")
fig.colorbar(cf, ax=ax, label="z value")
ax.set(
    xlabel="x",
    ylabel="y",
    title=r"InterpTable2D: $z(x,y) = \sin(x)\cos(y)$",
)
fig.tight_layout()
