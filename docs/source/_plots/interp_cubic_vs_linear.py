"""InterpTable1D cubic vs linear comparison for the Interpolation tutorial.

Rendered by the matplotlib ``plot`` directive (file-based form, so the image is
named after this script and cannot collide with other figures on the page). This
is a build asset, not a documentation page.
"""

import matplotlib.pyplot as plt
import numpy as np

import _brand  # sibling module; the plot directive puts _plots/ on sys.path
from tychopy import vector_functions as vf

_brand.apply()

# 8 coarse samples of a bumpy function.
ts_coarse = np.linspace(0.0, 2.0 * np.pi, 8)
vs_coarse = np.sin(ts_coarse) + 0.4 * np.sin(3.0 * ts_coarse)

cubic_table = vf.InterpTable1D(ts_coarse, vs_coarse, kind="cubic")
linear_table = vf.InterpTable1D(ts_coarse, vs_coarse, kind="linear")

t_dense = np.linspace(0.0, 2.0 * np.pi, 400)
y_cubic = np.array([float(cubic_table.interp(t)[0]) for t in t_dense])
y_linear = np.array([float(linear_table.interp(t)[0]) for t in t_dense])

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(t_dense, y_cubic, color=_brand.AMBER, lw=1.8, label="cubic spline")
ax.plot(t_dense, y_linear, color=_brand.STEEL, lw=1.5, ls="--", label="linear")
ax.scatter(ts_coarse, vs_coarse, color=_brand.AMBER_BRIGHT, zorder=5, s=40, label="samples")
ax.set(
    xlabel="t",
    ylabel="value",
    title="InterpTable1D: cubic vs linear (8 sample points)",
)
ax.legend()
ax.grid(True)
fig.tight_layout()
