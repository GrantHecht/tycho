"""Out-of-domain behavior: ChebTable clamping vs polynomial extrapolation.

Rendered by the matplotlib ``plot`` directive (file-based form, so the image is
named after this script and cannot collide with other figures on the page). This
is a build asset, not a documentation page.
"""

import matplotlib.pyplot as plt
import numpy as np

import _brand  # sibling module; the plot directive puts _plots/ on sys.path
from tychopy import vector_functions as vf

_brand.apply()


def f(x):
    return np.exp(-2.0 * x) * np.sin(3.0 * x) + 0.5


LB, UB = 0.0, 1.0
N = 12

# Build ChebTable on [0, 1].
cheb_tbl = vf.cheb_from_function(f, LB, UB, N - 1)

# Global polynomial extrapolant via numpy (shows divergence outside domain).
ts = np.linspace(LB, UB, N)
coeffs = np.polyfit(ts, f(ts), N - 1)

# Evaluate on the extended range.
x_ext = np.linspace(-0.3, 1.3, 500)
y_true = f(x_ext)
y_poly = np.polyval(coeffs, x_ext)
y_cheb = np.array([float(cheb_tbl.eval(x)[0]) for x in x_ext])

fig, ax = plt.subplots(figsize=(7, 4))
ax.axvspan(-0.3, LB, color=_brand.FAINT, alpha=0.18, label="outside domain")
ax.axvspan(UB, 1.3, color=_brand.FAINT, alpha=0.18)
ax.plot(x_ext, y_true, color=_brand.FAINT, lw=1.5, ls=":", label="true function")
ax.plot(
    x_ext,
    y_poly,
    color=_brand.STEEL,
    lw=1.8,
    label="polynomial extrapolation (diverges)",
)
ax.plot(
    x_ext,
    y_cheb,
    color=_brand.AMBER,
    lw=1.8,
    ls="--",
    label="ChebTable (clamps at endpoints)",
)
ax.set_ylim(-1.5, 1.8)
ax.set(
    xlabel="x",
    ylabel="value",
    title="Out-of-domain behavior: extrapolation diverges, ChebTable clamps",
)
ax.legend()
ax.grid(True)
fig.tight_layout()
