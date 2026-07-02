"""Spectral vs algebraic convergence of ChebTable and cubic InterpTable1D.

Rendered by the matplotlib ``plot`` directive (file-based form, so the image is
named after this script and cannot collide with other figures on the page). This
is a build asset, not a documentation page.
"""

import matplotlib.pyplot as plt
import numpy as np

import _brand  # sibling module; the plot directive puts _plots/ on sys.path
from tychopy import vector_functions as vf

_brand.apply()


def f_true(x):
    return np.exp(np.sin(3.0 * x))


LB, UB = 0.0, 2.0
x_ref = np.linspace(LB, UB, 2000)
y_ref = f_true(x_ref)

Ns = np.arange(5, 42)
err_cubic = []
err_cheb = []

for N in Ns:
    # Cubic spline on N equispaced samples (requires N > 4).
    ts = np.linspace(LB, UB, N)
    tbl = vf.InterpTable1D(ts, f_true(ts), kind="cubic")
    y_hat = np.array([float(tbl.interp(x)[0]) for x in x_ref])
    err_cubic.append(np.max(np.abs(y_hat - y_ref)))

    # Chebyshev table of order N-1.
    ct = vf.cheb_from_function(f_true, LB, UB, N - 1)
    y_hat_c = np.array([float(ct.eval(x)[0]) for x in x_ref])
    err_cheb.append(np.max(np.abs(y_hat_c - y_ref)))

err_cubic = np.array(err_cubic)
err_cheb = np.maximum(err_cheb, 1e-15)  # clip to avoid log(0)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(
    Ns,
    err_cubic,
    color=_brand.STEEL,
    lw=1.8,
    label="cubic InterpTable1D (equispaced)",
)
ax.semilogy(
    Ns,
    err_cheb,
    color=_brand.AMBER,
    lw=1.8,
    ls="--",
    label="ChebTable (spectral)",
)
ax.set(
    xlabel="N (sample count)",
    ylabel="max |error|",
    title=r"Convergence on $f(x)=e^{\sin(3x)}$: algebraic vs spectral",
)
ax.legend()
ax.grid(True, which="both")
fig.tight_layout()
