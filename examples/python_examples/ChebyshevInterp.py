"""Chebyshev interpolation demo: spectral (Runge-free) accuracy + adaptive fit.
Serves as an integration test in the Python example suite."""

import numpy as np

from tychopy.vector_functions import cheb_adaptive, cheb_from_function


def main():
    f = lambda t: 1.0 / (1.0 + 25.0 * t * t)  # Runge function on [-1, 1]
    tab = cheb_from_function(f, -1.0, 1.0, 64)
    ts = np.linspace(-0.99, 0.99, 401)
    err = max(abs(tab.eval(float(t))[0] - f(t)) for t in ts)
    print(f"Chebyshev(64) max error on Runge function: {err:.3e}")
    assert err < 1e-3, "Chebyshev should not exhibit Runge oscillation"

    g = lambda t: np.sin(3.0 * t) * np.exp(-0.2 * t)
    tab2 = cheb_adaptive(g, 0.0, 10.0, rtol=1e-12)
    ts2 = np.linspace(0.1, 9.9, 200)
    err2 = max(abs(tab2.eval(float(t))[0] - g(t)) for t in ts2)
    print(f"Adaptive Chebyshev max error: {err2:.3e}; order={tab2.order}")
    assert err2 < 1e-9

    print("Optimal Solution Found")  # informational sentinel


if __name__ == "__main__":
    main()
