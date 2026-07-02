"""N-D Chebyshev interpolation demo: 2-D and 3-D tensor-product spectral accuracy.

Builds ChebTable interpolants for smooth multi-dimensional fields using
cheb_from_function, then asserts max error < 1e-8 on a dense off-grid sample.
Serves as an integration test in the Python example suite.
"""

import numpy as np

from tychopy.vector_functions import Arguments, cheb_from_function


def main():
    # ------------------------------------------------------------------
    # 2-D: f(x, y) = sin(x) * cos(y) on [0, pi] x [0, pi]
    # ------------------------------------------------------------------
    f2d = lambda x: np.sin(x[0]) * np.cos(x[1])
    tab2d = cheb_from_function(f2d, [0.0, 0.0], [np.pi, np.pi], (16, 16))
    assert tab2d.input_dim == 2, "Expected 2-D table"
    assert tab2d.output_dim == 1, "Expected 1-channel output"

    # Dense off-grid sample on a 50x50 grid (not aligned to Chebyshev nodes)
    xs = np.linspace(0.05, np.pi - 0.05, 50)
    ys = np.linspace(0.05, np.pi - 0.05, 50)
    max_err_2d = 0.0
    for xi in xs:
        for yi in ys:
            pt = np.array([xi, yi])
            approx = float(tab2d.eval(pt)[0])
            exact = np.sin(xi) * np.cos(yi)
            max_err_2d = max(max_err_2d, abs(approx - exact))

    print(f"2-D ChebTable(16x16) max error: {max_err_2d:.3e}")
    assert max_err_2d < 1e-8, f"2-D error too large: {max_err_2d:.3e}"

    # Verify VectorFunction composition (compose tab2d into a 3-input argument space)
    a = Arguments(3)
    composed = tab2d(a[0], a[1])
    assert composed.input_rows() == 3
    assert composed.output_rows() == 1

    # Also verify .vf() path (2-input)
    vf2d = tab2d.vf()
    assert vf2d.input_rows() == 2
    assert vf2d.output_rows() == 1

    # ------------------------------------------------------------------
    # 3-D: g(x, y, z) = exp(-0.5*(x^2+y^2+z^2)) on [-2, 2]^3
    # ------------------------------------------------------------------
    f3d = lambda x: np.exp(-0.5 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    tab3d = cheb_from_function(f3d, [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], (18, 18, 18))
    assert tab3d.input_dim == 3, "Expected 3-D table"

    # Dense off-grid sample — use a coarser grid to keep runtime fast
    coords = np.linspace(-1.8, 1.8, 20)
    max_err_3d = 0.0
    for xi in coords:
        for yi in coords:
            for zi in coords:
                pt = np.array([xi, yi, zi])
                approx = float(tab3d.eval(pt)[0])
                exact = f3d(pt)
                max_err_3d = max(max_err_3d, abs(approx - exact))

    print(f"3-D ChebTable(18x18x18) max error: {max_err_3d:.3e}")
    assert max_err_3d < 1e-8, f"3-D error too large: {max_err_3d:.3e}"

    # Verify 3-D VectorFunction composition
    b = Arguments(3)
    composed3d = tab3d(b[0], b[1], b[2])
    assert composed3d.input_rows() == 3
    assert composed3d.output_rows() == 1

    print("Optimal Solution Found")


if __name__ == "__main__":
    main()
