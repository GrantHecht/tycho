import numpy as np

import tychopy.vector_functions as vf
from tychopy.vector_functions import cheb_adaptive, cheb_from_function

Args = vf.Arguments


def test_from_values_roundtrip():
    n, lb, ub = 24, 0.0, 6.0
    pts = vf.ChebTable.cheb_points(n, lb, ub)
    vals = (np.sin(0.7 * pts) + 0.3 * pts).reshape(1, -1)
    tab = vf.ChebTable.from_values(vals, lb, ub, n)
    for t in pts:
        assert abs(tab.eval(float(t))[0] - (np.sin(0.7 * t) + 0.3 * t)) < 1e-11


def test_from_function_accuracy():
    tab = cheb_from_function(lambda t: np.sin(0.7 * t) + 0.3 * t, 0.0, 6.0, 28)
    for t in np.linspace(0.1, 5.9, 17):
        assert abs(tab.eval(float(t))[0] - (np.sin(0.7 * t) + 0.3 * t)) < 1e-8


def test_adaptive_converges():
    tab = cheb_adaptive(lambda t: np.cos(1.3 * t), -2.0, 2.0, rtol=1e-12)
    for t in np.linspace(-1.9, 1.9, 21):
        assert abs(tab.eval(float(t))[0] - np.cos(1.3 * t)) < 1e-9


def test_compose_to_vectorfunction():
    """Composing a ChebTable with a scalar argument yields a VectorFunction
    whose evaluation matches the table's own eval (a plan acceptance criterion)."""
    n, lb, ub = 20, -1.0, 1.0

    def f(t):
        return np.sin(t)

    tab = cheb_from_function(f, lb, ub, n)

    # Compose with a scalar Argument -> VectorFunction.
    args = Args(1)
    g = tab(args[0])
    for t in np.linspace(-0.9, 0.9, 11):
        out = g.compute([float(t)])
        assert abs(out[0] - tab.eval(float(t))[0]) < 1e-12

    # The vf() accessor returns an equivalent VectorFunction.
    gvf = tab.vf()
    for t in np.linspace(-0.9, 0.9, 11):
        out = gvf.compute([float(t)])
        assert abs(out[0] - tab.eval(float(t))[0]) < 1e-12


def test_multichannel_from_function():
    """A callable returning a multi-element array builds a multi-channel table;
    each channel is accurate independently."""
    lb, ub, n = 0.0, 2.0, 24

    def f(t):
        return np.array([np.sin(t), np.cos(t), t * t])

    tab = cheb_from_function(f, lb, ub, n)
    assert tab.output_dim == 3
    for t in np.linspace(0.1, 1.9, 13):
        v = tab.eval(float(t))
        assert abs(v[0] - np.sin(t)) < 1e-9
        assert abs(v[1] - np.cos(t)) < 1e-9
        assert abs(v[2] - t * t) < 1e-9


def test_binding_eval_deriv1():
    """ChebTable.eval_deriv1 returns the analytic first derivative."""
    lb, ub, n = 0.0, 6.0, 30

    def f(t):
        return np.sin(0.7 * t) + 0.3 * t

    def df(t):
        return 0.7 * np.cos(0.7 * t) + 0.3

    tab = cheb_from_function(f, lb, ub, n)
    for t in (1.7, 2.9, 4.1):
        v, dv = tab.eval_deriv1(t)
        assert abs(v[0] - f(t)) < 1e-9
        assert abs(dv[0] - df(t)) < 1e-7
        # finite-difference cross-check
        e = 1e-5
        fd = (f(t + e) - f(t - e)) / (2 * e)
        assert abs(dv[0] - fd) < 1e-6


def test_adaptive_inverted_bounds_raises():
    """order0 > max_order must raise rather than silently over-sample."""
    try:
        cheb_adaptive(lambda t: np.cos(t), -1.0, 1.0, order0=64, max_order=16)
    except ValueError:
        pass
    else:
        raise AssertionError("cheb_adaptive did not raise on order0 > max_order")


def test_nthreads_zero_smoke():
    """nthreads=0 (auto) does not throw and yields the same result as default."""
    lb, ub, n = -1.0, 1.0, 20

    def f(t):
        return np.sin(3.0 * t)

    tab1 = cheb_from_function(f, lb, ub, n, nthreads=1)
    tab0 = cheb_from_function(f, lb, ub, n, nthreads=0)
    for t in np.linspace(-0.9, 0.9, 11):
        assert abs(tab0.eval(float(t))[0] - tab1.eval(float(t))[0]) < 1e-14
