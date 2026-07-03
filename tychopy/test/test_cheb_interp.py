import warnings

import numpy as np
import pytest

import tychopy.vector_functions as vf
from tychopy.vector_functions import (
    ChebConvergenceWarning,
    cheb_adaptive,
    cheb_from_function,
)

Args = vf.Arguments


def test_from_values_roundtrip():
    n, lb, ub = 24, 0.0, 6.0
    pts = vf.ChebTable.cheb_points(n, lb, ub)
    vals = (np.sin(0.7 * pts) + 0.3 * pts).reshape(-1, 1)
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


# ---------------------------------------------------------------------------
# N-D tests
# ---------------------------------------------------------------------------


def test_nd_cheb_points_2d():
    """N-D cheb_points returns per-axis node arrays of correct shape."""
    orders = [6, 8]
    lb = np.array([0.0, -1.0])
    ub = np.array([2.0, 1.0])
    nodes = vf.ChebTable.cheb_points(orders, lb, ub)
    assert len(nodes) == 2
    assert nodes[0].shape == (7,)
    assert nodes[1].shape == (9,)
    # bounds: nodes should be in [lb, ub]
    assert nodes[0].min() >= 0.0 - 1e-12 and nodes[0].max() <= 2.0 + 1e-12
    assert nodes[1].min() >= -1.0 - 1e-12 and nodes[1].max() <= 1.0 + 1e-12


def test_nd_from_values_2d():
    """N-D from_values builds a table that reproduces grid values at nodes."""
    orders = [8, 8]
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    nodes = vf.ChebTable.cheb_points(orders, lb, ub)
    nx, ny = len(nodes[0]), len(nodes[1])
    # f(x,y) = sin(x)*cos(y)
    vals = np.zeros((nx * ny, 1))
    for i, x in enumerate(nodes[0]):
        for j, y in enumerate(nodes[1]):
            vals[i * ny + j, 0] = np.sin(x) * np.cos(y)
    tab = vf.ChebTable.from_values(vals, lb, ub, orders)
    assert tab.input_dim == 2
    assert tab.output_dim == 1
    # check round-trip at nodes
    for i, x in enumerate(nodes[0]):
        for j, y in enumerate(nodes[1]):
            v = tab.eval(np.array([x, y]))
            assert abs(v[0] - np.sin(x) * np.cos(y)) < 1e-10


def test_nd_from_function_2d_accuracy():
    """cheb_from_function N-D: f(x,y)=sin(x)*cos(y) accurate to 1e-8 at off-grid points."""

    def f(xy):
        return np.array([np.sin(xy[0]) * np.cos(xy[1])])

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    orders = [12, 12]
    tab = cheb_from_function(f, lb, ub, orders)
    assert tab.input_dim == 2
    assert tab.output_dim == 1
    rng = np.random.default_rng(42)
    test_pts = rng.uniform([-1.0, -1.0], [1.0, 1.0], (20, 2))
    for xy in test_pts:
        v = tab.eval(xy)
        expected = np.sin(xy[0]) * np.cos(xy[1])
        assert abs(v[0] - expected) < 1e-8, f"error at {xy}: {abs(v[0] - expected)}"


def test_nd_from_function_3d_accuracy():
    """cheb_from_function N-D 3D: f(x,y,z)=exp(-x^2)*sin(y)*cos(z)."""

    def f(xyz):
        return np.array([np.exp(-(xyz[0] ** 2)) * np.sin(xyz[1]) * np.cos(xyz[2])])

    lb = np.array([-1.0, -1.0, -1.0])
    ub = np.array([1.0, 1.0, 1.0])
    orders = [14, 14, 14]
    tab = cheb_from_function(f, lb, ub, orders)
    assert tab.input_dim == 3
    rng = np.random.default_rng(7)
    test_pts = rng.uniform([-0.9, -0.9, -0.9], [0.9, 0.9, 0.9], (10, 3))
    for xyz in test_pts:
        v = tab.eval(xyz)
        expected = np.exp(-(xyz[0] ** 2)) * np.sin(xyz[1]) * np.cos(xyz[2])
        assert abs(v[0] - expected) < 1e-8, f"error at {xyz}: {abs(v[0] - expected)}"


def test_nd_compose_2arg():
    """N-D compose: tab(args[0], args[1]) builds a VF matching tab.eval."""

    def f(xy):
        return np.array([np.sin(xy[0]) * np.cos(xy[1]), xy[0] + xy[1]])

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    orders = [10, 10]
    tab = cheb_from_function(f, lb, ub, orders)
    assert tab.output_dim == 2

    args = Args(2)
    g = tab(args[0], args[1])

    for xy in [np.array([0.3, -0.4]), np.array([-0.7, 0.6]), np.array([0.1, 0.2])]:
        out = np.array(g.compute(xy))
        ref = tab.eval(xy)
        assert np.allclose(out, ref, atol=1e-12), f"mismatch at {xy}: {out} vs {ref}"


def test_nd_compose_3arg():
    """N-D compose: tab(args[0], args[1], args[2]) builds a VF matching tab.eval."""

    def f(xyz):
        return np.array([np.sin(xyz[0]) * np.cos(xyz[1]) + 0.1 * xyz[2]])

    lb = np.array([-1.0, -1.0, -1.0])
    ub = np.array([1.0, 1.0, 1.0])
    orders = [8, 8, 8]
    tab = cheb_from_function(f, lb, ub, orders)
    assert tab.output_dim == 1

    args = Args(3)
    g = tab(args[0], args[1], args[2])

    for xyz in [np.array([0.3, -0.4, 0.1]), np.array([-0.5, 0.6, -0.2])]:
        out = np.array(g.compute(xyz))
        ref = tab.eval(xyz)
        assert np.allclose(out, ref, atol=1e-12), f"mismatch at {xyz}: {out} vs {ref}"


def test_nd_vf_method():
    """tab.vf() returns a VectorFunction for a 2-D table."""

    def f(xy):
        return np.array([np.sin(xy[0]) * np.cos(xy[1])])

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    orders = [10, 10]
    tab = cheb_from_function(f, lb, ub, orders)
    g = tab.vf()
    for xy in [np.array([0.3, -0.4]), np.array([0.0, 0.0])]:
        out = np.array(g.compute(xy))
        ref = tab.eval(xy)
        assert np.allclose(out, ref, atol=1e-12)


def test_nd_adaptive_2d():
    """cheb_adaptive N-D converges for f(x,y)=sin(3x)*cos(2y)."""

    def f(xy):
        return np.array([np.sin(3 * xy[0]) * np.cos(2 * xy[1])])

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    tab = cheb_adaptive(f, lb, ub, rtol=1e-10)
    rng = np.random.default_rng(99)
    test_pts = rng.uniform([-0.9, -0.9], [0.9, 0.9], (15, 2))
    for xy in test_pts:
        v = tab.eval(xy)
        expected = np.sin(3 * xy[0]) * np.cos(2 * xy[1])
        assert abs(v[0] - expected) < 1e-8, f"error {abs(v[0] - expected)} at {xy}"


def test_nd_adaptive_terminates_mixed_convergence():
    """N-D adaptive must TERMINATE even when one axis cannot converge within
    a low max_order while another converges quickly (regression: infinite loop).

    Axis 0 is low-frequency (converges fast); axis 1 is very high-frequency so
    it hits max_order without converging.  The call simply returning proves the
    loop terminates rather than spinning forever."""

    def f(xy):
        # low-frequency in x, very high-frequency in y (won't converge at low order)
        return np.array([np.sin(0.5 * xy[0]) * np.cos(40.0 * xy[1])])

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    # Small max_order so axis 1 cannot converge; axis 0 converges early.  The
    # capped axis must raise ChebConvergenceWarning (and reaching this line at
    # all proves the loop terminates rather than hanging).
    with pytest.warns(ChebConvergenceWarning):
        tab = cheb_adaptive(f, lb, ub, rtol=1e-12, order0=8, max_order=16)
    assert tab.input_dim == 2
    assert tab.output_dim == 1


def test_nd_from_function_multichannel_2d():
    """N-D from_function works with multichannel output."""

    def f(xy):
        return np.array([np.sin(xy[0]), np.cos(xy[1]), xy[0] * xy[1]])

    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    orders = [10, 10]
    tab = cheb_from_function(f, lb, ub, orders)
    assert tab.output_dim == 3
    assert tab.input_dim == 2
    for xy in [np.array([0.3, 0.7]), np.array([0.5, 0.5])]:
        v = tab.eval(xy)
        assert abs(v[0] - np.sin(xy[0])) < 1e-9
        assert abs(v[1] - np.cos(xy[1])) < 1e-9
        assert abs(v[2] - xy[0] * xy[1]) < 1e-9


def test_1d_behavior_preserved():
    """Confirm all 1-D ChebTable functionality is unaffected by N-D changes."""
    n, lb, ub = 20, -1.0, 1.0
    pts = vf.ChebTable.cheb_points(n, lb, ub)
    assert pts.shape == (n + 1,)
    vals = np.sin(pts).reshape(-1, 1)
    tab = vf.ChebTable.from_values(vals, lb, ub, n)
    assert tab.order == n
    assert tab.output_dim == 1
    for t in np.linspace(-0.9, 0.9, 9):
        assert abs(tab.eval(float(t))[0] - np.sin(t)) < 1e-10
    v, dv = tab.eval_deriv1(0.5)
    assert abs(v[0] - np.sin(0.5)) < 1e-9
    assert abs(dv[0] - np.cos(0.5)) < 1e-7
    args = Args(1)
    g = tab(args[0])
    out = g.compute([0.3])
    assert abs(out[0] - np.sin(0.3)) < 1e-10


# ---------------------------------------------------------------------------
# Review-response coverage
# ---------------------------------------------------------------------------


def test_binding_eval_deriv2():
    """ChebTable.eval_deriv2 returns analytic first + second derivatives."""
    lb, ub, n = 0.0, 6.0, 30
    pts = vf.ChebTable.cheb_points(n, lb, ub)
    vals = np.sin(0.7 * pts).reshape(-1, 1)
    tab = vf.ChebTable.from_values(vals, lb, ub, n)
    for t in [0.5, 2.9, 5.2]:
        v, dv, d2 = tab.eval_deriv2(float(t))
        assert abs(v[0] - np.sin(0.7 * t)) < 1e-9
        assert abs(dv[0] - 0.7 * np.cos(0.7 * t)) < 1e-7
        assert abs(d2[0] - (-0.49 * np.sin(0.7 * t))) < 1e-5


def test_bounds_and_contains():
    """lb/ub properties and contains() predicate (1-D scalar and N-D array)."""
    tab = cheb_from_function(np.sin, -2.0, 3.0, 12)
    assert tab.lb == pytest.approx(-2.0)
    assert tab.ub == pytest.approx(3.0)
    assert tab.contains(0.0)
    assert not tab.contains(4.0)

    tab2 = cheb_from_function(
        lambda x: np.sin(x[0]) * x[1], [-1.0, 0.0], [1.0, 2.0], [6, 4]
    )
    np.testing.assert_allclose(tab2.lb, [-1.0, 0.0])
    np.testing.assert_allclose(tab2.ub, [1.0, 2.0])
    assert tab2.contains(np.array([0.0, 1.0]))
    assert not tab2.contains(np.array([0.0, 3.0]))


def test_nan_sample_rejected_1d():
    """A callable returning NaN raises ValueError instead of poisoning the table."""

    def f(t):
        return np.nan if t > 0.5 else t

    with pytest.raises(ValueError):
        cheb_from_function(f, 0.0, 1.0, 8)


def test_nan_sample_rejected_nd():
    def f(x):
        return np.inf if x[0] > 0.0 else x[0] + x[1]

    with pytest.raises(ValueError):
        cheb_from_function(f, [-1.0, -1.0], [1.0, 1.0], [4, 4])


def test_scalar_eval_on_nd_table_raises():
    tab = cheb_from_function(lambda x: x[0] + x[1], [0.0, 0.0], [1.0, 1.0], [4, 4])
    with pytest.raises(ValueError):
        tab.eval(0.5)


def test_wrong_length_query_raises():
    tab = cheb_from_function(lambda x: x[0] + x[1], [0.0, 0.0], [1.0, 1.0], [4, 4])
    with pytest.raises(ValueError):
        tab.eval(np.array([0.1, 0.2, 0.3]))


def test_nonconvergence_warns_1d():
    """A discontinuous function cannot converge; cheb_adaptive warns and returns best-effort."""

    def step(t):
        return 1.0 if t > 0.0 else -1.0

    with pytest.warns(ChebConvergenceWarning):
        tab = cheb_adaptive(step, -1.0, 1.0, rtol=1e-12, order0=8, max_order=32)
    assert tab.order <= 32  # returned a usable (best-effort) table


def test_converged_1d_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        cheb_adaptive(lambda t: np.cos(1.3 * t), -2.0, 2.0, rtol=1e-11)


def test_periodic_wrap_1d():
    """Periodic policy wraps out-of-domain queries modulo the span."""
    lb, ub = 0.0, 2.0 * np.pi
    tab = cheb_from_function(np.cos, lb, ub, 24, periodic=True)
    t = 1.3
    assert tab.eval(t + 2.0 * (ub - lb))[0] == pytest.approx(tab.eval(t)[0], abs=1e-10)
    assert tab.eval(t + (ub - lb))[0] == pytest.approx(np.cos(t), abs=1e-9)


def test_periodic_per_axis_nd():
    lb = [0.0, -1.0]
    ub = [2.0 * np.pi, 1.0]
    tab = cheb_from_function(
        lambda x: np.cos(x[0]) * x[1], lb, ub, [20, 3], periodic=[True, False]
    )
    q = np.array([1.1, 0.4])
    qw = np.array([1.1 + (ub[0] - lb[0]), 0.4])
    assert tab.eval(qw)[0] == pytest.approx(tab.eval(q)[0], abs=1e-9)


def test_residual_net_warns_on_underresolved_coupling():
    """The N-D coupling-residual net warns when a strongly coupled function is
    under-resolved (the per-axis marginal test cannot see cross-axis coupling)."""
    from tychopy.vector_functions.cheb import _warn_if_residual_exceeds

    def f(xy):
        return np.array([np.sin(8.0 * xy[0] * xy[1])])

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    # Order 6 is far too low to resolve sin(8*x*y) -> large pointwise residual.
    tab = cheb_from_function(f, lb, ub, [6, 6])
    with pytest.warns(ChebConvergenceWarning):
        _warn_if_residual_exceeds(f, tab, lb, ub)


def test_residual_net_silent_when_resolved():
    """The residual net must NOT warn on a well-resolved separable field."""
    from tychopy.vector_functions.cheb import _warn_if_residual_exceeds

    def f(xy):
        return np.array([np.sin(2.0 * xy[0]) * np.cos(2.0 * xy[1])])

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    tab = cheb_from_function(f, lb, ub, [24, 24])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ChebConvergenceWarning)
        _warn_if_residual_exceeds(f, tab, lb, ub)  # must not raise


def test_clamp_derivative_zero_outside_1d():
    """Under Clamp the value is flat outside [lb, ub], so eval_deriv1/2 return 0
    there; the boundary retains the (nonzero) one-sided endpoint slope."""
    tab = cheb_from_function(lambda t: np.sin(0.7 * t), 0.0, 2.0, 20)
    for t in (2.0 + 3.0, 0.0 - 3.0):
        v, dv = tab.eval_deriv1(t)
        v2, dv2, d2 = tab.eval_deriv2(t)
        assert dv[0] == pytest.approx(0.0, abs=1e-13)
        assert dv2[0] == pytest.approx(0.0, abs=1e-13)
        assert d2[0] == pytest.approx(0.0, abs=1e-13)
    # Boundary keeps the endpoint slope (nonzero).
    _, dv_b = tab.eval_deriv1(2.0)
    assert abs(dv_b[0]) > 1e-6
