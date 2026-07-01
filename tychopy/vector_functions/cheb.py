"""Pure-Python construction helpers over the C++ ChebTable primitives.

Sampling a Python callable and the adaptive order-doubling loop live here so no
Python callable is ever marshalled into a C++ std::function.  The numerical core
(DCT-I build, Clenshaw eval, analytic derivatives) is in C++.

Both 1-D and N-D tensor-product cases are supported.  Dispatch is on whether
``lb``/``ub``/``order`` are scalars (1-D) or sequences (N-D).
"""

import itertools

import _tychopy.vector_functions as _vf
import numpy as np


def _is_nd(lb):
    """True if lb is a sequence of bounds (N-D); False if scalar (1-D)."""
    if isinstance(lb, (str, bytes)):
        return False
    return np.asarray(lb).ndim >= 1


# ---------------------------------------------------------------------------
# 1-D helpers
# ---------------------------------------------------------------------------


def _sample_1d(callable_fn, lb, ub, order):
    """Evaluate callable_fn at the order+1 Chebyshev nodes; return (olen, order+1) array."""
    pts = _vf.ChebTable.cheb_points(order, lb, ub)
    cols = [np.atleast_1d(np.asarray(callable_fn(float(t)), dtype=float)) for t in pts]
    olen = cols[0].size
    vals = np.empty((olen, order + 1), dtype=float)
    for j, c in enumerate(cols):
        if c.size != olen:
            raise ValueError("cheb sampler: inconsistent output size across nodes")
        vals[:, j] = c
    return vals


def _build_1d(callable_fn, lb, ub, order, nthreads):
    """Sample callable_fn and build a 1-D ChebTable; return (table, sampled values)."""
    vals = _sample_1d(callable_fn, lb, ub, order)
    table = _vf.ChebTable.from_values(vals, lb, ub, order, nthreads)
    return table, vals


# ---------------------------------------------------------------------------
# N-D helpers
# ---------------------------------------------------------------------------


def _sample_nd(callable_fn, lb, ub, orders):
    """Evaluate callable_fn on the N-D Chebyshev tensor-product grid.

    Returns (grid_flat, olen) where grid_flat has shape (tsize, olen) and
    rows follow row-major (axis-0-outer) ordering.
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    orders = list(orders)
    nodes = _vf.ChebTable.cheb_points(orders, lb, ub)  # list of per-axis arrays
    shapes = [len(n) for n in nodes]  # [n0+1, ..., n_{D-1}+1]
    tsize = 1
    for s in shapes:
        tsize *= s

    # Build all grid points in row-major order (axis 0 outer)
    indices = itertools.product(*[range(s) for s in shapes])
    olen = None
    grid_flat = None
    for flat_idx, multi_idx in enumerate(indices):
        pt = np.array([nodes[d][i] for d, i in enumerate(multi_idx)], dtype=float)
        val = np.atleast_1d(np.asarray(callable_fn(pt), dtype=float))
        if flat_idx == 0:
            olen = val.size
            grid_flat = np.empty((tsize, olen), dtype=float)
        elif val.size != olen:
            raise ValueError(
                "cheb_from_function N-D: inconsistent output size across grid points"
            )
        grid_flat[flat_idx] = val

    return grid_flat, olen


def _build_nd(callable_fn, lb, ub, orders, nthreads):
    """Sample callable_fn on the N-D grid and build a ChebTable; return (table, grid_flat)."""
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    orders = list(orders)
    grid_flat, _ = _sample_nd(callable_fn, lb_arr, ub_arr, orders)
    table = _vf.ChebTable.from_values(grid_flat, lb_arr, ub_arr, orders, nthreads)
    return table, grid_flat


# ---------------------------------------------------------------------------
# Public API — kept backward-compatible for 1-D; dispatches to N-D on sequences
# ---------------------------------------------------------------------------


# Keep old names for internal backward compat
_sample = _sample_1d
_build = _build_1d


def cheb_from_function(callable_fn, lb, ub, order, nthreads=1):
    """Build a ChebTable by sampling a Python callable at the Chebyshev nodes.

    Supports both 1-D and N-D tensor-product cases.

    Parameters
    ----------
    callable_fn : callable
        - 1-D: ``f(t) -> scalar or array_like`` evaluated at ``order+1`` nodes.
        - N-D: ``f(x) -> scalar or array_like`` where x is a length-D array,
          evaluated at all tensor-product grid points.
    lb : float or array_like, shape (D,)
        Lower bound(s) of the interpolation domain.
    ub : float or array_like, shape (D,)
        Upper bound(s) of the interpolation domain.
    order : int or sequence of int
        - 1-D: polynomial order (``>= 1``); ``order+1`` nodes used.
        - N-D: per-axis polynomial orders (sequence of length D).
    nthreads : int, optional
        Thread count forwarded to the DCT-I transform in ``from_values``.
        Defaults to 1; ``0`` = auto (all cores); negative raises ``ValueError``.

    Returns
    -------
    ChebTable
        Chebyshev interpolant built from the sampled values.
    """
    if _is_nd(lb):
        D = len(np.asarray(lb))
        orders = [int(order)] * D if isinstance(order, int) else list(order)
        table, _ = _build_nd(callable_fn, lb, ub, orders, nthreads)
    else:
        table, _ = _build_1d(callable_fn, float(lb), float(ub), int(order), nthreads)
    return table


def cheb_adaptive(callable_fn, lb, ub, rtol=1e-12, order0=8, max_order=512, nthreads=1):
    """Build a ChebTable by doubling the order until spectral convergence.

    Supports both 1-D and N-D tensor-product cases.

    For 1-D: doubles the order until the trailing-coefficient norm falls below
    ``rtol * scale``.

    For N-D: independently doubles each axis's order until all axes have
    converged.  Convergence for each axis is assessed by building a marginal
    1-D table (evaluating the function along that axis at the centre of all
    other axes) and checking its trailing-coefficient norm.

    Parameters
    ----------
    callable_fn : callable
        - 1-D: ``f(t) -> scalar or array_like``.
        - N-D: ``f(x) -> scalar or array_like`` where x is a length-D array.
    lb : float or array_like, shape (D,)
        Lower bound(s) of the interpolation domain.
    ub : float or array_like, shape (D,)
        Upper bound(s) of the interpolation domain.
    rtol : float, optional
        Relative convergence tolerance on the trailing-coefficient norm.
        Default is ``1e-12``.
    order0 : int or sequence of int, optional
        Starting polynomial order(s).  Scalar for both 1-D and N-D.  Default 8.
    max_order : int, optional
        Maximum polynomial order (per axis for N-D) before giving up.  Default 512.
    nthreads : int, optional
        Thread count forwarded to the DCT-I transform.  Defaults to 1.

    Returns
    -------
    ChebTable
        Converged (or max-order) Chebyshev interpolant.
    """
    if _is_nd(lb):
        return _cheb_adaptive_nd(callable_fn, lb, ub, rtol, order0, max_order, nthreads)
    else:
        return _cheb_adaptive_1d(
            callable_fn, float(lb), float(ub), rtol, order0, max_order, nthreads
        )


def _cheb_adaptive_1d(callable_fn, lb, ub, rtol, order0, max_order, nthreads):
    """1-D adaptive order-doubling loop (original logic)."""
    if order0 > max_order:
        raise ValueError(
            f"cheb_adaptive: order0 ({order0}) must not exceed max_order ({max_order})"
        )
    order = int(order0)
    table, vals = _build_1d(callable_fn, lb, ub, order, nthreads)
    while order < max_order:
        tail = np.max(table.coeff_tail_norm())
        scale = max(np.max(np.abs(vals)), 1.0)
        if tail <= rtol * scale:
            return table
        order *= 2
        table, vals = _build_1d(callable_fn, lb, ub, order, nthreads)
    return table


def _cheb_adaptive_nd(callable_fn, lb, ub, rtol, order0, max_order, nthreads):
    """N-D adaptive order-doubling loop with per-axis convergence checking.

    Each axis is independently refined: marginal 1-D tables are built for
    each axis (holding all other axes at several non-degenerate base points,
    offset from the domain centres to avoid axes where the function is
    identically zero), and their trailing-coefficient norm is used as the
    per-axis convergence indicator.
    """
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    D = len(lb_arr)

    if isinstance(order0, int):
        orders = [order0] * D
    else:
        orders = list(order0)

    if any(o > max_order for o in orders):
        raise ValueError(
            f"cheb_adaptive: order0 {orders} exceeds max_order ({max_order})"
        )

    # Evaluation points for marginals: use several offsets to avoid degenerate
    # points where the function is identically zero along an axis.
    _offsets = np.array(
        [0.3, -0.4, 0.6]
    )  # fractional offsets in [-1,1] of normalised domain

    def _eval_points():
        """Return 3 non-degenerate base points (offset from the domain centres)."""
        pts = []
        for alpha in _offsets:
            pt = lb_arr + 0.5 * (ub_arr - lb_arr) * (1.0 + alpha)
            pt = np.clip(pt, lb_arr, ub_arr)
            pts.append(pt)
        return pts

    def converged_axis(d, order_d):
        """Check convergence along axis d using several marginal 1-D functions."""
        eval_pts = _eval_points()
        # Check convergence for each fixed point of the other axes; axis d is
        # converged only when ALL marginals are below rtol.
        for base_pt in eval_pts:

            def marginal(t, _base=base_pt):
                pt = _base.copy()
                pt[d] = float(t)
                return callable_fn(pt)

            vals_1d = _sample_1d(marginal, lb_arr[d], ub_arr[d], order_d)
            tab1d = _vf.ChebTable.from_values(
                vals_1d, lb_arr[d], ub_arr[d], order_d, nthreads
            )
            tail = np.max(tab1d.coeff_tail_norm())
            scale = max(float(np.max(np.abs(vals_1d))), 1.0)
            if tail > rtol * scale:
                return False
        return True

    # Refine axes until all converged or max_order reached
    axis_done = [False] * D
    while True:
        all_done = True
        for d in range(D):
            if axis_done[d]:
                continue
            if converged_axis(d, orders[d]):
                axis_done[d] = True
            else:
                all_done = False
                if orders[d] < max_order:
                    orders[d] = min(orders[d] * 2, max_order)
        if all_done or all(axis_done[d] or orders[d] >= max_order for d in range(D)):
            break

    table, _ = _build_nd(callable_fn, lb_arr, ub_arr, orders, nthreads)
    return table
