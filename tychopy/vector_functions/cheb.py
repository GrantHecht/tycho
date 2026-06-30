"""Pure-Python construction helpers over the C++ ChebTable primitives.

Sampling a Python callable and the adaptive order-doubling loop live here so no
Python callable is ever marshalled into a C++ std::function.  The numerical core
(DCT-I build, Clenshaw eval, analytic derivatives) is in C++.
"""

import _tychopy.vector_functions as _vf
import numpy as np


def _sample(callable_fn, lb, ub, order):
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


def _build(callable_fn, lb, ub, order, nthreads):
    """Sample callable_fn and build a ChebTable; return (table, sampled values).

    Returns both so callers (e.g. the adaptive loop) can reuse the sampled
    values without a second pass over the callable.
    """
    vals = _sample(callable_fn, lb, ub, order)
    table = _vf.ChebTable.from_values(vals, lb, ub, order, nthreads)
    return table, vals


def cheb_from_function(callable_fn, lb, ub, order, nthreads=1):
    """Build a ChebTable by sampling a Python callable at the Chebyshev nodes.

    Parameters
    ----------
    callable_fn : callable
        A function ``f(t) -> scalar or array_like`` evaluated at the
        ``order+1`` Chebyshev nodes on ``[lb, ub]``.
    lb : float
        Lower bound of the interpolation domain.
    ub : float
        Upper bound of the interpolation domain.
    order : int
        Polynomial order (``>= 1``).  The interpolant uses ``order+1`` nodes.
    nthreads : int, optional
        Thread count forwarded to the DCT-I transform in ``from_values``.
        Defaults to 1; ``0`` = auto (all cores); a negative value raises
        ``ValueError``.

    Returns
    -------
    ChebTable
        1-D Chebyshev interpolant built from the sampled values.
    """
    table, _ = _build(callable_fn, lb, ub, order, nthreads)
    return table


def cheb_adaptive(callable_fn, lb, ub, rtol=1e-12, order0=8, max_order=512, nthreads=1):
    """Build a ChebTable by doubling the order until spectral convergence.

    The order is doubled until the trailing-coefficient norm, normalized by
    the peak function magnitude, falls below ``rtol``.

    Parameters
    ----------
    callable_fn : callable
        A function ``f(t) -> scalar or array_like``.
    lb : float
        Lower bound of the interpolation domain.
    ub : float
        Upper bound of the interpolation domain.
    rtol : float, optional
        Relative convergence tolerance on the trailing-coefficient norm.
        Default is ``1e-12``.
    order0 : int, optional
        Starting polynomial order.  Default is 8.
    max_order : int, optional
        Maximum polynomial order before giving up.  Default is 512.
    nthreads : int, optional
        Thread count forwarded to the DCT-I transform in ``from_values``.
        Defaults to 1; ``0`` = auto (all cores); a negative value raises
        ``ValueError``.

    Returns
    -------
    ChebTable
        Converged (or max-order) Chebyshev interpolant.
    """
    if order0 > max_order:
        raise ValueError(
            f"cheb_adaptive: order0 ({order0}) must not exceed max_order ({max_order})"
        )
    order = int(order0)
    table, vals = _build(callable_fn, lb, ub, order, nthreads)
    while order < max_order:
        tail = np.max(table.coeff_tail_norm())
        # scale from the current (possibly under-resolved) samples. An
        # under-estimate only makes the convergence test stricter (more
        # doubling), never looser, so it can't cause premature convergence.
        scale = max(np.max(np.abs(vals)), 1.0)
        if tail <= rtol * scale:
            return table
        order *= 2
        table, vals = _build(callable_fn, lb, ub, order, nthreads)
    return table
