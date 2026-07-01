"""Pure-Python construction helpers over the C++ ChebTable primitives.

Sampling a Python callable and the adaptive order-doubling loop live here so no
Python callable is ever marshalled into a C++ std::function.  The numerical core
(DCT-I build, Clenshaw eval, analytic derivatives) is in C++.

Both 1-D and N-D tensor-product cases are supported.  Dispatch is on whether
``lb``/``ub``/``order`` are scalars (1-D) or sequences (N-D).
"""

import itertools
import warnings

import _tychopy.vector_functions as _vf
import numpy as np


class ChebConvergenceWarning(UserWarning):
    """Emitted when adaptive Chebyshev construction stops without meeting rtol.

    Issued (as a warning, not an exception) when :func:`cheb_adaptive` reaches
    ``max_order`` on one or more axes before the trailing-coefficient norm falls
    below ``rtol * scale``.  The returned table is the best-available fit; filter
    this category with :py:mod:`warnings` to escalate non-convergence to an error.
    """


def _oob_policy(periodic, D):
    """Translate a ``periodic`` argument into per-axis ``OutOfDomain`` enums (length D).

    ``periodic`` may be None/False (all Clamp), True (all Periodic), or a length-D
    sequence of per-axis booleans.
    """
    clamp = _vf.ChebTable.OutOfDomain.Clamp
    periodic_enum = _vf.ChebTable.OutOfDomain.Periodic
    if periodic is None or periodic is False:
        return [clamp] * D
    if periodic is True:
        return [periodic_enum] * D
    flags = list(periodic)
    if len(flags) != D:
        raise ValueError(
            f"cheb: periodic must have length {D} (one flag per axis), got {len(flags)}"
        )
    return [periodic_enum if bool(f) else clamp for f in flags]


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
        if not np.isfinite(c).all():
            raise ValueError(
                f"cheb sampler: callable returned a non-finite value at t={float(pts[j])!r}"
            )
        vals[:, j] = c
    return vals


def _build_1d(callable_fn, lb, ub, order, nthreads, oob=None):
    """Sample callable_fn and build a 1-D ChebTable; return (table, sampled values)."""
    vals = _sample_1d(callable_fn, lb, ub, order)
    if oob is None:
        oob = _vf.ChebTable.OutOfDomain.Clamp
    table = _vf.ChebTable.from_values(vals, lb, ub, order, nthreads, oob)
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
        if not np.isfinite(val).all():
            raise ValueError(
                f"cheb_from_function N-D: callable returned a non-finite value at x={pt.tolist()!r}"
            )
        grid_flat[flat_idx] = val

    return grid_flat, olen


def _build_nd(callable_fn, lb, ub, orders, nthreads, oob=None):
    """Sample callable_fn on the N-D grid and build a ChebTable; return (table, grid_flat)."""
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    orders = list(orders)
    grid_flat, _ = _sample_nd(callable_fn, lb_arr, ub_arr, orders)
    if oob is None:
        oob = []
    table = _vf.ChebTable.from_values(grid_flat, lb_arr, ub_arr, orders, nthreads, oob)
    return table, grid_flat


# ---------------------------------------------------------------------------
# Public API — kept backward-compatible for 1-D; dispatches to N-D on sequences
# ---------------------------------------------------------------------------


# Keep old names for internal backward compat
_sample = _sample_1d
_build = _build_1d


def cheb_from_function(callable_fn, lb, ub, order, nthreads=1, periodic=None):
    """Build a ChebTable by sampling a Python callable at the Chebyshev nodes.

    Supports both 1-D and N-D tensor-product cases.

    Parameters
    ----------
    callable_fn : callable
        - 1-D: ``f(t) -> scalar or array_like`` evaluated at ``order+1`` nodes.
        - N-D: ``f(x) -> scalar or array_like`` where x is a length-D array,
          evaluated at all tensor-product grid points.
        Non-finite (NaN/Inf) samples raise ``ValueError`` naming the offending node.
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
    periodic : bool or sequence of bool, optional
        Per-axis out-of-domain policy.  ``None``/``False`` (default) clamps
        out-of-domain queries to the nearest endpoint; ``True`` wraps them modulo
        the axis span (for angle-like / invariant-manifold axes — the sampled
        function should satisfy ``f(lb) ~= f(ub)`` on that axis).  For N-D, pass a
        length-D sequence to select the policy per axis.

    Returns
    -------
    ChebTable
        Chebyshev interpolant built from the sampled values.
    """
    if _is_nd(lb):
        D = len(np.asarray(lb))
        orders = [int(order)] * D if isinstance(order, int) else list(order)
        table, _ = _build_nd(
            callable_fn, lb, ub, orders, nthreads, _oob_policy(periodic, D)
        )
    else:
        table, _ = _build_1d(
            callable_fn,
            float(lb),
            float(ub),
            int(order),
            nthreads,
            _oob_policy(periodic, 1)[0],
        )
    return table


def cheb_adaptive(
    callable_fn, lb, ub, rtol=1e-12, order0=8, max_order=512, nthreads=1, periodic=None
):
    """Build a ChebTable by doubling the order until spectral convergence.

    Supports both 1-D and N-D tensor-product cases.

    For 1-D: doubles the order until the trailing-coefficient norm falls below
    ``rtol * scale``.

    For N-D: independently doubles each axis's order until all axes have
    converged.  Convergence for each axis is assessed by building a marginal
    1-D table (evaluating the function along that axis at several non-degenerate
    base points of the other axes) and checking its trailing-coefficient norm.
    This per-axis marginal test can under-resolve strongly coupled functions; a
    final residual check over random in-domain points guards against that and
    warns (see ``ChebConvergenceWarning``) if the assembled table misses ``rtol``.

    If ``max_order`` is reached before convergence, the best-available table is
    returned and a :class:`ChebConvergenceWarning` is issued (not an exception).

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
        Starting polynomial order(s).  1-D: a scalar int.  N-D: a scalar int
        (applied to every axis) or a length-D sequence of per-axis starting
        orders.  Default 8.
    max_order : int, optional
        Maximum polynomial order (per axis for N-D) before giving up.  Default 512.
    nthreads : int, optional
        Thread count forwarded to the DCT-I transform.  Defaults to 1.
    periodic : bool or sequence of bool, optional
        Per-axis out-of-domain policy for the assembled table; see
        :func:`cheb_from_function`.  Default ``None`` (clamp on every axis).

    Returns
    -------
    ChebTable
        Converged Chebyshev interpolant, or the best-available table (with a
        ``ChebConvergenceWarning``) if ``max_order`` was reached first.
    """
    if _is_nd(lb):
        return _cheb_adaptive_nd(
            callable_fn, lb, ub, rtol, order0, max_order, nthreads, periodic
        )
    else:
        return _cheb_adaptive_1d(
            callable_fn,
            float(lb),
            float(ub),
            rtol,
            order0,
            max_order,
            nthreads,
            periodic,
        )


def _cheb_adaptive_1d(
    callable_fn, lb, ub, rtol, order0, max_order, nthreads, periodic=None
):
    """1-D adaptive order-doubling loop.

    Checks convergence of the *current* table at the top of each iteration (so the
    final max_order table is checked too), and warns with ``ChebConvergenceWarning``
    if max_order is reached without meeting ``rtol``.
    """
    if order0 > max_order:
        raise ValueError(
            f"cheb_adaptive: order0 ({order0}) must not exceed max_order ({max_order})"
        )
    oob = _oob_policy(periodic, 1)[0]
    order = int(order0)
    table, vals = _build_1d(callable_fn, lb, ub, order, nthreads, oob)
    while True:
        tail = np.max(table.coeff_tail_norm())
        scale = max(np.max(np.abs(vals)), 1.0)
        if tail <= rtol * scale:
            return table
        if order >= max_order:
            warnings.warn(
                f"cheb_adaptive (1-D) did not converge to rtol={rtol:.2e} at "
                f"max_order={max_order}: trailing-coefficient norm {tail:.3e} vs "
                f"target {rtol * scale:.3e} on [{lb}, {ub}]. Returning best-available "
                f"order-{order} table (increase max_order, or check for a "
                f"discontinuity/singularity in the domain).",
                ChebConvergenceWarning,
                stacklevel=2,
            )
            return table
        order = min(order * 2, max_order)
        table, vals = _build_1d(callable_fn, lb, ub, order, nthreads, oob)


def _warn_if_residual_exceeds(callable_fn, table, lb_arr, ub_arr):
    """Coarse coupling check: warn if the assembled N-D table is grossly under-resolved.

    The per-axis marginal convergence test can be fooled by strong cross-axis
    coupling (e.g. ``sin(x*y)``): every 1-D slice looks converged while the full
    field is not.  This samples the callable at fixed pseudo-random in-domain
    points and warns (``ChebConvergenceWarning``) if the pointwise residual
    exceeds a deliberately coarse relative tolerance.  It is a gross-failure net,
    not a precision gate, so its threshold is independent of ``rtol``.
    """
    resid_rtol = 1e-6
    D = len(lb_arr)
    rng = np.random.default_rng(0)  # fixed seed → deterministic warnings
    max_res = 0.0
    scale = 1.0
    for _ in range(64):
        x = lb_arr + (ub_arr - lb_arr) * rng.random(D)
        f_true = np.atleast_1d(np.asarray(callable_fn(x), dtype=float))
        f_hat = np.atleast_1d(np.asarray(table.eval(x), dtype=float))
        max_res = max(max_res, float(np.max(np.abs(f_true - f_hat))))
        scale = max(scale, float(np.max(np.abs(f_true))))
    if max_res > resid_rtol * scale:
        warnings.warn(
            f"cheb_adaptive (N-D): per-axis marginals reported convergence but the "
            f"assembled table has pointwise residual {max_res:.3e} (> {resid_rtol:.0e} "
            f"* scale) at random in-domain points — the function may be strongly "
            f"coupled across axes. Build with explicit higher per-axis orders if so.",
            ChebConvergenceWarning,
            stacklevel=3,
        )


def _cheb_adaptive_nd(
    callable_fn, lb, ub, rtol, order0, max_order, nthreads, periodic=None
):
    """N-D adaptive order-doubling loop with per-axis convergence checking.

    Each axis is independently refined: marginal 1-D tables are built for
    each axis (holding all other axes at several non-degenerate base points,
    offset from the domain centres to avoid axes where the function is
    identically zero), and their trailing-coefficient norm is used as the
    per-axis convergence indicator.  Axes that reach ``max_order`` without
    converging trigger a ``ChebConvergenceWarning``; a final residual check
    guards against the marginal heuristic being fooled by cross-axis coupling.
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
            # Marginal probes are tiny (order_d+1)-point DCTs; spinning up a
            # thread pool per probe would cost more than it saves, so force
            # single-threaded here and reserve `nthreads` for the final build.
            tab1d = _vf.ChebTable.from_values(vals_1d, lb_arr[d], ub_arr[d], order_d, 1)
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

    table, _ = _build_nd(
        callable_fn, lb_arr, ub_arr, orders, nthreads, _oob_policy(periodic, D)
    )

    # Non-convergence signal: axes that reached max_order without converging.
    capped = [d for d in range(D) if not axis_done[d]]
    if capped:
        warnings.warn(
            f"cheb_adaptive (N-D) did not converge to rtol={rtol:.2e} on axes "
            f"{capped} at max_order={max_order}; returning best-available table "
            f"(orders={orders}).",
            ChebConvergenceWarning,
            stacklevel=2,
        )
    else:
        # All axes' marginals converged — verify the assembled field didn't get
        # fooled by cross-axis coupling.
        _warn_if_residual_exceeds(callable_fn, table, lb_arr, ub_arr)

    return table
