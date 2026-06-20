from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

from . import kepler as kepler
import _tychopy.vector_functions


@overload
def cartesian_to_classic(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Convert a Cartesian state to classical orbital elements (mean anomaly).

    Parameters
    ----------
    XV : array_like, shape (6,)
        Cartesian state ``[rx, ry, rz, vx, vy, vz]``.  Units must be
        consistent with ``mu`` (e.g. km and km/s).
    mu : float
        Gravitational parameter (km³/s² or consistent units).  Must be > 0.

    Returns
    -------
    ndarray, shape (6,)
        Classical elements ``[a, e, i, Omega, omega, M]``, where

        * ``a`` — semi-major axis (< 0 for hyperbolic orbits).
        * ``e`` — eccentricity.
        * ``i``, ``Omega``, ``omega`` — inclination, RAAN, argument of perigee (radians).
        * ``M`` — mean anomaly (radians): elliptic (M = E − e sin E) for e < 1,
          hyperbolic (M = e sinh H − H) for e > 1.

    See Also
    --------
    cartesian_to_classic_true : Same conversion but returns true anomaly.
    classic_to_cartesian : Inverse conversion.
    """

@overload
def cartesian_to_classic(arg0: _tychopy.vector_functions.VectorFunction, arg1: float, /) -> _tychopy.vector_functions.VectorFunction:
    """
    Compose a Cartesian-to-classical conversion with a VectorFunction.

    Returns a new VectorFunction that evaluates ``RV`` and then converts its
    Cartesian output ``[rx, ry, rz, vx, vy, vz]`` to classical orbital elements
    ``[a, e, i, Omega, omega, M]`` (M = mean anomaly).

    Parameters
    ----------
    RV : VectorFunction
        A VectorFunction whose output is a Cartesian state vector (shape 6).
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    VectorFunction
        Composed VectorFunction with output shape (6,) in classical elements.

    Notes
    -----
    For numeric-array input use the ``(array, mu)`` overload.
    """

@overload
def cartesian_to_modified(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Convert a Cartesian state to Modified Equinoctial Elements (MEE).

    Direct closed-form conversion with no Kepler-equation iteration.
    Singular at exactly i = 180° (retrograde equatorial); all other
    inclinations including non-equatorial retrograde are supported.

    Parameters
    ----------
    XV : array_like, shape (6,)
        Cartesian state ``[rx, ry, rz, vx, vy, vz]``.
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    ndarray, shape (6,)
        MEE ``[p, f, g, h, k, L]`` where ``L`` is the true longitude (radians).

    See Also
    --------
    modified_to_cartesian : Inverse conversion.
    """

@overload
def cartesian_to_modified(arg0: _tychopy.vector_functions.VectorFunction, arg1: float, /) -> _tychopy.vector_functions.VectorFunction:
    """
    Compose a Cartesian-to-MEE conversion with a VectorFunction.

    Returns a new VectorFunction that evaluates ``RV`` and then converts its
    Cartesian output ``[rx, ry, rz, vx, vy, vz]`` to MEE ``[p, f, g, h, k, L]``.

    Parameters
    ----------
    RV : VectorFunction
        A VectorFunction whose output is a Cartesian state vector (shape 6).
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    VectorFunction
        Composed VectorFunction with output shape (6,) in MEE.

    Notes
    -----
    For numeric-array input use the ``(array, mu)`` overload.
    """

def classic_to_cartesian(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Convert classical orbital elements to a Cartesian state.

    Solves Kepler's equation via Newton iteration (elliptic) or the hyperbolic
    analogue, then rotates to the inertial frame via the Euler-angle sequence
    (Omega, i, omega).

    Parameters
    ----------
    oelems : array_like, shape (6,)
        Classical elements ``[a, e, i, Omega, omega, M]``.  For hyperbolic
        orbits ``a < 0`` by convention and ``M`` is the hyperbolic mean anomaly
        (M = e sinh H − H).  All angles in radians.
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    ndarray, shape (6,)
        Cartesian state ``[rx, ry, rz, vx, vy, vz]``.

    See Also
    --------
    cartesian_to_classic : Inverse conversion.
    """

def classic_to_modified(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Convert classical orbital elements to Modified Equinoctial Elements (MEE).

    Solves Kepler's equation for the true anomaly from the mean anomaly, then
    computes MEE algebraically.

    Parameters
    ----------
    oelems : array_like, shape (6,)
        Classical elements ``[a, e, i, Omega, omega, M]``.  All angles in
        radians.  ``mu`` is accepted for API symmetry but is not used in the
        algebraic conversion.
    mu : float
        Gravitational parameter (not used; accepted for API symmetry).

    Returns
    -------
    ndarray, shape (6,)
        MEE ``[p, f, g, h, k, L]`` where ``L`` is the true longitude (radians).

    See Also
    --------
    modified_to_classic : Inverse conversion.
    """

@overload
def modified_to_cartesian(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Convert Modified Equinoctial Elements (MEE) to a Cartesian state.

    Direct closed-form conversion using the standard MEE frame vectors.

    Parameters
    ----------
    meelems : array_like, shape (6,)
        MEE ``[p, f, g, h, k, L]``: semi-latus rectum ``p``, equinoctial
        eccentricity components ``f``, ``g``, inclination components ``h``,
        ``k``, and true longitude ``L`` (radians).
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    ndarray, shape (6,)
        Cartesian state ``[rx, ry, rz, vx, vy, vz]``.

    See Also
    --------
    cartesian_to_modified : Inverse conversion.
    """

@overload
def modified_to_cartesian(arg0: _tychopy.vector_functions.VectorFunction, arg1: float, /) -> _tychopy.vector_functions.VectorFunction:
    """
    Compose a MEE-to-Cartesian conversion with a VectorFunction.

    Returns a new VectorFunction that evaluates ``meelems`` and then converts
    its MEE output ``[p, f, g, h, k, L]`` to Cartesian ``[rx, ry, rz, vx, vy, vz]``.

    Parameters
    ----------
    meelems : VectorFunction
        A VectorFunction whose output is MEE ``[p, f, g, h, k, L]``.
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    VectorFunction
        Composed VectorFunction with output shape (6,) in Cartesian coordinates.

    Notes
    -----
    For numeric-array input use the ``(array, mu)`` overload.
    """

def modified_to_classic(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Convert Modified Equinoctial Elements (MEE) to classical orbital elements.

    Uses algebraic relations from MEE to classical elements, then solves the
    true-anomaly-to-mean-anomaly step analytically.

    Parameters
    ----------
    meelems : array_like, shape (6,)
        MEE ``[p, f, g, h, k, L]`` where ``L`` is the true longitude (radians).
    mu : float
        Gravitational parameter (not used; accepted for API symmetry).

    Returns
    -------
    ndarray, shape (6,)
        Classical elements ``[a, e, i, Omega, omega, M]``, where ``M`` is the
        elliptic mean anomaly (e < 1) or hyperbolic mean anomaly (e > 1).
        All angles in radians.

    See Also
    --------
    classic_to_modified : Inverse conversion.
    """

def cartesian_to_classic_true(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Convert a Cartesian state to classical orbital elements (true anomaly).

    Like :func:`cartesian_to_classic` but returns true anomaly ``nu`` as the
    sixth element instead of mean anomaly ``M``.

    Parameters
    ----------
    XV : array_like, shape (6,)
        Cartesian state ``[rx, ry, rz, vx, vy, vz]``.
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    ndarray, shape (6,)
        Classical elements ``[a, e, i, Omega, omega, nu]`` where ``nu`` is the
        true anomaly (radians).

    See Also
    --------
    cartesian_to_classic : Identical conversion but returns mean anomaly.
    """

class ModifiedToCartesian:
    """
    MEE-to-Cartesian conversion VectorFunction (IR=6, OR=6).

    Accepts Modified Equinoctial Elements ``[p, f, g, h, k, L]`` and returns
    Cartesian state ``[rx, ry, rz, vx, vy, vz]`` via direct closed-form
    conversion.  Useful for embedding element conversions inside an optimal
    control expression graph.

    Parameters
    ----------
    mu : float
        Gravitational parameter (must be > 0).

    See Also
    --------
    modified_to_cartesian : Scalar overload for numeric arrays.
    """

    def __init__(self, arg: float, /) -> None: ...

    def input_rows(self) -> int:
        """
        Number of inputs this VectorFunction expects.

        Returns
        -------
        int
            Length of the input vector ``x`` passed to :meth:`compute`, :meth:`jacobian`,
            and the adjoint methods.
        """

    def output_rows(self) -> int:
        """
        Number of scalar outputs this VectorFunction produces.

        Returns
        -------
        int
            Length of the output vector ``f(x)`` returned by :meth:`compute`.
        """

    def name(self) -> str:
        """
        Human-readable name of this VectorFunction.

        The name is derived from the C++ type name of the underlying expression.
        It is primarily used for diagnostics and display.

        Returns
        -------
        str
            Type-based name string.
        """

    def input_domain(self) -> Annotated[NDArray[numpy.int32], dict(shape=(2, None), order='F')]:
        """
        Return the sparsity domain describing which input indices this function uses.

        The input domain encodes which contiguous sub-ranges of the input vector
        this function actually depends on.  Dynamic-size functions compute the domain
        at construction time; static-size functions return the full range ``[0, IR)``.

        Returns
        -------
        ndarray, shape (2, k), dtype int
            Column matrix with ``k`` sub-ranges.  Row 0 contains start indices,
            row 1 contains the corresponding lengths.
        """

    def is_linear(self) -> bool:
        """
        Whether this VectorFunction is known to be linear.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.
        For type-erased ``GenericFunction`` objects this value is determined at
        construction time and cached; for concrete (statically-typed) functions
        it is a compile-time constant propagated through the type system.

        Returns
        -------
        bool
            ``True`` if the function is linear, ``False`` otherwise.
        """

    @overload
    def compute(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.  Accepts a NumPy array (zero-copy) or any Python
            sequence (list, tuple) that can be converted to a 1-D float64 vector.

        Returns
        -------
        ndarray, shape (output_rows,)
            Output vector ``f(x)``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`compute` overload.
        """

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  This overload accepts a numeric vector.
        To compose with another VectorFunction use :meth:`eval` or pass a
        VectorFunction argument — those are handled by separate ``__call__``
        overloads.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.  Accepts a NumPy array (zero-copy) or any Python sequence.

        Returns
        -------
        ndarray, shape (output_rows,)
            Output vector ``f(x)``.
        """

    @overload
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray:
        """Overload accepting a Python list or tuple; see :meth:`compute`."""

    @overload
    def jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Evaluate the Jacobian of the function at a point.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate the Jacobian.

        Returns
        -------
        ndarray, shape (output_rows, input_rows)
            Jacobian matrix ``J(x)`` where entry ``(i, j)`` is
            ``∂f_i/∂x_j`` evaluated at ``x``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`jacobian` overload.
        """

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        For many derivative modes this avoids redundant function evaluations
        compared to two separate calls to :meth:`compute` and :meth:`jacobian`.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.

        Returns
        -------
        fx : ndarray, shape (output_rows,)
            Function value ``f(x)``.
        jx : ndarray, shape (output_rows, input_rows)
            Jacobian ``J(x)``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`compute_jacobian` overload.
        """

    @overload
    def adjointgradient(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Compute the adjoint (Lagrange-multiplier-weighted) gradient.

        Evaluates ``g = Jᵀ λ`` where ``J`` is the Jacobian of ``self`` at ``x`` and
        ``λ`` (``lm``) is the adjoint (multiplier) vector.  This is the first-order
        sensitivity of the Lagrangian with respect to the inputs.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector, one entry per output.

        Returns
        -------
        ndarray, shape (input_rows,)
            Adjoint gradient vector ``Jᵀ λ``.

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`adjointgradient` overload.
        """

    @overload
    def adjointhessian(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Compute the adjoint (Lagrange-multiplier-weighted) Hessian.

        Evaluates the symmetric matrix ``H = Σ_k λ_k ∇²f_k(x)``, i.e. the
        multiplier-weighted sum of the per-output Hessians.  This is the
        second-order term of the Lagrangian's Hessian with respect to the inputs.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector, one entry per output.

        Returns
        -------
        ndarray, shape (input_rows, input_rows)
            Adjoint Hessian matrix (symmetric).

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`adjointhessian` overload.
        """

    @overload
    def computeall(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function value, Jacobian, adjoint gradient, and adjoint Hessian in one call.

        All four quantities are computed together, sharing internal buffers, which is
        more efficient than invoking them separately.  This is the entry point used
        by PSIOPT during second-order optimization.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector used to form the weighted gradient
            and Hessian.

        Returns
        -------
        fx : ndarray, shape (output_rows,)
            Function value ``f(x)``.
        jx : ndarray, shape (output_rows, input_rows)
            Jacobian ``J(x)``.
        gx : ndarray, shape (input_rows,)
            Adjoint gradient ``Jᵀ λ``.
        hx : ndarray, shape (input_rows, input_rows)
            Adjoint Hessian ``Σ_k λ_k ∇²f_k(x)``.

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`computeall` overload.
        """

    def vf(self) -> _tychopy.vector_functions.VectorFunction:
        """
        Type-erase this expression into a generic VectorFunction.

        Wraps the current expression in a fully dynamic ``GenericFunction`` that can
        be stored, passed to functions expecting a ``GenericFunction``, or used as an
        operand in mixed-type expressions without exposing the underlying expression
        template type.

        Returns
        -------
        GenericFunction
            A dynamically-typed wrapper around this function.
        """

class CartesianToClassic:
    """
    Cartesian-to-classical orbital elements differentiable VectorFunction (IR=6, OR=6).

    Accepts Cartesian state ``[rx, ry, rz, vx, vy, vz]`` and computes classical
    orbital elements ``[a, e, i, Omega, omega, M]`` (with mean anomaly ``M``) as a
    differentiable VectorFunction expression for embedding inside optimal-control
    expression graphs.

    Parameters
    ----------
    mu : float
        Gravitational parameter (must be > 0).

    Notes
    -----
    Produces the same classical elements as the scalar
    :func:`cartesian_to_classic` (mean anomaly ``M``), as a differentiable
    VectorFunction suitable for embedding in expression graphs. For direct numeric
    conversion of array inputs, use :func:`cartesian_to_classic`.
    """

    def __init__(self, arg: float, /) -> None: ...

    def input_rows(self) -> int:
        """
        Number of inputs this VectorFunction expects.

        Returns
        -------
        int
            Length of the input vector ``x`` passed to :meth:`compute`, :meth:`jacobian`,
            and the adjoint methods.
        """

    def output_rows(self) -> int:
        """
        Number of scalar outputs this VectorFunction produces.

        Returns
        -------
        int
            Length of the output vector ``f(x)`` returned by :meth:`compute`.
        """

    def name(self) -> str:
        """
        Human-readable name of this VectorFunction.

        The name is derived from the C++ type name of the underlying expression.
        It is primarily used for diagnostics and display.

        Returns
        -------
        str
            Type-based name string.
        """

    def input_domain(self) -> Annotated[NDArray[numpy.int32], dict(shape=(2, None), order='F')]:
        """
        Return the sparsity domain describing which input indices this function uses.

        The input domain encodes which contiguous sub-ranges of the input vector
        this function actually depends on.  Dynamic-size functions compute the domain
        at construction time; static-size functions return the full range ``[0, IR)``.

        Returns
        -------
        ndarray, shape (2, k), dtype int
            Column matrix with ``k`` sub-ranges.  Row 0 contains start indices,
            row 1 contains the corresponding lengths.
        """

    def is_linear(self) -> bool:
        """
        Whether this VectorFunction is known to be linear.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.
        For type-erased ``GenericFunction`` objects this value is determined at
        construction time and cached; for concrete (statically-typed) functions
        it is a compile-time constant propagated through the type system.

        Returns
        -------
        bool
            ``True`` if the function is linear, ``False`` otherwise.
        """

    @overload
    def compute(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.  Accepts a NumPy array (zero-copy) or any Python
            sequence (list, tuple) that can be converted to a 1-D float64 vector.

        Returns
        -------
        ndarray, shape (output_rows,)
            Output vector ``f(x)``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`compute` overload.
        """

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  This overload accepts a numeric vector.
        To compose with another VectorFunction use :meth:`eval` or pass a
        VectorFunction argument — those are handled by separate ``__call__``
        overloads.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.  Accepts a NumPy array (zero-copy) or any Python sequence.

        Returns
        -------
        ndarray, shape (output_rows,)
            Output vector ``f(x)``.
        """

    @overload
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray:
        """Overload accepting a Python list or tuple; see :meth:`compute`."""

    @overload
    def jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Evaluate the Jacobian of the function at a point.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate the Jacobian.

        Returns
        -------
        ndarray, shape (output_rows, input_rows)
            Jacobian matrix ``J(x)`` where entry ``(i, j)`` is
            ``∂f_i/∂x_j`` evaluated at ``x``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`jacobian` overload.
        """

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        For many derivative modes this avoids redundant function evaluations
        compared to two separate calls to :meth:`compute` and :meth:`jacobian`.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.

        Returns
        -------
        fx : ndarray, shape (output_rows,)
            Function value ``f(x)``.
        jx : ndarray, shape (output_rows, input_rows)
            Jacobian ``J(x)``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`compute_jacobian` overload.
        """

    @overload
    def adjointgradient(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Compute the adjoint (Lagrange-multiplier-weighted) gradient.

        Evaluates ``g = Jᵀ λ`` where ``J`` is the Jacobian of ``self`` at ``x`` and
        ``λ`` (``lm``) is the adjoint (multiplier) vector.  This is the first-order
        sensitivity of the Lagrangian with respect to the inputs.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector, one entry per output.

        Returns
        -------
        ndarray, shape (input_rows,)
            Adjoint gradient vector ``Jᵀ λ``.

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`adjointgradient` overload.
        """

    @overload
    def adjointhessian(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Compute the adjoint (Lagrange-multiplier-weighted) Hessian.

        Evaluates the symmetric matrix ``H = Σ_k λ_k ∇²f_k(x)``, i.e. the
        multiplier-weighted sum of the per-output Hessians.  This is the
        second-order term of the Lagrangian's Hessian with respect to the inputs.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector, one entry per output.

        Returns
        -------
        ndarray, shape (input_rows, input_rows)
            Adjoint Hessian matrix (symmetric).

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`adjointhessian` overload.
        """

    @overload
    def computeall(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function value, Jacobian, adjoint gradient, and adjoint Hessian in one call.

        All four quantities are computed together, sharing internal buffers, which is
        more efficient than invoking them separately.  This is the entry point used
        by PSIOPT during second-order optimization.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector used to form the weighted gradient
            and Hessian.

        Returns
        -------
        fx : ndarray, shape (output_rows,)
            Function value ``f(x)``.
        jx : ndarray, shape (output_rows, input_rows)
            Jacobian ``J(x)``.
        gx : ndarray, shape (input_rows,)
            Adjoint gradient ``Jᵀ λ``.
        hx : ndarray, shape (input_rows, input_rows)
            Adjoint Hessian ``Σ_k λ_k ∇²f_k(x)``.

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`computeall` overload.
        """

    def vf(self) -> _tychopy.vector_functions.VectorFunction:
        """
        Type-erase this expression into a generic VectorFunction.

        Wraps the current expression in a fully dynamic ``GenericFunction`` that can
        be stored, passed to functions expecting a ``GenericFunction``, or used as an
        operand in mixed-type expressions without exposing the underlying expression
        template type.

        Returns
        -------
        GenericFunction
            A dynamically-typed wrapper around this function.
        """

class CartesianToMEE:
    """
    Cartesian-to-MEE conversion VectorFunction (IR=6, OR=6).

    Accepts Cartesian state ``[rx, ry, rz, vx, vy, vz]`` and computes Modified
    Equinoctial Elements ``[p, f, g, h, k, L]`` (L = true longitude).
    Implements the direct closed-form conversion as a differentiable
    VectorFunction for use inside optimal control expression graphs.

    Parameters
    ----------
    mu : float
        Gravitational parameter (must be > 0).

    See Also
    --------
    cartesian_to_modified : Scalar overload for numeric arrays.
    """

    def __init__(self, arg: float, /) -> None: ...

    def input_rows(self) -> int:
        """
        Number of inputs this VectorFunction expects.

        Returns
        -------
        int
            Length of the input vector ``x`` passed to :meth:`compute`, :meth:`jacobian`,
            and the adjoint methods.
        """

    def output_rows(self) -> int:
        """
        Number of scalar outputs this VectorFunction produces.

        Returns
        -------
        int
            Length of the output vector ``f(x)`` returned by :meth:`compute`.
        """

    def name(self) -> str:
        """
        Human-readable name of this VectorFunction.

        The name is derived from the C++ type name of the underlying expression.
        It is primarily used for diagnostics and display.

        Returns
        -------
        str
            Type-based name string.
        """

    def input_domain(self) -> Annotated[NDArray[numpy.int32], dict(shape=(2, None), order='F')]:
        """
        Return the sparsity domain describing which input indices this function uses.

        The input domain encodes which contiguous sub-ranges of the input vector
        this function actually depends on.  Dynamic-size functions compute the domain
        at construction time; static-size functions return the full range ``[0, IR)``.

        Returns
        -------
        ndarray, shape (2, k), dtype int
            Column matrix with ``k`` sub-ranges.  Row 0 contains start indices,
            row 1 contains the corresponding lengths.
        """

    def is_linear(self) -> bool:
        """
        Whether this VectorFunction is known to be linear.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.
        For type-erased ``GenericFunction`` objects this value is determined at
        construction time and cached; for concrete (statically-typed) functions
        it is a compile-time constant propagated through the type system.

        Returns
        -------
        bool
            ``True`` if the function is linear, ``False`` otherwise.
        """

    @overload
    def compute(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.  Accepts a NumPy array (zero-copy) or any Python
            sequence (list, tuple) that can be converted to a 1-D float64 vector.

        Returns
        -------
        ndarray, shape (output_rows,)
            Output vector ``f(x)``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`compute` overload.
        """

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  This overload accepts a numeric vector.
        To compose with another VectorFunction use :meth:`eval` or pass a
        VectorFunction argument — those are handled by separate ``__call__``
        overloads.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.  Accepts a NumPy array (zero-copy) or any Python sequence.

        Returns
        -------
        ndarray, shape (output_rows,)
            Output vector ``f(x)``.
        """

    @overload
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray:
        """Overload accepting a Python list or tuple; see :meth:`compute`."""

    @overload
    def jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Evaluate the Jacobian of the function at a point.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate the Jacobian.

        Returns
        -------
        ndarray, shape (output_rows, input_rows)
            Jacobian matrix ``J(x)`` where entry ``(i, j)`` is
            ``∂f_i/∂x_j`` evaluated at ``x``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`jacobian` overload.
        """

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        For many derivative modes this avoids redundant function evaluations
        compared to two separate calls to :meth:`compute` and :meth:`jacobian`.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector.

        Returns
        -------
        fx : ndarray, shape (output_rows,)
            Function value ``f(x)``.
        jx : ndarray, shape (output_rows, input_rows)
            Jacobian ``J(x)``.

        Raises
        ------
        ValueError
            If ``x`` has the wrong length.
        """

    @overload
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Overload accepting a Python list or tuple; see the primary :meth:`compute_jacobian` overload.
        """

    @overload
    def adjointgradient(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Compute the adjoint (Lagrange-multiplier-weighted) gradient.

        Evaluates ``g = Jᵀ λ`` where ``J`` is the Jacobian of ``self`` at ``x`` and
        ``λ`` (``lm``) is the adjoint (multiplier) vector.  This is the first-order
        sensitivity of the Lagrangian with respect to the inputs.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector, one entry per output.

        Returns
        -------
        ndarray, shape (input_rows,)
            Adjoint gradient vector ``Jᵀ λ``.

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`adjointgradient` overload.
        """

    @overload
    def adjointhessian(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Compute the adjoint (Lagrange-multiplier-weighted) Hessian.

        Evaluates the symmetric matrix ``H = Σ_k λ_k ∇²f_k(x)``, i.e. the
        multiplier-weighted sum of the per-output Hessians.  This is the
        second-order term of the Lagrangian's Hessian with respect to the inputs.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector, one entry per output.

        Returns
        -------
        ndarray, shape (input_rows, input_rows)
            Adjoint Hessian matrix (symmetric).

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`adjointhessian` overload.
        """

    @overload
    def computeall(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function value, Jacobian, adjoint gradient, and adjoint Hessian in one call.

        All four quantities are computed together, sharing internal buffers, which is
        more efficient than invoking them separately.  This is the entry point used
        by PSIOPT during second-order optimization.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate.
        lm : array_like, shape (output_rows,)
            Adjoint (Lagrange multiplier) vector used to form the weighted gradient
            and Hessian.

        Returns
        -------
        fx : ndarray, shape (output_rows,)
            Function value ``f(x)``.
        jx : ndarray, shape (output_rows, input_rows)
            Jacobian ``J(x)``.
        gx : ndarray, shape (input_rows,)
            Adjoint gradient ``Jᵀ λ``.
        hx : ndarray, shape (input_rows, input_rows)
            Adjoint Hessian ``Σ_k λ_k ∇²f_k(x)``.

        Raises
        ------
        ValueError
            If ``x`` or ``lm`` has the wrong length.
        """

    @overload
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Overload accepting Python lists or tuples; see the primary :meth:`computeall` overload.
        """

    def vf(self) -> _tychopy.vector_functions.VectorFunction:
        """
        Type-erase this expression into a generic VectorFunction.

        Wraps the current expression in a fully dynamic ``GenericFunction`` that can
        be stored, passed to functions expecting a ``GenericFunction``, or used as an
        operand in mixed-type expressions without exposing the underlying expression
        template type.

        Returns
        -------
        GenericFunction
            A dynamically-typed wrapper around this function.
        """

def propagate_cartesian(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, arg2: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Propagate a Cartesian state forward under two-body gravity.

    Uses the universal-variable LCD (Stumpff-function) iteration followed by
    the Goodyear f-g map.  Supports elliptic and hyperbolic trajectories.

    Parameters
    ----------
    RV : array_like, shape (6,)
        Initial Cartesian state ``[rx, ry, rz, vx, vy, vz]``.
    dt : float
        Time-of-flight.  Units must be consistent with ``mu``.
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    ndarray, shape (6,)
        Propagated Cartesian state ``[rx', ry', rz', vx', vy', vz']``.

    Raises
    ------
    ValueError
        If ``mu <= 0``, ``dt`` is non-finite, ``RV`` contains non-finite
        entries, or ``|R0| == 0``.
    RuntimeError
        If the LCD iteration fails to converge (near-parabolic orbits, very
        large time steps) or the hyperbolic-asymptote guard fires
        (``sqrt(-alpha)*X ≈ 30``).
    """

def propagate_classic(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, arg2: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Propagate classical orbital elements forward by a time step.

    Advances the mean anomaly analytically: ``M_new = M + n*dt`` where
    ``n = sqrt(mu / |a|^3)`` is the mean motion.  Only the mean anomaly
    element (index 5) is updated; all other elements are unchanged.

    Parameters
    ----------
    oelems : array_like, shape (6,)
        Classical elements ``[a, e, i, Omega, omega, M]``.  All angles in
        radians.
    dt : float
        Time-of-flight.  Units consistent with ``mu``.
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    ndarray, shape (6,)
        Propagated classical elements with updated mean anomaly.

    Raises
    ------
    ValueError
        If ``mu <= 0`` or the semi-major axis is zero or non-finite.
    RuntimeError
        If the propagation produces non-finite output.
    """

def propagate_modified(arg0: Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')], arg1: float, arg2: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(6), order='C')]:
    """
    Propagate Modified Equinoctial Elements (MEE) forward by a time step.

    Round-trips through classical elements for the analytic mean-anomaly update:
    MEE → classic → propagate_classic → classic → MEE.

    Parameters
    ----------
    meelems : array_like, shape (6,)
        MEE ``[p, f, g, h, k, L]`` where ``L`` is the true longitude (radians).
    dt : float
        Time-of-flight.  Units consistent with ``mu``.
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    ndarray, shape (6,)
        Propagated MEE state.

    Raises
    ------
    ValueError
        If ``mu <= 0``, or if the semi-major axis derived from the MEE
        elements is zero or non-finite (raised by the internal
        ``propagate_classic`` call).
    RuntimeError
        If the propagation produces non-finite output.
    """

@overload
def lambert_izzo(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg2: float, arg3: float, arg4: bool, /) -> tuple:
    """
    Solve a Lambert boundary-value problem (zero-revolution, scalar).

    Uses Izzo's algorithm to find the transfer orbit connecting ``R1`` to ``R2``
    in time ``dt``.

    Parameters
    ----------
    R1 : array_like, shape (3,)
        Departure position vector.  Units consistent with ``mu``.
    R2 : array_like, shape (3,)
        Arrival position vector.  Same units as ``R1``.
    dt : float
        Transfer time-of-flight (must be > 0).
    mu : float
        Gravitational parameter (must be > 0).
    longway : bool
        ``True`` for the long-way transfer (transfer angle > 180°);
        ``False`` for the short-way transfer.

    Returns
    -------
    tuple[ndarray shape (3,), ndarray shape (3,)]
        ``(V1, V2)`` — departure and arrival velocity vectors.

    Raises
    ------
    RuntimeError
        If the iteration fails to converge within 20 iterations.

    See Also
    --------
    lambert_izzo_multirev : Multi-revolution overload.
    """

@overload
def lambert_izzo(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg2: float, arg3: float, arg4: bool, arg5: int, arg6: bool, /) -> tuple:
    """
    Solve a Lambert boundary-value problem (multi-revolution, scalar).

    Overload that additionally accepts a revolution count and branch selector.
    For zero revolutions (``Nrevs=0``) this is equivalent to the 5-argument
    overload.

    Parameters
    ----------
    R1 : array_like, shape (3,)
        Departure position vector.
    R2 : array_like, shape (3,)
        Arrival position vector.
    dt : float
        Transfer time-of-flight (must be > 0).
    mu : float
        Gravitational parameter (must be > 0).
    longway : bool
        ``True`` for the long-way (>180°) transfer.
    Nrevs : int
        Number of complete revolutions (0 = direct transfer).
    rightbranch : bool
        For ``Nrevs > 0``: ``True`` selects the right branch of the
        multi-revolution solution family, ``False`` selects the left branch.

    Returns
    -------
    tuple[ndarray shape (3,), ndarray shape (3,)]
        ``(V1, V2)`` — departure and arrival velocity vectors.

    Raises
    ------
    RuntimeError
        If the iteration fails to converge within 20 iterations.
    """

@overload
def lambert_izzo(arg0: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)], arg2: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], arg3: float, arg4: Sequence[bool], arg5: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], arg6: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], arg7: int, arg8: bool, /) -> numpy.ndarray:
    """
    Solve a batch of Lambert problems in-place (vectorized overload).

    Solves ``N`` independent Lambert problems simultaneously, writing results
    directly into pre-allocated output matrices.  Uses 8-wide SuperScalar SIMD
    packing when ``vectorize=True`` for throughput-critical applications.

    Parameters
    ----------
    R1s : ndarray, shape (3, N) or (N, 3)
        Departure position vectors.  Layout selected by ``axis``.
    R2s : ndarray, shape (3, N) or (N, 3)
        Arrival position vectors.  Same layout as ``R1s``.
    dts : ndarray, shape (N,)
        Transfer times for each problem.
    mu : float
        Gravitational parameter (scalar, same for all problems).
    longways : list of bool, length N
        Long-way flag for each problem.
    V1s : ndarray, shape (3, N) or (N, 3), writeable
        Output buffer for departure velocities.  Must be pre-allocated.
    V2s : ndarray, shape (3, N) or (N, 3), writeable
        Output buffer for arrival velocities.  Must be pre-allocated.
    axis : int
        ``0`` if vectors are stored as columns (shape ``(3, N)``);
        ``1`` if vectors are stored as rows (shape ``(N, 3)``).
    vectorize : bool
        If ``True``, processes problems in packs of 8 using SuperScalar SIMD
        for higher throughput.  The remainder (``N mod 8``) is handled
        scalar-wise.

    Returns
    -------
    ndarray of int, shape (N,)
        Per-problem exit codes: ``0`` = converged, ``1`` = not converged
        within 20 iterations.

    Notes
    -----
    Unlike the scalar overloads this function does **not** raise on
    non-convergence; inspect the returned exit-code vector instead.
    """

def lambert_izzo_multirev(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg2: float, arg3: float, arg4: bool, arg5: int, arg6: bool, /) -> tuple:
    """
    Solve a multi-revolution Lambert problem (alias for ``lambert_izzo`` 7-arg overload).

    Equivalent to ``lambert_izzo(R1, R2, dt, mu, longway, Nrevs, rightbranch)``.
    Provided as a named alias to make call sites self-documenting.

    Parameters
    ----------
    R1 : array_like, shape (3,)
        Departure position vector.
    R2 : array_like, shape (3,)
        Arrival position vector.
    dt : float
        Transfer time-of-flight (must be > 0).
    mu : float
        Gravitational parameter (must be > 0).
    longway : bool
        ``True`` for the long-way (>180°) transfer.
    Nrevs : int
        Number of complete revolutions (must be > 0 for a multi-rev solution).
    rightbranch : bool
        ``True`` selects the right branch; ``False`` selects the left branch.

    Returns
    -------
    tuple[ndarray shape (3,), ndarray shape (3,)]
        ``(V1, V2)`` — departure and arrival velocity vectors.

    Raises
    ------
    RuntimeError
        If the iteration fails to converge within 20 iterations.
    """

def modified_dynamics(arg: float, /) -> _tychopy.vector_functions.VectorFunction:
    """
    Build the MEE two-body + RSW-thrust dynamics VectorFunction (IR=9, OR=6).

    Returns a VectorFunction suitable for use as the ODE of an optimal-control
    :class:`~tychopy.optimal_control.PhaseInterface`.

    Input layout ``[state(6), control(3)]``:

    * State (indices 0–5): ``[p, f, g, h, k, L]`` — Modified Equinoctial
      Elements where ``L`` is the true longitude (radians).
    * Control (indices 6–8): ``[ur, ut, un]`` — RSW-frame accelerations
      (radial, tangential/along-track, out-of-plane/normal).

    Output (6,): state derivative ``[dp/dt, df/dt, dg/dt, dh/dt, dk/dt, dL/dt]``.

    Parameters
    ----------
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    VectorFunction (IR=9, OR=6)
        MEE equations of motion with analytic Jacobian and Hessian.

    See Also
    --------
    cartesian_dynamics : Equivalent model in Cartesian coordinates.
    """

def cartesian_dynamics(arg: float, /) -> _tychopy.vector_functions.VectorFunction:
    """
    Build the Cartesian two-body + thrust dynamics VectorFunction (IR=9, OR=6).

    Returns a VectorFunction suitable for use as the ODE of an optimal-control
    :class:`~tychopy.optimal_control.PhaseInterface`.

    Input layout ``[state(6), control(3)]``:

    * State (indices 0–5): ``[rx, ry, rz, vx, vy, vz]`` — inertial Cartesian
      position and velocity.
    * Control (indices 6–8): ``[ax_ctrl, ay_ctrl, az_ctrl]`` — inertial control
      acceleration.

    Output (6,): state derivative
    ``[vx, vy, vz, ax_grav + ax_ctrl, ay_grav + ay_ctrl, az_grav + az_ctrl]``.

    Parameters
    ----------
    mu : float
        Gravitational parameter (must be > 0).

    Returns
    -------
    VectorFunction (IR=9, OR=6)
        Cartesian equations of motion with analytic Jacobian and Hessian.

    See Also
    --------
    modified_dynamics : Equivalent model in Modified Equinoctial Elements.
    """

def crtbp_dynamics(arg: float, /) -> _tychopy.vector_functions.VectorFunction:
    """
    Build the CR3BP dynamics VectorFunction in the synodic frame (IR=9, OR=6).

    Returns a VectorFunction suitable for use as the ODE of an optimal-control
    :class:`~tychopy.optimal_control.PhaseInterface`.

    Input layout ``[state(6), control(3)]``:

    * State (indices 0–5): ``[x, y, z, vx, vy, vz]`` — position and velocity
      in the rotating (synodic) frame with the primaries on the x-axis.
    * Control (indices 6–8): ``[ax_ctrl, ay_ctrl, az_ctrl]`` — control
      acceleration in the synodic frame.

    Output (6,): state derivative
    ``[vx, vy, vz, x_ddot, y_ddot, z_ddot]`` in the synodic frame
    (includes Coriolis and centrifugal terms).

    Parameters
    ----------
    mu : float
        CR3BP mass ratio ``mu = m2 / (m1 + m2)``, must satisfy ``0 < mu < 1``.
        This is **not** a gravitational parameter — it is the dimensionless
        mass fraction of the smaller primary.

    Returns
    -------
    VectorFunction (IR=9, OR=6)
        CR3BP equations of motion with analytic Jacobian and Hessian.
    """

def j2_cartesian(arg0: float, arg1: float, arg2: float, /) -> _tychopy.vector_functions.VectorFunction:
    """
    Build the J2 perturbation acceleration VectorFunction in Cartesian coordinates.

    Returns a VectorFunction that computes the J2 gravitational perturbation
    acceleration given position and north-pole direction.

    Input layout (IR=6): ``[rx, ry, rz, north_px, north_py, north_pz]``.
    The north-pole direction need not be pre-normalized.

    Output (OR=3): J2 acceleration vector ``[ax_J2, ay_J2, az_J2]``.

    Parameters
    ----------
    mu : float
        Gravitational parameter (must be > 0).
    J2 : float
        J2 zonal harmonic coefficient (dimensionless, typically ~1.08e-3 for
        Earth).
    Rb : float
        Reference body radius (same units as position).

    Returns
    -------
    VectorFunction (IR=6, OR=3)
        J2 perturbation acceleration with analytic derivatives.
    """

def non_ideal_solar_sail(arg0: float, arg1: float, arg2: float, arg3: float, arg4: float, /) -> _tychopy.vector_functions.VectorFunction:
    """
    Build the non-ideal solar-sail thrust acceleration VectorFunction.

    Returns a VectorFunction that models a non-ideal solar sail, accounting for
    absorption efficiency and tangential force, in addition to the ideal
    radiation-pressure term.

    Input layout (IR=6): ``[rx, ry, rz, nx, ny, nz]`` — inertial position and
    sail-normal direction.  The sail normal need not be pre-normalized.

    Output (OR=3): sail acceleration vector ``[ax, ay, az]``.

    Parameters
    ----------
    mu : float
        Gravitational parameter of the central body (used to scale the
        radiation-pressure force via the lightness number; must be > 0).
    beta : float
        Lightness number (ratio of solar radiation pressure to gravity at
        the reference distance).
    n1 : float
        Normal force efficiency coefficient.
    n2 : float
        Absorption efficiency coefficient.
    t1 : float
        Tangential force efficiency coefficient.

    Returns
    -------
    VectorFunction (IR=6, OR=3)
        Non-ideal solar-sail acceleration with analytic derivatives.
    """
