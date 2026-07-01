"""SubModule Containing Vector and Scalar Function Types and Functions"""

from collections.abc import Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import _tychopy.astro
import _tychopy.astro.kepler
import _tychopy.optimal_control
import _tychopy.optimal_control.ode_6
import _tychopy.optimal_control.ode_6_3
import _tychopy.optimal_control.ode_7_3
import _tychopy.optimal_control.ode_x
import _tychopy.optimal_control.ode_x_u
import _tychopy.optimal_control.ode_x_u_p


class VectorFunction:
    """
    The central symbolic, differentiable vector-to-vector map in Tycho.

    A ``VectorFunction`` represents a differentiable mapping
    ``f : R^n -> R^m`` that can be evaluated, differentiated (Jacobian,
    adjoint gradient, adjoint Hessian), and composed with other
    VectorFunctions to build dynamics, constraints, and objectives for
    optimal control problems. Concrete expressions are constructed by
    indexing and operating on ``Arguments`` objects; the result is
    implicitly convertible to ``VectorFunction`` for use in phase APIs.
    """

    @overload
    def __init__(self, arg: VectorFunction) -> None:
        """
        Construct a GenericFunction as a shared-ownership copy of another.

        Parameters
        ----------
        src : GenericFunction
            Source function to copy. The new instance shares ownership of the
            underlying erased function (O(1) refcount increment — no deep clone).

        Raises
        ------
        ValueError
            If *src* is an empty (null) GenericFunction.
        """

    @overload
    def __init__(self, arg: ScalarFunction, /) -> None:
        """
        Construct a fully-dynamic GenericFunction from a scalar-output GenericFunction.

        Absorbs a scalar-valued (output rows = 1) GenericFunction into a fully-dynamic
        ``GenericFunction[-1, -1]`` by cloning it into dynamic storage.

        Parameters
        ----------
        src : GenericFunction
            Scalar-output source function to promote. Must be non-empty.

        Raises
        ------
        ValueError
            If *src* is an empty (null) GenericFunction.
        """

    @overload
    def __init__(self, arg: Segment, /) -> None: ...

    @overload
    def __init__(self, arg: Element, /) -> None: ...

    @overload
    def __init__(self, arg: Arguments, /) -> None: ...

    @overload
    def __init__(self, arg: Segment2, /) -> None: ...

    @overload
    def __init__(self, arg: Segment3, /) -> None: ...

    @overload
    def __init__(self, arg: PyVectorFunction, /) -> None: ...

    @overload
    def __init__(self, arg: PyScalarFunction, /) -> None: ...

    @overload
    def __init__(self, arg: NumbaVectorFunction, /) -> None: ...

    @overload
    def __init__(self, arg: NumbaScalarFunction, /) -> None: ...

    @overload
    def __init__(self, arg: ConstantVector, /) -> None: ...

    @overload
    def __init__(self, arg: ConstantScalar, /) -> None: ...

    @overload
    def __init__(self, arg: IOScaled, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.InterpFunction, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.InterpFunction_1, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.InterpFunction_3, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.InterpFunction_6, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_x.ode, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_x_u.ode, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_x_u_p.ode, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_6_3.ode, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_7_3.ode, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_6.ode, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_x.integrator, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_x_u.integrator, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_x_u_p.integrator, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_6_3.integrator, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_7_3.integrator, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.ode_6.integrator, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.astro.kepler.ode, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.astro.kepler.KeplerPropagator, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.astro.kepler.integrator, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.astro.ModifiedToCartesian, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.astro.CartesianToClassic, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.astro.CartesianToMEE, /) -> None: ...

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
    def __call__(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Compose this VectorFunction with an inner VectorFunction (``self(g)``).

        When called with a VectorFunction argument rather than a numeric vector,
        returns the functional composition ``x ↦ self(g(x))`` rather than evaluating
        ``self`` at a point.  See :meth:`eval` for the non-operator form.

        Parameters
        ----------
        g : VectorFunction
            Inner function to substitute into the input of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(g(x))``.
        """

    @overload
    def __call__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __call__(self, arg: Segment, /) -> VectorFunction: ...

    @overload
    def __call__(self, arg: Segment2, /) -> VectorFunction: ...

    @overload
    def __call__(self, arg: Segment3, /) -> VectorFunction: ...

    @overload
    def __call__(self, arg: Sequence[ScalarFunction], /) -> VectorFunction:
        """
        Compose this function by substituting a list of scalar functions as its input.

        The argument functions are vertically stacked — their outputs are concatenated in
        order to form a single input vector — and the result is used as the input to this
        function.  Concretely, if ``funcs = [g0, g1, ...]`` and the shared domain of all
        ``gi`` is ``x``, then the returned function computes ``self([g0(x); g1(x); ...])``.

        All functions in ``funcs`` must share the same input size, and their concatenated
        output size must equal ``self.input_rows()``.

        Parameters
        ----------
        funcs : list[ScalarFunction]
            Homogeneous list of scalar-output (1-output-row) VectorFunctions whose
            stacked outputs form the input to this function.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.

        Raises
        ------
        ValueError
            If the stacked output size does not match ``self.input_rows()``, if the
            functions in ``funcs`` do not share a common input size, or if ``funcs``
            is empty.
        """

    @overload
    def __call__(self, arg: Sequence[VectorFunction], /) -> VectorFunction:
        """
        Compose this function by substituting a list of vector-output functions as its input.

        The functions in ``funcs`` are vertically stacked and their concatenated outputs
        are used as the input to this function.  Equivalent to the scalar-list overload
        but accepts VectorFunctions with any output dimension.

        Parameters
        ----------
        funcs : list[VectorFunction]
            List of VectorFunctions whose stacked outputs form the input to this function.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.
        """

    @overload
    def __call__(self, arg0: VectorFunction, /, *args) -> VectorFunction:
        """
        Compose this function with a variadic sequence of VectorFunctions.

        Convenience variadic form of the list overload.  All positional arguments after
        ``self`` are stacked in order and used as the input to this function.

        Parameters
        ----------
        first : VectorFunction
            First function in the stack; its ``input_rows()`` sets the shared domain size.
        *args : VectorFunction
            Additional VectorFunctions to stack after ``first``.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.
        """

    @overload
    def __call__(self, arg0: float, /, *args) -> VectorFunction:
        """
        Compose this function, prepending a scalar constant to the stacked input.

        The scalar ``first`` is promoted to a constant scalar VectorFunction whose domain
        size is inferred from the first VectorFunction in ``*args``, then stacked with the
        remaining arguments to form the full input.

        Parameters
        ----------
        first : float
            Scalar constant prepended to the input stack.
        *args : VectorFunction
            VectorFunctions whose stacked outputs (together with ``first``) form the input
            to this function.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.
        """

    @overload
    def __call__(self, arg0: numpy.ndarray, /, *args) -> VectorFunction:
        """
        Compose this function, prepending a constant vector to the stacked input.

        The array ``first`` is promoted to a constant vector VectorFunction whose domain
        size is inferred from the first VectorFunction in ``*args``, then stacked with the
        remaining arguments to form the full input.

        Parameters
        ----------
        first : array_like
            Constant vector prepended to the input stack.
        *args : VectorFunction
            VectorFunctions whose stacked outputs (together with ``first``) form the input
            to this function.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.
        """

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

    def vf(self) -> VectorFunction:
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

    def sum(self) -> ScalarFunction:
        """
        Sum all output components into a scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``Σ_i self(x)[i]``.
        """

    @overload
    def normalized_power3(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Shifted and normalized-power-3 of this VectorFunction.

        Computes ``(self(x) + b) / ||self(x) + b||³``.  This operation appears
        frequently in gravitational and electrostatic force expressions.

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset added to the output of ``self`` before normalizing.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction:
        """
        Scaled shifted normalized-power-3: ``s * (self(x) + b) / ||self(x) + b||³``; see :meth:`normalized_power3`.
        """

    @overload
    def normalized_power3(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power3`."""

    def norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`norm`."""

    def squared_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`squared_norm`."""

    def cubed_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`cubed_norm`."""

    def inverse_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_norm`."""

    def inverse_squared_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_squared_norm`."""

    def inverse_cubed_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_cubed_norm`."""

    def inverse_four_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_four_norm`."""

    def normalized(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized`."""

    def normalized_power2(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power2`."""

    def normalized_power4(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power4`."""

    def normalized_power5(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power5`."""

    @overload
    def cross(self, arg: Segment, /) -> VectorFunction:
        """
        3-D cross product of this VectorFunction with another.

        Both operands must produce 3-element output vectors.  The result
        is the standard right-hand-rule cross product ``self(x) × other(x)``.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (3,)
            Right-hand operand.  Accepts a VectorFunction of any concrete type
            or a constant 3-element vector.

        Returns
        -------
        VectorFunction
            3-element VectorFunction evaluating ``self(x) × other(x)``.
        """

    @overload
    def cross(self, arg: Segment3, /) -> VectorFunction:
        """
        Overload for a fixed-size-3 segment right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction:
        """
        Overload accepting a constant 3-element array right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cross`.
        """

    @overload
    def dot(self, arg: Segment, /) -> ScalarFunction:
        """Overload for a generic segment right-hand operand; see :meth:`dot`."""

    @overload
    def dot(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Dot product of this VectorFunction with another equal-length one.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) · other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`dot`.
        """

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise product of this VectorFunction with a segment VectorFunction.

        Parameters
        ----------
        other : Segment VectorFunction, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.
            This overload accepts a generic segment; sibling overloads handle
            other concrete VectorFunction types or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) .* other(x)`` element-wise.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise quotient of this VectorFunction divided by a segment VectorFunction.

        Parameters
        ----------
        other : Segment VectorFunction, shape (n,)
            Divisor; must have the same output dimension as ``self``.
            This overload accepts a generic segment; sibling overloads handle
            other concrete VectorFunction types or a constant vector divisor.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) ./ other(x)`` element-wise.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector divisor; see :meth:`cwise_quotient`.
        """

    __array_ufunc__: None = None

    @overload
    def __add__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Add a constant vector to this VectorFunction (``self + b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset to add element-wise to the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + b``.
        """

    @overload
    def __add__(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Add two VectorFunctions of the same type (``self + other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + other(x)``.
        """

    def __radd__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __sub__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Subtract a constant vector from this VectorFunction (``self - b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant to subtract element-wise from the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - b``.
        """

    @overload
    def __sub__(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Subtract two VectorFunctions of the same type (``self - other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - other(x)``.
        """

    def __rsub__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __mul__(self, arg: float, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar constant (``self * b``).

        Parameters
        ----------
        b : float
            Scalar multiplier applied to every output component.

        Returns
        -------
        VectorFunction
            Expression evaluating ``b * self(x)``.
        """

    @overload
    def __mul__(self, arg: Element, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar-valued VectorFunction (``self * s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; its runtime value scales every output
            component of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``s(x) * self(x)`` element-wise.
        """

    @overload
    def __mul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: float, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def __neg__(self) -> VectorFunction:
        """
        Negate this VectorFunction (``-self``).

        Returns
        -------
        VectorFunction
            Expression evaluating ``-self(x)``.
        """

    @overload
    def __truediv__(self, arg: float, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar constant (``self / b``).

        A separate overload accepts a scalar VectorFunction divisor.

        Parameters
        ----------
        b : float
            Constant divisor; every output component is divided by ``b``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / b``.
        """

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar-valued VectorFunction (``self / s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; every output component of ``self`` is
            divided by the runtime value of ``s``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / s(x)`` element-wise.
        """

    @overload
    def __truediv__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def padded_lower(self, arg: int, /) -> VectorFunction:
        """
        Append zero rows below the output of this VectorFunction.

        Parameters
        ----------
        lpad : int
            Number of zero rows to append after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[self(x); 0, …, 0]`` with ``lpad`` trailing zeros.
        """

    def padded_upper(self, arg: int, /) -> VectorFunction:
        """
        Prepend zero rows above the output of this VectorFunction.

        Parameters
        ----------
        upad : int
            Number of zero rows to prepend before the first output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0, …, 0; self(x)]`` with ``upad`` leading zeros.
        """

    def padded(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Pad the output of this VectorFunction with zeros above and below.

        Parameters
        ----------
        upad : int
            Number of zero rows prepended before the first output component.
        lpad : int
            Number of zero rows appended after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0…; self(x); 0…]``.
        """

    def segment(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Extract a contiguous sub-vector of the output.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.
        size : int
            Number of consecutive output elements to extract.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + size]``.
        """

    def head(self, arg: int, /) -> VectorFunction:
        """
        Extract the first ``size`` output components.

        Parameters
        ----------
        size : int
            Number of leading output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:size]``.
        """

    def tail(self, arg: int, /) -> VectorFunction:
        """
        Extract the last ``size`` output components.

        Parameters
        ----------
        size : int
            Number of trailing output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-size:]``.
        """

    def coeff(self, arg: int, /) -> ScalarFunction:
        """
        Extract a single scalar output component by index.

        Parameters
        ----------
        elem : int
            Zero-based index of the output element to select.

        Returns
        -------
        ScalarFunction
            Expression whose output is the scalar ``self(x)[elem]``.
        """

    @overload
    def __getitem__(self, arg: int, /) -> ScalarFunction:
        """
        Extract a single output element by integer index (``self[i]``).

        A ``slice`` argument is handled by a sibling overload and raises
        ``ValueError`` if the slice is empty, backward, or non-contiguous.

        Parameters
        ----------
        index : int
            Zero-based index of the output element to select.

        Returns
        -------
        VectorFunction
            Scalar expression for the selected output element.
        """

    @overload
    def __getitem__(self, arg: slice, /) -> VectorFunction: ...

    def segment_2(self, arg: int, /) -> VectorFunction:
        """
        Extract a 2-element sub-vector of the output starting at a given index.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + 2]``.
        """

    def head_2(self) -> VectorFunction:
        """
        Extract the first 2 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:2]``.
        """

    def tail_2(self) -> VectorFunction:
        """
        Extract the last 2 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-2:]``.
        """

    def segment2(self, arg: int, /) -> VectorFunction:
        """Alias for :meth:`segment_2`."""

    def head2(self) -> VectorFunction:
        """Alias for :meth:`head_2`."""

    def tail2(self) -> VectorFunction:
        """Alias for :meth:`tail_2`."""

    def segment_3(self, arg: int, /) -> VectorFunction:
        """
        Extract a 3-element sub-vector of the output starting at a given index.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + 3]``.
        """

    def head_3(self) -> VectorFunction:
        """
        Extract the first 3 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:3]``.
        """

    def tail_3(self) -> VectorFunction:
        """
        Extract the last 3 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-3:]``.
        """

    def segment3(self, arg: int, /) -> VectorFunction:
        """Alias for :meth:`segment_3`."""

    def head3(self) -> VectorFunction:
        """Alias for :meth:`head_3`."""

    def tail3(self) -> VectorFunction:
        """Alias for :meth:`tail_3`."""

    @overload
    def eval(self, arg: Element, /) -> VectorFunction:
        """
        Compose this VectorFunction with a scalar-segment inner function: ``self(g(.))``.

        Substitutes the inner function ``g`` into the input of ``self``, forming
        the composition ``x ↦ self(g(x))``.  The inner function's output dimension
        must match ``self``'s input dimension.  Sibling overloads accept other
        inner-function types (generic segment, fixed-size segments, GenericFunction).

        Parameters
        ----------
        g : Segment (scalar, ``Segment[-1, 1, -1]``)
            Inner function whose output feeds ``self``'s input.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(g(x))``.
        """

    @overload
    def eval(self, arg: Segment, /) -> VectorFunction:
        """Overload for a generic segment inner function; see :meth:`eval`."""

    @overload
    def eval(self, arg: Segment2, /) -> VectorFunction:
        """Overload for a fixed-size-2 segment inner function; see :meth:`eval`."""

    @overload
    def eval(self, arg: Segment3, /) -> VectorFunction:
        """Overload for a fixed-size-3 segment inner function; see :meth:`eval`."""

    @overload
    def eval(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction inner function; see :meth:`eval`.
        """

    @overload
    def eval(self, arg0: int, arg1: numpy.ndarray, /) -> VectorFunction:
        """
        Compose this function with a remapped-input projection.

        Creates the composition ``x ↦ self(x[v[0]], x[v[1]], …, x[v[k-1]])`` where
        ``x`` is a vector of length ``ir``.  Use this to route selected elements of a
        larger input vector into the inputs of ``self``.

        Parameters
        ----------
        ir : int
            Total number of inputs in the outer (ambient) vector ``x``.
        v : array_like of int, shape (input_rows,)
            Index vector; ``v[i]`` is the position in ``x`` of the ``i``-th input
            of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x[v])``, with ``input_rows`` equal to ``ir``.
        """

    @overload
    def apply(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Apply an outer VectorFunction to the output of this one: ``g(self(.))``.

        Forms the composition ``x ↦ g(self(x))``, with ``self`` as the *inner*
        function and ``g`` as the *outer* function.  This is the reverse of
        :meth:`eval`; use it when you have an outer function ``g`` and want to
        pipe ``self`` through it.

        Parameters
        ----------
        g : VectorFunction
            Outer function; its input dimension must match ``self``'s output dimension.

        Returns
        -------
        VectorFunction
            Expression evaluating ``g(self(x))``.
        """

    @overload
    def apply(self, arg: ScalarFunction, /) -> ScalarFunction:
        """Overload for a scalar outer function; see :meth:`apply`."""

class ScalarFunction:
    """
    A VectorFunction specialization whose output is a single scalar value.

    ``ScalarFunction`` is a ``VectorFunction`` with exactly one output row
    (``m == 1``). All VectorFunction operations are available; in addition a
    ``ScalarFunction`` can be used directly wherever a scalar objective or
    scalar constraint value is expected. Any ``ScalarFunction`` is implicitly
    convertible to ``VectorFunction``.
    """

    @overload
    def __init__(self, arg: ScalarFunction) -> None:
        """
        Construct a GenericFunction as a shared-ownership copy of another.

        Parameters
        ----------
        src : GenericFunction
            Source function to copy. The new instance shares ownership of the
            underlying erased function (O(1) refcount increment — no deep clone).

        Raises
        ------
        ValueError
            If *src* is an empty (null) GenericFunction.
        """

    @overload
    def __init__(self, arg: Element, /) -> None: ...

    @overload
    def __init__(self, arg: PyScalarFunction, /) -> None: ...

    @overload
    def __init__(self, arg: NumbaScalarFunction, /) -> None: ...

    @overload
    def __init__(self, arg: ConstantScalar, /) -> None: ...

    @overload
    def __init__(self, arg: _tychopy.optimal_control.InterpFunction_1, /) -> None: ...

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
    def __call__(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Compose this VectorFunction with an inner VectorFunction (``self(g)``).

        When called with a VectorFunction argument rather than a numeric vector,
        returns the functional composition ``x ↦ self(g(x))`` rather than evaluating
        ``self`` at a point.  See :meth:`eval` for the non-operator form.

        Parameters
        ----------
        g : VectorFunction
            Inner function to substitute into the input of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(g(x))``.
        """

    @overload
    def __call__(self, arg: Element, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: Segment, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: Segment2, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: Segment3, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: Sequence[ScalarFunction], /) -> ScalarFunction:
        """
        Compose this function by substituting a list of scalar functions as its input.

        The argument functions are vertically stacked — their outputs are concatenated in
        order to form a single input vector — and the result is used as the input to this
        function.  Concretely, if ``funcs = [g0, g1, ...]`` and the shared domain of all
        ``gi`` is ``x``, then the returned function computes ``self([g0(x); g1(x); ...])``.

        All functions in ``funcs`` must share the same input size, and their concatenated
        output size must equal ``self.input_rows()``.

        Parameters
        ----------
        funcs : list[ScalarFunction]
            Homogeneous list of scalar-output (1-output-row) VectorFunctions whose
            stacked outputs form the input to this function.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.

        Raises
        ------
        ValueError
            If the stacked output size does not match ``self.input_rows()``, if the
            functions in ``funcs`` do not share a common input size, or if ``funcs``
            is empty.
        """

    @overload
    def __call__(self, arg: Sequence[VectorFunction], /) -> ScalarFunction:
        """
        Compose this function by substituting a list of vector-output functions as its input.

        The functions in ``funcs`` are vertically stacked and their concatenated outputs
        are used as the input to this function.  Equivalent to the scalar-list overload
        but accepts VectorFunctions with any output dimension.

        Parameters
        ----------
        funcs : list[VectorFunction]
            List of VectorFunctions whose stacked outputs form the input to this function.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.
        """

    @overload
    def __call__(self, arg0: VectorFunction, /, *args) -> ScalarFunction:
        """
        Compose this function with a variadic sequence of VectorFunctions.

        Convenience variadic form of the list overload.  All positional arguments after
        ``self`` are stacked in order and used as the input to this function.

        Parameters
        ----------
        first : VectorFunction
            First function in the stack; its ``input_rows()`` sets the shared domain size.
        *args : VectorFunction
            Additional VectorFunctions to stack after ``first``.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.
        """

    @overload
    def __call__(self, arg0: float, /, *args) -> ScalarFunction:
        """
        Compose this function, prepending a scalar constant to the stacked input.

        The scalar ``first`` is promoted to a constant scalar VectorFunction whose domain
        size is inferred from the first VectorFunction in ``*args``, then stacked with the
        remaining arguments to form the full input.

        Parameters
        ----------
        first : float
            Scalar constant prepended to the input stack.
        *args : VectorFunction
            VectorFunctions whose stacked outputs (together with ``first``) form the input
            to this function.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.
        """

    @overload
    def __call__(self, arg0: numpy.ndarray, /, *args) -> ScalarFunction:
        """
        Compose this function, prepending a constant vector to the stacked input.

        The array ``first`` is promoted to a constant vector VectorFunction whose domain
        size is inferred from the first VectorFunction in ``*args``, then stacked with the
        remaining arguments to form the full input.

        Parameters
        ----------
        first : array_like
            Constant vector prepended to the input stack.
        *args : VectorFunction
            VectorFunctions whose stacked outputs (together with ``first``) form the input
            to this function.

        Returns
        -------
        VectorFunction
            A new VectorFunction representing the composition.
        """

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

    def vf(self) -> VectorFunction:
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

    def sf(self) -> ScalarFunction:
        """
        Type-erase this scalar-output expression into a scalar GenericFunction.

        Available only on VectorFunctions with a single output row.  Equivalent to
        :meth:`vf` but the returned wrapper carries the compile-time knowledge that
        the output dimension is 1, which enables scalar-specific operations.

        Returns
        -------
        ScalarFunction
            A dynamically-typed scalar-output wrapper around this function.
        """

    __array_ufunc__: None = None

    @overload
    def __add__(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Add a constant vector to this VectorFunction (``self + b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset to add element-wise to the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + b``.
        """

    @overload
    def __add__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __add__(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Add two VectorFunctions of the same type (``self + other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + other(x)``.
        """

    @overload
    def __radd__(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def __radd__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __sub__(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Subtract a constant vector from this VectorFunction (``self - b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant to subtract element-wise from the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - b``.
        """

    @overload
    def __sub__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __sub__(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Subtract two VectorFunctions of the same type (``self - other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - other(x)``.
        """

    @overload
    def __rsub__(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def __rsub__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __mul__(self, arg: float, /) -> ScalarFunction:
        """
        Scale this VectorFunction by a scalar constant (``self * b``).

        Parameters
        ----------
        b : float
            Scalar multiplier applied to every output component.

        Returns
        -------
        VectorFunction
            Expression evaluating ``b * self(x)``.
        """

    @overload
    def __mul__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __mul__(self, arg: Element, /) -> ScalarFunction:
        """
        Scale this VectorFunction by a scalar-valued VectorFunction (``self * s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; its runtime value scales every output
            component of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``s(x) * self(x)`` element-wise.
        """

    @overload
    def __mul__(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def __rmul__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __rmul__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    def __neg__(self) -> ScalarFunction:
        """
        Negate this VectorFunction (``-self``).

        Returns
        -------
        VectorFunction
            Expression evaluating ``-self(x)``.
        """

    @overload
    def __truediv__(self, arg: float, /) -> ScalarFunction:
        """
        Divide this VectorFunction by a scalar constant (``self / b``).

        A separate overload accepts a scalar VectorFunction divisor.

        Parameters
        ----------
        b : float
            Constant divisor; every output component is divided by ``b``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / b``.
        """

    @overload
    def __truediv__(self, arg: Element, /) -> ScalarFunction: ...

    @overload
    def __truediv__(self, arg: Element, /) -> ScalarFunction:
        """
        Divide this VectorFunction by a scalar-valued VectorFunction (``self / s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; every output component of ``self`` is
            divided by the runtime value of ``s``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / s(x)`` element-wise.
        """

    @overload
    def __truediv__(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    def __rtruediv__(self, arg: float, /) -> ScalarFunction: ...

    def sin(self) -> ScalarFunction:
        """
        Sine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``sin(self(x))``.
        """

    def cos(self) -> ScalarFunction:
        """
        Cosine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``cos(self(x))``.
        """

    def tan(self) -> ScalarFunction:
        """
        Tangent of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``tan(self(x))``.
        """

    def sqrt(self) -> ScalarFunction:
        """
        Square root of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``sqrt(self(x))``.
        """

    def exp(self) -> ScalarFunction:
        """
        Natural exponential of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``exp(self(x))``.
        """

    def log(self) -> ScalarFunction:
        """
        Natural logarithm of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``log(self(x))``.
        """

    def squared(self) -> ScalarFunction:
        """
        Square of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x)²``.
        """

    def arcsin(self) -> ScalarFunction:
        """
        Inverse sine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arcsin(self(x))``, in radians, range ``[-π/2, π/2]``.
        """

    def arccos(self) -> ScalarFunction:
        """
        Inverse cosine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arccos(self(x))``, in radians, range ``[0, π]``.
        """

    def arctan(self) -> ScalarFunction:
        """
        Inverse tangent of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arctan(self(x))``, in radians, range ``(-π/2, π/2)``.
        """

    def sinh(self) -> ScalarFunction:
        """
        Hyperbolic sine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``sinh(self(x))``.
        """

    def cosh(self) -> ScalarFunction:
        """
        Hyperbolic cosine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``cosh(self(x))``.
        """

    def tanh(self) -> ScalarFunction:
        """
        Hyperbolic tangent of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``tanh(self(x))``.
        """

    def arcsinh(self) -> ScalarFunction:
        """
        Inverse hyperbolic sine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arcsinh(self(x))``.
        """

    def arccosh(self) -> ScalarFunction:
        """
        Inverse hyperbolic cosine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arccosh(self(x))``, defined for input ≥ 1.
        """

    def arctanh(self) -> ScalarFunction:
        """
        Inverse hyperbolic tangent of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arctanh(self(x))``, defined for input in ``(-1, 1)``.
        """

    def __abs__(self) -> ScalarFunction:
        """
        Absolute value of this scalar VectorFunction (``abs(self)``).

        Returns
        -------
        ScalarFunction
            Expression evaluating ``|self(x)|``.
        """

    def sign(self) -> ScalarFunction:
        """
        Sign of this scalar VectorFunction: ``-1``, ``0``, or ``+1``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``sign(self(x))``.
        """

    def pow(self, arg: float, /) -> ScalarFunction:
        """
        Raise this scalar VectorFunction to a real power.

        Parameters
        ----------
        power : float
            Exponent to raise ``self(x)`` to.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) ** power``.
        """

    @overload
    def __pow__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __pow__(self, arg: int, /) -> ScalarFunction: ...

    @overload
    def dot(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Dot product of this VectorFunction with another equal-length one.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) · other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`dot`.
        """

    @overload
    def cwise_product(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_quotient(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector divisor; see :meth:`cwise_quotient`.
        """

    @overload
    def eval(self, arg: Element, /) -> ScalarFunction:
        """
        Compose this VectorFunction with a scalar-segment inner function: ``self(g(.))``.

        Substitutes the inner function ``g`` into the input of ``self``, forming
        the composition ``x ↦ self(g(x))``.  The inner function's output dimension
        must match ``self``'s input dimension.  Sibling overloads accept other
        inner-function types (generic segment, fixed-size segments, GenericFunction).

        Parameters
        ----------
        g : Segment (scalar, ``Segment[-1, 1, -1]``)
            Inner function whose output feeds ``self``'s input.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(g(x))``.
        """

    @overload
    def eval(self, arg: Segment, /) -> ScalarFunction:
        """Overload for a generic segment inner function; see :meth:`eval`."""

    @overload
    def eval(self, arg: Segment2, /) -> ScalarFunction:
        """Overload for a fixed-size-2 segment inner function; see :meth:`eval`."""

    @overload
    def eval(self, arg: Segment3, /) -> ScalarFunction:
        """Overload for a fixed-size-3 segment inner function; see :meth:`eval`."""

    @overload
    def eval(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Overload for a generic VectorFunction inner function; see :meth:`eval`.
        """

    @overload
    def eval(self, arg0: int, arg1: numpy.ndarray, /) -> ScalarFunction:
        """
        Compose this function with a remapped-input projection.

        Creates the composition ``x ↦ self(x[v[0]], x[v[1]], …, x[v[k-1]])`` where
        ``x`` is a vector of length ``ir``.  Use this to route selected elements of a
        larger input vector into the inputs of ``self``.

        Parameters
        ----------
        ir : int
            Total number of inputs in the outer (ambient) vector ``x``.
        v : array_like of int, shape (input_rows,)
            Index vector; ``v[i]`` is the position in ``x`` of the ``i``-th input
            of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x[v])``, with ``input_rows`` equal to ``ir``.
        """

    @overload
    def apply(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Apply an outer VectorFunction to the output of this one: ``g(self(.))``.

        Forms the composition ``x ↦ g(self(x))``, with ``self`` as the *inner*
        function and ``g`` as the *outer* function.  This is the reverse of
        :meth:`eval`; use it when you have an outer function ``g`` and want to
        pipe ``self`` through it.

        Parameters
        ----------
        g : VectorFunction
            Outer function; its input dimension must match ``self``'s output dimension.

        Returns
        -------
        VectorFunction
            Expression evaluating ``g(self(x))``.
        """

    @overload
    def apply(self, arg: ScalarFunction, /) -> ScalarFunction:
        """Overload for a scalar outer function; see :meth:`apply`."""

    def padded_lower(self, arg: int, /) -> VectorFunction:
        """
        Append zero rows below the output of this VectorFunction.

        Parameters
        ----------
        lpad : int
            Number of zero rows to append after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[self(x); 0, …, 0]`` with ``lpad`` trailing zeros.
        """

    def padded_upper(self, arg: int, /) -> VectorFunction:
        """
        Prepend zero rows above the output of this VectorFunction.

        Parameters
        ----------
        upad : int
            Number of zero rows to prepend before the first output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0, …, 0; self(x)]`` with ``upad`` leading zeros.
        """

    def padded(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Pad the output of this VectorFunction with zeros above and below.

        Parameters
        ----------
        upad : int
            Number of zero rows prepended before the first output component.
        lpad : int
            Number of zero rows appended after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0…; self(x); 0…]``.
        """

    def segment(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Extract a contiguous sub-vector of the output.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.
        size : int
            Number of consecutive output elements to extract.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + size]``.
        """

    def head(self, arg: int, /) -> VectorFunction:
        """
        Extract the first ``size`` output components.

        Parameters
        ----------
        size : int
            Number of leading output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:size]``.
        """

    def tail(self, arg: int, /) -> VectorFunction:
        """
        Extract the last ``size`` output components.

        Parameters
        ----------
        size : int
            Number of trailing output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-size:]``.
        """

    def coeff(self, arg: int, /) -> ScalarFunction:
        """
        Extract a single scalar output component by index.

        Parameters
        ----------
        elem : int
            Zero-based index of the output element to select.

        Returns
        -------
        ScalarFunction
            Expression whose output is the scalar ``self(x)[elem]``.
        """

    @overload
    def __getitem__(self, arg: int, /) -> ScalarFunction:
        """
        Extract a single output element by integer index (``self[i]``).

        A ``slice`` argument is handled by a sibling overload and raises
        ``ValueError`` if the slice is empty, backward, or non-contiguous.

        Parameters
        ----------
        index : int
            Zero-based index of the output element to select.

        Returns
        -------
        VectorFunction
            Scalar expression for the selected output element.
        """

    @overload
    def __getitem__(self, arg: slice, /) -> VectorFunction: ...

    @overload
    def __lt__(self, arg: float, /) -> Conditional: ...

    @overload
    def __lt__(self, arg: ScalarFunction, /) -> Conditional: ...

    @overload
    def __gt__(self, arg: float, /) -> Conditional: ...

    @overload
    def __gt__(self, arg: ScalarFunction, /) -> Conditional: ...

    def __rlt__(self, arg: float, /) -> Conditional: ...

    def __rgt__(self, arg: float, /) -> Conditional: ...

    @overload
    def __le__(self, arg: float, /) -> Conditional: ...

    @overload
    def __le__(self, arg: ScalarFunction, /) -> Conditional: ...

    @overload
    def __ge__(self, arg: float, /) -> Conditional: ...

    @overload
    def __ge__(self, arg: ScalarFunction, /) -> Conditional: ...

class Segment:
    """
    A contiguous sub-vector view of a VectorFunction's input.

    ``Segment(input_rows, output_rows, start)`` extracts ``output_rows``
    consecutive components of the input vector beginning at index ``start``,
    producing a new vector-valued VectorFunction. It is the standard mechanism for
    slicing the symbolic input — obtained by indexing ``Arguments`` objects (e.g.
    ``x[2:5]``) or by calling ``.segment(start, size)`` on any VectorFunction.
    """

    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None:
        """
        Construct a Segment by specifying input size, segment length, and start index.

        Parameters
        ----------
        input_rows : int
            Total number of input rows.
        output_rows : int
            Number of consecutive input components to extract (segment length).
        start : int
            Index of the first input component to extract.
        """

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

    def vf(self) -> VectorFunction:
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

    __array_ufunc__: None = None

    @overload
    def __add__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Add a constant vector to this VectorFunction (``self + b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset to add element-wise to the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + b``.
        """

    @overload
    def __add__(self, arg: Segment, /) -> VectorFunction:
        """
        Add two VectorFunctions of the same type (``self + other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + other(x)``.
        """

    @overload
    def __add__(self, arg: VectorFunction, /) -> VectorFunction: ...

    def __radd__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __sub__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Subtract a constant vector from this VectorFunction (``self - b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant to subtract element-wise from the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - b``.
        """

    @overload
    def __sub__(self, arg: Segment, /) -> VectorFunction:
        """
        Subtract two VectorFunctions of the same type (``self - other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - other(x)``.
        """

    @overload
    def __sub__(self, arg: VectorFunction, /) -> VectorFunction: ...

    def __rsub__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __mul__(self, arg: float, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar constant (``self * b``).

        Parameters
        ----------
        b : float
            Scalar multiplier applied to every output component.

        Returns
        -------
        VectorFunction
            Expression evaluating ``b * self(x)``.
        """

    @overload
    def __mul__(self, arg: Element, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar-valued VectorFunction (``self * s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; its runtime value scales every output
            component of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``s(x) * self(x)`` element-wise.
        """

    @overload
    def __mul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: float, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def __neg__(self) -> VectorFunction:
        """
        Negate this VectorFunction (``-self``).

        Returns
        -------
        VectorFunction
            Expression evaluating ``-self(x)``.
        """

    @overload
    def __truediv__(self, arg: float, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar constant (``self / b``).

        A separate overload accepts a scalar VectorFunction divisor.

        Parameters
        ----------
        b : float
            Constant divisor; every output component is divided by ``b``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / b``.
        """

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar-valued VectorFunction (``self / s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; every output component of ``self`` is
            divided by the runtime value of ``s``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / s(x)`` element-wise.
        """

    @overload
    def __truediv__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def sum(self) -> ScalarFunction:
        """
        Sum all output components into a scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``Σ_i self(x)[i]``.
        """

    @overload
    def normalized_power3(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Shifted and normalized-power-3 of this VectorFunction.

        Computes ``(self(x) + b) / ||self(x) + b||³``.  This operation appears
        frequently in gravitational and electrostatic force expressions.

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset added to the output of ``self`` before normalizing.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction:
        """
        Scaled shifted normalized-power-3: ``s * (self(x) + b) / ||self(x) + b||³``; see :meth:`normalized_power3`.
        """

    @overload
    def normalized_power3(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power3`."""

    def norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`norm`."""

    def squared_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`squared_norm`."""

    def cubed_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`cubed_norm`."""

    def inverse_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_norm`."""

    def inverse_squared_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_squared_norm`."""

    def inverse_cubed_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_cubed_norm`."""

    def inverse_four_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_four_norm`."""

    def normalized(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized`."""

    def normalized_power2(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power2`."""

    def normalized_power4(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power4`."""

    def normalized_power5(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power5`."""

    @overload
    def cross(self, arg: Segment3, /) -> VectorFunction:
        """
        Overload for a fixed-size-3 segment right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction:
        """
        Overload accepting a constant 3-element array right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: Segment, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cross`.
        """

    @overload
    def dot(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`dot`.
        """

    @overload
    def dot(self, arg: Segment, /) -> ScalarFunction:
        """
        Dot product of this VectorFunction with another equal-length one.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) · other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`dot`.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector divisor; see :meth:`cwise_quotient`.
        """

    @overload
    def apply(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Apply an outer VectorFunction to the output of this one: ``g(self(.))``.

        Forms the composition ``x ↦ g(self(x))``, with ``self`` as the *inner*
        function and ``g`` as the *outer* function.  This is the reverse of
        :meth:`eval`; use it when you have an outer function ``g`` and want to
        pipe ``self`` through it.

        Parameters
        ----------
        g : VectorFunction
            Outer function; its input dimension must match ``self``'s output dimension.

        Returns
        -------
        VectorFunction
            Expression evaluating ``g(self(x))``.
        """

    @overload
    def apply(self, arg: ScalarFunction, /) -> ScalarFunction:
        """Overload for a scalar outer function; see :meth:`apply`."""

    def padded_lower(self, arg: int, /) -> VectorFunction:
        """
        Append zero rows below the output of this VectorFunction.

        Parameters
        ----------
        lpad : int
            Number of zero rows to append after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[self(x); 0, …, 0]`` with ``lpad`` trailing zeros.
        """

    def padded_upper(self, arg: int, /) -> VectorFunction:
        """
        Prepend zero rows above the output of this VectorFunction.

        Parameters
        ----------
        upad : int
            Number of zero rows to prepend before the first output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0, …, 0; self(x)]`` with ``upad`` leading zeros.
        """

    def padded(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Pad the output of this VectorFunction with zeros above and below.

        Parameters
        ----------
        upad : int
            Number of zero rows prepended before the first output component.
        lpad : int
            Number of zero rows appended after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0…; self(x); 0…]``.
        """

    def segment(self, arg0: int, arg1: int, /) -> Segment:
        """
        Extract a contiguous sub-vector of the output.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.
        size : int
            Number of consecutive output elements to extract.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + size]``.
        """

    def head(self, arg: int, /) -> Segment:
        """
        Extract the first ``size`` output components.

        Parameters
        ----------
        size : int
            Number of leading output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:size]``.
        """

    def tail(self, arg: int, /) -> Segment:
        """
        Extract the last ``size`` output components.

        Parameters
        ----------
        size : int
            Number of trailing output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-size:]``.
        """

    def coeff(self, arg: int, /) -> Element:
        """
        Extract a single scalar output component by index.

        Parameters
        ----------
        elem : int
            Zero-based index of the output element to select.

        Returns
        -------
        ScalarFunction
            Expression whose output is the scalar ``self(x)[elem]``.
        """

    @overload
    def __getitem__(self, arg: int, /) -> Element:
        """
        Extract a single output element by integer index (``self[i]``).

        A ``slice`` argument is handled by a sibling overload and raises
        ``ValueError`` if the slice is empty, backward, or non-contiguous.

        Parameters
        ----------
        index : int
            Zero-based index of the output element to select.

        Returns
        -------
        VectorFunction
            Scalar expression for the selected output element.
        """

    @overload
    def __getitem__(self, arg: slice, /) -> Segment: ...

    def segment_2(self, arg: int, /) -> Segment2:
        """
        Extract a 2-element sub-vector of the output starting at a given index.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + 2]``.
        """

    def head_2(self) -> Segment2:
        """
        Extract the first 2 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:2]``.
        """

    def tail_2(self) -> Segment2:
        """
        Extract the last 2 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-2:]``.
        """

    def segment2(self, arg: int, /) -> Segment2:
        """Alias for :meth:`segment_2`."""

    def head2(self) -> Segment2:
        """Alias for :meth:`head_2`."""

    def tail2(self) -> Segment2:
        """Alias for :meth:`tail_2`."""

    def segment_3(self, arg: int, /) -> Segment3:
        """
        Extract a 3-element sub-vector of the output starting at a given index.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + 3]``.
        """

    def head_3(self) -> Segment3:
        """
        Extract the first 3 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:3]``.
        """

    def tail_3(self) -> Segment3:
        """
        Extract the last 3 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-3:]``.
        """

    def segment3(self, arg: int, /) -> Segment3:
        """Alias for :meth:`segment_3`."""

    def head3(self) -> Segment3:
        """Alias for :meth:`head_3`."""

    def tail3(self) -> Segment3:
        """Alias for :meth:`tail_3`."""

    @overload
    def tolist(self) -> list[Element]:
        """
        Split the function into a list of scalar element functions.

        Returns one scalar-valued VectorFunction per output row, in order, each
        extracting a single component of the original output vector.

        Returns
        -------
        list of Element
            A list of ``output_rows`` scalar Element functions, where element *i*
            extracts output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]:
        """
        Split selected output components into a list of scalar element functions.

        Returns one scalar-valued VectorFunction for each index in *coeffs*, each
        extracting the corresponding output component.

        Parameters
        ----------
        coeffs : list of int
            Output-component indices to extract, in the requested order.

        Returns
        -------
        list of Element
            A list of scalar Element functions, one per entry in *coeffs*.
        """

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]:
        """
        Split the function into a mixed list of element and segment functions.

        Each entry in *seglist* is a ``(start, size)`` pair. A pair with ``size == 1``
        produces a scalar Element function; ``size == 2`` or ``3`` produces a
        fixed-width Segment; any other size produces a dynamic-length Segment.

        Parameters
        ----------
        seglist : list of (int, int)
            List of ``(start, size)`` pairs specifying which contiguous slices of the
            output vector to extract.

        Returns
        -------
        list of Element or Segment
            A heterogeneous list of VectorFunctions, one per entry in *seglist*.
        """

class Element:
    """
    A contiguous sub-vector view of a VectorFunction's input.

    ``Segment(input_rows, output_rows, start)`` extracts ``output_rows``
    consecutive components of the input vector beginning at index ``start``,
    producing a new vector-valued VectorFunction. It is the standard mechanism for
    slicing the symbolic input — obtained by indexing ``Arguments`` objects (e.g.
    ``x[2:5]``) or by calling ``.segment(start, size)`` on any VectorFunction.
    """

    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None:
        """
        Construct a Segment by specifying input size, segment length, and start index.

        Parameters
        ----------
        input_rows : int
            Total number of input rows.
        output_rows : int
            Number of consecutive input components to extract (segment length).
        start : int
            Index of the first input component to extract.
        """

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

    def vf(self) -> VectorFunction:
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

    def sf(self) -> ScalarFunction:
        """
        Type-erase this scalar-output expression into a scalar GenericFunction.

        Available only on VectorFunctions with a single output row.  Equivalent to
        :meth:`vf` but the returned wrapper carries the compile-time knowledge that
        the output dimension is 1, which enables scalar-specific operations.

        Returns
        -------
        ScalarFunction
            A dynamically-typed scalar-output wrapper around this function.
        """

    __array_ufunc__: None = None

    @overload
    def __add__(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Add a constant vector to this VectorFunction (``self + b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset to add element-wise to the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + b``.
        """

    @overload
    def __add__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __add__(self, arg: Element, /) -> ScalarFunction:
        """
        Add two VectorFunctions of the same type (``self + other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + other(x)``.
        """

    @overload
    def __add__(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def __radd__(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def __radd__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __sub__(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Subtract a constant vector from this VectorFunction (``self - b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant to subtract element-wise from the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - b``.
        """

    @overload
    def __sub__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __sub__(self, arg: Element, /) -> ScalarFunction:
        """
        Subtract two VectorFunctions of the same type (``self - other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - other(x)``.
        """

    @overload
    def __sub__(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def __rsub__(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def __rsub__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __mul__(self, arg: float, /) -> ScalarFunction:
        """
        Scale this VectorFunction by a scalar constant (``self * b``).

        Parameters
        ----------
        b : float
            Scalar multiplier applied to every output component.

        Returns
        -------
        VectorFunction
            Expression evaluating ``b * self(x)``.
        """

    @overload
    def __mul__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __mul__(self, arg: Element, /) -> ScalarFunction:
        """
        Scale this VectorFunction by a scalar-valued VectorFunction (``self * s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; its runtime value scales every output
            component of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``s(x) * self(x)`` element-wise.
        """

    @overload
    def __mul__(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def __rmul__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __rmul__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    def __neg__(self) -> ScalarFunction:
        """
        Negate this VectorFunction (``-self``).

        Returns
        -------
        VectorFunction
            Expression evaluating ``-self(x)``.
        """

    @overload
    def __truediv__(self, arg: float, /) -> ScalarFunction:
        """
        Divide this VectorFunction by a scalar constant (``self / b``).

        A separate overload accepts a scalar VectorFunction divisor.

        Parameters
        ----------
        b : float
            Constant divisor; every output component is divided by ``b``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / b``.
        """

    @overload
    def __truediv__(self, arg: Element, /) -> ScalarFunction: ...

    @overload
    def __truediv__(self, arg: Element, /) -> ScalarFunction:
        """
        Divide this VectorFunction by a scalar-valued VectorFunction (``self / s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; every output component of ``self`` is
            divided by the runtime value of ``s``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / s(x)`` element-wise.
        """

    @overload
    def __truediv__(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    def __rtruediv__(self, arg: float, /) -> ScalarFunction: ...

    def sin(self) -> ScalarFunction:
        """
        Sine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``sin(self(x))``.
        """

    def cos(self) -> ScalarFunction:
        """
        Cosine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``cos(self(x))``.
        """

    def tan(self) -> ScalarFunction:
        """
        Tangent of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``tan(self(x))``.
        """

    def sqrt(self) -> ScalarFunction:
        """
        Square root of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``sqrt(self(x))``.
        """

    def exp(self) -> ScalarFunction:
        """
        Natural exponential of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``exp(self(x))``.
        """

    def log(self) -> ScalarFunction:
        """
        Natural logarithm of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``log(self(x))``.
        """

    def squared(self) -> ScalarFunction:
        """
        Square of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x)²``.
        """

    def arcsin(self) -> ScalarFunction:
        """
        Inverse sine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arcsin(self(x))``, in radians, range ``[-π/2, π/2]``.
        """

    def arccos(self) -> ScalarFunction:
        """
        Inverse cosine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arccos(self(x))``, in radians, range ``[0, π]``.
        """

    def arctan(self) -> ScalarFunction:
        """
        Inverse tangent of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arctan(self(x))``, in radians, range ``(-π/2, π/2)``.
        """

    def sinh(self) -> ScalarFunction:
        """
        Hyperbolic sine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``sinh(self(x))``.
        """

    def cosh(self) -> ScalarFunction:
        """
        Hyperbolic cosine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``cosh(self(x))``.
        """

    def tanh(self) -> ScalarFunction:
        """
        Hyperbolic tangent of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``tanh(self(x))``.
        """

    def arcsinh(self) -> ScalarFunction:
        """
        Inverse hyperbolic sine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arcsinh(self(x))``.
        """

    def arccosh(self) -> ScalarFunction:
        """
        Inverse hyperbolic cosine of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arccosh(self(x))``, defined for input ≥ 1.
        """

    def arctanh(self) -> ScalarFunction:
        """
        Inverse hyperbolic tangent of this scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``arctanh(self(x))``, defined for input in ``(-1, 1)``.
        """

    def __abs__(self) -> ScalarFunction:
        """
        Absolute value of this scalar VectorFunction (``abs(self)``).

        Returns
        -------
        ScalarFunction
            Expression evaluating ``|self(x)|``.
        """

    def sign(self) -> ScalarFunction:
        """
        Sign of this scalar VectorFunction: ``-1``, ``0``, or ``+1``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``sign(self(x))``.
        """

    def pow(self, arg: float, /) -> ScalarFunction:
        """
        Raise this scalar VectorFunction to a real power.

        Parameters
        ----------
        power : float
            Exponent to raise ``self(x)`` to.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) ** power``.
        """

    @overload
    def __pow__(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def __pow__(self, arg: int, /) -> ScalarFunction: ...

    @overload
    def dot(self, arg: Element, /) -> ScalarFunction:
        """
        Dot product of this VectorFunction with another equal-length one.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) · other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`dot`.
        """

    @overload
    def cwise_product(self, arg: Element, /) -> ScalarFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_quotient(self, arg: Element, /) -> ScalarFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector divisor; see :meth:`cwise_quotient`.
        """

    @overload
    def apply(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Apply an outer VectorFunction to the output of this one: ``g(self(.))``.

        Forms the composition ``x ↦ g(self(x))``, with ``self`` as the *inner*
        function and ``g`` as the *outer* function.  This is the reverse of
        :meth:`eval`; use it when you have an outer function ``g`` and want to
        pipe ``self`` through it.

        Parameters
        ----------
        g : VectorFunction
            Outer function; its input dimension must match ``self``'s output dimension.

        Returns
        -------
        VectorFunction
            Expression evaluating ``g(self(x))``.
        """

    @overload
    def apply(self, arg: ScalarFunction, /) -> ScalarFunction:
        """Overload for a scalar outer function; see :meth:`apply`."""

    def padded_lower(self, arg: int, /) -> VectorFunction:
        """
        Append zero rows below the output of this VectorFunction.

        Parameters
        ----------
        lpad : int
            Number of zero rows to append after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[self(x); 0, …, 0]`` with ``lpad`` trailing zeros.
        """

    def padded_upper(self, arg: int, /) -> VectorFunction:
        """
        Prepend zero rows above the output of this VectorFunction.

        Parameters
        ----------
        upad : int
            Number of zero rows to prepend before the first output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0, …, 0; self(x)]`` with ``upad`` leading zeros.
        """

    def padded(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Pad the output of this VectorFunction with zeros above and below.

        Parameters
        ----------
        upad : int
            Number of zero rows prepended before the first output component.
        lpad : int
            Number of zero rows appended after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0…; self(x); 0…]``.
        """

    def segment(self, arg0: int, arg1: int, /) -> Segment:
        """
        Extract a contiguous sub-vector of the output.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.
        size : int
            Number of consecutive output elements to extract.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + size]``.
        """

    def head(self, arg: int, /) -> Segment:
        """
        Extract the first ``size`` output components.

        Parameters
        ----------
        size : int
            Number of leading output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:size]``.
        """

    def tail(self, arg: int, /) -> Segment:
        """
        Extract the last ``size`` output components.

        Parameters
        ----------
        size : int
            Number of trailing output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-size:]``.
        """

    def coeff(self, arg: int, /) -> Element:
        """
        Extract a single scalar output component by index.

        Parameters
        ----------
        elem : int
            Zero-based index of the output element to select.

        Returns
        -------
        ScalarFunction
            Expression whose output is the scalar ``self(x)[elem]``.
        """

    @overload
    def __getitem__(self, arg: int, /) -> Element:
        """
        Extract a single output element by integer index (``self[i]``).

        A ``slice`` argument is handled by a sibling overload and raises
        ``ValueError`` if the slice is empty, backward, or non-contiguous.

        Parameters
        ----------
        index : int
            Zero-based index of the output element to select.

        Returns
        -------
        VectorFunction
            Scalar expression for the selected output element.
        """

    @overload
    def __getitem__(self, arg: slice, /) -> Segment: ...

    @overload
    def __lt__(self, arg: float, /) -> Conditional: ...

    @overload
    def __lt__(self, arg: Element, /) -> Conditional: ...

    @overload
    def __lt__(self, arg: ScalarFunction, /) -> Conditional: ...

    @overload
    def __gt__(self, arg: float, /) -> Conditional: ...

    @overload
    def __gt__(self, arg: Element, /) -> Conditional: ...

    @overload
    def __gt__(self, arg: ScalarFunction, /) -> Conditional: ...

    def __rlt__(self, arg: float, /) -> Conditional: ...

    def __rgt__(self, arg: float, /) -> Conditional: ...

    @overload
    def __le__(self, arg: float, /) -> Conditional: ...

    @overload
    def __le__(self, arg: Element, /) -> Conditional: ...

    @overload
    def __le__(self, arg: ScalarFunction, /) -> Conditional: ...

    @overload
    def __ge__(self, arg: float, /) -> Conditional: ...

    @overload
    def __ge__(self, arg: Element, /) -> Conditional: ...

    @overload
    def __ge__(self, arg: ScalarFunction, /) -> Conditional: ...

    @overload
    def tolist(self) -> list[Element]:
        """
        Split the function into a list of scalar element functions.

        Returns one scalar-valued VectorFunction per output row, in order, each
        extracting a single component of the original output vector.

        Returns
        -------
        list of Element
            A list of ``output_rows`` scalar Element functions, where element *i*
            extracts output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]:
        """
        Split selected output components into a list of scalar element functions.

        Returns one scalar-valued VectorFunction for each index in *coeffs*, each
        extracting the corresponding output component.

        Parameters
        ----------
        coeffs : list of int
            Output-component indices to extract, in the requested order.

        Returns
        -------
        list of Element
            A list of scalar Element functions, one per entry in *coeffs*.
        """

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]:
        """
        Split the function into a mixed list of element and segment functions.

        Each entry in *seglist* is a ``(start, size)`` pair. A pair with ``size == 1``
        produces a scalar Element function; ``size == 2`` or ``3`` produces a
        fixed-width Segment; any other size produces a dynamic-length Segment.

        Parameters
        ----------
        seglist : list of (int, int)
            List of ``(start, size)`` pairs specifying which contiguous slices of the
            output vector to extract.

        Returns
        -------
        list of Element or Segment
            A heterogeneous list of VectorFunctions, one per entry in *seglist*.
        """

class Arguments:
    """
    The identity VectorFunction — the symbolic input vector for building expressions.

    ``Arguments(n)`` is the identity map on ``n``-dimensional inputs: it returns
    its input unchanged. It is the standard starting point for constructing
    VectorFunction expressions — use indexing (``x[i]``, ``x[i:j]``) or methods
    such as ``x.constant(v)`` to derive sub-expressions and build up dynamics,
    constraints, and objectives.
    """

    def __init__(self, arg: int, /) -> None:
        """
        Construct an identity-passthrough VectorFunction for a given input size.

        ``Arguments(n)`` is the identity function on ``n``-dimensional inputs: its
        output is the full input vector. It is the usual starting point for building
        VectorFunction expressions — index into it to form element or segment functions.

        Parameters
        ----------
        input_rows : int
            Number of input rows (dimension of the input/output vector).
        """

    @overload
    def constant(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Build a constant VectorFunction whose output is a fixed vector.

        Creates a VectorFunction that returns the same vector *v* for any input,
        with input dimension matching this ``Arguments`` object. The Jacobian and
        adjoint Hessian of a constant function are identically zero.

        Parameters
        ----------
        v : array_like, shape (output_rows,)
            Fixed output vector to return at every input point.

        Returns
        -------
        GenericFunction
            Constant function with ``input_rows`` matching this object and
            ``output_rows`` equal to ``len(v)``.

        Examples
        --------
        >>> import numpy as np
        >>> from tychopy import vector_functions as vf
        >>> x = vf.Arguments(3)
        >>> c = x.constant(np.array([1.0, 2.0, 3.0]))
        >>> c.compute(np.zeros(3))
        array([1., 2., 3.])
        """

    @overload
    def constant(self, arg: float, /) -> ScalarFunction:
        """
        Build a scalar constant VectorFunction whose output is a fixed scalar.

        Creates a scalar-output VectorFunction that returns *v* for any input, with
        input dimension matching this ``Arguments`` object.

        Parameters
        ----------
        v : float
            Fixed scalar output value.

        Returns
        -------
        ScalarFunction
            Constant scalar function with ``input_rows`` matching this object.
        """

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

    def vf(self) -> VectorFunction:
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

    __array_ufunc__: None = None

    @overload
    def __add__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Add a constant vector to this VectorFunction (``self + b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset to add element-wise to the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + b``.
        """

    @overload
    def __add__(self, arg: Arguments, /) -> VectorFunction:
        """
        Add two VectorFunctions of the same type (``self + other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + other(x)``.
        """

    def __radd__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __sub__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Subtract a constant vector from this VectorFunction (``self - b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant to subtract element-wise from the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - b``.
        """

    @overload
    def __sub__(self, arg: Arguments, /) -> VectorFunction:
        """
        Subtract two VectorFunctions of the same type (``self - other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - other(x)``.
        """

    def __rsub__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __mul__(self, arg: float, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar constant (``self * b``).

        Parameters
        ----------
        b : float
            Scalar multiplier applied to every output component.

        Returns
        -------
        VectorFunction
            Expression evaluating ``b * self(x)``.
        """

    @overload
    def __mul__(self, arg: Element, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar-valued VectorFunction (``self * s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; its runtime value scales every output
            component of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``s(x) * self(x)`` element-wise.
        """

    @overload
    def __mul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: float, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def __neg__(self) -> VectorFunction:
        """
        Negate this VectorFunction (``-self``).

        Returns
        -------
        VectorFunction
            Expression evaluating ``-self(x)``.
        """

    @overload
    def __truediv__(self, arg: float, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar constant (``self / b``).

        A separate overload accepts a scalar VectorFunction divisor.

        Parameters
        ----------
        b : float
            Constant divisor; every output component is divided by ``b``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / b``.
        """

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar-valued VectorFunction (``self / s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; every output component of ``self`` is
            divided by the runtime value of ``s``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / s(x)`` element-wise.
        """

    @overload
    def __truediv__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def sum(self) -> ScalarFunction:
        """
        Sum all output components into a scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``Σ_i self(x)[i]``.
        """

    @overload
    def normalized_power3(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Shifted and normalized-power-3 of this VectorFunction.

        Computes ``(self(x) + b) / ||self(x) + b||³``.  This operation appears
        frequently in gravitational and electrostatic force expressions.

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset added to the output of ``self`` before normalizing.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction:
        """
        Scaled shifted normalized-power-3: ``s * (self(x) + b) / ||self(x) + b||³``; see :meth:`normalized_power3`.
        """

    @overload
    def normalized_power3(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power3`."""

    def norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`norm`."""

    def squared_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`squared_norm`."""

    def cubed_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`cubed_norm`."""

    def inverse_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_norm`."""

    def inverse_squared_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_squared_norm`."""

    def inverse_cubed_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_cubed_norm`."""

    def inverse_four_norm(self) -> ScalarFunction:
        """Dynamic-size dispatch overload; see :meth:`inverse_four_norm`."""

    def normalized(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized`."""

    def normalized_power2(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power2`."""

    def normalized_power4(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power4`."""

    def normalized_power5(self) -> VectorFunction:
        """Dynamic-size dispatch overload; see :meth:`normalized_power5`."""

    @overload
    def cross(self, arg: Segment, /) -> VectorFunction:
        """
        3-D cross product of this VectorFunction with another.

        Both operands must produce 3-element output vectors.  The result
        is the standard right-hand-rule cross product ``self(x) × other(x)``.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (3,)
            Right-hand operand.  Accepts a VectorFunction of any concrete type
            or a constant 3-element vector.

        Returns
        -------
        VectorFunction
            3-element VectorFunction evaluating ``self(x) × other(x)``.
        """

    @overload
    def cross(self, arg: Segment3, /) -> VectorFunction:
        """
        Overload for a fixed-size-3 segment right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction:
        """
        Overload accepting a constant 3-element array right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: Arguments, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cross`.
        """

    @overload
    def dot(self, arg: Segment, /) -> ScalarFunction:
        """Overload for a generic segment right-hand operand; see :meth:`dot`."""

    @overload
    def dot(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`dot`.
        """

    @overload
    def dot(self, arg: Arguments, /) -> ScalarFunction:
        """
        Dot product of this VectorFunction with another equal-length one.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) · other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`dot`.
        """

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise product of this VectorFunction with a segment VectorFunction.

        Parameters
        ----------
        other : Segment VectorFunction, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.
            This overload accepts a generic segment; sibling overloads handle
            other concrete VectorFunction types or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) .* other(x)`` element-wise.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: Arguments, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise quotient of this VectorFunction divided by a segment VectorFunction.

        Parameters
        ----------
        other : Segment VectorFunction, shape (n,)
            Divisor; must have the same output dimension as ``self``.
            This overload accepts a generic segment; sibling overloads handle
            other concrete VectorFunction types or a constant vector divisor.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) ./ other(x)`` element-wise.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: Arguments, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector divisor; see :meth:`cwise_quotient`.
        """

    @overload
    def apply(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Apply an outer VectorFunction to the output of this one: ``g(self(.))``.

        Forms the composition ``x ↦ g(self(x))``, with ``self`` as the *inner*
        function and ``g`` as the *outer* function.  This is the reverse of
        :meth:`eval`; use it when you have an outer function ``g`` and want to
        pipe ``self`` through it.

        Parameters
        ----------
        g : VectorFunction
            Outer function; its input dimension must match ``self``'s output dimension.

        Returns
        -------
        VectorFunction
            Expression evaluating ``g(self(x))``.
        """

    @overload
    def apply(self, arg: ScalarFunction, /) -> ScalarFunction:
        """Overload for a scalar outer function; see :meth:`apply`."""

    def padded_lower(self, arg: int, /) -> VectorFunction:
        """
        Append zero rows below the output of this VectorFunction.

        Parameters
        ----------
        lpad : int
            Number of zero rows to append after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[self(x); 0, …, 0]`` with ``lpad`` trailing zeros.
        """

    def padded_upper(self, arg: int, /) -> VectorFunction:
        """
        Prepend zero rows above the output of this VectorFunction.

        Parameters
        ----------
        upad : int
            Number of zero rows to prepend before the first output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0, …, 0; self(x)]`` with ``upad`` leading zeros.
        """

    def padded(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Pad the output of this VectorFunction with zeros above and below.

        Parameters
        ----------
        upad : int
            Number of zero rows prepended before the first output component.
        lpad : int
            Number of zero rows appended after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0…; self(x); 0…]``.
        """

    def segment(self, arg0: int, arg1: int, /) -> Segment:
        """
        Extract a contiguous sub-vector of the output.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.
        size : int
            Number of consecutive output elements to extract.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + size]``.
        """

    def head(self, arg: int, /) -> Segment:
        """
        Extract the first ``size`` output components.

        Parameters
        ----------
        size : int
            Number of leading output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:size]``.
        """

    def tail(self, arg: int, /) -> Segment:
        """
        Extract the last ``size`` output components.

        Parameters
        ----------
        size : int
            Number of trailing output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-size:]``.
        """

    def coeff(self, arg: int, /) -> Element:
        """
        Extract a single scalar output component by index.

        Parameters
        ----------
        elem : int
            Zero-based index of the output element to select.

        Returns
        -------
        ScalarFunction
            Expression whose output is the scalar ``self(x)[elem]``.
        """

    @overload
    def __getitem__(self, arg: int, /) -> Element:
        """
        Extract a single output element by integer index (``self[i]``).

        A ``slice`` argument is handled by a sibling overload and raises
        ``ValueError`` if the slice is empty, backward, or non-contiguous.

        Parameters
        ----------
        index : int
            Zero-based index of the output element to select.

        Returns
        -------
        VectorFunction
            Scalar expression for the selected output element.
        """

    @overload
    def __getitem__(self, arg: slice, /) -> Segment: ...

    def segment_2(self, arg: int, /) -> Segment2:
        """
        Extract a 2-element sub-vector of the output starting at a given index.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + 2]``.
        """

    def head_2(self) -> Segment2:
        """
        Extract the first 2 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:2]``.
        """

    def tail_2(self) -> Segment2:
        """
        Extract the last 2 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-2:]``.
        """

    def segment2(self, arg: int, /) -> Segment2:
        """Alias for :meth:`segment_2`."""

    def head2(self) -> Segment2:
        """Alias for :meth:`head_2`."""

    def tail2(self) -> Segment2:
        """Alias for :meth:`tail_2`."""

    def segment_3(self, arg: int, /) -> Segment3:
        """
        Extract a 3-element sub-vector of the output starting at a given index.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + 3]``.
        """

    def head_3(self) -> Segment3:
        """
        Extract the first 3 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:3]``.
        """

    def tail_3(self) -> Segment3:
        """
        Extract the last 3 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-3:]``.
        """

    def segment3(self, arg: int, /) -> Segment3:
        """Alias for :meth:`segment_3`."""

    def head3(self) -> Segment3:
        """Alias for :meth:`head_3`."""

    def tail3(self) -> Segment3:
        """Alias for :meth:`tail_3`."""

    @overload
    def tolist(self) -> list[Element]:
        """
        Split the function into a list of scalar element functions.

        Returns one scalar-valued VectorFunction per output row, in order, each
        extracting a single component of the original output vector.

        Returns
        -------
        list of Element
            A list of ``output_rows`` scalar Element functions, where element *i*
            extracts output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]:
        """
        Split selected output components into a list of scalar element functions.

        Returns one scalar-valued VectorFunction for each index in *coeffs*, each
        extracting the corresponding output component.

        Parameters
        ----------
        coeffs : list of int
            Output-component indices to extract, in the requested order.

        Returns
        -------
        list of Element
            A list of scalar Element functions, one per entry in *coeffs*.
        """

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]:
        """
        Split the function into a mixed list of element and segment functions.

        Each entry in *seglist* is a ``(start, size)`` pair. A pair with ``size == 1``
        produces a scalar Element function; ``size == 2`` or ``3`` produces a
        fixed-width Segment; any other size produces a dynamic-length Segment.

        Parameters
        ----------
        seglist : list of (int, int)
            List of ``(start, size)`` pairs specifying which contiguous slices of the
            output vector to extract.

        Returns
        -------
        list of Element or Segment
            A heterogeneous list of VectorFunctions, one per entry in *seglist*.
        """

class Segment2:
    """
    A contiguous sub-vector view of a VectorFunction's input.

    ``Segment(input_rows, output_rows, start)`` extracts ``output_rows``
    consecutive components of the input vector beginning at index ``start``,
    producing a new vector-valued VectorFunction. It is the standard mechanism for
    slicing the symbolic input — obtained by indexing ``Arguments`` objects (e.g.
    ``x[2:5]``) or by calling ``.segment(start, size)`` on any VectorFunction.
    """

    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None:
        """
        Construct a Segment by specifying input size, segment length, and start index.

        Parameters
        ----------
        input_rows : int
            Total number of input rows.
        output_rows : int
            Number of consecutive input components to extract (segment length).
        start : int
            Index of the first input component to extract.
        """

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

    def vf(self) -> VectorFunction:
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

    __array_ufunc__: None = None

    @overload
    def __add__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Add a constant vector to this VectorFunction (``self + b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset to add element-wise to the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + b``.
        """

    @overload
    def __add__(self, arg: Segment2, /) -> VectorFunction:
        """
        Add two VectorFunctions of the same type (``self + other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + other(x)``.
        """

    @overload
    def __add__(self, arg: VectorFunction, /) -> VectorFunction: ...

    def __radd__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __sub__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Subtract a constant vector from this VectorFunction (``self - b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant to subtract element-wise from the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - b``.
        """

    @overload
    def __sub__(self, arg: Segment2, /) -> VectorFunction:
        """
        Subtract two VectorFunctions of the same type (``self - other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - other(x)``.
        """

    @overload
    def __sub__(self, arg: VectorFunction, /) -> VectorFunction: ...

    def __rsub__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __mul__(self, arg: float, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar constant (``self * b``).

        Parameters
        ----------
        b : float
            Scalar multiplier applied to every output component.

        Returns
        -------
        VectorFunction
            Expression evaluating ``b * self(x)``.
        """

    @overload
    def __mul__(self, arg: Element, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar-valued VectorFunction (``self * s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; its runtime value scales every output
            component of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``s(x) * self(x)`` element-wise.
        """

    @overload
    def __mul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: float, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def __neg__(self) -> VectorFunction:
        """
        Negate this VectorFunction (``-self``).

        Returns
        -------
        VectorFunction
            Expression evaluating ``-self(x)``.
        """

    @overload
    def __truediv__(self, arg: float, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar constant (``self / b``).

        A separate overload accepts a scalar VectorFunction divisor.

        Parameters
        ----------
        b : float
            Constant divisor; every output component is divided by ``b``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / b``.
        """

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar-valued VectorFunction (``self / s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; every output component of ``self`` is
            divided by the runtime value of ``s``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / s(x)`` element-wise.
        """

    @overload
    def __truediv__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def sum(self) -> ScalarFunction:
        """
        Sum all output components into a scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``Σ_i self(x)[i]``.
        """

    @overload
    def normalized_power3(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Shifted and normalized-power-3 of this VectorFunction.

        Computes ``(self(x) + b) / ||self(x) + b||³``.  This operation appears
        frequently in gravitational and electrostatic force expressions.

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset added to the output of ``self`` before normalizing.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction:
        """
        Scaled shifted normalized-power-3: ``s * (self(x) + b) / ||self(x) + b||³``; see :meth:`normalized_power3`.
        """

    @overload
    def normalized_power3(self) -> VectorFunction:
        """
        Output divided by the cubed Euclidean norm: ``self(x) / ||self(x)||³``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂³``.
        """

    def norm(self) -> ScalarFunction:
        """
        Euclidean norm of the output vector: ``||self(x)||``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``||self(x)||₂``.
        """

    def squared_norm(self) -> ScalarFunction:
        """
        Squared Euclidean norm of the output vector: ``||self(x)||²``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``||self(x)||₂²``.
        """

    def cubed_norm(self) -> ScalarFunction:
        """
        Euclidean norm of the output raised to the third power: ``||self(x)||³``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``||self(x)||₂³``.
        """

    def inverse_norm(self) -> ScalarFunction:
        """
        Reciprocal Euclidean norm of the output: ``1 / ||self(x)||``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``1 / ||self(x)||₂``.
        """

    def inverse_squared_norm(self) -> ScalarFunction:
        """
        Reciprocal squared Euclidean norm: ``1 / ||self(x)||²``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``1 / ||self(x)||₂²``.
        """

    def inverse_cubed_norm(self) -> ScalarFunction:
        """
        Reciprocal cubed Euclidean norm: ``1 / ||self(x)||³``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``1 / ||self(x)||₂³``.
        """

    def inverse_four_norm(self) -> ScalarFunction:
        """
        Reciprocal fourth-power Euclidean norm: ``1 / ||self(x)||⁴``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``1 / ||self(x)||₂⁴``.
        """

    def normalized(self) -> VectorFunction:
        """
        Normalize the output to unit Euclidean length: ``self(x) / ||self(x)||``.

        Returns
        -------
        VectorFunction
            Expression evaluating the unit vector ``self(x) / ||self(x)||₂``.
        """

    def normalized_power2(self) -> VectorFunction:
        """
        Output divided by the squared Euclidean norm: ``self(x) / ||self(x)||²``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂²``.
        """

    def normalized_power4(self) -> VectorFunction:
        """
        Output divided by the fourth power of its Euclidean norm: ``self(x) / ||self(x)||⁴``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂⁴``.
        """

    def normalized_power5(self) -> VectorFunction:
        """
        Output divided by the fifth power of its Euclidean norm: ``self(x) / ||self(x)||⁵``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂⁵``.
        """

    @overload
    def dot(self, arg: Segment, /) -> ScalarFunction:
        """Overload for a generic segment right-hand operand; see :meth:`dot`."""

    @overload
    def dot(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`dot`.
        """

    @overload
    def dot(self, arg: Segment2, /) -> ScalarFunction:
        """
        Dot product of this VectorFunction with another equal-length one.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) · other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`dot`.
        """

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise product of this VectorFunction with a segment VectorFunction.

        Parameters
        ----------
        other : Segment VectorFunction, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.
            This overload accepts a generic segment; sibling overloads handle
            other concrete VectorFunction types or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) .* other(x)`` element-wise.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: Segment2, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise quotient of this VectorFunction divided by a segment VectorFunction.

        Parameters
        ----------
        other : Segment VectorFunction, shape (n,)
            Divisor; must have the same output dimension as ``self``.
            This overload accepts a generic segment; sibling overloads handle
            other concrete VectorFunction types or a constant vector divisor.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) ./ other(x)`` element-wise.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: Segment2, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector divisor; see :meth:`cwise_quotient`.
        """

    @overload
    def apply(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Apply an outer VectorFunction to the output of this one: ``g(self(.))``.

        Forms the composition ``x ↦ g(self(x))``, with ``self`` as the *inner*
        function and ``g`` as the *outer* function.  This is the reverse of
        :meth:`eval`; use it when you have an outer function ``g`` and want to
        pipe ``self`` through it.

        Parameters
        ----------
        g : VectorFunction
            Outer function; its input dimension must match ``self``'s output dimension.

        Returns
        -------
        VectorFunction
            Expression evaluating ``g(self(x))``.
        """

    @overload
    def apply(self, arg: ScalarFunction, /) -> ScalarFunction:
        """Overload for a scalar outer function; see :meth:`apply`."""

    def padded_lower(self, arg: int, /) -> VectorFunction:
        """
        Append zero rows below the output of this VectorFunction.

        Parameters
        ----------
        lpad : int
            Number of zero rows to append after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[self(x); 0, …, 0]`` with ``lpad`` trailing zeros.
        """

    def padded_upper(self, arg: int, /) -> VectorFunction:
        """
        Prepend zero rows above the output of this VectorFunction.

        Parameters
        ----------
        upad : int
            Number of zero rows to prepend before the first output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0, …, 0; self(x)]`` with ``upad`` leading zeros.
        """

    def padded(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Pad the output of this VectorFunction with zeros above and below.

        Parameters
        ----------
        upad : int
            Number of zero rows prepended before the first output component.
        lpad : int
            Number of zero rows appended after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0…; self(x); 0…]``.
        """

    def segment(self, arg0: int, arg1: int, /) -> Segment:
        """
        Extract a contiguous sub-vector of the output.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.
        size : int
            Number of consecutive output elements to extract.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + size]``.
        """

    def head(self, arg: int, /) -> Segment:
        """
        Extract the first ``size`` output components.

        Parameters
        ----------
        size : int
            Number of leading output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:size]``.
        """

    def tail(self, arg: int, /) -> Segment:
        """
        Extract the last ``size`` output components.

        Parameters
        ----------
        size : int
            Number of trailing output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-size:]``.
        """

    def coeff(self, arg: int, /) -> Element:
        """
        Extract a single scalar output component by index.

        Parameters
        ----------
        elem : int
            Zero-based index of the output element to select.

        Returns
        -------
        ScalarFunction
            Expression whose output is the scalar ``self(x)[elem]``.
        """

    @overload
    def __getitem__(self, arg: int, /) -> Element:
        """
        Extract a single output element by integer index (``self[i]``).

        A ``slice`` argument is handled by a sibling overload and raises
        ``ValueError`` if the slice is empty, backward, or non-contiguous.

        Parameters
        ----------
        index : int
            Zero-based index of the output element to select.

        Returns
        -------
        VectorFunction
            Scalar expression for the selected output element.
        """

    @overload
    def __getitem__(self, arg: slice, /) -> Segment: ...

    @overload
    def tolist(self) -> list[Element]:
        """
        Split the function into a list of scalar element functions.

        Returns one scalar-valued VectorFunction per output row, in order, each
        extracting a single component of the original output vector.

        Returns
        -------
        list of Element
            A list of ``output_rows`` scalar Element functions, where element *i*
            extracts output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]:
        """
        Split selected output components into a list of scalar element functions.

        Returns one scalar-valued VectorFunction for each index in *coeffs*, each
        extracting the corresponding output component.

        Parameters
        ----------
        coeffs : list of int
            Output-component indices to extract, in the requested order.

        Returns
        -------
        list of Element
            A list of scalar Element functions, one per entry in *coeffs*.
        """

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]:
        """
        Split the function into a mixed list of element and segment functions.

        Each entry in *seglist* is a ``(start, size)`` pair. A pair with ``size == 1``
        produces a scalar Element function; ``size == 2`` or ``3`` produces a
        fixed-width Segment; any other size produces a dynamic-length Segment.

        Parameters
        ----------
        seglist : list of (int, int)
            List of ``(start, size)`` pairs specifying which contiguous slices of the
            output vector to extract.

        Returns
        -------
        list of Element or Segment
            A heterogeneous list of VectorFunctions, one per entry in *seglist*.
        """

class Segment3:
    """
    A contiguous sub-vector view of a VectorFunction's input.

    ``Segment(input_rows, output_rows, start)`` extracts ``output_rows``
    consecutive components of the input vector beginning at index ``start``,
    producing a new vector-valued VectorFunction. It is the standard mechanism for
    slicing the symbolic input — obtained by indexing ``Arguments`` objects (e.g.
    ``x[2:5]``) or by calling ``.segment(start, size)`` on any VectorFunction.
    """

    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None:
        """
        Construct a Segment by specifying input size, segment length, and start index.

        Parameters
        ----------
        input_rows : int
            Total number of input rows.
        output_rows : int
            Number of consecutive input components to extract (segment length).
        start : int
            Index of the first input component to extract.
        """

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

    def vf(self) -> VectorFunction:
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

    __array_ufunc__: None = None

    @overload
    def __add__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Add a constant vector to this VectorFunction (``self + b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset to add element-wise to the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + b``.
        """

    @overload
    def __add__(self, arg: Segment3, /) -> VectorFunction:
        """
        Add two VectorFunctions of the same type (``self + other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) + other(x)``.
        """

    @overload
    def __add__(self, arg: VectorFunction, /) -> VectorFunction: ...

    def __radd__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __sub__(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Subtract a constant vector from this VectorFunction (``self - b``).

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant to subtract element-wise from the output of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - b``.
        """

    @overload
    def __sub__(self, arg: Segment3, /) -> VectorFunction:
        """
        Subtract two VectorFunctions of the same type (``self - other``).

        Parameters
        ----------
        other : VectorFunction
            Right-hand operand; must have the same input and output dimensions as ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) - other(x)``.
        """

    @overload
    def __sub__(self, arg: VectorFunction, /) -> VectorFunction: ...

    def __rsub__(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def __mul__(self, arg: float, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar constant (``self * b``).

        Parameters
        ----------
        b : float
            Scalar multiplier applied to every output component.

        Returns
        -------
        VectorFunction
            Expression evaluating ``b * self(x)``.
        """

    @overload
    def __mul__(self, arg: Element, /) -> VectorFunction:
        """
        Scale this VectorFunction by a scalar-valued VectorFunction (``self * s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; its runtime value scales every output
            component of ``self``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``s(x) * self(x)`` element-wise.
        """

    @overload
    def __mul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: float, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __rmul__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def __neg__(self) -> VectorFunction:
        """
        Negate this VectorFunction (``-self``).

        Returns
        -------
        VectorFunction
            Expression evaluating ``-self(x)``.
        """

    @overload
    def __truediv__(self, arg: float, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar constant (``self / b``).

        A separate overload accepts a scalar VectorFunction divisor.

        Parameters
        ----------
        b : float
            Constant divisor; every output component is divided by ``b``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / b``.
        """

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction: ...

    @overload
    def __truediv__(self, arg: Element, /) -> VectorFunction:
        """
        Divide this VectorFunction by a scalar-valued VectorFunction (``self / s``).

        Parameters
        ----------
        s : ScalarFunction
            A single-output VectorFunction; every output component of ``self`` is
            divided by the runtime value of ``s``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / s(x)`` element-wise.
        """

    @overload
    def __truediv__(self, arg: ScalarFunction, /) -> VectorFunction: ...

    def sum(self) -> ScalarFunction:
        """
        Sum all output components into a scalar VectorFunction.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``Σ_i self(x)[i]``.
        """

    @overload
    def normalized_power3(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Shifted and normalized-power-3 of this VectorFunction.

        Computes ``(self(x) + b) / ||self(x) + b||³``.  This operation appears
        frequently in gravitational and electrostatic force expressions.

        Parameters
        ----------
        b : array_like, shape (output_rows,)
            Constant offset added to the output of ``self`` before normalizing.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction:
        """
        Scaled shifted normalized-power-3: ``s * (self(x) + b) / ||self(x) + b||³``; see :meth:`normalized_power3`.
        """

    @overload
    def normalized_power3(self) -> VectorFunction:
        """
        Output divided by the cubed Euclidean norm: ``self(x) / ||self(x)||³``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂³``.
        """

    def norm(self) -> ScalarFunction:
        """
        Euclidean norm of the output vector: ``||self(x)||``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``||self(x)||₂``.
        """

    def squared_norm(self) -> ScalarFunction:
        """
        Squared Euclidean norm of the output vector: ``||self(x)||²``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``||self(x)||₂²``.
        """

    def cubed_norm(self) -> ScalarFunction:
        """
        Euclidean norm of the output raised to the third power: ``||self(x)||³``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``||self(x)||₂³``.
        """

    def inverse_norm(self) -> ScalarFunction:
        """
        Reciprocal Euclidean norm of the output: ``1 / ||self(x)||``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``1 / ||self(x)||₂``.
        """

    def inverse_squared_norm(self) -> ScalarFunction:
        """
        Reciprocal squared Euclidean norm: ``1 / ||self(x)||²``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``1 / ||self(x)||₂²``.
        """

    def inverse_cubed_norm(self) -> ScalarFunction:
        """
        Reciprocal cubed Euclidean norm: ``1 / ||self(x)||³``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``1 / ||self(x)||₂³``.
        """

    def inverse_four_norm(self) -> ScalarFunction:
        """
        Reciprocal fourth-power Euclidean norm: ``1 / ||self(x)||⁴``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``1 / ||self(x)||₂⁴``.
        """

    def normalized(self) -> VectorFunction:
        """
        Normalize the output to unit Euclidean length: ``self(x) / ||self(x)||``.

        Returns
        -------
        VectorFunction
            Expression evaluating the unit vector ``self(x) / ||self(x)||₂``.
        """

    def normalized_power2(self) -> VectorFunction:
        """
        Output divided by the squared Euclidean norm: ``self(x) / ||self(x)||²``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂²``.
        """

    def normalized_power4(self) -> VectorFunction:
        """
        Output divided by the fourth power of its Euclidean norm: ``self(x) / ||self(x)||⁴``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂⁴``.
        """

    def normalized_power5(self) -> VectorFunction:
        """
        Output divided by the fifth power of its Euclidean norm: ``self(x) / ||self(x)||⁵``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂⁵``.
        """

    @overload
    def cross(self, arg: Segment, /) -> VectorFunction:
        """
        3-D cross product of this VectorFunction with another.

        Both operands must produce 3-element output vectors.  The result
        is the standard right-hand-rule cross product ``self(x) × other(x)``.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (3,)
            Right-hand operand.  Accepts a VectorFunction of any concrete type
            or a constant 3-element vector.

        Returns
        -------
        VectorFunction
            3-element VectorFunction evaluating ``self(x) × other(x)``.
        """

    @overload
    def cross(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction:
        """
        Overload accepting a constant 3-element array right-hand operand; see :meth:`cross`.
        """

    @overload
    def cross(self, arg: Segment3, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cross`.
        """

    @overload
    def dot(self, arg: Segment, /) -> ScalarFunction:
        """Overload for a generic segment right-hand operand; see :meth:`dot`."""

    @overload
    def dot(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`dot`.
        """

    @overload
    def dot(self, arg: Segment3, /) -> ScalarFunction:
        """
        Dot product of this VectorFunction with another equal-length one.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.

        Returns
        -------
        ScalarFunction
            Expression evaluating ``self(x) · other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`dot`.
        """

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise product of this VectorFunction with a segment VectorFunction.

        Parameters
        ----------
        other : Segment VectorFunction, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.
            This overload accepts a generic segment; sibling overloads handle
            other concrete VectorFunction types or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) .* other(x)`` element-wise.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: Segment3, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_product`.
        """

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector right-hand operand; see :meth:`cwise_product`.
        """

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise quotient of this VectorFunction divided by a segment VectorFunction.

        Parameters
        ----------
        other : Segment VectorFunction, shape (n,)
            Divisor; must have the same output dimension as ``self``.
            This overload accepts a generic segment; sibling overloads handle
            other concrete VectorFunction types or a constant vector divisor.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) ./ other(x)`` element-wise.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Overload for a generic VectorFunction right-hand operand; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: Segment3, /) -> VectorFunction:
        """
        Overload for two operands of the same concrete type; see :meth:`cwise_quotient`.
        """

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Overload accepting a constant vector divisor; see :meth:`cwise_quotient`.
        """

    @overload
    def apply(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Apply an outer VectorFunction to the output of this one: ``g(self(.))``.

        Forms the composition ``x ↦ g(self(x))``, with ``self`` as the *inner*
        function and ``g`` as the *outer* function.  This is the reverse of
        :meth:`eval`; use it when you have an outer function ``g`` and want to
        pipe ``self`` through it.

        Parameters
        ----------
        g : VectorFunction
            Outer function; its input dimension must match ``self``'s output dimension.

        Returns
        -------
        VectorFunction
            Expression evaluating ``g(self(x))``.
        """

    @overload
    def apply(self, arg: ScalarFunction, /) -> ScalarFunction:
        """Overload for a scalar outer function; see :meth:`apply`."""

    def padded_lower(self, arg: int, /) -> VectorFunction:
        """
        Append zero rows below the output of this VectorFunction.

        Parameters
        ----------
        lpad : int
            Number of zero rows to append after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[self(x); 0, …, 0]`` with ``lpad`` trailing zeros.
        """

    def padded_upper(self, arg: int, /) -> VectorFunction:
        """
        Prepend zero rows above the output of this VectorFunction.

        Parameters
        ----------
        upad : int
            Number of zero rows to prepend before the first output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0, …, 0; self(x)]`` with ``upad`` leading zeros.
        """

    def padded(self, arg0: int, arg1: int, /) -> VectorFunction:
        """
        Pad the output of this VectorFunction with zeros above and below.

        Parameters
        ----------
        upad : int
            Number of zero rows prepended before the first output component.
        lpad : int
            Number of zero rows appended after the last output component.

        Returns
        -------
        VectorFunction
            Expression whose output is ``[0…; self(x); 0…]``.
        """

    def segment(self, arg0: int, arg1: int, /) -> Segment:
        """
        Extract a contiguous sub-vector of the output.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.
        size : int
            Number of consecutive output elements to extract.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + size]``.
        """

    def head(self, arg: int, /) -> Segment:
        """
        Extract the first ``size`` output components.

        Parameters
        ----------
        size : int
            Number of leading output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:size]``.
        """

    def tail(self, arg: int, /) -> Segment:
        """
        Extract the last ``size`` output components.

        Parameters
        ----------
        size : int
            Number of trailing output elements to keep.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-size:]``.
        """

    def coeff(self, arg: int, /) -> Element:
        """
        Extract a single scalar output component by index.

        Parameters
        ----------
        elem : int
            Zero-based index of the output element to select.

        Returns
        -------
        ScalarFunction
            Expression whose output is the scalar ``self(x)[elem]``.
        """

    @overload
    def __getitem__(self, arg: int, /) -> Element:
        """
        Extract a single output element by integer index (``self[i]``).

        A ``slice`` argument is handled by a sibling overload and raises
        ``ValueError`` if the slice is empty, backward, or non-contiguous.

        Parameters
        ----------
        index : int
            Zero-based index of the output element to select.

        Returns
        -------
        VectorFunction
            Scalar expression for the selected output element.
        """

    @overload
    def __getitem__(self, arg: slice, /) -> Segment: ...

    def segment_2(self, arg: int, /) -> Segment2:
        """
        Extract a 2-element sub-vector of the output starting at a given index.

        Parameters
        ----------
        start : int
            Zero-based index of the first output element to include.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[start : start + 2]``.
        """

    def head_2(self) -> Segment2:
        """
        Extract the first 2 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[:2]``.
        """

    def tail_2(self) -> Segment2:
        """
        Extract the last 2 output components.

        Returns
        -------
        VectorFunction
            Expression whose output is ``self(x)[-2:]``.
        """

    def segment2(self, arg: int, /) -> Segment2:
        """Alias for :meth:`segment_2`."""

    def head2(self) -> Segment2:
        """Alias for :meth:`head_2`."""

    def tail2(self) -> Segment2:
        """Alias for :meth:`tail_2`."""

    @overload
    def tolist(self) -> list[Element]:
        """
        Split the function into a list of scalar element functions.

        Returns one scalar-valued VectorFunction per output row, in order, each
        extracting a single component of the original output vector.

        Returns
        -------
        list of Element
            A list of ``output_rows`` scalar Element functions, where element *i*
            extracts output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]:
        """
        Split selected output components into a list of scalar element functions.

        Returns one scalar-valued VectorFunction for each index in *coeffs*, each
        extracting the corresponding output component.

        Parameters
        ----------
        coeffs : list of int
            Output-component indices to extract, in the requested order.

        Returns
        -------
        list of Element
            A list of scalar Element functions, one per entry in *coeffs*.
        """

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]:
        """
        Split the function into a mixed list of element and segment functions.

        Each entry in *seglist* is a ``(start, size)`` pair. A pair with ``size == 1``
        produces a scalar Element function; ``size == 2`` or ``3`` produces a
        fixed-width Segment; any other size produces a dynamic-length Segment.

        Parameters
        ----------
        seglist : list of (int, int)
            List of ``(start, size)`` pairs specifying which contiguous slices of the
            output vector to extract.

        Returns
        -------
        list of Element or Segment
            A heterogeneous list of VectorFunctions, one per entry in *seglist*.
        """

@overload
def stack(arg: Sequence[ScalarFunction], /) -> VectorFunction:
    """
    Concatenate scalar-output VectorFunctions vertically into one VectorFunction.

    Each function in *funcs* must share the same input dimension.  Their scalar
    outputs are stacked in order to form a single multi-output VectorFunction
    whose output dimension equals ``len(funcs)``.

    Parameters
    ----------
    funcs : list[VectorFunction]
        Scalar-output (output dimension 1) functions sharing the same input
        dimension.

    Returns
    -------
    VectorFunction
        A VectorFunction whose output is the vertical concatenation of all
        inputs; its output dimension is ``len(funcs)``.

    Raises
    ------
    ValueError
        If any two functions have mismatched input dimensions.

    Examples
    --------
    >>> v = vf.Arguments(3)
    >>> f = vf.stack([v[0], v[1] + v[2]])
    >>> f.output_rows()
    2
    """

@overload
def stack(arg: Sequence[VectorFunction], /) -> VectorFunction: ...

@overload
def stack(arg0: VectorFunction, /, *args) -> VectorFunction: ...

@overload
def stack(arg0: float, /, *args) -> VectorFunction: ...

@overload
def stack(arg0: numpy.ndarray, /, *args) -> VectorFunction: ...

@overload
def stack(arg: Sequence[ScalarFunction], /) -> VectorFunction: ...

@overload
def stack(arg: Sequence[VectorFunction], /) -> VectorFunction: ...

@overload
def stack_scalar(arg: Sequence[ScalarFunction], /) -> VectorFunction:
    """
    Concatenate scalar-output VectorFunctions vertically into one VectorFunction.

    Equivalent to :func:`stack` but restricted to scalar-output (output
    dimension 1) functions.  All functions in *funcs* must share the same input
    dimension; their outputs are concatenated in order.

    Parameters
    ----------
    funcs : list[VectorFunction]
        Scalar-output (output dimension 1) functions sharing the same input
        dimension.

    Returns
    -------
    VectorFunction
        A VectorFunction whose output dimension equals ``len(funcs)`` formed by
        concatenating the scalar outputs of each element.

    Raises
    ------
    ValueError
        If any two functions have mismatched input dimensions.
    """

@overload
def stack_scalar(arg0: ScalarFunction, /, *args) -> VectorFunction: ...

@overload
def stack_scalar(arg0: float, /, *args) -> VectorFunction: ...

@overload
def stack_scalar(arg: Sequence[ScalarFunction], /) -> VectorFunction: ...

@overload
def sum(arg: Sequence[ScalarFunction], /) -> ScalarFunction:
    """
    Elementwise sum of several equal-output scalar VectorFunctions.

    Each function in *funcs* must have the same input dimension and the same
    output dimension (1).  The result evaluates ``f1(x) + f2(x) + ...``
    elementwise; its Jacobian and higher derivatives are accumulated from all
    addends.

    This overload requires scalar (output dimension 1) addends; the general
    ``sum(list[VectorFunction])`` overload accepts equal-output functions of any
    dimension.

    Parameters
    ----------
    funcs : list[VectorFunction]
        Scalar-output (output dimension 1) functions sharing the same input and
        output dimension.  The list must contain at least one element.

    Returns
    -------
    VectorFunction
        A scalar-output VectorFunction whose value is the elementwise sum of all
        inputs.

    Raises
    ------
    ValueError
        If any two functions have mismatched input or output dimensions, or the
        list is empty.
    """

@overload
def sum(arg: Sequence[VectorFunction], /) -> VectorFunction: ...

@overload
def sum(arg0: ScalarFunction, /, *args) -> ScalarFunction: ...

@overload
def sum(arg0: float, /, *args) -> ScalarFunction: ...

@overload
def sum(arg0: VectorFunction, /, *args) -> VectorFunction: ...

@overload
def sum(arg0: numpy.ndarray, /, *args) -> VectorFunction: ...

@overload
def sum(arg: Sequence[ScalarFunction], /) -> ScalarFunction: ...

@overload
def sum(arg: Sequence[VectorFunction], /) -> VectorFunction: ...

def sum_scalar(arg: Sequence[ScalarFunction], /) -> ScalarFunction:
    """
    Elementwise sum of several scalar-output VectorFunctions.

    Equivalent to :func:`sum` but restricted to scalar-output (output dimension
    1) functions.  Each function in *funcs* must share the same input dimension;
    the result evaluates ``f1(x) + f2(x) + ...`` and accumulates Jacobians and
    higher derivatives from all addends.

    Parameters
    ----------
    funcs : list[VectorFunction]
        Scalar-output (output dimension 1) functions sharing the same input
        dimension.  The list must contain at least one element.

    Returns
    -------
    VectorFunction
        A scalar-output VectorFunction whose value is the elementwise sum of all
        inputs.

    Raises
    ------
    ValueError
        If any two functions have mismatched input dimensions, or the list is
        empty.
    """

@overload
def sum_elems(arg: Sequence[Element], /) -> ScalarFunction:
    """
    Sum several scalar segment-output VectorFunctions into one scalar.

    Sums a list of single-element segment VectorFunctions (each extracting one
    output component) into a single scalar-output VectorFunction.  All segments
    must share the same input dimension.  Useful for building scalar objectives
    or constraints from individual state or control components.

    Parameters
    ----------
    funcs : list[Segment]
        Single-element segment VectorFunctions (output dimension 1) sharing the
        same input dimension.  The list must contain at least one element.

    Returns
    -------
    VectorFunction
        A scalar-output VectorFunction whose value is the sum of all segment
        values.

    Raises
    ------
    ValueError
        If the list is empty or any two elements have mismatched input
        dimensions.
    """

@overload
def sum_elems(arg0: Sequence[Element], arg1: Sequence[float], /) -> ScalarFunction:
    """
    Weighted sum of several scalar segment-output VectorFunctions.

    Scales each function in *funcs* by the corresponding entry in *scales* and
    returns their sum as a single scalar-output VectorFunction.  All segments
    must share the same input dimension.  Equivalent to
    ``sum_elems([s * f for s, f in zip(scales, funcs)])``.

    Parameters
    ----------
    funcs : list[Segment]
        Single-element segment VectorFunctions sharing the same input dimension.
        The list must contain at least one element.
    scales : list[float]
        Scalar multiplier for each function.  Must have at least as many entries
        as *funcs*; no length check is performed — passing a shorter *scales*
        causes undefined behavior.

    Returns
    -------
    VectorFunction
        A scalar-output VectorFunction whose value is
        ``scales[0]*funcs[0](x) + scales[1]*funcs[1](x) + ...``.

    Raises
    ------
    ValueError
        If the list is empty or any two elements have mismatched input
        dimensions.
    """

@overload
def normalize(arg: numpy.ndarray, /) -> numpy.ndarray:
    """
    Normalize a constant vector to unit length.

    Divides the input array by its Euclidean (L2) norm. This overload acts on a
    plain NumPy array rather than a VectorFunction; for the VectorFunction variant
    see the overloads that accept a VectorFunction argument.

    Parameters
    ----------
    x : array_like, shape (n,)
        Non-zero vector to normalize.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Unit vector in the direction of *x*, i.e. ``x / ||x||_2``.
    """

@overload
def normalize(arg: Arguments, /) -> object: ...

@overload
def normalize(arg: Segment, /) -> object: ...

@overload
def normalize(arg: Segment2, /) -> object: ...

@overload
def normalize(arg: Segment3, /) -> object: ...

@overload
def normalize(arg: VectorFunction, /) -> object:
    """
    Unit-vector direction of a vector-valued VectorFunction.

    Alias for :func:`normalized`.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction to normalise.

    Returns
    -------
    VectorFunction
        Vector VectorFunction with the same output dimension as *fun*, evaluating
        ``fun(x) / ||fun(x)||_2``.
    """

@overload
def normalized(arg: numpy.ndarray, /) -> numpy.ndarray:
    """
    Normalize a constant vector to unit length.

    Alias for :func:`normalize` when given a plain array. Divides the input by
    its Euclidean (L2) norm.

    Parameters
    ----------
    x : array_like, shape (n,)
        Non-zero vector to normalize.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Unit vector in the direction of *x*, i.e. ``x / ||x||_2``.
    """

@overload
def normalized(arg: Arguments, /) -> object: ...

@overload
def normalized(arg: Segment, /) -> object: ...

@overload
def normalized(arg: Segment2, /) -> object: ...

@overload
def normalized(arg: Segment3, /) -> object: ...

@overload
def normalized(arg: VectorFunction, /) -> object:
    """
    Unit-vector direction of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction to normalise.

    Returns
    -------
    VectorFunction
        Vector VectorFunction with the same output dimension as *fun*, evaluating
        ``fun(x) / ||fun(x)||_2``.
    """

@overload
def norm(arg: Arguments, /) -> object: ...

@overload
def norm(arg: Segment, /) -> object: ...

@overload
def norm(arg: Segment2, /) -> object: ...

@overload
def norm(arg: Segment3, /) -> object: ...

@overload
def norm(arg: VectorFunction, /) -> object:
    """
    Euclidean (L2) norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction whose output norm is computed.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating ``||fun(x)||_2``.
    """

@overload
def squared_norm(arg: Arguments, /) -> object: ...

@overload
def squared_norm(arg: Segment, /) -> object: ...

@overload
def squared_norm(arg: Segment2, /) -> object: ...

@overload
def squared_norm(arg: Segment3, /) -> object: ...

@overload
def squared_norm(arg: VectorFunction, /) -> object:
    """
    Squared Euclidean norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction whose squared norm is computed.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating ``||fun(x)||_2^2``.
    """

@overload
def cubed_norm(arg: Arguments, /) -> object: ...

@overload
def cubed_norm(arg: Segment, /) -> object: ...

@overload
def cubed_norm(arg: Segment2, /) -> object: ...

@overload
def cubed_norm(arg: Segment3, /) -> object: ...

@overload
def cubed_norm(arg: VectorFunction, /) -> object:
    """
    Euclidean norm raised to the third power of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction whose cubed norm is computed.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating ``||fun(x)||_2^3``.
    """

@overload
def inverse_norm(arg: Arguments, /) -> object: ...

@overload
def inverse_norm(arg: Segment, /) -> object: ...

@overload
def inverse_norm(arg: Segment2, /) -> object: ...

@overload
def inverse_norm(arg: Segment3, /) -> object: ...

@overload
def inverse_norm(arg: VectorFunction, /) -> object:
    """
    Reciprocal Euclidean norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction whose reciprocal norm is computed.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating ``1 / ||fun(x)||_2``.
    """

@overload
def inverse_squared_norm(arg: Arguments, /) -> object: ...

@overload
def inverse_squared_norm(arg: Segment, /) -> object: ...

@overload
def inverse_squared_norm(arg: Segment2, /) -> object: ...

@overload
def inverse_squared_norm(arg: Segment3, /) -> object: ...

@overload
def inverse_squared_norm(arg: VectorFunction, /) -> object:
    """
    Reciprocal squared Euclidean norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction whose reciprocal squared norm is computed.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating ``1 / ||fun(x)||_2^2``.
    """

@overload
def inverse_cubed_norm(arg: Arguments, /) -> object: ...

@overload
def inverse_cubed_norm(arg: Segment, /) -> object: ...

@overload
def inverse_cubed_norm(arg: Segment2, /) -> object: ...

@overload
def inverse_cubed_norm(arg: Segment3, /) -> object: ...

@overload
def inverse_cubed_norm(arg: VectorFunction, /) -> object:
    """
    Reciprocal cubed Euclidean norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction whose reciprocal cubed norm is computed.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating ``1 / ||fun(x)||_2^3``.
    """

@overload
def inverse_four_norm(arg: Arguments, /) -> object: ...

@overload
def inverse_four_norm(arg: Segment, /) -> object: ...

@overload
def inverse_four_norm(arg: Segment2, /) -> object: ...

@overload
def inverse_four_norm(arg: Segment3, /) -> object: ...

@overload
def inverse_four_norm(arg: VectorFunction, /) -> object:
    """
    Reciprocal fourth-power Euclidean norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction whose reciprocal fourth-power norm is computed.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating ``1 / ||fun(x)||_2^4``.
    """

@overload
def normalized_power2(arg: Arguments, /) -> object: ...

@overload
def normalized_power2(arg: Segment, /) -> object: ...

@overload
def normalized_power2(arg: Segment2, /) -> object: ...

@overload
def normalized_power2(arg: Segment3, /) -> object: ...

@overload
def normalized_power2(arg: VectorFunction, /) -> object:
    """
    Vector divided by the squared norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction.

    Returns
    -------
    VectorFunction
        Vector VectorFunction evaluating ``fun(x) / ||fun(x)||_2^2``.
    """

@overload
def normalized_power3(arg: Arguments, /) -> object: ...

@overload
def normalized_power3(arg: Segment, /) -> object: ...

@overload
def normalized_power3(arg: Segment2, /) -> object: ...

@overload
def normalized_power3(arg: Segment3, /) -> object: ...

@overload
def normalized_power3(arg: VectorFunction, /) -> object:
    """
    Vector divided by the cubed norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction.

    Returns
    -------
    VectorFunction
        Vector VectorFunction evaluating ``fun(x) / ||fun(x)||_2^3``.
    """

@overload
def normalized_power4(arg: Arguments, /) -> object: ...

@overload
def normalized_power4(arg: Segment, /) -> object: ...

@overload
def normalized_power4(arg: Segment2, /) -> object: ...

@overload
def normalized_power4(arg: Segment3, /) -> object: ...

@overload
def normalized_power4(arg: VectorFunction, /) -> object:
    """
    Vector divided by the fourth-power norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction.

    Returns
    -------
    VectorFunction
        Vector VectorFunction evaluating ``fun(x) / ||fun(x)||_2^4``.
    """

@overload
def normalized_power5(arg: Arguments, /) -> object: ...

@overload
def normalized_power5(arg: Segment, /) -> object: ...

@overload
def normalized_power5(arg: Segment2, /) -> object: ...

@overload
def normalized_power5(arg: Segment3, /) -> object: ...

@overload
def normalized_power5(arg: VectorFunction, /) -> object:
    """
    Vector divided by the fifth-power norm of a vector-valued VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Vector-valued VectorFunction.

    Returns
    -------
    VectorFunction
        Vector VectorFunction evaluating ``fun(x) / ||fun(x)||_2^5``.
    """

@overload
def cross(arg0: Segment3, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object:
    """
    3D cross product of two 3-vector quantities.

    Each operand may be a VectorFunction or a constant 3-vector array, but at
    least one must be a VectorFunction. Both operands must produce 3-element
    output vectors. The accepted operand-type combinations are listed in the
    overload signatures.

    Parameters
    ----------
    a, b : VectorFunction or array_like, shape (3,)
        The two 3-vector operands.

    Returns
    -------
    VectorFunction
        A 3-vector VectorFunction evaluating the cross product ``a(x) x b(x)``.

    Examples
    --------
    >>> v = vf.Arguments(3)
    >>> f = vf.cross(v, v)       # zero vector (self cross product)
    >>> f.output_rows()
    3
    """

@overload
def cross(arg0: Segment, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def cross(arg0: VectorFunction, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def cross(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: Segment3, /) -> object: ...

@overload
def cross(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: Segment, /) -> object: ...

@overload
def cross(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: VectorFunction, /) -> object: ...

@overload
def cross(arg0: Segment3, arg1: Segment3, /) -> object: ...

@overload
def cross(arg0: Segment3, arg1: Segment, /) -> object: ...

@overload
def cross(arg0: Segment3, arg1: VectorFunction, /) -> object: ...

@overload
def cross(arg0: Segment, arg1: Segment, /) -> object: ...

@overload
def cross(arg0: Segment, arg1: Segment3, /) -> object: ...

@overload
def cross(arg0: Segment, arg1: VectorFunction, /) -> object: ...

@overload
def cross(arg0: VectorFunction, arg1: Segment3, /) -> object: ...

@overload
def cross(arg0: VectorFunction, arg1: Segment, /) -> object: ...

@overload
def cross(arg0: VectorFunction, arg1: VectorFunction, /) -> object: ...

@overload
def doublecross(arg0: Segment3, arg1: Segment3, arg2: Segment3, /) -> VectorFunction:
    """
    Double (iterated) cross product of three 3-element Segment VectorFunctions.

    Computes ``(a x b) x c``.  This overload requires all three arguments to be
    fixed 3-element ``Segment`` types (``Segment<-1, 3, -1>``); for general
    VectorFunctions use the overload that accepts plain ``VectorFunction`` arguments.

    Parameters
    ----------
    a, b, c : Segment
        Three fixed 3-element segment VectorFunctions (output dimension 3).

    Returns
    -------
    VectorFunction
        A 3-vector VectorFunction evaluating ``(a(x) x b(x)) x c(x)``.
    """

@overload
def doublecross(arg0: VectorFunction, arg1: VectorFunction, arg2: VectorFunction, /) -> VectorFunction: ...

@overload
def cwise_product(arg0: Segment2, arg1: Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')], /) -> object:
    """
    Elementwise (Hadamard) product of two equal-length vector quantities.

    Each operand may be a VectorFunction or a constant array, but at least one
    must be a VectorFunction. Both operands must produce the same number of output
    elements. The result at each index ``i`` is ``a_i(x) * b_i(x)``.

    Parameters
    ----------
    a, b : VectorFunction or array_like
        The two operands, each with the same output dimension.

    Returns
    -------
    VectorFunction
        Vector VectorFunction evaluating the element-wise product ``a(x) * b(x)``
        with the same output dimension as the inputs.
    """

@overload
def cwise_product(arg0: Segment3, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def cwise_product(arg0: Segment, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def cwise_product(arg0: VectorFunction, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def cwise_product(arg0: Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')], arg1: Segment2, /) -> object: ...

@overload
def cwise_product(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: Segment3, /) -> object: ...

@overload
def cwise_product(arg0: numpy.ndarray, arg1: Segment, /) -> object: ...

@overload
def cwise_product(arg0: numpy.ndarray, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_product(arg0: Segment2, arg1: Segment2, /) -> object: ...

@overload
def cwise_product(arg0: Segment2, arg1: Segment, /) -> object: ...

@overload
def cwise_product(arg0: Segment2, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_product(arg0: Segment3, arg1: Segment3, /) -> object: ...

@overload
def cwise_product(arg0: Segment3, arg1: Segment, /) -> object: ...

@overload
def cwise_product(arg0: Segment3, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_product(arg0: Segment, arg1: Segment, /) -> object: ...

@overload
def cwise_product(arg0: Segment, arg1: Segment2, /) -> object: ...

@overload
def cwise_product(arg0: Segment, arg1: Segment3, /) -> object: ...

@overload
def cwise_product(arg0: Segment, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_product(arg0: VectorFunction, arg1: Segment2, /) -> object: ...

@overload
def cwise_product(arg0: VectorFunction, arg1: Segment3, /) -> object: ...

@overload
def cwise_product(arg0: VectorFunction, arg1: Segment, /) -> object: ...

@overload
def cwise_product(arg0: VectorFunction, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment2, arg1: Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')], /) -> object:
    """
    Elementwise quotient of two equal-length vector quantities.

    Each operand may be a VectorFunction or a constant array, but at least one
    must be a VectorFunction. Both operands must produce the same number of output
    elements. The result at each index ``i`` is ``a_i(x) / b_i(x)``.

    Parameters
    ----------
    a, b : VectorFunction or array_like
        Numerator and denominator, each with the same output dimension. The
        denominator must be non-zero at all evaluation points.

    Returns
    -------
    VectorFunction
        Vector VectorFunction evaluating the element-wise quotient ``a(x) / b(x)``
        with the same output dimension as the inputs.
    """

@overload
def cwise_quotient(arg0: Segment3, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def cwise_quotient(arg0: Segment, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def cwise_quotient(arg0: VectorFunction, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def cwise_quotient(arg0: Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')], arg1: Segment2, /) -> object: ...

@overload
def cwise_quotient(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: Segment3, /) -> object: ...

@overload
def cwise_quotient(arg0: numpy.ndarray, arg1: Segment, /) -> object: ...

@overload
def cwise_quotient(arg0: numpy.ndarray, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment2, arg1: Segment2, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment2, arg1: Segment, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment2, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment3, arg1: Segment3, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment3, arg1: Segment, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment3, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment, arg1: Segment, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment, arg1: Segment2, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment, arg1: Segment3, /) -> object: ...

@overload
def cwise_quotient(arg0: Segment, arg1: VectorFunction, /) -> object: ...

@overload
def cwise_quotient(arg0: VectorFunction, arg1: Segment2, /) -> object: ...

@overload
def cwise_quotient(arg0: VectorFunction, arg1: Segment3, /) -> object: ...

@overload
def cwise_quotient(arg0: VectorFunction, arg1: Segment, /) -> object: ...

@overload
def cwise_quotient(arg0: VectorFunction, arg1: VectorFunction, /) -> object: ...

@overload
def dot(arg0: Segment2, arg1: Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')], /) -> object:
    """
    Dot product of two equal-length vector quantities.

    Each operand may be a VectorFunction or a constant array, but at least one must
    be a VectorFunction, and both must have the same output dimension. The accepted
    operand-type combinations are listed in the overload signatures.

    Parameters
    ----------
    a, b : VectorFunction or array_like, shape (n,)
        The two operands.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating the dot product ``a(x) . b(x)``.

    Examples
    --------
    >>> v = vf.Arguments(3)
    >>> f = vf.dot(v, v)          # squared norm of the 3-vector
    >>> f.output_rows()
    1
    """

@overload
def dot(arg0: Segment3, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def dot(arg0: Segment, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def dot(arg0: VectorFunction, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> object: ...

@overload
def dot(arg0: Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')], arg1: Segment2, /) -> object: ...

@overload
def dot(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], arg1: Segment3, /) -> object: ...

@overload
def dot(arg0: numpy.ndarray, arg1: Segment, /) -> object: ...

@overload
def dot(arg0: numpy.ndarray, arg1: VectorFunction, /) -> object: ...

@overload
def dot(arg0: Segment2, arg1: Segment2, /) -> object: ...

@overload
def dot(arg0: Segment2, arg1: Segment, /) -> object: ...

@overload
def dot(arg0: Segment2, arg1: VectorFunction, /) -> object: ...

@overload
def dot(arg0: Segment3, arg1: Segment3, /) -> object: ...

@overload
def dot(arg0: Segment3, arg1: Segment, /) -> object: ...

@overload
def dot(arg0: Segment3, arg1: VectorFunction, /) -> object: ...

@overload
def dot(arg0: Segment, arg1: Segment, /) -> object: ...

@overload
def dot(arg0: Segment, arg1: Segment2, /) -> object: ...

@overload
def dot(arg0: Segment, arg1: Segment3, /) -> object: ...

@overload
def dot(arg0: Segment, arg1: VectorFunction, /) -> object: ...

@overload
def dot(arg0: VectorFunction, arg1: Segment2, /) -> object: ...

@overload
def dot(arg0: VectorFunction, arg1: Segment3, /) -> object: ...

@overload
def dot(arg0: VectorFunction, arg1: Segment, /) -> object: ...

@overload
def dot(arg0: VectorFunction, arg1: VectorFunction, /) -> object: ...

@overload
def scalar_root_finder(arg0: ScalarFunction, arg1: ScalarFunction, arg2: int, arg3: float, /) -> ScalarFunction:
    """
    Embed a scalar Newton root-finder as a VectorFunction.

    Constructs a VectorFunction that, when evaluated, runs Newton iteration on
    *fx* to find the root of the scalar residual with respect to the first input
    element (the iteration variable). The converged root is returned as a scalar
    output; its derivatives with respect to the remaining inputs are computed via
    the implicit-function theorem.

    The first element of the input vector is treated as the initial guess for the
    iteration variable; the iteration variable is updated internally during each
    Newton step (the caller's input array is not mutated — the iteration runs on an
    internal copy). The remaining elements are differentiated-through parameters.

    Parameters
    ----------
    fx : VectorFunction
        Scalar-valued residual function whose root is sought.
    dfx : VectorFunction
        Scalar-valued derivative of *fx* with respect to the iteration variable
        (first input element). When provided, this overload skips the automatic
        Jacobian evaluation inside the Newton loop.
    iter : int
        Maximum number of Newton iterations.
    tol : float
        Convergence tolerance on the absolute residual value.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction returning the converged root of *fx*.
    """

@overload
def scalar_root_finder(arg0: ScalarFunction, arg1: int, arg2: float, /) -> ScalarFunction: ...

@overload
def quat_product(arg0: Segment, arg1: Segment, /) -> VectorFunction:
    """
    Quaternion product of two 4-vector VectorFunctions.

    Computes the Hamilton product ``q1 ⊗ q2`` where each operand is a
    4-element quaternion in ``[x, y, z, w]`` storage order (vector part first,
    scalar part last).

    Parameters
    ----------
    q1 : VectorFunction
        Left quaternion operand, output dimension 4.
    q2 : VectorFunction
        Right quaternion operand, output dimension 4.

    Returns
    -------
    VectorFunction
        A 4-vector VectorFunction evaluating the quaternion product ``q1(x) ⊗ q2(x)``.
    """

@overload
def quat_product(arg0: Segment, arg1: VectorFunction, /) -> VectorFunction: ...

@overload
def quat_product(arg0: VectorFunction, arg1: Segment, /) -> VectorFunction: ...

@overload
def quat_product(arg0: VectorFunction, arg1: VectorFunction, /) -> VectorFunction: ...

@overload
def quat_rotate(arg0: VectorFunction, arg1: VectorFunction, /) -> VectorFunction:
    """
    Rotate a 3-vector by a unit quaternion.

    Computes the active rotation ``q ⊗ [v; 0] ⊗ q*``, returning only the vector
    part. The quaternion *q* must be stored in ``[x, y, z, w]`` order (vector part
    first, scalar part last) and should be a unit quaternion at all evaluation
    points.

    Parameters
    ----------
    q : VectorFunction
        Unit quaternion operand, output dimension 4, in ``[x, y, z, w]`` order.
    v : VectorFunction or array_like, shape (3,)
        The 3-vector to rotate.

    Returns
    -------
    VectorFunction
        A 3-vector VectorFunction evaluating the rotated vector ``q(x) * v(x)``.
    """

@overload
def quat_rotate(arg0: VectorFunction, arg1: Segment3, /) -> VectorFunction: ...

@overload
def quat_rotate(arg0: VectorFunction, arg1: Segment, /) -> VectorFunction: ...

@overload
def quat_rotate(arg0: VectorFunction, arg1: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction: ...

@overload
def sin(arg: Element, /) -> object: ...

@overload
def sin(arg: ScalarFunction, /) -> object:
    """
    Elementwise sine of a scalar VectorFunction (argument in radians).

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``sin(fun(x))``.

    Examples
    --------
    >>> x = vf.Arguments(1)[0]
    >>> f = vf.sin(x)
    >>> f.output_rows()
    1
    """

@overload
def cos(arg: Element, /) -> object: ...

@overload
def cos(arg: ScalarFunction, /) -> object:
    """
    Elementwise cosine of a scalar VectorFunction (argument in radians).

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``cos(fun(x))``.
    """

@overload
def tan(arg: Element, /) -> object: ...

@overload
def tan(arg: ScalarFunction, /) -> object:
    """
    Elementwise tangent of a scalar VectorFunction (argument in radians).

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``tan(fun(x))``.
    """

@overload
def sqrt(arg: Element, /) -> object: ...

@overload
def sqrt(arg: ScalarFunction, /) -> object:
    """
    Elementwise square root of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction. The value must be
        non-negative at all evaluation points.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``sqrt(fun(x))``.
    """

@overload
def exp(arg: Element, /) -> object: ...

@overload
def exp(arg: ScalarFunction, /) -> object:
    """
    Elementwise natural exponential of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``exp(fun(x))``.
    """

@overload
def log(arg: Element, /) -> object: ...

@overload
def log(arg: ScalarFunction, /) -> object:
    """
    Elementwise natural logarithm of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction. The value must be
        strictly positive at all evaluation points.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``log(fun(x))`` (natural logarithm).
    """

@overload
def squared(arg: Element, /) -> object: ...

@overload
def squared(arg: ScalarFunction, /) -> object:
    """
    Elementwise square of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``fun(x)**2``.
    """

@overload
def arcsin(arg: Element, /) -> object: ...

@overload
def arcsin(arg: ScalarFunction, /) -> object:
    """
    Elementwise arcsine of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction. The value must lie in
        ``[-1, 1]`` at all evaluation points.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``arcsin(fun(x))``, with result in
        ``[-pi/2, pi/2]``.
    """

@overload
def arccos(arg: Element, /) -> object: ...

@overload
def arccos(arg: ScalarFunction, /) -> object:
    """
    Elementwise arccosine of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction. The value must lie in
        ``[-1, 1]`` at all evaluation points.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``arccos(fun(x))``, with result in
        ``[0, pi]``.
    """

@overload
def arctan(arg: Element, /) -> object: ...

@overload
def arctan(arg: ScalarFunction, /) -> object:
    """
    Elementwise arctangent of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``arctan(fun(x))``, with result in
        ``(-pi/2, pi/2)``.
    """

@overload
def sinh(arg: Element, /) -> object: ...

@overload
def sinh(arg: ScalarFunction, /) -> object:
    """
    Elementwise hyperbolic sine of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``sinh(fun(x))``.
    """

@overload
def cosh(arg: Element, /) -> object: ...

@overload
def cosh(arg: ScalarFunction, /) -> object:
    """
    Elementwise hyperbolic cosine of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``cosh(fun(x))``.
    """

@overload
def tanh(arg: Element, /) -> object: ...

@overload
def tanh(arg: ScalarFunction, /) -> object:
    """
    Elementwise hyperbolic tangent of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``tanh(fun(x))``.
    """

@overload
def pow(arg0: Element, arg1: float, /) -> object: ...

@overload
def pow(arg0: ScalarFunction, arg1: float, /) -> object:
    """
    Raise a scalar VectorFunction to a real power.

    Equivalent to the method form ``fun.pow(power)``.  Note that the ``**``
    operator may dispatch to a different expression for certain integer exponents:
    for example, ``f ** 2`` uses the ``.square()`` path (a ``CwiseSquare``
    expression) rather than ``CwisePow``, so the symbolic expression trees produced
    by ``vf.pow(f, 2)`` and ``f ** 2`` are not identical, even though they evaluate
    to the same values and derivatives.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction. The base value must be
        positive at all evaluation points for non-integer exponents.
    power : float
        The real exponent.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``fun(x) ** power`` via the
        ``CwisePow`` expression.
    """

@overload
def arcsinh(arg: Element, /) -> object: ...

@overload
def arcsinh(arg: ScalarFunction, /) -> object:
    """
    Elementwise inverse hyperbolic sine of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``arcsinh(fun(x))``.
    """

@overload
def arccosh(arg: Element, /) -> object: ...

@overload
def arccosh(arg: ScalarFunction, /) -> object:
    """
    Elementwise inverse hyperbolic cosine of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction. The value must be
        greater than or equal to 1 at all evaluation points.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``arccosh(fun(x))``.
    """

@overload
def arctanh(arg: Element, /) -> object: ...

@overload
def arctanh(arg: ScalarFunction, /) -> object:
    """
    Elementwise inverse hyperbolic tangent of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction. The value must lie
        strictly within ``(-1, 1)`` at all evaluation points.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``arctanh(fun(x))``.
    """

@overload
def sign(arg: Element, /) -> object: ...

@overload
def sign(arg: ScalarFunction, /) -> object:
    """
    Elementwise sign of a scalar VectorFunction.

    The sign function returns -1 for negative values, 0 for zero, and +1 for
    positive values. Its derivative is zero almost everywhere; Jacobian and
    Hessian contributions are not propagated.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``sign(fun(x))`` in ``{-1, 0, 1}``.
    """

@overload
def abs(arg: Element, /) -> object: ...

@overload
def abs(arg: ScalarFunction, /) -> object:
    """
    Elementwise absolute value of a scalar VectorFunction.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``|fun(x)|``.
    """

@overload
def ifelse(arg0: Conditional, arg1: ScalarFunction, arg2: ScalarFunction, /) -> ScalarFunction:
    """
    Build a scalar VectorFunction that selects between two scalar branches based on a predicate.

    At each evaluation point the predicate *test* is checked. If true, *tf* is
    evaluated; otherwise *ff* is evaluated. Derivatives are taken from whichever
    branch is active, so the result is only piecewise differentiable across the
    switching boundary.

    All three arguments must share the same input dimension, and both branch
    functions must be scalar-valued (output rows = 1).

    Parameters
    ----------
    test : Conditional
        Predicate that selects the active branch.
    tf : GenericFunction (scalar)
        Function evaluated when *test* is true.
    ff : GenericFunction (scalar)
        Function evaluated when *test* is false.

    Returns
    -------
    GenericFunction (scalar)
        Scalar function returning ``tf(x)`` when ``test(x)`` is true, else ``ff(x)``.

    Raises
    ------
    ValueError
        If *test*, *tf*, and *ff* do not share the same input dimension, or if
        *tf* and *ff* have different output dimensions.
    """

@overload
def ifelse(arg0: Conditional, arg1: float, arg2: ScalarFunction, /) -> ScalarFunction:
    """
    Build a scalar VectorFunction that returns a constant when the predicate is true, else a scalar function.

    Parameters
    ----------
    test : Conditional
        Predicate that selects the active branch.
    tfv : float
        Constant value returned when *test* is true.
    ff : GenericFunction (scalar)
        Function evaluated when *test* is false.

    Returns
    -------
    GenericFunction (scalar)
        Scalar function returning *tfv* when ``test(x)`` is true, else ``ff(x)``.
    """

@overload
def ifelse(arg0: Conditional, arg1: ScalarFunction, arg2: float, /) -> ScalarFunction:
    """
    Build a scalar VectorFunction that returns a scalar function when the predicate is true, else a constant.

    Parameters
    ----------
    test : Conditional
        Predicate that selects the active branch.
    tf : GenericFunction (scalar)
        Function evaluated when *test* is true.
    ffv : float
        Constant value returned when *test* is false.

    Returns
    -------
    GenericFunction (scalar)
        Scalar function returning ``tf(x)`` when ``test(x)`` is true, else *ffv*.
    """

@overload
def ifelse(arg0: Conditional, arg1: float, arg2: float, /) -> ScalarFunction:
    """
    Build a scalar VectorFunction that selects between two scalar constants based on a predicate.

    Parameters
    ----------
    test : Conditional
        Predicate that selects the active branch.
    tfv : float
        Constant value returned when *test* is true.
    ffv : float
        Constant value returned when *test* is false.

    Returns
    -------
    GenericFunction (scalar)
        Scalar function returning *tfv* when ``test(x)`` is true, else *ffv*.
    """

@overload
def ifelse(arg0: Conditional, arg1: VectorFunction, arg2: VectorFunction, /) -> VectorFunction:
    """
    Build a vector-valued VectorFunction that selects between two vector-valued branches based on a predicate.

    At each evaluation point the predicate *test* is checked. If true, *tf* is
    evaluated; otherwise *ff* is evaluated. Both branch functions must share the
    same input and output dimensions as one another, and their input dimension
    must match that of *test*. Derivatives are taken from the active branch
    only, so the result is piecewise differentiable across the switching
    boundary.

    Parameters
    ----------
    test : Conditional
        Predicate that selects the active branch.
    tf : GenericFunction
        Vector-valued function evaluated when *test* is true.
    ff : GenericFunction
        Vector-valued function evaluated when *test* is false.

    Returns
    -------
    GenericFunction
        Vector function returning ``tf(x)`` when ``test(x)`` is true, else ``ff(x)``.

    Raises
    ------
    ValueError
        If *test*, *tf*, and *ff* do not share the same input dimension, or if
        *tf* and *ff* have different output dimensions.
    """

@overload
def ifelse(arg0: Conditional, arg1: numpy.ndarray, arg2: VectorFunction, /) -> VectorFunction:
    """
    Build a vector-valued VectorFunction that returns a constant vector when the predicate is true, else a vector-valued function.

    Parameters
    ----------
    test : Conditional
        Predicate that selects the active branch.
    v : array_like
        Constant vector returned when *test* is true; must have the same length
        as ``ff.output_rows()``.
    ff : GenericFunction
        Vector-valued function evaluated when *test* is false.

    Returns
    -------
    GenericFunction
        Vector function returning *v* when ``test(x)`` is true, else ``ff(x)``.
    """

@overload
def ifelse(arg0: Conditional, arg1: VectorFunction, arg2: numpy.ndarray, /) -> VectorFunction:
    """
    Build a vector-valued VectorFunction that returns a vector-valued function when the predicate is true, else a constant vector.

    Parameters
    ----------
    test : Conditional
        Predicate that selects the active branch.
    tf : GenericFunction
        Vector-valued function evaluated when *test* is true.
    v : array_like
        Constant vector returned when *test* is false; must have the same length
        as ``tf.output_rows()``.

    Returns
    -------
    GenericFunction
        Vector function returning ``tf(x)`` when ``test(x)`` is true, else *v*.
    """

@overload
def ifelse(arg0: Conditional, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> VectorFunction:
    """
    Build a vector-valued VectorFunction that selects between two constant vectors based on a predicate.

    Parameters
    ----------
    test : Conditional
        Predicate that selects the active branch.
    v1 : array_like
        Constant vector returned when *test* is true.
    v2 : array_like
        Constant vector returned when *test* is false; must have the same length
        as *v1*.

    Returns
    -------
    GenericFunction
        Vector function returning *v1* when ``test(x)`` is true, else *v2*.
    """

@overload
def arctan2(arg0: ScalarFunction, arg1: ScalarFunction, /) -> ScalarFunction:
    """
    Quadrant-aware arctangent of two scalar VectorFunctions.

    Computes ``atan2(y(x), x(x))`` with analytic first and second derivatives.
    The result lies in ``(-pi, pi]`` and correctly identifies the quadrant based
    on the signs of both operands.

    Parameters
    ----------
    y : VectorFunction
        Scalar-valued VectorFunction for the opposite (sine-like) component.
    x : VectorFunction
        Scalar-valued VectorFunction for the adjacent (cosine-like) component.

    Returns
    -------
    VectorFunction
        Scalar VectorFunction evaluating ``atan2(y(x), x(x))``.
    """

@overload
def arctan2(arg0: Element, arg1: Element, /) -> ScalarFunction: ...

@overload
def arctan2(arg0: Element, arg1: ScalarFunction, /) -> ScalarFunction: ...

@overload
def arctan2(arg0: ScalarFunction, arg1: Element, /) -> ScalarFunction: ...

@overload
def divtest(arg0: VectorFunction, arg1: ScalarFunction, /) -> VectorFunction:
    """
    Divide a vector VectorFunction by a scalar VectorFunction (internal testing utility).

    Divides each element of *a* by the scalar *b*. This function exists primarily
    to exercise the ``VectorScalarFunctionDivision`` implementation and is not
    intended for production use.

    Parameters
    ----------
    a : VectorFunction
        Vector-valued VectorFunction (numerator).
    b : VectorFunction
        Scalar-valued VectorFunction (denominator). Must be non-zero at all
        evaluation points.

    Returns
    -------
    VectorFunction
        Vector VectorFunction evaluating ``a(x) / b(x)`` element-wise.
    """

@overload
def divtest(arg0: VectorFunction, arg1: Element, /) -> VectorFunction: ...

class ColMatrix:
    """
    Column-major matrix view of a VectorFunction.

    A ``ColMatrix`` wraps a flat VectorFunction whose output has length
    ``rows * cols`` and interprets that output as a column-major (Fortran-order)
    matrix.  The object supports matrix algebra — products, sums, scalar
    multiplication, transpose, and inverse — all expressed symbolically as new
    VectorFunctions so that derivatives propagate through them automatically.

    Construct a ``ColMatrix`` either from a backing VectorFunction with explicit
    shape, or from a Python list of same-length VectorFunctions that are stacked
    into columns.

    See Also
    --------
    RowMatrix : Row-major counterpart.
    matmul : Free-function matrix product for mixed operand types.
    """

    @overload
    def __init__(self, arg0: VectorFunction, arg1: int, arg2: int, /) -> None:
        """
        Construct a ColMatrix by reshaping a VectorFunction output into a column-major matrix.

        Parameters
        ----------
        f : VectorFunction
            Backing VectorFunction whose flat output (length ``rows * cols``) is reinterpreted
            as a column-major matrix.
        rows : int
            Number of rows in the matrix view.  Must satisfy ``rows * cols == f.output_rows()``.
        cols : int
            Number of columns in the matrix view.

        Raises
        ------
        ValueError
            If ``rows`` or ``cols`` is non-positive, or if ``rows * cols`` does not equal
            ``f.output_rows()``.
        """

    @overload
    def __init__(self, arg: Sequence[VectorFunction], /) -> None:
        """
        Construct a ColMatrix from a list of column VectorFunctions.

        Each element of ``colfuns`` becomes one column of the resulting matrix.
        All functions must share the same output length (which becomes the number
        of rows); the number of columns equals ``len(colfuns)``.

        Parameters
        ----------
        colfuns : list of VectorFunction
            Column vectors to stack.  Every element must have the same
            ``output_rows()``.

        Raises
        ------
        ValueError
            If ``colfuns`` is empty or if the functions do not all have the same
            output length.
        """

    @overload
    def __mul__(self, arg: ColMatrix, /) -> ColMatrix:
        """
        Matrix product of two ColMatrix VectorFunctions (``self @ other``).

        Computes ``M1(x) @ M2(x)`` where both factors are evaluated at the same input
        ``x``.  The inner dimension of ``M1`` must equal the outer dimension of ``M2``.
        Other right-hand-factor types (``RowMatrix``, plain ``VectorFunction``, ``float``)
        are handled by separate overloads.

        Parameters
        ----------
        other : ColMatrix
            Right-hand ColMatrix factor.

        Returns
        -------
        ColMatrix
            Column-major matrix representing the product.

        Raises
        ------
        ValueError
            If the inner dimensions of the two matrix functions do not match.
        """

    @overload
    def __mul__(self, arg: RowMatrix, /) -> ColMatrix:
        """
        Matrix product of a ColMatrix with a RowMatrix VectorFunction (``self @ other``).

        Overload of ``__mul__`` accepting a ``RowMatrix`` right-hand factor.
        See the primary ``__mul__`` overload for full documentation.
        """

    @overload
    def __mul__(self, arg: float, /) -> ColMatrix:
        """
        Scalar multiplication (``self * scale``).

        Parameters
        ----------
        scale : float
            Scalar factor applied to every element of the matrix function.

        Returns
        -------
        ColMatrix
            A new ColMatrix scaled by ``scale``.
        """

    @overload
    def __mul__(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Matrix-vector product of a ColMatrix and a column VectorFunction (``self @ v``).

        Treats ``v`` as a column vector and returns ``M(x) @ v(x)`` as a flat
        VectorFunction.  The output length of ``v`` must equal the number of columns
        of ``self``.

        Parameters
        ----------
        v : VectorFunction
            Column vector whose ``output_rows()`` equals ``self.cols``.

        Returns
        -------
        VectorFunction
            Flat VectorFunction of length ``self.rows`` representing the product.

        Raises
        ------
        ValueError
            If the inner dimension does not match.
        """

    def __rmul__(self, arg: float, /) -> ColMatrix:
        """
        Scalar multiplication with the scalar on the left (``scale * self``).

        Parameters
        ----------
        scale : float
            Scalar factor applied to every element of the matrix function.

        Returns
        -------
        ColMatrix
            A new ColMatrix scaled by ``scale``.
        """

    @overload
    def __add__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> ColMatrix:
        """
        Add a constant matrix offset to a ColMatrix VectorFunction (``self + mshift``).

        Adds the constant numpy matrix ``mshift`` element-wise to the matrix produced
        by ``self`` at every evaluation point.  The dimensions must match exactly.

        Parameters
        ----------
        mshift : array_like, shape (rows, cols)
            Constant matrix offset with the same shape as ``self``.

        Returns
        -------
        ColMatrix
            A new ColMatrix evaluating ``M(x) + mshift``.

        Raises
        ------
        ValueError
            If ``mshift`` does not have the same number of rows and columns as ``self``.
        """

    @overload
    def __add__(self, arg: ColMatrix, /) -> ColMatrix:
        """
        Element-wise sum of two ColMatrix VectorFunctions (``self + other``).

        Computes ``M1(x) + M2(x)`` pointwise.  Both matrix functions must have the
        same shape.

        Parameters
        ----------
        other : ColMatrix
            Right-hand summand with the same number of rows and columns as ``self``.

        Returns
        -------
        ColMatrix
            A new ColMatrix evaluating the element-wise sum.

        Raises
        ------
        ValueError
            If the two matrix functions do not have the same shape.
        """

    def __radd__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> ColMatrix:
        """
        Add a constant matrix offset with the constant on the left (``mshift + self``).

        Commutative companion to ``__add__``; produces the same result as
        ``self + mshift``.

        Parameters
        ----------
        mshift : array_like, shape (rows, cols)
            Constant matrix offset with the same shape as ``self``.

        Returns
        -------
        ColMatrix
            A new ColMatrix evaluating ``mshift + M(x)``.

        Raises
        ------
        ValueError
            If ``mshift`` does not have the same number of rows and columns as ``self``.
        """

    __array_ufunc__: None = None

    def inverse(self) -> ColMatrix:
        """
        Compute the symbolic matrix inverse of this square ColMatrix VectorFunction.

        Returns a new ColMatrix VectorFunction that, for any input ``x``, evaluates
        to the inverse of the matrix ``M(x)``.  Dispatches to analytically-derived
        2×2 or 3×3 inverse kernels for small matrices and to a general LU-based
        kernel otherwise.

        Returns
        -------
        ColMatrix
            A ColMatrix representing ``M(x)^{-1}``.

        Raises
        ------
        ValueError
            If the matrix is not square (``rows != cols``).
        """

    def transpose(self) -> RowMatrix:
        """
        Transpose this ColMatrix VectorFunction.

        Returns a RowMatrix VectorFunction whose (i, j) entry equals the (j, i) entry
        of ``self``.  Construction is O(1): the transposed view shares the same backing
        VectorFunction object (reference-shared, not copied).  No output data is
        copied or recomputed during evaluation; the flat output buffer is simply
        reinterpreted with swapped row/column counts.

        Returns
        -------
        RowMatrix
            A row-major matrix view of the transposed function.
        """

    def vf(self) -> VectorFunction:
        """
        Extract the underlying flat VectorFunction from this ColMatrix.

        Returns the backing VectorFunction whose output is the column-major
        flattening of the matrix (column-by-column, length ``rows * cols``).
        Use this to pass a matrix VectorFunction to APIs that expect a plain
        VectorFunction.

        Returns
        -------
        VectorFunction
            The flat VectorFunction underlying this ColMatrix.
        """

class RowMatrix:
    """
    Row-major matrix view of a VectorFunction.

    A ``RowMatrix`` wraps a flat VectorFunction whose output has length
    ``rows * cols`` and interprets that output as a row-major (C-order) matrix.
    Like ``ColMatrix``, it supports symbolic matrix algebra — products, sums,
    scalar multiplication, transpose, and inverse — all expressed as new
    VectorFunctions so that derivatives propagate through them automatically.

    Construct a ``RowMatrix`` either from a backing VectorFunction with explicit
    shape, or from a Python list of same-length VectorFunctions that are stacked
    into rows.

    See Also
    --------
    ColMatrix : Column-major counterpart.
    matmul : Free-function matrix product for mixed operand types.
    """

    @overload
    def __init__(self, arg0: VectorFunction, arg1: int, arg2: int, /) -> None:
        """
        Construct a RowMatrix by reshaping a VectorFunction output into a row-major matrix.

        Parameters
        ----------
        f : VectorFunction
            Backing VectorFunction whose flat output (length ``rows * cols``) is reinterpreted
            as a row-major matrix.
        rows : int
            Number of rows in the matrix view.  Must satisfy ``rows * cols == f.output_rows()``.
        cols : int
            Number of columns in the matrix view.

        Raises
        ------
        ValueError
            If ``rows`` or ``cols`` is non-positive, or if ``rows * cols`` does not equal
            ``f.output_rows()``.
        """

    @overload
    def __init__(self, arg: Sequence[VectorFunction], /) -> None:
        """
        Construct a RowMatrix from a list of row VectorFunctions.

        Each element of ``rowfuns`` becomes one row of the resulting matrix.
        All functions must share the same output length (which becomes the number
        of columns); the number of rows equals ``len(rowfuns)``.

        Parameters
        ----------
        rowfuns : list of VectorFunction
            Row vectors to stack.  Every element must have the same
            ``output_rows()``.

        Raises
        ------
        ValueError
            If ``rowfuns`` is empty or if the functions do not all have the same
            output length.
        """

    @overload
    def __mul__(self, arg: ColMatrix, /) -> ColMatrix:
        """
        Matrix product of this RowMatrix with a ColMatrix (``self @ other``).

        Computes ``M1(x) @ M2(x)`` where both factors are evaluated at the same input
        ``x``.  The inner dimension of ``M1`` must equal the outer dimension of ``M2``.
        Other right-hand-factor types (``RowMatrix``, plain ``VectorFunction``, ``float``)
        are handled by separate overloads.

        Parameters
        ----------
        other : ColMatrix
            Right-hand ColMatrix factor.

        Returns
        -------
        ColMatrix
            The matrix product returned as a column-major matrix.

        Raises
        ------
        ValueError
            If the inner dimensions of the two matrix functions do not match.
        """

    @overload
    def __mul__(self, arg: RowMatrix, /) -> ColMatrix:
        """
        Matrix product of two RowMatrix VectorFunctions (``self @ other``).

        Overload of ``__mul__`` when both factors are ``RowMatrix`` instances.
        The result is always returned as a ``ColMatrix``.
        See the primary ``__mul__`` overload for full documentation.
        """

    @overload
    def __mul__(self, arg: float, /) -> RowMatrix:
        """
        Scalar multiplication (``self * scale``).

        Parameters
        ----------
        scale : float
            Scalar factor applied to every element of the matrix function.

        Returns
        -------
        RowMatrix
            A new RowMatrix scaled by ``scale``.
        """

    @overload
    def __mul__(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Matrix-vector product of a RowMatrix and a column VectorFunction (``self @ v``).

        Treats ``v`` as a column vector and returns ``M(x) @ v(x)`` as a flat
        VectorFunction.  The output length of ``v`` must equal the number of columns
        of ``self``.

        Parameters
        ----------
        v : VectorFunction
            Column vector whose ``output_rows()`` equals ``self.cols``.

        Returns
        -------
        VectorFunction
            Flat VectorFunction of length ``self.rows`` representing the product.

        Raises
        ------
        ValueError
            If the inner dimension does not match.
        """

    def __rmul__(self, arg: float, /) -> RowMatrix:
        """
        Scalar multiplication with the scalar on the left (``scale * self``).

        Parameters
        ----------
        scale : float
            Scalar factor applied to every element of the matrix function.

        Returns
        -------
        RowMatrix
            A new RowMatrix scaled by ``scale``.
        """

    @overload
    def __add__(self, arg: RowMatrix, /) -> RowMatrix:
        """
        Element-wise sum of two RowMatrix VectorFunctions (``self + other``).

        Computes ``M1(x) + M2(x)`` pointwise.  Both matrix functions must have the
        same shape.  Adding a constant numpy matrix offset is handled by a separate
        overload.

        Parameters
        ----------
        other : RowMatrix
            Right-hand RowMatrix summand with the same number of rows and columns as
            ``self``.

        Returns
        -------
        RowMatrix
            A new RowMatrix evaluating the element-wise sum.

        Raises
        ------
        ValueError
            If the two matrix functions do not have the same shape.
        """

    @overload
    def __add__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> RowMatrix:
        """
        Add a constant matrix offset to a RowMatrix VectorFunction (``self + mshift``).

        Adds the constant numpy matrix ``mshift`` element-wise to the matrix produced
        by ``self`` at every evaluation point.  The dimensions must match exactly.
        The constant is converted to row-major storage internally.

        Parameters
        ----------
        mshift : array_like, shape (rows, cols)
            Constant matrix offset with the same shape as ``self``.

        Returns
        -------
        RowMatrix
            A new RowMatrix evaluating ``M(x) + mshift``.

        Raises
        ------
        ValueError
            If ``mshift`` does not have the same number of rows and columns as ``self``.
        """

    __array_ufunc__: None = None

    def __radd__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> RowMatrix:
        """
        Add a constant matrix offset with the constant on the left (``mshift + self``).

        Commutative companion to ``__add__``; produces the same result as
        ``self + mshift``.

        Parameters
        ----------
        mshift : array_like, shape (rows, cols)
            Constant matrix offset with the same shape as ``self``.

        Returns
        -------
        RowMatrix
            A new RowMatrix evaluating ``mshift + M(x)``.

        Raises
        ------
        ValueError
            If ``mshift`` does not have the same number of rows and columns as ``self``.
        """

    def inverse(self) -> RowMatrix:
        """
        Compute the symbolic matrix inverse of this square RowMatrix VectorFunction.

        Returns a new RowMatrix VectorFunction that, for any input ``x``, evaluates
        to the inverse of the matrix ``M(x)``.  Dispatches to analytically-derived
        2×2 or 3×3 inverse kernels for small matrices and to a general LU-based
        kernel otherwise.

        Returns
        -------
        RowMatrix
            A RowMatrix representing ``M(x)^{-1}``.

        Raises
        ------
        ValueError
            If the matrix is not square (``rows != cols``).
        """

    def transpose(self) -> ColMatrix:
        """
        Transpose this RowMatrix VectorFunction.

        Returns a ColMatrix VectorFunction whose (i, j) entry equals the (j, i) entry
        of ``self``.  Construction is O(1): the transposed view shares the same backing
        VectorFunction object (reference-shared, not copied).  No output data is
        copied or recomputed during evaluation; the flat output buffer is simply
        reinterpreted with swapped row/column counts.

        Returns
        -------
        ColMatrix
            A column-major matrix view of the transposed function.
        """

    def vf(self) -> VectorFunction:
        """
        Extract the underlying flat VectorFunction from this RowMatrix.

        Returns the backing VectorFunction whose output is the row-major
        flattening of the matrix (row-by-row, length ``rows * cols``).
        Use this to pass a matrix VectorFunction to APIs that expect a plain
        VectorFunction.

        Returns
        -------
        VectorFunction
            The flat VectorFunction underlying this RowMatrix.
        """

@overload
def matmul(arg0: ColMatrix, arg1: ColMatrix, /) -> ColMatrix:
    """
    Matrix product of two matrix-valued VectorFunctions or a matrix and a vector/constant.

    Computes ``M1(x) @ M2(x)`` (or ``M(x) @ v`` when the right operand is a
    VectorFunction column vector or a constant numpy vector).  All matrix and
    vector operands are evaluated at the same input ``x``.  Multiple overloads
    accept ``ColMatrix``, ``RowMatrix``, ``VectorFunction``, ``numpy.ndarray``
    (constant vector), and small constant matrices (2×2, 3×3, or general
    ``numpy.ndarray``).

    Parameters
    ----------
    m1 : ColMatrix or RowMatrix or array_like
        Left-hand factor.  May be a matrix VectorFunction or a constant numpy
        matrix (2×2, 3×3, or general ``ndarray``).
    m2 : ColMatrix or RowMatrix or VectorFunction or array_like
        Right-hand factor.  May be a matrix VectorFunction, a plain VectorFunction
        (treated as a column vector), or a constant numpy vector.

    Returns
    -------
    ColMatrix or VectorFunction
        Column-major matrix VectorFunction, or a flat VectorFunction when the
        right operand is a column vector.

    Raises
    ------
    ValueError
        If the inner dimensions of the two factors do not match.
    """

@overload
def matmul(arg0: ColMatrix, arg1: RowMatrix, /) -> ColMatrix:
    """
    Overload of ``matmul`` for a ColMatrix left factor and a RowMatrix right factor.

    See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: ColMatrix, arg1: VectorFunction, /) -> VectorFunction:
    """
    Overload of ``matmul`` for a ColMatrix left factor and a VectorFunction column vector.

    Treats ``m2`` as a column vector and returns the matrix-vector product as a
    flat VectorFunction.  See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: ColMatrix, arg1: numpy.ndarray, /) -> VectorFunction:
    """
    Overload of ``matmul`` for a ColMatrix left factor and a constant numpy vector.

    Multiplies the matrix VectorFunction ``m1`` by the fixed column vector ``v``,
    returning a flat VectorFunction of length ``m1.rows``.
    See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: RowMatrix, arg1: ColMatrix, /) -> ColMatrix:
    """
    Overload of ``matmul`` for a RowMatrix left factor and a ColMatrix right factor.

    See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: RowMatrix, arg1: RowMatrix, /) -> ColMatrix:
    """
    Overload of ``matmul`` for two RowMatrix factors.

    See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: RowMatrix, arg1: VectorFunction, /) -> VectorFunction:
    """
    Overload of ``matmul`` for a RowMatrix left factor and a VectorFunction column vector.

    Treats ``m2`` as a column vector and returns the matrix-vector product as a
    flat VectorFunction.  See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: RowMatrix, arg1: numpy.ndarray, /) -> VectorFunction:
    """
    Overload of ``matmul`` for a RowMatrix left factor and a constant numpy vector.

    Multiplies the matrix VectorFunction ``m1`` by the fixed column vector ``v``,
    returning a flat VectorFunction of length ``m1.rows``.
    See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: Annotated[NDArray[numpy.float64], dict(shape=(2, 2), order='F')], arg1: VectorFunction, /) -> VectorFunction:
    """
    Overload of ``matmul`` for a constant 2×2 matrix and a VectorFunction column vector.

    Scales the VectorFunction ``vec`` by the fixed 2×2 matrix ``mat``, returning a
    flat VectorFunction of length 2.  Uses a specialized 2×2 kernel for efficiency.
    See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')], arg1: VectorFunction, /) -> VectorFunction:
    """
    Overload of ``matmul`` for a constant 3×3 matrix and a VectorFunction column vector.

    Scales the VectorFunction ``vec`` by the fixed 3×3 matrix ``mat``, returning a
    flat VectorFunction of length 3.  Uses a specialized 3×3 kernel for efficiency.
    See the primary ``matmul`` overload for full documentation.
    """

@overload
def matmul(arg0: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], arg1: VectorFunction, /) -> VectorFunction:
    """
    Overload of ``matmul`` for an arbitrary constant matrix and a VectorFunction column vector.

    Scales the VectorFunction ``vec`` by the fixed matrix ``mat`` using a general
    kernel.  The number of columns of ``mat`` must equal ``vec.output_rows()``.
    See the primary ``matmul`` overload for full documentation.
    """

class Conditional:
    """
    Type-erased boolean predicate over an input vector.

    A ``Conditional`` wraps any comparison or logical combination of
    scalar-valued VectorFunctions behind a single uniform interface. It is the
    predicate type that drives :func:`ifelse` branching.

    Leaf predicates are formed by applying comparison operators (``<``, ``<=``,
    ``>``, ``>=``) to pairs of scalar VectorFunctions; these operators
    return ``Conditional`` objects. Leaf predicates can then be combined with
    ``&`` (:meth:`__and__`) and ``|`` (:meth:`__or__`) to build compound
    logical expressions. The resulting ``Conditional`` can be passed to
    :func:`ifelse` to select between two VectorFunction branches.

    ``compute(x)`` evaluates the predicate at the input vector *x* and returns
    the resulting ``bool``. Derivatives are not defined on ``Conditional``
    itself — they are computed from whichever branch is active in the enclosing
    :func:`ifelse` or ``min`` / ``max`` expression.
    """

    def compute(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> bool:
        """
        Evaluate the conditional predicate at a given input vector.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate the predicate.

        Returns
        -------
        bool
            Result of the comparison or logical combination at *x*.
        """

    def __and__(self, arg: Conditional, /) -> Conditional:
        """
        Combine two conditionals with logical AND.

        Returns a new Conditional that is true only when both *self* and *other* are
        true at the same input point.

        Parameters
        ----------
        other : Conditional
            Right-hand conditional to combine.

        Returns
        -------
        Conditional
            Logical AND of the two predicates.
        """

    def __or__(self, arg: Conditional, /) -> Conditional:
        """
        Combine two conditionals with logical OR.

        Returns a new Conditional that is true when at least one of *self* or *other*
        is true at the same input point.

        Parameters
        ----------
        other : Conditional
            Right-hand conditional to combine.

        Returns
        -------
        Conditional
            Logical OR of the two predicates.
        """

    @overload
    def ifelse(self, arg0: ScalarFunction, arg1: ScalarFunction, /) -> ScalarFunction:
        """
        Build a scalar VectorFunction that selects between two scalar branches based on a predicate.

        At each evaluation point the predicate *test* is checked. If true, *tf* is
        evaluated; otherwise *ff* is evaluated. Derivatives are taken from whichever
        branch is active, so the result is only piecewise differentiable across the
        switching boundary.

        All three arguments must share the same input dimension, and both branch
        functions must be scalar-valued (output rows = 1).

        Parameters
        ----------
        test : Conditional
            Predicate that selects the active branch.
        tf : GenericFunction (scalar)
            Function evaluated when *test* is true.
        ff : GenericFunction (scalar)
            Function evaluated when *test* is false.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning ``tf(x)`` when ``test(x)`` is true, else ``ff(x)``.

        Raises
        ------
        ValueError
            If *test*, *tf*, and *ff* do not share the same input dimension, or if
            *tf* and *ff* have different output dimensions.
        """

    @overload
    def ifelse(self, arg0: float, arg1: ScalarFunction, /) -> ScalarFunction:
        """
        Build a scalar VectorFunction that returns a constant when the predicate is true, else a scalar function.

        Parameters
        ----------
        test : Conditional
            Predicate that selects the active branch.
        tfv : float
            Constant value returned when *test* is true.
        ff : GenericFunction (scalar)
            Function evaluated when *test* is false.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning *tfv* when ``test(x)`` is true, else ``ff(x)``.
        """

    @overload
    def ifelse(self, arg0: ScalarFunction, arg1: float, /) -> ScalarFunction:
        """
        Build a scalar VectorFunction that returns a scalar function when the predicate is true, else a constant.

        Parameters
        ----------
        test : Conditional
            Predicate that selects the active branch.
        tf : GenericFunction (scalar)
            Function evaluated when *test* is true.
        ffv : float
            Constant value returned when *test* is false.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning ``tf(x)`` when ``test(x)`` is true, else *ffv*.
        """

    @overload
    def ifelse(self, arg0: float, arg1: float, /) -> ScalarFunction:
        """
        Build a scalar VectorFunction that selects between two scalar constants based on a predicate.

        Parameters
        ----------
        test : Conditional
            Predicate that selects the active branch.
        tfv : float
            Constant value returned when *test* is true.
        ffv : float
            Constant value returned when *test* is false.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning *tfv* when ``test(x)`` is true, else *ffv*.
        """

    @overload
    def ifelse(self, arg0: VectorFunction, arg1: VectorFunction, /) -> VectorFunction:
        """
        Build a vector-valued VectorFunction that selects between two vector-valued branches based on a predicate.

        At each evaluation point the predicate *test* is checked. If true, *tf* is
        evaluated; otherwise *ff* is evaluated. Both branch functions must share the
        same input and output dimensions as one another, and their input dimension
        must match that of *test*. Derivatives are taken from the active branch
        only, so the result is piecewise differentiable across the switching
        boundary.

        Parameters
        ----------
        test : Conditional
            Predicate that selects the active branch.
        tf : GenericFunction
            Vector-valued function evaluated when *test* is true.
        ff : GenericFunction
            Vector-valued function evaluated when *test* is false.

        Returns
        -------
        GenericFunction
            Vector function returning ``tf(x)`` when ``test(x)`` is true, else ``ff(x)``.

        Raises
        ------
        ValueError
            If *test*, *tf*, and *ff* do not share the same input dimension, or if
            *tf* and *ff* have different output dimensions.
        """

    @overload
    def ifelse(self, arg0: numpy.ndarray, arg1: VectorFunction, /) -> VectorFunction:
        """
        Build a vector-valued VectorFunction that returns a constant vector when the predicate is true, else a vector-valued function.

        Parameters
        ----------
        test : Conditional
            Predicate that selects the active branch.
        v : array_like
            Constant vector returned when *test* is true; must have the same length
            as ``ff.output_rows()``.
        ff : GenericFunction
            Vector-valued function evaluated when *test* is false.

        Returns
        -------
        GenericFunction
            Vector function returning *v* when ``test(x)`` is true, else ``ff(x)``.
        """

    @overload
    def ifelse(self, arg0: VectorFunction, arg1: numpy.ndarray, /) -> VectorFunction:
        """
        Build a vector-valued VectorFunction that returns a vector-valued function when the predicate is true, else a constant vector.

        Parameters
        ----------
        test : Conditional
            Predicate that selects the active branch.
        tf : GenericFunction
            Vector-valued function evaluated when *test* is true.
        v : array_like
            Constant vector returned when *test* is false; must have the same length
            as ``tf.output_rows()``.

        Returns
        -------
        GenericFunction
            Vector function returning ``tf(x)`` when ``test(x)`` is true, else *v*.
        """

    @overload
    def ifelse(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> VectorFunction:
        """
        Build a vector-valued VectorFunction that selects between two constant vectors based on a predicate.

        Parameters
        ----------
        test : Conditional
            Predicate that selects the active branch.
        v1 : array_like
            Constant vector returned when *test* is true.
        v2 : array_like
            Constant vector returned when *test* is false; must have the same length
            as *v1*.

        Returns
        -------
        GenericFunction
            Vector function returning *v1* when ``test(x)`` is true, else *v2*.
        """

class Comparative:
    """
    Type-erased predicate used as the branch selector for min/max VectorFunctions.

    A ``Comparative`` wraps a pairwise min or max comparison between two
    scalar-valued (or vector-valued) VectorFunctions behind a uniform boolean
    interface. It is produced internally by the ``min`` and ``max``
    methods on ``Comparative`` and is distinct from :class:`Conditional` so
    that the two types are separately registered in the Python module.

    At each evaluation point ``compute(x)`` returns the boolean result of the
    underlying comparison (i.e. which operand is selected). The associated
    ``min`` / ``max`` combinators return a new VectorFunction (not a
    ``Comparative``) whose value and derivatives track the selected branch.

    The ``Comparative`` object itself is rarely needed directly — it surfaces
    only when you want to inspect the switching predicate independently of the
    enclosing min/max function.
    """

    def compute(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> bool:
        """
        Evaluate the comparative function at a given input vector.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate the min/max function.

        Returns
        -------
        bool
            ``True`` if the first operand (*f1*) is selected (i.e. ``f1(x)`` wins
            the min/max comparison); ``False`` if the second operand (*f2*) is
            selected.
        """

    @overload
    def max(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Return a VectorFunction that evaluates the pointwise maximum of two scalar functions.

        Both operands must be scalar-valued (output rows = 1) and share the same input
        dimension. At each evaluation point the result equals ``max(f1(x), f2(x))``.
        The derivative is taken from whichever branch is selected, so the result is
        piecewise differentiable across the switching boundary.

        Parameters
        ----------
        f1 : GenericFunction (scalar)
            First scalar operand.
        f2 : GenericFunction (scalar)
            Second scalar operand.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning the pointwise maximum.

        Raises
        ------
        ValueError
            If *f1* and *f2* have different input dimensions.
        """

    @overload
    def max(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Return a VectorFunction evaluating the pointwise maximum of a scalar constant and a scalar function.

        Parameters
        ----------
        v1 : float
            Constant scalar to compare against.
        f2 : GenericFunction (scalar)
            Scalar function operand.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning ``max(v1, f2(x))``.
        """

    @overload
    def max(self, arg: float, /) -> ScalarFunction:
        """
        Return a VectorFunction evaluating the pointwise maximum of a scalar function and a scalar constant.

        Parameters
        ----------
        f1 : GenericFunction (scalar)
            Scalar function operand.
        v2 : float
            Constant scalar to compare against.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning ``max(f1(x), v2)``.
        """

    @overload
    def max(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Return a VectorFunction evaluating the element-wise maximum of two vector-valued functions.

        Both operands must have the same input dimension and the same output
        dimension. The element-wise maximum is taken independently for each output
        component. Derivatives track whichever branch is selected per component.

        Parameters
        ----------
        f1 : GenericFunction
            First vector-valued operand.
        f2 : GenericFunction
            Second vector-valued operand.

        Returns
        -------
        GenericFunction
            Vector function returning the element-wise maximum of *f1* and *f2*.

        Raises
        ------
        ValueError
            If *f1* and *f2* have different input or output dimensions.
        """

    @overload
    def max(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Return a VectorFunction evaluating the element-wise maximum of a constant vector and a vector-valued function.

        Parameters
        ----------
        v1 : array_like
            Constant vector; must have the same length as ``f2.output_rows()``.
        f2 : GenericFunction
            Vector-valued function operand.

        Returns
        -------
        GenericFunction
            Vector function returning the element-wise maximum of *v1* and *f2(x)*.
        """

    @overload
    def max(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Return a VectorFunction evaluating the element-wise maximum of a vector-valued function and a constant vector.

        Parameters
        ----------
        f1 : GenericFunction
            Vector-valued function operand.
        v2 : array_like
            Constant vector; must have the same length as ``f1.output_rows()``.

        Returns
        -------
        GenericFunction
            Vector function returning the element-wise maximum of *f1(x)* and *v2*.
        """

    @overload
    def min(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Return a VectorFunction that evaluates the pointwise minimum of two scalar functions.

        Both operands must be scalar-valued (output rows = 1) and share the same input
        dimension. At each evaluation point the result equals ``min(f1(x), f2(x))``.
        The derivative is taken from whichever branch is selected, so the result is
        piecewise differentiable across the switching boundary.

        Parameters
        ----------
        f1 : GenericFunction (scalar)
            First scalar operand.
        f2 : GenericFunction (scalar)
            Second scalar operand.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning the pointwise minimum.

        Raises
        ------
        ValueError
            If *f1* and *f2* have different input dimensions.
        """

    @overload
    def min(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Return a VectorFunction evaluating the pointwise minimum of a scalar constant and a scalar function.

        Parameters
        ----------
        v1 : float
            Constant scalar to compare against.
        f2 : GenericFunction (scalar)
            Scalar function operand.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning ``min(v1, f2(x))``.
        """

    @overload
    def min(self, arg: float, /) -> ScalarFunction:
        """
        Return a VectorFunction evaluating the pointwise minimum of a scalar function and a scalar constant.

        Parameters
        ----------
        f1 : GenericFunction (scalar)
            Scalar function operand.
        v2 : float
            Constant scalar to compare against.

        Returns
        -------
        GenericFunction (scalar)
            Scalar function returning ``min(f1(x), v2)``.
        """

    @overload
    def min(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Return a VectorFunction evaluating the element-wise minimum of two vector-valued functions.

        Both operands must have the same input dimension and the same output
        dimension. The element-wise minimum is taken independently for each output
        component. Derivatives track whichever branch is selected per component.

        Parameters
        ----------
        f1 : GenericFunction
            First vector-valued operand.
        f2 : GenericFunction
            Second vector-valued operand.

        Returns
        -------
        GenericFunction
            Vector function returning the element-wise minimum of *f1* and *f2*.

        Raises
        ------
        ValueError
            If *f1* and *f2* have different input or output dimensions.
        """

    @overload
    def min(self, arg: VectorFunction, /) -> VectorFunction:
        """
        Return a VectorFunction evaluating the element-wise minimum of a constant vector and a vector-valued function.

        Parameters
        ----------
        v1 : array_like
            Constant vector; must have the same length as ``f2.output_rows()``.
        f2 : GenericFunction
            Vector-valued function operand.

        Returns
        -------
        GenericFunction
            Vector function returning the element-wise minimum of *v1* and *f2(x)*.
        """

    @overload
    def min(self, arg: numpy.ndarray, /) -> VectorFunction:
        """
        Return a VectorFunction evaluating the element-wise minimum of a vector-valued function and a constant vector.

        Parameters
        ----------
        f1 : GenericFunction
            Vector-valued function operand.
        v2 : array_like
            Constant vector; must have the same length as ``f1.output_rows()``.

        Returns
        -------
        GenericFunction
            Vector function returning the element-wise minimum of *f1(x)* and *v2*.
        """

class PyVectorFunction:
    """
    VectorFunction defined by an arbitrary Python callable, with finite-difference derivatives.

    Wrap any Python function ``f(x, *args)`` as a first-class VectorFunction that can
    be used anywhere in the Tycho optimal-control and trajectory-design API.  The
    callable receives a 1-D NumPy array ``x`` of length ``i_rows`` (plus any extra
    positional arguments supplied via ``args``) and must return a 1-D array-like of
    length ``o_rows``.  Jacobians and Hessians are approximated automatically using
    forward finite differences.

    Because the Python GIL must be acquired on every evaluation, ``PyVectorFunction``
    cannot execute in parallel across PSIOPT solver threads — calls are serialized by
    the GIL.  For parallel-safe evaluation, use ``NumbaVectorFunction`` instead.

    Parameters
    ----------
    i_rows : int
        Number of input components (length of the input vector passed to ``func``).
    o_rows : int
        Number of output components (length of the array returned by ``func``).
        Omitted for the scalar-output variant; fixed to 1.
    func : callable
        Python function ``f(x, *args) -> array_like`` implementing the VectorFunction.
    jstepsize : float, optional
        Forward finite-difference step size for Jacobian approximation (default 1e-6).
    hstepsize : float, optional
        Forward finite-difference step size for Hessian approximation (default 1e-4).
    args : tuple, optional
        Extra positional arguments forwarded to ``func`` after the input vector.
    """

    def __init__(self, i_rows: int, o_rows: int, func: object, jstepsize: float = 1e-06, hstepsize: float = 0.0001, args: tuple = ()) -> None:
        """
        Wrap a Python callable as a VectorFunction using finite-difference derivatives.

        The callable is invoked with a 1-D NumPy array of length ``i_rows`` (plus any
        extra positional arguments in ``args``) and must return a 1-D array of length
        ``o_rows``.  Jacobians and Hessians are approximated by forward finite differences
        with step sizes ``jstepsize`` and ``hstepsize`` respectively.

        Note: ``PyVectorFunction`` always acquires the Python GIL and cannot execute in
        parallel.  Use ``NumbaVectorFunction`` for thread-safe (GIL-free) evaluation.

        Parameters
        ----------
        i_rows : int
            Number of input components (input vector length).
        o_rows : int
            Number of output components (output vector length).
        func : callable
            Python function ``f(x, *args) -> array_like`` implementing the VectorFunction.
        jstepsize : float, optional
            Finite-difference step size used for Jacobian approximation (default 1e-6).
        hstepsize : float, optional
            Finite-difference step size used for Hessian approximation (default 1e-4).
        args : tuple, optional
            Extra positional arguments forwarded to ``func`` after the input vector.
        """

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

    def vf(self) -> VectorFunction:
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

    @property
    def thread_safe(self) -> bool:
        """
        Whether this function is safe to call from multiple threads simultaneously.

        Always ``False`` for ``PyVectorFunction`` — every evaluation acquires the Python
        GIL, so parallel calls are serialized.  Attempting to set this to ``True`` raises
        ``ValueError``.  Use ``NumbaVectorFunction`` for genuinely thread-safe evaluation.
        """

    @thread_safe.setter
    def thread_safe(self, arg: bool, /) -> None: ...

class PyScalarFunction:
    """
    VectorFunction defined by an arbitrary Python callable, with finite-difference derivatives.

    Wrap any Python function ``f(x, *args)`` as a first-class VectorFunction that can
    be used anywhere in the Tycho optimal-control and trajectory-design API.  The
    callable receives a 1-D NumPy array ``x`` of length ``i_rows`` (plus any extra
    positional arguments supplied via ``args``) and must return a 1-D array-like of
    length ``o_rows``.  Jacobians and Hessians are approximated automatically using
    forward finite differences.

    Because the Python GIL must be acquired on every evaluation, ``PyVectorFunction``
    cannot execute in parallel across PSIOPT solver threads — calls are serialized by
    the GIL.  For parallel-safe evaluation, use ``NumbaVectorFunction`` instead.

    Parameters
    ----------
    i_rows : int
        Number of input components (length of the input vector passed to ``func``).
    o_rows : int
        Number of output components (length of the array returned by ``func``).
        Omitted for the scalar-output variant; fixed to 1.
    func : callable
        Python function ``f(x, *args) -> array_like`` implementing the VectorFunction.
    jstepsize : float, optional
        Forward finite-difference step size for Jacobian approximation (default 1e-6).
    hstepsize : float, optional
        Forward finite-difference step size for Hessian approximation (default 1e-4).
    args : tuple, optional
        Extra positional arguments forwarded to ``func`` after the input vector.
    """

    def __init__(self, i_rows: int, func: object, jstepsize: float = 1e-06, hstepsize: float = 0.0001, args: tuple = ()) -> None:
        """
        Scalar-output variant: ``o_rows`` is fixed to 1 and may be omitted.

        Parameters
        ----------
        i_rows : int
            Number of input components (length of the input vector passed to ``func``).
        func : callable
            Python function ``f(x, *args) -> float`` implementing the scalar VectorFunction.
        jstepsize : float, optional
            Forward finite-difference step size for Jacobian approximation (default 1e-6).
        hstepsize : float, optional
            Forward finite-difference step size for Hessian approximation (default 1e-4).
        args : tuple, optional
            Extra positional arguments forwarded to ``func`` after the input vector.
        """

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

    def vf(self) -> VectorFunction:
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

    def sf(self) -> ScalarFunction:
        """
        Type-erase this scalar-output expression into a scalar GenericFunction.

        Available only on VectorFunctions with a single output row.  Equivalent to
        :meth:`vf` but the returned wrapper carries the compile-time knowledge that
        the output dimension is 1, which enables scalar-specific operations.

        Returns
        -------
        ScalarFunction
            A dynamically-typed scalar-output wrapper around this function.
        """

    @property
    def thread_safe(self) -> bool:
        """
        Whether this function is safe to call from multiple threads simultaneously.

        Always ``False`` for ``PyVectorFunction`` — every evaluation acquires the Python
        GIL, so parallel calls are serialized.  Attempting to set this to ``True`` raises
        ``ValueError``.  Use ``NumbaVectorFunction`` for genuinely thread-safe evaluation.
        """

    @thread_safe.setter
    def thread_safe(self, arg: bool, /) -> None: ...

class NumbaVectorFunction:
    """
    VectorFunction defined by a Numba JIT-compiled C callback, with finite-difference derivatives.

    Wrap a Numba ``@cfunc``-decorated (or any compatible C-ABI) function as a
    first-class VectorFunction that can be used anywhere in the Tycho
    optimal-control and trajectory-design API.  The function pointer must have
    the C signature ``void f(double* x, double* fx, int i_rows, int o_rows)`` and
    is passed as the integer address returned by, for example, the ``.address``
    attribute of a Numba ``@cfunc`` object.

    Unlike ``PyVectorFunction``, evaluation does NOT hold the Python GIL, so
    ``NumbaVectorFunction`` instances can safely be evaluated in parallel across
    PSIOPT solver threads once ``thread_safe`` is set to ``True``.  Jacobians and
    Hessians are approximated automatically using forward finite differences.

    Parameters
    ----------
    i_rows : int
        Number of input components (length of the ``x`` buffer passed to the callback).
    o_rows : int
        Number of output components (length of the ``fx`` buffer filled by the callback).
        Omitted for the scalar-output (``ORR == 1``) and fully-static variants.
    func : int
        Integer address of a C function with signature
        ``void(double*, double*, int, int)``.
    jstepsize : float, optional
        Forward finite-difference step size for Jacobian approximation (default 1e-6).
    hstepsize : float, optional
        Forward finite-difference step size for Hessian approximation (default 1e-6).
    """

    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: float, arg4: float, /) -> None:
        """
        Wrap a Numba JIT-compiled function pointer as a VectorFunction.

        The function pointer must have the C signature
        ``void f(double* x, double* fx, int i_rows, int o_rows)`` and is passed as an
        integer holding its address (e.g. the ``.address`` attribute of a Numba
        ``@cfunc``-decorated function).  Derivatives are approximated by forward finite
        differences with the given step sizes.  Unlike ``PyVectorFunction``, evaluation
        does not hold the Python GIL and is safe to call from multiple threads in parallel
        once ``thread_safe`` is set to ``True`` (default ``False``).

        Parameters
        ----------
        i_rows : int
            Number of input components (input vector length).
        o_rows : int
            Number of output components (output vector length).
        func : int
            Integer address of a C function with signature
            ``void(double*, double*, int, int)``.
        jstepsize : float
            Finite-difference step size used for Jacobian approximation.
        hstepsize : float
            Finite-difference step size used for Hessian approximation.
        """

    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @property
    def thread_safe(self) -> bool:
        """
        Whether this function is safe to call from multiple threads simultaneously.

        Set to ``True`` to allow PSIOPT solver threads to evaluate this function in
        parallel without acquiring the Python GIL.  The underlying C callback must
        itself be thread-safe (i.e., it must not use shared mutable state).
        Defaults to ``False``.
        """

    @thread_safe.setter
    def thread_safe(self, arg: bool, /) -> None: ...

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

    def vf(self) -> VectorFunction:
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

class NumbaScalarFunction:
    """
    VectorFunction defined by a Numba JIT-compiled C callback, with finite-difference derivatives.

    Wrap a Numba ``@cfunc``-decorated (or any compatible C-ABI) function as a
    first-class VectorFunction that can be used anywhere in the Tycho
    optimal-control and trajectory-design API.  The function pointer must have
    the C signature ``void f(double* x, double* fx, int i_rows, int o_rows)`` and
    is passed as the integer address returned by, for example, the ``.address``
    attribute of a Numba ``@cfunc`` object.

    Unlike ``PyVectorFunction``, evaluation does NOT hold the Python GIL, so
    ``NumbaVectorFunction`` instances can safely be evaluated in parallel across
    PSIOPT solver threads once ``thread_safe`` is set to ``True``.  Jacobians and
    Hessians are approximated automatically using forward finite differences.

    Parameters
    ----------
    i_rows : int
        Number of input components (length of the ``x`` buffer passed to the callback).
    o_rows : int
        Number of output components (length of the ``fx`` buffer filled by the callback).
        Omitted for the scalar-output (``ORR == 1``) and fully-static variants.
    func : int
        Integer address of a C function with signature
        ``void(double*, double*, int, int)``.
    jstepsize : float, optional
        Forward finite-difference step size for Jacobian approximation (default 1e-6).
    hstepsize : float, optional
        Forward finite-difference step size for Hessian approximation (default 1e-6).
    """

    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: float, arg4: float, /) -> None:
        """
        Wrap a Numba JIT-compiled function pointer as a VectorFunction.

        The function pointer must have the C signature
        ``void f(double* x, double* fx, int i_rows, int o_rows)`` and is passed as an
        integer holding its address (e.g. the ``.address`` attribute of a Numba
        ``@cfunc``-decorated function).  Derivatives are approximated by forward finite
        differences with the given step sizes.  Unlike ``PyVectorFunction``, evaluation
        does not hold the Python GIL and is safe to call from multiple threads in parallel
        once ``thread_safe`` is set to ``True`` (default ``False``).

        Parameters
        ----------
        i_rows : int
            Number of input components (input vector length).
        o_rows : int
            Number of output components (output vector length).
        func : int
            Integer address of a C function with signature
            ``void(double*, double*, int, int)``.
        jstepsize : float
            Finite-difference step size used for Jacobian approximation.
        hstepsize : float
            Finite-difference step size used for Hessian approximation.
        """

    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg0: int, arg1: int, arg2: float, arg3: float, /) -> None: ...

    @overload
    def __init__(self, arg0: int, arg1: int, /) -> None: ...

    @property
    def thread_safe(self) -> bool:
        """
        Whether this function is safe to call from multiple threads simultaneously.

        Set to ``True`` to allow PSIOPT solver threads to evaluate this function in
        parallel without acquiring the Python GIL.  The underlying C callback must
        itself be thread-safe (i.e., it must not use shared mutable state).
        Defaults to ``False``.
        """

    @thread_safe.setter
    def thread_safe(self, arg: bool, /) -> None: ...

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

    def vf(self) -> VectorFunction:
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

    def sf(self) -> ScalarFunction:
        """
        Type-erase this scalar-output expression into a scalar GenericFunction.

        Available only on VectorFunctions with a single output row.  Equivalent to
        :meth:`vf` but the returned wrapper carries the compile-time knowledge that
        the output dimension is 1, which enables scalar-specific operations.

        Returns
        -------
        ScalarFunction
            A dynamically-typed scalar-output wrapper around this function.
        """

class ConstantVector:
    """
    A VectorFunction that always returns the same fixed output vector.

    Regardless of the input, ``Constant`` evaluates to a compile-time or
    runtime-specified constant vector. Its Jacobian and Hessian are identically
    zero. Construct one via ``Arguments.constant(v)`` or directly with
    ``ConstantVector(input_rows, v)`` / ``ConstantScalar(input_rows, v)``.
    """

    def __init__(self, arg0: int, arg1: numpy.ndarray, /) -> None:
        """
        Construct a constant VectorFunction with a given input size and fixed output vector.

        Parameters
        ----------
        input_rows : int
            Number of input rows (dimension of the input vector).
        value : array_like, shape (output_rows,)
            Fixed output vector returned for every input. Its length sets the output
            dimension.
        """

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

    def vf(self) -> VectorFunction:
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

class ConstantScalar:
    """
    A VectorFunction that always returns the same fixed output vector.

    Regardless of the input, ``Constant`` evaluates to a compile-time or
    runtime-specified constant vector. Its Jacobian and Hessian are identically
    zero. Construct one via ``Arguments.constant(v)`` or directly with
    ``ConstantVector(input_rows, v)`` / ``ConstantScalar(input_rows, v)``.
    """

    def __init__(self, arg0: int, arg1: numpy.ndarray, /) -> None:
        """
        Construct a constant VectorFunction with a given input size and fixed output vector.

        Parameters
        ----------
        input_rows : int
            Number of input rows (dimension of the input vector).
        value : array_like, shape (output_rows,)
            Fixed output vector returned for every input. Its length sets the output
            dimension.
        """

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

    def vf(self) -> VectorFunction:
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

    def sf(self) -> ScalarFunction:
        """
        Type-erase this scalar-output expression into a scalar GenericFunction.

        Available only on VectorFunctions with a single output row.  Equivalent to
        :meth:`vf` but the returned wrapper carries the compile-time knowledge that
        the output dimension is 1, which enables scalar-specific operations.

        Returns
        -------
        ScalarFunction
            A dynamically-typed scalar-output wrapper around this function.
        """

class IOScaled:
    """
    An input/output scaling wrapper that non-dimensionalizes a VectorFunction.

    ``IOScaled(func, input_scales, output_scales)`` multiplies each input
    component by the corresponding entry of ``input_scales`` before passing it to
    *func*, and multiplies each output component by the corresponding entry of
    ``output_scales`` after evaluation. Jacobian, adjoint gradient, and adjoint
    Hessian are all transformed consistently. Use this to rescale poorly-conditioned
    dynamics or constraint functions before handing them to PSIOPT.
    """

    def __init__(self, arg0: VectorFunction, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None:
        """
        Construct an IO-scaled wrapper around a VectorFunction.

        Each input component is multiplied by the corresponding entry of ``input_scales``
        before being passed to the wrapped function, and each output component is
        multiplied by the corresponding entry of ``output_scales`` after evaluation.
        Jacobian rows are scaled by ``output_scales`` and columns by ``input_scales``;
        the adjoint gradient and Hessian are transformed consistently.

        Parameters
        ----------
        func : VectorFunction
            The function whose input and output are to be scaled.
        input_scales : array_like, shape (input_rows,)
            Per-component scale factors applied to the input before evaluation.
        output_scales : array_like, shape (output_rows,)
            Per-component scale factors applied to the output after evaluation.

        Returns
        -------
        IOScaled
            Wrapped function with scaled input and output.

        Notes
        -----
        The scale-vector lengths are not validated against ``func``'s dimensions;
        supplying ``input_scales`` or ``output_scales`` of the wrong length leads to
        undefined behavior at evaluation time.
        """

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

    def vf(self) -> VectorFunction:
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

class InterpType(enum.Enum):
    Cubic = 0

    Linear = 1

class InterpTable1D:
    """
    Lookup-table VectorFunction that interpolates tabulated data over one independent variable.

    ``InterpTable1D`` stores a set of sample points and associated values and evaluates
    them at arbitrary query points via cubic (Hermite) or linear interpolation.  The
    table can hold either a single scalar channel (``vlen == 1``) or multiple parallel
    channels stacked as rows in a matrix.  Once constructed it can be used as a plain
    callable for immediate numeric evaluation, or composed with a scalar VectorFunction
    (or a scalar ``Segment``) via ``__call__`` to produce a new VectorFunction suitable
    for use in an optimal-control problem.  Derivatives are evaluated analytically.

    Available interpolation kinds: ``"cubic"`` (default, C1 Hermite spline) and
    ``"linear"`` (piecewise linear, C0).
    """

    @overload
    def __init__(self, ts: numpy.ndarray, vs: numpy.ndarray, kind: str = 'cubic') -> None:
        """
        Construct a 1-D interpolation table from a 1-D value array.

        Parameters
        ----------
        ts : array_like, shape (N,)
            Strictly ascending sample coordinates.  Must have at least 5 elements.
        vs : array_like, shape (N,)
            Scalar values at each sample point.
        kind : str, optional
            Interpolation method: ``"cubic"`` (default) or ``"linear"``.
        """

    @overload
    def __init__(self, ts: numpy.ndarray, vs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], axis: int = 0, kind: str = 'cubic') -> None:
        """
        Construct a 1-D interpolation table from a 2-D value matrix.

        Parameters
        ----------
        ts : array_like, shape (N,)
            Strictly ascending sample coordinates.  Must have at least 5 elements.
        vs : array_like, shape (N, C) or (C, N)
            Multi-channel tabulated values.  ``axis`` selects which dimension
            contains the sample axis (``0`` for rows, ``1`` for columns).
        axis : int, optional
            Axis of ``vs`` that corresponds to the sample coordinates.
            ``0`` (default) means rows are samples; ``1`` means columns are samples.
        kind : str, optional
            Interpolation method: ``"cubic"`` (default) or ``"linear"``.
        """

    @overload
    def __init__(self, vts: list[numpy.ndarray], tvar: int = -1, kind: str = 'cubic') -> None:
        """
        Construct a 1-D interpolation table from a list of value-plus-time vectors.

        Each element of ``vts`` is a vector whose components are the channel values
        together with the independent coordinate at that sample.  The coordinate
        component is identified by ``tvar`` and stripped before storing the channels.

        Parameters
        ----------
        vts : list of array_like
            List of length N; each element is a 1-D array of the same length.
            One element per sample, with the independent coordinate embedded at
            position ``tvar``.  All vectors must be the same length (>= 2).
        tvar : int, optional
            Index of the independent-coordinate component within each vector.
            Negative values count from the end (default ``-1`` selects the last
            component).
        kind : str, optional
            Interpolation method: ``"cubic"`` (default) or ``"linear"``.
        """

    @overload
    def __init__(self, ts: numpy.ndarray, vs: numpy.ndarray, kind: InterpType = InterpType.Cubic) -> None: ...

    @overload
    def __init__(self, ts: numpy.ndarray, vs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], axis: int = 0, kind: InterpType = InterpType.Cubic) -> None: ...

    @overload
    def __init__(self, vts: list[numpy.ndarray], tvar: int = -1, kind: InterpType = InterpType.Cubic) -> None: ...

    @overload
    def interp(self, t: float) -> numpy.ndarray:
        """
        Evaluate the table at a single coordinate value.

        Parameters
        ----------
        t : float
            Query coordinate.  Must lie within the table's coordinate range.

        Returns
        -------
        numpy.ndarray, shape (vlen,)
            Interpolated channel values at ``t``.

        Raises
        ------
        ValueError
            If ``t`` is outside the table's coordinate range.
        """

    @overload
    def interp(self, t_vals: numpy.ndarray) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Evaluate the table at multiple coordinate values.

        Parameters
        ----------
        t_vals : array_like, shape (M,)
            Query coordinates.  Each must lie within the table's coordinate range.

        Returns
        -------
        numpy.ndarray, shape (vlen, M)
            Interpolated channel values; column ``i`` corresponds to ``t_vals[i]``.
        """

    @overload
    def __call__(self, arg: float, /) -> numpy.ndarray:
        """
        Evaluate the table numerically — scalar coordinate.

        Equivalent to :py:meth:`interp(t) <InterpTable1D.interp>`.
        """

    @overload
    def __call__(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        Evaluate the table numerically — vector of coordinates.

        Equivalent to :py:meth:`interp(t_vals) <InterpTable1D.interp>`.
        """

    @overload
    def __call__(self, arg: ScalarFunction, /) -> object:
        """
        Compose the table with a scalar VectorFunction to produce a new VectorFunction.

        Parameters
        ----------
        t : VectorFunction (scalar output)
            Scalar-output VectorFunction whose result is used as the query coordinate.

        Returns
        -------
        VectorFunction
            If ``vlen == 1``: scalar VectorFunction evaluating the table at ``t(x)``.
            If ``vlen > 1``: vector VectorFunction (output size = ``vlen``) evaluating
            all channels at ``t(x)``.
        """

    @overload
    def __call__(self, arg: Element, /) -> object:
        """
        Compose the table with a scalar Segment to produce a new VectorFunction.

        Parameters
        ----------
        t : Segment (scalar output)
            Scalar-output ``Segment`` (a single element of a state/control vector)
            whose value is used as the query coordinate.

        Returns
        -------
        VectorFunction
            Scalar or vector VectorFunction as described in the ``GenericFunction``
            overload above, depending on ``vlen``.
        """

    def interp_deriv1(self, t: float) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Evaluate the table and its first derivative at a single coordinate.

        Parameters
        ----------
        t : float
            Query coordinate within the table's range.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(v, dv_dt)`` where both arrays have shape ``(vlen,)``.
            ``v`` is the interpolated value and ``dv_dt`` its derivative with respect
            to ``t``.
        """

    def interp_deriv2(self, t: float) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Evaluate the table, its first derivative, and its second derivative.

        Parameters
        ----------
        t : float
            Query coordinate within the table's range.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(v, dv_dt, d2v_dt2)`` where all arrays have shape ``(vlen,)``.
            For linear interpolation the second derivative is not computed; its
            contents are undefined — do not rely on them being zero.
        """

    def sf(self) -> ScalarFunction:
        """
        Return a scalar VectorFunction that wraps this table.

        The returned VectorFunction takes a single-element input (the coordinate)
        and produces a single-element output (the interpolated value).  Raises
        ``ValueError`` if the table stores more than one channel (``vlen > 1``);
        use :py:meth:`vf` in that case.

        Returns
        -------
        VectorFunction
            Scalar-in, scalar-out VectorFunction backed by this table.
        """

    def vf(self) -> VectorFunction:
        """
        Return a vector VectorFunction that wraps this table.

        The returned VectorFunction takes a single-element input (the coordinate)
        and produces a ``vlen``-element output (all interpolated channels).  Works
        for both scalar (``vlen == 1``) and multi-channel tables.

        Returns
        -------
        VectorFunction
            Scalar-in, ``vlen``-out VectorFunction backed by this table.
        """

class InterpTable2D:
    """
    Lookup-table VectorFunction that interpolates tabulated scalar data over two independent variables.

    ``InterpTable2D`` stores a 2-D grid of scalar samples and evaluates them at
    arbitrary ``(x, y)`` query points via bicubic or bilinear interpolation.  The
    output is always a single scalar.  Once constructed the table can be used as a
    plain callable for immediate numeric evaluation, or composed with two scalar
    VectorFunctions (or a 2-element ``Segment``) via ``__call__`` to produce a new
    scalar VectorFunction suitable for use in an optimal-control problem.  Derivatives
    are evaluated analytically.

    Available interpolation kinds: ``"cubic"`` (default, bicubic) and ``"linear"``
    (bilinear).
    """

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, z: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], kind: str = 'cubic') -> None:
        """
        Construct a 2-D interpolation table.

        Parameters
        ----------
        xs : array_like, shape (Nx,)
            Strictly ascending x-coordinates.  Must have at least 5 elements.
            Correspond to **columns** of ``z``.
        ys : array_like, shape (Ny,)
            Strictly ascending y-coordinates.  Must have at least 5 elements.
            Correspond to **rows** of ``z``.
        z : array_like, shape (Ny, Nx)
            Scalar values on the ``(xs, ys)`` grid in row-major order.
        kind : str, optional
            Interpolation method: ``"cubic"`` (default) or ``"linear"``.
        """

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, z: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], kind: InterpType = InterpType.Cubic) -> None: ...

    @overload
    def interp(self, x: float, y: float) -> float:
        """
        Evaluate the table at a single ``(x, y)`` point.

        Parameters
        ----------
        x : float
            X-coordinate within the table's x range.
        y : float
            Y-coordinate within the table's y range.

        Returns
        -------
        float
            Interpolated scalar value at ``(x, y)``.

        Raises
        ------
        ValueError
            If ``x`` or ``y`` is outside the table's range.
        """

    @overload
    def interp(self, x_vals: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], y_vals: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')]:
        """
        Evaluate the table element-wise over two matrices of coordinates.

        Parameters
        ----------
        x_vals : array_like, shape (M, K)
            X-coordinates of the query points.
        y_vals : array_like, shape (M, K)
            Y-coordinates of the query points; must have the same shape as ``x_vals``.

        Returns
        -------
        numpy.ndarray, shape (M, K)
            Interpolated scalar values at each ``(x_vals[i,j], y_vals[i,j])``.
        """

    def interp_deriv1(self, x: float, y: float) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')]]:
        """
        Evaluate the table and its first partial derivatives at ``(x, y)``.

        Parameters
        ----------
        x : float
            X-coordinate within the table's range.
        y : float
            Y-coordinate within the table's range.

        Returns
        -------
        tuple[float, numpy.ndarray]
            ``(z, dz_dxy)`` where ``dz_dxy`` is a length-2 array containing
            ``[dz/dx, dz/dy]``.
        """

    def interp_deriv2(self, x: float, y: float) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')], Annotated[NDArray[numpy.float64], dict(shape=(2, 2), order='F')]]:
        """
        Evaluate the table, first, and second partial derivatives at ``(x, y)``.

        Parameters
        ----------
        x : float
            X-coordinate within the table's range.
        y : float
            Y-coordinate within the table's range.

        Returns
        -------
        tuple[float, numpy.ndarray, numpy.ndarray]
            ``(z, dz_dxy, d2z_dxy2)`` where ``dz_dxy`` has shape ``(2,)`` containing
            ``[dz/dx, dz/dy]``, and ``d2z_dxy2`` is a ``(2, 2)`` symmetric Hessian
            matrix.  For linear interpolation the diagonal second derivatives
            (d²z/dx², d²z/dy²) are zero, but the off-diagonal cross-derivative
            (d²z/dxdy) is in general non-zero.
        """

    def find_elem(self, vals: numpy.ndarray, v: float) -> int:
        """
        Binary-search helper: find the raw interval index for ``v`` in ``vals``.

        Parameters
        ----------
        vals : array_like, shape (N,)
            Sorted coordinate vector to search (e.g. the internal ``xs_`` or ``ys_``
            array).
        v : float
            Value to locate.

        Returns
        -------
        int
            Raw upper-bound-based index: ``upper_bound(vals, v) - vals.begin() - 1``.
            This may be negative or greater than ``N-2`` when ``v`` is outside the
            sample range; no clamping is applied here.  Clamping to ``[0, N-2]``
            happens later in ``get_xyelems``.
        """

    @overload
    def __call__(self, arg0: float, arg1: float, /) -> float:
        """
        Evaluate the table numerically at a single ``(x, y)`` point.

        Equivalent to :py:meth:`interp(x, y) <InterpTable2D.interp>`.
        """

    @overload
    def __call__(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')]:
        """
        Evaluate the table numerically over matrices of coordinates.

        Equivalent to :py:meth:`interp(x_vals, y_vals) <InterpTable2D.interp>`.
        """

    @overload
    def __call__(self, arg0: ScalarFunction, arg1: ScalarFunction, /) -> ScalarFunction:
        """
        Compose the table with two scalar VectorFunctions to produce a new VectorFunction.

        Parameters
        ----------
        x : VectorFunction (scalar output)
            VectorFunction providing the x-coordinate.
        y : VectorFunction (scalar output)
            VectorFunction providing the y-coordinate.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating ``table(x(v), y(v))``.
        """

    @overload
    def __call__(self, arg0: Element, arg1: Element, /) -> ScalarFunction:
        """
        Compose the table with two scalar Segments to produce a new VectorFunction.

        Parameters
        ----------
        x : Segment (scalar output)
            Scalar ``Segment`` providing the x-coordinate.
        y : Segment (scalar output)
            Scalar ``Segment`` providing the y-coordinate.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at the two segment values.
        """

    @overload
    def __call__(self, arg: Segment2, /) -> ScalarFunction:
        """
        Compose the table with a 2-element Segment to produce a new VectorFunction.

        Parameters
        ----------
        xy : Segment (2-element output)
            Two-element ``Segment`` whose first element is the x-coordinate and
            second element is the y-coordinate.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at ``(xy[0], xy[1])``.
        """

    @overload
    def __call__(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Compose the table with a 2-output VectorFunction to produce a new VectorFunction.

        Parameters
        ----------
        xy : VectorFunction (2-element output)
            Two-output VectorFunction whose first output is the x-coordinate and
            second output is the y-coordinate.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at ``(xy(v)[0], xy(v)[1])``.
        """

    def sf(self) -> ScalarFunction:
        """
        Return a scalar VectorFunction that wraps this table.

        The returned VectorFunction takes a 2-element input ``[x, y]`` and returns
        a single scalar output.  Equivalent to :py:meth:`vf` for 2-D tables.

        Returns
        -------
        VectorFunction
            2-in, 1-out VectorFunction backed by this table.
        """

    def vf(self) -> VectorFunction:
        """
        Return a vector VectorFunction that wraps this table.

        The returned VectorFunction takes a 2-element input ``[x, y]`` and returns
        a single scalar output (output size is dynamic, but 1 at runtime).  Use
        :py:meth:`sf` when a statically-typed 1-output ``GenericFunction<-1,1>``
        is required.

        Returns
        -------
        VectorFunction
            2-in, dynamic-out VectorFunction backed by this table (1 output at runtime).
        """

class InterpTable3D:
    """
    Lookup-table VectorFunction that interpolates tabulated scalar data over three independent variables.

    ``InterpTable3D`` stores a 3-D grid of scalar samples (a rank-3 tensor) and
    evaluates them at arbitrary ``(x, y, z)`` query points via tricubic or trilinear
    interpolation.  The output is always a single scalar.  Once constructed the table
    can be used as a plain callable for immediate numeric evaluation, or composed with
    three scalar VectorFunctions (or a 3-element ``Segment``) via ``__call__`` to
    produce a new scalar VectorFunction suitable for use in an optimal-control problem.
    Derivatives are evaluated analytically.

    Available interpolation kinds: ``"cubic"`` (default, tricubic) and ``"linear"``
    (trilinear).  The ``cache`` option pre-computes and stores the per-cell polynomial
    coefficients to accelerate repeated evaluations at the cost of additional memory.
    """

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, zs: numpy.ndarray, fs: numpy.ndarray, kind: str = 'cubic', cache: bool = False) -> None:
        """
        Construct a 3-D interpolation table.

        Parameters
        ----------
        xs : array_like, shape (Nx,)
            Strictly ascending x-coordinates.  Must have at least 5 elements.
            Correspond to the first dimension of ``fs``.
        ys : array_like, shape (Ny,)
            Strictly ascending y-coordinates.  Must have at least 5 elements.
            Correspond to the second dimension of ``fs``.
        zs : array_like, shape (Nz,)
            Strictly ascending z-coordinates.  Must have at least 5 elements.
            Correspond to the third dimension of ``fs``.
        fs : array_like, shape (Nx, Ny, Nz)
            Scalar values on the ``(xs, ys, zs)`` grid in NumPy meshgrid ``ij``
            format (first index = x, second = y, third = z).
        kind : str, optional
            Interpolation method: ``"cubic"`` (default) or ``"linear"``.
        cache : bool, optional
            If ``True``, pre-compute and cache all per-cell polynomial coefficients
            at construction time.  Speeds up repeated queries; uses more memory.
            Default is ``False``.
        """

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, zs: numpy.ndarray, fs: numpy.ndarray, kind: InterpType = InterpType.Cubic, cache: bool = False) -> None: ...

    def interp(self, x: float, y: float, z: float) -> float:
        """
        Evaluate the table at a single ``(x, y, z)`` point.

        Parameters
        ----------
        x : float
            X-coordinate within the table's x range.
        y : float
            Y-coordinate within the table's y range.
        z : float
            Z-coordinate within the table's z range.

        Returns
        -------
        float
            Interpolated scalar value at ``(x, y, z)``.

        Raises
        ------
        ValueError
            If any coordinate is outside the table's range.
        """

    def interp_deriv1(self, x: float, y: float, z: float) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]]:
        """
        Evaluate the table and its first partial derivatives at ``(x, y, z)``.

        Parameters
        ----------
        x : float
            X-coordinate within the table's range.
        y : float
            Y-coordinate within the table's range.
        z : float
            Z-coordinate within the table's range.

        Returns
        -------
        tuple[float, numpy.ndarray]
            ``(f, df_dxyz)`` where ``df_dxyz`` has shape ``(3,)`` containing
            ``[df/dx, df/dy, df/dz]``.
        """

    def interp_deriv2(self, x: float, y: float, z: float) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')]]:
        """
        Evaluate the table, first, and second partial derivatives at ``(x, y, z)``.

        Parameters
        ----------
        x : float
            X-coordinate within the table's range.
        y : float
            Y-coordinate within the table's range.
        z : float
            Z-coordinate within the table's range.

        Returns
        -------
        tuple[float, numpy.ndarray, numpy.ndarray]
            ``(f, df_dxyz, d2f_dxyz2)`` where ``df_dxyz`` has shape ``(3,)`` and
            ``d2f_dxyz2`` is a symmetric ``(3, 3)`` Hessian matrix.
        """

    @overload
    def __call__(self, arg0: float, arg1: float, arg2: float, /) -> float:
        """
        Evaluate the table numerically at a single ``(x, y, z)`` point.

        Equivalent to :py:meth:`interp(x, y, z) <InterpTable3D.interp>`.
        """

    @overload
    def __call__(self, arg0: ScalarFunction, arg1: ScalarFunction, arg2: ScalarFunction, /) -> ScalarFunction:
        """
        Compose the table with three scalar VectorFunctions to produce a new VectorFunction.

        Parameters
        ----------
        x : VectorFunction (scalar output)
            VectorFunction providing the x-coordinate.
        y : VectorFunction (scalar output)
            VectorFunction providing the y-coordinate.
        z : VectorFunction (scalar output)
            VectorFunction providing the z-coordinate.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating ``table(x(v), y(v), z(v))``.
        """

    @overload
    def __call__(self, arg0: Element, arg1: Element, arg2: Element, /) -> ScalarFunction:
        """
        Compose the table with three scalar Segments to produce a new VectorFunction.

        Parameters
        ----------
        x : Segment (scalar output)
            Scalar ``Segment`` providing the x-coordinate.
        y : Segment (scalar output)
            Scalar ``Segment`` providing the y-coordinate.
        z : Segment (scalar output)
            Scalar ``Segment`` providing the z-coordinate.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at the three segment values.
        """

    @overload
    def __call__(self, arg: Segment3, /) -> ScalarFunction:
        """
        Compose the table with a 3-element Segment to produce a new VectorFunction.

        Parameters
        ----------
        xyz : Segment (3-element output)
            Three-element ``Segment`` whose components supply the ``(x, y, z)``
            coordinates in order.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at ``(xyz[0], xyz[1], xyz[2])``.
        """

    @overload
    def __call__(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Compose the table with a 3-output VectorFunction to produce a new VectorFunction.

        Parameters
        ----------
        xyz : VectorFunction (3-element output)
            Three-output VectorFunction whose outputs supply the ``(x, y, z)``
            coordinates in order.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at ``(xyz(v)[0], xyz(v)[1], xyz(v)[2])``.
        """

    def sf(self) -> ScalarFunction:
        """
        Return a scalar VectorFunction that wraps this table.

        The returned VectorFunction takes a 3-element input ``[x, y, z]`` and returns
        a single scalar output.

        Returns
        -------
        VectorFunction
            3-in, 1-out VectorFunction backed by this table.
        """

    def vf(self) -> VectorFunction:
        """
        Return a vector VectorFunction that wraps this table (output size 1).

        The returned VectorFunction takes a 3-element input ``[x, y, z]`` and returns
        a single scalar output (output size 1).

        Returns
        -------
        VectorFunction
            3-in, 1-out VectorFunction backed by this table.
        """

class InterpTable4D:
    """
    Lookup-table VectorFunction that interpolates tabulated scalar data over four independent variables.

    ``InterpTable4D`` stores a 4-D grid of scalar samples (a rank-4 tensor) and
    evaluates them at arbitrary ``(x, y, z, w)`` query points via quadricubic or
    quadrilinear interpolation.  The output is always a single scalar.  Once
    constructed the table can be used as a plain callable for immediate numeric
    evaluation, or composed with four scalar VectorFunctions (or a 4-element
    ``Segment``) via ``__call__`` to produce a new scalar VectorFunction suitable
    for use in an optimal-control problem.  Derivatives are evaluated analytically.

    Available interpolation kinds: ``"cubic"`` (default, quadricubic) and ``"linear"``
    (quadrilinear).  The ``cache`` option pre-computes and stores the per-cell
    polynomial coefficients to accelerate repeated evaluations at the cost of
    additional memory.
    """

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, zs: numpy.ndarray, ws: numpy.ndarray, fs: numpy.ndarray, kind: str = 'cubic', cache: bool = False) -> None:
        """
        Construct a 4-D interpolation table.

        Parameters
        ----------
        xs : array_like, shape (Nx,)
            Strictly ascending x-coordinates.  Must have at least 5 elements.
            Correspond to the first dimension of ``fs``.
        ys : array_like, shape (Ny,)
            Strictly ascending y-coordinates.  Must have at least 5 elements.
            Correspond to the second dimension of ``fs``.
        zs : array_like, shape (Nz,)
            Strictly ascending z-coordinates.  Must have at least 5 elements.
            Correspond to the third dimension of ``fs``.
        ws : array_like, shape (Nw,)
            Strictly ascending w-coordinates.  Must have at least 5 elements.
            Correspond to the fourth dimension of ``fs``.
        fs : array_like, shape (Nx, Ny, Nz, Nw)
            Scalar values on the ``(xs, ys, zs, ws)`` grid in NumPy meshgrid ``ij``
            format (first index = x, second = y, third = z, fourth = w).
        kind : str, optional
            Interpolation method: ``"cubic"`` (default) or ``"linear"``.
        cache : bool, optional
            If ``True``, pre-compute and cache all per-cell polynomial coefficients
            at construction time.  Speeds up repeated queries; uses more memory.
            Default is ``False``.
        """

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, zs: numpy.ndarray, ws: numpy.ndarray, fs: numpy.ndarray, kind: InterpType = InterpType.Cubic, cache: bool = False) -> None: ...

    def interp(self, x: float, y: float, z: float, w: float) -> float:
        """
        Evaluate the table at a single ``(x, y, z, w)`` point.

        Parameters
        ----------
        x : float
            X-coordinate within the table's x range.
        y : float
            Y-coordinate within the table's y range.
        z : float
            Z-coordinate within the table's z range.
        w : float
            W-coordinate within the table's w range.

        Returns
        -------
        float
            Interpolated scalar value at ``(x, y, z, w)``.

        Raises
        ------
        ValueError
            If any coordinate is outside the table's range.
        """

    def interp_deriv1(self, x: float, y: float, z: float, w: float) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')]]:
        """
        Evaluate the table and its first partial derivatives at ``(x, y, z, w)``.

        Parameters
        ----------
        x : float
            X-coordinate within the table's range.
        y : float
            Y-coordinate within the table's range.
        z : float
            Z-coordinate within the table's range.
        w : float
            W-coordinate within the table's range.

        Returns
        -------
        tuple[float, numpy.ndarray]
            ``(f, df_dxyzw)`` where ``df_dxyzw`` has shape ``(4,)`` containing
            ``[df/dx, df/dy, df/dz, df/dw]``.
        """

    def interp_deriv2(self, x: float, y: float, z: float, w: float) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')], Annotated[NDArray[numpy.float64], dict(shape=(4, 4), order='F')]]:
        """
        Evaluate the table, first, and second partial derivatives at ``(x, y, z, w)``.

        Parameters
        ----------
        x : float
            X-coordinate within the table's range.
        y : float
            Y-coordinate within the table's range.
        z : float
            Z-coordinate within the table's range.
        w : float
            W-coordinate within the table's range.

        Returns
        -------
        tuple[float, numpy.ndarray, numpy.ndarray]
            ``(f, df_dxyzw, d2f_dxyzw2)`` where ``df_dxyzw`` has shape ``(4,)``
            and ``d2f_dxyzw2`` is a symmetric ``(4, 4)`` Hessian matrix.
        """

    @overload
    def __call__(self, arg0: float, arg1: float, arg2: float, arg3: float, /) -> float:
        """
        Evaluate the table numerically at a single ``(x, y, z, w)`` point.

        Equivalent to :py:meth:`interp(x, y, z, w) <InterpTable4D.interp>`.
        """

    @overload
    def __call__(self, arg0: ScalarFunction, arg1: ScalarFunction, arg2: ScalarFunction, arg3: ScalarFunction, /) -> ScalarFunction:
        """
        Compose the table with four scalar VectorFunctions to produce a new VectorFunction.

        Parameters
        ----------
        x : VectorFunction (scalar output)
            VectorFunction providing the x-coordinate.
        y : VectorFunction (scalar output)
            VectorFunction providing the y-coordinate.
        z : VectorFunction (scalar output)
            VectorFunction providing the z-coordinate.
        w : VectorFunction (scalar output)
            VectorFunction providing the w-coordinate.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating ``table(x(v), y(v), z(v), w(v))``.
        """

    @overload
    def __call__(self, arg0: Element, arg1: Element, arg2: Element, arg3: Element, /) -> ScalarFunction:
        """
        Compose the table with four scalar Segments to produce a new VectorFunction.

        Parameters
        ----------
        x : Segment (scalar output)
            Scalar ``Segment`` providing the x-coordinate.
        y : Segment (scalar output)
            Scalar ``Segment`` providing the y-coordinate.
        z : Segment (scalar output)
            Scalar ``Segment`` providing the z-coordinate.
        w : Segment (scalar output)
            Scalar ``Segment`` providing the w-coordinate.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at the four segment values.
        """

    @overload
    def __call__(self, arg: Segment, /) -> ScalarFunction:
        """
        Compose the table with a dynamically-sized Segment to produce a new VectorFunction.

        Parameters
        ----------
        xyzw : Segment (dynamic size)
            A ``Segment`` whose first four components supply the ``(x, y, z, w)``
            coordinates in order.  The size is not checked at compile time; the
            caller is responsible for ensuring at least four components are present.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at
            ``(xyzw[0], xyzw[1], xyzw[2], xyzw[3])``.
        """

    @overload
    def __call__(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Compose the table with a 4-output VectorFunction to produce a new VectorFunction.

        Parameters
        ----------
        xyzw : VectorFunction (4-element output)
            Four-output VectorFunction whose outputs supply the ``(x, y, z, w)``
            coordinates in order.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating the table at
            ``(xyzw(v)[0], xyzw(v)[1], xyzw(v)[2], xyzw(v)[3])``.
        """

    def sf(self) -> ScalarFunction:
        """
        Return a scalar VectorFunction that wraps this table.

        The returned VectorFunction takes a 4-element input ``[x, y, z, w]`` and
        returns a single scalar output.

        Returns
        -------
        VectorFunction
            4-in, 1-out VectorFunction backed by this table.
        """

    def vf(self) -> VectorFunction:
        """
        Return a vector VectorFunction that wraps this table (output size 1).

        The returned VectorFunction takes a 4-element input ``[x, y, z, w]`` and
        returns a single scalar output (output size 1).

        Returns
        -------
        VectorFunction
            4-in, 1-out VectorFunction backed by this table.
        """

class ChebTable:
    """
    Chebyshev (2nd-kind / DCT-I) interpolant data class — 1-D and N-D.

    Holds Chebyshev coefficients of one or more output channels and evaluates them
    numerically via Clenshaw recurrence with analytic first and second derivatives.
    This is the interpolant *data* — it is not itself a VectorFunction.  Build from
    values sampled at :py:meth:`cheb_points`, then obtain a VectorFunction with
    :py:meth:`vf` or a ``__call__`` compose overload for use in optimal-control problems.

    Derivative-series coefficients are precomputed once at construction time, so
    solve-time evaluation incurs no per-call allocation or recurrence work.
    """

    class OutOfDomain(enum.Enum):
        """
        Per-axis behaviour for queries outside ``[lb, ub]``.

        ``Clamp`` (default) snaps the query to the nearest endpoint (a global Chebyshev
        polynomial diverges outside its domain, so extrapolation is never desired).
        ``Periodic`` wraps the query modulo the axis span back into the domain — for
        angle-like / invariant-manifold axes; the sampled function must satisfy
        ``f(lb) ~= f(ub)`` on that axis for the interpolant to be continuous.
        """

        Clamp = 0

        Periodic = 1

    @overload
    @staticmethod
    def cheb_points(order: int, lb: float, ub: float) -> numpy.ndarray:
        """
        Return the order+1 second-kind Chebyshev nodes on [lb, ub] (1-D).

        The nodes are ordered so that ``t[0] = ub`` and ``t[order] = lb`` (i.e. the
        cosine mapping places the first node at the right endpoint).  Pass the returned
        array to :py:meth:`from_values` to build a :py:class:`ChebTable`.

        Parameters
        ----------
        order : int
            Polynomial order ``n >= 1``.  The returned array has ``n+1`` elements.
        lb : float
            Lower bound of the physical domain.
        ub : float
            Upper bound of the physical domain.  Must satisfy ``ub > lb``.

        Returns
        -------
        numpy.ndarray, shape (order+1,)
            Second-kind Chebyshev node coordinates on ``[lb, ub]``.
        """

    @overload
    @staticmethod
    def cheb_points(orders: Sequence[int], lb: numpy.ndarray, ub: numpy.ndarray) -> list:
        """
        Return per-axis Chebyshev nodes for an N-D tensor-product grid.

        Parameters
        ----------
        orders : list[int]
            Per-axis polynomial orders; axis ``d`` has ``orders[d]+1`` nodes.
        lb : array_like, shape (D,)
            Per-axis lower bounds.
        ub : array_like, shape (D,)
            Per-axis upper bounds.  Must satisfy ``ub[d] > lb[d]`` for all d.

        Returns
        -------
        list of numpy.ndarray
            ``orders[d]+1`` node coordinates for each axis d.
        """

    @overload
    @staticmethod
    def from_values(grid_values: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], lb: float, ub: float, order: int, nthreads: int = 1, out_of_domain: OutOfDomain = OutOfDomain.Clamp) -> ChebTable:
        """
        Build a ChebTable from values sampled at the Chebyshev nodes (1-D).

        The values must be sampled at the ``order+1`` nodes returned by
        :py:meth:`cheb_points`.  Coefficients are computed via a DCT-I transform
        (pocketfft) and derivative-series coefficients are precomputed immediately,
        so subsequent evaluation calls incur no per-call allocation.

        Parameters
        ----------
        grid_values : array_like, shape (olen, order+1)
            Sampled values.  Rows are output channels; columns correspond to the
            ``order+1`` Chebyshev nodes in the order returned by :py:meth:`cheb_points`.
        lb : float
            Lower bound of the physical domain.
        ub : float
            Upper bound of the physical domain.  Must satisfy ``ub > lb``.
        order : int
            Polynomial order ``n >= 1``.
        nthreads : int, optional
            Thread count forwarded to pocketfft for the DCT-I transform.  Defaults
            to 1; pass 0 to use all available cores.  Negative values raise
            ``ValueError``.
        out_of_domain : ChebTable.OutOfDomain, optional
            Behaviour for queries outside ``[lb, ub]``.  ``Clamp`` (default) snaps to
            the nearest endpoint; ``Periodic`` wraps modulo the span (see
            :py:class:`ChebTable.OutOfDomain`).

        Returns
        -------
        ChebTable
            Constructed interpolant ready for evaluation or composition.
        """

    @overload
    @staticmethod
    def from_values(grid_values_flat: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], lb: numpy.ndarray, ub: numpy.ndarray, orders: Sequence[int], nthreads: int = 1, out_of_domain: Sequence[OutOfDomain] = []) -> ChebTable:
        """
        Build a ChebTable from values on an N-D tensor-product Chebyshev grid.

        Parameters
        ----------
        grid_values_flat : array_like, shape (tsize, olen)
            Flattened grid values in row-major (axis-0-outer) order, where
            ``tsize = prod(orders[d]+1)``.  Columns are output channels.
            Obtain the grid from :py:meth:`cheb_points` and flatten row-major.
        lb : array_like, shape (D,)
            Per-axis lower bounds.
        ub : array_like, shape (D,)
            Per-axis upper bounds.
        orders : list[int]
            Per-axis polynomial orders.
        nthreads : int, optional
            Thread count for the DCT-I transforms.  Defaults to 1.
        out_of_domain : list[ChebTable.OutOfDomain], optional
            Per-axis out-of-domain policy (length D).  Empty (default) means ``Clamp``
            on every axis.  Use ``Periodic`` on an axis to wrap queries modulo its span.

        Returns
        -------
        ChebTable
            N-D Chebyshev interpolant ready for evaluation or composition.
        """

    @overload
    def eval(self, t: float) -> numpy.ndarray:
        """
        Evaluate all output channels at scalar coordinate t (1-D).

        Parameters
        ----------
        t : float
            Query coordinate.  Values outside ``[lb, ub]`` are clamped to the
            nearest endpoint (or wrapped, if the axis policy is Periodic); no
            extrapolation error is raised.  Requires a 1-D table — calling this on an
            N-D table raises ``ValueError`` (pass a length-D array instead).

        Returns
        -------
        numpy.ndarray, shape (olen,)
            Interpolated channel values at ``t``.
        """

    @overload
    def eval(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Evaluate all output channels at a physical point x (N-D).

        Parameters
        ----------
        x : array_like, shape (D,)
            Query coordinates.  Each coordinate is mapped into its axis interval per
            that axis's out-of-domain policy (clamped by default; wrapped if Periodic).
            Its length must equal ``input_dim`` or a ``ValueError`` is raised.

        Returns
        -------
        numpy.ndarray, shape (olen,)
            Interpolated channel values at ``x``.
        """

    @overload
    def __call__(self, t: float) -> numpy.ndarray:
        """
        Evaluate all output channels at scalar coordinate t (numeric call).

        Equivalent to :py:meth:`eval`.
        """

    @overload
    def __call__(self, arg: ScalarFunction, /) -> VectorFunction:
        """
        Compose the table with a scalar VectorFunction to produce a new VectorFunction.

        Parameters
        ----------
        t : VectorFunction (scalar output)
            Scalar-output VectorFunction whose result is used as the query coordinate.

        Returns
        -------
        VectorFunction
            VectorFunction evaluating all channels at ``t(x)``.  Output size equals
            ``output_dim`` at runtime.
        """

    @overload
    def __call__(self, arg: Element, /) -> VectorFunction:
        """
        Compose the table with a scalar Segment to produce a new VectorFunction.

        Parameters
        ----------
        t : Segment (scalar output)
            Scalar-output ``Segment`` (a single element of a state/control vector)
            whose value is used as the query coordinate.

        Returns
        -------
        VectorFunction
            VectorFunction evaluating all channels at the segment value.  Output size
            equals ``output_dim`` at runtime.
        """

    @overload
    def __call__(self, arg0: ScalarFunction, arg1: ScalarFunction, /) -> VectorFunction:
        """
        Compose the table with two scalar VectorFunctions (2-D input).

        Parameters
        ----------
        x : VectorFunction (scalar output)
            VectorFunction providing the first coordinate.
        y : VectorFunction (scalar output)
            VectorFunction providing the second coordinate.

        Returns
        -------
        VectorFunction
            VectorFunction evaluating all channels at ``(x(v), y(v))``.
        """

    @overload
    def __call__(self, arg0: Element, arg1: Element, /) -> VectorFunction:
        """
        Compose the table with two scalar Segments (2-D input).

        Parameters
        ----------
        x : Segment (scalar output)
        y : Segment (scalar output)

        Returns
        -------
        VectorFunction
            VectorFunction evaluating all channels at the two segment values.
        """

    @overload
    def __call__(self, arg0: ScalarFunction, arg1: ScalarFunction, arg2: ScalarFunction, /) -> VectorFunction:
        """
        Compose the table with three scalar VectorFunctions (3-D input).

        Parameters
        ----------
        x : VectorFunction (scalar output)
        y : VectorFunction (scalar output)
        z : VectorFunction (scalar output)

        Returns
        -------
        VectorFunction
            VectorFunction evaluating all channels at ``(x(v), y(v), z(v))``.
        """

    @overload
    def __call__(self, arg0: Element, arg1: Element, arg2: Element, /) -> VectorFunction:
        """
        Compose the table with three scalar Segments (3-D input).

        Parameters
        ----------
        x : Segment (scalar output)
        y : Segment (scalar output)
        z : Segment (scalar output)

        Returns
        -------
        VectorFunction
            VectorFunction evaluating all channels at the three segment values.
        """

    def eval_deriv1(self, t: float) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Evaluate value and first derivative at t.

        Parameters
        ----------
        t : float
            Query coordinate within or clamped to ``[lb, ub]``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(value, deriv1)`` where both arrays have shape ``(olen,)``.
            ``deriv1`` contains ``df/dt`` for each output channel.
        """

    def eval_deriv2(self, t: float) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Evaluate value, first derivative, and second derivative at t.

        Parameters
        ----------
        t : float
            Query coordinate within or clamped to ``[lb, ub]``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(value, deriv1, deriv2)`` where all arrays have shape ``(olen,)``.
            ``deriv1`` contains ``df/dt`` and ``deriv2`` contains ``d²f/dt²`` for
            each output channel.
        """

    def coeff_tail_norm(self) -> numpy.ndarray:
        """
        Per-channel norm of the trailing-half Chebyshev coefficients (1-D).

        Returns the L2 norm of the upper half of the coefficient series for each
        output channel.  Used by the adaptive construction loop
        (:py:func:`cheb_adaptive`) as a convergence indicator: when this norm is
        small relative to the function scale the series has converged.

        Returns
        -------
        numpy.ndarray, shape (olen,)
            Tail norms; one entry per output channel.
        """

    @property
    def output_dim(self) -> int:
        """Number of output channels (olen)."""

    @property
    def input_dim(self) -> int:
        """Number of input dimensions (1 for 1-D tables, D for N-D)."""

    @property
    def order(self) -> int:
        """Polynomial order n for 1-D tables (orders()[0])."""

    @property
    def orders(self) -> list:
        """Per-axis polynomial orders as a list of int."""

    @property
    def lb(self) -> object:
        """Lower domain bound(s): a float for 1-D tables, an ndarray for N-D."""

    @property
    def ub(self) -> object:
        """Upper domain bound(s): a float for 1-D tables, an ndarray for N-D."""

    @property
    def out_of_domain(self) -> list:
        """Per-axis out-of-domain policy as a list of ChebTable.OutOfDomain."""

    @overload
    def contains(self, t: float) -> bool:
        """
        True iff scalar ``t`` lies within ``[lb, ub]`` (1-D tables).

        Lets callers guard against out-of-domain queries (which otherwise clamp, or wrap
        on Periodic axes) before evaluating.
        """

    @overload
    def contains(self, x: numpy.ndarray) -> bool:
        """
        True iff every coordinate of ``x`` lies within ``[lb, ub]`` (N-D tables).
        """

    def in_domain(self, x: numpy.ndarray) -> bool:
        """Alias of :py:meth:`contains` for an array query."""

    def vf(self) -> VectorFunction:
        """
        Return a VectorFunction backed by this table (any dimension).

        For a 1-D table, takes a single-element input.
        For an N-D table, takes an N-element input vector.

        Returns
        -------
        VectorFunction
            input_dim-in, ``output_dim``-out VectorFunction backed by this table.
        """

def scalar_dynamic_stack_test(arg: Sequence[ScalarFunction], /) -> VectorFunction:
    """
    Internal testing helper: stack a list of scalar functions into a VectorFunction.

    Constructs a ``DynamicStackedOutputs`` from a homogeneous list of scalar-output
    (1-output-row) VectorFunctions and type-erases the result to ``VectorFunction``.
    This function exists to exercise the dynamic-stacking code path from Python tests;
    it is not part of the public API.

    Parameters
    ----------
    funcs : list[ScalarFunction]
        Non-empty list of scalar-output VectorFunctions sharing the same input size.

    Returns
    -------
    VectorFunction
        A VectorFunction whose output is the vertical concatenation of all inputs.
    """

def dynamic_stack_test(arg: Sequence[VectorFunction], /) -> VectorFunction:
    """
    Internal testing helper: stack a list of VectorFunctions into a VectorFunction.

    Constructs a ``DynamicStackedOutputs`` from a homogeneous list of multi-output
    VectorFunctions and type-erases the result to ``VectorFunction``.  This function
    exists to exercise the dynamic-stacking code path from Python tests; it is not
    part of the public API.

    Parameters
    ----------
    funcs : list[VectorFunction]
        Non-empty list of VectorFunctions sharing the same input size.

    Returns
    -------
    VectorFunction
        A VectorFunction whose output is the vertical concatenation of all inputs.
    """
