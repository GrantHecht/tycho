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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
            If the stacked output size does not match ``self.input_rows()``, or if the
            functions in ``funcs`` do not share a common input size.
        """

    @overload
    def __call__(self, arg: Sequence[VectorFunction], /) -> VectorFunction: ...

    @overload
    def __call__(self, arg0: VectorFunction, /, *args) -> VectorFunction: ...

    @overload
    def __call__(self, arg0: float, /, *args) -> VectorFunction: ...

    @overload
    def __call__(self, arg0: numpy.ndarray, /, *args) -> VectorFunction: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
        s : float, optional
            Scalar multiplier applied to the result.  When omitted the result is
            not scaled.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``, or the
            scaled variant when ``s`` is provided.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction: ...

    @overload
    def normalized_power3(self) -> VectorFunction: ...

    def norm(self) -> ScalarFunction: ...

    def squared_norm(self) -> ScalarFunction: ...

    def cubed_norm(self) -> ScalarFunction: ...

    def inverse_norm(self) -> ScalarFunction: ...

    def inverse_squared_norm(self) -> ScalarFunction: ...

    def inverse_cubed_norm(self) -> ScalarFunction: ...

    def inverse_four_norm(self) -> ScalarFunction: ...

    def normalized(self) -> VectorFunction: ...

    def normalized_power2(self) -> VectorFunction: ...

    def normalized_power4(self) -> VectorFunction: ...

    def normalized_power5(self) -> VectorFunction: ...

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
    def cross(self, arg: Segment3, /) -> VectorFunction: ...

    @overload
    def cross(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction: ...

    @overload
    def cross(self, arg: VectorFunction, /) -> VectorFunction: ...

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
    def dot(self, arg: VectorFunction, /) -> ScalarFunction:
        """
        Dot product of this VectorFunction with another equal-length one.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.

        Returns
        -------
        VectorFunction
            Scalar VectorFunction evaluating ``self(x) . other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise product of this VectorFunction with another.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.
            Accepts a VectorFunction of any concrete type or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) .* other(x)`` element-wise.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise quotient of this VectorFunction divided by another.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Divisor; must have the same output dimension as ``self``.
            Accepts a VectorFunction of any concrete type or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) ./ other(x)`` element-wise.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction: ...

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

        Parameters
        ----------
        b : float or scalar VectorFunction
            Divisor.  When a float, every output component is divided by ``b``.
            When a scalar VectorFunction, each output is divided element-wise by
            the runtime value of ``b(x)``.

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
        Extract a single output element or a contiguous slice (``self[i]`` or ``self[i:j]``).

        Parameters
        ----------
        index : int or slice
            Integer index selects a single scalar output.  A ``slice`` selects a
            contiguous sub-vector (step must be 1 and the slice must be non-empty
            and forward-only).

        Returns
        -------
        VectorFunction
            Scalar expression for integer index; sub-vector expression for a slice.

        Raises
        ------
        ValueError
            If the index is out of range, the slice is empty, backward, or non-contiguous.
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

    def segment2(self, arg: int, /) -> VectorFunction: ...

    def head2(self) -> VectorFunction: ...

    def tail2(self) -> VectorFunction: ...

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

    def segment3(self, arg: int, /) -> VectorFunction: ...

    def head3(self) -> VectorFunction: ...

    def tail3(self) -> VectorFunction: ...

    @overload
    def eval(self, arg: Element, /) -> VectorFunction:
        """
        Compose this VectorFunction with an inner VectorFunction: ``self(g(.))``.

        Substitutes the inner function ``g`` into the input of ``self``, forming
        the composition ``x ↦ self(g(x))``.  The inner function's output dimension
        must match ``self``'s input dimension.

        Parameters
        ----------
        g : VectorFunction
            Inner function whose output feeds ``self``'s input.  Accepts all
            concrete VectorFunction types (scalar, segment, generic, etc.).

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(g(x))``.
        """

    @overload
    def eval(self, arg: Segment, /) -> VectorFunction: ...

    @overload
    def eval(self, arg: Segment2, /) -> VectorFunction: ...

    @overload
    def eval(self, arg: Segment3, /) -> VectorFunction: ...

    @overload
    def eval(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def eval(self, arg0: int, arg1: numpy.ndarray, /) -> VectorFunction: ...

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
    def apply(self, arg: ScalarFunction, /) -> VectorFunction: ...

class ScalarFunction:
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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
            If the stacked output size does not match ``self.input_rows()``, or if the
            functions in ``funcs`` do not share a common input size.
        """

    @overload
    def __call__(self, arg: Sequence[VectorFunction], /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg0: VectorFunction, /, *args) -> ScalarFunction: ...

    @overload
    def __call__(self, arg0: float, /, *args) -> ScalarFunction: ...

    @overload
    def __call__(self, arg0: numpy.ndarray, /, *args) -> ScalarFunction: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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

        Parameters
        ----------
        b : float or scalar VectorFunction
            Divisor.  When a float, every output component is divided by ``b``.
            When a scalar VectorFunction, each output is divided element-wise by
            the runtime value of ``b(x)``.

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
        VectorFunction
            Scalar VectorFunction evaluating ``self(x) . other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_quotient(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def eval(self, arg: Element, /) -> ScalarFunction:
        """
        Compose this VectorFunction with an inner VectorFunction: ``self(g(.))``.

        Substitutes the inner function ``g`` into the input of ``self``, forming
        the composition ``x ↦ self(g(x))``.  The inner function's output dimension
        must match ``self``'s input dimension.

        Parameters
        ----------
        g : VectorFunction
            Inner function whose output feeds ``self``'s input.  Accepts all
            concrete VectorFunction types (scalar, segment, generic, etc.).

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(g(x))``.
        """

    @overload
    def eval(self, arg: Segment, /) -> ScalarFunction: ...

    @overload
    def eval(self, arg: Segment2, /) -> ScalarFunction: ...

    @overload
    def eval(self, arg: Segment3, /) -> ScalarFunction: ...

    @overload
    def eval(self, arg: VectorFunction, /) -> ScalarFunction: ...

    @overload
    def eval(self, arg0: int, arg1: numpy.ndarray, /) -> ScalarFunction: ...

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
    def apply(self, arg: ScalarFunction, /) -> VectorFunction: ...

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
        Extract a single output element or a contiguous slice (``self[i]`` or ``self[i:j]``).

        Parameters
        ----------
        index : int or slice
            Integer index selects a single scalar output.  A ``slice`` selects a
            contiguous sub-vector (step must be 1 and the slice must be non-empty
            and forward-only).

        Returns
        -------
        VectorFunction
            Scalar expression for integer index; sub-vector expression for a slice.

        Raises
        ------
        ValueError
            If the index is out of range, the slice is empty, backward, or non-contiguous.
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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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

        Parameters
        ----------
        b : float or scalar VectorFunction
            Divisor.  When a float, every output component is divided by ``b``.
            When a scalar VectorFunction, each output is divided element-wise by
            the runtime value of ``b(x)``.

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
        s : float, optional
            Scalar multiplier applied to the result.  When omitted the result is
            not scaled.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``, or the
            scaled variant when ``s`` is provided.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction: ...

    @overload
    def normalized_power3(self) -> VectorFunction: ...

    def norm(self) -> ScalarFunction: ...

    def squared_norm(self) -> ScalarFunction: ...

    def cubed_norm(self) -> ScalarFunction: ...

    def inverse_norm(self) -> ScalarFunction: ...

    def inverse_squared_norm(self) -> ScalarFunction: ...

    def inverse_cubed_norm(self) -> ScalarFunction: ...

    def inverse_four_norm(self) -> ScalarFunction: ...

    def normalized(self) -> VectorFunction: ...

    def normalized_power2(self) -> VectorFunction: ...

    def normalized_power4(self) -> VectorFunction: ...

    def normalized_power5(self) -> VectorFunction: ...

    @overload
    def cross(self, arg: Segment3, /) -> VectorFunction: ...

    @overload
    def cross(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cross(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction: ...

    @overload
    def cross(self, arg: Segment, /) -> VectorFunction: ...

    @overload
    def dot(self, arg: VectorFunction, /) -> ScalarFunction: ...

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
        VectorFunction
            Scalar VectorFunction evaluating ``self(x) . other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction: ...

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
    def apply(self, arg: ScalarFunction, /) -> VectorFunction: ...

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
        Extract a single output element or a contiguous slice (``self[i]`` or ``self[i:j]``).

        Parameters
        ----------
        index : int or slice
            Integer index selects a single scalar output.  A ``slice`` selects a
            contiguous sub-vector (step must be 1 and the slice must be non-empty
            and forward-only).

        Returns
        -------
        VectorFunction
            Scalar expression for integer index; sub-vector expression for a slice.

        Raises
        ------
        ValueError
            If the index is out of range, the slice is empty, backward, or non-contiguous.
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

    def segment2(self, arg: int, /) -> Segment2: ...

    def head2(self) -> Segment2: ...

    def tail2(self) -> Segment2: ...

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

    def segment3(self, arg: int, /) -> Segment3: ...

    def head3(self) -> Segment3: ...

    def tail3(self) -> Segment3: ...

    @overload
    def tolist(self) -> list[Element]:
        """
        Split the function into a list of scalar element functions.

        Returns one scalar-valued VectorFunction per output row, in order, each
        extracting a single component of the original output vector.

        Returns
        -------
        list of GenericFunction (scalar)
            A list of ``output_rows`` scalar functions, where element *i* extracts
            output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]: ...

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]: ...

class Element:
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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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

        Parameters
        ----------
        b : float or scalar VectorFunction
            Divisor.  When a float, every output component is divided by ``b``.
            When a scalar VectorFunction, each output is divided element-wise by
            the runtime value of ``b(x)``.

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
        VectorFunction
            Scalar VectorFunction evaluating ``self(x) . other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: Element, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_quotient(self, arg: Element, /) -> ScalarFunction: ...

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

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
    def apply(self, arg: ScalarFunction, /) -> VectorFunction: ...

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
        Extract a single output element or a contiguous slice (``self[i]`` or ``self[i:j]``).

        Parameters
        ----------
        index : int or slice
            Integer index selects a single scalar output.  A ``slice`` selects a
            contiguous sub-vector (step must be 1 and the slice must be non-empty
            and forward-only).

        Returns
        -------
        VectorFunction
            Scalar expression for integer index; sub-vector expression for a slice.

        Raises
        ------
        ValueError
            If the index is out of range, the slice is empty, backward, or non-contiguous.
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
        list of GenericFunction (scalar)
            A list of ``output_rows`` scalar functions, where element *i* extracts
            output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]: ...

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]: ...

class Arguments:
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
    def constant(self, arg: float, /) -> ScalarFunction: ...

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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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

        Parameters
        ----------
        b : float or scalar VectorFunction
            Divisor.  When a float, every output component is divided by ``b``.
            When a scalar VectorFunction, each output is divided element-wise by
            the runtime value of ``b(x)``.

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
        s : float, optional
            Scalar multiplier applied to the result.  When omitted the result is
            not scaled.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``, or the
            scaled variant when ``s`` is provided.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction: ...

    @overload
    def normalized_power3(self) -> VectorFunction: ...

    def norm(self) -> ScalarFunction: ...

    def squared_norm(self) -> ScalarFunction: ...

    def cubed_norm(self) -> ScalarFunction: ...

    def inverse_norm(self) -> ScalarFunction: ...

    def inverse_squared_norm(self) -> ScalarFunction: ...

    def inverse_cubed_norm(self) -> ScalarFunction: ...

    def inverse_four_norm(self) -> ScalarFunction: ...

    def normalized(self) -> VectorFunction: ...

    def normalized_power2(self) -> VectorFunction: ...

    def normalized_power4(self) -> VectorFunction: ...

    def normalized_power5(self) -> VectorFunction: ...

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
    def cross(self, arg: Segment3, /) -> VectorFunction: ...

    @overload
    def cross(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cross(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction: ...

    @overload
    def cross(self, arg: Arguments, /) -> VectorFunction: ...

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
    def dot(self, arg: VectorFunction, /) -> ScalarFunction: ...

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
        VectorFunction
            Scalar VectorFunction evaluating ``self(x) . other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise product of this VectorFunction with another.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.
            Accepts a VectorFunction of any concrete type or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) .* other(x)`` element-wise.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: Arguments, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise quotient of this VectorFunction divided by another.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Divisor; must have the same output dimension as ``self``.
            Accepts a VectorFunction of any concrete type or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) ./ other(x)`` element-wise.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: Arguments, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction: ...

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
    def apply(self, arg: ScalarFunction, /) -> VectorFunction: ...

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
        Extract a single output element or a contiguous slice (``self[i]`` or ``self[i:j]``).

        Parameters
        ----------
        index : int or slice
            Integer index selects a single scalar output.  A ``slice`` selects a
            contiguous sub-vector (step must be 1 and the slice must be non-empty
            and forward-only).

        Returns
        -------
        VectorFunction
            Scalar expression for integer index; sub-vector expression for a slice.

        Raises
        ------
        ValueError
            If the index is out of range, the slice is empty, backward, or non-contiguous.
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

    def segment2(self, arg: int, /) -> Segment2: ...

    def head2(self) -> Segment2: ...

    def tail2(self) -> Segment2: ...

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

    def segment3(self, arg: int, /) -> Segment3: ...

    def head3(self) -> Segment3: ...

    def tail3(self) -> Segment3: ...

    @overload
    def tolist(self) -> list[Element]:
        """
        Split the function into a list of scalar element functions.

        Returns one scalar-valued VectorFunction per output row, in order, each
        extracting a single component of the original output vector.

        Returns
        -------
        list of GenericFunction (scalar)
            A list of ``output_rows`` scalar functions, where element *i* extracts
            output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]: ...

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]: ...

class Segment2:
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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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

        Parameters
        ----------
        b : float or scalar VectorFunction
            Divisor.  When a float, every output component is divided by ``b``.
            When a scalar VectorFunction, each output is divided element-wise by
            the runtime value of ``b(x)``.

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
        s : float, optional
            Scalar multiplier applied to the result.  When omitted the result is
            not scaled.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``, or the
            scaled variant when ``s`` is provided.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction: ...

    @overload
    def normalized_power3(self) -> VectorFunction: ...

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
        Output normalized and divided by an additional factor of its norm: ``self(x) / ||self(x)||²``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂²``.
        """

    def normalized_power4(self) -> VectorFunction:
        """
        Output normalized and divided by the cube of its norm: ``self(x) / ||self(x)||⁴``.

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
    def dot(self, arg: VectorFunction, /) -> ScalarFunction: ...

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
        VectorFunction
            Scalar VectorFunction evaluating ``self(x) . other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise product of this VectorFunction with another.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.
            Accepts a VectorFunction of any concrete type or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) .* other(x)`` element-wise.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: Segment2, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise quotient of this VectorFunction divided by another.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Divisor; must have the same output dimension as ``self``.
            Accepts a VectorFunction of any concrete type or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) ./ other(x)`` element-wise.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: Segment2, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction: ...

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
    def apply(self, arg: ScalarFunction, /) -> VectorFunction: ...

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
        Extract a single output element or a contiguous slice (``self[i]`` or ``self[i:j]``).

        Parameters
        ----------
        index : int or slice
            Integer index selects a single scalar output.  A ``slice`` selects a
            contiguous sub-vector (step must be 1 and the slice must be non-empty
            and forward-only).

        Returns
        -------
        VectorFunction
            Scalar expression for integer index; sub-vector expression for a slice.

        Raises
        ------
        ValueError
            If the index is out of range, the slice is empty, backward, or non-contiguous.
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
        list of GenericFunction (scalar)
            A list of ``output_rows`` scalar functions, where element *i* extracts
            output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]: ...

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]: ...

class Segment3:
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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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

        Parameters
        ----------
        b : float or scalar VectorFunction
            Divisor.  When a float, every output component is divided by ``b``.
            When a scalar VectorFunction, each output is divided element-wise by
            the runtime value of ``b(x)``.

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
        s : float, optional
            Scalar multiplier applied to the result.  When omitted the result is
            not scaled.

        Returns
        -------
        VectorFunction
            Expression evaluating ``(self(x) + b) / ||self(x) + b||³``, or the
            scaled variant when ``s`` is provided.
        """

    @overload
    def normalized_power3(self, arg0: numpy.ndarray, arg1: float, /) -> VectorFunction: ...

    @overload
    def normalized_power3(self) -> VectorFunction: ...

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
        Output normalized and divided by an additional factor of its norm: ``self(x) / ||self(x)||²``.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) / ||self(x)||₂²``.
        """

    def normalized_power4(self) -> VectorFunction:
        """
        Output normalized and divided by the cube of its norm: ``self(x) / ||self(x)||⁴``.

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
    def cross(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cross(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> VectorFunction: ...

    @overload
    def cross(self, arg: Segment3, /) -> VectorFunction: ...

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
    def dot(self, arg: VectorFunction, /) -> ScalarFunction: ...

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
        VectorFunction
            Scalar VectorFunction evaluating ``self(x) . other(x)``.
        """

    @overload
    def dot(self, arg: numpy.ndarray, /) -> ScalarFunction: ...

    @overload
    def cwise_product(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise product of this VectorFunction with another.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Right-hand operand; must have the same output dimension as ``self``.
            Accepts a VectorFunction of any concrete type or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) .* other(x)`` element-wise.
        """

    @overload
    def cwise_product(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: Segment3, /) -> VectorFunction: ...

    @overload
    def cwise_product(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: Segment, /) -> VectorFunction:
        """
        Element-wise quotient of this VectorFunction divided by another.

        Parameters
        ----------
        other : VectorFunction or array_like, shape (n,)
            Divisor; must have the same output dimension as ``self``.
            Accepts a VectorFunction of any concrete type or a constant vector.

        Returns
        -------
        VectorFunction
            Expression evaluating ``self(x) ./ other(x)`` element-wise.
        """

    @overload
    def cwise_quotient(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: Segment3, /) -> VectorFunction: ...

    @overload
    def cwise_quotient(self, arg: numpy.ndarray, /) -> VectorFunction: ...

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
    def apply(self, arg: ScalarFunction, /) -> VectorFunction: ...

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
        Extract a single output element or a contiguous slice (``self[i]`` or ``self[i:j]``).

        Parameters
        ----------
        index : int or slice
            Integer index selects a single scalar output.  A ``slice`` selects a
            contiguous sub-vector (step must be 1 and the slice must be non-empty
            and forward-only).

        Returns
        -------
        VectorFunction
            Scalar expression for integer index; sub-vector expression for a slice.

        Raises
        ------
        ValueError
            If the index is out of range, the slice is empty, backward, or non-contiguous.
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

    def segment2(self, arg: int, /) -> Segment2: ...

    def head2(self) -> Segment2: ...

    def tail2(self) -> Segment2: ...

    @overload
    def tolist(self) -> list[Element]:
        """
        Split the function into a list of scalar element functions.

        Returns one scalar-valued VectorFunction per output row, in order, each
        extracting a single component of the original output vector.

        Returns
        -------
        list of GenericFunction (scalar)
            A list of ``output_rows`` scalar functions, where element *i* extracts
            output component *i*.
        """

    @overload
    def tolist(self, arg: Sequence[int], /) -> list[Element]: ...

    @overload
    def tolist(self, arg: Sequence[tuple[int, int]], /) -> list[object]: ...

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
        dimension.  The list must be non-empty.

    Returns
    -------
    VectorFunction
        A VectorFunction whose output is the vertical concatenation of all
        inputs; its output dimension is ``len(funcs)``.

    Raises
    ------
    ValueError
        If any two functions have mismatched input dimensions, or the list is
        empty.

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
        dimension.  The list must be non-empty.

    Returns
    -------
    VectorFunction
        A VectorFunction whose output dimension equals ``len(funcs)`` formed by
        concatenating the scalar outputs of each element.

    Raises
    ------
    ValueError
        If any two functions have mismatched input dimensions, or the list is
        empty.
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
    Elementwise sum of several scalar segment-output VectorFunctions.

    Sums a list of single-element segment VectorFunctions (each extracting one
    output component) into a single scalar-output VectorFunction.  All segments
    must share the same input dimension.  Useful for building scalar objectives
    or constraints from individual state or control components.

    Parameters
    ----------
    funcs : list[VectorFunction]
        Single-element segment VectorFunctions sharing the same input dimension.
        The list must contain at least one element.

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
def sum_elems(arg0: Sequence[Element], arg1: Sequence[float], /) -> ScalarFunction: ...

@overload
def normalize(arg: numpy.ndarray, /) -> numpy.ndarray: ...

@overload
def normalize(arg: Arguments, /) -> object:
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
def normalize(arg: Segment, /) -> object:
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
def normalize(arg: Segment2, /) -> object:
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
def normalize(arg: Segment3, /) -> object:
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
def normalized(arg: numpy.ndarray, /) -> numpy.ndarray: ...

@overload
def normalized(arg: Arguments, /) -> object:
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
def normalized(arg: Segment, /) -> object:
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
def normalized(arg: Segment2, /) -> object:
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
def normalized(arg: Segment3, /) -> object:
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
def norm(arg: Arguments, /) -> object:
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
def norm(arg: Segment, /) -> object: ...

@overload
def norm(arg: Segment2, /) -> object: ...

@overload
def norm(arg: Segment3, /) -> object: ...

@overload
def norm(arg: VectorFunction, /) -> object: ...

@overload
def squared_norm(arg: Arguments, /) -> object:
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
def squared_norm(arg: Segment, /) -> object: ...

@overload
def squared_norm(arg: Segment2, /) -> object: ...

@overload
def squared_norm(arg: Segment3, /) -> object: ...

@overload
def squared_norm(arg: VectorFunction, /) -> object: ...

@overload
def cubed_norm(arg: Arguments, /) -> object:
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
def cubed_norm(arg: Segment, /) -> object: ...

@overload
def cubed_norm(arg: Segment2, /) -> object: ...

@overload
def cubed_norm(arg: Segment3, /) -> object: ...

@overload
def cubed_norm(arg: VectorFunction, /) -> object: ...

@overload
def inverse_norm(arg: Arguments, /) -> object:
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
def inverse_norm(arg: Segment, /) -> object: ...

@overload
def inverse_norm(arg: Segment2, /) -> object: ...

@overload
def inverse_norm(arg: Segment3, /) -> object: ...

@overload
def inverse_norm(arg: VectorFunction, /) -> object: ...

@overload
def inverse_squared_norm(arg: Arguments, /) -> object:
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
def inverse_squared_norm(arg: Segment, /) -> object: ...

@overload
def inverse_squared_norm(arg: Segment2, /) -> object: ...

@overload
def inverse_squared_norm(arg: Segment3, /) -> object: ...

@overload
def inverse_squared_norm(arg: VectorFunction, /) -> object: ...

@overload
def inverse_cubed_norm(arg: Arguments, /) -> object:
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
def inverse_cubed_norm(arg: Segment, /) -> object: ...

@overload
def inverse_cubed_norm(arg: Segment2, /) -> object: ...

@overload
def inverse_cubed_norm(arg: Segment3, /) -> object: ...

@overload
def inverse_cubed_norm(arg: VectorFunction, /) -> object: ...

@overload
def inverse_four_norm(arg: Arguments, /) -> object:
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
def inverse_four_norm(arg: Segment, /) -> object: ...

@overload
def inverse_four_norm(arg: Segment2, /) -> object: ...

@overload
def inverse_four_norm(arg: Segment3, /) -> object: ...

@overload
def inverse_four_norm(arg: VectorFunction, /) -> object: ...

@overload
def normalized_power2(arg: Arguments, /) -> object:
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
def normalized_power2(arg: Segment, /) -> object: ...

@overload
def normalized_power2(arg: Segment2, /) -> object: ...

@overload
def normalized_power2(arg: Segment3, /) -> object: ...

@overload
def normalized_power2(arg: VectorFunction, /) -> object: ...

@overload
def normalized_power3(arg: Arguments, /) -> object:
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
def normalized_power3(arg: Segment, /) -> object: ...

@overload
def normalized_power3(arg: Segment2, /) -> object: ...

@overload
def normalized_power3(arg: Segment3, /) -> object: ...

@overload
def normalized_power3(arg: VectorFunction, /) -> object: ...

@overload
def normalized_power4(arg: Arguments, /) -> object:
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
def normalized_power4(arg: Segment, /) -> object: ...

@overload
def normalized_power4(arg: Segment2, /) -> object: ...

@overload
def normalized_power4(arg: Segment3, /) -> object: ...

@overload
def normalized_power4(arg: VectorFunction, /) -> object: ...

@overload
def normalized_power5(arg: Arguments, /) -> object:
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
def normalized_power5(arg: Segment, /) -> object: ...

@overload
def normalized_power5(arg: Segment2, /) -> object: ...

@overload
def normalized_power5(arg: Segment3, /) -> object: ...

@overload
def normalized_power5(arg: VectorFunction, /) -> object: ...

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
    Double (iterated) cross product of three 3-vector quantities.

    Computes ``(a x b) x c``, where each argument must produce a 3-element output
    vector.

    Parameters
    ----------
    a, b, c : VectorFunction
        Three 3-vector-valued VectorFunctions.

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
    iteration variable and is updated in place during each evaluation; the remaining
    elements are differentiated-through parameters.

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
def sin(arg: Element, /) -> object:
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
def sin(arg: ScalarFunction, /) -> object: ...

@overload
def cos(arg: Element, /) -> object:
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
def cos(arg: ScalarFunction, /) -> object: ...

@overload
def tan(arg: Element, /) -> object:
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
def tan(arg: ScalarFunction, /) -> object: ...

@overload
def sqrt(arg: Element, /) -> object:
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
def sqrt(arg: ScalarFunction, /) -> object: ...

@overload
def exp(arg: Element, /) -> object:
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
def exp(arg: ScalarFunction, /) -> object: ...

@overload
def log(arg: Element, /) -> object:
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
        A new scalar VectorFunction evaluating ``ln(fun(x))``.
    """

@overload
def log(arg: ScalarFunction, /) -> object: ...

@overload
def squared(arg: Element, /) -> object:
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
def squared(arg: ScalarFunction, /) -> object: ...

@overload
def arcsin(arg: Element, /) -> object:
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
def arcsin(arg: ScalarFunction, /) -> object: ...

@overload
def arccos(arg: Element, /) -> object:
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
def arccos(arg: ScalarFunction, /) -> object: ...

@overload
def arctan(arg: Element, /) -> object:
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
def arctan(arg: ScalarFunction, /) -> object: ...

@overload
def sinh(arg: Element, /) -> object:
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
def sinh(arg: ScalarFunction, /) -> object: ...

@overload
def cosh(arg: Element, /) -> object:
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
def cosh(arg: ScalarFunction, /) -> object: ...

@overload
def tanh(arg: Element, /) -> object:
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
def tanh(arg: ScalarFunction, /) -> object: ...

@overload
def pow(arg: Element, /) -> object:
    """
    Elementwise real power of a scalar VectorFunction.

    Delegates to the ``.pow(exponent)`` method on *fun*. Prefer using the method
    form ``fun.pow(p)`` or the ``**`` operator directly, as the free-function form
    does not forward the exponent argument.

    Parameters
    ----------
    fun : VectorFunction
        Scalar-valued (output dimension 1) VectorFunction. The base value must be
        positive at all evaluation points for non-integer exponents.

    Returns
    -------
    VectorFunction
        A new scalar VectorFunction evaluating ``fun(x)**p``.
    """

@overload
def pow(arg: ScalarFunction, /) -> object: ...

@overload
def arcsinh(arg: Element, /) -> object:
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
def arcsinh(arg: ScalarFunction, /) -> object: ...

@overload
def arccosh(arg: Element, /) -> object:
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
def arccosh(arg: ScalarFunction, /) -> object: ...

@overload
def arctanh(arg: Element, /) -> object:
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
def arctanh(arg: ScalarFunction, /) -> object: ...

@overload
def sign(arg: Element, /) -> object:
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
def sign(arg: ScalarFunction, /) -> object: ...

@overload
def abs(arg: Element, /) -> object:
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
def abs(arg: ScalarFunction, /) -> object: ...

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
def ifelse(arg0: Conditional, arg1: float, arg2: ScalarFunction, /) -> ScalarFunction: ...

@overload
def ifelse(arg0: Conditional, arg1: ScalarFunction, arg2: float, /) -> ScalarFunction: ...

@overload
def ifelse(arg0: Conditional, arg1: float, arg2: float, /) -> ScalarFunction: ...

@overload
def ifelse(arg0: Conditional, arg1: VectorFunction, arg2: VectorFunction, /) -> VectorFunction: ...

@overload
def ifelse(arg0: Conditional, arg1: numpy.ndarray, arg2: VectorFunction, /) -> VectorFunction: ...

@overload
def ifelse(arg0: Conditional, arg1: VectorFunction, arg2: numpy.ndarray, /) -> VectorFunction: ...

@overload
def ifelse(arg0: Conditional, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> VectorFunction: ...

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
    def __init__(self, arg: Sequence[VectorFunction], /) -> None: ...

    @overload
    def __mul__(self, arg: ColMatrix, /) -> ColMatrix:
        """
        Matrix product of two column-major matrix VectorFunctions (``self @ other``).

        Computes ``M1(x) @ M2(x)`` where both factors are evaluated at the same input
        ``x``.  The inner dimension of ``M1`` must equal the outer dimension of ``M2``.

        Parameters
        ----------
        other : ColMatrix or RowMatrix or VectorFunction or float
            Right-hand factor.  When ``other`` is a plain VectorFunction it is treated
            as a column vector; when it is a ``float`` every element is scaled by that
            constant.

        Returns
        -------
        ColMatrix or VectorFunction
            Column-major matrix (or flat VectorFunction when ``other`` is a column
            vector) representing the product.

        Raises
        ------
        ValueError
            If the inner dimensions of the two matrix functions do not match.
        """

    @overload
    def __mul__(self, arg: RowMatrix, /) -> ColMatrix: ...

    @overload
    def __mul__(self, arg: float, /) -> ColMatrix: ...

    @overload
    def __mul__(self, arg: VectorFunction, /) -> VectorFunction: ...

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
    def __add__(self, arg: ColMatrix, /) -> ColMatrix: ...

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
        of ``self``.  The transposed matrix shares the same backing VectorFunction;
        no copy of the output data is made at evaluation time.

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
    def __init__(self, arg: Sequence[VectorFunction], /) -> None: ...

    @overload
    def __mul__(self, arg: ColMatrix, /) -> ColMatrix:
        """
        Matrix product of two row-major matrix VectorFunctions (``self @ other``).

        Computes ``M1(x) @ M2(x)`` where both factors are evaluated at the same input
        ``x``.  The inner dimension of ``M1`` must equal the outer dimension of ``M2``.

        Parameters
        ----------
        other : ColMatrix or RowMatrix or VectorFunction or float
            Right-hand factor.  When ``other`` is a plain VectorFunction it is treated
            as a column vector; when it is a ``float`` every element is scaled by that
            constant.

        Returns
        -------
        ColMatrix or RowMatrix
            Column-major matrix (or row-major when both operands are RowMatrix)
            representing the product.

        Raises
        ------
        ValueError
            If the inner dimensions of the two matrix functions do not match.
        """

    @overload
    def __mul__(self, arg: RowMatrix, /) -> ColMatrix: ...

    @overload
    def __mul__(self, arg: float, /) -> RowMatrix: ...

    @overload
    def __mul__(self, arg: VectorFunction, /) -> VectorFunction: ...

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
        same shape.

        Parameters
        ----------
        other : RowMatrix or array_like, shape (rows, cols)
            Right-hand summand.  When ``other`` is a constant numpy matrix it is
            broadcast as a fixed offset.

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
    def __add__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> RowMatrix: ...

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
        of ``self``.  The transposed matrix shares the same backing VectorFunction;
        no copy of the output data is made at evaluation time.

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
def matmul(arg0: ColMatrix, arg1: RowMatrix, /) -> ColMatrix: ...

@overload
def matmul(arg0: ColMatrix, arg1: VectorFunction, /) -> VectorFunction: ...

@overload
def matmul(arg0: ColMatrix, arg1: numpy.ndarray, /) -> VectorFunction: ...

@overload
def matmul(arg0: RowMatrix, arg1: ColMatrix, /) -> ColMatrix: ...

@overload
def matmul(arg0: RowMatrix, arg1: RowMatrix, /) -> ColMatrix: ...

@overload
def matmul(arg0: RowMatrix, arg1: VectorFunction, /) -> VectorFunction: ...

@overload
def matmul(arg0: RowMatrix, arg1: numpy.ndarray, /) -> VectorFunction: ...

@overload
def matmul(arg0: Annotated[NDArray[numpy.float64], dict(shape=(2, 2), order='F')], arg1: VectorFunction, /) -> VectorFunction: ...

@overload
def matmul(arg0: Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')], arg1: VectorFunction, /) -> VectorFunction: ...

@overload
def matmul(arg0: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], arg1: VectorFunction, /) -> VectorFunction: ...

class Conditional:
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
    def ifelse(self, arg0: float, arg1: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def ifelse(self, arg0: ScalarFunction, arg1: float, /) -> ScalarFunction: ...

    @overload
    def ifelse(self, arg0: float, arg1: float, /) -> ScalarFunction: ...

    @overload
    def ifelse(self, arg0: VectorFunction, arg1: VectorFunction, /) -> VectorFunction: ...

    @overload
    def ifelse(self, arg0: numpy.ndarray, arg1: VectorFunction, /) -> VectorFunction: ...

    @overload
    def ifelse(self, arg0: VectorFunction, arg1: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def ifelse(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> VectorFunction: ...

class Comparative:
    def compute(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> bool:
        """
        Evaluate the comparative function at a given input vector.

        Parameters
        ----------
        x : array_like, shape (input_rows,)
            Input vector at which to evaluate the min/max function.

        Returns
        -------
        numpy.ndarray, shape (output_rows,)
            Output of the comparative (min or max) function at *x*.
        """

    @overload
    def max(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Return a VectorFunction that evaluates the elementwise maximum of two scalar functions.

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
    def max(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def max(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def max(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def max(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def max(self, arg: numpy.ndarray, /) -> VectorFunction: ...

    @overload
    def min(self, arg: ScalarFunction, /) -> ScalarFunction:
        """
        Return a VectorFunction that evaluates the elementwise minimum of two scalar functions.

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
    def min(self, arg: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def min(self, arg: float, /) -> ScalarFunction: ...

    @overload
    def min(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def min(self, arg: VectorFunction, /) -> VectorFunction: ...

    @overload
    def min(self, arg: numpy.ndarray, /) -> VectorFunction: ...

class PyVectorFunction:
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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def thread_safe(self) -> bool: ...

    @thread_safe.setter
    def thread_safe(self, arg: bool, /) -> None: ...

class PyScalarFunction:
    def __init__(self, i_rows: int, func: object, jstepsize: float = 1e-06, hstepsize: float = 0.0001, args: tuple = ()) -> None: ...

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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def thread_safe(self) -> bool: ...

    @thread_safe.setter
    def thread_safe(self, arg: bool, /) -> None: ...

class NumbaVectorFunction:
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: float, arg4: float, /) -> None:
        """
        Wrap a Numba JIT-compiled function pointer as a thread-safe VectorFunction.

        The function pointer must have the C signature
        ``void f(double* x, double* fx, int i_rows, int o_rows)`` and is passed as an
        integer holding its address (as returned by ``numba.extending.get_cython_function_address``
        or equivalent).  Derivatives are approximated by forward finite differences with
        the given step sizes.  Unlike ``PyVectorFunction``, evaluation does not hold the
        Python GIL and is safe to call from multiple threads in parallel.

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
    def thread_safe(self) -> bool: ...

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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: float, arg4: float, /) -> None:
        """
        Wrap a Numba JIT-compiled function pointer as a thread-safe VectorFunction.

        The function pointer must have the C signature
        ``void f(double* x, double* fx, int i_rows, int o_rows)`` and is passed as an
        integer holding its address (as returned by ``numba.extending.get_cython_function_address``
        or equivalent).  Derivatives are approximated by forward finite differences with
        the given step sizes.  Unlike ``PyVectorFunction``, evaluation does not hold the
        Python GIL and is safe to call from multiple threads in parallel.

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
    def thread_safe(self) -> bool: ...

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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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

        Raises
        ------
        ValueError
            If the length of ``input_scales`` or ``output_scales`` does not match the
            corresponding dimension of ``func``.
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
        Whether this VectorFunction is known to be linear at compile time.

        A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
        this flag to skip second-derivative computations for linear expressions.

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
    def compute(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> numpy.ndarray:
        """
        Evaluate the function at a point (``f(x)``).

        Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
        argument for functional composition — see :meth:`eval` for that overload.

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
    def __call__(self, arg: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def jacobian(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def compute_jacobian(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
        """
        Evaluate the function and its Jacobian simultaneously.

        Computing both at once is cheaper than two separate calls because internal
        temporary buffers are shared.

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
    def compute_jacobian(self, arg: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    def adjointgradient(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> numpy.ndarray: ...

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
    def adjointhessian(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

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
    def computeall(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> tuple[numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], numpy.ndarray, Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]: ...

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
    @overload
    def __init__(self, ts: numpy.ndarray, vs: numpy.ndarray, kind: str = 'cubic') -> None: ...

    @overload
    def __init__(self, ts: numpy.ndarray, vs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], axis: int = 0, kind: str = 'cubic') -> None: ...

    @overload
    def __init__(self, vts: list[numpy.ndarray], tvar: int = -1, kind: str = 'cubic') -> None: ...

    @overload
    def __init__(self, ts: numpy.ndarray, vs: numpy.ndarray, kind: InterpType = InterpType.Cubic) -> None: ...

    @overload
    def __init__(self, ts: numpy.ndarray, vs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], axis: int = 0, kind: InterpType = InterpType.Cubic) -> None: ...

    @overload
    def __init__(self, vts: list[numpy.ndarray], tvar: int = -1, kind: InterpType = InterpType.Cubic) -> None: ...

    @overload
    def interp(self, arg: float, /) -> numpy.ndarray: ...

    @overload
    def interp(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def __call__(self, arg: float, /) -> numpy.ndarray: ...

    @overload
    def __call__(self, arg: numpy.ndarray, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @overload
    def __call__(self, arg: ScalarFunction, /) -> object: ...

    @overload
    def __call__(self, arg: Element, /) -> object: ...

    def interp_deriv1(self, arg: float, /) -> tuple[numpy.ndarray, numpy.ndarray]: ...

    def interp_deriv2(self, arg: float, /) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ...

    def sf(self) -> ScalarFunction: ...

    def vf(self) -> VectorFunction: ...

class InterpTable2D:
    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, z: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], kind: str = 'cubic') -> None: ...

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, z: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], kind: InterpType = InterpType.Cubic) -> None: ...

    @overload
    def interp(self, arg0: float, arg1: float, /) -> float: ...

    @overload
    def interp(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')]: ...

    def interp_deriv1(self, arg0: float, arg1: float, /) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')]]: ...

    def interp_deriv2(self, arg0: float, arg1: float, /) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')], Annotated[NDArray[numpy.float64], dict(shape=(2, 2), order='F')]]: ...

    def find_elem(self, arg0: numpy.ndarray, arg1: float, /) -> int: ...

    @overload
    def __call__(self, arg0: float, arg1: float, /) -> float: ...

    @overload
    def __call__(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')]: ...

    @overload
    def __call__(self, arg0: ScalarFunction, arg1: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg0: Element, arg1: Element, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: Segment2, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: VectorFunction, /) -> ScalarFunction: ...

    def sf(self) -> ScalarFunction: ...

    def vf(self) -> VectorFunction: ...

class InterpTable3D:
    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, zs: numpy.ndarray, fs: numpy.ndarray, kind: str = 'cubic', cache: bool = False) -> None: ...

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, zs: numpy.ndarray, fs: numpy.ndarray, kind: InterpType = InterpType.Cubic, cache: bool = False) -> None: ...

    def interp(self, arg0: float, arg1: float, arg2: float, /) -> float: ...

    def interp_deriv1(self, arg0: float, arg1: float, arg2: float, /) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]]: ...

    def interp_deriv2(self, arg0: float, arg1: float, arg2: float, /) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')]]: ...

    @overload
    def __call__(self, arg0: float, arg1: float, arg2: float, /) -> float: ...

    @overload
    def __call__(self, arg0: ScalarFunction, arg1: ScalarFunction, arg2: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg0: Element, arg1: Element, arg2: Element, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: Segment3, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: VectorFunction, /) -> ScalarFunction: ...

    def sf(self) -> ScalarFunction: ...

    def vf(self) -> VectorFunction: ...

class InterpTable4D:
    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, zs: numpy.ndarray, ws: numpy.ndarray, fs: numpy.ndarray, kind: str = 'cubic', cache: bool = False) -> None: ...

    @overload
    def __init__(self, xs: numpy.ndarray, ys: numpy.ndarray, zs: numpy.ndarray, ws: numpy.ndarray, fs: numpy.ndarray, kind: InterpType = InterpType.Cubic, cache: bool = False) -> None: ...

    def interp(self, arg0: float, arg1: float, arg2: float, arg3: float, /) -> float: ...

    def interp_deriv1(self, arg0: float, arg1: float, arg2: float, arg3: float, /) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')]]: ...

    def interp_deriv2(self, arg0: float, arg1: float, arg2: float, arg3: float, /) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')], Annotated[NDArray[numpy.float64], dict(shape=(4, 4), order='F')]]: ...

    @overload
    def __call__(self, arg0: float, arg1: float, arg2: float, arg3: float, /) -> float: ...

    @overload
    def __call__(self, arg0: ScalarFunction, arg1: ScalarFunction, arg2: ScalarFunction, arg3: ScalarFunction, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg0: Element, arg1: Element, arg2: Element, arg3: Element, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: Segment, /) -> ScalarFunction: ...

    @overload
    def __call__(self, arg: VectorFunction, /) -> ScalarFunction: ...

    def sf(self) -> ScalarFunction: ...

    def vf(self) -> VectorFunction: ...

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
