"""SubModule Containing Optimal Control ODEs, Phases, and Utilities"""

from collections.abc import Sequence
import enum
from typing import Annotated, Union, overload

import numpy
from numpy.typing import NDArray
from . import (
    ode_6 as ode_6,
    ode_6_3 as ode_6_3,
    ode_7_3 as ode_7_3,
    ode_x as ode_x,
    ode_x_u as ode_x_u,
    ode_x_u_p as ode_x_u_p
)
import _tychopy.solvers
import _tychopy.vector_functions


class IVPAlg(enum.Enum):
    DOPRI54 = 0

    DOPRI87 = 1

    Tsit5 = 2

    BS3 = 3

    BS5 = 4

    Vern7 = 5

    Vern8 = 6

    Vern9 = 7

class IVPController(enum.Enum):
    I = 0

    PI = 1

    PID = 2

class ErrorNormType(enum.Enum):
    RMS = 0

    MAX = 1

class TranscriptionModes(enum.Enum):
    """
    Collocation/transcription scheme used to discretize a phase.

    Passed (as an enum or its string name, e.g. ``"LGL3"``) when constructing
    a phase to select how the continuous dynamics are converted into a
    finite-dimensional set of collocation constraints.
    """

    LGL3 = 0
    """3rd-order Legendre-Gauss-Lobatto collocation."""

    LGL5 = 1
    """5th-order Legendre-Gauss-Lobatto collocation."""

    LGL7 = 2
    """7th-order Legendre-Gauss-Lobatto collocation."""

    Trapezoidal = 3
    """Trapezoidal (2nd-order) collocation."""

    CentralShooting = 4
    """Central (forward/backward) shooting transcription."""

class IntegralModes(enum.Enum):
    """Quadrature rule used to evaluate integral terms over a phase."""

    BaseIntegral = 0
    """Default quadrature matched to the transcription order."""

    TrapIntegral = 2
    """Trapezoidal-rule quadrature."""

class ControlModes(enum.Enum):
    """Interpolation/representation scheme for the control history."""

    HighestOrderSpline = 0
    """Spline matching the transcription's highest order."""

    FirstOrderSpline = 1
    """Piecewise-linear (first-order) control spline."""

    NoSpline = 2
    """Independent control values, no inter-node spline."""

    BlockConstant = 3
    """Control held constant over each transcription block."""

class PhaseRegionFlags(enum.Enum):
    """
    Region of a phase that a state function (constraint or objective) acts on.

    Selects which discretized state(s) -- endpoints, interior nodes, or
    parameter blocks -- are passed as arguments to a state function when it is
    attached to a phase. Most phase methods accept either the enum value or its
    string name (e.g. ``"Front"``, ``"Path"``).
    """

    Front = 1
    """The first (initial-time) node of the phase."""

    Back = 2
    """The last (final-time) node of the phase."""

    Path = 5
    """Every node of the phase (full path constraint)."""

    NodalPath = 7
    """Every nodal (mesh) point along the path."""

    FrontandBack = 3
    """Both endpoints, front state followed by back state."""

    BackandFront = 4
    """Both endpoints, back state followed by front state."""

    Params = 12
    """The combined parameter vector of the phase."""

    InnerPath = 6
    """Every interior node, excluding the two endpoints."""

    ODEParams = 13
    """The ODE (dynamics) parameter sub-vector."""

    StaticParams = 14
    """The static (non-ODE) parameter sub-vector."""

    PairWisePath = 9
    """Each consecutive pair of nodes along the path."""

class ScaleModes(enum.Enum):
    """Source of the variable/equation scaling applied to a phase."""

    AUTO = 0
    """Scales are computed automatically from the problem."""

    CUSTOM = 1
    """Scales are supplied explicitly by the user."""

    NONE = 2
    """No scaling is applied (unit scales)."""

class MeshErrorEstimators(enum.Enum):
    """Method used to estimate the per-interval mesh (discretization) error."""

    DEBOOR = 0
    """de Boor collocation-residual error estimate."""

    INTEGRATOR = 1
    """Error estimate from a reference numerical integrator."""

class MeshErrorAggregation(enum.Enum):
    """Reduction used to aggregate per-interval mesh errors into a scalar."""

    MAX = 0
    """Maximum error over all intervals."""

    AVG = 1
    """Arithmetic mean of per-interval errors."""

    GEOMETRIC = 2
    """Geometric mean of per-interval errors."""

    ENDTOEND = 3
    """End-to-end (accumulated) trajectory error."""

def strto_phase_region_flag(arg: str, /) -> PhaseRegionFlags: ...

class StateConstraint:
    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, arg3: numpy.ndarray, arg4: numpy.ndarray, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, arg3: PhaseRegionFlags, arg4: numpy.ndarray, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, /) -> None: ...

class StateObjective:
    @overload
    def __init__(self, arg0: _tychopy.vector_functions.ScalarFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, arg3: numpy.ndarray, arg4: numpy.ndarray, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.ScalarFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, arg3: PhaseRegionFlags, arg4: numpy.ndarray, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.ScalarFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, /) -> None: ...

class MeshIterateInfo:
    """
    Per-iteration diagnostics of an adaptive-mesh refinement pass.

    One :class:`MeshIterateInfo` is recorded per mesh-refinement iteration of a
    phase; the full history is returned by ``phase.get_mesh_iters()``. The arrays
    describe the node times, the per-segment error estimate, and the resulting
    error distribution that drives the next round of node redistribution.
    """

    @property
    def times(self) -> numpy.ndarray:
        """
        numpy.ndarray, shape (numsegs + 1,): node times of the mesh for this iteration, normalized to ``[0, 1]``.
        """

    @property
    def error(self) -> numpy.ndarray:
        """
        numpy.ndarray, shape (numsegs,): per-segment estimated discretization error.
        """

    @property
    def distribution(self) -> numpy.ndarray:
        """
        numpy.ndarray: error-density distribution over the mesh, used to redistribute nodes for the next iteration.
        """

    @property
    def distintegral(self) -> numpy.ndarray:
        """numpy.ndarray: normalized cumulative integral of :attr:`distribution`."""

    @property
    def avg_error(self) -> float:
        """float: time-weighted average per-segment error."""

    @property
    def max_error(self) -> float:
        """float: maximum per-segment error over the mesh."""

    @property
    def numsegs(self) -> int:
        """int: number of segments in this iteration's mesh."""

    @property
    def converged(self) -> bool:
        """bool: whether the mesh met the error tolerance on this iteration."""

class EventPack:
    @overload
    def __init__(self, vf: _tychopy.vector_functions.ScalarFunction, direction: int = 0, stop_count: int = 0) -> None: ...

    @overload
    def __init__(self, t: tuple) -> None: ...

    @property
    def vf(self) -> _tychopy.vector_functions.ScalarFunction: ...

    @vf.setter
    def vf(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> None: ...

    @property
    def direction(self) -> int: ...

    @direction.setter
    def direction(self, arg: int, /) -> None: ...

    @property
    def stop_count(self) -> int: ...

    @stop_count.setter
    def stop_count(self, arg: int, /) -> None: ...

class PhaseInterface(_tychopy.solvers.OptimizationProblemBase):
    """
    A single phase of an optimal control problem.

    A phase couples an ODE, a discretized trajectory, and the boundary values,
    bounds, constraints, and objectives that define a direct-collocation problem.
    Phases are not constructed directly; build one from an ODE with
    ``ode.phase(transcription_mode, traj, num_segments)`` (see
    :class:`~tychopy.optimal_control.ODEBase`), then attach problem data with the
    ``add_*`` methods below.

    Variables are addressed within a region of the phase. A *region* (see
    :class:`PhaseRegionFlags`) selects which discretized state(s) a function or
    bound applies to -- e.g. ``"Front"`` (initial node), ``"Back"`` (final node),
    or ``"Path"`` (every node). Region selectors accept either the enum value or
    its string name. Variable indices refer to the packed node layout
    ``[x, t, u, p]``.

    Once configured, solve the phase with :meth:`optimize` (or :meth:`solve`),
    then retrieve the result with :meth:`return_traj`.

    Notes
    -----
    :meth:`optimize`, :meth:`solve`, and the ``optimizer`` handle are inherited
    from the shared optimization-problem base; ``phase.optimizer`` exposes the
    underlying PSIOPT solver (e.g. ``phase.optimizer.set_print_level(0)``).

    Examples
    --------
    >>> phase = ode.phase("LGL3", Xs, 32)
    >>> phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
    >>> phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
    >>> phase.add_boundary_value("Back", [0, 1], [xf, yf])
    >>> phase.add_delta_time_objective(1.0)
    >>> phase.optimize()
    >>> traj = phase.return_traj()
    """

    def enable_vectorization(self, arg: bool, /) -> None: ...

    @overload
    def set_traj(self, arg0: list[numpy.ndarray], arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def set_traj(self, arg0: list[numpy.ndarray], arg1: numpy.ndarray, arg2: numpy.ndarray, arg3: bool, /) -> None: ...

    @overload
    def set_traj(self, arg0: list[numpy.ndarray], arg1: int, /) -> None: ...

    @overload
    def set_traj(self, arg0: list[numpy.ndarray], arg1: int, arg2: bool, /) -> None: ...

    @overload
    def set_traj(self, arg: list[numpy.ndarray], /) -> None:
        """
        Set (or replace) the phase's initial-guess trajectory.

        Parameters
        ----------
        mesh : list of numpy.ndarray
            Trajectory nodes, each packed as ``[x, t, u, p]``, in increasing time
            order. This becomes the initial guess that collocation refines.

        Notes
        -----
        Additional overloads accept a defect-segment count or an explicit
        defects-binding/defects-per-block specification; see the C++ reference.
        """

    @overload
    def switch_transcription_mode(self, arg0: TranscriptionModes, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg: TranscriptionModes, /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg0: str, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg: str, /) -> None: ...

    def transcribe(self, arg0: bool, arg1: bool, /) -> None:
        """
        Transcribe the phase into its underlying nonlinear program.

        Builds the collocation NLP (defect constraints, user constraints, and
        objectives) from the current configuration. This is performed automatically
        by :meth:`optimize`/:meth:`solve`; call it explicitly only to inspect the
        transcribed problem or to force a rebuild.

        Parameters
        ----------
        showstats : bool
            Print transcription statistics (variable/constraint counts).
        showfuns : bool
            Print the set of transcribed functions.
        """

    @overload
    def refine_traj_manual(self, arg: int, /) -> None:
        """
        Resample the current trajectory onto a new number of segments.

        Parameters
        ----------
        num : int
            New segment count to interpolate the existing trajectory onto.
        """

    @overload
    def refine_traj_manual(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> None:
        """
        Resample the current trajectory onto an explicit mesh specification.

        Parameters
        ----------
        bin_times : numpy.ndarray
            Normalized bin boundary times in ``[0, 1]``.
        bin_segments : numpy.ndarray of int
            Number of segments to place in each bin.
        """

    def refine_traj_equal(self, arg: int, /) -> list[numpy.ndarray]:
        """Resample the current trajectory onto an equally-spaced mesh."""

    @overload
    def set_static_params(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> None: ...

    @overload
    def set_static_params(self, arg: numpy.ndarray, /) -> None: ...

    @overload
    def add_static_params(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> None: ...

    @overload
    def add_static_params(self, arg: numpy.ndarray, /) -> None: ...

    def add_static_param_vgroups(self, arg: "std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, Eigen::Matrix<int, -1, 1, 0, -1, 1> > > >", /) -> None: ...

    def set_static_param_vgroups(self, arg: "std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, Eigen::Matrix<int, -1, 1, 0, -1, 1> > > >", /) -> None: ...

    @overload
    def add_static_param_vgroup(self, arg0: numpy.ndarray, arg1: str, /) -> None: ...

    @overload
    def add_static_param_vgroup(self, arg0: int, arg1: str, /) -> None: ...

    @overload
    def set_control_mode(self, arg: ControlModes, /) -> None: ...

    @overload
    def set_control_mode(self, arg: str, /) -> None: ...

    def set_integral_mode(self, arg: IntegralModes, /) -> None: ...

    def sub_static_params(self, arg: numpy.ndarray, /) -> None: ...

    @overload
    def sub_variables(self, arg0: PhaseRegionFlags, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def sub_variables(self, arg0: str, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def sub_variable(self, arg0: PhaseRegionFlags, arg1: int, arg2: float, /) -> None: ...

    @overload
    def sub_variable(self, arg0: str, arg1: int, arg2: float, /) -> None: ...

    def return_traj(self) -> list[numpy.ndarray]:
        """
        Return the current discretized trajectory.

        Returns
        -------
        list of numpy.ndarray
            One packed state vector ``[x, t, u, p]`` per node, in time order. After a
            successful :meth:`optimize`/:meth:`solve` this is the optimized solution.

        Examples
        --------
        >>> phase.optimize()
        >>> traj = phase.return_traj()
        """

    def return_traj_range(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]:
        """
        Return the trajectory resampled to ``num`` points over a time range.

        Parameters
        ----------
        num : int
            Number of output samples.
        tl : float
            Lower time bound.
        th : float
            Upper time bound.

        Returns
        -------
        list of numpy.ndarray
            ``num`` interpolated node vectors spanning ``[tl, th]``.
        """

    def return_traj_range_nd(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]: ...

    def return_traj_table(self) -> LGLInterpTable: ...

    def return_costate_traj(self) -> list[numpy.ndarray]:
        """
        Return the costate (adjoint/dual) trajectory.

        Returns
        -------
        list of numpy.ndarray
            One costate vector per node, recovered from the dynamics-defect
            Lagrange multipliers of the converged solution. Only meaningful after a
            successful :meth:`optimize`/:meth:`solve`.
        """

    def return_traj_error(self) -> list[numpy.ndarray]: ...

    def return_u_spline_con_lmults(self) -> list[numpy.ndarray]: ...

    def return_u_spline_con_vals(self) -> list[numpy.ndarray]: ...

    def return_equal_con_lmults(self, arg: int, /) -> list[numpy.ndarray]: ...

    def return_equal_con_vals(self, arg: int, /) -> list[numpy.ndarray]: ...

    def return_equal_con_scales(self, arg: int, /) -> numpy.ndarray: ...

    def return_inequal_con_lmults(self, arg: int, /) -> list[numpy.ndarray]: ...

    def return_inequal_con_vals(self, arg: int, /) -> list[numpy.ndarray]: ...

    def return_inequal_con_scales(self, arg: int, /) -> numpy.ndarray: ...

    def return_integral_objective_scales(self, arg: int, /) -> numpy.ndarray: ...

    def return_integral_param_function_scales(self, arg: int, /) -> numpy.ndarray: ...

    def return_state_objective_scales(self, arg: int, /) -> numpy.ndarray: ...

    def return_ode_output_scales(self) -> numpy.ndarray: ...

    def return_static_params(self) -> numpy.ndarray: ...

    def test_partitions(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    def remove_equal_con(self, arg: int, /) -> None: ...

    def remove_inequal_con(self, arg: int, /) -> None: ...

    def remove_state_objective(self, arg: int, /) -> None: ...

    def remove_integral_objective(self, arg: int, /) -> None: ...

    def remove_integral_param_function(self, arg: int, /) -> None: ...

    @overload
    def add_equal_con(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.VectorFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_equal_con(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.VectorFunction, input_index: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_equal_con(self, arg: StateConstraint, /) -> int: ...

    def add_boundary_value(self, phase_region: PhaseRegionFlags | str, index: int | Sequence[int] | str | list[str], value: float | numpy.ndarray | Sequence[float], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Fix selected region variables to given values (a boundary-value constraint).

        Adds an equality constraint pinning the named variables -- evaluated in the
        given region -- to ``value``. This is the usual way to set initial/final
        conditions.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variables are read from, e.g. ``"Front"`` for the initial node
            or ``"Back"`` for the final node.
        index : int or sequence of int
            Variable index or indices (into the packed ``[x, t, u, p]`` layout) to fix.
        value : float or numpy.ndarray
            Target value(s). A scalar is broadcast to all selected variables; a vector
            must match the number of selected indices.
        auto_scale : ScaleModes or str, optional
            Constraint output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``).
            Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.

        Examples
        --------
        >>> phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
        >>> phase.add_boundary_value("Back", [0, 1], [xf, yf])
        """

    def add_delta_var_equal_con(self, var: int | Sequence[int] | str | list[str], value: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_delta_time_equal_con(self, value: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_value_lock(self, reg: PhaseRegionFlags | str, vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_periodicity_con(self, vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_inequal_con(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.VectorFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_inequal_con(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.VectorFunction, input_index: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_inequal_con(self, arg: StateConstraint, /) -> int: ...

    def add_lu_var_bound(self, phase_region: PhaseRegionFlags | str, var: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound a variable from below and above (an inequality constraint).

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variable is read from, e.g. ``"Path"`` to bound it at every node.
        var : int
            Variable index (into the packed ``[x, t, u, p]`` layout) to bound.
        lowerbound : float
            Lower bound.
        upperbound : float
            Upper bound.
        scale : float, optional
            Constraint scaling applied to both the lower and upper bound. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.

        Examples
        --------
        >>> phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
        """

    def add_lower_var_bound(self, phase_region: PhaseRegionFlags | str, var: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound a variable from below (an inequality constraint).

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variable is read from.
        var : int
            Variable index (into the packed ``[x, t, u, p]`` layout) to bound.
        lowerbound : float
            Lower bound.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_upper_var_bound(self, phase_region: PhaseRegionFlags | str, var: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound a variable from above (an inequality constraint).

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variable is read from.
        var : int
            Variable index (into the packed ``[x, t, u, p]`` layout) to bound.
        upperbound : float
            Upper bound.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    @overload
    def add_lu_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_lu_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_lower_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_lower_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_upper_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_upper_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_up_vars: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_lu_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_lu_squared_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_lower_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_lower_squared_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_upper_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_upper_squared_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_lower_delta_var_bound(self, var: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_lower_delta_time_bound(self, lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_upper_delta_var_bound(self, var: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_upper_delta_time_bound(self, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_state_objective(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_state_objective(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, input_index: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add a (terminal/boundary) objective evaluated at a region of the phase.

        Adds the scalar value of ``func``, evaluated over the selected variables in
        the given region, to the problem objective (a Mayer-type term).

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the function is evaluated over, e.g. ``"Back"`` for a terminal cost.
        func : VectorFunction
            Scalar-valued function of the selected variables.
        input_index : int or sequence of int
            Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
            ``func``.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new objective.
        """

    @overload
    def add_state_objective(self, arg: StateObjective, /) -> int: ...

    def add_value_objective(self, phase_region: PhaseRegionFlags | str, var: int | Sequence[int] | str | list[str], scale: float, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an objective equal to a scaled single variable at a region.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variable is read from.
        var : int
            Variable index (into the packed ``[x, t, u, p]`` layout).
        scale : float
            Multiplier applied to the variable's value in the objective. Use a
            negative value to maximize the variable.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new objective.
        """

    def add_delta_var_objective(self, var: int | Sequence[int] | str | list[str], scale: float, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an objective on the front-to-back change of a variable.

        Parameters
        ----------
        var : int
            Variable index (into the packed ``[x, t, u, p]`` layout).
        scale : float
            Multiplier applied to the change (back value minus front value).
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new objective.
        """

    def add_delta_time_objective(self, scale: float, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an objective on the total elapsed time of the phase.

        A scaled minimum-time objective: minimizes ``scale * (t_back - t_front)``.

        Parameters
        ----------
        scale : float
            Multiplier applied to the elapsed time. Use ``1.0`` for a minimum-time
            objective.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new objective.

        Examples
        --------
        >>> phase.add_delta_time_objective(1.0)
        """

    @overload
    def add_integral_objective(self, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_integral_objective(self, func: _tychopy.vector_functions.ScalarFunction, input_index: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add a running-cost (Lagrange) objective integrated over the phase.

        Adds the time integral of ``func`` -- evaluated along the trajectory -- to the
        problem objective.

        Parameters
        ----------
        func : VectorFunction
            Scalar-valued integrand, a function of the selected variables.
        input_index : int or sequence of int
            Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
            ``func`` at each node.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new objective.
        """

    @overload
    def add_integral_objective(self, arg: StateObjective, /) -> int: ...

    @overload
    def add_integral_param_function(self, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], int_param: int, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_integral_param_function(self, func: _tychopy.vector_functions.ScalarFunction, input_index: int | Sequence[int] | str | list[str], int_param: int, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_integral_param_function(self, arg0: StateObjective, arg1: int, /) -> int: ...

    @overload
    def add_lu_var_bounds(self, arg0: PhaseRegionFlags, arg1: numpy.ndarray, arg2: float, arg3: float, arg4: float, /) -> numpy.ndarray: ...

    @overload
    def add_lu_var_bounds(self, arg0: str, arg1: numpy.ndarray, arg2: float, arg3: float, arg4: float, /) -> numpy.ndarray: ...

    def get_mesh_info(self, arg0: bool, arg1: int, /) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ...

    def refine_traj_auto(self) -> None:
        """
        Run one automatic mesh-refinement step.

        Estimates the per-interval discretization error and redistributes mesh nodes
        to drive it below :attr:`mesh_tol`. This is invoked automatically when
        :attr:`adaptive_mesh` is enabled; call it directly for manual control of the
        refinement loop.
        """

    def calc_global_error(self) -> numpy.ndarray:
        """
        Estimate the global (end-to-end) discretization error of the trajectory.

        Returns
        -------
        numpy.ndarray
            Per-state global error estimate of the current trajectory.
        """

    def get_mesh_iters(self) -> list[MeshIterateInfo]:
        """
        Return the per-iteration adaptive-mesh refinement history.

        Returns
        -------
        list of MeshIterateInfo
            One :class:`MeshIterateInfo` record per mesh-refinement iteration that was
            performed during the last solve.
        """

    @property
    def adaptive_mesh(self) -> bool: ...

    @adaptive_mesh.setter
    def adaptive_mesh(self, arg: bool, /) -> None: ...

    @property
    def auto_scaling(self) -> bool: ...

    @auto_scaling.setter
    def auto_scaling(self, arg: bool, /) -> None: ...

    @property
    def sync_objective_scales(self) -> bool: ...

    @sync_objective_scales.setter
    def sync_objective_scales(self, arg: bool, /) -> None: ...

    def set_auto_scaling(self, auto_scaling: bool = True) -> None: ...

    def set_adaptive_mesh(self, adaptive_mesh: bool = True) -> None: ...

    @overload
    def set_units(self, **kwargs) -> None: ...

    @overload
    def set_units(self, arg: numpy.ndarray, /) -> None: ...

    def set_mesh_tol(self, arg: float, /) -> None: ...

    def set_mesh_red_factor(self, arg: float, /) -> None: ...

    def set_mesh_inc_factor(self, arg: float, /) -> None: ...

    def set_mesh_err_factor(self, arg: float, /) -> None: ...

    def set_max_mesh_iters(self, arg: int, /) -> None: ...

    def set_min_segments(self, arg: int, /) -> None: ...

    def set_max_segments(self, arg: int, /) -> None: ...

    @overload
    def set_mesh_error_criteria(self, arg: MeshErrorAggregation, /) -> None: ...

    @overload
    def set_mesh_error_criteria(self, arg: str, /) -> None: ...

    @overload
    def set_mesh_error_estimator(self, arg: MeshErrorEstimators, /) -> None: ...

    @overload
    def set_mesh_error_estimator(self, arg: str, /) -> None: ...

    @overload
    def set_mesh_error_distributor(self, arg: MeshErrorAggregation, /) -> None: ...

    @overload
    def set_mesh_error_distributor(self, arg: str, /) -> None: ...

    @property
    def print_mesh_info(self) -> bool: ...

    @print_mesh_info.setter
    def print_mesh_info(self, arg: bool, /) -> None: ...

    @property
    def max_mesh_iters(self) -> int: ...

    @max_mesh_iters.setter
    def max_mesh_iters(self, arg: int, /) -> None: ...

    @property
    def mesh_tol(self) -> float: ...

    @mesh_tol.setter
    def mesh_tol(self, arg: float, /) -> None: ...

    @property
    def mesh_error_estimator(self) -> MeshErrorEstimators: ...

    @mesh_error_estimator.setter
    def mesh_error_estimator(self, arg: object, /) -> None: ...

    @property
    def mesh_error_criteria(self) -> MeshErrorAggregation: ...

    @mesh_error_criteria.setter
    def mesh_error_criteria(self, arg: object, /) -> None: ...

    @property
    def mesh_error_distributor(self) -> MeshErrorAggregation: ...

    @mesh_error_distributor.setter
    def mesh_error_distributor(self, arg: object, /) -> None: ...

    @property
    def solve_only_first(self) -> bool: ...

    @solve_only_first.setter
    def solve_only_first(self, arg: bool, /) -> None: ...

    @property
    def force_one_mesh_iter(self) -> bool: ...

    @force_one_mesh_iter.setter
    def force_one_mesh_iter(self, arg: bool, /) -> None: ...

    @property
    def new_error(self) -> bool: ...

    @new_error.setter
    def new_error(self, arg: bool, /) -> None: ...

    @property
    def detect_control_switches(self) -> bool: ...

    @detect_control_switches.setter
    def detect_control_switches(self, arg: bool, /) -> None: ...

    @property
    def rel_switch_tol(self) -> float: ...

    @rel_switch_tol.setter
    def rel_switch_tol(self, arg: float, /) -> None: ...

    @property
    def abs_switch_tol(self) -> float: ...

    @abs_switch_tol.setter
    def abs_switch_tol(self, arg: float, /) -> None: ...

    @property
    def mesh_abort_flag(self) -> _tychopy.solvers.ConvergenceFlags: ...

    @mesh_abort_flag.setter
    def mesh_abort_flag(self, arg: _tychopy.solvers.ConvergenceFlags, /) -> None: ...

    @property
    def num_extra_segs(self) -> int: ...

    @num_extra_segs.setter
    def num_extra_segs(self, arg: int, /) -> None: ...

    @property
    def mesh_red_factor(self) -> float: ...

    @mesh_red_factor.setter
    def mesh_red_factor(self, arg: float, /) -> None: ...

    @property
    def mesh_inc_factor(self) -> float: ...

    @mesh_inc_factor.setter
    def mesh_inc_factor(self, arg: float, /) -> None: ...

    @property
    def mesh_err_factor(self) -> float: ...

    @mesh_err_factor.setter
    def mesh_err_factor(self, arg: float, /) -> None: ...

    @property
    def mesh_converged(self) -> bool: ...

class OptimalControlProblem(_tychopy.solvers.OptimizationProblemBase):
    """
    A multi-phase optimal control problem.

    Collects one or more :class:`PhaseInterface` phases and the *link* constraints
    and objectives that couple them (e.g. continuity between the end of one phase
    and the start of the next, or a cost spanning several phases). Each phase is
    configured independently; the problem adds the cross-phase relationships and
    solves all phases together.

    Phases are referenced by an integer index (assigned in :meth:`add_phase`
    order), by the phase object itself, or by a registered name. Solve the whole
    problem with :meth:`optimize` (or :meth:`solve`), inherited from the shared
    optimization-problem base.

    Examples
    --------
    >>> ocp = OptimalControlProblem()
    >>> ocp.add_phase(phase0)
    >>> ocp.add_phase(phase1)
    >>> ocp.add_forward_link_equal_con(0, 1, range(0, 5))
    >>> ocp.optimize()
    """

    def __init__(self) -> None:
        """Construct an empty optimal control problem with no phases."""

    @overload
    def add_link_equal_con(self, func: _tychopy.vector_functions.VectorFunction, phasepack: Sequence[tuple[int | PhaseInterface | str, PhaseRegionFlags | str, int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str]]], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_equal_con(self, func: _tychopy.vector_functions.VectorFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, xt_u_vars0: int | Sequence[int] | str | list[str], op_vars0: int | Sequence[int] | str | list[str], sp_vars0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, xt_u_vars1: int | Sequence[int] | str | list[str], op_vars1: int | Sequence[int] | str | list[str], sp_vars1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_equal_con(self, func: _tychopy.vector_functions.VectorFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, v0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, v1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_equal_con(self, func: _tychopy.vector_functions.VectorFunction, link_parms: Sequence[numpy.ndarray], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_equal_con(self, func: _tychopy.vector_functions.VectorFunction, link_parms: numpy.ndarray, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_forward_link_equal_con(self, phase0: int | PhaseInterface | str, phase1: int | PhaseInterface | str, vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> list[int]:
        """
        Add a continuity (forward-link) constraint between two phases.

        Constrains the selected variables at the *back* of ``phase0`` to equal the
        same variables at the *front* of ``phase1`` -- the standard way to chain
        phases into a continuous trajectory.

        Parameters
        ----------
        phase0 : int, PhaseInterface, or str
            The upstream phase (its back/final node is used).
        phase1 : int, PhaseInterface, or str
            The downstream phase (its front/initial node is used).
        vars : int or sequence of int
            Variable index/indices (into the packed ``[x, t, u, p]`` layout) to make
            continuous across the link.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Examples
        --------
        >>> ocp.add_forward_link_equal_con(0, 1, range(0, 5))
        """

    def add_param_link_equal_con(self, phase0: int | PhaseInterface | str, phase1: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> list[int]: ...

    def add_direct_link_equal_con(self, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, v0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, v1: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, phasepack: Sequence[tuple[int | PhaseInterface | str, PhaseRegionFlags | str, int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str]]], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, xt_u_vars0: int | Sequence[int] | str | list[str], op_vars0: int | Sequence[int] | str | list[str], sp_vars0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, xt_u_vars1: int | Sequence[int] | str | list[str], op_vars1: int | Sequence[int] | str | list[str], sp_vars1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, v0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, v1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, link_parms: Sequence[numpy.ndarray], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, link_parms: numpy.ndarray, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_objective(self, func: _tychopy.vector_functions.ScalarFunction, phasepack: Sequence[tuple[int | PhaseInterface | str, PhaseRegionFlags | str, int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str]]], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_objective(self, func: _tychopy.vector_functions.ScalarFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, xt_u_vars0: int | Sequence[int] | str | list[str], op_vars0: int | Sequence[int] | str | list[str], sp_vars0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, xt_u_vars1: int | Sequence[int] | str | list[str], op_vars1: int | Sequence[int] | str | list[str], sp_vars1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_objective(self, func: _tychopy.vector_functions.ScalarFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, v0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, v1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an objective coupling variables of two phases.

        Adds the scalar value of ``func`` -- evaluated on selected variables drawn
        from a region of ``phase0`` and a region of ``phase1`` (and optionally the
        shared link parameters) -- to the problem objective.

        Parameters
        ----------
        func : VectorFunction
            Scalar-valued function of the linked variables.
        phase0 : int, PhaseInterface, or str
            First phase.
        reg0 : PhaseRegionFlags or str
            Region of ``phase0`` the variables ``v0`` are read from.
        v0 : int or sequence of int
            Variable index/indices from ``phase0``.
        phase1 : int, PhaseInterface, or str
            Second phase.
        reg1 : PhaseRegionFlags or str
            Region of ``phase1`` the variables ``v1`` are read from.
        v1 : int or sequence of int
            Variable index/indices from ``phase1``.
        linkparams : numpy.ndarray of int, optional
            Indices into the shared link-parameter vector to also pass to ``func``.
            Default empty.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new link objective.
        """

    @overload
    def add_link_param_objective(self, func: _tychopy.vector_functions.ScalarFunction, link_parms: Sequence[numpy.ndarray], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_objective(self, func: _tychopy.vector_functions.ScalarFunction, link_parms: numpy.ndarray, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def remove_link_equal_con(self, arg: int, /) -> None: ...

    def remove_link_inequal_con(self, arg: int, /) -> None: ...

    def remove_link_objective(self, arg: int, /) -> None: ...

    def add_phase(self, arg: PhaseInterface, /) -> int:
        """
        Add a phase to the problem.

        Parameters
        ----------
        phase : PhaseInterface
            The phase to append. Its problem index is assigned in call order
            (0, 1, ...).
        """

    def add_phases(self, arg: Sequence[PhaseInterface], /) -> list[int]:
        """
        Add several phases at once.

        Parameters
        ----------
        phases : sequence of PhaseInterface
            Phases to append, in order.
        """

    def get_phase_num(self, arg: PhaseInterface, /) -> int:
        """
        Return the problem index of a previously-added phase.

        Parameters
        ----------
        phase : PhaseInterface
            A phase that was added to this problem.

        Returns
        -------
        int
            The phase's index within the problem.
        """

    def remove_phase(self, arg: int, /) -> None:
        """
        Remove a phase from the problem by index.

        Parameters
        ----------
        pnum : int
            Index of the phase to remove.
        """

    def phase(self, arg: int, /) -> PhaseInterface:
        """
        Return a phase of the problem by index.

        Parameters
        ----------
        pnum : int
            Index of the phase to retrieve.

        Returns
        -------
        PhaseInterface
            The requested phase.
        """

    @overload
    def set_link_params(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> None: ...

    @overload
    def set_link_params(self, arg: numpy.ndarray, /) -> None: ...

    def add_link_param_vgroups(self, arg: "std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, Eigen::Matrix<int, -1, 1, 0, -1, 1> > > >", /) -> None: ...

    def set_link_param_vgroups(self, arg: "std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, Eigen::Matrix<int, -1, 1, 0, -1, 1> > > >", /) -> None: ...

    @overload
    def add_link_param_vgroup(self, arg0: numpy.ndarray, arg1: str, /) -> None: ...

    @overload
    def add_link_param_vgroup(self, arg0: int, arg1: str, /) -> None: ...

    def return_link_params(self) -> numpy.ndarray: ...

    def transcribe(self, arg0: bool, arg1: bool, /) -> None: ...

    @property
    def phases(self) -> list[PhaseInterface]: ...

    def return_link_equal_con_vals(self, arg: int, /) -> list[numpy.ndarray]: ...

    def return_link_equal_con_lmults(self, arg: int, /) -> list[numpy.ndarray]: ...

    def return_link_inequal_con_vals(self, arg: int, /) -> list[numpy.ndarray]: ...

    def return_link_inequal_con_lmults(self, arg: int, /) -> list[numpy.ndarray]: ...

    def return_link_equal_con_scales(self, arg: int, /) -> numpy.ndarray: ...

    def return_link_inequal_con_scales(self, arg: int, /) -> numpy.ndarray: ...

    def return_link_objective_scales(self, arg: int, /) -> numpy.ndarray: ...

    @property
    def auto_scaling(self) -> bool: ...

    @auto_scaling.setter
    def auto_scaling(self, arg: bool, /) -> None: ...

    @property
    def sync_objective_scales(self) -> bool: ...

    @sync_objective_scales.setter
    def sync_objective_scales(self, arg: bool, /) -> None: ...

    @property
    def adaptive_mesh(self) -> bool: ...

    @adaptive_mesh.setter
    def adaptive_mesh(self, arg: bool, /) -> None: ...

    @property
    def print_mesh_info(self) -> bool: ...

    @print_mesh_info.setter
    def print_mesh_info(self, arg: bool, /) -> None: ...

    @property
    def max_mesh_iters(self) -> int: ...

    @max_mesh_iters.setter
    def max_mesh_iters(self, arg: int, /) -> None: ...

    @property
    def mesh_converged(self) -> bool: ...

    @property
    def solve_only_first(self) -> bool: ...

    @solve_only_first.setter
    def solve_only_first(self, arg: bool, /) -> None: ...

    @property
    def mesh_abort_flag(self) -> _tychopy.solvers.ConvergenceFlags: ...

    @mesh_abort_flag.setter
    def mesh_abort_flag(self, arg: _tychopy.solvers.ConvergenceFlags, /) -> None: ...

    def set_adaptive_mesh(self, adaptive_mesh: bool = True, apply_to_phases: bool = True) -> None: ...

    def set_auto_scaling(self, auto_scaling: bool = True, apply_to_phases: bool = True) -> None: ...

    def set_mesh_tol(self, arg: float, /) -> None: ...

    def set_mesh_red_factor(self, arg: float, /) -> None: ...

    def set_mesh_inc_factor(self, arg: float, /) -> None: ...

    def set_mesh_err_factor(self, arg: float, /) -> None: ...

    def set_max_mesh_iters(self, arg: int, /) -> None: ...

    def set_min_segments(self, arg: int, /) -> None: ...

    def set_max_segments(self, arg: int, /) -> None: ...

    @overload
    def set_mesh_error_criteria(self, arg: MeshErrorAggregation, /) -> None: ...

    @overload
    def set_mesh_error_criteria(self, arg: str, /) -> None: ...

    @overload
    def set_mesh_error_estimator(self, arg: MeshErrorEstimators, /) -> None: ...

    @overload
    def set_mesh_error_estimator(self, arg: str, /) -> None: ...

    @overload
    def set_mesh_error_distributor(self, arg: MeshErrorAggregation, /) -> None: ...

    @overload
    def set_mesh_error_distributor(self, arg: str, /) -> None: ...

class LGLInterpTable:
    """
    Piecewise-polynomial interpolation table over a discretized trajectory.

    Builds a Legendre-Gauss-Lobatto (LGL) interpolant of a trajectory so the
    state (and optionally control) can be evaluated at arbitrary times between
    the stored nodes. A table can be constructed from an ODE (which supplies node
    derivatives) plus trajectory data, or directly from trajectory data with the
    derivatives recovered by finite difference.

    A trajectory is a list of packed node vectors, each ordered ``[x, t, u, p]``
    (the same layout used by phases). The ``time axis`` defaults to the time
    component of that layout.

    Notes
    -----
    Calling the table like a function -- ``table(t)`` -- is shorthand for
    :meth:`interpolate` at time ``t``.
    """

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: TranscriptionModes, arg4: list[numpy.ndarray], arg5: int, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: str, arg4: list[numpy.ndarray], arg5: int, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: int, arg4: str, arg5: list[numpy.ndarray], arg6: int, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: list[numpy.ndarray], /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: int, arg4: list[numpy.ndarray], /) -> None: ...

    @overload
    def __init__(self, arg0: int, arg1: list[numpy.ndarray], arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg: list[numpy.ndarray], /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: TranscriptionModes, /) -> None: ...

    @overload
    def __init__(self, arg0: int, arg1: int, arg2: TranscriptionModes, /) -> None: ...

    def load_even_data(self, arg: list[numpy.ndarray], /) -> None:
        """
        Load evenly-time-spaced node data, deriving node derivatives.

        Parameters
        ----------
        xtudat : list of numpy.ndarray
            Trajectory nodes, each packed as ``[x, t, u, p]``, evenly spaced in time.
            Derivatives are computed from the bound ODE (or by finite difference if
            the table was constructed without one).
        """

    def get_table_ptr(self) -> LGLInterpTable: ...

    def load_uneven_data(self, arg0: int, arg1: list[numpy.ndarray], /) -> None:
        """
        Load arbitrarily-spaced node data and resample onto an even mesh.

        Parameters
        ----------
        dnum : int
            Number of evenly-spaced segments to resample the trajectory onto.
        xtudat : list of numpy.ndarray
            Trajectory nodes, each packed as ``[x, t, u, p]``, in strictly
            increasing time order.
        """

    def interpolate(self, arg: float, /) -> numpy.ndarray:
        """
        Interpolate the trajectory at a single time.

        Parameters
        ----------
        t : float
            Time at which to evaluate, within ``[t0, tf]``.

        Returns
        -------
        numpy.ndarray
            The interpolated packed node vector ``[x, t, u, p]`` at time ``t``.
        """

    def new_error_integral(self) -> list[numpy.ndarray]: ...

    def __call__(self, arg: float, /) -> numpy.ndarray:
        """
        Interpolate the trajectory at a single time (calling shorthand for :meth:`interpolate`).

        Parameters
        ----------
        t : float
            Time at which to evaluate, within ``[t0, tf]``.

        Returns
        -------
        numpy.ndarray
            The interpolated packed node vector ``[x, t, u, p]`` at time ``t``.
        """

    def interpolate_deriv(self, arg: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, 2), order='F')]:
        """
        Interpolate the trajectory and its time derivative at a single time.

        Parameters
        ----------
        t : float
            Time at which to evaluate, within ``[t0, tf]``.

        Returns
        -------
        tuple of numpy.ndarray
            The interpolated value and its time derivative at ``t``.
        """

    def make_periodic(self) -> None:
        """
        Mark the trajectory periodic, snapping the last node onto the first state.
        """

    def interp_range(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]:
        """
        Interpolate the trajectory at evenly-spaced samples over a time range.

        Parameters
        ----------
        num : int
            Number of output samples.
        tl : float
            Lower time bound.
        th : float
            Upper time bound.

        Returns
        -------
        list of numpy.ndarray
            ``num`` interpolated node vectors spanning ``[tl, th]``.
        """

    def interp_whole_range(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Interpolate the trajectory at evenly-spaced samples over ``[t0, tf]``.

        Parameters
        ----------
        num : int
            Number of output samples.

        Returns
        -------
        list of numpy.ndarray
            ``num`` interpolated node vectors spanning the full table range.
        """

    def error_integral(self, arg: int, /) -> list[numpy.ndarray]: ...

    @property
    def t0(self) -> float:
        """float: initial time of the table."""

    @property
    def tf(self) -> float:
        """float: final time of the table."""

    def interp_non_dim(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]: ...

class InterpFunction:
    def __init__(self, arg0: LGLInterpTable, arg1: numpy.ndarray, /) -> None: ...

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
    def __call__(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> _tychopy.vector_functions.VectorFunction: ...

    @overload
    def __call__(self, arg: _tychopy.vector_functions.Element, /) -> _tychopy.vector_functions.VectorFunction: ...

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

class InterpFunction_1:
    def __init__(self, arg: LGLInterpTable, /) -> None: ...

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
    def __call__(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> _tychopy.vector_functions.VectorFunction: ...

    @overload
    def __call__(self, arg: _tychopy.vector_functions.Element, /) -> _tychopy.vector_functions.VectorFunction: ...

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

    def sf(self) -> _tychopy.vector_functions.ScalarFunction:
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

class InterpFunction_3:
    def __init__(self, arg: LGLInterpTable, /) -> None: ...

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
    def __call__(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> _tychopy.vector_functions.VectorFunction: ...

    @overload
    def __call__(self, arg: _tychopy.vector_functions.Element, /) -> _tychopy.vector_functions.VectorFunction: ...

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

class InterpFunction_6:
    def __init__(self, arg: LGLInterpTable, /) -> None: ...

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
    def __call__(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> _tychopy.vector_functions.VectorFunction: ...

    @overload
    def __call__(self, arg: _tychopy.vector_functions.Element, /) -> _tychopy.vector_functions.VectorFunction: ...

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

class FiniteDiffTable:
    def __init__(self, arg0: int, arg1: list[numpy.ndarray], /) -> None: ...

    def all_derivs(self, arg0: int, arg1: int, /) -> list[numpy.ndarray]: ...

    def deriv(self, arg0: int, arg1: int, arg2: int, /) -> numpy.ndarray: ...

class ODEArguments(_tychopy.vector_functions.Arguments):
    """
    Symbolic argument vector for defining ODE right-hand sides.

    ``ODEArguments(Xvars, Uvars, Pvars)`` constructs the ordered input
    ``[x, t, u, p]`` of an ODE -- the state ``x`` (size ``Xvars``), time ``t``,
    control ``u`` (size ``Uvars``), and ODE parameters ``p`` (size ``Pvars``).
    The accessors below slice out named sub-vectors and individual scalars as
    :class:`~tychopy.vector_functions.VectorFunction` expressions, which are then
    combined arithmetically to build the dynamics.

    Examples
    --------
    >>> XtU = ODEArguments(3, 1)        # 3 states, 1 control
    >>> x, y, v = XtU.x_vec().tolist()
    >>> theta = XtU.u_var(0)
    """

    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None:
        """
        Construct with explicit state, control, and parameter counts.

        Parameters
        ----------
        Xvars : int
            Number of state variables.
        Uvars : int
            Number of control variables.
        Pvars : int
            Number of ODE parameters.
        """

    @overload
    def __init__(self, arg0: int, arg1: int, /) -> None:
        """
        Construct with state and control counts (no ODE parameters).

        Parameters
        ----------
        Xvars : int
            Number of state variables.
        Uvars : int
            Number of control variables.
        """

    @overload
    def __init__(self, arg: int, /) -> None:
        """
        Construct with state count only (no controls or parameters).

        Parameters
        ----------
        Xvars : int
            Number of state variables.
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

    def x_vec(self) -> _tychopy.vector_functions.Segment:
        """
        State sub-vector ``x``.

        Returns
        -------
        VectorFunction
            Expression selecting the state block, of size ``Xvars``.
        """

    def x_var(self, arg: int, /) -> _tychopy.vector_functions.Element:
        """
        A single state variable.

        Parameters
        ----------
        i : int
            Zero-based index into the state block.

        Returns
        -------
        VectorFunction
            Scalar expression for state ``x[i]``.
        """

    def xt_vec(self) -> _tychopy.vector_functions.Segment:
        """
        Combined state-and-time sub-vector ``[x, t]``.

        Returns
        -------
        VectorFunction
            Expression of size ``Xvars + 1``.
        """

    def t_var(self) -> _tychopy.vector_functions.Element:
        """
        The time variable ``t``.

        Returns
        -------
        VectorFunction
            Scalar expression for time.
        """

    def u_vec(self) -> _tychopy.vector_functions.Segment:
        """
        Control sub-vector ``u``.

        Returns
        -------
        VectorFunction
            Expression selecting the control block, of size ``Uvars``.
        """

    def u_var(self, arg: int, /) -> _tychopy.vector_functions.Element:
        """
        A single control variable.

        Parameters
        ----------
        i : int
            Zero-based index into the control block.

        Returns
        -------
        VectorFunction
            Scalar expression for control ``u[i]``.
        """

    def p_vec(self) -> _tychopy.vector_functions.Segment:
        """
        ODE-parameter sub-vector ``p``.

        Returns
        -------
        VectorFunction
            Expression selecting the parameter block, of size ``Pvars``.
        """

    def p_var(self, arg: int, /) -> _tychopy.vector_functions.Element:
        """
        A single ODE parameter.

        Parameters
        ----------
        i : int
            Zero-based index into the parameter block.

        Returns
        -------
        VectorFunction
            Scalar expression for parameter ``p[i]``.
        """
