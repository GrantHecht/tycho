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
    """
    Adaptive Runge-Kutta algorithm used by an integrator or mesh-error estimator.

    Pass as the first argument to ``integrator(alg, dt)`` (or its string name,
    e.g. ``"DOPRI87"``) to select the RK scheme. Only the eight values listed here
    are runtime-selectable; the remaining internal variants are used by the
    transcription and collocation subsystems.
    """

    DOPRI54 = 0
    """
    Dormand-Prince 5(4) — 7 stages, FSAL, adaptive (5th-order solution, 4th-order error estimate).
    """

    DOPRI87 = 1
    """
    Dormand-Prince 8(7) — 13 stages, adaptive (8th-order solution, 7th-order error estimate). Default algorithm.
    """

    Tsit5 = 2
    """
    Tsitouras 5(4) — 7 stages, FSAL, adaptive (5th-order solution, 4th-order error estimate).
    """

    BS3 = 3
    """
    Bogacki-Shampine 3(2) — 4 stages, FSAL, adaptive (3rd-order solution, 2nd-order error estimate).
    """

    BS5 = 4
    """
    Bogacki-Shampine 5(4) — 8 stages, adaptive with lazy interpolant (5th-order solution, 4th-order error estimate).
    """

    Vern7 = 5
    """
    Verner 7(6) — 10 stages, adaptive with lazy interpolant (7th-order solution, 6th-order error estimate).
    """

    Vern8 = 6
    """
    Verner 8(7) — 13 stages, adaptive with lazy interpolant (8th-order solution, 7th-order error estimate).
    """

    Vern9 = 7
    """
    Verner 9(8) — 16 stages, adaptive with lazy interpolant (9th-order solution, 8th-order error estimate).
    """

class IVPController(enum.Enum):
    """
    Adaptive step-size controller used by the integrator.

    Controls how the step size is grown or shrunk after each accepted or rejected
    step based on the local error estimate. Select via ``integrator.set_controller``
    or an equivalent constructor argument.
    """

    I = 0
    """
    Integral (I) controller — step size scaled by EEst^(1/k) / gamma. Simple and robust; default for most methods.
    """

    PI = 1
    """
    Proportional-integral (PI) controller — uses current and previous error estimates for smoother step growth.
    """

    PID = 2
    """
    Proportional-integral-derivative (PID) controller with Soderlind limiter — uses three successive error estimates for the smoothest adaptation.
    """

class ErrorNormType(enum.Enum):
    """
    Norm used to reduce per-component local errors to a scalar for step control.

    The scalar error estimate (EEst) is compared against 1.0 to decide
    accept/reject and to scale the next step size.
    """

    RMS = 0
    """
    Root-mean-square of scaled per-component errors. Matches Julia OrdinaryDiffEq default norm; default choice.
    """

    MAX = 1
    """
    Maximum (L-infinity) of scaled per-component errors. More conservative; useful for stiff or sensitivity-critical problems.
    """

class TranscriptionModes(enum.Enum):
    """
    Transcription scheme used to discretize a phase.

    Passed (as an enum or its string name, e.g. ``"LGL3"``) when constructing
    a phase to select how the continuous dynamics are converted into a
    finite-dimensional set of constraints -- either collocation defects (the LGL
    and ``Trapezoidal`` schemes) or shooting defects (``CentralShooting``).
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
    """End-to-end re-integration trajectory error."""

def strto_phase_region_flag(name: str) -> PhaseRegionFlags:
    """
    Convert a string name to the corresponding ``PhaseRegionFlags`` value.

    Raises ``ValueError`` if ``name`` does not match any known flag.

    Parameters
    ----------
    name : str
        Case-sensitive string name of the phase-region flag, e.g. ``"Front"``,
        ``"Back"``, ``"Path"``, ``"FrontandBack"``, ``"BackandFront"``,
        ``"Params"``, ``"NodalPath"``, ``"InnerPath"``, ``"ODEParams"``,
        ``"StaticParams"``, ``"PairWisePath"``.

    Returns
    -------
    PhaseRegionFlags
        The enum value corresponding to ``name``.
    """

class StateConstraint:
    """
    A vector-valued constraint bound to a phase region.

    Bundles a VectorFunction with the phase region it applies to and the
    variable indices it reads. This is the internal record a phase stores
    for each equality or inequality constraint added via
    :meth:`~PhaseInterface.add_equal_con` / :meth:`~PhaseInterface.add_inequal_con`.
    Users typically do not construct ``StateConstraint`` objects directly --
    they are built and registered by the ``add_*`` methods on a phase.

    Parameters
    ----------
    func : VectorFunction
        The constraint function (vector-valued, output dimension = number of
        constraints).
    region : PhaseRegionFlags or str
        Phase region the constraint acts on (e.g. ``"Front"``, ``"Back"``,
        ``"Path"``).
    vars : numpy.ndarray of int
        Variable indices within each node vector that the function reads.
    """

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, arg3: numpy.ndarray, arg4: numpy.ndarray, /) -> None:
        """
        Bind a function to a phase region with explicit variable index groups.

        Parameters
        ----------
        func : VectorFunction
            The vector function to attach.
        region : PhaseRegionFlags or str
            Phase region the function acts on (e.g. ``"Front"``, ``"Back"``,
            ``"Path"``).
        xtu_vars : numpy.ndarray of int
            State/time/control variable indices within each node vector that the
            function reads.
        op_vars : numpy.ndarray of int
            ODE-parameter variable indices the function reads.
        sp_vars : numpy.ndarray of int
            Static-parameter variable indices the function reads.
        """

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, arg3: PhaseRegionFlags, arg4: numpy.ndarray, /) -> None:
        """
        Bind a function spanning both a state region and a parameter region.

        Parameters
        ----------
        func : VectorFunction
            The vector function to attach.
        region : PhaseRegionFlags or str
            Phase region for the state/time/control variables.
        xtu_vars : numpy.ndarray of int
            State/time/control variable indices the function reads.
        param_region : PhaseRegionFlags or str
            Parameter region — must be ``"ODEParams"`` or ``"StaticParams"``.
        param_vars : numpy.ndarray of int
            Parameter variable indices the function reads.
        """

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, /) -> None:
        """
        Bind a function to a phase region with a single variable index group.

        Parameters
        ----------
        func : VectorFunction
            The vector function to attach.
        region : PhaseRegionFlags or str
            Phase region the function acts on. When set to ``"ODEParams"`` or
            ``"StaticParams"``, ``vars`` is interpreted as the corresponding
            parameter indices; otherwise it is the state/time/control indices.
        vars : numpy.ndarray of int
            Variable indices, interpreted according to ``region``.
        """

class StateObjective:
    """
    A scalar objective bound to a phase region.

    Bundles a scalar VectorFunction (output rows = 1) with the phase region
    it applies to and the variable indices it reads. This is the internal
    record a phase stores for each objective added via
    :meth:`~PhaseInterface.add_state_objective`. Users typically do not construct
    ``StateObjective`` objects directly -- they are built and registered by
    the ``add_*`` methods on a phase.

    Parameters
    ----------
    func : VectorFunction
        The objective function (scalar-valued, output rows = 1).
    region : PhaseRegionFlags or str
        Phase region the objective acts on (e.g. ``"Front"``, ``"Back"``,
        ``"Path"``).
    vars : numpy.ndarray of int
        Variable indices within each node vector that the function reads.
    """

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.ScalarFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, arg3: numpy.ndarray, arg4: numpy.ndarray, /) -> None:
        """
        Bind a function to a phase region with explicit variable index groups.

        Parameters
        ----------
        func : VectorFunction
            The vector function to attach.
        region : PhaseRegionFlags or str
            Phase region the function acts on (e.g. ``"Front"``, ``"Back"``,
            ``"Path"``).
        xtu_vars : numpy.ndarray of int
            State/time/control variable indices within each node vector that the
            function reads.
        op_vars : numpy.ndarray of int
            ODE-parameter variable indices the function reads.
        sp_vars : numpy.ndarray of int
            Static-parameter variable indices the function reads.
        """

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.ScalarFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, arg3: PhaseRegionFlags, arg4: numpy.ndarray, /) -> None:
        """
        Bind a function spanning both a state region and a parameter region.

        Parameters
        ----------
        func : VectorFunction
            The vector function to attach.
        region : PhaseRegionFlags or str
            Phase region for the state/time/control variables.
        xtu_vars : numpy.ndarray of int
            State/time/control variable indices the function reads.
        param_region : PhaseRegionFlags or str
            Parameter region — must be ``"ODEParams"`` or ``"StaticParams"``.
        param_vars : numpy.ndarray of int
            Parameter variable indices the function reads.
        """

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.ScalarFunction, arg1: PhaseRegionFlags, arg2: numpy.ndarray, /) -> None:
        """
        Bind a function to a phase region with a single variable index group.

        Parameters
        ----------
        func : VectorFunction
            The vector function to attach.
        region : PhaseRegionFlags or str
            Phase region the function acts on. When set to ``"ODEParams"`` or
            ``"StaticParams"``, ``vars`` is interpreted as the corresponding
            parameter indices; otherwise it is the state/time/control indices.
        vars : numpy.ndarray of int
            Variable indices, interpreted according to ``region``.
        """

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
    """
    Specification for a single event (zero-crossing) to be tracked by an integrator.

    An event function is a scalar-output :class:`VectorFunction` that is evaluated
    at each step. When its sign changes between two successive steps the integrator
    records the bracketed interval as a crossing. The ``direction`` filter
    restricts which sign changes are recorded, and ``stop_count`` can halt
    integration after a given number of events.

    A list of ``EventPack`` objects is passed to the integrator via its
    ``integrate_events`` family of methods. Each element of the list corresponds
    to an independent event function, and the returned event-time arrays are
    indexed accordingly.

    ``EventPack`` objects are implicitly constructible from a 3-tuple
    ``(vf, direction, stop_count)`` so callers may pass plain tuples wherever an
    ``EventPack`` is expected.
    """

    @overload
    def __init__(self, vf: _tychopy.vector_functions.ScalarFunction, direction: int = 0, stop_count: int = 0) -> None:
        """
        Construct an event specification.

        Parameters
        ----------
        vf : VectorFunction
            Scalar-output (output rows == 1) function over the ODE input
            ``[x, t, u, p]``. A zero crossing of this function defines the event.
        direction : int, optional
            Crossing direction filter.  ``-1`` records only falling crossings
            (positive-to-negative); ``0`` (default) records any crossing;
            ``+1`` records only rising crossings (negative-to-positive).
        stop_count : int, optional
            Number of crossings after which integration halts.  ``0`` (default)
            means "record crossings but never stop"; ``N > 0`` stops after the
            *N*-th recorded crossing.

        Raises
        ------
        ValueError
            If ``direction`` is not in ``{-1, 0, +1}`` or ``stop_count < 0``.
        """

    @overload
    def __init__(self, t: tuple) -> None:
        """
        Construct from a 3-tuple ``(vf, direction, stop_count)``.

        Convenience form used by the implicit conversion from ``tuple``.  The
        elements are interpreted the same way as the explicit 3-argument constructor.

        Raises
        ------
        TypeError
            If the tuple does not have exactly 3 elements.
        ValueError
            If ``direction`` is not in ``{-1, 0, +1}`` or ``stop_count < 0``.
        """

    @property
    def vf(self) -> _tychopy.vector_functions.ScalarFunction:
        """
        The scalar-output event function.

        A :class:`VectorFunction` with exactly one output row.  The integrator
        evaluates this function at every accepted step point; a sign change between
        consecutive evaluations triggers an event recording.
        """

    @vf.setter
    def vf(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> None: ...

    @property
    def direction(self) -> int:
        """
        Crossing direction filter: ``-1`` (falling), ``0`` (any), or ``+1`` (rising).

        Only crossings matching the specified direction are recorded.  Assigning a
        value outside ``{-1, 0, +1}`` raises ``ValueError``.
        """

    @direction.setter
    def direction(self, arg: int, /) -> None: ...

    @property
    def stop_count(self) -> int:
        """
        Number of crossings after which integration stops.

        ``0`` (default) means all crossings are recorded but integration is never
        halted by this event.  ``N > 0`` halts integration after the *N*-th
        crossing is recorded.  Assigning a negative value raises ``ValueError``.
        """

    @stop_count.setter
    def stop_count(self, arg: int, /) -> None: ...

class PhaseInterface(_tychopy.solvers.OptimizationProblemBase):
    """
    A single phase of an optimal control problem.

    A phase couples an ODE, a discretized trajectory, and the boundary values,
    bounds, constraints, and objectives that transcribe a continuous optimal
    control problem into a finite-dimensional nonlinear program. The
    ``transcription_mode`` selects how the dynamics are enforced between nodes:
    direct collocation (the LGL schemes and ``Trapezoidal``) or central shooting
    (``CentralShooting``), which numerically propagates both bounding nodes of each
    interval to its midpoint and constrains the two results to agree. Phases are not
    constructed directly; build one from an ODE with
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
    Build a phase, constrain it, solve, and read the trajectory::

        phase = ode.phase("LGL3", Xs, 32)
        phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
        phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
        phase.add_boundary_value("Back", [0, 1], [xf, yf])
        phase.add_delta_time_objective(1.0)
        phase.optimize()
        traj = phase.return_traj()
    """

    def enable_vectorization(self, arg: bool, /) -> None:
        """
        Enable or disable SuperScalar (vectorized) function evaluation.

        Parameters
        ----------
        b : bool
            Pass ``True`` to enable vectorization (the default), ``False`` to disable.
        """

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
            order. This becomes the initial guess that the optimizer refines.

        Notes
        -----
        Additional overloads accept a defect-segment count or an explicit
        defects-binding/defects-per-block specification; see the C++ reference.
        """

    @overload
    def switch_transcription_mode(self, arg0: TranscriptionModes, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None:
        """
        Switch the transcription scheme and reload the trajectory onto a new mesh.

        Parameters
        ----------
        mode : TranscriptionModes or str
            The new transcription scheme (e.g. ``"LGL3"``, ``"LGL5"``, ``"Trapezoidal"``,
            ``"CentralShooting"``).
        bin_times : numpy.ndarray, optional
            Normalized defect-bin boundary times in ``[0, 1]``.
        bin_segments : numpy.ndarray of int, optional
            Number of defect intervals to place in each bin.

        Notes
        -----
        If ``bin_times`` and ``bin_segments`` are omitted the current mesh spacing is
        preserved. Invalidates the cached transcription and post-solve data.
        """

    @overload
    def switch_transcription_mode(self, arg: TranscriptionModes, /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg0: str, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg: str, /) -> None: ...

    def transcribe(self, arg0: bool, arg1: bool, /) -> None:
        """
        Transcribe the phase into its underlying nonlinear program.

        Builds the transcribed NLP (defect constraints, user constraints, and
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

    def refine_traj_equal(self, n: int) -> list[numpy.ndarray]:
        """
        Resample the current trajectory onto an equally-spaced mesh.

        Parameters
        ----------
        n : int
            Number of equally-spaced segments (defect intervals) to divide the mesh
            into.

        Returns
        -------
        list of numpy.ndarray
            The re-interpolated trajectory on the new equally-spaced mesh.
        """

    @overload
    def set_static_params(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> None:
        """
        Set the static (non-ODE) parameter vector and its per-element scaling units.

        Parameters
        ----------
        parm : numpy.ndarray
            Static-parameter values.
        units : numpy.ndarray
            Per-parameter scaling units; must be the same length as ``parm``.

        Notes
        -----
        Calling this overload with a ``units`` vector replaces any previously
        registered static parameters and invalidates the cached transcription.
        An overload that accepts only ``parm`` (with unit scaling) is also available.
        """

    @overload
    def set_static_params(self, arg: numpy.ndarray, /) -> None: ...

    @overload
    def add_static_params(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> None:
        """
        Append additional static parameters and their per-element scaling units.

        Parameters
        ----------
        parm : numpy.ndarray
            Static-parameter values to append.
        units : numpy.ndarray
            Per-parameter scaling units for the appended values; must match ``parm`` in size.

        Notes
        -----
        An overload without ``units`` (unit scaling assumed) is also available.
        """

    @overload
    def add_static_params(self, arg: numpy.ndarray, /) -> None: ...

    def add_static_param_vgroups(self, arg: "std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, Eigen::Matrix<int, -1, 1, 0, -1, 1> > > >", /) -> None:
        """
        Merge additional named static-parameter variable groups into the phase.

        Parameters
        ----------
        spidxs : dict[str, numpy.ndarray of int]
            Map of group name to static-parameter index array. Existing groups with the
            same name are overwritten.
        """

    def set_static_param_vgroups(self, arg: "std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, Eigen::Matrix<int, -1, 1, 0, -1, 1> > > >", /) -> None:
        """
        Replace all named static-parameter variable groups.

        Parameters
        ----------
        spidxs : dict[str, numpy.ndarray of int]
            Map of group name to static-parameter index array; replaces any previously
            registered groups entirely.
        """

    @overload
    def add_static_param_vgroup(self, arg0: numpy.ndarray, arg1: str, /) -> None:
        """
        Register a named group of static-parameter indices.

        Parameters
        ----------
        idx : numpy.ndarray of int
            Static-parameter indices that belong to this group.
        key : str
            Name of the group (used with string-based variable selectors).
        """

    @overload
    def add_static_param_vgroup(self, arg0: int, arg1: str, /) -> None: ...

    @overload
    def set_control_mode(self, arg: ControlModes, /) -> None:
        """
        Set the control representation used during transcription.

        Parameters
        ----------
        mode : ControlModes or str
            Control mode, e.g. ``"FirstOrderSpline"`` (default) or ``"BlockConstant"``.
            Invalidates the cached transcription.
        """

    @overload
    def set_control_mode(self, arg: str, /) -> None: ...

    def set_integral_mode(self, arg: IntegralModes, /) -> None:
        """
        Set the quadrature rule used for integral objectives and parameter functions.

        Parameters
        ----------
        mode : IntegralModes
            Integral quadrature mode. Invalidates the cached transcription.
        """

    def sub_static_params(self, arg: numpy.ndarray, /) -> None:
        """
        Update static-parameter values in-place without triggering re-transcription.

        If the size of ``parm`` matches the current static-parameter count the values are
        substituted directly; otherwise the parameters are fully replaced (which does
        invalidate the transcription).

        Parameters
        ----------
        parm : numpy.ndarray
            New static-parameter values.
        """

    @overload
    def sub_variables(self, arg0: PhaseRegionFlags, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None:
        """
        Substitute fixed values into selected trajectory variables at a region.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            The region (e.g. ``"Front"``, ``"Back"``, ``"Path"``) whose variables are
            updated.
        indices : numpy.ndarray of int
            Variable indices (into the packed ``[x, t, u, p]`` layout) to overwrite.
        vals : numpy.ndarray
            Replacement values; must match ``indices`` in length.
        """

    @overload
    def sub_variables(self, arg0: str, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def sub_variable(self, arg0: PhaseRegionFlags, arg1: int, arg2: float, /) -> None:
        """
        Substitute a fixed value into a single trajectory variable at a region.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            The region whose variable is updated.
        var : int
            Variable index (into the packed ``[x, t, u, p]`` layout) to overwrite.
        val : float
            Replacement value.
        """

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
        Solve the phase and read back the optimized trajectory::

            phase.optimize()
            traj = phase.return_traj()
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

    def return_traj_range_nd(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]:
        """
        Resample the trajectory over a non-dimensional (fractional) time range.

        Resamples the trajectory into ``num`` equally-spaced segments spanning the
        fractional time range ``[tl, th]``, where ``0.0`` is the start of the
        trajectory and ``1.0`` is the end. Nodes within each segment follow the
        transcription's LGL node spacing, so the returned list is longer than ``num``.

        Parameters
        ----------
        num : int
            Number of equally-spaced segments to divide the range into.
        tl : float
            Lower non-dimensional (fractional) time bound in ``[0, 1]``.
        th : float
            Upper non-dimensional (fractional) time bound in ``[0, 1]``.

        Returns
        -------
        list of numpy.ndarray
            The resampled node vectors spanning the fractional range ``[tl, th]``.
        """

    def return_traj_table(self) -> LGLInterpTable:
        """
        Return an LGL interpolation table built from the exact current trajectory.

        Returns
        -------
        LGLInterpTable
            An interpolation table loaded with the exact (non-interpolated) trajectory
            data. Useful for custom dense output queries.
        """

    def return_costate_traj(self) -> list[numpy.ndarray]:
        """
        Return the costate (adjoint/dual) trajectory.

        Returns
        -------
        list of numpy.ndarray
            One costate vector per trajectory node (same count as :meth:`return_traj`),
            each of length ``x_vars + 1`` -- the ``x_vars`` costate components plus a
            time entry at the time index. Costates are recovered from the
            dynamics-defect Lagrange multipliers, which are defined at the defect
            points and then linearly interpolated (extrapolated at the endpoints) onto
            the trajectory nodes. Only meaningful after a successful
            :meth:`optimize`/:meth:`solve`.
        """

    def return_traj_error(self) -> list[numpy.ndarray]:
        """
        Return the discretization-error estimate of the current trajectory.

        Returns
        -------
        list of numpy.ndarray
            One error vector per interior collocation point (``num_card_states - 1``
            per defect interval, so the count is the number of trajectory nodes minus
            one). Each vector has length ``x_vars + 1`` -- the ``x_vars`` per-state
            error components plus a time entry. Only available after a successful
            :meth:`optimize`/:meth:`solve` (raises ``RuntimeError`` otherwise).
        """

    def return_u_spline_con_lmults(self) -> list[numpy.ndarray]:
        """
        Return the control-spline constraint Lagrange multipliers from the last solve.

        Returns
        -------
        list of numpy.ndarray
            Per-node multiplier vectors for the control-spline continuity constraint,
            or an empty list if no control-spline constraint is active. Only meaningful
            after a successful :meth:`optimize`/:meth:`solve`.
        """

    def return_u_spline_con_vals(self) -> list[numpy.ndarray]:
        """
        Return the control-spline constraint residual values from the last solve.

        Returns
        -------
        list of numpy.ndarray
            Per-node residual vectors for the control-spline continuity constraint,
            or an empty list if no control-spline constraint is active. Only meaningful
            after a successful :meth:`optimize`/:meth:`solve`.
        """

    def return_equal_con_lmults(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Return the Lagrange multipliers of an equality constraint from the last solve.

        Parameters
        ----------
        index : int
            The equality-constraint index returned by :meth:`add_equal_con` (or a
            similar ``add_*`` call).

        Returns
        -------
        list of numpy.ndarray
            Per-application multiplier vectors for the constraint. Only meaningful
            after a successful :meth:`optimize`/:meth:`solve`.
        """

    def return_equal_con_vals(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Return the residual values of an equality constraint from the last solve.

        Parameters
        ----------
        index : int
            The equality-constraint index.

        Returns
        -------
        list of numpy.ndarray
            Per-application residual vectors. Only meaningful after a successful
            :meth:`optimize`/:meth:`solve`.
        """

    def return_equal_con_scales(self, arg: int, /) -> numpy.ndarray:
        """
        Return the output scales of a registered equality constraint.

        Parameters
        ----------
        index : int
            The equality-constraint index.

        Returns
        -------
        numpy.ndarray
            Per-output scale vector for the constraint.
        """

    def return_inequal_con_lmults(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Return the Lagrange multipliers of an inequality constraint from the last solve.

        Parameters
        ----------
        index : int
            The inequality-constraint index returned by :meth:`add_inequal_con` (or a
            similar ``add_*`` call).

        Returns
        -------
        list of numpy.ndarray
            Per-application multiplier vectors. Only meaningful after a successful
            :meth:`optimize`/:meth:`solve`.
        """

    def return_inequal_con_vals(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Return the residual values of an inequality constraint from the last solve.

        Parameters
        ----------
        index : int
            The inequality-constraint index.

        Returns
        -------
        list of numpy.ndarray
            Per-application residual vectors. Only meaningful after a successful
            :meth:`optimize`/:meth:`solve`.
        """

    def return_inequal_con_scales(self, arg: int, /) -> numpy.ndarray:
        """
        Return the output scales of a registered inequality constraint.

        Parameters
        ----------
        index : int
            The inequality-constraint index.

        Returns
        -------
        numpy.ndarray
            Per-output scale vector for the constraint.
        """

    def return_integral_objective_scales(self, arg: int, /) -> numpy.ndarray:
        """
        Return the output scales of a registered integral objective.

        Parameters
        ----------
        index : int
            The integral-objective index returned by :meth:`add_integral_objective`.

        Returns
        -------
        numpy.ndarray
            Per-output scale vector for the objective.
        """

    def return_integral_param_function_scales(self, arg: int, /) -> numpy.ndarray:
        """
        Return the output scales of a registered integral-parameter function.

        Parameters
        ----------
        index : int
            The integral-parameter-function index returned by
            :meth:`add_integral_param_function`.

        Returns
        -------
        numpy.ndarray
            Per-output scale vector for the function.
        """

    def return_state_objective_scales(self, arg: int, /) -> numpy.ndarray:
        """
        Return the output scales of a registered state (terminal) objective.

        Parameters
        ----------
        index : int
            The state-objective index returned by :meth:`add_state_objective`.

        Returns
        -------
        numpy.ndarray
            Per-output scale vector for the objective.
        """

    def return_ode_output_scales(self) -> numpy.ndarray:
        """
        Return the output scales applied to the ODE dynamics (defect) equations.

        Returns
        -------
        numpy.ndarray
            Per-state-derivative output scale vector, derived from the current unit
            scaling of the state and time variables.
        """

    def return_static_params(self) -> numpy.ndarray:
        """
        Return the current static-parameter vector.

        Returns
        -------
        numpy.ndarray
            The active static-parameter values. After a successful
            :meth:`optimize`/:meth:`solve` this reflects the optimized values.
        """

    def test_partitions(self, arg0: int, arg1: int, arg2: int, /) -> None:
        """
        Benchmark threaded function partitioning over a range of thread counts.

        .. note::
           Internal diagnostic / benchmarking helper. Not part of the normal
           problem-solving workflow; most users never need to call it.

        Parameters
        ----------
        i : int
            First partition count to test.
        j : int
            Last partition count to test.
        n : int
            Number of timing repetitions per count.
        """

    def remove_equal_con(self, arg: int, /) -> None:
        """
        Remove a previously added equality constraint by index.

        Parameters
        ----------
        index : int
            The constraint index to remove, or ``-1`` to remove the most recently added.
        """

    def remove_inequal_con(self, arg: int, /) -> None:
        """
        Remove a previously added inequality constraint by index.

        Parameters
        ----------
        index : int
            The constraint index to remove, or ``-1`` to remove the most recently added.
        """

    def remove_state_objective(self, arg: int, /) -> None:
        """
        Remove a previously added state (terminal) objective by index.

        Parameters
        ----------
        index : int
            The objective index to remove, or ``-1`` to remove the most recently added.
        """

    def remove_integral_objective(self, arg: int, /) -> None:
        """
        Remove a previously added integral (running-cost) objective by index.

        Parameters
        ----------
        index : int
            The objective index to remove, or ``-1`` to remove the most recently added.
        """

    def remove_integral_param_function(self, arg: int, /) -> None:
        """
        Remove a previously added integral-parameter function by index.

        Parameters
        ----------
        index : int
            The function index to remove, or ``-1`` to remove the most recently added.
        """

    @overload
    def add_equal_con(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.VectorFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an equality constraint f=0 over a region with full variable bindings.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region over which the constraint is evaluated.
        func : VectorFunction
            Vector-valued constraint function; enforced to be zero.
        xt_u_vars : int, sequence of int, or str
            State/time/control variable indices (or named group) passed to ``func``.
        op_vars : int, sequence of int, or str
            ODE-parameter variable indices passed to ``func``.
        sp_vars : int, sequence of int, or str
            Static-parameter variable indices passed to ``func``.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    @overload
    def add_equal_con(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.VectorFunction, input_index: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an equality constraint f=0 over a region (state/time/control variables only).

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region over which the constraint is evaluated.
        func : VectorFunction
            Vector-valued constraint function; enforced to be zero.
        input_index : int, sequence of int, or str
            Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
            ``func``.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    @overload
    def add_equal_con(self, arg: StateConstraint, /) -> int:
        """
        Add a pre-built equality constraint to the phase.

        Parameters
        ----------
        con : StateConstraint
            A region-bound equality constraint constructed externally.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

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
        Pin the front state fully and the back position to a target::

            phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
            phase.add_boundary_value("Back", [0, 1], [xf, yf])
        """

    def add_delta_var_equal_con(self, var: int | Sequence[int] | str | list[str], value: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Constrain the front-to-back change of a variable to a fixed value.

        Parameters
        ----------
        var : int or sequence of int
            Variable index (into the packed ``[x, t, u, p]`` layout).
        value : float
            Target change (back value minus front value).
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    def add_delta_time_equal_con(self, value: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Constrain the total elapsed time of the phase to a fixed value.

        Parameters
        ----------
        value : float
            Target elapsed time (``t_back - t_front``).
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    def add_value_lock(self, reg: PhaseRegionFlags | str, vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Lock selected region variables to their current trajectory values.

        Adds an equality constraint that pins the named variables to whatever values
        they currently hold in the trajectory (useful to freeze a variable while
        optimizing others).

        Parameters
        ----------
        reg : PhaseRegionFlags or str
            Region the variables are read from.
        vars : int or sequence of int
            Variable index/indices to lock.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    def add_periodicity_con(self, vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Enforce periodicity (front == back) on selected variables.

        Parameters
        ----------
        vars : int or sequence of int
            Variable index/indices (into the packed ``[x, t, u, p]`` layout) that must
            match between the first and last trajectory node.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    @overload
    def add_inequal_con(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.VectorFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an inequality constraint f<=0 over a region with full variable bindings.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region over which the constraint is evaluated.
        func : VectorFunction
            Vector-valued constraint function; each output must be ≤ 0.
        xt_u_vars : int, sequence of int, or str
            State/time/control variable indices (or named group) passed to ``func``.
        op_vars : int, sequence of int, or str
            ODE-parameter variable indices passed to ``func``.
        sp_vars : int, sequence of int, or str
            Static-parameter variable indices passed to ``func``.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    @overload
    def add_inequal_con(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.VectorFunction, input_index: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an inequality constraint f<=0 over a region (state/time/control variables only).

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region over which the constraint is evaluated.
        func : VectorFunction
            Vector-valued constraint function; each output must be ≤ 0.
        input_index : int, sequence of int, or str
            Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
            ``func``.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    @overload
    def add_inequal_con(self, arg: StateConstraint, /) -> int:
        """
        Add a pre-built inequality constraint to the phase.

        Parameters
        ----------
        con : StateConstraint
            A region-bound inequality constraint constructed externally.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

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
        Bound the control variable along the whole path::

            phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
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
    def add_lu_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound a scalar function of region variables from below and above.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region over which the function is evaluated.
        func : ScalarFunction
            Scalar-valued function to bound.
        xt_u_vars : int, sequence of int, or str
            State/time/control variable indices passed to ``func``.
        op_vars : int, sequence of int, or str
            ODE-parameter variable indices passed to ``func``.
        sp_vars : int, sequence of int, or str
            Static-parameter variable indices passed to ``func``.
        lowerbound : float
            Lower bound on the function value.
        upperbound : float
            Upper bound on the function value.
        scale : float, optional
            Shared constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    @overload
    def add_lu_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_lower_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound a scalar function of region variables from below.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region over which the function is evaluated.
        func : ScalarFunction
            Scalar-valued function to bound.
        xt_u_vars : int, sequence of int, or str
            State/time/control variable indices passed to ``func``.
        op_vars : int, sequence of int, or str
            ODE-parameter variable indices passed to ``func``.
        sp_vars : int, sequence of int, or str
            Static-parameter variable indices passed to ``func``.
        lowerbound : float
            Lower bound on the function value.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    @overload
    def add_lower_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_upper_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound a scalar function of region variables from above.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region over which the function is evaluated.
        func : ScalarFunction
            Scalar-valued function to bound.
        xt_u_vars : int, sequence of int, or str
            State/time/control variable indices passed to ``func``.
        op_vars : int, sequence of int, or str
            ODE-parameter variable indices passed to ``func``.
        sp_vars : int, sequence of int, or str
            Static-parameter variable indices passed to ``func``.
        upperbound : float
            Upper bound on the function value.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    @overload
    def add_upper_func_bound(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, xt_up_vars: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_lu_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the Euclidean norm of selected variables from below and above.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variables are read from.
        xt_up_vars : int or sequence of int
            Variable indices whose norm is bounded.
        lowerbound : float
            Lower bound on the norm.
        upperbound : float
            Upper bound on the norm.
        scale : float, optional
            Shared constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_lu_squared_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the squared Euclidean norm of selected variables from below and above.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variables are read from.
        xt_up_vars : int or sequence of int
            Variable indices whose squared norm is bounded.
        lowerbound : float
            Lower bound on the squared norm.
        upperbound : float
            Upper bound on the squared norm.
        scale : float, optional
            Shared constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_lower_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the Euclidean norm of selected variables from below.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variables are read from.
        xt_up_vars : int or sequence of int
            Variable indices whose norm is bounded.
        lowerbound : float
            Lower bound on the norm.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_lower_squared_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the squared Euclidean norm of selected variables from below.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variables are read from.
        xt_up_vars : int or sequence of int
            Variable indices whose squared norm is bounded.
        lowerbound : float
            Lower bound on the squared norm.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_upper_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the Euclidean norm of selected variables from above.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variables are read from.
        xt_up_vars : int or sequence of int
            Variable indices whose norm is bounded.
        upperbound : float
            Upper bound on the norm.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_upper_squared_norm_bound(self, phase_region: PhaseRegionFlags | str, xt_up_vars: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the squared Euclidean norm of selected variables from above.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variables are read from.
        xt_up_vars : int or sequence of int
            Variable indices whose squared norm is bounded.
        upperbound : float
            Upper bound on the squared norm.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_lower_delta_var_bound(self, var: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the front-to-back change of a variable from below.

        Parameters
        ----------
        var : int or sequence of int
            Variable index (into the packed ``[x, t, u, p]`` layout).
        lowerbound : float
            Lower bound on the change (back minus front).
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_lower_delta_time_bound(self, lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the elapsed time of the phase from below.

        Parameters
        ----------
        lowerbound : float
            Lower bound on ``t_back - t_front``.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_upper_delta_var_bound(self, var: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the front-to-back change of a variable from above.

        Parameters
        ----------
        var : int or sequence of int
            Variable index (into the packed ``[x, t, u, p]`` layout).
        upperbound : float
            Upper bound on the change (back minus front).
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

    def add_upper_delta_time_bound(self, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Bound the elapsed time of the phase from above.

        Parameters
        ----------
        upperbound : float
            Upper bound on ``t_back - t_front``.
        scale : float, optional
            Constraint scaling. Default 1.0.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new bound constraint.
        """

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
        func : ScalarFunction
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
    def add_state_objective(self, arg: StateObjective, /) -> int:
        """
        Add a pre-built state (terminal) objective to the phase.

        Parameters
        ----------
        obj : StateObjective
            A region-bound scalar objective constructed externally.

        Returns
        -------
        int
            The index assigned to the new objective.
        """

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
        Minimize the phase duration::

            phase.add_delta_time_objective(1.0)
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
        func : ScalarFunction
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
    def add_integral_objective(self, arg: StateObjective, /) -> int:
        """
        Add a pre-built integral (running-cost) objective to the phase.

        Parameters
        ----------
        obj : StateObjective
            A region-bound scalar integrand objective constructed externally.

        Returns
        -------
        int
            The index assigned to the new objective.
        """

    @overload
    def add_integral_param_function(self, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], int_param: int, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an integral that accumulates into a parameter, with full variable bindings.

        Integrates ``func`` along the path and adds the result to the ODE parameter
        at index ``int_param``.

        Parameters
        ----------
        func : ScalarFunction
            Scalar-valued integrand.
        xt_u_vars : int, sequence of int, or str
            State/time/control variable indices passed to ``func``.
        op_vars : int, sequence of int, or str
            ODE-parameter variable indices passed to ``func``.
        sp_vars : int, sequence of int, or str
            Static-parameter variable indices passed to ``func``.
        int_param : int
            Index of the ODE parameter the integral accumulates into.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new integral-parameter function.
        """

    @overload
    def add_integral_param_function(self, func: _tychopy.vector_functions.ScalarFunction, input_index: int | Sequence[int] | str | list[str], int_param: int, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an integral that accumulates into a parameter (state/time/control variables only).

        Parameters
        ----------
        func : ScalarFunction
            Scalar-valued integrand.
        input_index : int, sequence of int, or str
            Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
            ``func``.
        int_param : int
            Index of the ODE parameter the integral accumulates into.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new integral-parameter function.
        """

    @overload
    def add_integral_param_function(self, arg0: StateObjective, arg1: int, /) -> int:
        """
        Add a pre-built integral-parameter function that accumulates into a parameter.

        Parameters
        ----------
        obj : StateObjective
            A region-bound scalar integrand constructed externally.
        int_param : int
            Index of the ODE parameter the integral value accumulates into.

        Returns
        -------
        int
            The index assigned to the new integral-parameter function.
        """

    @overload
    def add_lu_var_bounds(self, arg0: PhaseRegionFlags, arg1: numpy.ndarray, arg2: float, arg3: float, arg4: float, /) -> numpy.ndarray:
        """
        Bound several variables from below and above with a shared scale.

        Parameters
        ----------
        phase_region : PhaseRegionFlags or str
            Region the variables are read from.
        vars : numpy.ndarray of int
            Variable indices (into the packed ``[x, t, u, p]`` layout) to bound.
        lowerbound : float
            Lower bound applied to each variable.
        upperbound : float
            Upper bound applied to each variable.
        scale : float
            Shared constraint scaling.

        Returns
        -------
        numpy.ndarray of int
            Indices assigned to the new bound constraints (one per variable).
        """

    @overload
    def add_lu_var_bounds(self, arg0: str, arg1: numpy.ndarray, arg2: float, arg3: float, arg4: float, /) -> numpy.ndarray: ...

    def get_mesh_info(self, arg0: bool, arg1: int, /) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Compute mesh error, equidistributed bin boundaries, and per-segment error.

        Parameters
        ----------
        integ : bool
            Use the reference integrator estimator (``True``) or the de Boor estimator
            (``False``).
        n : int
            Number of equidistributing bins to produce.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            ``(node_times, bin_boundaries, per_segment_error)`` — the non-dimensional
            node times, the ``n+1`` equidistributed bin boundary times, and the
            per-segment error vector.
        """

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
    def adaptive_mesh(self) -> bool:
        """
        Enable adaptive mesh refinement (bool, default ``False``).

        When ``True``, :meth:`optimize`/:meth:`solve` runs the automatic mesh-refinement
        loop until the per-interval error falls below :attr:`mesh_tol` or
        :attr:`max_mesh_iters` is reached.
        """

    @adaptive_mesh.setter
    def adaptive_mesh(self, arg: bool, /) -> None: ...

    @property
    def auto_scaling(self) -> bool:
        """
        Enable automatic variable/equation scaling (bool, default ``False``).

        When ``True``, the transcription scales each decision variable and constraint
        by values derived from the current trajectory. Writing this attribute directly
        sets the flag only; use :meth:`set_auto_scaling` to update it and reset the
        cached transcription in one step.
        """

    @auto_scaling.setter
    def auto_scaling(self, arg: bool, /) -> None: ...

    @property
    def sync_objective_scales(self) -> bool:
        """
        Synchronize objective scales with auto-scaling (bool, default ``True``).

        When ``True`` and :attr:`auto_scaling` is enabled, objective scale factors are
        updated together with the variable/constraint scales.
        """

    @sync_objective_scales.setter
    def sync_objective_scales(self, arg: bool, /) -> None: ...

    def set_auto_scaling(self, auto_scaling: bool = True) -> None:
        """
        Enable or disable automatic variable/equation scaling.

        Parameters
        ----------
        auto_scaling : bool, optional
            ``True`` to enable automatic scaling (default), ``False`` to disable.
            Invalidates the cached transcription.
        """

    def set_adaptive_mesh(self, adaptive_mesh: bool = True) -> None:
        """
        Enable or disable adaptive mesh refinement.

        Parameters
        ----------
        adaptive_mesh : bool, optional
            ``True`` to enable adaptive refinement (default), ``False`` to disable.
        """

    @overload
    def set_units(self, **kwargs) -> None:
        """
        Set per-variable scaling units for the state/time/control/ODE-param block.

        Accepts keyword arguments whose names are variable-group names registered on the
        phase (e.g. ``x``, ``t``, ``u``, ``p``); each value is a scalar or array of
        units for that group. Alternatively, call the overload that accepts a raw
        ``numpy.ndarray`` of the full unit vector.

        Parameters
        ----------
        **kwargs
            Group name to unit value/array mapping. Scalar values are broadcast to the
            whole group; arrays must match the group size.
        """

    @overload
    def set_units(self, arg: numpy.ndarray, /) -> None:
        """
        Set per-variable scaling units from a raw vector.

        Parameters
        ----------
        xtup_units : numpy.ndarray
            Per-variable scaling units for the full ``[x, t, u, p]`` layout, length
            equal to ``x_vars + 1 + u_vars + p_vars``.
        """

    def set_mesh_tol(self, arg: float, /) -> None:
        """
        Set the target mesh-error tolerance for adaptive refinement.

        Parameters
        ----------
        t : float
            Tolerance (absolute value used); refinement stops when the estimated
            per-interval error falls below this threshold.
        """

    def set_mesh_red_factor(self, arg: float, /) -> None:
        """
        Set the factor by which segment counts may be reduced during mesh refinement.

        Parameters
        ----------
        t : float
            Reduction factor (absolute value used); segments in low-error intervals
            are multiplied by this value.
        """

    def set_mesh_inc_factor(self, arg: float, /) -> None:
        """
        Set the factor by which segment counts may be increased during mesh refinement.

        Parameters
        ----------
        t : float
            Increase factor (absolute value used); segments in high-error intervals
            are multiplied by this value.
        """

    def set_mesh_err_factor(self, arg: float, /) -> None:
        """
        Set the error-overshoot factor that triggers mesh refinement.

        Parameters
        ----------
        t : float
            Error factor (absolute value used); refinement is triggered when the
            estimated error exceeds ``mesh_tol * mesh_err_factor``.
        """

    def set_max_mesh_iters(self, arg: int, /) -> None:
        """
        Set the maximum number of adaptive mesh-refinement iterations.

        Parameters
        ----------
        it : int
            Iteration cap (absolute value used).
        """

    def set_min_segments(self, arg: int, /) -> None:
        """
        Set the minimum number of mesh segments allowed during adaptive refinement.

        Parameters
        ----------
        it : int
            Minimum segment count (absolute value used).
        """

    def set_max_segments(self, arg: int, /) -> None:
        """
        Set the maximum number of mesh segments allowed during adaptive refinement.

        Parameters
        ----------
        it : int
            Maximum segment count (absolute value used).
        """

    @overload
    def set_mesh_error_criteria(self, arg: MeshErrorAggregation, /) -> None:
        """
        Set the aggregation function used to determine mesh convergence.

        Parameters
        ----------
        mode : MeshErrorAggregation or str
            Aggregation mode, e.g. ``"MAX"`` (default) or ``"AVG"``.  The mesh is
            considered converged when the aggregated per-interval error is below
            :attr:`mesh_tol`.
        """

    @overload
    def set_mesh_error_criteria(self, arg: str, /) -> None: ...

    @overload
    def set_mesh_error_estimator(self, arg: MeshErrorEstimators, /) -> None:
        """
        Set the method used to estimate per-interval discretization error.

        Parameters
        ----------
        mode : MeshErrorEstimators or str
            Estimator, e.g. ``"DEBOOR"`` (default) or ``"Integrator"``.
        """

    @overload
    def set_mesh_error_estimator(self, arg: str, /) -> None: ...

    @overload
    def set_mesh_error_distributor(self, arg: MeshErrorAggregation, /) -> None:
        """
        Set the aggregation used to distribute the mesh-error density across segments.

        Parameters
        ----------
        mode : MeshErrorAggregation or str
            Distribution aggregation mode, e.g. ``"AVG"`` (default) or ``"MAX"``.
        """

    @overload
    def set_mesh_error_distributor(self, arg: str, /) -> None: ...

    @property
    def print_mesh_info(self) -> bool:
        """
        Print mesh-refinement diagnostics during the adaptive loop (bool, default ``True``).
        """

    @print_mesh_info.setter
    def print_mesh_info(self, arg: bool, /) -> None: ...

    @property
    def max_mesh_iters(self) -> int:
        """
        Maximum number of adaptive mesh-refinement iterations (int, default 10).
        """

    @max_mesh_iters.setter
    def max_mesh_iters(self, arg: int, /) -> None: ...

    @property
    def mesh_tol(self) -> float:
        """
        Target per-interval mesh-error tolerance for adaptive refinement (float, default 1e-6).
        """

    @mesh_tol.setter
    def mesh_tol(self, arg: float, /) -> None: ...

    @property
    def mesh_error_estimator(self) -> MeshErrorEstimators:
        """
        Method used to estimate per-interval discretization error (MeshErrorEstimators or str).

        Accepts a :class:`MeshErrorEstimators` enum value or its string name (e.g.
        ``"DEBOOR"`` or ``"Integrator"``). Default: ``MeshErrorEstimators.DEBOOR``.
        """

    @mesh_error_estimator.setter
    def mesh_error_estimator(self, arg: object, /) -> None: ...

    @property
    def mesh_error_criteria(self) -> MeshErrorAggregation:
        """
        Aggregation function used to determine mesh convergence (MeshErrorAggregation or str).

        Accepts a :class:`MeshErrorAggregation` enum value or its string name (e.g.
        ``"MAX"`` or ``"AVG"``). The mesh is converged when the aggregated per-interval
        error is below :attr:`mesh_tol`. Default: ``MeshErrorAggregation.MAX``.
        """

    @mesh_error_criteria.setter
    def mesh_error_criteria(self, arg: object, /) -> None: ...

    @property
    def mesh_error_distributor(self) -> MeshErrorAggregation:
        """
        Aggregation used to distribute error density when redistributing mesh segments (MeshErrorAggregation or str).

        Accepts a :class:`MeshErrorAggregation` enum value or its string name (e.g.
        ``"AVG"`` or ``"MAX"``). Default: ``MeshErrorAggregation.AVG``.
        """

    @mesh_error_distributor.setter
    def mesh_error_distributor(self, arg: object, /) -> None: ...

    @property
    def solve_only_first(self) -> bool:
        """
        Only solve (not optimize) on the first mesh-refinement iteration (bool, default ``True``).
        """

    @solve_only_first.setter
    def solve_only_first(self, arg: bool, /) -> None: ...

    @property
    def force_one_mesh_iter(self) -> bool:
        """
        Force at least one mesh-refinement iteration even if the initial mesh already meets tolerance (bool, default ``False``).
        """

    @force_one_mesh_iter.setter
    def force_one_mesh_iter(self, arg: bool, /) -> None: ...

    @property
    def new_error(self) -> bool:
        """
        Use the newer mesh-error formulation instead of the legacy one (bool, default ``False``).
        """

    @new_error.setter
    def new_error(self, arg: bool, /) -> None: ...

    @property
    def detect_control_switches(self) -> bool:
        """
        Detect and refine the mesh around control switches (bool, default ``False``).
        """

    @detect_control_switches.setter
    def detect_control_switches(self, arg: bool, /) -> None: ...

    @property
    def rel_switch_tol(self) -> float:
        """Relative tolerance for control-switch detection (float, default 0.3)."""

    @rel_switch_tol.setter
    def rel_switch_tol(self, arg: float, /) -> None: ...

    @property
    def abs_switch_tol(self) -> float:
        """Absolute tolerance for control-switch detection (float, default 1e-6)."""

    @abs_switch_tol.setter
    def abs_switch_tol(self, arg: float, /) -> None: ...

    @property
    def mesh_abort_flag(self) -> _tychopy.solvers.ConvergenceFlags:
        """
        Solver convergence flag that aborts the mesh-refinement loop early (ConvergenceFlags, default ``DIVERGING``).
        """

    @mesh_abort_flag.setter
    def mesh_abort_flag(self, arg: _tychopy.solvers.ConvergenceFlags, /) -> None: ...

    @property
    def num_extra_segs(self) -> int:
        """Extra segments added when refining a mesh interval (int, default 4)."""

    @num_extra_segs.setter
    def num_extra_segs(self, arg: int, /) -> None: ...

    @property
    def mesh_red_factor(self) -> float:
        """
        Factor by which segment counts may be reduced during adaptive refinement (float, default 0.5).
        """

    @mesh_red_factor.setter
    def mesh_red_factor(self, arg: float, /) -> None: ...

    @property
    def mesh_inc_factor(self) -> float:
        """
        Factor by which segment counts may be increased during adaptive refinement (float, default 5.0).
        """

    @mesh_inc_factor.setter
    def mesh_inc_factor(self, arg: float, /) -> None: ...

    @property
    def mesh_err_factor(self) -> float:
        """
        Error-overshoot factor that triggers segment count increase during refinement (float, default 10.0).
        """

    @mesh_err_factor.setter
    def mesh_err_factor(self, arg: float, /) -> None: ...

    @property
    def mesh_converged(self) -> bool:
        """
        ``True`` if the last adaptive mesh-refinement loop met the :attr:`mesh_tol` criterion (read-only bool).
        """

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
    Assemble a multi-phase problem, link adjacent phases, and solve::

        ocp = OptimalControlProblem()
        ocp.add_phase(phase0)
        ocp.add_phase(phase1)
        ocp.add_forward_link_equal_con(0, 1, range(0, 5))
        ocp.optimize()
    """

    def __init__(self) -> None:
        """Construct an empty optimal control problem with no phases."""

    @overload
    def add_link_equal_con(self, func: _tychopy.vector_functions.VectorFunction, phasepack: Sequence[tuple[int | PhaseInterface | str, PhaseRegionFlags | str, int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str]]], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add a link equality constraint ``func == 0`` over a list of phase targets.

        Evaluates ``func`` on variables drawn from each entry of ``phasepack`` (and
        optionally the shared link parameters) and enforces the result to be zero.

        Parameters
        ----------
        func : VectorFunction
            Vector-valued function; its output is constrained to zero.
        phasepack : list of tuple
            Per-phase link targets, each specifying a phase reference, region, and
            variable selectors.
        linkparams : numpy.ndarray of int, optional
            Indices into the shared link-parameter vector to also pass to ``func``.
            Default empty.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new link equality constraint.
        """

    @overload
    def add_link_equal_con(self, func: _tychopy.vector_functions.VectorFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, xt_u_vars0: int | Sequence[int] | str | list[str], op_vars0: int | Sequence[int] | str | list[str], sp_vars0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, xt_u_vars1: int | Sequence[int] | str | list[str], op_vars1: int | Sequence[int] | str | list[str], sp_vars1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_equal_con(self, func: _tychopy.vector_functions.VectorFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, v0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, v1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_equal_con(self, func: _tychopy.vector_functions.VectorFunction, link_parms: Sequence[numpy.ndarray], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an equality constraint ``func == 0`` on the shared link parameters.

        Parameters
        ----------
        func : VectorFunction
            Vector-valued function of the selected link parameters.
        link_parms : list of numpy.ndarray of int
            Per-application link-parameter index vectors.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

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
            Variable index/indices (into the packed ``[x, t, u, p]`` node layout, where
            ``p`` are the ODE parameters -- not the static parameters) to make
            continuous across the link.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
            ``"auto"``.

        Returns
        -------
        list of int
            The index/indices assigned to the added link constraint(s).

        Examples
        --------
        Make the first five variables continuous across phases 0 and 1::

            ocp.add_forward_link_equal_con(0, 1, range(0, 5))
        """

    def add_param_link_equal_con(self, phase0: int | PhaseInterface | str, phase1: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> list[int]:
        """
        Chain consecutive phases with equal-parameter continuity constraints.

        For each consecutive pair of phases from ``phase0`` to ``phase1``, adds a
        link equality constraint enforcing that the selected parameter variables
        are identical across the join.

        Parameters
        ----------
        phase0 : int, PhaseInterface, or str
            Starting phase reference.
        phase1 : int, PhaseInterface, or str
            Ending phase reference (must have a higher index than ``phase0``).
        reg0 : PhaseRegionFlags or str
            Parameter region (must be ``ODEParams`` or ``StaticParams``).
        vars : int or sequence of int
            Parameter variable index/indices to make continuous.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        list of int
            The indices assigned to the added constraints (one per adjacent phase pair).
        """

    def add_direct_link_equal_con(self, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, v0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, v1: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Constrain selected variables from two phase regions to be exactly equal.

        Adds a link equality constraint enforcing ``v0 - v1 == 0`` between the
        selected variables drawn from ``phase0`` and ``phase1``.

        Parameters
        ----------
        phase0 : int, PhaseInterface, or str
            First phase reference.
        reg0 : PhaseRegionFlags or str
            Region of ``phase0`` to read ``v0`` from.
        v0 : int or sequence of int
            Variable index/indices from ``phase0``.
        phase1 : int, PhaseInterface, or str
            Second phase reference.
        reg1 : PhaseRegionFlags or str
            Region of ``phase1`` to read ``v1`` from.
        v1 : int or sequence of int
            Variable index/indices from ``phase1``.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

    @overload
    def add_link_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, phasepack: Sequence[tuple[int | PhaseInterface | str, PhaseRegionFlags | str, int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str], int | Sequence[int] | str | list[str]]], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add a link inequality constraint ``func <= 0`` over a list of phase targets.

        Evaluates ``func`` on variables drawn from each entry of ``phasepack`` (and
        optionally the shared link parameters) and enforces the result to be
        non-positive.

        Parameters
        ----------
        func : VectorFunction
            Vector-valued function; each output is constrained to be ``<= 0``.
        phasepack : list of tuple
            Per-phase link targets, each specifying a phase reference, region, and
            variable selectors.
        linkparams : numpy.ndarray of int, optional
            Indices into the shared link-parameter vector to also pass to ``func``.
            Default empty.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new link inequality constraint.
        """

    @overload
    def add_link_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, xt_u_vars0: int | Sequence[int] | str | list[str], op_vars0: int | Sequence[int] | str | list[str], sp_vars0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, xt_u_vars1: int | Sequence[int] | str | list[str], op_vars1: int | Sequence[int] | str | list[str], sp_vars1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, v0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, v1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_inequal_con(self, func: _tychopy.vector_functions.VectorFunction, link_parms: Sequence[numpy.ndarray], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add an inequality constraint ``func <= 0`` on the shared link parameters.

        Parameters
        ----------
        func : VectorFunction
            Vector-valued function of the selected link parameters.
        link_parms : list of numpy.ndarray of int
            Per-application link-parameter index vectors.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new constraint.
        """

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
        func : ScalarFunction
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
    def add_link_param_objective(self, func: _tychopy.vector_functions.ScalarFunction, link_parms: Sequence[numpy.ndarray], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int:
        """
        Add a scalar objective on the shared link parameters.

        Parameters
        ----------
        func : ScalarFunction
            Scalar-valued function of the selected link parameters.
        link_parms : list of numpy.ndarray of int
            Per-application link-parameter index vectors.
        auto_scale : ScaleModes or str, optional
            Output-scaling mode. Defaults to ``"auto"``.

        Returns
        -------
        int
            The index assigned to the new link objective.
        """

    @overload
    def add_link_param_objective(self, func: _tychopy.vector_functions.ScalarFunction, link_parms: numpy.ndarray, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def remove_link_equal_con(self, arg: int, /) -> None:
        """
        Remove a previously added link equality constraint by index.

        Parameters
        ----------
        index : int
            Index of the constraint to remove, or ``-1`` to remove the most recently
            added one.
        """

    def remove_link_inequal_con(self, arg: int, /) -> None:
        """
        Remove a previously added link inequality constraint by index.

        Parameters
        ----------
        index : int
            Index of the constraint to remove, or ``-1`` to remove the most recently
            added one.
        """

    def remove_link_objective(self, arg: int, /) -> None:
        """
        Remove a previously added link objective by index.

        Parameters
        ----------
        index : int
            Index of the objective to remove, or ``-1`` to remove the most recently
            added one.
        """

    def add_phase(self, arg: PhaseInterface, /) -> int:
        """
        Add a phase to the problem.

        Parameters
        ----------
        phase : PhaseInterface
            The phase to append. Its problem index is assigned in call order
            (0, 1, ...).

        Returns
        -------
        int
            The index assigned to the new phase.
        """

    def add_phases(self, arg: Sequence[PhaseInterface], /) -> list[int]:
        """
        Add several phases at once.

        Parameters
        ----------
        phases : sequence of PhaseInterface
            Phases to append, in order.

        Returns
        -------
        list of int
            The indices assigned to the new phases, in add order.
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
    def set_link_params(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> None:
        """
        Set the shared link-parameter vector and its per-parameter scaling units.

        Parameters
        ----------
        parm : numpy.ndarray
            Link-parameter values.
        units : numpy.ndarray
            Per-parameter scaling units; must be the same length as ``parm``.
        """

    @overload
    def set_link_params(self, arg: numpy.ndarray, /) -> None: ...

    def add_link_param_vgroups(self, arg: "std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, Eigen::Matrix<int, -1, 1, 0, -1, 1> > > >", /) -> None:
        """
        Merge additional named link-parameter variable groups into the problem.

        Parameters
        ----------
        lpidxs : dict[str, numpy.ndarray of int]
            Map of group name to link-parameter index array. Existing groups with
            the same name are overwritten.
        """

    def set_link_param_vgroups(self, arg: "std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, Eigen::Matrix<int, -1, 1, 0, -1, 1> > > >", /) -> None:
        """
        Replace all named link-parameter variable groups.

        Parameters
        ----------
        lpidxs : dict[str, numpy.ndarray of int]
            Map of group name to link-parameter index array. Any previously
            registered groups are discarded.
        """

    @overload
    def add_link_param_vgroup(self, arg0: numpy.ndarray, arg1: str, /) -> None:
        """
        Register a named group of link-parameter indices.

        Parameters
        ----------
        idx : numpy.ndarray of int
            Link-parameter indices belonging to the group.
        key : str
            Name for the group. Used to reference these indices by name in
            link-function variable selectors (the ``link_parms`` arguments of
            :meth:`add_link_equal_con`, :meth:`add_link_objective`, etc.).
        """

    @overload
    def add_link_param_vgroup(self, arg0: int, arg1: str, /) -> None: ...

    def return_link_params(self) -> numpy.ndarray:
        """
        Return the current shared link-parameter vector.

        Returns
        -------
        numpy.ndarray
            The link-parameter values set by :meth:`set_link_params`.
        """

    def transcribe(self, arg0: bool, arg1: bool, /) -> None:
        """
        Transcribe all phases and link functions into the shared NLP.

        Parameters
        ----------
        showstats : bool
            If ``True``, print problem-size statistics after transcription.
        showfuns : bool
            If ``True``, print per-function details after transcription.
        """

    @property
    def phases(self) -> list[PhaseInterface]:
        """The list of phases registered with this problem, in add order."""

    def return_link_equal_con_vals(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Return the residual values of a link equality constraint from the last solve.

        Parameters
        ----------
        index : int
            Index of the link equality constraint.

        Returns
        -------
        list of numpy.ndarray
            Per-application residual vectors (one entry per application of the constraint).
        """

    def return_link_equal_con_lmults(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Return the Lagrange multipliers of a link equality constraint from the last solve.

        Parameters
        ----------
        index : int
            Index of the link equality constraint.

        Returns
        -------
        list of numpy.ndarray
            Per-application Lagrange multiplier vectors.
        """

    def return_link_inequal_con_vals(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Return the residual values of a link inequality constraint from the last solve.

        Parameters
        ----------
        index : int
            Index of the link inequality constraint.

        Returns
        -------
        list of numpy.ndarray
            Per-application residual vectors.
        """

    def return_link_inequal_con_lmults(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Return the Lagrange multipliers of a link inequality constraint from the last solve.

        Parameters
        ----------
        index : int
            Index of the link inequality constraint.

        Returns
        -------
        list of numpy.ndarray
            Per-application Lagrange multiplier vectors.
        """

    def return_link_equal_con_scales(self, arg: int, /) -> numpy.ndarray:
        """
        Return the per-output scale vector of a link equality constraint.

        Parameters
        ----------
        index : int
            Index of the link equality constraint.

        Returns
        -------
        numpy.ndarray
            Per-output scale factors.
        """

    def return_link_inequal_con_scales(self, arg: int, /) -> numpy.ndarray:
        """
        Return the per-output scale vector of a link inequality constraint.

        Parameters
        ----------
        index : int
            Index of the link inequality constraint.

        Returns
        -------
        numpy.ndarray
            Per-output scale factors.
        """

    def return_link_objective_scales(self, arg: int, /) -> numpy.ndarray:
        """
        Return the per-output scale vector of a link objective.

        Parameters
        ----------
        index : int
            Index of the link objective.

        Returns
        -------
        numpy.ndarray
            Per-output scale factors.
        """

    @property
    def auto_scaling(self) -> bool:
        """Enable automatic output scaling for link functions and phases."""

    @auto_scaling.setter
    def auto_scaling(self, arg: bool, /) -> None: ...

    @property
    def sync_objective_scales(self) -> bool:
        """Keep link-objective scales in sync when automatic scaling is updated."""

    @sync_objective_scales.setter
    def sync_objective_scales(self, arg: bool, /) -> None: ...

    @property
    def adaptive_mesh(self) -> bool:
        """Enable problem-wide adaptive mesh refinement across all phases."""

    @adaptive_mesh.setter
    def adaptive_mesh(self, arg: bool, /) -> None: ...

    @property
    def print_mesh_info(self) -> bool:
        """Print mesh-refinement diagnostics after each mesh iteration."""

    @print_mesh_info.setter
    def print_mesh_info(self, arg: bool, /) -> None: ...

    @property
    def max_mesh_iters(self) -> int:
        """Maximum number of adaptive mesh-refinement iterations."""

    @max_mesh_iters.setter
    def max_mesh_iters(self, arg: int, /) -> None: ...

    @property
    def mesh_converged(self) -> bool:
        """
        ``True`` if all adaptive phases satisfied their mesh-error tolerance on the last run.
        """

    @property
    def solve_only_first(self) -> bool:
        """
        Run only the initial *solve* step on the first mesh-refinement iteration.

        Only affects the compound ``SolveOptimize`` and ``SolveOptimizeSolve`` job
        modes: when ``True``, subsequent mesh iterations drop the leading solve step
        (running ``Optimize`` / ``OptimizeSolve`` instead). Has no effect on plain
        ``solve`` or ``optimize``.
        """

    @solve_only_first.setter
    def solve_only_first(self, arg: bool, /) -> None: ...

    @property
    def mesh_abort_flag(self) -> _tychopy.solvers.ConvergenceFlags:
        """
        Severity threshold at which the adaptive mesh-refinement loop aborts early.

        The loop aborts as soon as the solver's convergence flag is at or above this
        severity (the check is ``flag >= mesh_abort_flag``). The flag ordering is
        ``CONVERGED < ACCEPTABLE < NOTCONVERGED < DIVERGING``. Default is
        ``DIVERGING`` (only divergence aborts); set it to ``NOTCONVERGED`` to also
        abort when a mesh iteration fails to converge.
        """

    @mesh_abort_flag.setter
    def mesh_abort_flag(self, arg: _tychopy.solvers.ConvergenceFlags, /) -> None: ...

    def set_adaptive_mesh(self, adaptive_mesh: bool = True, apply_to_phases: bool = True) -> None:
        """
        Enable or disable adaptive mesh refinement, optionally propagating to all phases.

        Parameters
        ----------
        adaptive_mesh : bool, optional
            Whether to enable adaptive mesh refinement. Default ``True``.
        apply_to_phases : bool, optional
            If ``True``, also set the flag on every registered phase. Default ``True``.
        """

    def set_auto_scaling(self, auto_scaling: bool = True, apply_to_phases: bool = True) -> None:
        """
        Enable or disable automatic output scaling, optionally propagating to all phases.

        Parameters
        ----------
        auto_scaling : bool, optional
            Whether to enable automatic scaling. Default ``True``.
        apply_to_phases : bool, optional
            If ``True``, also set the flag on every registered phase. Default ``True``.
        """

    def set_mesh_tol(self, arg: float, /) -> None:
        """
        Set the mesh-error tolerance on every registered phase.

        Parameters
        ----------
        t : float
            Desired mesh-error tolerance.
        """

    def set_mesh_red_factor(self, arg: float, /) -> None:
        """
        Set the mesh segment-count reduction factor on every registered phase.

        Parameters
        ----------
        t : float
            Reduction factor applied when the mesh is too fine.
        """

    def set_mesh_inc_factor(self, arg: float, /) -> None:
        """
        Set the mesh segment-count increase factor on every registered phase.

        Parameters
        ----------
        t : float
            Increase factor applied when the mesh is too coarse.
        """

    def set_mesh_err_factor(self, arg: float, /) -> None:
        """
        Set the mesh error-overshoot factor on every registered phase.

        Parameters
        ----------
        t : float
            Overshoot factor used when estimating how many segments are needed.
        """

    def set_max_mesh_iters(self, arg: int, /) -> None:
        """
        Set the maximum number of problem-level mesh-refinement iterations.

        Parameters
        ----------
        it : int
            Iteration cap.
        """

    def set_min_segments(self, arg: int, /) -> None:
        """
        Set the minimum mesh segment count on every registered phase.

        Parameters
        ----------
        it : int
            Minimum number of collocation segments per phase.
        """

    def set_max_segments(self, arg: int, /) -> None:
        """
        Set the maximum mesh segment count on every registered phase.

        Parameters
        ----------
        it : int
            Maximum number of collocation segments per phase.
        """

    @overload
    def set_mesh_error_criteria(self, arg: MeshErrorAggregation, /) -> None:
        """
        Set the mesh-error convergence aggregation on every registered phase.

        Parameters
        ----------
        m : MeshErrorAggregation
            Aggregation mode (e.g. ``Max``, ``Mean``) used to decide whether the mesh
            has converged.
        """

    @overload
    def set_mesh_error_criteria(self, arg: str, /) -> None: ...

    @overload
    def set_mesh_error_estimator(self, arg: MeshErrorEstimators, /) -> None:
        """
        Set the mesh-error estimator on every registered phase.

        Parameters
        ----------
        m : MeshErrorEstimators
            Estimator used to compute the per-segment mesh error.
        """

    @overload
    def set_mesh_error_estimator(self, arg: str, /) -> None: ...

    @overload
    def set_mesh_error_distributor(self, arg: MeshErrorAggregation, /) -> None:
        """
        Set the mesh-error distribution aggregation on every registered phase.

        Parameters
        ----------
        m : MeshErrorAggregation
            Aggregation mode used to distribute error across segments when refining.
        """

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
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: TranscriptionModes, arg4: list[numpy.ndarray], arg5: int, /) -> None:
        """
        Construct from an ODE, dimensions, transcription mode, trajectory data, and segment count.

        Parameters
        ----------
        ode : VectorFunction
            The ODE used to compute node derivatives.
        x_vars : int
            Number of state variables.
        u_vars : int
            Number of control variables.
        mode : TranscriptionModes
            Transcription mode that determines polynomial order (e.g.
            ``TranscriptionModes.LGL3``).
        traj : list of numpy.ndarray
            Trajectory nodes, each packed as ``[x, t, u]``, in strictly
            increasing time order.
        num_segments : int
            Number of defect intervals to resample the trajectory onto.
        """

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: str, arg4: list[numpy.ndarray], arg5: int, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: int, arg4: str, arg5: list[numpy.ndarray], arg6: int, /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: list[numpy.ndarray], /) -> None: ...

    @overload
    def __init__(self, arg0: _tychopy.vector_functions.VectorFunction, arg1: int, arg2: int, arg3: int, arg4: list[numpy.ndarray], /) -> None: ...

    @overload
    def __init__(self, arg0: int, arg1: list[numpy.ndarray], arg2: int, /) -> None:
        """
        Construct a state-only table without an ODE, deriving derivatives by finite difference.

        Parameters
        ----------
        x_vars : int
            Number of state variables (the last element of each node is taken as
            time).
        traj : list of numpy.ndarray
            Trajectory nodes packed as ``[x, t]``, in strictly increasing time
            order.
        num_segments : int
            Number of defect intervals to resample the trajectory onto.
        """

    @overload
    def __init__(self, arg: list[numpy.ndarray], /) -> None:
        """
        Construct a state-only table, inferring dimensions from the data.

        The last component of each node vector is treated as time. Node
        derivatives are estimated by finite difference and the table is
        resampled onto ``len(traj) - 1`` segments.

        Parameters
        ----------
        traj : list of numpy.ndarray
            Trajectory nodes packed as ``[x, t]``, in strictly increasing time
            order. The state dimension is inferred as ``len(traj[0]) - 1``.
        """

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

    def get_table_ptr(self) -> LGLInterpTable:
        """
        Return a non-owning shared pointer aliasing this table.

        Used internally so that phase components (e.g. interpolation functions)
        can hold a reference to the table without taking ownership of it.

        Returns
        -------
        LGLInterpTable
            A shared pointer that aliases ``self`` with no ownership transfer.
        """

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

    def new_error_integral(self) -> list[numpy.ndarray]:
        """
        Compute the updated mesh-error integral over the trajectory.

        Estimates the per-segment discretization error using the highest-order
        polynomial coefficient of the interpolant (the de Boor formulation),
        then integrates it over the non-dimensional time axis.

        Returns
        -------
        list of numpy.ndarray
            A three-element list ``[tsnd, errs2, errint]``:

            * ``tsnd`` -- non-dimensional segment boundary times in ``[0, 1]``
              (length ``num_segments + 1``).
            * ``errs2`` -- per-segment absolute error estimate weighted by
              ``h^(order+1)`` (same length as ``tsnd``).
            * ``errint`` -- cumulative error integral over ``tsnd`` (same
              length as ``tsnd``).
        """

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

    def error_integral(self, arg: int, /) -> list[numpy.ndarray]:
        """
        Compute the cumulative ODE residual error integral over the trajectory.

        Evaluates the ODE at ``num_samps`` uniformly-spaced times, compares the
        ODE output to the interpolated state derivative, and accumulates the
        mismatch to produce a cumulative error profile. Requires that the table
        was constructed with an ODE.

        Parameters
        ----------
        num_samps : int
            Number of uniformly-spaced time samples at which to evaluate the
            ODE residual.

        Returns
        -------
        list of numpy.ndarray
            One two-element array per sample, ``[cumulative_error, time]``,
            where the cumulative error is the running integral of the
            ``(order+1)``-th root of the ODE residual norm.
        """

    @property
    def t0(self) -> float:
        """float: initial time of the table."""

    @property
    def tf(self) -> float:
        """float: final time of the table."""

    def interp_non_dim(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]:
        """
        Resample the trajectory onto an evenly-spaced mesh over a fractional sub-range.

        Uses the table's default within-block node spacing and resamples
        ``num_segments`` equally-sized defect intervals that span the fraction
        ``[low, high]`` of the full trajectory time range.

        Parameters
        ----------
        num_segments : int
            Number of defect intervals to produce.
        low : float
            Lower fractional time bound in ``[0, 1]`` (0 = ``t0``, 1 = ``tf``).
        high : float
            Upper fractional time bound in ``[0, 1]``.

        Returns
        -------
        list of numpy.ndarray
            Resampled trajectory nodes packed as ``[x, t, u]``, ready to be
            passed to a phase as an initial trajectory.
        """

class InterpFunction:
    """
    VectorFunction that interpolates selected trajectory variables over time.

    Maps a scalar time input to the interpolated values of the requested
    state/time/control variables read from an :class:`LGLInterpTable`, with
    analytic first and second derivatives.  The compile-time output dimension
    ``OR`` is the number of variables interpolated; use the dynamic variant
    ``InterpFunction`` (``OR == -1``) when the selection is not known at
    compile time.

    Notes
    -----
    Instances are typically obtained from a solved phase's interpolation table
    rather than constructed directly.  The ``__call__`` operator composes this
    function with a scalar VectorFunction of time so that the interpolated
    trajectory can be embedded directly into NLP expressions.

    See Also
    --------
    LGLInterpTable, PhaseInterface
    """

    def __init__(self, arg0: LGLInterpTable, arg1: numpy.ndarray, /) -> None:
        """
        Construct interpolating an explicit subset of trajectory variables.

        Parameters
        ----------
        table : LGLInterpTable
            The trajectory interpolation table to read from.
        vars : numpy.ndarray of int
            Indices (into the packed ``[x, t, u]`` node vector) of the variables
            to interpolate.

        Raises
        ------
        ValueError
            If any index in ``vars`` exceeds the table's ``xtu_vars`` dimension.
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
    def __call__(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> _tychopy.vector_functions.VectorFunction:
        """
        Compose this interpolant with a scalar-valued time function.

        Parameters
        ----------
        t : ScalarFunction
            A scalar VectorFunction whose single output is interpreted as time.

        Returns
        -------
        VectorFunction
            A new VectorFunction that evaluates the interpolant at the time
            produced by ``t``.
        """

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
    """
    VectorFunction that interpolates selected trajectory variables over time.

    Maps a scalar time input to the interpolated values of the requested
    state/time/control variables read from an :class:`LGLInterpTable`, with
    analytic first and second derivatives.  The compile-time output dimension
    ``OR`` is the number of variables interpolated; use the dynamic variant
    ``InterpFunction`` (``OR == -1``) when the selection is not known at
    compile time.

    Notes
    -----
    Instances are typically obtained from a solved phase's interpolation table
    rather than constructed directly.  The ``__call__`` operator composes this
    function with a scalar VectorFunction of time so that the interpolated
    trajectory can be embedded directly into NLP expressions.

    See Also
    --------
    LGLInterpTable, PhaseInterface
    """

    def __init__(self, arg: LGLInterpTable, /) -> None:
        """
        Construct interpolating all state variables of the trajectory table.

        Parameters
        ----------
        table : LGLInterpTable
            The trajectory interpolation table to read from.  Its state dimension
            must match the compile-time output size ``OR``, and it must contain no
            control variables.

        Raises
        ------
        ValueError
            If the table's state size differs from ``OR`` or it has control variables.
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
    def __call__(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> _tychopy.vector_functions.VectorFunction:
        """
        Compose this interpolant with a scalar-valued time function.

        Parameters
        ----------
        t : ScalarFunction
            A scalar VectorFunction whose single output is interpreted as time.

        Returns
        -------
        VectorFunction
            A new VectorFunction that evaluates the interpolant at the time
            produced by ``t``.
        """

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
    """
    VectorFunction that interpolates selected trajectory variables over time.

    Maps a scalar time input to the interpolated values of the requested
    state/time/control variables read from an :class:`LGLInterpTable`, with
    analytic first and second derivatives.  The compile-time output dimension
    ``OR`` is the number of variables interpolated; use the dynamic variant
    ``InterpFunction`` (``OR == -1``) when the selection is not known at
    compile time.

    Notes
    -----
    Instances are typically obtained from a solved phase's interpolation table
    rather than constructed directly.  The ``__call__`` operator composes this
    function with a scalar VectorFunction of time so that the interpolated
    trajectory can be embedded directly into NLP expressions.

    See Also
    --------
    LGLInterpTable, PhaseInterface
    """

    def __init__(self, arg: LGLInterpTable, /) -> None:
        """
        Construct interpolating all state variables of the trajectory table.

        Parameters
        ----------
        table : LGLInterpTable
            The trajectory interpolation table to read from.  Its state dimension
            must match the compile-time output size ``OR``, and it must contain no
            control variables.

        Raises
        ------
        ValueError
            If the table's state size differs from ``OR`` or it has control variables.
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
    def __call__(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> _tychopy.vector_functions.VectorFunction:
        """
        Compose this interpolant with a scalar-valued time function.

        Parameters
        ----------
        t : ScalarFunction
            A scalar VectorFunction whose single output is interpreted as time.

        Returns
        -------
        VectorFunction
            A new VectorFunction that evaluates the interpolant at the time
            produced by ``t``.
        """

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
    """
    VectorFunction that interpolates selected trajectory variables over time.

    Maps a scalar time input to the interpolated values of the requested
    state/time/control variables read from an :class:`LGLInterpTable`, with
    analytic first and second derivatives.  The compile-time output dimension
    ``OR`` is the number of variables interpolated; use the dynamic variant
    ``InterpFunction`` (``OR == -1``) when the selection is not known at
    compile time.

    Notes
    -----
    Instances are typically obtained from a solved phase's interpolation table
    rather than constructed directly.  The ``__call__`` operator composes this
    function with a scalar VectorFunction of time so that the interpolated
    trajectory can be embedded directly into NLP expressions.

    See Also
    --------
    LGLInterpTable, PhaseInterface
    """

    def __init__(self, arg: LGLInterpTable, /) -> None:
        """
        Construct interpolating all state variables of the trajectory table.

        Parameters
        ----------
        table : LGLInterpTable
            The trajectory interpolation table to read from.  Its state dimension
            must match the compile-time output size ``OR``, and it must contain no
            control variables.

        Raises
        ------
        ValueError
            If the table's state size differs from ``OR`` or it has control variables.
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
    def __call__(self, arg: _tychopy.vector_functions.ScalarFunction, /) -> _tychopy.vector_functions.VectorFunction:
        """
        Compose this interpolant with a scalar-valued time function.

        Parameters
        ----------
        t : ScalarFunction
            A scalar VectorFunction whose single output is interpreted as time.

        Returns
        -------
        VectorFunction
            A new VectorFunction that evaluates the interpolant at the time
            produced by ``t``.
        """

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
    """
    Finite-difference differentiator for non-uniformly-sampled vector data.

    Computes per-sample derivatives of a sequence of vector samples with
    respect to a chosen independent axis. The samples do not need to be
    equally spaced: stencil weights are solved from the actual node offsets
    at each evaluation point via a Vandermonde system, using forward,
    central, or backward stencils near the boundaries.

    Parameters
    ----------
    axis : int
        Index of the independent axis (e.g. time) within each sample vector.
    data : list of numpy.ndarray
        The sampled data series.
    """

    def __init__(self, arg0: int, arg1: list[numpy.ndarray], /) -> None:
        """
        Construct a differentiator with the given axis index and data series.

        Parameters
        ----------
        axis : int
            Index of the independent axis within each sample vector.
        data : list of numpy.ndarray
            The sampled data series to differentiate.
        """

    def all_derivs(self, arg0: int, arg1: int, /) -> list[numpy.ndarray]:
        """
        Compute finite-difference derivatives at every sample index.

        Parameters
        ----------
        order : int
            Order of the derivative to compute (e.g. ``1`` for first derivative).
        accuracy : int
            Finite-difference accuracy order (rounded up to the nearest even
            number internally).

        Returns
        -------
        list of numpy.ndarray
            One derivative vector per sample, in the same order as the input data.
        """

    def deriv(self, arg0: int, arg1: int, arg2: int, /) -> numpy.ndarray:
        """
        Compute the finite-difference derivative at a single sample index.

        Parameters
        ----------
        i : int
            Zero-based sample index at which to evaluate the derivative.
        order : int
            Order of the derivative to compute.
        accuracy : int
            Finite-difference accuracy order (rounded up to the nearest even
            number internally).

        Returns
        -------
        numpy.ndarray
            The derivative vector at sample index ``i``.
        """

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
    Build the ODE argument set and slice out named state/control expressions::

        XtU = ODEArguments(3, 1)        # 3 states, 1 control
        x, y, v = XtU.x_vec().tolist()
        theta = XtU.u_var(0)
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
