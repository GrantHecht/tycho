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
    LGL3 = 0

    LGL5 = 1

    LGL7 = 2

    Trapezoidal = 3

    CentralShooting = 4

class IntegralModes(enum.Enum):
    BaseIntegral = 0

    TrapIntegral = 2

class ControlModes(enum.Enum):
    HighestOrderSpline = 0

    FirstOrderSpline = 1

    NoSpline = 2

    BlockConstant = 3

class PhaseRegionFlags(enum.Enum):
    Front = 1

    Back = 2

    Path = 5

    NodalPath = 7

    FrontandBack = 3

    BackandFront = 4

    Params = 12

    InnerPath = 6

    ODEParams = 13

    StaticParams = 14

    PairWisePath = 9

class ScaleModes(enum.Enum):
    AUTO = 0

    CUSTOM = 1

    NONE = 2

class MeshErrorEstimators(enum.Enum):
    DEBOOR = 0

    INTEGRATOR = 1

class MeshErrorAggregation(enum.Enum):
    MAX = 0

    AVG = 1

    GEOMETRIC = 2

    ENDTOEND = 3

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
    @property
    def times(self) -> numpy.ndarray: ...

    @property
    def error(self) -> numpy.ndarray: ...

    @property
    def distribution(self) -> numpy.ndarray: ...

    @property
    def distintegral(self) -> numpy.ndarray: ...

    @property
    def avg_error(self) -> float: ...

    @property
    def max_error(self) -> float: ...

    @property
    def numsegs(self) -> int: ...

    @property
    def converged(self) -> bool: ...

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
    """Base Class for All Optimal Control Phases"""

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
    def set_traj(self, arg: list[numpy.ndarray], /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg0: TranscriptionModes, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg: TranscriptionModes, /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg0: str, arg1: numpy.ndarray, arg2: numpy.ndarray, /) -> None: ...

    @overload
    def switch_transcription_mode(self, arg: str, /) -> None: ...

    def transcribe(self, arg0: bool, arg1: bool, /) -> None: ...

    @overload
    def refine_traj_manual(self, arg: int, /) -> None: ...

    @overload
    def refine_traj_manual(self, arg0: numpy.ndarray, arg1: numpy.ndarray, /) -> None: ...

    def refine_traj_equal(self, arg: int, /) -> list[numpy.ndarray]: ...

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

    def return_traj(self) -> list[numpy.ndarray]: ...

    def return_traj_range(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]: ...

    def return_traj_range_nd(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]: ...

    def return_traj_table(self) -> LGLInterpTable: ...

    def return_costate_traj(self) -> list[numpy.ndarray]: ...

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

    def add_boundary_value(self, phase_region: PhaseRegionFlags | str, index: int | Sequence[int] | str | list[str], value: float | numpy.ndarray | Sequence[float], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

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

    def add_lu_var_bound(self, phase_region: PhaseRegionFlags | str, var: int | Sequence[int] | str | list[str], lowerbound: float, upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_lower_var_bound(self, phase_region: PhaseRegionFlags | str, var: int | Sequence[int] | str | list[str], lowerbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_upper_var_bound(self, phase_region: PhaseRegionFlags | str, var: int | Sequence[int] | str | list[str], upperbound: float, scale: float = 1.0, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

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
    def add_state_objective(self, phase_region: PhaseRegionFlags | str, func: _tychopy.vector_functions.ScalarFunction, input_index: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_state_objective(self, arg: StateObjective, /) -> int: ...

    def add_value_objective(self, phase_region: PhaseRegionFlags | str, var: int | Sequence[int] | str | list[str], scale: float, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_delta_var_objective(self, var: int | Sequence[int] | str | list[str], scale: float, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def add_delta_time_objective(self, var: float, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_integral_objective(self, func: _tychopy.vector_functions.ScalarFunction, xt_u_vars: int | Sequence[int] | str | list[str], op_vars: int | Sequence[int] | str | list[str], sp_vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_integral_objective(self, func: _tychopy.vector_functions.ScalarFunction, input_index: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

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

    def refine_traj_auto(self) -> None: ...

    def calc_global_error(self) -> numpy.ndarray: ...

    def get_mesh_iters(self) -> list[MeshIterateInfo]: ...

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
    def __init__(self) -> None: ...

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

    def add_forward_link_equal_con(self, phase0: int | PhaseInterface | str, phase1: int | PhaseInterface | str, vars: int | Sequence[int] | str | list[str], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> list[int]: ...

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
    def add_link_objective(self, func: _tychopy.vector_functions.ScalarFunction, phase0: int | PhaseInterface | str, reg0: PhaseRegionFlags | str, v0: int | Sequence[int] | str | list[str], phase1: int | PhaseInterface | str, reg1: PhaseRegionFlags | str, v1: int | Sequence[int] | str | list[str], linkparams: int | Sequence[int] | str | list[str] = ..., auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_objective(self, func: _tychopy.vector_functions.ScalarFunction, link_parms: Sequence[numpy.ndarray], auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    @overload
    def add_link_param_objective(self, func: _tychopy.vector_functions.ScalarFunction, link_parms: numpy.ndarray, auto_scale: Union[float, numpy.ndarray, ScaleModes, str, None] | None = 'auto') -> int: ...

    def remove_link_equal_con(self, arg: int, /) -> None: ...

    def remove_link_inequal_con(self, arg: int, /) -> None: ...

    def remove_link_objective(self, arg: int, /) -> None: ...

    def add_phase(self, arg: PhaseInterface, /) -> int: ...

    def add_phases(self, arg: Sequence[PhaseInterface], /) -> list[int]: ...

    def get_phase_num(self, arg: PhaseInterface, /) -> int: ...

    def remove_phase(self, arg: int, /) -> None: ...

    def phase(self, arg: int, /) -> PhaseInterface: ...

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

    def load_even_data(self, arg: list[numpy.ndarray], /) -> None: ...

    def get_table_ptr(self) -> LGLInterpTable: ...

    def load_uneven_data(self, arg0: int, arg1: list[numpy.ndarray], /) -> None: ...

    def interpolate(self, arg: float, /) -> numpy.ndarray: ...

    def new_error_integral(self) -> list[numpy.ndarray]: ...

    def __call__(self, arg: float, /) -> numpy.ndarray: ...

    def interpolate_deriv(self, arg: float, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, 2), order='F')]: ...

    def make_periodic(self) -> None: ...

    def interp_range(self, arg0: int, arg1: float, arg2: float, /) -> list[numpy.ndarray]: ...

    def interp_whole_range(self, arg: int, /) -> list[numpy.ndarray]: ...

    def error_integral(self, arg: int, /) -> list[numpy.ndarray]: ...

    @property
    def t0(self) -> float: ...

    @property
    def tf(self) -> float: ...

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
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    @overload
    def __init__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __init__(self, arg: int, /) -> None: ...

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

    def x_vec(self) -> _tychopy.vector_functions.Segment: ...

    def x_var(self, arg: int, /) -> _tychopy.vector_functions.Element: ...

    def xt_vec(self) -> _tychopy.vector_functions.Segment: ...

    def t_var(self) -> _tychopy.vector_functions.Element: ...

    def u_vec(self) -> _tychopy.vector_functions.Segment: ...

    def u_var(self, arg: int, /) -> _tychopy.vector_functions.Element: ...

    def p_vec(self) -> _tychopy.vector_functions.Segment: ...

    def p_var(self, arg: int, /) -> _tychopy.vector_functions.Element: ...
