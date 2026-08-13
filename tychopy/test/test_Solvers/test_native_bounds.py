"""Coverage for the native-bounds Settings surface: InteriorPointSolver's
``fixed_variable_treatment``, ``bound_relax_factor``, and
``bound_interval_push`` properties (bound onto ``InteriorPointSolver::Settings::
fixed_variable_treatment_`` / ``bound_relax_factor_`` / ``bound_interval_push_``
in ``include/tycho/detail/solvers/interior_point_solver.h``, validated through
``set_fixed_variable_treatment`` / ``set_bound_relax_factor`` /
``set_bound_interval_push`` in ``src/solvers/interior_point_solver_settings.cpp``), the
``FixedVariableTreatments`` enum, and the phase-level variable-bound
declarations (``add_lu_var_bound``/``add_lower_var_bound``/
``add_upper_var_bound``), which are staged onto the solver's variable-bound
contract by ``ODEPhaseBase::transcribe_var_bounds`` in
``src/optimal_control/ode_phase_base.cpp`` rather than becoming inequality
rows.

Each property round-trips within its documented range and rejects an
out-of-range value with ``ValueError`` (the validated setter raises
immediately on assignment, mirroring every other validated Settings field --
see test_interior_point_solver_globalization_settings.py's ``test_MaxSocRoundTrip`` etc.).
"""

import numpy as np
import pytest

import tychopy as typy

vf = typy.vector_functions
oc = typy.optimal_control
solvs = typy.solvers


# =============================================================================
# fixed_variable_treatment / bound_relax_factor / bound_interval_push
# =============================================================================


def test_fixed_variable_treatment_default_is_make_parameter():
    opt = solvs.InteriorPointSolver()
    assert opt.fixed_variable_treatment == solvs.FixedVariableTreatments.MakeParameter


def test_fixed_variable_treatment_round_trip():
    opt = solvs.InteriorPointSolver()
    opt.fixed_variable_treatment = solvs.FixedVariableTreatments.MakeConstraint
    assert opt.fixed_variable_treatment == solvs.FixedVariableTreatments.MakeConstraint
    opt.fixed_variable_treatment = solvs.FixedVariableTreatments.RelaxBounds
    assert opt.fixed_variable_treatment == solvs.FixedVariableTreatments.RelaxBounds
    opt.fixed_variable_treatment = solvs.FixedVariableTreatments.MakeParameter
    assert opt.fixed_variable_treatment == solvs.FixedVariableTreatments.MakeParameter


def test_fixed_variable_treatment_enum_importable():
    # The three values named in the brief must exist under solvers and be
    # distinct.
    values = {
        solvs.FixedVariableTreatments.MakeParameter,
        solvs.FixedVariableTreatments.MakeConstraint,
        solvs.FixedVariableTreatments.RelaxBounds,
    }
    assert len(values) == 3


def test_fixed_variable_treatment_rejects_raw_int_assignment():
    opt = solvs.InteriorPointSolver()
    with pytest.raises(TypeError):
        opt.fixed_variable_treatment = 7


def test_fixed_variable_treatment_invalid_raw_value_rejected():
    with pytest.raises(ValueError):
        solvs.FixedVariableTreatments(99)


def test_bound_relax_factor_default():
    opt = solvs.InteriorPointSolver()
    assert opt.bound_relax_factor == pytest.approx(1.0e-8)


def test_bound_relax_factor_round_trip():
    opt = solvs.InteriorPointSolver()
    opt.bound_relax_factor = 1.0e-4
    assert opt.bound_relax_factor == pytest.approx(1.0e-4)
    # Both ends of the closed range [0, 1e-2] are accepted.
    opt.bound_relax_factor = 0.0
    assert opt.bound_relax_factor == 0.0
    opt.bound_relax_factor = 1.0e-2
    assert opt.bound_relax_factor == pytest.approx(1.0e-2)


def test_bound_relax_factor_rejects_out_of_range():
    opt = solvs.InteriorPointSolver()
    with pytest.raises(ValueError, match="bound_relax_factor"):
        opt.bound_relax_factor = -1.0e-9
    # Rejected write must not clobber the prior valid (default) value.
    assert opt.bound_relax_factor == pytest.approx(1.0e-8)

    with pytest.raises(ValueError, match="bound_relax_factor"):
        opt.bound_relax_factor = 2.0e-2
    assert opt.bound_relax_factor == pytest.approx(1.0e-8)


def test_bound_interval_push_default():
    opt = solvs.InteriorPointSolver()
    assert opt.bound_interval_push == pytest.approx(1.0e-2)


def test_bound_interval_push_round_trip():
    opt = solvs.InteriorPointSolver()
    opt.bound_interval_push = 0.25
    assert opt.bound_interval_push == pytest.approx(0.25)
    opt.bound_interval_push = 1.0e-2
    assert opt.bound_interval_push == pytest.approx(1.0e-2)


def test_bound_interval_push_rejects_out_of_range():
    opt = solvs.InteriorPointSolver()
    # The interval is open, so both endpoints (0.0 and 0.5) are rejected too.
    with pytest.raises(ValueError, match="bound_interval_push"):
        opt.bound_interval_push = 0.0
    assert opt.bound_interval_push == pytest.approx(1.0e-2)

    with pytest.raises(ValueError, match="bound_interval_push"):
        opt.bound_interval_push = 0.5
    assert opt.bound_interval_push == pytest.approx(1.0e-2)

    with pytest.raises(ValueError, match="bound_interval_push"):
        opt.bound_interval_push = -0.1
    assert opt.bound_interval_push == pytest.approx(1.0e-2)


# =============================================================================
# Variable bounds end to end through a phase problem
# =============================================================================


class _HarmonicOscillatorODE(oc.ODEBase):
    """Simple harmonic oscillator: x'' = -x. State (x, v), no controls --
    the cheapest phase fixture that reaches ODEPhaseBase's variable-bound
    declaration path (add_lu_var_bound -> transcribe_var_bounds) without an
    objective or a solve. Mirrors test_ErrorMessages.py's SHO_Ode.
    """

    def __init__(self):
        XVars = 2
        args = oc.ODEArguments(XVars)
        x = args.x_var(0)
        v = args.x_var(1)
        xdot = v
        vdot = (-1.0) * x
        ode = vf.stack([xdot, vdot])
        super().__init__(ode, XVars)


def _make_oscillator_phase():
    ode = _HarmonicOscillatorODE()
    traj_ig = [np.array([np.cos(t), -np.sin(t), t]) for t in np.linspace(0.0, 1.0, 20)]
    return ode.phase("LGL3", traj_ig, 4)


class _BrachistochroneODE(oc.ODEBase):
    """Classic brachistochrone dynamics, mirroring
    examples/python_examples/Brachistochrone.py. State (x, y, v), single
    control theta.
    """

    def __init__(self, g):
        XVars = 3
        UVars = 1
        args = oc.ODEArguments(XVars, UVars)
        x, y, v = args.x_vec().tolist()
        theta = args.u_var(0)
        xdot = vf.sin(theta) * v
        ydot = -1.0 * vf.cos(theta) * v
        vdot = g * vf.cos(theta)
        ode = vf.stack([xdot, ydot, vdot])
        super().__init__(ode, XVars, UVars)


# Control bound applied to the phase below; the harder case for the barrier
# since the bound is inactive at the brachistochrone optimum, so every
# committed iterate has to stay strictly inside a box it never needs to
# touch.
_CONTROL_BOUND = (-0.1, 2.0)


def _make_bounded_brachistochrone_phase():
    g = 9.81
    ode = _BrachistochroneODE(g)

    x0, y0, v0, theta0 = 0.0, 10.0, 0.0, 1.0
    xf, yf = 10.0, 5.0
    tf = 1.0

    traj_ig = []
    for t in np.linspace(0.0, tf, 20):
        X = np.zeros(5)
        X[0] = x0 + (xf - x0) * t / tf
        X[1] = y0 + (yf - y0) * t / tf
        X[2] = g * t * np.cos(theta0)
        X[3] = t
        X[4] = theta0
        traj_ig.append(X)

    phase = ode.phase("LGL3", traj_ig, 16)
    phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0.0])
    lower, upper = _CONTROL_BOUND
    phase.add_lu_var_bound("Path", 4, lower, upper)
    phase.add_boundary_value("Back", [0, 1], [xf, yf])
    phase.add_delta_time_objective(1.0)
    phase.optimizer.print_level = 3  # fully silent
    return phase


def test_bounded_phase_solves_and_returns_in_bounds_trajectory():
    phase = _make_bounded_brachistochrone_phase()
    flag = phase.optimize()
    assert flag == solvs.ConvergenceFlags.CONVERGED

    traj = np.array(phase.return_traj())
    control = traj[:, 4]  # packed layout is [x, y, v, t, theta]

    lower, upper = _CONTROL_BOUND
    # bound_relax_factor widens the recorded box by a small factor before the
    # barrier divides by it, so allow a small margin rather than requiring
    # exact closure at the declared edge.
    margin = 1.0e-6
    assert np.all(control >= lower - margin)
    assert np.all(control <= upper + margin)


def test_bound_declarations_take_no_scale_argument():
    # A variable bound adds no constraint row, so there is no output for a
    # constraint scale to act on and none of the three declarations accept one.
    phase = _make_oscillator_phase()
    with pytest.raises(TypeError):
        phase.add_lu_var_bound("Front", 0, -1.0, 1.0, 1.0)
    with pytest.raises(TypeError):
        phase.add_lower_var_bound("Front", 0, -1.0, 1.0)
    with pytest.raises(TypeError):
        phase.add_upper_var_bound("Front", 0, 1.0, 1.0)


def test_conflicting_bound_declaration_raises_value_error_naming_the_variable():
    phase = _make_oscillator_phase()
    phase.add_lu_var_bound("Front", 0, 5.0, 10.0)
    phase.add_lu_var_bound("Front", 0, 20.0, 30.0)

    with pytest.raises(ValueError, match=r"variable index 0"):
        phase.transcribe(False, False)
