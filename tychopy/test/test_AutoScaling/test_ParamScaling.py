import unittest

import numpy as np

import tychopy.optimal_control as oc
from tychopy.vector_functions import Arguments as Args


class ParamODE(oc.ODEBase):
    """Minimal ODE carrying one ODE (dynamics) parameter: xdot = u + p.

    Sizes: 1 state, 1 control, 1 ODE parameter. The full input node vector is
    ``[x, t, u, p]`` (length ``xtu_p_vars`` == 4), so ``xtu_vars`` == 3 and the
    ODE parameter occupies absolute input index 3.
    """

    def __init__(self):
        XVars = 1
        UVars = 1
        PVars = 1

        args = oc.ODEArguments(XVars, UVars, PVars)
        u = args.u_var(0)
        p = args.p_var(0)
        xdot = u + p
        super().__init__(xdot, XVars, UVars, PVars)


class ParamScaling(unittest.TestCase):
    """Characterizes auto-scaling of parameter-region function bindings (OC §1.11).

    The problem has the analytic solution x(t) = 2 t, u = 0.5, once the
    boundary values, the ODE-param pin (p = 1.5), and the static-param pins
    (s0 = 0.7, s1 = 0.9) are enforced and the integral of u^2 is minimized:
    x(0) = 0, x(1) = 2 and xdot = u + p force integral(u) = 0.5, and
    minimizing integral(u^2) makes u constant.

    With auto-scaling ON and deliberately non-unit, all-distinct units, the
    param pins are only satisfied in PHYSICAL values if the auto-scaling
    machinery (calc_auto_scales -> get_input_scale/get_test_inputs -> IOScaled
    wrapping at transcription) scales each parameter input by its own unit:

    * the ODE param by ``xtup_units_[0 + xtu_vars()]`` -- if the +xtu_vars()
      offset were dropped, the pin would resolve to p = 1.5 * 11 / 2 = 8.25,
      failing the assertion;
    * each static param by ``sp_units_[i]`` -- if state units were used
      instead, s0 would resolve to 0.7 * 13 / 2 != 0.7.

    The static-param pins are registered through BOTH registration routes:
    s0 via the ``make_func_impl`` selector path (which remaps the
    ``StaticParams`` selector to the ``Params`` region), and s1 via a
    pre-built ``StateConstraint`` carrying a verbatim ``StaticParams`` region
    flag with its index in ``sp_vars_`` -- valid, and it must scale via
    ``sp_units_`` just the same.
    """

    # ODE sizes (see ParamODE); xtu_vars = XVars + t + UVars = 1 + 1 + 1 = 3.
    XTU_VARS = 3

    # Deliberately non-unit, all-distinct units. Layout: [x, t, u, p_ode].
    XTUP_UNITS = [2.0, 3.0, 5.0, 11.0]
    SP_UNITS = [13.0, 17.0]

    # Pinned values and the analytic solution.
    P_PIN = 1.5
    S0_PIN = 0.7
    S1_PIN = 0.9
    XF = 2.0
    U_STAR = 0.5

    TOL = 1.0e-6

    def _build_phase(self, auto_scale):
        ode = ParamODE()

        nsegs = 16
        traj_ig = [[2.0 * t, t, 0.5, self.P_PIN] for t in np.linspace(0.0, 1.0, 20)]

        phase = ode.phase("LGL3", traj_ig, nsegs)
        phase.set_units(np.array(self.XTUP_UNITS))
        phase.set_static_params(np.array([0.5, 0.4]), np.array(self.SP_UNITS))

        phase.add_boundary_value("Front", [0, 1], [0.0, 0.0])
        phase.add_boundary_value("Back", [1], [1.0])
        phase.add_boundary_value("Back", [0], [self.XF])

        # make_func_impl selector routes: ODEParams and StaticParams selectors
        # are remapped to the Params region with indices in op_vars_/sp_vars_.
        phase.add_boundary_value("ODEParams", [0], [self.P_PIN])
        phase.add_boundary_value("StaticParams", [0], [self.S0_PIN])

        # Direct-construction route: a pre-built StateConstraint with a
        # verbatim StaticParams region flag (bypasses make_func_impl). The
        # index sits in sp_vars_; xtu_vars_ is empty.
        empty = np.array([], dtype=np.int32)
        sp1 = np.array([1], dtype=np.int32)
        pin_s1 = oc.StateConstraint(
            Args(1)[0] - self.S1_PIN,
            oc.PhaseRegionFlags.StaticParams,
            empty,
            empty,
            sp1,
        )
        phase.add_equal_con(pin_s1)

        phase.add_integral_objective(Args(1)[0] ** 2, [2])

        phase.set_auto_scaling(auto_scale)
        phase.set_num_partitions(1)

        if __name__ != "__main__":
            phase.optimizer.print_level = 3

        return phase

    def _check_solution(self, phase):
        traj = phase.return_traj()
        sp = phase.return_static_params()

        # Analytic trajectory: x(1) = 2, u = 0.5 everywhere.
        self.assertAlmostEqual(traj[-1][0], self.XF, delta=self.TOL)
        self.assertAlmostEqual(traj[0][2], self.U_STAR, delta=self.TOL)
        self.assertAlmostEqual(traj[-1][2], self.U_STAR, delta=self.TOL)

        # ODE param pinned in PHYSICAL units on every node row: fails if the
        # +xtu_vars() offset is dropped anywhere in the scaling chain.
        for row in traj:
            self.assertAlmostEqual(row[self.XTU_VARS], self.P_PIN, delta=self.TOL)

        # Static params pinned in PHYSICAL units: s0 through the selector
        # (make_func_impl) route, s1 through the pre-built param-flag route.
        self.assertAlmostEqual(sp[0], self.S0_PIN, delta=self.TOL)
        self.assertAlmostEqual(sp[1], self.S1_PIN, delta=self.TOL)

    def test_param_pins_no_autoscale(self):
        # Ground truth: same problem without auto-scaling.
        phase = self._build_phase(False)
        phase.optimize()
        self._check_solution(phase)

    def test_param_pins_survive_autoscaled_optimize(self):
        # optimize() -> transcribe() -> calc_auto_scales() runs the full
        # auto-scaling machinery over every registered function, including the
        # remapped Params-region pins and the verbatim StaticParams-flag pin.
        phase = self._build_phase(True)
        phase.optimize()
        self._check_solution(phase)

    def test_param_flag_with_xtu_indices_raises(self):
        # A param-region flag with indices in the state-block (xtu) slot is a
        # malformed binding (the phase indexer binds param regions from
        # op_vars/sp_vars only) and must be rejected at construction.
        empty = np.array([], dtype=np.int32)
        one = np.array([0], dtype=np.int32)
        fun = Args(1)[0] - 1.0

        for reg in (
            oc.PhaseRegionFlags.Params,
            oc.PhaseRegionFlags.ODEParams,
            oc.PhaseRegionFlags.StaticParams,
        ):
            with self.subTest(region=reg):
                with self.assertRaises(ValueError):
                    oc.StateConstraint(fun, reg, one, empty, empty)

        # Single-index-group (3-arg) ctor: a Params flag stores the indices in
        # xtu_vars_ verbatim (no remap is possible -- a combined parameter
        # vector index cannot be split into op/sp without phase dimensions),
        # so it must be rejected too.
        with self.assertRaises(ValueError):
            oc.StateConstraint(fun, oc.PhaseRegionFlags.Params, one)

        # ODEParams/StaticParams single-group selectors remap the indices into
        # op_vars_/sp_vars_ and must remain constructible.
        oc.StateConstraint(fun, oc.PhaseRegionFlags.ODEParams, one)
        oc.StateConstraint(fun, oc.PhaseRegionFlags.StaticParams, one)


if __name__ == "__main__":
    unittest.main(exit=False)
