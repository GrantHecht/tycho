import unittest

import numpy as np

import tychopy.optimal_control as oc
import tychopy.vector_functions as vf


class ParamODE(oc.ODEBase):
    """Minimal ODE carrying a single ODE (dynamics) parameter.

    Sizes: 2 states, 1 control, 1 ODE parameter. The full input node vector is
    ``[x0, x1, t, u, p]`` (length ``xtu_p_vars`` == 5), so ``xtu_vars`` == 4 and
    the ODE parameter sits at absolute input index 4.
    """

    def __init__(self):
        XVars = 2
        UVars = 1
        PVars = 1

        args = oc.ODEArguments(XVars, UVars, PVars)
        x1 = args.x_var(1)
        u = args.u_var(0)
        p = args.p_var(0)
        xdot = vf.stack([x1, u * p])
        super().__init__(xdot, XVars, UVars, PVars)


class ParamScaling(unittest.TestCase):
    """Characterizes auto-scaling of parameter-region function bindings (OC §1.11).

    ``ODEPhaseBase::get_input_scale`` draws its per-input scale factors from
    ``xtup_units_`` (state/time/control/ODE-param block) and ``sp_units_``
    (static-param block). ``make_func_impl`` remaps any ``ODEParams`` /
    ``StaticParams`` selector to the ``Params`` region, placing the parameter
    indices in ``op_vars_`` / ``sp_vars_`` (never in the state-indexed
    ``xtu_vars_``). These tests lock in that:

      * a static-parameter input is scaled by ``sp_units_[i]``;
      * an ODE-parameter input is scaled by ``xtup_units_[i + xtu_vars()]`` --
        i.e. the ``+ xtu_vars()`` offset is applied, so a param is never
        mis-scaled by a state unit.
    """

    # ODE sizes (see ParamODE); xtu_vars = XVars + t + UVars = 2 + 1 + 1 = 4.
    XVARS = 2
    UVARS = 1
    PVARS = 1
    XTU_VARS = XVARS + 1 + UVARS  # 4
    XTU_P_VARS = XTU_VARS + PVARS  # 5

    # Deliberately non-unit, all-distinct units so any mis-indexing is caught.
    # Layout matches the node vector: [x0, x1, t, u, p_ode].
    XTUP_UNITS = [2.0, 3.0, 5.0, 7.0, 11.0]
    SP_VALUES = [0.5, 0.9]
    SP_UNITS = [13.0, 17.0]

    def _build_phase(self):
        ode = ParamODE()

        nsegs = 4
        traj_ig = [[1.0, 0.0, t, 0.4, 0.0] for t in np.linspace(0.0, 1.0, 20)]

        phase = ode.phase("LGL3", traj_ig, nsegs)
        phase.set_units(np.array(self.XTUP_UNITS))
        phase.set_static_params(np.array(self.SP_VALUES), np.array(self.SP_UNITS))
        return phase

    @staticmethod
    def _empty():
        return np.array([], dtype=np.int32)

    @staticmethod
    def _idx(*values):
        return np.array(list(values), dtype=np.int32)

    def test_static_param_autoscale_uses_sp_units(self):
        phase = self._build_phase()

        # Register an AUTO-scaled StaticParams objective: make_func_impl remaps the
        # StaticParams selector to the Params region with the index in sp_vars_.
        phase.add_value_objective("StaticParams", 0, 1.0)

        # The auto-scaler computes the static-param input scale from sp_units_.
        scales = phase.get_input_scale(
            oc.PhaseRegionFlags.Params, self._empty(), self._empty(), self._idx(0, 1)
        )

        self.assertAlmostEqual(scales[0], self.SP_UNITS[0], places=10)
        self.assertAlmostEqual(scales[1], self.SP_UNITS[1], places=10)

    def test_ode_param_autoscale_uses_xtu_offset(self):
        phase = self._build_phase()

        # The ODE-param input scale is xtup_units_[i + xtu_vars()], not the
        # state-slot xtup_units_[i]; the +xtu_vars() offset must be applied.
        scales = phase.get_input_scale(
            oc.PhaseRegionFlags.Params, self._empty(), self._idx(0), self._empty()
        )

        expected = self.XTUP_UNITS[self.XTU_VARS]  # 11.0, the ODE-param unit
        wrong_if_offset_dropped = self.XTUP_UNITS[0]  # 2.0, a state unit

        self.assertAlmostEqual(scales[0], expected, places=10)
        self.assertNotAlmostEqual(scales[0], wrong_if_offset_dropped, places=10)

    def test_mixed_param_scale_ordering(self):
        phase = self._build_phase()

        # OP indices are packed before SP indices (get_input_scale loop order).
        scales = phase.get_input_scale(
            oc.PhaseRegionFlags.Params, self._empty(), self._idx(0), self._idx(1)
        )

        self.assertAlmostEqual(scales[0], self.XTUP_UNITS[self.XTU_VARS], places=10)
        self.assertAlmostEqual(scales[1], self.SP_UNITS[1], places=10)


if __name__ == "__main__":
    unittest.main(exit=False)
