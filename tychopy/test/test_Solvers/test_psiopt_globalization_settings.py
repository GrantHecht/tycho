"""Coverage for PSIOPT's globalization-knob bindings.

Exercises the Python surface for the five new PSIOPT.Settings fields
(``acceptance_strategy``, ``merit_penalty_rule``, ``max_soc``,
``ls_extended_iters``, ``watchdog``) and the three new SolveResult
diagnostics (``last_soc_steps``, ``last_watchdog_activations``,
``last_recovery_depth_histogram``): property round-trips, per-field
validation, invalid enum construction, and the
``acceptance_strategy=merit`` + classic-path-recovery combo guard in
``Settings::validate()`` (src/solvers/psiopt.cpp).
"""

import unittest

import _tychopy as ast

solvs = ast.solvers
vf = ast.vector_functions
Args = vf.Arguments


def RosenBrockObj(xy=Args(2)):
    x = xy[0]
    y = xy[1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def DiskCon():
    return Args(2).squared_norm() - 2.0


def _make_problem():
    """Smallest available problem that can drive PSIOPT::run_phase_sequence
    (and therefore Settings::validate()) via optimize()/solve(). Mirrors
    test_psiopt_init_time.py's RosenBrock+disk-constraint problem -- the repo
    has no standalone double-integrator OCP fixture, and this static NLP is
    equally fast and sufficient to reach validate() before any iteration.
    """
    prob = solvs.OptimizationProblem()
    prob.set_vars([-1, -1])
    prob.add_objective(RosenBrockObj(), [0, 1])
    prob.add_inequal_con(DiskCon(), [0, 1])
    prob.optimizer.print_level = 3  # fully silent
    return prob


class test_AcceptanceStrategyRoundTrip(unittest.TestCase):
    def test_default_is_classic_merit(self):
        prob = _make_problem()
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.classic_merit
        )

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.merit
        )
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.classic_merit
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.classic_merit
        )


class test_MeritPenaltyRuleRoundTrip(unittest.TestCase):
    def test_default_is_wmno(self):
        prob = _make_problem()
        self.assertEqual(
            prob.optimizer.merit_penalty_rule, solvs.MeritPenaltyRules.wmno
        )

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.merit_penalty_rule = solvs.MeritPenaltyRules.flexible
        self.assertEqual(
            prob.optimizer.merit_penalty_rule, solvs.MeritPenaltyRules.flexible
        )
        prob.optimizer.merit_penalty_rule = solvs.MeritPenaltyRules.wmno
        self.assertEqual(
            prob.optimizer.merit_penalty_rule, solvs.MeritPenaltyRules.wmno
        )


class test_MaxSocRoundTrip(unittest.TestCase):
    def test_default_is_zero(self):
        prob = _make_problem()
        self.assertEqual(prob.optimizer.max_soc, 0)

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.max_soc = 4
        self.assertEqual(prob.optimizer.max_soc, 4)

    def test_rejects_negative(self):
        prob = _make_problem()
        with self.assertRaises(ValueError):
            prob.optimizer.max_soc = -1
        # Rejected write must not clobber the prior valid value.
        self.assertEqual(prob.optimizer.max_soc, 0)


class test_LsExtendedItersRoundTrip(unittest.TestCase):
    def test_default_is_zero(self):
        prob = _make_problem()
        self.assertEqual(prob.optimizer.ls_extended_iters, 0)

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.ls_extended_iters = 3
        self.assertEqual(prob.optimizer.ls_extended_iters, 3)

    def test_rejects_negative(self):
        prob = _make_problem()
        with self.assertRaises(ValueError):
            prob.optimizer.ls_extended_iters = -1
        self.assertEqual(prob.optimizer.ls_extended_iters, 0)


class test_WatchdogRoundTrip(unittest.TestCase):
    def test_default_is_false(self):
        prob = _make_problem()
        self.assertEqual(prob.optimizer.watchdog, False)

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.watchdog = True
        self.assertEqual(prob.optimizer.watchdog, True)
        prob.optimizer.watchdog = False
        self.assertEqual(prob.optimizer.watchdog, False)


class test_BadEnumValues(unittest.TestCase):
    """nanobind's enum constructor validates a raw int against the
    registered members and raises ValueError on mismatch (mirroring
    Python stdlib Enum semantics) -- see dep/nanobind/src/nb_enum.cpp.
    """

    def test_acceptance_strategy_invalid_raw_value_rejected(self):
        with self.assertRaises(ValueError):
            solvs.AcceptanceStrategies(99)

    def test_merit_penalty_rule_invalid_raw_value_rejected(self):
        with self.assertRaises(ValueError):
            solvs.MeritPenaltyRules(99)

    def test_enum_property_rejects_raw_int_assignment(self):
        # Assigning a raw int to the enum-typed PROPERTY goes through
        # nanobind's convert path, which rejects non-member values with
        # TypeError (a different path than the ValueError-raising enum
        # constructor above) -- pin both exception types.
        opt = solvs.PSIOPT()
        with self.assertRaises(TypeError):
            opt.acceptance_strategy = 7
        with self.assertRaises(TypeError):
            opt.merit_penalty_rule = 7


class test_AcceptanceMeritRecoveryComboGuard(unittest.TestCase):
    """Settings::validate() rejects acceptance_strategy=merit combined with
    max_soc > 0 or ls_extended_iters > 0: the modern merit acceptance path
    does not implement the classic-path recovery links (SOC / extended
    backtracking re-drive the fused classic line search). validate() runs
    at PSIOPT::run_phase_sequence() entry -- i.e. on the very first call to
    optimize()/solve(), before any iteration -- so triggering it only
    requires a single solve() call on an otherwise-trivial problem.
    """

    def test_max_soc_combo_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        prob.optimizer.max_soc = 4
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("max_soc", msg)
        self.assertIn("ls_extended_iters", msg)
        # Validation fires before any solve activity -- no iterations recorded.
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_ls_extended_iters_combo_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        prob.optimizer.ls_extended_iters = 2
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("max_soc", msg)
        self.assertIn("ls_extended_iters", msg)
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_watchdog_alone_is_compatible_with_merit(self):
        # The watchdog link is compatible with either acceptance strategy --
        # only SOC / extended backtracking are rejected in combination with
        # acceptance_strategy=merit.
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        prob.optimizer.watchdog = True
        flag = prob.optimize()
        self.assertEqual(flag, ast.solvers.ConvergenceFlags.CONVERGED)


class test_SolveDiagnostics(unittest.TestCase):
    """SolveResult diagnostics default to zero/empty when their features are off."""

    def test_defaults_are_zero_after_solve(self):
        prob = _make_problem()
        prob.optimize()
        self.assertEqual(prob.optimizer.last_soc_steps, 0)
        self.assertEqual(prob.optimizer.last_watchdog_activations, 0)
        hist = prob.optimizer.last_recovery_depth_histogram
        self.assertEqual(len(hist), 4)
        for count in hist:
            self.assertIsInstance(count, int)


if __name__ == "__main__":
    unittest.main(exit=False)
