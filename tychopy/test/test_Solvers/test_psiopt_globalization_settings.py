"""Coverage for PSIOPT's globalization-knob bindings.

Exercises the Python surface for the five new PSIOPT.Settings fields
(``acceptance_strategy``, ``merit_penalty_rule``, ``max_soc``,
``ls_extended_iters``, ``watchdog``) and the three new SolveResult
diagnostics (``last_soc_steps``, ``last_watchdog_activations``,
``last_recovery_depth_histogram``): property round-trips, per-field
validation, invalid enum construction, and the classic-path-recovery
combo guard in ``Settings::validate()`` (src/solvers/psiopt.cpp), which
rejects ``max_soc``/``ls_extended_iters`` in combination with any of the
three generic-path acceptance strategies (``merit``, ``funnel``,
``filter``). Also covers enum-from-int coercion for ``funnel``/``filter``
(``AcceptanceStrategies(2)``/``AcceptanceStrategies(3)``), which the
corpus harness relies on.

Also regression-tests the "component construction staleness" review
finding: ``acceptance_``/``mechanism_``/``governor_``/``recovery_`` used to
be (re)built only inside ``PSIOPT::set_nlp()`` (i.e. only on
(re)transcription), so construction-time knobs like
``acceptance_strategy`` changed AFTER a first solve were silently ignored
by a later re-solve that did not retranscribe. See
``test_ComponentRebuildTakesEffectWithoutRetranscription`` below.
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

    def test_round_trip_funnel(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.funnel
        )
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.classic_merit
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.classic_merit
        )

    def test_round_trip_filter(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.filter
        )
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.classic_merit
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.classic_merit
        )

    def test_round_trip_funnel_via_int_coercion(self):
        # The corpus harness selects strategies by raw int; enum-from-int
        # coercion must resolve to the same member as the named attribute.
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies(2)
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.funnel
        )

    def test_round_trip_filter_via_int_coercion(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies(3)
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.filter
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
    """Settings::validate() rejects any non-classic acceptance strategy
    (merit, funnel, filter) combined with max_soc > 0 or
    ls_extended_iters > 0: none of the three generic-path acceptance
    strategies implement the classic-path recovery links (SOC / extended
    backtracking re-drive the fused classic line search). validate() runs
    at PSIOPT::run_phase_sequence() entry -- i.e. on the very first call to
    optimize()/solve(), before any iteration -- so triggering it only
    requires a single solve() call on an otherwise-trivial problem. The
    watchdog link is compatible with all four acceptance strategies.
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

    def test_funnel_max_soc_combo_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        prob.optimizer.max_soc = 4
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("max_soc", msg)
        self.assertIn("ls_extended_iters", msg)
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_funnel_ls_extended_iters_combo_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        prob.optimizer.ls_extended_iters = 2
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("max_soc", msg)
        self.assertIn("ls_extended_iters", msg)
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_filter_max_soc_combo_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        prob.optimizer.max_soc = 4
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("max_soc", msg)
        self.assertIn("ls_extended_iters", msg)
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_filter_ls_extended_iters_combo_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        prob.optimizer.ls_extended_iters = 2
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("max_soc", msg)
        self.assertIn("ls_extended_iters", msg)
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_watchdog_alone_is_compatible_with_funnel(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        prob.optimizer.watchdog = True
        flag = prob.optimize()
        self.assertEqual(flag, ast.solvers.ConvergenceFlags.CONVERGED)

    def test_watchdog_alone_is_compatible_with_filter(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
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


class test_AcceptanceDiagnostics(unittest.TestCase):
    """SolveResult.last_funnel_width / last_filter_size / last_filter_resets
    (src/solvers/psiopt_globalization.cpp's FunnelAcceptance::
    append_diagnostics / FilterAcceptance::append_diagnostics, collected once
    per phase in PSIOPT::run_phase_sequence -- see psiopt.h for the sentinel
    semantics). Each is -1.0/-1/-1 ("not applicable") unless the matching
    acceptance strategy is selected; a solve with that strategy selected
    populates its own field(s) and leaves the other strategy's field(s) at
    their sentinel.
    """

    def test_funnel_solve_reports_width_filter_sentinel(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertGreater(prob.optimizer.last_funnel_width, 0.0)
        self.assertEqual(prob.optimizer.last_filter_size, -1)
        self.assertEqual(prob.optimizer.last_filter_resets, -1)

    def test_filter_solve_reports_size_funnel_sentinel(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertGreaterEqual(prob.optimizer.last_filter_size, 0)
        self.assertGreaterEqual(prob.optimizer.last_filter_resets, 0)
        self.assertEqual(prob.optimizer.last_funnel_width, -1.0)

    def test_default_solve_reports_all_sentinels(self):
        prob = _make_problem()
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_funnel_width, -1.0)
        self.assertEqual(prob.optimizer.last_filter_size, -1)
        self.assertEqual(prob.optimizer.last_filter_resets, -1)

    def test_merit_solve_reports_all_sentinels(self):
        # classic_merit's generic-path sibling (merit) also has no
        # funnel/filter state to report -- the default AcceptanceStrategy::
        # append_diagnostics() no-op applies to it too.
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_funnel_width, -1.0)
        self.assertEqual(prob.optimizer.last_filter_size, -1)
        self.assertEqual(prob.optimizer.last_filter_resets, -1)


class test_ComponentRebuildTakesEffectWithoutRetranscription(unittest.TestCase):
    """Regression test for the "component construction staleness" review
    finding on PSIOPT::set_nlp() / PSIOPT::rebuild_globalization_components()
    (src/solvers/psiopt.cpp): the four globalization components used to be
    constructed only in set_nlp() (which runs only on (re)transcription),
    so a construction-time knob like ``acceptance_strategy`` changed AFTER
    a first solve was silently ignored by a re-solve that did not
    retranscribe.

    Reachability from Python: OptimizationProblem::optimize()
    (include/tycho/detail/solvers/optimization_problem.h) gates
    ``transcribe()`` (and therefore ``set_nlp()``) on ``do_transcription_``
    alone, which is cleared once transcribe() runs and set back to True
    only by problem-EDITING calls (add_objective/add_*_con/
    reset_transcription) -- never by writing an optimizer setting, and
    NOT by ``set_vars()`` (a plain assignment to active_variables_). So:
    optimize(), set_vars(<original guess>), flip acceptance_strategy,
    optimize() again reaches run_phase_sequence() a second time WITHOUT an
    intervening set_nlp() call, from the SAME cold start as the first
    solve -- the exact repro path for the staleness bug, with the
    warm-start confound removed.

    The observable: on this Rosenbrock+disk fixture from [-1, -1] the two
    acceptance strategies take genuinely different iterate paths (measured
    on the dev toolchain: classic_merit 22 iterations, merit/wmno 20).
    Rather than pin those exact counts, the test computes BOTH cold-start
    reference counts from fresh problems, requires the fixture to
    discriminate (skip if a toolchain ever makes them coincide -- then
    this observable is void, not failing), and asserts the
    re-solve-without-retranscription run reproduces the MERIT reference
    count, not the classic one. Under the pre-fix code the re-solve still
    ran the stale ClassicMeritAcceptance and reproduced the classic count.
    """

    def test_acceptance_strategy_switch_is_live_on_resolve(self):
        # Cold-start reference counts, each from a fresh problem (each
        # first optimize() transcribes, so both references are built the
        # ordinary way).
        ref_classic = _make_problem()
        self.assertEqual(ref_classic.optimize(), solvs.ConvergenceFlags.CONVERGED)
        iters_classic = ref_classic.optimizer.last_iter_num

        ref_merit = _make_problem()
        ref_merit.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        self.assertEqual(ref_merit.optimize(), solvs.ConvergenceFlags.CONVERGED)
        iters_merit = ref_merit.optimizer.last_iter_num

        if iters_classic == iters_merit:
            self.skipTest(
                "classic_merit and merit converge in the same iteration "
                "count on this fixture/toolchain; the iteration-count "
                "observable cannot discriminate the strategies here"
            )

        # The regression scenario: solve with defaults (transcribes once),
        # restore the ORIGINAL cold start via set_vars (which does NOT set
        # do_transcription_), flip the construction-time knob, re-optimize.
        # No transcribe()/set_nlp() runs before the second optimize().
        prob = _make_problem()
        self.assertEqual(prob.optimize(), solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_iter_num, iters_classic)

        prob.set_vars([-1, -1])
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        self.assertEqual(prob.optimize(), solvs.ConvergenceFlags.CONVERGED)
        iters_resolve = prob.optimizer.last_iter_num

        # Identical solver configuration + identical cold start as the
        # merit reference => identical iterate sequence; under the pre-fix
        # code this came out equal to iters_classic instead (stale
        # ClassicMeritAcceptance still installed).
        self.assertEqual(
            iters_resolve,
            iters_merit,
            "acceptance_strategy=merit did not take effect on a re-solve "
            "without retranscription -- construction-staleness regression "
            "(globalization components not rebuilt per solve invocation)",
        )


if __name__ == "__main__":
    unittest.main(exit=False)
