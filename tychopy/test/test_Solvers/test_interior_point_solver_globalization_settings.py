"""Coverage for InteriorPointSolver's globalization-knob bindings.

Exercises the Python surface for the ten InteriorPointSolver.Settings fields
(``acceptance_strategy``, ``merit_penalty_rule``, ``max_soc``,
``ls_extended_iters``, ``watchdog``, ``barrier_governor``,
``never_monotone``, ``restoration_mode``, ``max_feas_rest``,
``inertia_mode``) and the ten SolveResult diagnostics (``last_soc_steps``,
``last_watchdog_activations``, ``last_recovery_depth_histogram``,
``last_monotone_switches``, ``last_monotone_iters``,
``last_feas_rest_entries``, ``last_feas_rest_iters``,
``last_prox_reg_primal``, ``last_prox_reg_dual``,
``last_eval_exception``): property
round-trips, per-field validation, invalid enum construction, the
composition of the SOC and extended-backtracking recovery links with
every acceptance strategy (``max_soc``/``ls_extended_iters`` now re-drive
the acceptance backtrack through the mechanism, so they compose with the
generic-path strategies ``merit``/``funnel``/``filter`` as well as
``classic_merit`` -- there is no longer a classic-only restriction), and
the barrier-governor combo guard, which rejects ``funnel``/``filter``
paired with ``barrier_governor=classic_adaptive`` (the default) unless
``never_monotone`` is set, and rejects ``never_monotone=True`` paired with
``barrier_governor=monitored`` as a direct contradiction. Also covers
enum-from-int coercion for ``funnel``/``filter``
(``AcceptanceStrategies(2)``/``AcceptanceStrategies(3)``), for
``monitored`` (``BarrierGovernors(1)``), for ``proximal_switch``/
``l1_nested`` (``RestorationModes(1)``/``RestorationModes(2)``), and for
``proximal_regularization`` (``InertiaModes(1)``), which the corpus
harness relies on, and the feasibility-restoration entry-budget
validation (``max_feas_rest`` must be non-negative) plus restoration
composing with every acceptance strategy and barrier governor combination
(no matrix restrictions, unlike the guards above) -- exercised for BOTH
``restoration_mode`` values (``proximal_switch`` and the nested l1 elastic
mode, ``l1_nested``), since neither mode restricts the combination.
``inertia_mode=proximal_regularization`` composes the same way -- it only
changes factor_impl's KKT diagonal shifts, so no combination guard applies
to it either -- and is exercised against the same acceptance/governor/
restoration matrix, plus the ``last_prox_reg_primal``/``last_prox_reg_dual``
diagnostics it populates.

Also regression-tests the "component construction staleness" review
finding: ``acceptance_``/``mechanism_``/``governor_``/``recovery_`` used to
be (re)built only inside ``InteriorPointSolver::set_nlp()`` (i.e. only on
(re)transcription), so construction-time knobs like
``acceptance_strategy`` changed AFTER a first solve were silently ignored
by a later re-solve that did not retranscribe. See
``test_ComponentRebuildTakesEffectWithoutRetranscription`` below.

Finally, covers ``InteriorPointSolver.apply_preset(name)`` (the five named
configuration presets in ``kInteriorPointSolverPresets``,
include/tycho/detail/solvers/interior_point_solver_presets.h): each preset name round-
tripped through representative properties above, every preset validating
and solving, the unrecognized-name ``ValueError`` naming all five presets,
and the binding docstring naming all five presets. See
``test_ApplyPreset`` below.
"""

import math
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
    """Smallest available problem that can drive InteriorPointSolver::run_phase_sequence
    (and therefore Settings::validate()) via optimize()/solve(). Mirrors
    test_interior_point_solver_init_time.py's RosenBrock+disk-constraint problem -- the repo
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


class test_BarrierGovernorRoundTrip(unittest.TestCase):
    def test_default_is_classic_adaptive(self):
        prob = _make_problem()
        self.assertEqual(
            prob.optimizer.barrier_governor, solvs.BarrierGovernors.classic_adaptive
        )

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        self.assertEqual(
            prob.optimizer.barrier_governor, solvs.BarrierGovernors.monitored
        )
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.classic_adaptive
        self.assertEqual(
            prob.optimizer.barrier_governor, solvs.BarrierGovernors.classic_adaptive
        )

    def test_round_trip_via_int_coercion(self):
        # The corpus harness selects the governor by raw int; enum-from-int
        # coercion must resolve to the same member as the named attribute.
        prob = _make_problem()
        prob.optimizer.barrier_governor = solvs.BarrierGovernors(1)
        self.assertEqual(
            prob.optimizer.barrier_governor, solvs.BarrierGovernors.monitored
        )


class test_NeverMonotoneRoundTrip(unittest.TestCase):
    def test_default_is_false(self):
        prob = _make_problem()
        self.assertEqual(prob.optimizer.never_monotone, False)

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.never_monotone = True
        self.assertEqual(prob.optimizer.never_monotone, True)
        prob.optimizer.never_monotone = False
        self.assertEqual(prob.optimizer.never_monotone, False)


class test_RestorationModeRoundTrip(unittest.TestCase):
    def test_default_is_off(self):
        prob = _make_problem()
        self.assertEqual(prob.optimizer.restoration_mode, solvs.RestorationModes.off)

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.restoration_mode = solvs.RestorationModes.proximal_switch
        self.assertEqual(
            prob.optimizer.restoration_mode, solvs.RestorationModes.proximal_switch
        )
        prob.optimizer.restoration_mode = solvs.RestorationModes.off
        self.assertEqual(prob.optimizer.restoration_mode, solvs.RestorationModes.off)

    def test_round_trip_via_int_coercion(self):
        # The corpus harness selects the mode by raw int; enum-from-int
        # coercion must resolve to the same member as the named attribute.
        prob = _make_problem()
        prob.optimizer.restoration_mode = solvs.RestorationModes(1)
        self.assertEqual(
            prob.optimizer.restoration_mode, solvs.RestorationModes.proximal_switch
        )

    def test_round_trip_l1_nested(self):
        prob = _make_problem()
        prob.optimizer.restoration_mode = solvs.RestorationModes.l1_nested
        self.assertEqual(
            prob.optimizer.restoration_mode, solvs.RestorationModes.l1_nested
        )
        prob.optimizer.restoration_mode = solvs.RestorationModes.off
        self.assertEqual(prob.optimizer.restoration_mode, solvs.RestorationModes.off)

    def test_round_trip_l1_nested_via_int_coercion(self):
        # Mirrors test_round_trip_via_int_coercion for the third member.
        prob = _make_problem()
        prob.optimizer.restoration_mode = solvs.RestorationModes(2)
        self.assertEqual(
            prob.optimizer.restoration_mode, solvs.RestorationModes.l1_nested
        )


class test_MaxFeasRestRoundTrip(unittest.TestCase):
    def test_default_is_two(self):
        prob = _make_problem()
        self.assertEqual(prob.optimizer.max_feas_rest, 2)

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.max_feas_rest = 5
        self.assertEqual(prob.optimizer.max_feas_rest, 5)
        prob.optimizer.max_feas_rest = 2
        self.assertEqual(prob.optimizer.max_feas_rest, 2)

    def test_zero_is_accepted(self):
        # 0 disables restoration entry entirely but is itself a valid budget.
        prob = _make_problem()
        prob.optimizer.max_feas_rest = 0
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_zero_disables_restoration_entry(self):
        # Strengthens test_zero_is_accepted: proves 0 doesn't merely validate
        # but actually disables entry when restoration_mode is on, by
        # asserting the entry counter stays at 0 for a proximal_switch solve.
        prob = _make_problem()
        prob.optimizer.restoration_mode = solvs.RestorationModes.proximal_switch
        prob.optimizer.max_feas_rest = 0
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_feas_rest_entries, 0)

    def test_zero_disables_restoration_entry_l1_nested(self):
        # Budget semantics parity: max_feas_rest gates NestedL1Restoration::
        # entry_permitted() exactly as it gates ProximalSwitchRestoration's
        # (see l1_restoration.h's entry_permitted -- same budget check,
        # same shared Settings field). Mirrors
        # test_zero_disables_restoration_entry above with restoration_mode
        # swapped to l1_nested.
        prob = _make_problem()
        prob.optimizer.restoration_mode = solvs.RestorationModes.l1_nested
        prob.optimizer.max_feas_rest = 0
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_feas_rest_entries, 0)

    def test_rejects_negative(self):
        # max_feas_rest_ now has a dedicated validated setter mirroring
        # max_soc/ls_extended_iters, so the raise fires immediately on
        # assignment rather than being deferred to validate() at
        # optimize()/solve() time.
        prob = _make_problem()
        with self.assertRaises(ValueError) as ctx:
            prob.optimizer.max_feas_rest = -1
        self.assertIn("max_feas_rest", str(ctx.exception))
        # Rejected write must not clobber the prior valid value.
        self.assertEqual(prob.optimizer.max_feas_rest, 2)


class test_BadRestorationModeValue(unittest.TestCase):
    def test_invalid_raw_value_rejected(self):
        with self.assertRaises(ValueError):
            solvs.RestorationModes(99)

    def test_enum_property_rejects_raw_int_assignment(self):
        opt = solvs.InteriorPointSolver()
        with self.assertRaises(TypeError):
            opt.restoration_mode = 7


class test_BarrierGovernorComboGuard(unittest.TestCase):
    """Settings::validate() rejects funnel/filter combined with
    barrier_governor=classic_adaptive (the default) and never_monotone=False
    -- those acceptance strategies are designed to operate above a monotone
    barrier safeguard, which classic_adaptive does not provide.
    never_monotone=True combined with barrier_governor=monitored is a direct
    contradiction (monitored already provides the safeguard never_monotone
    forfeits) and is rejected regardless of acceptance_strategy. Both
    opt-ins (barrier_governor=monitored, never_monotone=True) individually
    validate and solve with funnel/filter; classic_merit/merit are
    unaffected by this guard in every combination.
    """

    def test_funnel_classic_adaptive_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("barrier_governor", msg)
        self.assertIn("never_monotone", msg)
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_filter_classic_adaptive_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("barrier_governor", msg)
        self.assertIn("never_monotone", msg)
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_never_monotone_with_monitored_raises_before_any_iteration(self):
        prob = _make_problem()
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        prob.optimizer.never_monotone = True
        with self.assertRaises(ValueError) as ctx:
            prob.optimize()
        msg = str(ctx.exception)
        self.assertIn("barrier_governor", msg)
        self.assertIn("never_monotone", msg)
        self.assertEqual(prob.optimizer.last_iter_num, 0)

    def test_funnel_with_monitored_governor_validates_and_solves(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_filter_with_never_monotone_validates_and_solves(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        prob.optimizer.never_monotone = True
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_classic_merit_unaffected_by_classic_adaptive_governor(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.classic_merit
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_merit_unaffected_by_classic_adaptive_governor(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_classic_merit_with_monitored_governor_is_allowed_opt_in(self):
        # classic_merit + monitored is allowed opt-in -- bit-identity is
        # about the DEFAULT governor selection, not about excluding
        # classic_merit from pairing with the monitored governor.
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.classic_merit
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)


class test_MonotoneDiagnostics(unittest.TestCase):
    """SolveResult.last_monotone_switches / last_monotone_iters
    (MonitoredBarrierGovernor::append_diagnostics, collected once per phase
    in InteriorPointSolver::run_phase_sequence -- see interior_point_solver.h for the sentinel
    semantics). Sentinel -1/-1 unless barrier_governor is monitored.
    """

    def test_default_solve_reports_sentinels(self):
        prob = _make_problem()
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_monotone_switches, -1)
        self.assertEqual(prob.optimizer.last_monotone_iters, -1)

    def test_monitored_solve_populates_diagnostics(self):
        prob = _make_problem()
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertGreaterEqual(prob.optimizer.last_monotone_switches, 0)
        self.assertGreaterEqual(prob.optimizer.last_monotone_iters, 0)


class test_FeasRestDiagnostics(unittest.TestCase):
    """SolveResult.last_feas_rest_entries / last_feas_rest_iters
    (ProximalSwitchRestoration::append_diagnostics /
    NestedL1Restoration::append_diagnostics, collected once per phase in
    InteriorPointSolver::run_phase_sequence -- see interior_point_solver.h for the sentinel semantics).
    Sentinel -1/-1 unless restoration_mode is proximal_switch or l1_nested
    (no restoration strategy is constructed when it is off); once a
    strategy is constructed, both fields report >= 0 even if restoration
    was never actually entered during the solve. Both concrete strategies
    populate the same pair of fields identically (see the field docstrings
    in interior_point_solver_bind.cpp for the nested-mode counting note).
    """

    def test_default_solve_reports_sentinels(self):
        prob = _make_problem()
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_feas_rest_entries, -1)
        self.assertEqual(prob.optimizer.last_feas_rest_iters, -1)

    def test_proximal_switch_solve_reports_non_negative_counts(self):
        prob = _make_problem()
        prob.optimizer.restoration_mode = solvs.RestorationModes.proximal_switch
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertGreaterEqual(prob.optimizer.last_feas_rest_entries, 0)
        self.assertGreaterEqual(prob.optimizer.last_feas_rest_iters, 0)

    def test_l1_nested_solve_reports_non_negative_counts(self):
        prob = _make_problem()
        prob.optimizer.restoration_mode = solvs.RestorationModes.l1_nested
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertGreaterEqual(prob.optimizer.last_feas_rest_entries, 0)
        self.assertGreaterEqual(prob.optimizer.last_feas_rest_iters, 0)


class test_RestorationComboMatrix(unittest.TestCase):
    """restoration_mode in {proximal_switch, l1_nested} composes with every
    acceptance_strategy x barrier_governor combination -- no combo guard
    in Settings::validate() restricts either mode (unlike the
    acceptance/governor guards exercised in test_BarrierGovernorComboGuard /
    test_AcceptanceMeritRecoveryComboGuard above), since every shipped
    acceptance strategy implements the restoration exit test both modes
    rely on. Spot-checks the matrix for BOTH restoration modes (each test
    method loops the two modes via subTest), including the funnel/filter +
    monitored combinations named in the feature's review notes.
    """

    # Both concrete RestorationStrategy implementations wired today --
    # extending this tuple is the only change needed to add a third mode
    # to every combo test below.
    RESTORATION_MODES = (
        solvs.RestorationModes.proximal_switch,
        solvs.RestorationModes.l1_nested,
    )

    def _solve_with(
        self, acceptance_strategy, barrier_governor, restoration_mode, **extra_settings
    ):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = acceptance_strategy
        prob.optimizer.barrier_governor = barrier_governor
        prob.optimizer.restoration_mode = restoration_mode
        for name, value in extra_settings.items():
            setattr(prob.optimizer, name, value)
        return prob.optimize()

    def test_classic_merit_classic_adaptive_with_restoration_solves(self):
        for mode in self.RESTORATION_MODES:
            with self.subTest(restoration_mode=mode):
                flag = self._solve_with(
                    solvs.AcceptanceStrategies.classic_merit,
                    solvs.BarrierGovernors.classic_adaptive,
                    mode,
                )
                self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_merit_classic_adaptive_with_restoration_solves(self):
        for mode in self.RESTORATION_MODES:
            with self.subTest(restoration_mode=mode):
                flag = self._solve_with(
                    solvs.AcceptanceStrategies.merit,
                    solvs.BarrierGovernors.classic_adaptive,
                    mode,
                )
                self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_classic_merit_monitored_with_restoration_solves(self):
        for mode in self.RESTORATION_MODES:
            with self.subTest(restoration_mode=mode):
                flag = self._solve_with(
                    solvs.AcceptanceStrategies.classic_merit,
                    solvs.BarrierGovernors.monitored,
                    mode,
                )
                self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_funnel_monitored_with_restoration_solves(self):
        for mode in self.RESTORATION_MODES:
            with self.subTest(restoration_mode=mode):
                flag = self._solve_with(
                    solvs.AcceptanceStrategies.funnel,
                    solvs.BarrierGovernors.monitored,
                    mode,
                )
                self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_filter_monitored_with_restoration_solves(self):
        for mode in self.RESTORATION_MODES:
            with self.subTest(restoration_mode=mode):
                flag = self._solve_with(
                    solvs.AcceptanceStrategies.filter,
                    solvs.BarrierGovernors.monitored,
                    mode,
                )
                self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_filter_never_monotone_with_restoration_solves(self):
        for mode in self.RESTORATION_MODES:
            with self.subTest(restoration_mode=mode):
                flag = self._solve_with(
                    solvs.AcceptanceStrategies.filter,
                    solvs.BarrierGovernors.classic_adaptive,
                    mode,
                    never_monotone=True,
                )
                self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)


class test_InertiaModeRoundTrip(unittest.TestCase):
    def test_default_is_classic(self):
        prob = _make_problem()
        self.assertEqual(prob.optimizer.inertia_mode, solvs.InertiaModes.classic)

    def test_round_trip(self):
        prob = _make_problem()
        prob.optimizer.inertia_mode = solvs.InertiaModes.proximal_regularization
        self.assertEqual(
            prob.optimizer.inertia_mode, solvs.InertiaModes.proximal_regularization
        )
        prob.optimizer.inertia_mode = solvs.InertiaModes.classic
        self.assertEqual(prob.optimizer.inertia_mode, solvs.InertiaModes.classic)

    def test_round_trip_via_int_coercion(self):
        # The corpus harness selects the mode by raw int; enum-from-int
        # coercion must resolve to the same member as the named attribute.
        prob = _make_problem()
        prob.optimizer.inertia_mode = solvs.InertiaModes(1)
        self.assertEqual(
            prob.optimizer.inertia_mode, solvs.InertiaModes.proximal_regularization
        )


class test_BadInertiaModeValue(unittest.TestCase):
    def test_invalid_raw_value_rejected(self):
        with self.assertRaises(ValueError):
            solvs.InertiaModes(99)

    def test_enum_property_rejects_raw_int_assignment(self):
        opt = solvs.InteriorPointSolver()
        with self.assertRaises(TypeError):
            opt.inertia_mode = 7


class test_InertiaModeComboMatrix(unittest.TestCase):
    """inertia_mode=proximal_regularization composes with every
    acceptance_strategy x barrier_governor combination the existing
    matrices cover (see test_BarrierGovernorComboGuard /
    test_RestorationComboMatrix above), plus every restoration_mode and
    the SOC/extended-backtracking/watchdog recovery links -- the mode only
    changes factor_impl's KKT diagonal shifts and adds no new combination
    guard in Settings::validate().
    """

    def _solve_with(
        self,
        acceptance_strategy,
        barrier_governor,
        restoration_mode=solvs.RestorationModes.off,
        **extra_settings,
    ):
        prob = _make_problem()
        prob.optimizer.inertia_mode = solvs.InertiaModes.proximal_regularization
        prob.optimizer.acceptance_strategy = acceptance_strategy
        prob.optimizer.barrier_governor = barrier_governor
        prob.optimizer.restoration_mode = restoration_mode
        for name, value in extra_settings.items():
            setattr(prob.optimizer, name, value)
        return prob.optimize()

    def test_classic_merit_classic_adaptive_solves(self):
        flag = self._solve_with(
            solvs.AcceptanceStrategies.classic_merit,
            solvs.BarrierGovernors.classic_adaptive,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_merit_classic_adaptive_solves(self):
        flag = self._solve_with(
            solvs.AcceptanceStrategies.merit,
            solvs.BarrierGovernors.classic_adaptive,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_classic_merit_monitored_solves(self):
        flag = self._solve_with(
            solvs.AcceptanceStrategies.classic_merit,
            solvs.BarrierGovernors.monitored,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_funnel_monitored_solves(self):
        flag = self._solve_with(
            solvs.AcceptanceStrategies.funnel,
            solvs.BarrierGovernors.monitored,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_filter_monitored_solves(self):
        flag = self._solve_with(
            solvs.AcceptanceStrategies.filter,
            solvs.BarrierGovernors.monitored,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_filter_never_monotone_solves(self):
        flag = self._solve_with(
            solvs.AcceptanceStrategies.filter,
            solvs.BarrierGovernors.classic_adaptive,
            never_monotone=True,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_with_proximal_switch_restoration_solves(self):
        flag = self._solve_with(
            solvs.AcceptanceStrategies.classic_merit,
            solvs.BarrierGovernors.classic_adaptive,
            restoration_mode=solvs.RestorationModes.proximal_switch,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_with_l1_nested_restoration_solves(self):
        flag = self._solve_with(
            solvs.AcceptanceStrategies.classic_merit,
            solvs.BarrierGovernors.classic_adaptive,
            restoration_mode=solvs.RestorationModes.l1_nested,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_watchdog_max_soc_ls_extended_iters_compose(self):
        # inertia_mode is orthogonal to the recovery-link knobs too.
        flag = self._solve_with(
            solvs.AcceptanceStrategies.filter,
            solvs.BarrierGovernors.monitored,
            max_soc=4,
            ls_extended_iters=2,
            watchdog=True,
        )
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)


class test_ProxRegDiagnostics(unittest.TestCase):
    """SolveResult.last_prox_reg_primal / last_prox_reg_dual (written at
    alg_impl phase close from the last FACTORIZED iteration's applied
    shifts -- see interior_point_solver.h for the sentinel semantics). Sentinel -1.0/-1.0
    unless inertia_mode is proximal_regularization; both fields report
    >= 0.0 once that mode is selected and the solve factorizes at least
    once.
    """

    def test_default_solve_reports_sentinels(self):
        prob = _make_problem()
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_prox_reg_primal, -1.0)
        self.assertEqual(prob.optimizer.last_prox_reg_dual, -1.0)

    def test_proximal_regularization_solve_reports_non_negative_shifts(self):
        prob = _make_problem()
        prob.optimizer.inertia_mode = solvs.InertiaModes.proximal_regularization
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertGreaterEqual(prob.optimizer.last_prox_reg_primal, 0.0)
        self.assertGreaterEqual(prob.optimizer.last_prox_reg_dual, 0.0)

    def test_mode_off_after_mode_on_restores_sentinels(self):
        # Guards both the mode-gate on the phase-close write and the
        # reset_accumulators() coverage: a mode-on solve leaves real values
        # behind, so a following mode-off solve on the SAME optimizer only
        # reports -1.0 again if the fields were actually re-sentineled and
        # the classic path really skips the write.
        prob = _make_problem()
        prob.optimizer.inertia_mode = solvs.InertiaModes.proximal_regularization
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertGreaterEqual(prob.optimizer.last_prox_reg_primal, 0.0)
        prob.optimizer.inertia_mode = solvs.InertiaModes.classic
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertEqual(prob.optimizer.last_prox_reg_primal, -1.0)
        self.assertEqual(prob.optimizer.last_prox_reg_dual, -1.0)


class test_EvalExceptionDiagnostic(unittest.TestCase):
    """SolveResult.last_eval_exception (the message of the most recent
    trial-point evaluation exception absorbed by the acceptance machinery
    during the most recent solve call -- see interior_point_solver.h for the empty-string
    sentinel semantics). Read-only, like every other SolveResult diagnostic
    above.
    """

    def test_fresh_optimizer_reports_empty_sentinel(self):
        prob = _make_problem()
        self.assertEqual(prob.optimizer.last_eval_exception, "")
        # A clean solve exercises the per-solve reset path and must leave the
        # sentinel untouched.
        prob.optimize()
        self.assertEqual(prob.optimizer.last_eval_exception, "")

    def test_property_rejects_assignment(self):
        prob = _make_problem()
        with self.assertRaises(AttributeError):
            prob.optimizer.last_eval_exception = "boom"


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

    def test_barrier_governor_invalid_raw_value_rejected(self):
        with self.assertRaises(ValueError):
            solvs.BarrierGovernors(99)

    def test_enum_property_rejects_raw_int_assignment(self):
        # Assigning a raw int to the enum-typed PROPERTY goes through
        # nanobind's convert path, which rejects non-member values with
        # TypeError (a different path than the ValueError-raising enum
        # constructor above) -- pin both exception types.
        opt = solvs.InteriorPointSolver()
        with self.assertRaises(TypeError):
            opt.acceptance_strategy = 7
        with self.assertRaises(TypeError):
            opt.merit_penalty_rule = 7
        with self.assertRaises(TypeError):
            opt.barrier_governor = 7


class test_AcceptanceGenericRecoveryComposition(unittest.TestCase):
    """The SOC (max_soc > 0) and extended-backtracking (ls_extended_iters >
    0) recovery links compose with every acceptance strategy. They re-drive
    the acceptance backtrack through GlobalizationMechanism::
    run_acceptance_backtrack, which dispatches to the classic merit test or
    to the generic AcceptanceStrategy::is_iterate_acceptable surface
    (merit / funnel / filter) as appropriate, so a corrected or extended
    step is tested against the SAME acceptance criteria the ordinary step
    faced. Settings::validate() therefore no longer rejects these knobs in
    combination with a generic-path strategy (the former classic-only
    restriction has been lifted). funnel/filter still require their own
    monotone-barrier opt-in (barrier_governor=monitored or
    never_monotone=True -- test_BarrierGovernorComboGuard), so the
    funnel/filter cases below set it. The watchdog link was always
    compatible with all four acceptance strategies.
    """

    def test_merit_max_soc_composes_and_solves(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        prob.optimizer.max_soc = 4
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_merit_ls_extended_iters_composes_and_solves(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        prob.optimizer.ls_extended_iters = 2
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_watchdog_alone_is_compatible_with_merit(self):
        # The watchdog link is compatible with every acceptance strategy;
        # SOC / extended backtracking now are too (see the sibling tests).
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        prob.optimizer.watchdog = True
        flag = prob.optimize()
        self.assertEqual(flag, ast.solvers.ConvergenceFlags.CONVERGED)

    def test_funnel_max_soc_composes_and_solves(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        prob.optimizer.max_soc = 4
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_funnel_ls_extended_iters_composes_and_solves(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        prob.optimizer.ls_extended_iters = 2
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_filter_max_soc_composes_and_solves(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        prob.optimizer.never_monotone = True
        prob.optimizer.max_soc = 4
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_filter_ls_extended_iters_composes_and_solves(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        prob.optimizer.never_monotone = True
        prob.optimizer.ls_extended_iters = 2
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_filter_soc_and_extended_and_watchdog_compose_and_solve(self):
        # All three recovery links enabled at once under a generic strategy.
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        prob.optimizer.max_soc = 4
        prob.optimizer.ls_extended_iters = 2
        prob.optimizer.watchdog = True
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_watchdog_alone_is_compatible_with_funnel(self):
        # The watchdog link is compatible with every acceptance strategy (as
        # are SOC / extended backtracking now). funnel/filter also require an
        # explicit barrier_governor/never_monotone opt-in (see
        # test_BarrierGovernorRoundTrip / test_BarrierGovernorComboGuard
        # below) -- never_monotone is used here, barrier_governor=monitored
        # is used in the filter sibling below, to exercise both opt-ins.
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        prob.optimizer.watchdog = True
        prob.optimizer.never_monotone = True
        flag = prob.optimize()
        self.assertEqual(flag, ast.solvers.ConvergenceFlags.CONVERGED)

    def test_watchdog_alone_is_compatible_with_filter(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        prob.optimizer.watchdog = True
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
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
        # 5 buckets: SOC, extended backtracking, watchdog, unresolved,
        # restoration -- the restoration bucket only increments when a
        # restoration mode (proximal_switch or l1_nested) is enabled.
        self.assertEqual(len(hist), 5)
        for count in hist:
            self.assertIsInstance(count, int)


class test_AcceptanceDiagnostics(unittest.TestCase):
    """SolveResult.last_funnel_width / last_filter_size / last_filter_resets
    (src/solvers/interior_point_solver_globalization.cpp's FunnelAcceptance::
    append_diagnostics / FilterAcceptance::append_diagnostics, collected once
    per phase in InteriorPointSolver::run_phase_sequence -- see interior_point_solver.h for the sentinel
    semantics). Each is -1.0/-1/-1 ("not applicable") unless the matching
    acceptance strategy is selected; a solve with that strategy selected
    populates its own field(s) and leaves the other strategy's field(s) at
    their sentinel.
    """

    def test_funnel_solve_reports_width_filter_sentinel(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.funnel
        # funnel requires an explicit barrier_governor/never_monotone opt-in
        # (Settings::validate()) -- exercise the barrier_governor=monitored
        # flavor here (the filter sibling below uses never_monotone).
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        flag = prob.optimize()
        self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)
        self.assertGreater(prob.optimizer.last_funnel_width, 0.0)
        self.assertTrue(math.isfinite(prob.optimizer.last_funnel_width))
        self.assertEqual(prob.optimizer.last_filter_size, -1)
        self.assertEqual(prob.optimizer.last_filter_resets, -1)

    def test_filter_solve_reports_size_funnel_sentinel(self):
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.filter
        # never_monotone opt-in flavor (see the funnel sibling above).
        prob.optimizer.never_monotone = True
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
    finding on InteriorPointSolver::set_nlp() / InteriorPointSolver::rebuild_globalization_components()
    (src/solvers/interior_point_solver.cpp): the four globalization components used to be
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


class test_ApplyPreset(unittest.TestCase):
    """InteriorPointSolver.apply_preset(name) (src/solvers/interior_point_solver_settings.cpp, table-driven
    from kInteriorPointSolverPresets in include/tycho/detail/solvers/interior_point_solver_presets.h).
    Round-trips each of the five named presets through the existing
    property surface exercised throughout this file, checks that an
    unrecognized name raises ValueError naming all five valid options, and
    pins the docstring's five-name coverage (the binding's single
    obligation beyond forwarding to the C++ method).
    """

    PRESET_NAMES = (
        "classic",
        "filter_l1",
        "soc_recovery_l1",
        "soc_proximal",
        "merit_l1",
    )

    def test_classic_resets_a_dirtied_optimizer(self):
        # Every field classic assigns also happens to be Settings{}'s
        # default, so applying it to a fresh optimizer would trivially
        # match without proving the preset actually writes anything.
        # Dirty every field first, then confirm apply_preset("classic")
        # overwrites all of them back to the stock configuration.
        prob = _make_problem()
        prob.optimizer.acceptance_strategy = solvs.AcceptanceStrategies.merit
        prob.optimizer.barrier_governor = solvs.BarrierGovernors.monitored
        prob.optimizer.restoration_mode = solvs.RestorationModes.l1_nested
        prob.optimizer.inertia_mode = solvs.InertiaModes.proximal_regularization
        prob.optimizer.max_soc = 4
        prob.optimizer.ls_extended_iters = 2
        prob.optimizer.watchdog = True

        prob.optimizer.apply_preset("classic")

        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.classic_merit
        )
        self.assertEqual(
            prob.optimizer.barrier_governor, solvs.BarrierGovernors.classic_adaptive
        )
        self.assertEqual(prob.optimizer.restoration_mode, solvs.RestorationModes.off)
        self.assertEqual(prob.optimizer.inertia_mode, solvs.InertiaModes.classic)
        self.assertEqual(prob.optimizer.max_soc, 0)
        self.assertEqual(prob.optimizer.ls_extended_iters, 0)
        self.assertEqual(prob.optimizer.watchdog, False)

    def test_filter_l1(self):
        prob = _make_problem()
        prob.optimizer.apply_preset("filter_l1")
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.filter
        )
        self.assertEqual(
            prob.optimizer.barrier_governor, solvs.BarrierGovernors.monitored
        )
        self.assertEqual(
            prob.optimizer.restoration_mode, solvs.RestorationModes.l1_nested
        )

    def test_soc_recovery_l1(self):
        prob = _make_problem()
        prob.optimizer.apply_preset("soc_recovery_l1")
        self.assertEqual(prob.optimizer.max_soc, 4)
        self.assertEqual(prob.optimizer.ls_extended_iters, 2)
        self.assertEqual(prob.optimizer.watchdog, True)
        self.assertEqual(
            prob.optimizer.restoration_mode, solvs.RestorationModes.l1_nested
        )

    def test_soc_proximal(self):
        prob = _make_problem()
        prob.optimizer.apply_preset("soc_proximal")
        self.assertEqual(
            prob.optimizer.restoration_mode, solvs.RestorationModes.proximal_switch
        )
        self.assertEqual(prob.optimizer.max_soc, 4)
        # Distinguishes this preset from soc_recovery_l1, which shares the
        # same acceptance/governor/inertia/max_soc fields but enables
        # watchdog and ls_extended_iters.
        self.assertEqual(prob.optimizer.watchdog, False)

    def test_merit_l1(self):
        prob = _make_problem()
        prob.optimizer.apply_preset("merit_l1")
        self.assertEqual(
            prob.optimizer.acceptance_strategy, solvs.AcceptanceStrategies.merit
        )
        self.assertEqual(
            prob.optimizer.barrier_governor, solvs.BarrierGovernors.classic_adaptive
        )
        self.assertEqual(
            prob.optimizer.restoration_mode, solvs.RestorationModes.l1_nested
        )

    def test_every_preset_solves(self):
        # Every preset must pass Settings::validate() -- smoke each one
        # through a full optimize() call.
        for name in self.PRESET_NAMES:
            with self.subTest(preset=name):
                prob = _make_problem()
                prob.optimizer.apply_preset(name)
                flag = prob.optimize()
                self.assertEqual(flag, solvs.ConvergenceFlags.CONVERGED)

    def test_unrecognized_name_raises_value_error_naming_all_presets(self):
        prob = _make_problem()
        with self.assertRaises(ValueError) as ctx:
            prob.optimizer.apply_preset("not_a_real_preset")
        msg = str(ctx.exception)
        for name in self.PRESET_NAMES:
            self.assertIn(name, msg)

    def test_docstring_names_all_five_presets(self):
        doc = solvs.InteriorPointSolver.apply_preset.__doc__
        self.assertIsNotNone(doc)
        for name in self.PRESET_NAMES:
            self.assertIn(name, doc)


if __name__ == "__main__":
    unittest.main(exit=False)
