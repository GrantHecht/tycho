"""The solve() binding surface: solve(engine, mode=, presolve=, polish=,
warm=), the Mode/SolveResult/StageResult/PhaseResult/WarmStartData types and
their pickling, and the IPM/SqpSolver kwargs constructors."""

import pickle

import numpy as np
import pytest

import tychopy.optimal_control as oc
import tychopy.solvers as solvs
import tychopy.vector_functions as vf
from tychopy.vector_functions import Arguments as Args


def _small_problem():
    """A tiny, fast-converging equality-constrained QP: minimize the
    squared distance to (1, 2) subject to lying on the circle of radius 2."""
    prob = solvs.OptimizationProblem()
    prob.set_vars([0.5, 1.7])
    prob.add_objective((Args(2) - [1.0, 2.0]).squared_norm(), [0, 1])
    prob.add_equal_con(Args(2).squared_norm() - 4.0, [0, 1])
    return prob


def _rosenbrock_problem():
    """A harder problem (RosenBrock + disk constraint) that does not converge
    in a handful of iterations from its default start -- used to prove an
    iteration cap actually takes effect."""
    prob = solvs.OptimizationProblem()
    prob.set_vars([-1.0, -1.0])
    xy = Args(2)
    x, y = xy[0], xy[1]
    prob.add_objective((1 - x) ** 2 + 100 * (y - x**2) ** 2, [0, 1])
    prob.add_inequal_con(Args(2).squared_norm() - 2.0, [0, 1])
    return prob


def _quiet_ipm(**kwargs):
    ipm = solvs.IPM(**kwargs)
    ipm.print_level = 0
    return ipm


# ---------------------------------------------------------------------------
# mode=: enum, string, and unknown-string refusal
# ---------------------------------------------------------------------------


def test_solve_accepts_mode_enum():
    r = _small_problem().solve(_quiet_ipm(), mode=solvs.Mode.Optimal)
    assert r.flag == solvs.ConvergenceFlags.CONVERGED


@pytest.mark.parametrize("mode_str", ["optimal", "feasible", "Optimal", "FEASIBLE"])
def test_solve_accepts_mode_string(mode_str):
    r = _small_problem().solve(_quiet_ipm(), mode=mode_str)
    assert bool(r)


def test_solve_unknown_mode_string_raises_value_error():
    with pytest.raises(ValueError, match=r'Unknown solve mode "bogus"'):
        _small_problem().solve(_quiet_ipm(), mode="bogus")


def test_solve_bad_mode_type_raises():
    with pytest.raises(ValueError, match="mode:"):
        _small_problem().solve(_quiet_ipm(), mode=3.0)


# ---------------------------------------------------------------------------
# Refusal-matrix messages surface verbatim.
# ---------------------------------------------------------------------------


def test_polish_with_feasible_mode_refused_verbatim():
    with pytest.raises(
        ValueError,
        match=r"^polish= is an optimality refinement; it cannot follow mode=Feasible$",
    ):
        _small_problem().solve(
            _quiet_ipm(), mode=solvs.Mode.Feasible, polish=_quiet_ipm()
        )


def test_presolve_with_feasible_mode_refused_verbatim():
    with pytest.raises(
        ValueError,
        match=r"^presolve= runs a feasibility stage; mode=Feasible already is one$",
    ):
        _small_problem().solve(_quiet_ipm(), mode=solvs.Mode.Feasible, presolve=True)


def test_stale_warm_stamp_refused():
    r1 = _small_problem().solve(_quiet_ipm())

    prob2 = _small_problem()
    prob2.add_inequal_con(Args(2)[0] - 100.0, [0, 1])  # different declared problem
    with pytest.raises(
        ValueError,
        match=r"does not match the current transcription's declaration key",
    ):
        prob2.solve(_quiet_ipm(), warm=r1.warm)


# ---------------------------------------------------------------------------
# bool(result) / result.stages[i].role
# ---------------------------------------------------------------------------


def test_result_bool_reflects_convergence():
    r = _small_problem().solve(_quiet_ipm())
    assert bool(r) is True
    assert r.converged() is True


def test_result_stage_role_and_engine_name():
    r = _small_problem().solve(_quiet_ipm())
    assert len(r.stages) == 1
    assert r.stages[0].role == "main"
    assert r.stages[0].engine_name == "InteriorPointSolver"


def test_result_presolve_and_main_stage_roles():
    r = _small_problem().solve(_quiet_ipm(), presolve=True)
    assert [s.role for s in r.stages] == ["presolve", "main"]


# ---------------------------------------------------------------------------
# presolve=/polish=/warm= accepted paths.
# ---------------------------------------------------------------------------


def test_solve_presolve_none_aliases_false():
    r = _small_problem().solve(_quiet_ipm(), presolve=None)
    assert [s.role for s in r.stages] == ["main"]


def test_presolve_engine_instance_runs_two_stages():
    r = _small_problem().solve(_quiet_ipm(), presolve=_quiet_ipm())
    assert [s.role for s in r.stages] == ["presolve", "main"]


def test_polish_engine_instance_runs_polish_stage():
    r = _small_problem().solve(_quiet_ipm(), polish=_quiet_ipm())
    assert [s.role for s in r.stages] == ["main", "polish"]


def test_warm_accepts_solve_result_and_warm_start_data():
    r1 = _small_problem().solve(_quiet_ipm())
    assert bool(_small_problem().solve(_quiet_ipm(), warm=r1))
    assert bool(_small_problem().solve(_quiet_ipm(), warm=r1.warm))


# ---------------------------------------------------------------------------
# Warm-chain sanity on a continuation-shaped problem.
# ---------------------------------------------------------------------------


def _circle_problem(target, radius):
    """Same declared shape as _small_problem (2 variables, 1 equality row):
    minimize the squared distance to `target` subject to lying on the circle
    of the given `radius`. A continuation step moves `target`/`radius` --
    values baked into the equality row's own numeric constants, not the
    declared problem's dimensions or bound structure -- so consecutive calls
    with different arguments key the SAME declaration, exactly the
    OrbitContinuation pattern (examples/python_examples/OrbitContinuation.py)
    of re-solving a perturbed instance of one fixed declared problem."""
    prob = solvs.OptimizationProblem()
    prob.set_vars([0.5, 1.7])
    prob.add_objective((Args(2) - target).squared_norm(), [0, 1])
    prob.add_equal_con(Args(2).squared_norm() - radius**2, [0, 1])
    return prob


def test_warm_chain_continuation_both_solves_converge(record_property):
    """Two warm-chain legs off one cold solve, both accepted and
    stamp-matched (test_stale_warm_stamp_refused above is this same check's
    negative):

    UNPERTURBED leg -- re-solve the SAME declared problem (same target,
    same radius) with warm=r1. This is the hard, deterministic proof that
    the warm payload is genuinely applied rather than silently dropped to a
    cold start: on this 2-variable, 1-equality-row problem there is no
    barrier state and no partition nondeterminism, so nothing else could
    explain a strictly-fewer-iterations result -- measured, 1 warm iteration
    vs 5 cold. That strict inequality is asserted; a warm start that was
    silently ignored would instead reproduce the 5-iteration cold count
    exactly.

    PERTURBED leg -- re-solve a MODESTLY PERTURBED instance (radius moved by
    0.05) with warm=r1. Convergence is the only hard gate here, per the
    brief's non-gating evidence carve-out: measured, this leg takes
    dramatically MORE iterations than cold (24 warm vs 5 cold at the same
    perturbed radius), not fewer. The cause is not barrier state -- this
    fixture declares no variable bounds and no inequality rows, so there is
    no barrier term to seed, and the solver log confirms mu/Bar Obj/Bar Inf
    stay pinned at their initial values for every iteration of both solves.
    It is that the warm primal is r1's converged point on the
    PRE-perturbation circle, which sits off the perturbed one, and the line
    search pins at a small step (alpha ~= 0.25) for many consecutive
    iterations creeping back onto the curved equality constraint from
    there -- a real, occasionally counter-productive consequence of
    warm-starting a primal point near a curved constraint, not a defect in
    the warm-start plumbing. Both iteration counts are RECORDED, via
    record_property (so they survive pytest's default output capture on a
    passing test, unlike a bare print), rather than asserted: the direction
    of this effect is a property of this fixture's constraint curvature, not
    a guarantee the warm-start path owes a caller in general."""
    r1 = _circle_problem([1.0, 2.0], 2.0).solve(_quiet_ipm())
    assert bool(r1)

    r_unperturbed = _circle_problem([1.0, 2.0], 2.0).solve(_quiet_ipm(), warm=r1)
    r2 = _circle_problem([1.0, 2.0], 2.05).solve(_quiet_ipm(), warm=r1)
    assert bool(r2)

    iterations = {
        "cold": r1.iterations(),
        "warm_unperturbed": r_unperturbed.iterations(),
        "warm_perturbed": r2.iterations(),
    }

    assert r_unperturbed.iterations() < r1.iterations(), (
        f"warm-started re-solve of the SAME declared problem took "
        f"{r_unperturbed.iterations()} iterations, not fewer than the cold "
        f"solve's {r1.iterations()} -- the warm payload may not have been "
        f"applied (iterations={iterations})"
    )

    record_property("warm_chain_iterations", iterations)


# ---------------------------------------------------------------------------
# A warm= payload that cannot be used degrades to a cold start.
# ---------------------------------------------------------------------------


def test_empty_warm_payload_runs_cold_and_records_why():
    """An empty payload is not a stale stamp: it costs the seeding, and the
    reason is on the first stage's annex. (A default-constructed payload
    carries a default stamp too, so the emptiness test has to run first or
    the refusal would name the wrong cause.)"""
    r = _small_problem().solve(_quiet_ipm(), warm=solvs.WarmStartData())
    assert bool(r)
    assert "empty" in r.stages[0].engine_notes["warm_payload"]


def test_non_finite_warm_payload_runs_cold_and_records_why():
    """The documented retry idiom's own case: the payload handed back after a
    non-convergent solve is the one a diverged stage exported."""
    prob = solvs.OptimizationProblem()
    prob.set_vars([1000.0])
    prob.add_equal_con(Args(1)[0].exp() - 5.0, [0])

    diverged = prob.solve(_quiet_ipm(), mode="feasible")
    assert not bool(diverged)
    assert len(diverged.warm.primal) > 0
    assert not np.isfinite(
        np.concatenate(
            [
                np.asarray(diverged.warm.primal),
                np.asarray(diverged.warm.eq_lmults),
                np.asarray(diverged.warm.iq_lmults),
                np.asarray(diverged.warm.bound_lmults),
            ]
        )
    ).all()

    retried = prob.solve(_quiet_ipm(), mode="feasible", warm=diverged)
    assert "non-finite" in retried.stages[0].engine_notes["warm_payload"]


# ---------------------------------------------------------------------------
# A warm= payload taken from a feasibility solve seeds the primal only.
# ---------------------------------------------------------------------------


class _BrachOde(oc.ODEBase):
    """The classic brachistochrone, the smallest fixture in this file whose
    solve is long enough for a multiplier seed to change the iteration
    count."""

    def __init__(self, g=9.81):
        xtu = oc.ODEArguments(3, 1)
        _, _, v = xtu.x_vec().tolist()
        theta = xtu.u_var(0)
        ode = vf.stack([vf.sin(theta) * v, -1.0 * vf.cos(theta) * v, g * vf.cos(theta)])
        super().__init__(ode, 3, 1)


def _brach_phase(n_pts=20, n_defects=4):
    g = 9.81
    states = []
    for s in np.linspace(0.0, 1.0, n_pts):
        states.append(np.array([10.0 * s, 10.0 - 5.0 * s, g * s * np.cos(1.0), s, 1.0]))
    phase = _BrachOde(g).phase("LGL3", states, n_defects)
    phase.add_boundary_value("Front", range(0, 4), [0.0, 10.0, 0.0, 0.0])
    phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
    phase.add_boundary_value("Back", [0, 1], [10.0, 5.0])
    phase.add_delta_time_objective(1.0)
    return phase


def test_warm_from_a_feasibility_solve_forwards_the_primal_only():
    """The multipliers a mode="feasible" solve ends on are duals of the
    feasibility measure it minimized, not of the objective, so an optimality
    solve seeded from that result takes its point and derives its own prices.
    Measured while those duals were still forwarded, the optimality solve took
    20 iterations on this fixture against the 10 it takes from the same point
    unseeded."""
    phase = _brach_phase()
    feasible = phase.solve(_quiet_ipm(), mode="feasible")
    assert bool(feasible)

    seeded = phase.solve(_quiet_ipm(), warm=feasible)
    assert bool(seeded)
    assert "primal" in seeded.stages[0].engine_notes["warm_payload"]

    # Reference: the same optimality solve from the same point, with no warm=
    # at all -- which is what "primal only" means here, the point being
    # carried by the problem's own write-back either way.
    reference_phase = _brach_phase()
    assert bool(reference_phase.solve(_quiet_ipm(), mode="feasible"))
    reference = reference_phase.solve(_quiet_ipm())
    assert bool(reference)

    assert seeded.iterations() == reference.iterations()


def test_warm_from_an_optimality_solve_still_seeds_the_duals():
    """The rule is about a feasibility stage's multipliers, not about warm=
    in general: an optimality result's duals are duals of the same objective
    and still travel."""
    r1 = _circle_problem([1.0, 2.0], 2.0).solve(_quiet_ipm())
    assert bool(r1)

    r2 = _circle_problem([1.0, 2.0], 2.0).solve(_quiet_ipm(), warm=r1)
    assert bool(r2)
    assert "warm_payload" not in r2.stages[0].engine_notes
    # The payload really was applied (this fixture's cold solve takes more).
    assert r2.iterations() < r1.iterations()


def test_raw_warm_start_data_from_a_feasibility_solve_passes_through():
    """A bare WarmStartData carries no record of the stage that produced it,
    so it is taken as given -- the primal-only rule applies to a SolveResult,
    which knows which mode its stages ran."""
    phase = _brach_phase()
    feasible = phase.solve(_quiet_ipm(), mode="feasible")
    assert bool(feasible)

    seeded = phase.solve(_quiet_ipm(), warm=feasible.warm)
    assert bool(seeded)
    assert "warm_payload" not in seeded.stages[0].engine_notes


def test_stage_mode_reports_which_objective_the_stage_pursued():
    r = _small_problem().solve(_quiet_ipm(), presolve=True)
    assert [s.mode for s in r.stages] == [solvs.Mode.Feasible, solvs.Mode.Optimal]
    assert _small_problem().solve(_quiet_ipm(), mode="feasible").stages[0].mode == (
        solvs.Mode.Feasible
    )


def test_presolve_engine_without_feasible_mode_refused_naming_both_parts():
    with pytest.raises(ValueError, match=r"presolve=.*SqpSolver"):
        _small_problem().solve(_quiet_ipm(), presolve=solvs.SqpSolver())
    with pytest.raises(ValueError, match=r"presolve=.*SqpSolver"):
        _small_problem().solve(solvs.SqpSolver(), presolve=True)
    with pytest.raises(ValueError, match=r"presolve=.*SqpSolver"):
        _small_problem().set_jet_job(solvs.SqpSolver(), presolve=True)


# ---------------------------------------------------------------------------
# Pickling.
# ---------------------------------------------------------------------------


def test_solve_result_pickle_roundtrip():
    r = _small_problem().solve(_quiet_ipm())
    r2 = pickle.loads(pickle.dumps(r))

    assert r2.flag == r.flag
    assert bool(r2) == bool(r)
    assert len(r2.stages) == len(r.stages)
    assert r2.stages[0].role == r.stages[0].role
    assert r2.stages[0].engine_name == r.stages[0].engine_name
    assert r2.stages[0].mode == r.stages[0].mode
    assert r2.stages[0].iterations == r.stages[0].iterations
    assert r2.stages[0].objective == pytest.approx(r.stages[0].objective)
    assert r2.structure_key == r.structure_key
    assert (r2.warm.primal == r.warm.primal).all()


def test_warm_start_data_pickle_roundtrip_with_extension():
    r = _small_problem().solve(_quiet_ipm())
    warm = r.warm
    warm.extensions = [solvs.WarmExtension("test.solve_api.v1", b"hello-world")]

    warm2 = pickle.loads(pickle.dumps(warm))

    assert (warm2.primal == warm.primal).all()
    assert warm2.structure_key == warm.structure_key
    assert len(warm2.extensions) == 1
    assert warm2.extensions[0].tag == "test.solve_api.v1"
    assert bytes(warm2.extensions[0].payload) == b"hello-world"


def test_declaration_key_pickle_roundtrip():
    r = _small_problem().solve(_quiet_ipm())
    key2 = pickle.loads(pickle.dumps(r.structure_key))
    assert key2 == r.structure_key


def test_warm_start_data_setstate_rejects_garbage():
    # __setstate__ is a placement-new-style constructor: it only runs on a
    # freshly allocated, not-yet-initialized instance (exactly what
    # cls.__new__(cls) gives pickle -- __new__(cls) bypasses __init__, so
    # the instance stays uninitialized until __setstate__ runs on it).
    w = solvs.WarmStartData.__new__(solvs.WarmStartData)
    with pytest.raises(ValueError):
        w.__setstate__(b"not-a-warm-start-payload")


def test_warm_start_data_unpickle_rejects_truncated_payload():
    warm = _small_problem().solve(_quiet_ipm()).warm
    good_bytes = warm.__getstate__()
    truncated = good_bytes[: len(good_bytes) // 2]
    w2 = solvs.WarmStartData.__new__(solvs.WarmStartData)
    with pytest.raises(ValueError):
        w2.__setstate__(truncated)


# ---------------------------------------------------------------------------
# IPM(max_iters=...) kwargs constructor takes effect.
# ---------------------------------------------------------------------------


def test_ipm_max_iters_kwarg_takes_effect():
    ipm = _quiet_ipm(max_iters=3)
    assert ipm.max_iters == 3

    r = _rosenbrock_problem().solve(ipm)
    assert r.stages[0].iterations <= 3


def test_ipm_preset_applies_before_overrides():
    ipm = solvs.IPM(preset="soc_recovery_l1", max_soc=6)
    assert ipm.max_soc == 6
    # A field the preset itself sets (and that the override list doesn't
    # touch) should carry the preset's own value.
    assert ipm.restoration_mode == solvs.RestorationModes.l1_nested


# ---------------------------------------------------------------------------
# Unknown-kwarg refusal (TypeError, naming the argument).
# ---------------------------------------------------------------------------


def test_ipm_unknown_kwarg_raises_type_error_naming_it():
    with pytest.raises(TypeError, match="not_a_real_option"):
        solvs.IPM(not_a_real_option=3)


def test_sqp_unknown_kwarg_raises_type_error_naming_it():
    with pytest.raises(TypeError, match="not_a_real_option"):
        solvs.SqpSolver(not_a_real_option=3)


def test_sqp_kwargs_constructor_sets_fields():
    sqp = solvs.SqpSolver(kkt_tol=1e-8, max_iter=42, enable_soc=False)
    assert sqp.kkt_tol == pytest.approx(1e-8)
    assert sqp.max_iter == 42
    assert sqp.enable_soc is False


def test_ipm_bad_kwarg_value_type_raises_type_error_naming_it():
    with pytest.raises(TypeError, match="max_iters"):
        solvs.IPM(max_iters="not-an-int")


def test_ipm_bad_preset_value_type_raises_type_error():
    with pytest.raises(TypeError, match="preset"):
        solvs.IPM(preset=3)


def test_sqp_bad_kwarg_value_type_raises_type_error_naming_it():
    with pytest.raises(TypeError, match="max_iter"):
        solvs.SqpSolver(max_iter="not-an-int")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
