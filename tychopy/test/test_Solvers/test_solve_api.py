"""The M5 solve-API binding surface: solve(engine, mode=, presolve=, polish=,
warm=), the Mode/SolveResult/StageResult/PhaseResult/WarmStartData types and
their pickling, and the IPM/SqpSolver kwargs constructors."""

import pickle

import pytest

import tychopy.solvers as solvs
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
# presolve=/polish=/warm= accepted paths, and the legacy no-arg solve().
# ---------------------------------------------------------------------------


def test_legacy_no_arg_solve_still_returns_flag():
    prob = _small_problem()
    prob.optimizer.print_level = 0
    flag = prob.solve()
    assert flag == solvs.ConvergenceFlags.CONVERGED
    assert isinstance(flag, solvs.ConvergenceFlags)


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
