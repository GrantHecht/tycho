"""NLP solver backend selection surface (nlp_solver / ipopt_options /
last_ipopt_result / ipopt_available)."""

import pytest

import tychopy.solvers as solvs
from tychopy.vector_functions import Arguments as Args


def _small_problem():
    prob = solvs.OptimizationProblem()
    prob.set_vars([0.5, 1.7])
    prob.add_objective((Args(2) - [1.0, 2.0]).squared_norm(), [0, 1])
    prob.add_equal_con(Args(2).squared_norm() - 4.0, [0, 1])
    return prob


def test_nlp_solver_default_and_roundtrip():
    prob = _small_problem()
    assert prob.nlp_solver == solvs.NLPSolvers.interior_point
    prob.nlp_solver = solvs.NLPSolvers.ipopt
    assert prob.nlp_solver == solvs.NLPSolvers.ipopt


def test_ipopt_options_roundtrip():
    prob = _small_problem()
    assert prob.ipopt_options == {}
    prob.ipopt_options = {"linear_solver": "pardisomkl"}
    assert prob.ipopt_options == {"linear_solver": "pardisomkl"}


def test_last_ipopt_result_sentinels():
    prob = _small_problem()
    info = prob.last_ipopt_result
    assert info.ran is False
    assert info.iterations == -1


def test_ipopt_available_is_bool():
    assert isinstance(solvs.ipopt_available(), bool)


@pytest.mark.skipif(solvs.ipopt_available(), reason="built with Ipopt support")
def test_ipopt_without_build_support_raises():
    prob = _small_problem()
    prob.nlp_solver = solvs.NLPSolvers.ipopt
    with pytest.raises(RuntimeError, match="ENABLE_IPOPT"):
        prob.optimize()


@pytest.mark.skipif(not solvs.ipopt_available(), reason="built without Ipopt support")
def test_ipopt_backend_solves_and_reports():
    prob = _small_problem()
    prob.nlp_solver = solvs.NLPSolvers.ipopt
    flag = prob.optimize()
    assert flag in (solvs.ConvergenceFlags.CONVERGED, solvs.ConvergenceFlags.ACCEPTABLE)
    info = prob.last_ipopt_result
    assert info.ran is True
    assert info.iterations > 0
    assert info.normalized in ("converged", "acceptable")
