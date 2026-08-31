"""NLP solver backend selection surface: driving a solve through the
IpoptSolver engine (constructible only when the backend is compiled in) and
its availability guard (ipopt_available())."""

import pytest

import tychopy.solvers as solvs
from tychopy.vector_functions import Arguments as Args


def _small_problem():
    prob = solvs.OptimizationProblem()
    prob.set_vars([0.5, 1.7])
    prob.add_objective((Args(2) - [1.0, 2.0]).squared_norm(), [0, 1])
    prob.add_equal_con(Args(2).squared_norm() - 4.0, [0, 1])
    return prob


def test_ipopt_available_is_bool():
    assert isinstance(solvs.ipopt_available(), bool)


@pytest.mark.skipif(solvs.ipopt_available(), reason="built with Ipopt support")
def test_ipopt_without_build_support_raises():
    with pytest.raises(RuntimeError, match="ENABLE_IPOPT"):
        solvs.IpoptSolver()


@pytest.mark.skipif(not solvs.ipopt_available(), reason="built without Ipopt support")
def test_ipopt_options_roundtrip():
    engine = solvs.IpoptSolver()
    assert engine.options == {}
    engine.options = {"linear_solver": "pardisomkl"}
    assert engine.options == {"linear_solver": "pardisomkl"}


@pytest.mark.skipif(not solvs.ipopt_available(), reason="built without Ipopt support")
def test_ipopt_backend_solves_and_reports():
    prob = _small_problem()
    engine = solvs.IpoptSolver()
    result = prob.solve(engine)

    assert result.converged()
    assert result.flag in (
        solvs.ConvergenceFlags.CONVERGED,
        solvs.ConvergenceFlags.ACCEPTABLE,
    )
    assert result.iterations() > 0
    assert len(result.stages) == 1
    assert result.stages[0].engine_name == "Ipopt"
