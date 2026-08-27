"""Smoke test for the interior-point-solver robustness corpus (tests/corpus/).

Verifies the problem-module contract documented in tests/corpus/README.md
holds for every module registered in tests/corpus/registry.py, and that the
harness (scripts/run_corpus.py) runs end-to-end on two fast real problems
(one converging, one diverging).

This is the only pytest-gated piece of the corpus: the corpus problems
themselves are expected to fail/diverge on today's interior-point-solver defaults and are
scored (never gated) by the harness, not by this test.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tychopy.solvers as solvs

# This file lives at <repo>/tychopy/test/test_corpus_smoke.py, so
# parents[2] is the repo root when it runs in place. Some harnesses copy the
# test files elsewhere (e.g. the wheel-layout CI job runs pytest against the
# packaged wheel from a scratch directory) where that relative depth no
# longer holds — TYCHO_REPO_ROOT lets such a harness point back at the real
# checkout, which still holds tests/corpus/ and scripts/run_corpus.py.
REPO_ROOT = Path(os.environ.get("TYCHO_REPO_ROOT", Path(__file__).resolve().parents[2]))
CORPUS_DIR = REPO_ROOT / "tests" / "corpus"
RUN_CORPUS = REPO_ROOT / "scripts" / "run_corpus.py"

if str(CORPUS_DIR) not in sys.path:
    sys.path.insert(0, str(CORPUS_DIR))

import driver  # noqa: E402  (tests/corpus/ on sys.path, same as registry)
import registry  # noqa: E402  (tests/corpus/ must be on sys.path first)

VALID_TIERS = {"degenerate", "hard", "literature"}

# The reserved keys a SOLVE_CALL dict may use -- "mode"/"presolve"/"polish"/
# "warm" are forwarded straight to prob.solve(engine, **SOLVE_CALL);
# "feasible_fallback" is consumed by the driver itself (see
# tests/corpus/driver.py's module docstring for the full SOLVE_CALL
# convention, including the old SOLVE_MODE vocabulary's mapping onto it).
VALID_SOLVE_CALL_KEYS = {"mode", "presolve", "polish", "warm", "feasible_fallback"}
VALID_MODES = {"optimal", "feasible"}


@pytest.mark.parametrize("module_name", registry.ALL_PROBLEMS)
def test_problem_contract(module_name):
    """Every registered problem module exposes the required contract surface."""
    mod = importlib.import_module(f"problems.{module_name}")

    assert mod.TIER in VALID_TIERS, f"{module_name}: invalid TIER {mod.TIER!r}"
    assert isinstance(mod.TIMEOUT, int) and mod.TIMEOUT > 0, (
        f"{module_name}: TIMEOUT must be a positive int, got {mod.TIMEOUT!r}"
    )
    assert isinstance(mod.SOLVE_CALL, dict), (
        f"{module_name}: SOLVE_CALL must be a dict, got {mod.SOLVE_CALL!r}"
    )
    assert set(mod.SOLVE_CALL) <= VALID_SOLVE_CALL_KEYS, (
        f"{module_name}: unknown SOLVE_CALL key(s) "
        f"{set(mod.SOLVE_CALL) - VALID_SOLVE_CALL_KEYS!r}"
    )
    mode = str(mod.SOLVE_CALL.get("mode", "optimal")).lower()
    assert mode in VALID_MODES, f"{module_name}: invalid SOLVE_CALL mode {mode!r}"
    assert callable(mod.build), f"{module_name}: build must be callable"
    assert not (hasattr(mod, "NOTES") and hasattr(mod, "POST_SOLVE")), (
        f"{module_name}: define at most one of NOTES/POST_SOLVE"
    )
    configure_hook = getattr(mod, "configure", None)
    assert configure_hook is None or callable(configure_hook), (
        f"{module_name}: configure, if defined, must be callable"
    )


def test_harness_end_to_end_fast_problems(tmp_path):
    """The harness runs end-to-end and records the expected statuses.

    Uses the two fastest real corpus problems instead of the original
    throwaway stub problems (since deleted): ``deg_dup_equality`` (converges in
    3 iterations, ~1 s) and ``hard_zermelo_wrongbasin`` (diverges, ~0.8 s —
    the fastest genuine failure in the corpus; the other degenerate-tier
    failures grind through the full 500-iteration ``max_iters`` cap).
    ``hard_vanderpol`` previously filled the diverging slot, but the
    ``squared_norm`` derivative fix and the switch to native variable
    bounds cured its divergence, so it now converges and no longer serves
    the purpose of this test.
    Each is run in its own ``--filter`` invocation since the two problem
    names share no common substring.
    """
    by_problem = {}
    for name in ("deg_dup_equality", "hard_zermelo_wrongbasin"):
        out_path = tmp_path / f"{name}_result.jsonl"
        proc = subprocess.run(
            [sys.executable, str(RUN_CORPUS), "--filter", name, "--out", str(out_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"harness exited {proc.returncode}\n{proc.stderr}"

        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1, (
            f"expected 1 corpus record for {name}, got {len(lines)}: {lines}"
        )

        record = json.loads(lines[0])
        assert record["problem"] == name
        assert record["backend"] == "psiopt"
        by_problem[name] = record

    assert by_problem["deg_dup_equality"]["status"] == "converged"
    assert by_problem["hard_zermelo_wrongbasin"]["status"] == "diverged"


class _FakeResult:
    """Stand-in for SolveResult: a real ConvergenceFlags member (so
    ``result.flag != solvs.ConvergenceFlags.CONVERGED`` compares correctly,
    not just ``.flag.name``) plus a truthy/falsy convergence bit."""

    def __init__(self, flag_name, converged):
        self.flag = getattr(solvs.ConvergenceFlags, flag_name)
        self._converged = converged

    def __bool__(self):
        return self._converged


class _CallShapeProbeProblem:
    """Stub problem recording every ``solve(engine, **kwargs)`` call's kwargs."""

    def __init__(self):
        self.calls = []

    def solve(self, engine, mode="optimal", presolve=False, polish=None, warm=None):
        self.calls.append(
            {"mode": mode, "presolve": presolve, "polish": polish, "warm": warm}
        )
        return _FakeResult("CONVERGED", converged=True)


class _CallShapeProbeModule:
    SOLVE_CALL = {"mode": "optimal", "presolve": True}
    NOTES = ""

    @staticmethod
    def build():
        return _CallShapeProbeModule._prob


def test_driver_default_call_shape_runs_module_solve_call():
    # presolve=True forwards straight through to solve()'s own presolve=
    # argument as a single call -- the pipeline itself now tolerates a
    # diverging presolve stage (src/solvers/solve_pipeline.cpp's
    # warm_or_null), so the driver no longer needs to route around it with
    # two separate calls.
    prob = _CallShapeProbeProblem()
    _CallShapeProbeModule._prob = prob
    result = driver.run(_CallShapeProbeModule, lambda engine: None)
    assert prob.calls == [
        {"mode": "optimal", "presolve": True, "polish": None, "warm": None},
    ]
    assert result["call_shape"] == "module"


def test_driver_optimize_call_shape_overrides_solve_call():
    prob = _CallShapeProbeProblem()
    _CallShapeProbeModule._prob = prob
    result = driver.run(
        _CallShapeProbeModule, lambda engine: None, call_shape="optimize"
    )
    assert prob.calls == [
        {"mode": "optimal", "presolve": False, "polish": None, "warm": None}
    ]
    assert result["call_shape"] == "optimize"


class _FallbackProbeProblem:
    """Stub problem whose first solve() reports NOTCONVERGED and second
    (the feasible_fallback retry) reports CONVERGED -- exercises the
    driver's conditional chain for the retired optimize_solve()/
    solve_optimize_solve() combo methods' trailing conditional Solve
    phase."""

    def __init__(self):
        self.calls = []

    def solve(self, engine, mode="optimal", presolve=False, polish=None, warm=None):
        self.calls.append(
            {"mode": mode, "presolve": presolve, "polish": polish, "warm": warm}
        )
        converged = len(self.calls) > 1
        return _FakeResult("CONVERGED" if converged else "NOTCONVERGED", converged)


class _FallbackProbeModule:
    SOLVE_CALL = {"mode": "optimal", "feasible_fallback": True}
    NOTES = ""

    @staticmethod
    def build():
        return _FallbackProbeModule._prob


def test_driver_feasible_fallback_retries_on_non_convergence():
    prob = _FallbackProbeProblem()
    _FallbackProbeModule._prob = prob
    result = driver.run(_FallbackProbeModule, lambda engine: None)

    assert len(prob.calls) == 2
    assert prob.calls[0] == {
        "mode": "optimal",
        "presolve": False,
        "polish": None,
        "warm": None,
    }
    # No warm= on the retry: the retired combo method passed no warm
    # payload between its phases either, relying on in-place primal
    # continuation -- see driver._dispatch_psiopt_solve's docstring.
    assert prob.calls[1] == {
        "mode": "feasible",
        "presolve": False,
        "polish": None,
        "warm": None,
    }
    assert result["flag"] == "CONVERGED"


class _AcceptableProbeProblem:
    """Stub problem whose first solve() reports ACCEPTABLE (a converged --
    ``bool(result) is True`` -- but not exactly CONVERGED flag) and second
    (the feasible_fallback retry) reports CONVERGED. Regression case: the
    retired ``optimize_solve()``'s own conditional-phase gate skipped the
    trailing Solve phase only on an EXACT CONVERGED
    (``psiopt/src/psiopt.cpp``), so a merely-ACCEPTABLE
    main stage must still trigger the fallback -- the ``not result``/
    ``converged()`` idiom (which treats ACCEPTABLE as already "done") gets
    this wrong; ``result.flag != ConvergenceFlags.CONVERGED`` gets it
    right."""

    def __init__(self):
        self.calls = []

    def solve(self, engine, mode="optimal", presolve=False, polish=None, warm=None):
        self.calls.append(
            {"mode": mode, "presolve": presolve, "polish": polish, "warm": warm}
        )
        if len(self.calls) == 1:
            return _FakeResult("ACCEPTABLE", converged=True)
        return _FakeResult("CONVERGED", converged=True)


class _AcceptableProbeModule:
    SOLVE_CALL = {"mode": "optimal", "feasible_fallback": True}
    NOTES = ""

    @staticmethod
    def build():
        return _AcceptableProbeModule._prob


def test_driver_feasible_fallback_retries_on_acceptable_not_just_notconverged():
    prob = _AcceptableProbeProblem()
    _AcceptableProbeModule._prob = prob
    result = driver.run(_AcceptableProbeModule, lambda engine: None)

    assert len(prob.calls) == 2  # the ACCEPTABLE main stage still triggers the retry
    assert prob.calls[1]["mode"] == "feasible"
    assert result["flag"] == "CONVERGED"


class _NoFallbackNeededProbeProblem:
    """Stub problem whose first solve() already converges -- the
    feasible_fallback retry must NOT run when it isn't needed."""

    def __init__(self):
        self.calls = []

    def solve(self, engine, mode="optimal", presolve=False, polish=None, warm=None):
        self.calls.append(
            {"mode": mode, "presolve": presolve, "polish": polish, "warm": warm}
        )
        return _FakeResult("CONVERGED", converged=True)


class _NoFallbackNeededProbeModule:
    SOLVE_CALL = {"mode": "optimal", "feasible_fallback": True}
    NOTES = ""

    @staticmethod
    def build():
        return _NoFallbackNeededProbeModule._prob


def test_driver_feasible_fallback_skipped_when_first_solve_converges():
    prob = _NoFallbackNeededProbeProblem()
    _NoFallbackNeededProbeModule._prob = prob
    result = driver.run(_NoFallbackNeededProbeModule, lambda engine: None)

    assert len(prob.calls) == 1
    assert result["flag"] == "CONVERGED"
