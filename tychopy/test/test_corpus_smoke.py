"""Smoke test for the E2 G0 PSIOPT robustness corpus (tests/corpus/).

Verifies the problem-module contract documented in tests/corpus/README.md
holds for every module registered in tests/corpus/registry.py, and that the
harness (scripts/run_corpus.py) runs end-to-end on two fast real problems
(one converging, one diverging).

This is the only pytest-gated piece of the corpus: the corpus problems
themselves are expected to fail/diverge on today's PSIOPT defaults and are
scored (never gated) by the harness, not by this test.
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "corpus"
RUN_CORPUS = REPO_ROOT / "scripts" / "run_corpus.py"

if str(CORPUS_DIR) not in sys.path:
    sys.path.insert(0, str(CORPUS_DIR))

import registry  # noqa: E402  (tests/corpus/ must be on sys.path first)

VALID_TIERS = {"degenerate", "hard", "literature"}


@pytest.mark.parametrize("module_name", registry.ALL_PROBLEMS)
def test_problem_contract(module_name):
    """Every registered problem module exposes the required contract surface."""
    mod = importlib.import_module(f"problems.{module_name}")

    assert mod.TIER in VALID_TIERS, f"{module_name}: invalid TIER {mod.TIER!r}"
    assert isinstance(mod.TIMEOUT, int) and mod.TIMEOUT > 0, (
        f"{module_name}: TIMEOUT must be a positive int, got {mod.TIMEOUT!r}"
    )
    assert callable(mod.build_and_solve), (
        f"{module_name}: build_and_solve must be callable"
    )


def test_harness_end_to_end_fast_problems(tmp_path):
    """The harness runs end-to-end and records the expected statuses.

    Uses the two fastest real corpus problems instead of the Task 1
    throwaway stubs (deleted in Task 5): ``deg_dup_equality`` (converges in
    3 iterations, ~1 s) and ``hard_vanderpol`` (diverges in 1 iteration,
    ~1 s — the fastest genuine failure in the corpus; the other degenerate-
    tier failures grind through the full 500-iteration ``max_iters`` cap).
    Each is run in its own ``--filter`` invocation since the two problem
    names share no common substring.
    """
    by_problem = {}
    for name in ("deg_dup_equality", "hard_vanderpol"):
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
        by_problem[name] = record

    assert by_problem["deg_dup_equality"]["status"] == "converged"
    assert by_problem["hard_vanderpol"]["status"] == "diverged"
