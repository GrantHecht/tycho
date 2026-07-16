"""Smoke test for the E2 G0 PSIOPT robustness corpus (tests/corpus/).

Verifies the problem-module contract documented in tests/corpus/README.md
holds for every module registered in tests/corpus/registry.py, and that the
harness (scripts/run_corpus.py) runs end-to-end on the Task 1 stub problems.

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


def test_harness_end_to_end_stub_filter(tmp_path):
    """The harness runs both Task 1 stubs and records the expected statuses."""
    out_path = tmp_path / "stub_results.jsonl"
    proc = subprocess.run(
        [sys.executable, str(RUN_CORPUS), "--filter", "stub", "--out", str(out_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"harness exited {proc.returncode}\n{proc.stderr}"

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, f"expected 2 corpus records, got {len(lines)}: {lines}"

    records = [json.loads(line) for line in lines]
    by_problem = {r["problem"]: r for r in records}
    assert set(by_problem) == {"stub_converges", "stub_fails"}

    assert by_problem["stub_converges"]["status"] == "converged"
    assert by_problem["stub_fails"]["status"] in {"failed", "diverged", "error"}
