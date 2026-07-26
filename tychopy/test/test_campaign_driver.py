"""Unit tests for the globalization campaign sweep driver (scripts/run_campaign.py)."""

import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytest

# The relative-depth default assumes the suite runs from its in-repo location;
# harnesses that copy the suite elsewhere (e.g. the wheel-layout CI lane
# running from a scratch directory) set TYCHO_REPO_ROOT to point back at the
# checkout, the same contract test_corpus_smoke.py uses.
REPO = Path(os.environ.get("TYCHO_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO / "scripts"))

rc = pytest.importorskip(
    "run_campaign",
    reason="campaign driver tests need the repository checkout "
    "(scripts/run_campaign.py is not shipped); set TYCHO_REPO_ROOT when "
    "running the suite from a copied location",
)


def _write_cell_jsonl(tmp_path, cfg, repeat, statuses, iters=None):
    """Write one cell/repeat JSONL scorecard: `statuses[i]`/`iters[i]` for
    problem `p{i}`. `iters` defaults to 10 for every problem."""
    if iters is None:
        iters = [10] * len(statuses)
    h = rc.cell_hash(cfg)
    rows = [
        json.dumps(
            {
                "problem": f"p{i}",
                "status": s,
                "flag": "X",
                "iterations": it,
                "wall_s": 0.1,
                "objective": 0.0,
                "returncode": 0,
                "notes": "",
                "backend": "psiopt",
            }
        )
        for i, (s, it) in enumerate(zip(statuses, iters))
    ]
    (tmp_path / f"cell-{h}-r{repeat}.jsonl").write_text("\n".join(rows) + "\n")


def test_enumeration_covers_full_axis_product():
    cells = rc.enumerate_cells()
    assert len(cells) == 192

    # Every cell carries exactly the six-axis keys.
    key_sets = {frozenset(c) for c in cells}
    assert key_sets == {frozenset(rc.AXES)}

    # Both extremes of the axis product are present.
    all_min = {
        "acceptance_strategy": 0,
        "barrier_governor": 0,
        "restoration_mode": 0,
        "inertia_mode": 0,
        "max_soc": 0,
        "recovery": 0,
    }
    all_max = {
        "acceptance_strategy": 3,
        "barrier_governor": 1,
        "restoration_mode": 2,
        "inertia_mode": 1,
        "max_soc": 4,
        "recovery": 1,
    }
    assert all_min in cells
    assert all_max in cells


def test_cell_hash_stable_and_order_independent():
    a = {"acceptance_strategy": 1, "inertia_mode": 0}
    b = {"inertia_mode": 0, "acceptance_strategy": 1}
    assert rc.cell_hash(a) == rc.cell_hash(b)
    assert len(rc.cell_hash(a)) == 12
    assert rc.cell_hash({"acceptance_strategy": 2, "inertia_mode": 0}) != rc.cell_hash(
        a
    )


def test_resume_skips_complete_cells(tmp_path):
    cfg = {"acceptance_strategy": 1}
    h = rc.cell_hash(cfg)
    f = tmp_path / f"cell-{h}-r1.jsonl"
    rows = [
        json.dumps(
            {
                "problem": f"p{i}",
                "status": "converged",
                "flag": "CONVERGED",
                "iterations": 3,
                "wall_s": 0.1,
                "objective": 0.0,
                "returncode": 0,
                "notes": "",
                "backend": "psiopt",
            }
        )
        for i in range(rc.EXPECTED_PROBLEM_COUNT)
    ]
    f.write_text("\n".join(rows) + "\n")
    assert rc.is_cell_complete(tmp_path, cfg, repeat=1)
    assert not rc.is_cell_complete(tmp_path, cfg, repeat=2)
    f.write_text("\n".join(rows[:-1]) + "\n")  # truncated -> incomplete
    assert not rc.is_cell_complete(tmp_path, cfg, repeat=1)


def test_shortlist_band_stability_and_cap(tmp_path):
    # Synthesize 3 configs x 2 repeats over 4 problems:
    #   best: 4 solved, stable          -> shortlisted
    #   near: 3 solved, stable          -> shortlisted (within one of best)
    #   flaky: 4 solved r1 / 3 solved r2 -> excluded (status-unstable)
    def write(cfg, repeat, statuses):
        h = rc.cell_hash(cfg)
        rows = [
            json.dumps(
                {
                    "problem": f"p{i}",
                    "status": s,
                    "flag": "X",
                    "iterations": 10,
                    "wall_s": 0.1,
                    "objective": 0.0,
                    "returncode": 0,
                    "notes": "",
                    "backend": "psiopt",
                }
            )
            for i, s in enumerate(statuses)
        ]
        (tmp_path / f"cell-{h}-r{repeat}.jsonl").write_text("\n".join(rows) + "\n")

    best = {"acceptance_strategy": 1}
    near = {"acceptance_strategy": 2}
    flaky = {"acceptance_strategy": 3}
    for r in (1, 2):
        write(best, r, ["converged"] * 4)
        write(near, r, ["converged"] * 3 + ["failed"])
    write(flaky, 1, ["converged"] * 4)
    write(flaky, 2, ["converged"] * 3 + ["failed"])

    picks = rc.shortlist(
        tmp_path, cap=8, expected_problems=4, cells=[best, near, flaky]
    )
    hashes = [rc.cell_hash(c) for c in picks]
    assert rc.cell_hash(best) in hashes and rc.cell_hash(near) in hashes
    assert rc.cell_hash(flaky) not in hashes


def test_shortlist_tiebreak_prefers_lower_common_iterations(tmp_path):
    # Two stable cells, both solving all 4 problems (equal solve count),
    # but with different iteration totals on their (here: identical)
    # commonly-solved set -- the lower-total cell must rank first.
    fast = {"acceptance_strategy": 1}
    slow = {"acceptance_strategy": 2}
    for r in (1, 2):
        _write_cell_jsonl(tmp_path, fast, r, ["converged"] * 4, iters=[5, 5, 5, 5])
        _write_cell_jsonl(tmp_path, slow, r, ["converged"] * 4, iters=[50, 50, 50, 50])

    picks = rc.shortlist(tmp_path, cap=8, expected_problems=4, cells=[slow, fast])
    hashes = [rc.cell_hash(c) for c in picks]
    assert hashes == [rc.cell_hash(fast), rc.cell_hash(slow)]


def test_shortlist_tiebreak_uses_intersection_not_self_set(tmp_path):
    # Cell A solves {p0, p1, p3} and is extremely fast on p3 -- a problem
    # cell B does not solve at all. Cell B solves {p0, p1, p2} and is
    # genuinely much faster than A on the problems they both solve (p0,
    # p1). A's own solved-problem set has a lower total (100+100+1=201)
    # than B's own solved-problem set (5+5+200=210), so a *self-set*
    # tie-break would incorrectly rank A first. The correct cross-cell
    # tie-break restricts to the intersection {p0, p1}, where A totals 200
    # and B totals 10 -- B must rank first.
    cell_a = {"acceptance_strategy": 1}
    cell_b = {"acceptance_strategy": 2}
    for r in (1, 2):
        _write_cell_jsonl(
            tmp_path,
            cell_a,
            r,
            ["converged", "converged", "failed", "converged"],
            iters=[100, 100, 999, 1],
        )
        _write_cell_jsonl(
            tmp_path,
            cell_b,
            r,
            ["converged", "converged", "converged", "failed"],
            iters=[5, 5, 200, 999],
        )

    picks = rc.shortlist(tmp_path, cap=8, expected_problems=4, cells=[cell_a, cell_b])
    hashes = [rc.cell_hash(c) for c in picks]
    assert hashes == [rc.cell_hash(cell_b), rc.cell_hash(cell_a)]


def test_shortlist_empty_intersection_preserves_candidate_order(tmp_path):
    # Cell A solves only p0; cell B solves only p1 -- disjoint solved sets,
    # so the band's intersection is empty and the iteration tie-break must
    # not apply. Both have solve_count=1 (tied), so the band keeps its
    # input order (stable sort) -- A (listed first) stays first even
    # though it has the higher iteration count on its own solved problem.
    cell_a = {"acceptance_strategy": 1}
    cell_b = {"acceptance_strategy": 2}
    for r in (1, 2):
        _write_cell_jsonl(tmp_path, cell_a, r, ["converged", "failed"], iters=[999, 0])
        _write_cell_jsonl(tmp_path, cell_b, r, ["failed", "converged"], iters=[0, 1])

    picks = rc.shortlist(tmp_path, cap=8, expected_problems=2, cells=[cell_a, cell_b])
    hashes = [rc.cell_hash(c) for c in picks]
    assert hashes == [rc.cell_hash(cell_a), rc.cell_hash(cell_b)]


def test_aggregate_cell_summary_exposes_both_repeats(tmp_path):
    # When repeat_stable is False, the aggregate record must still expose
    # BOTH repeats' status strings/detail, not just the higher-indexed one.
    cfg = {"acceptance_strategy": 1}
    _write_cell_jsonl(tmp_path, cfg, 1, ["converged", "converged"], iters=[3, 4])
    _write_cell_jsonl(tmp_path, cfg, 2, ["converged", "failed"], iters=[3, 500])

    summary = rc._cell_summary(tmp_path, cfg, expected_problems=2)
    assert summary["repeat_stable"] is False
    assert summary["statuses"] == "p0:converged,p1:converged"
    assert summary["statuses_r2"] == "p0:converged,p1:failed"
    assert set(summary["detail"]) == {"1", "2"}
    assert [r["status"] for r in summary["detail"]["1"]] == ["converged", "converged"]
    assert [r["status"] for r in summary["detail"]["2"]] == ["converged", "failed"]


def test_cell_summary_call_shape_defaults_to_module_for_legacy_rows(tmp_path):
    # Pre-existing campaign scorecards (recorded before the call_shape column
    # existed) have rows with no "call_shape" key at all -- _write_cell_jsonl
    # reproduces that shape exactly (it never sets the key) -- so aggregation
    # must default to "module" rather than raising or leaving it missing.
    cfg = {"acceptance_strategy": 1}
    _write_cell_jsonl(tmp_path, cfg, 1, ["converged", "converged"], iters=[3, 4])

    summary = rc._cell_summary(tmp_path, cfg, expected_problems=2)
    assert summary["call_shape"] == "module"


def test_context_summary_call_shape_defaults_to_module_for_legacy_rows():
    rows = [{"problem": "p0", "status": "converged", "iterations": 1}]
    summary = rc._context_summary("baseline", rows)
    assert summary["call_shape"] == "module"


def test_aggregate_cell_summary_single_repeat_blank_statuses_r2(tmp_path):
    cfg = {"acceptance_strategy": 1}
    _write_cell_jsonl(tmp_path, cfg, 1, ["converged", "converged"], iters=[3, 4])

    summary = rc._cell_summary(tmp_path, cfg, expected_problems=2)
    assert summary["repeat_stable"] is None
    assert summary["statuses_r2"] == ""
    assert set(summary["detail"]) == {"1"}


def test_context_summary_repeat_stable_is_none_not_blank():
    rows = [
        {
            "problem": "p0",
            "status": "converged",
            "iterations": 1,
        }
    ]
    summary = rc._context_summary("baseline", rows)
    assert summary["repeat_stable"] is None


def test_parse_only_cell_bad_value_raises_clean_systemexit():
    with pytest.raises(SystemExit, match="acceptance_strategy.*not-an-int"):
        rc._parse_only_cell(
            "acceptance_strategy=not-an-int,barrier_governor=0,"
            "restoration_mode=0,inertia_mode=0,max_soc=0,recovery=0"
        )


def test_cmd_sweep_skips_validity_probe_when_all_repeats_complete(
    tmp_path, monkeypatch
):
    cfg = {
        "acceptance_strategy": 1,
        "barrier_governor": 0,
        "restoration_mode": 0,
        "inertia_mode": 0,
        "max_soc": 0,
        "recovery": 0,
    }
    for r in (1, 2):
        _write_cell_jsonl(tmp_path, cfg, r, ["converged"] * rc.EXPECTED_PROBLEM_COUNT)

    def _boom(_config):
        raise AssertionError(
            "is_cell_valid must not be probed when all repeats are complete"
        )

    monkeypatch.setattr(rc, "is_cell_valid", _boom)

    args = argparse.Namespace(
        store=str(tmp_path),
        repeats=2,
        dry_run=False,
        only_cell=",".join(f"{k}={v}" for k, v in cfg.items()),
    )
    rc.cmd_sweep(args)  # must not raise


def test_cmd_aggregate_missing_context_file_raises_clean_systemexit(tmp_path):
    args = argparse.Namespace(
        store=str(tmp_path),
        out_csv=str(tmp_path / "out.csv"),
        out_json=str(tmp_path / "out.json"),
        context=[f"baseline={tmp_path / 'does-not-exist.jsonl'}"],
    )
    with pytest.raises(SystemExit, match="baseline.*does-not-exist"):
        rc.cmd_aggregate(args)


def test_cmd_sweep_includes_call_shape_flag(tmp_path, monkeypatch):
    cfg = {
        "acceptance_strategy": 1,
        "barrier_governor": 0,
        "restoration_mode": 0,
        "inertia_mode": 0,
        "max_soc": 0,
        "recovery": 0,
    }
    monkeypatch.setattr(rc, "is_cell_valid", lambda _config: True)

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(rc.subprocess, "run", _fake_run)

    args = argparse.Namespace(
        store=str(tmp_path),
        repeats=1,
        dry_run=False,
        only_cell=",".join(f"{k}={v}" for k, v in cfg.items()),
        call_shape="optimize",
    )
    rc.cmd_sweep(args)

    cmd = captured["cmd"]
    assert "--call-shape" in cmd
    assert cmd[cmd.index("--call-shape") + 1] == "optimize"


def test_is_cell_valid_returns_false_on_malformed_config():
    # An out-of-range axis value (e.g. from a hand-typed --only-cell spec)
    # must be caught inside is_cell_valid and land the cell as invalid
    # (False), not crash the sweep with an uncaught ValueError/TypeError.
    bad_cell = {
        "acceptance_strategy": 99,
        "barrier_governor": 0,
        "restoration_mode": 0,
        "inertia_mode": 0,
        "max_soc": 0,
        "recovery": 0,
    }
    assert rc.is_cell_valid(bad_cell) is False
