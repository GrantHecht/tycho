"""Unit tests for the globalization campaign sweep driver (scripts/run_campaign.py)."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import run_campaign as rc


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
