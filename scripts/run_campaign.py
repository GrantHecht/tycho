#!/usr/bin/env python3
"""
run_campaign.py — globalization configuration-sweep campaign driver

Sweeps PSIOPT's globalization knobs across a six-axis configuration space,
driving the ``tests/corpus`` robustness harness (``scripts/run_corpus.py``)
through every valid combination, and aggregates the results into a
shortlist of promotion candidates.

Axes (``AXES`` below) and the real ``PSIOPT.Settings`` fields they select:

    acceptance_strategy in {0: classic_merit, 1: merit, 2: funnel, 3: filter}
    barrier_governor    in {0: classic_adaptive, 1: monitored}
    restoration_mode    in {0: off, 1: proximal_switch, 2: l1_nested}
    inertia_mode        in {0: classic, 1: proximal_regularization}
    max_soc             in {0, 4}
    recovery            in {0, 1} -- a composite axis (see ``expand_config()``):
        0 -> ls_extended_iters=0, watchdog=False (both recovery links off)
        1 -> ls_extended_iters=E, watchdog=True

``E`` (``RECOVERY_LS_EXTENDED_ITERS``) = 2. Source: watchdog.h's own
recovery-link constants (``kWatchdogShortenedIterTrigger=10``,
``kWatchdogTrialIterMax=3``) gate the WATCHDOG link's arm/trial window, not
``ls_extended_iters`` -- there is no library-side default for
``ls_extended_iters``'s *enable* value (0 disables it; any positive int
enables it, per ``Settings::validate()`` and
``test_LsExtendedItersRoundTrip`` in
``tychopy/test/test_Solvers/test_psiopt_globalization_settings.py``). The
enable value used here is instead pinned from that same test file, which
consistently exercises the "enabled" state at ``ls_extended_iters=2`` across
every composition test that turns the link on (e.g.
``test_merit_ls_extended_iters_composes_and_solves``,
``test_watchdog_max_soc_ls_extended_iters_compose``) -- 2 is therefore the
representative "on" value already used to certify the knob works in
combination with every acceptance strategy, and is reused here so the
sweep's ``recovery=1`` cell matches known-good coverage rather than an
untested magnitude.

192 cells total (4 x 2 x 3 x 2 x 2 x 2). Cell identity is the
sha1(sorted "k=v" join) of the RAW six-axis dict (pre-expansion), truncated
to 12 hex chars -- stable even if E is retuned later, since the hash never
sees the expanded ls_extended_iters/watchdog values.

Subcommands:

    sweep       For each valid cell x repeat, invoke ``run_corpus.py --cbwr
                --config <expanded k=v...> --out
                <store>/cell-<hash>-r<N>.jsonl``. A tiny 2-variable probe
                problem (``tychopy.solvers.OptimizationProblem`` +
                ``Arguments``; same shape as
                ``tychopy/test/test_Solvers/test_nlp_solver_backend.py``'s
                ``_small_problem``) checks ``Settings::validate()`` before
                any real corpus run; a cell that raises ``ValueError`` there
                is recorded in ``<store>/invalid.json`` and skipped for
                every repeat. Resumable: a cell/repeat whose file already
                holds >= ``EXPECTED_PROBLEM_COUNT`` parseable JSONL rows is
                skipped. ``--dry-run`` prints the planned commands without
                probing validity or running anything (no solver calls at
                all). ``--only-cell k=v,k=v,...`` runs exactly one cell
                (bypassing the full 192-cell product) -- used for targeted
                reruns and smoke tests.
    shortlist   Selects cells whose solve-or-acceptable count is within one
                problem of the best cell's AND whose per-problem status
                vector is IDENTICAL across both repeats (status-stability),
                capped at ``cap``, tie-broken by total iterations on the
                commonly solved set.
    aggregate   Reads every complete cell under ``--store`` into one CSV +
                one JSON summary (hash, axis values, per-status counts,
                repeat-stability, per-problem status string);
                ``--context NAME=SCORECARD`` appends a fixed reference row
                read from an existing ``run_corpus.py`` JSONL scorecard
                (e.g. the PSIOPT-defaults or Ipopt baselines) for
                side-by-side comparison.

Coercion note (``is_cell_valid``): ``expand_config()`` already returns typed
Python values (int/bool), so there is no CLI-string parsing step to reuse
from ``run_corpus.py``'s ``_parse_config_args``/``_parse_config_value``
(those coerce raw "KEY=VALUE" *strings*, not already-typed dict values --
not applicable here). What IS reused conceptually is the enum-typed-property
fallback in ``run_corpus.py``'s ``_run_child`` (its nested ``configure``
closure): PSIOPT's enum-typed properties reject a raw int via ``setattr``
with ``TypeError`` even for a numerically valid member (confirmed by
``test_enum_property_rejects_raw_int_assignment`` in
``test_psiopt_globalization_settings.py``), so on ``TypeError`` the fallback
recovers the property's current enum type and reconstructs it
(``enum_type(value)``). That closure is not a standalone importable
function (it is nested inside ``_run_child``), so it is replicated verbatim
below as ``_apply_config()``, with a comment naming its source.
"""

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / registry
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPTS_DIR.parent
RUN_CORPUS = SCRIPTS_DIR / "run_corpus.py"
CORPUS_DIR = REPO_ROOT / "tests" / "corpus"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(CORPUS_DIR) not in sys.path:
    sys.path.insert(0, str(CORPUS_DIR))

import registry  # tests/corpus/registry.py -- single source of truth for problem count

# EXPECTED_PROBLEM_COUNT is 17 at authoring time (see registry.ALL_PROBLEMS);
# computed at import time rather than hard-coded so it tracks the registry.
EXPECTED_PROBLEM_COUNT = len(registry.ALL_PROBLEMS)

# ---------------------------------------------------------------------------
# Axes / cell enumeration
# ---------------------------------------------------------------------------

AXES = {
    "acceptance_strategy": [0, 1, 2, 3],
    "barrier_governor": [0, 1],
    "restoration_mode": [0, 1, 2],
    "inertia_mode": [0, 1],
    "max_soc": [0, 4],
    "recovery": [0, 1],  # expands to ls_extended_iters/watchdog, see expand_config
}

# See the module docstring's "E (RECOVERY_LS_EXTENDED_ITERS)" section for the
# source of this value.
RECOVERY_LS_EXTENDED_ITERS = 2


def enumerate_cells() -> list[dict]:
    """Full product over AXES, one dict per cell (192 total)."""
    keys = list(AXES)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(AXES[k] for k in keys))
    ]


def cell_hash(config: dict) -> str:
    """12-hex sha1 of the sorted 'k=v' join over the RAW (pre-expansion) cell dict."""
    joined = "|".join(f"{k}={v}" for k, v in sorted(config.items()))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def expand_config(cell: dict) -> dict:
    """Map a raw six-axis cell to the real PSIOPT.Settings field names/values.

    Five axes pass through under their real setting names unchanged; the
    composite ``recovery`` axis expands to the two real fields it controls.
    """
    recovery = cell["recovery"]
    return {
        "acceptance_strategy": cell["acceptance_strategy"],
        "barrier_governor": cell["barrier_governor"],
        "restoration_mode": cell["restoration_mode"],
        "inertia_mode": cell["inertia_mode"],
        "max_soc": cell["max_soc"],
        "ls_extended_iters": RECOVERY_LS_EXTENDED_ITERS if recovery else 0,
        "watchdog": bool(recovery),
    }


def cell_file_path(store: Path, cell: dict, repeat: int) -> Path:
    return Path(store) / f"cell-{cell_hash(cell)}-r{repeat}.jsonl"


def _format_cell(cell: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(cell.items()))


# ---------------------------------------------------------------------------
# JSONL scorecard reading
# ---------------------------------------------------------------------------


def _parse_jsonl_rows(path: Path) -> list:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def is_cell_complete(store, cell: dict, repeat: int) -> bool:
    """True iff the cell/repeat file exists and holds >= EXPECTED_PROBLEM_COUNT
    parseable JSONL rows."""
    path = cell_file_path(store, cell, repeat)
    if not path.exists():
        return False
    return len(_parse_jsonl_rows(path)) >= EXPECTED_PROBLEM_COUNT


# ---------------------------------------------------------------------------
# Validity pre-probe
# ---------------------------------------------------------------------------


def _apply_config(optimizer, config: dict) -> None:
    """Apply an expand_config()-shaped dict to a PSIOPT optimizer via setattr.

    Replicates (does not import -- see the module docstring's "Coercion
    note") the setattr/enum-coercion fallback in scripts/run_corpus.py's
    _run_child, specifically its nested ``configure(optimizer)`` closure:
    enum-typed properties (acceptance_strategy, barrier_governor,
    restoration_mode, inertia_mode) reject a raw int via setattr with
    TypeError even when the int is a valid member value, so on TypeError we
    recover the property's current enum type and reconstruct the value
    through it.
    """
    for key, value in config.items():
        try:
            setattr(optimizer, key, value)
        except TypeError:
            current = getattr(optimizer, key)
            enum_type = type(current)
            if isinstance(value, int):
                setattr(optimizer, key, enum_type(value))
            else:
                setattr(optimizer, key, getattr(enum_type, str(value)))


def is_cell_valid(config: dict) -> bool:
    """Probe whether a raw six-axis cell survives Settings::validate().

    Builds the smallest problem that can drive PSIOPT::run_phase_sequence
    (2 variables, quadratic objective, one equality constraint -- the same
    shape as tychopy/test/test_Solvers/test_nlp_solver_backend.py's
    _small_problem), applies expand_config(config), and calls optimize()
    with print_level=0. Convergence is irrelevant; only whether validate()
    accepts the combination (raises ValueError) matters.
    """
    import tychopy.solvers as solvs
    from tychopy.vector_functions import Arguments as Args

    prob = solvs.OptimizationProblem()
    prob.set_vars([0.5, 1.7])
    prob.add_objective((Args(2) - [1.0, 2.0]).squared_norm(), [0, 1])
    prob.add_equal_con(Args(2).squared_norm() - 4.0, [0, 1])
    prob.optimizer.print_level = 0

    _apply_config(prob.optimizer, expand_config(config))

    try:
        prob.optimize()
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# --config k=v formatting for the run_corpus.py subprocess call
# ---------------------------------------------------------------------------


def _config_cli_pairs(config: dict) -> list[str]:
    """Format expand_config()'s typed dict as 'k=v' strings for run_corpus.py's
    --config flag.

    Booleans are rendered as 0/1 (not "True"/"False") so that
    run_corpus._parse_config_value's int-then-float-then-str coercion
    recovers a plain int, which the child's setattr/enum-coercion fallback
    (see _apply_config's docstring above) can turn into the matching bool
    via bool(1)/bool(0) -- round-tripping through the literal string
    "True"/"False" would instead be left as a str by that coercion and then
    fail the fallback's enum-reconstruction branch (bool has no attribute
    named "True").
    """
    pairs = []
    for key, value in config.items():
        if isinstance(value, bool):
            pairs.append(f"{key}={int(value)}")
        else:
            pairs.append(f"{key}={value}")
    return pairs


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def _parse_only_cell(spec: str) -> dict:
    cell = {}
    for pair in spec.split(","):
        if "=" not in pair:
            raise SystemExit(f"--only-cell expects k=v,k=v,..., got: {spec!r}")
        key, _, raw = pair.partition("=")
        if key not in AXES:
            raise SystemExit(
                f"--only-cell: unknown axis {key!r} (expected one of {sorted(AXES)})"
            )
        cell[key] = int(raw)
    missing = set(AXES) - set(cell)
    if missing:
        raise SystemExit(f"--only-cell is missing axes: {sorted(missing)}")
    return cell


def cmd_sweep(args) -> None:
    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)

    cells = [_parse_only_cell(args.only_cell)] if args.only_cell else enumerate_cells()
    total = len(cells) * args.repeats
    n = 0

    if args.dry_run:
        for cell in cells:
            h = cell_hash(cell)
            for repeat in range(1, args.repeats + 1):
                n += 1
                print(f"[{n}/{total}] cell {h} r{repeat}: {_format_cell(cell)}")
        return

    invalid_path = store / "invalid.json"
    invalid = {}
    if invalid_path.exists():
        try:
            invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            invalid = {}

    for cell in cells:
        h = cell_hash(cell)

        if h in invalid:
            n += args.repeats
            print(
                f"[{n}/{total}] cell {h}: previously invalid, skipped ({_format_cell(cell)})"
            )
            continue

        if not is_cell_valid(cell):
            invalid[h] = cell
            invalid_path.write_text(json.dumps(invalid, indent=2), encoding="utf-8")
            n += args.repeats
            print(
                f"[{n}/{total}] cell {h}: invalid (rejected by Settings::validate()), skipped ({_format_cell(cell)})"
            )
            continue

        config = expand_config(cell)
        for repeat in range(1, args.repeats + 1):
            n += 1
            if is_cell_complete(store, cell, repeat):
                print(f"[{n}/{total}] cell {h} r{repeat}: already complete, skipping")
                continue
            cellfile = cell_file_path(store, cell, repeat)
            cmd = [
                sys.executable,
                str(RUN_CORPUS),
                "--cbwr",
                "--out",
                str(cellfile),
                "--config",
                *_config_cli_pairs(config),
            ]
            print(f"[{n}/{total}] cell {h} r{repeat}: running -> {cellfile.name}")
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)


# ---------------------------------------------------------------------------
# shortlist
# ---------------------------------------------------------------------------


def _status_vector(rows: list) -> list:
    return [r["status"] for r in sorted(rows, key=lambda r: r["problem"])]


def shortlist(store, cap: int = 8, expected_problems=None, cells=None) -> list:
    """Shortlist rule: cells whose solve-or-acceptable count is within ONE
    problem of the best cell's, AND status-stable across both repeats; cap
    `cap`; tie-break by total iterations on the commonly solved set.
    Status-stability = identical per-problem status vectors across r1/r2.
    Solve-or-acceptable count from statuses in {"converged", "acceptable"}.
    Returns the winning cell dicts, sorted best-first.
    """
    store = Path(store)
    if expected_problems is None:
        expected_problems = EXPECTED_PROBLEM_COUNT
    if cells is None:
        cells = enumerate_cells()

    candidates = []
    for cell in cells:
        repeats = {}
        for repeat in (1, 2):
            path = cell_file_path(store, cell, repeat)
            if not path.exists():
                repeats = None
                break
            rows = _parse_jsonl_rows(path)
            if len(rows) < expected_problems:
                repeats = None
                break
            repeats[repeat] = rows
        if not repeats:
            continue

        vec1 = _status_vector(repeats[1])
        vec2 = _status_vector(repeats[2])
        if vec1 != vec2:
            continue  # status-unstable across repeats

        solve_count = sum(1 for s in vec1 if s in ("converged", "acceptable"))
        solved_problems = {
            r["problem"]
            for r in repeats[1]
            if r["status"] in ("converged", "acceptable")
        }
        total_iters = sum(
            r["iterations"] for r in repeats[1] if r["problem"] in solved_problems
        )

        candidates.append((cell, solve_count, total_iters))

    if not candidates:
        return []

    best_count = max(c[1] for c in candidates)
    band = [c for c in candidates if c[1] >= best_count - 1]
    band.sort(key=lambda c: (-c[1], c[2]))
    return [c[0] for c in band[:cap]]


def cmd_shortlist(args) -> None:
    picks = shortlist(Path(args.store), cap=args.cap)
    for cell in picks:
        print(f"{cell_hash(cell)}  {_format_cell(cell)}")
    print(f"\n{len(picks)} cell(s) shortlisted (cap={args.cap}).")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

_STATUS_COLUMNS = ("converged", "acceptable", "failed", "diverged", "error", "timeout")

_CSV_FIELDS = [
    "hash",
    "context",
    "acceptance_strategy",
    "barrier_governor",
    "restoration_mode",
    "inertia_mode",
    "max_soc",
    "recovery",
    *_STATUS_COLUMNS,
    "repeat_stable",
    "statuses",
]


def _status_counts(rows: list) -> dict:
    counts = dict.fromkeys(_STATUS_COLUMNS, 0)
    for r in rows:
        s = r.get("status")
        if s in counts:
            counts[s] += 1
    return counts


def _statuses_string(rows: list) -> str:
    return ",".join(
        f"{r['problem']}:{r['status']}"
        for r in sorted(rows, key=lambda r: r["problem"])
    )


def _cell_summary(store: Path, cell: dict, expected_problems: int):
    repeats = {}
    for repeat in (1, 2):
        path = cell_file_path(store, cell, repeat)
        if not path.exists():
            continue
        rows = _parse_jsonl_rows(path)
        if len(rows) >= expected_problems:
            repeats[repeat] = rows
    if not repeats:
        return None

    primary_rows = repeats[max(repeats)]
    repeat_stable = None
    if 1 in repeats and 2 in repeats:
        repeat_stable = _status_vector(repeats[1]) == _status_vector(repeats[2])

    return {
        "hash": cell_hash(cell),
        "context": "",
        **cell,
        **_status_counts(primary_rows),
        "repeat_stable": repeat_stable,
        "statuses": _statuses_string(primary_rows),
        "detail": primary_rows,
    }


def _context_summary(name: str, rows: list) -> dict:
    return {
        "hash": "",
        "context": name,
        "acceptance_strategy": "",
        "barrier_governor": "",
        "restoration_mode": "",
        "inertia_mode": "",
        "max_soc": "",
        "recovery": "",
        **_status_counts(rows),
        "repeat_stable": "",
        "statuses": _statuses_string(rows),
        "detail": rows,
    }


def _write_csv(path: Path, cell_records: list, context_records: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in (*cell_records, *context_records):
            writer.writerow(rec)


def _write_json(path: Path, cell_records: list, context_records: list) -> None:
    payload = {"cells": cell_records, "context": context_records}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cmd_aggregate(args) -> None:
    store = Path(args.store)

    cell_records = []
    for cell in enumerate_cells():
        summary = _cell_summary(store, cell, EXPECTED_PROBLEM_COUNT)
        if summary is not None:
            cell_records.append(summary)

    context_records = []
    for spec in args.context:
        if "=" not in spec:
            raise SystemExit(f"--context expects NAME=PATH, got: {spec!r}")
        name, _, path = spec.partition("=")
        context_records.append(_context_summary(name, _parse_jsonl_rows(Path(path))))

    _write_csv(Path(args.out_csv), cell_records, context_records)
    _write_json(Path(args.out_json), cell_records, context_records)
    print(
        f"Aggregated {len(cell_records)} cell(s) and {len(context_records)} context row(s) "
        f"-> {args.out_csv}, {args.out_json}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sweep = sub.add_parser(
        "sweep", help="Run the corpus over every valid cell x repeat."
    )
    p_sweep.add_argument(
        "--store", required=True, help="Directory holding per-cell JSONL scorecards."
    )
    p_sweep.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Repeats per cell (default: %(default)s).",
    )
    p_sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running anything.",
    )
    p_sweep.add_argument(
        "--only-cell",
        default=None,
        metavar="k=v,k=v,...",
        help="Run exactly one cell (comma-separated axis=value pairs covering all six "
        "axes) instead of the full 192-cell product.",
    )
    p_sweep.set_defaults(func=cmd_sweep)

    p_agg = sub.add_parser(
        "aggregate",
        help="Aggregate complete cells (+ optional context scorecards) into CSV/JSON.",
    )
    p_agg.add_argument(
        "--store", required=True, help="Directory holding per-cell JSONL scorecards."
    )
    p_agg.add_argument("--out-csv", required=True, help="Output CSV path.")
    p_agg.add_argument("--out-json", required=True, help="Output JSON path.")
    p_agg.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="NAME=SCORECARD",
        help="Append a fixed reference row read from an existing JSONL scorecard file "
        "(e.g. a PSIOPT-defaults or Ipopt baseline). Repeatable.",
    )
    p_agg.set_defaults(func=cmd_aggregate)

    p_short = sub.add_parser("shortlist", help="Print the shortlisted cells.")
    p_short.add_argument(
        "--store", required=True, help="Directory holding per-cell JSONL scorecards."
    )
    p_short.add_argument(
        "--cap",
        type=int,
        default=8,
        help="Maximum cells to shortlist (default: %(default)s).",
    )
    p_short.set_defaults(func=cmd_shortlist)

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
