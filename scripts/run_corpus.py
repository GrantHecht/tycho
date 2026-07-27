#!/usr/bin/env python3
"""
run_corpus.py — PSIOPT robustness corpus scoring harness

Runs the problem modules registered in ``tests/corpus/registry.py`` and
scores each against PSIOPT's convergence flag, subprocess-isolated (one
child process per problem, per the proven pattern in ``run_examples.py``)
so a hang or crash in one problem cannot take down the rest of the corpus.
Never gates: the corpus is *expected* to contain problems today's PSIOPT
struggles with — this script only produces scorecards (JSONL).

This file is also its own subprocess child entry point (``--_child``) —
there is no separate child script.

Usage:
    conda run -n tycho python scripts/run_corpus.py [options]

Options:
    --out PATH            Output JSONL path (default: corpus_results.jsonl
                           in the current directory).
    --backend {psiopt,ipopt}  NLP solver backend to drive every selected
                           problem through (default: psiopt). ipopt requires
                           a Tycho build configured with -DENABLE_IPOPT=ON
                           (checked up front via tychopy.solvers.ipopt_available()
                           before any child is spawned).
    --call-shape {module,optimize}  How to invoke each problem (default:
                           module, today's behavior). 'optimize' always runs
                           a single optimize() call, for cross-backend
                           comparability with the ipopt backend's
                           single-solve mapping.
    --config KEY=VALUE...  Zero or more KEY=VALUE pairs. On the psiopt
                           backend these are applied to the PSIOPT optimizer
                           via setattr() inside the child subprocess,
                           immediately before optimize/solve (values are
                           parsed as int, then float, then left as str, in
                           that order). On the ipopt backend these instead
                           populate problem.ipopt_options verbatim as
                           strings (Ipopt's own option parser does its own
                           type coercion).
    --preset NAME          Reserved for a future named-configuration table.
                           No presets are defined yet; passing this always
                           errors with a clear message.
    --filter SUBSTRING     Only run problems whose module name contains
                           SUBSTRING (default: run everything registered).
    --cbwr                 Set MKL_CBWR=AUTO,STRICT in the child environment
                           for bitwise-reproducible MKL reductions.
    --repeat N              Run every selected problem N times (default 1),
                           appending one JSONL record per run — used to
                           check repeat-to-repeat determinism.
    --diff A.jsonl B.jsonl  Instead of running anything, print a per-problem
                           table of status/iteration changes between two
                           previously recorded JSONL files, plus summary
                           counts, and exit. Records from before the
                           --backend column existed are treated as "psiopt".

See tests/corpus/README.md for the full problem-module contract, the exact
PSIOPT-ConvergenceFlags -> status mapping, and the JSONL schema.
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
CORPUS_DIR = REPO_ROOT / "tests" / "corpus"

# tests/corpus/ is not a package (no __init__.py) — it holds registry.py and
# the problems/ package side by side. Put it on sys.path so both `import
# registry` and `import problems.<name>` resolve, in this process and in
# every --_child subprocess spawned from this same file.
if str(CORPUS_DIR) not in sys.path:
    sys.path.insert(0, str(CORPUS_DIR))

DEFAULT_OUT = "corpus_results.jsonl"

# ANSI SGR sequences wrap PSIOPT's colored console output; strip them before
# hunting for the iteration count.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# The bitwise-reproducibility (CBWR) work's proven instrument: PSIOPT prints
# " Iterations : N" once per solve when print_level < 2 (the library
# default).
_ITER_RE = re.compile(r"Iterations : *([0-9]+)")

# Exhaustive PSIOPT ConvergenceFlags -> harness status mapping. Verified
# against _tychopy.solvers.ConvergenceFlags (4 members) — see README.md.
_FLAG_TO_STATUS = {
    "CONVERGED": "converged",
    "ACCEPTABLE": "acceptable",
    "NOTCONVERGED": "failed",
    "DIVERGING": "diverged",
}

# Relative ranking used by --diff to classify a status change as an
# improvement or a regression (lower is better).
_STATUS_RANK = {
    "converged": 0,
    "acceptable": 1,
    "failed": 2,
    "diverged": 3,
    "timeout": 4,
    "error": 5,
}


def _import_registry():
    import registry  # tests/corpus/registry.py

    return registry


# ---------------------------------------------------------------------------
# --config KEY=VALUE parsing
# ---------------------------------------------------------------------------


def _parse_config_value(raw: str):
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            pass
    return raw


def _parse_config_args(pairs, verbatim: bool = False) -> dict:
    """Parse --config KEY=VALUE pairs.

    ``verbatim`` selects the ipopt backend's semantics: every value stays a
    plain string (Ipopt's own option parser does its own type coercion) and
    populates ``problem.ipopt_options`` rather than being setattr()'d onto
    the PSIOPT optimizer, where values are cast int, then float, then left
    as str (see the psiopt-backend docstring at the top of this file).
    """
    config = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--config expects KEY=VALUE, got: {item!r}")
        key, _, raw = item.partition("=")
        config[key] = raw if verbatim else _parse_config_value(raw)
    return config


# ---------------------------------------------------------------------------
# Child mode: run ONE problem in this process, write its result to a file
# ---------------------------------------------------------------------------
#
# The result is written to a dedicated file rather than printed to stdout.
# PSIOPT's C++ console output goes through its own (fmt-buffered) stdio
# stream; when captured through a pipe, that buffer flushes on its own
# schedule relative to Python's `sys.stdout`, and the two interleave at the
# byte level rather than the line level (observed: a printed JSON result
# line landing mid-way through a still-buffered PSIOPT output line). A
# result *file*, read only after the child process has fully exited, sidesteps
# that race entirely; the iteration count is still parsed from captured
# stdout, which is unaffected (the parent only greps it for the regex, it
# never needs a specific line to be intact).


def _run_child(
    module_name: str, config: dict, result_file: str, backend: str, call_shape: str
) -> int:
    registry = _import_registry()
    if module_name not in registry.ALL_PROBLEMS:
        print(f"Unknown corpus problem module: {module_name!r}", file=sys.stderr)
        return 2

    import driver  # tests/corpus/driver.py

    mod = importlib.import_module(f"problems.{module_name}")

    def configure(optimizer):
        for key, value in config.items():
            try:
                setattr(optimizer, key, value)
            except TypeError:
                # Enum-typed properties reject raw ints/strings; coerce through
                # the property's own enum type (by value for ints, by member
                # name for strings) and let a genuine mismatch raise.
                current = getattr(optimizer, key)
                enum_type = type(current)
                if isinstance(value, int):
                    setattr(optimizer, key, enum_type(value))
                else:
                    setattr(optimizer, key, getattr(enum_type, str(value)))

    # config doubles as backend_options on the ipopt backend (see
    # _parse_config_args's verbatim mode, selected by the parent when
    # --backend ipopt); the psiopt-only `configure` closure above is simply
    # unused on that path (driver.run never calls it for backend="ipopt").
    result = driver.run(
        mod, configure, backend=backend, backend_options=config, call_shape=call_shape
    )
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


# ---------------------------------------------------------------------------
# Parent mode: spawn a child per problem, score it, tabulate
# ---------------------------------------------------------------------------


def _score_one(
    name: str,
    tier: str,
    timeout: int,
    config: dict,
    env: dict,
    backend: str,
    call_shape: str,
) -> dict:
    fd, result_path = tempfile.mkstemp(prefix="corpus_result_", suffix=".json")
    os.close(fd)
    os.remove(result_path)  # only reserve the name; the child creates the file
    try:
        cmd = [
            sys.executable,
            str(THIS_FILE),
            "--_child",
            name,
            "--_config",
            json.dumps(config),
            "--_result-file",
            result_path,
            "--_backend",
            backend,
            "--_call-shape",
            call_shape,
        ]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            wall_s = time.monotonic() - t0
            return {
                "problem": name,
                "tier": tier,
                "status": "timeout",
                "flag": None,
                "iterations": -1,
                "wall_s": wall_s,
                "objective": None,
                "returncode": None,
                "notes": f"killed after {timeout}s timeout",
                "backend": backend,
                "call_shape": call_shape,
            }

        wall_s = time.monotonic() - t0
        stdout = proc.stdout or ""
        stripped = _ANSI_RE.sub("", stdout)
        iter_matches = [int(m) for m in _ITER_RE.findall(stripped)]
        iterations = sum(iter_matches) if iter_matches else -1

        result_text = None
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as f:
                result_text = f.read()

        if proc.returncode != 0 or not result_text:
            stderr_tail = (proc.stderr or "").strip().splitlines()
            if not result_text and proc.returncode == 0:
                note = "no result file written by child"
            else:
                note = (
                    stderr_tail[-1]
                    if stderr_tail
                    else f"nonzero exit {proc.returncode}"
                )
            return {
                "problem": name,
                "tier": tier,
                "status": "error",
                "flag": None,
                "iterations": iterations,
                "wall_s": wall_s,
                "objective": None,
                "returncode": proc.returncode,
                "notes": note,
                "backend": backend,
                "call_shape": call_shape,
            }

        try:
            child_result = json.loads(result_text)
        except json.JSONDecodeError as exc:
            return {
                "problem": name,
                "tier": tier,
                "status": "error",
                "flag": None,
                "iterations": iterations,
                "wall_s": wall_s,
                "objective": None,
                "returncode": proc.returncode,
                "notes": f"malformed result file: {exc}",
                "backend": backend,
                "call_shape": call_shape,
            }
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)

    flag_name = child_result.get("flag")
    notes = child_result.get("notes") or ""
    status = _FLAG_TO_STATUS.get(flag_name)
    if status is None:
        status = "error"
        notes = f"{notes}; unknown convergence flag {flag_name!r}".strip("; ")

    if backend == "ipopt":
        # PSIOPT's own console printer never runs a solve under this
        # backend, so the "Iterations : N" instrument the psiopt path
        # relies on (see the module docstring's "Iteration counting"
        # description in tests/corpus/README.md) has nothing to match;
        # trust the child's own count (driver.py's ipopt branch reads it
        # straight from IpoptRunInfo.iterations) instead.
        child_iterations = child_result.get("iterations")
        if isinstance(child_iterations, int):
            iterations = child_iterations
        else:
            notes = f"{notes}; iterations unavailable from instrument".strip("; ")

    return {
        "problem": name,
        "tier": tier,
        "status": status,
        "flag": flag_name,
        "iterations": iterations,
        "wall_s": wall_s,
        "objective": child_result.get("objective"),
        "returncode": proc.returncode,
        "notes": notes,
        "backend": backend,
        "call_shape": call_shape,
    }


def _build_env(cbwr: bool) -> dict:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    if cbwr:
        env["MKL_CBWR"] = "AUTO,STRICT"
    return env


def _print_table(records: list, out_path: Path) -> None:
    if not records:
        print("No records to report.")
        return

    col = max(len(r["problem"]) for r in records) + 2
    print(f"Ran {len(records)} corpus record(s)  ->  {out_path}\n" + "=" * 78)
    print(
        f"  {'problem'.ljust(col)}{'tier':<12}{'backend':<9}{'status':<10}"
        f"{'iters':>7}{'wall_s':>9}{'objective':>16}"
    )
    for r in records:
        obj = "None" if r["objective"] is None else f"{r['objective']:.6g}"
        print(
            f"  {r['problem'].ljust(col)}{r['tier']:<12}{r['backend']:<9}{r['status']:<10}"
            f"{r['iterations']:>7}{r['wall_s']:>9.2f}{obj:>16}"
        )

    counts: dict = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + "=" * 78)
    print("Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


# ---------------------------------------------------------------------------
# --diff mode
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                # Records recorded before the backend column existed are
                # implicitly psiopt runs (that was the only backend then).
                record.setdefault("backend", "psiopt")
                records.append(record)
    return records


def _group_last_per_problem(records: list) -> dict:
    # Later repeats overwrite earlier ones: the diff compares each file's
    # most recently recorded run per problem.
    grouped = {}
    for r in records:
        grouped[r["problem"]] = r
    return grouped


def _print_diff(path_a: str, path_b: str) -> None:
    a = _group_last_per_problem(_load_jsonl(path_a))
    b = _group_last_per_problem(_load_jsonl(path_b))
    names = sorted(set(a) | set(b))
    col = max((len(n) for n in names), default=8) + 2

    print(f"Diff: {path_a}  ->  {path_b}\n" + "=" * 78)
    print(
        f"  {'problem'.ljust(col)}{'backend (a -> b)':<20}"
        f"{'status (a -> b)':<28}{'iterations (a -> b)'}"
    )

    unchanged = improved = regressed = only_a = only_b = 0
    for name in names:
        ra, rb = a.get(name), b.get(name)
        if ra is None:
            only_b += 1
            print(f"  {name.ljust(col)}{'':<20}{'(missing) -> ' + rb['status']}")
            continue
        if rb is None:
            only_a += 1
            print(f"  {name.ljust(col)}{'':<20}{ra['status'] + ' -> (missing)'}")
            continue

        ba, bb = ra["backend"], rb["backend"]
        backend_str = f"{ba} -> {bb}" if ba != bb else ba

        sa, sb = ra["status"], rb["status"]
        ia, ib = ra["iterations"], rb["iterations"]
        delta_str = f"{ia} -> {ib}"
        if ia >= 0 and ib >= 0:
            delta_str += f" ({ib - ia:+d})"

        status_str = f"{sa} -> {sb}" if sa != sb else sa
        print(f"  {name.ljust(col)}{backend_str:<20}{status_str:<28}{delta_str}")

        if sa == sb:
            unchanged += 1
        else:
            rank_a = _STATUS_RANK.get(sa, 99)
            rank_b = _STATUS_RANK.get(sb, 99)
            if rank_b < rank_a:
                improved += 1
            else:
                regressed += 1

    print("\n" + "=" * 78)
    print(
        f"Summary: {len(names)} problem(s); {unchanged} unchanged, {improved} improved, "
        f"{regressed} regressed, {only_a} only-in-a, {only_b} only-in-b"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path.")
    parser.add_argument(
        "--backend",
        choices=("psiopt", "ipopt"),
        default="psiopt",
        help="NLP solver backend to drive every selected problem through "
        "(default: psiopt). ipopt requires a Tycho build configured with "
        "-DENABLE_IPOPT=ON.",
    )
    parser.add_argument(
        "--call-shape",
        choices=("module", "optimize"),
        default="module",
        help="How to invoke each problem: 'module' runs the module's declared "
        "SOLVE_MODE (default, today's behavior); 'optimize' runs a single "
        "optimize() call regardless, for cross-backend comparability with "
        "the ipopt backend's single-solve mapping.",
    )
    parser.add_argument(
        "--config",
        nargs="*",
        action="extend",
        default=[],
        metavar="KEY=VALUE",
        help="KEY=VALUE pairs: setattr onto the optimizer in the child "
        "(--backend psiopt) or populate problem.ipopt_options verbatim as "
        "strings (--backend ipopt).",
    )
    parser.add_argument(
        "--preset",
        default=None,
        metavar="NAME",
        help="Reserved for a future named-configuration table (not yet implemented).",
    )
    parser.add_argument(
        "--filter",
        default="",
        metavar="SUBSTRING",
        help="Only run problems whose module name contains SUBSTRING.",
    )
    parser.add_argument(
        "--cbwr",
        action="store_true",
        help="Set MKL_CBWR=AUTO,STRICT in the child environment.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run every selected problem this many times (default: 1).",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        default=None,
        metavar=("A.jsonl", "B.jsonl"),
        help="Print a status/iteration diff between two recorded JSONL files and exit.",
    )
    # Private child-mode flags: this file is its own subprocess entry point.
    parser.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_result-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_backend", default="psiopt", help=argparse.SUPPRESS)
    parser.add_argument("--_call-shape", default="module", help=argparse.SUPPRESS)
    return parser.parse_args()


def _check_ipopt_available() -> None:
    """Parent-side fast-fail for --backend ipopt: don't spawn 17 doomed children."""
    import tychopy.solvers as solvs

    if not solvs.ipopt_available():
        raise SystemExit(
            "--backend ipopt requires a Tycho build configured with "
            "-DENABLE_IPOPT=ON (tychopy.solvers.ipopt_available() is False "
            "in this environment)."
        )


def main() -> None:
    args = _parse_args()

    if args._child:
        config = json.loads(args._config) if args._config else {}
        sys.exit(
            _run_child(
                args._child,
                config,
                args._result_file,
                args._backend,
                args._call_shape,
            )
        )

    if args.diff is not None:
        _print_diff(args.diff[0], args.diff[1])
        return

    if args.preset is not None:
        raise SystemExit(
            "--preset is reserved for a future named-configuration table; no "
            "presets are defined yet. Use --config KEY=VALUE instead."
        )

    if args.backend == "ipopt":
        _check_ipopt_available()

    registry = _import_registry()
    config = _parse_config_args(args.config, verbatim=(args.backend == "ipopt"))
    names = [n for n in registry.ALL_PROBLEMS if args.filter in n]
    if not names:
        print(f"No corpus problems match filter {args.filter!r}.")
        sys.exit(1)

    env = _build_env(args.cbwr)

    records = []
    for name in names:
        mod = importlib.import_module(f"problems.{name}")
        tier = mod.TIER
        timeout = mod.TIMEOUT
        for _ in range(args.repeat):
            records.append(
                _score_one(
                    name, tier, timeout, config, env, args.backend, args.call_shape
                )
            )

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    _print_table(records, out_path)


if __name__ == "__main__":
    main()
