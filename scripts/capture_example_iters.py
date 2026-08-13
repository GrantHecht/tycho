#!/usr/bin/env python3
"""Capture per-example total interior-point solver iteration counts (CBWR gate instrument).

Runs every example registered in run_examples.py's ALL_EXAMPLES (or the
--start:--end slice of it), sums the "Iterations : N" lines the interior-point solver's own
console printer emits on each example's captured stdout, and writes one CSV
row per example: (example, total_iters, status).

CBWR environment: every example subprocess is run with MPLBACKEND=Agg (as
run_examples.py itself does) and MKL_CBWR=AUTO,STRICT, forcing
bitwise-reproducible MKL reductions -- the same environment
scripts/run_corpus.py's --cbwr flag sets, so total_iters is directly
comparable run-to-run and toolchain-to-toolchain (modulo genuine
solver/library changes).

CSV schema: header row `example,total_iters,status`, one data row per
example run, in run_examples.ALL_EXAMPLES order (or the --start:--end slice
of it). `total_iters` is -1 only when the example timed out; a run whose output
contains no "Iterations : N" line records 0 with its exit status
unchanged (several examples legitimately print none); `status` is "ok"
(exit 0), "exitN"
(nonzero exit N), or "timeout".

Known-noisy trio: MultiSpacecraftOptimization.py, ParallelParking.py, and
SimpleLowThrust.py are, by convention, excluded when DIFFING two captures
against each other for CBWR bit-exactness (see
docs/dev/analysis/2026-07-e2-g6-implicit-tr-regularization.md, "known-noisy
trio excluded as always"). This script does not itself filter them out --
every example in the selected slice is captured and written to the CSV; the
exclusion, when it matters, is applied downstream by whatever compares two
of this script's CSVs.

Reconstruction note: this is the promoted, argparse-ified form of a session
instrument (itself a reconstruction of an earlier gate instrument lost to a
/tmp wipe) -- the measurement itself (environment, regex, per-example
summation) is unchanged from that reconstruction; only the CLI surface
(positional sys.argv -> argparse) changed.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_examples as re_mod

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITER = re.compile(r"Iterations : *([0-9]+)")

DEFAULT_TIMEOUT = 300


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index into run_examples.ALL_EXAMPLES (default: from the beginning).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) into run_examples.ALL_EXAMPLES (default: through the end).",
    )
    parser.add_argument(
        "--timeout-default",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-example timeout in seconds for examples with no entry in "
        "run_examples.TIMEOUT_OVERRIDES (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sl = slice(args.start, args.end)

    env = dict(os.environ, MPLBACKEND="Agg", MKL_CBWR="AUTO,STRICT")
    extra = str(re_mod.EXAMPLES_DIR)
    env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")

    rows = []
    for rel in re_mod.ALL_EXAMPLES[sl]:
        script = re_mod.EXAMPLES_DIR / rel
        timeout = re_mod.TIMEOUT_OVERRIDES.get(rel, args.timeout_default)
        try:
            p = subprocess.run(
                [sys.executable, str(script)],
                cwd=re_mod.EXAMPLES_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            text = ANSI.sub("", p.stdout)
            total = sum(int(m) for m in ITER.findall(text))
            status = "ok" if p.returncode == 0 else f"exit{p.returncode}"
        except subprocess.TimeoutExpired:
            total, status = -1, "timeout"
        rows.append((rel, total, status))
        print(f"{rel}: iters={total} ({status})", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example", "total_iters", "status"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
