#!/usr/bin/env python3
"""
run_examples.py — Tycho example test runner

Runs every example under examples/ using the non-interactive Agg matplotlib
backend and reports PASS / FAIL / SKIP for each.  Exits 0 if all runnable
examples pass, 1 if any fail.

Usage:
    python run_examples.py [--timeout SECONDS] [--filter SUBSTRING]

    --timeout   Per-example time limit in seconds (default: 300).
                Mesh-refinement and heavy astrodynamics examples have
                individual overrides defined below.
    --filter    Only run examples whose relative path contains SUBSTRING.

Environment:
    Activate the tycho conda environment before running:
        conda activate tycho && python run_examples.py
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "python_examples"

DEFAULT_TIMEOUT = 300  # seconds

# Examples that are genuinely slow get a higher ceiling.
TIMEOUT_OVERRIDES = {
    "BettsLowThrust.py": 600,
    "DionysusLowThrust.py": 600,
    "Heteroclinic.py": 600,
    "OrbitContinuation.py": 600,
    "MultiSpacecraftOptimization.py": 600,
    "MeshRefinement/CartPole.py": 600,
    "MeshRefinement/Delta3Launch.py": 600,
    "MeshRefinement/Reentry.py": 600,
    "MeshRefinement/HyperSensLong.py": 900,
    "UpdatedInterface/BettsLowThrust.py": 600,
}

# Ordered list of all examples to run, relative to EXAMPLES_DIR.
# Order matters: tables-only helpers run before the examples that import them.
ALL_EXAMPLES = [
    # --- Core examples ---
    "AnalyticExample.py",
    "BikeObstacle.py",
    "Brachistochrone.py",
    "BrysonDenham.py",
    "CartPole.py",
    "FreeFlyingRobotExample.py",
    "GoddardRocket.py",
    "HangingChain.py",
    "Heteroclinic.py",
    "HyperSens.py",
    "MinimumTimeToClimbTables.py",
    "MinimumTimeToClimb.py",
    "MountainCar.py",
    "MultiPhaseCannon.py",
    "MultiPhaseZermelo.py",
    "MultiSpacecraftOptimization.py",
    "OptimalDocking.py",
    "OrbitContinuation.py",
    "ParallelParking.py",
    "SimpleLowThrust.py",
    "TopputtoLowThrust.py",
    "VanDerPol.py",
    "Zermelo.py",
    # Basemap examples
    "BettsLowThrust.py",
    "Delta3Launch.py",
    "DionysusLowThrust.py",
    "Reentry.py",
    # --- Mesh refinement ---
    "MeshRefinement/CartPole.py",
    "MeshRefinement/Delta3Launch.py",
    "MeshRefinement/HyperSensLong.py",
    "MeshRefinement/Reentry.py",
    # --- Updated interface ---
    "UpdatedInterface/GoddardRocket.py",
    "UpdatedInterface/MultiPhaseCannon.py",
    "UpdatedInterface/Reentry.py",
    "UpdatedInterface/MinimumTimeToClimbTables.py",
    "UpdatedInterface/MinimumTimeToClimb.py",
    "UpdatedInterface/Delta3Launch.py",
    "UpdatedInterface/BettsLowThrust.py",
]

# Optional imports: if any listed module cannot be imported the example is
# skipped rather than failed, so the suite stays green when optional system
# packages are absent.
OPTIONAL_DEPS: dict[str, list[str]] = {
    "BettsLowThrust.py": ["mpl_toolkits.basemap"],
    "Delta3Launch.py": ["mpl_toolkits.basemap"],
    "DionysusLowThrust.py": ["mpl_toolkits.basemap"],
    "Reentry.py": ["mpl_toolkits.basemap"],
    "MeshRefinement/Delta3Launch.py": ["mpl_toolkits.basemap"],
    "MeshRefinement/Reentry.py": ["mpl_toolkits.basemap"],
    "UpdatedInterface/BettsLowThrust.py": ["mpl_toolkits.basemap"],
    "UpdatedInterface/Delta3Launch.py": ["mpl_toolkits.basemap"],
}

# Output directories that some examples assume already exist.
REQUIRED_DIRS = [
    "Plots/Zermelo",
    "Plots/OrbitContinuation",
    "Plots/MultiSpacecraftOptimization",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _can_import(module: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
    )
    return r.returncode == 0


def _check_optional_deps() -> dict[str, bool]:
    all_mods = {m for deps in OPTIONAL_DEPS.values() for m in deps}
    available = {m: _can_import(m) for m in all_mods}
    missing = [m for m, ok in available.items() if not ok]
    if missing:
        print(f"Optional dependencies not found (affected examples will be skipped):")
        for m in missing:
            print(f"  - {m}")
        print()
    return available


def _build_env() -> dict[str, str]:
    """Subprocess environment: force non-interactive matplotlib backend and
    ensure examples/ is on PYTHONPATH so subdirectory scripts can import
    helpers (e.g. UpdatedInterface/MinimumTimeToClimb.py → MinimumTimeToClimbTables)."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    existing_pythonpath = env.get("PYTHONPATH", "")
    extra = str(EXAMPLES_DIR)
    env["PYTHONPATH"] = (
        f"{extra}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else extra
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Default per-example timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--filter",
        metavar="SUBSTRING",
        default="",
        help="Only run examples whose path contains SUBSTRING",
    )
    args = parser.parse_args()

    dep_cache = _check_optional_deps()

    # Pre-create directories that examples write plot files into.
    for d in REQUIRED_DIRS:
        (EXAMPLES_DIR / d).mkdir(parents=True, exist_ok=True)

    env = _build_env()

    examples = [e for e in ALL_EXAMPLES if args.filter in e]
    if not examples:
        print(f"No examples match filter '{args.filter}'.")
        sys.exit(1)

    col_width = max(len(e) for e in examples) + 2
    print(f"Running {len(examples)} examples  (cwd: {EXAMPLES_DIR})\n" + "=" * 72)

    results: list[tuple[str, str, float, str]] = []  # (path, status, seconds, detail)

    for rel in examples:
        label = rel.ljust(col_width)

        # Skip if optional deps are missing.
        missing_deps = [
            m for m in OPTIONAL_DEPS.get(rel, []) if not dep_cache.get(m, True)
        ]
        if missing_deps:
            print(
                f"  {label} {YELLOW}SKIP{RESET}  (missing: {', '.join(missing_deps)})"
            )
            results.append((rel, "SKIP", 0.0, f"missing: {', '.join(missing_deps)}"))
            continue

        timeout = TIMEOUT_OVERRIDES.get(rel, args.timeout)
        script = EXAMPLES_DIR / rel
        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(EXAMPLES_DIR),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0

            if proc.returncode == 0:
                print(f"  {label} {GREEN}PASS{RESET}  ({elapsed:.1f}s)")
                results.append((rel, "PASS", elapsed, ""))
            else:
                # Show the last few lines of stderr for quick diagnosis.
                err_lines = proc.stderr.strip().splitlines()
                summary = err_lines[-1] if err_lines else "(no stderr)"
                print(f"  {label} {RED}FAIL{RESET}  ({elapsed:.1f}s)  {summary}")
                results.append((rel, "FAIL", elapsed, proc.stderr.strip()))

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            msg = f"timed out after {timeout}s"
            print(f"  {label} {RED}FAIL{RESET}  {msg}")
            results.append((rel, "FAIL", elapsed, msg))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = [r for r in results if r[1] == "PASS"]
    failed = [r for r in results if r[1] == "FAIL"]
    skipped = [r for r in results if r[1] == "SKIP"]

    print("\n" + "=" * 72)
    print(
        f"Results: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped\n"
    )

    if failed:
        print(f"{RED}Failed examples:{RESET}")
        for name, _, elapsed, detail in failed:
            print(f"  {name}  ({elapsed:.1f}s)")
            if detail:
                for line in detail.splitlines()[-8:]:
                    print(f"    {line}")
        print()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
