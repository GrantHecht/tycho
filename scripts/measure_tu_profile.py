#!/usr/bin/env python3
"""Measure peak RSS, wall time, and generate -ftime-trace for heavy TUs.

Usage:
    python scripts/measure_tu_profile.py [--top N] [--trace-dir DIR] [--stats]

Reads .ninja_log to find the heaviest TUs, extracts their compile commands,
and rebuilds each with /usr/bin/time -v (for peak RSS) and optionally
-ftime-trace (for template profiling) and -Xclang -print-stats.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def get_current_targets():
    """Get list of current .o targets from ninja."""
    result = subprocess.run(
        ["ninja", "-t", "targets", "all"],
        capture_output=True,
        text=True,
        cwd=BUILD,
    )
    targets = []
    for line in result.stdout.splitlines():
        if line.endswith(".cpp.o: CXX_COMPILER__"):
            # Strip the ": CXX_COMPILER__..." suffix
            target = line.split(":")[0].strip()
            targets.append(target)
        elif ".cpp.o:" in line:
            target = line.split(":")[0].strip()
            if target.endswith(".cpp.o"):
                targets.append(target)
    return set(targets)


def parse_ninja_log():
    """Parse .ninja_log for per-TU wall-clock times. Returns dict of target -> seconds."""
    log_path = BUILD / ".ninja_log"
    best = {}
    with open(log_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            try:
                start_ms = int(parts[0])
                end_ms = int(parts[1])
                target = parts[3]
            except (ValueError, IndexError):
                continue
            if not target.endswith(".cpp.o"):
                continue
            time_s = (end_ms - start_ms) / 1000.0
            if target not in best or time_s > best[target]:
                best[target] = time_s
    return best


def get_compile_command(target):
    """Extract compile command for a target from compile_commands.json."""
    cc_path = BUILD / "compile_commands.json"
    with open(cc_path) as f:
        commands = json.load(f)

    # Match by output field
    for entry in commands:
        if entry.get("output", "").endswith(target) or target.endswith(
            entry.get("output", "")
        ):
            return entry["command"], entry["directory"]

    # Fallback: match by source file
    # Extract source file from target path
    # e.g., bench/cpp/CMakeFiles/bench_all.dir/solvers/bench_solvers.cpp.o
    # -> bench/cpp/solvers/bench_solvers.cpp (approximate)
    # Better: use ninja -t commands
    result = subprocess.run(
        ["ninja", "-t", "commands", target],
        capture_output=True,
        text=True,
        cwd=BUILD,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip(), str(ROOT)

    return None, None


def measure_tu(target, trace_dir=None, collect_stats=False):
    """Rebuild a single TU with /usr/bin/time -v and optional -ftime-trace."""
    cmd, directory = get_compile_command(target)
    if not cmd:
        return {"target": target, "error": "Could not find compile command"}

    # Remove the output file to force recompilation
    output_path = BUILD / target
    if output_path.exists():
        output_path.unlink()

    # Add -ftime-trace if requested
    extra_flags = []
    if trace_dir:
        extra_flags.append("-ftime-trace")
    if collect_stats:
        extra_flags.extend(["-Xclang", "-print-stats"])

    if extra_flags:
        # Insert flags after the compiler path, skipping any launcher (ccache, sccache)
        parts = cmd.split(" ")
        insert_idx = 1
        if parts[0].endswith(("ccache", "sccache")):
            insert_idx = 2
        cmd = " ".join(parts[:insert_idx] + extra_flags + parts[insert_idx:])

    # Wrap with /usr/bin/time -v
    full_cmd = f"/usr/bin/time -v {cmd}"

    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=directory,
    )

    # Parse /usr/bin/time output from stderr
    time_output = result.stderr
    peak_rss_kb = None
    wall_time = None
    user_time = None
    sys_time = None

    for line in time_output.splitlines():
        line = line.strip()
        if "Maximum resident set size" in line:
            m = re.search(r"(\d+)", line)
            if m:
                peak_rss_kb = int(m.group(1))
        elif "Elapsed (wall clock) time" in line:
            # Format: h:mm:ss or m:ss or m:ss.ss
            m = re.search(r"(\d+):(\d+\.\d+)", line)
            if m:
                mins = int(m.group(1))
                secs = float(m.group(2))
                wall_time = mins * 60 + secs
            else:
                m = re.search(r"(\d+):(\d+):(\d+\.\d+)", line)
                if m:
                    hrs = int(m.group(1))
                    mins = int(m.group(2))
                    secs = float(m.group(3))
                    wall_time = hrs * 3600 + mins * 60 + secs
        elif "User time (seconds)" in line:
            m = re.search(r"([\d.]+)", line)
            if m:
                user_time = float(m.group(1))
        elif "System time (seconds)" in line:
            m = re.search(r"([\d.]+)", line)
            if m:
                sys_time = float(m.group(1))

    # Collect -print-stats output if requested
    stats_text = None
    if collect_stats:
        # Stats go to stderr along with time output
        stats_lines = []
        in_stats = False
        for line in time_output.splitlines():
            if "===" in line and (
                "Clang" in line or "LLVM" in line or "Stats" in line.title()
            ):
                in_stats = True
            if in_stats:
                stats_lines.append(line)
        if stats_lines:
            stats_text = "\n".join(stats_lines)

    # Move -ftime-trace JSON if it was generated
    trace_file = None
    if trace_dir:
        json_path = output_path.with_suffix(".json")
        if json_path.exists():
            dest = Path(trace_dir) / json_path.name
            shutil.move(str(json_path), str(dest))
            trace_file = str(dest)

    return {
        "target": target,
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_gb": round(peak_rss_kb / 1048576, 2) if peak_rss_kb else None,
        "wall_time_s": round(wall_time, 1) if wall_time else None,
        "user_time_s": round(user_time, 1) if user_time else None,
        "sys_time_s": round(sys_time, 1) if sys_time else None,
        "compile_ok": result.returncode == 0,
        "trace_file": trace_file,
        "stats_text": stats_text,
    }


def short_name(target):
    """Extract a human-readable short name from a target path."""
    # bench/cpp/CMakeFiles/bench_all.dir/solvers/bench_solvers.cpp.o -> bench_solvers.cpp
    return target.rsplit("/", 1)[-1].replace(".o", "")


def categorize(target):
    if target.startswith("src/bindings/"):
        return "bindings"
    elif target.startswith("src/"):
        return "core"
    elif target.startswith("tests/"):
        return "tests"
    elif target.startswith("bench/"):
        return "bench"
    elif target.startswith("examples/"):
        return "examples"
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Profile heavy TUs")
    parser.add_argument(
        "--top", type=int, default=20, help="Number of heaviest TUs to profile"
    )
    parser.add_argument(
        "--trace-dir", type=str, default=None, help="Dir to collect -ftime-trace JSONs"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Also collect -Xclang -print-stats"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)

    # Get current targets and timing data
    current = get_current_targets()
    timing = parse_ninja_log()

    # Filter to current targets and sort by time
    ranked = sorted(
        [(t, timing[t]) for t in timing if t in current],
        key=lambda x: -x[1],
    )

    top_n = ranked[: args.top]
    print(f"Profiling top {len(top_n)} heaviest TUs (of {len(ranked)} current targets)")
    print(f"{'=' * 90}")

    results = []
    for i, (target, ninja_time) in enumerate(top_n):
        name = short_name(target)
        cat = categorize(target)
        print(f"\n[{i + 1}/{len(top_n)}] {name} ({cat}, ninja log: {ninja_time:.0f}s)")
        print(f"  Target: {target}")

        result = measure_tu(target, args.trace_dir, args.stats)
        result["ninja_time_s"] = round(ninja_time, 1)
        result["category"] = cat
        result["short_name"] = name
        results.append(result)

        if result.get("peak_rss_gb"):
            print(f"  Peak RSS: {result['peak_rss_gb']:.2f} GB")
            print(f"  Wall time: {result['wall_time_s']:.1f}s")
        elif result.get("error"):
            print(f"  ERROR: {result['error']}")

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"{'TU':<45} {'Cat':<8} {'RSS (GB)':>8} {'Wall (s)':>8} {'Ninja (s)':>9}")
    print(f"{'-' * 45} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9}")
    for r in sorted(results, key=lambda x: -(x.get("peak_rss_kb") or 0)):
        name = r["short_name"][:44]
        cat = r["category"]
        rss = f"{r['peak_rss_gb']:.2f}" if r.get("peak_rss_gb") else "N/A"
        wall = f"{r['wall_time_s']:.1f}" if r.get("wall_time_s") else "N/A"
        ninja = f"{r['ninja_time_s']:.1f}"
        print(f"{name:<45} {cat:<8} {rss:>8} {wall:>8} {ninja:>9}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
