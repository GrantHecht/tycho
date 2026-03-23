#!/usr/bin/env python3
"""
Benchmark runner for Tycho.

Usage:
    python run_benchmarks.py [options]

Options:
    --runs N            Number of runs per Python benchmark (default: 10)
    --cpp-runs N        Number of runs per C++ benchmark (default: 5)
    --output FILE       Save results as JSON
    --report FILE       Write a Markdown report (single-run summary)
    --compare FILE      Compare against a previous JSON file
    --label LABEL       Name for this run (default: current git branch)
    --compare-label L   Name for the comparison run (default: "baseline")
    --binary-sizes      Measure sizes of _tychopy extension and bench binary
    --skip-python       Skip Python benchmarks
    --skip-cpp          Skip C++ benchmarks
    --suite SUITE       Run only benchmarks in the given subdirectory (e.g. basic,
                        control, astrodynamics, low_thrust, multi_phase, aircraft)
    --from-json FILE    Load existing results JSON instead of running benchmarks
                        (skips all measurements; useful for re-generating reports)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Matches a standalone floating-point number (e.g. "0.1234", "1.2e-3").
# Requires a decimal point so integer diagnostic counts don't match.
_FLOAT_LINE_RE = re.compile(r"^-?[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _parse_timing(stdout: str) -> float:
    """Extract the timing float from benchmark stdout.

    Benchmark scripts print a single float to stdout, but PSIOPT may emit
    occasional diagnostic lines (e.g. 'Potential Rank Deficiency Detected!!!')
    that slip through even with PrintLevel=4.  This function scans from the
    end and returns the last line that is purely a floating-point number.
    """
    for line in reversed(stdout.strip().split("\n")):
        clean = _ANSI_RE.sub("", line).strip()
        if _FLOAT_LINE_RE.match(clean):
            return float(clean)
    raise ValueError(f"No timing value found in stdout:\n{stdout!r}")


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_python_benchmark(
    script: str, runs: int = 10, timeout: int = 600
) -> Tuple[float, float]:
    """Run a Python benchmark script *runs* times; return (mean, std) of solver times."""
    times = []
    for _ in range(runs):
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Benchmark failed:\n{result.stderr}")
        times.append(_parse_timing(result.stdout))

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    return mean, variance**0.5


def run_cpp_benchmark(binary: str, runs: int = 5) -> Tuple[float, float]:
    """Run a C++ benchmark binary *runs* times; return (mean, std) of solver times."""
    times = []
    for _ in range(runs):
        result = subprocess.run(
            [binary],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Benchmark failed:\n{result.stderr}")
        times.append(float(result.stdout.strip()))

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    return mean, variance**0.5


# ---------------------------------------------------------------------------
# Binary size helpers
# ---------------------------------------------------------------------------


def find_tychopy_extension(build_dir: Path) -> Optional[Path]:
    """Return the path to the _tychopy*.so file, searching build dir then site-packages."""
    # Look inside the build tree first (covers in-tree builds)
    matches = list(build_dir.glob("**/_tychopy*.so"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_size)

    # Fall back to the installed module
    result = subprocess.run(
        [sys.executable, "-c", "import _tychopy; print(_tychopy.__file__)"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        p = Path(result.stdout.strip())
        if p.exists():
            return p
    return None


def measure_binary_sizes(build_dir: Path) -> Dict[str, object]:
    """Return a dict of {name: bytes, name_path: str} for key binaries."""
    sizes: Dict[str, object] = {}

    so = find_tychopy_extension(build_dir)
    if so:
        sizes["_tychopy.so"] = so.stat().st_size
        sizes["_tychopy.so_path"] = str(so)
    else:
        print("  Warning: _tychopy extension not found — skipping .so size")

    bench = (
        build_dir
        / "examples"
        / "cpp_examples"
        / "brachistochrone"
        / "brachistochrone_bench"
    )
    if bench.exists():
        sizes["brachistochrone_bench"] = bench.stat().st_size
        sizes["brachistochrone_bench_path"] = str(bench)
    else:
        print(f"  Warning: bench binary not found at {bench} — skipping bench size")

    return sizes


def format_bytes(n: int) -> str:
    """Human-readable byte count."""
    for unit, divisor in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if n >= divisor:
            return f"{n / divisor:.2f} {unit}"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def generate_report(results: dict, label: str) -> str:
    """Return a Markdown single-run summary report."""
    lines = [
        f"# Tycho Benchmark Report — {label}",
        f"Generated: {results['metadata']['timestamp']}",
        "",
    ]

    compile = results.get("compile")
    if compile:
        lines += [
            "## Compilation Time",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean build time | {compile['mean_s']:.1f}s |",
            f"| Std dev | {compile['std_s']:.1f}s |",
            f"| Trials | {len(compile.get('trials', []))} |",
            "",
        ]

    py = results.get("python", {})
    if py:
        lines += [
            "## Python Benchmarks",
            "",
            "| Benchmark | Mean (s) | Std Dev (s) |",
            "|-----------|----------|-------------|",
        ]
        for name, data in py.items():
            lines.append(f"| {name} | {data['mean']:.4f} | {data['std']:.4f} |")
        lines.append("")

    cpp = results.get("cpp", {})
    if cpp:
        lines += [
            "## C++ Benchmarks",
            "",
            "| Benchmark | Mean (s) | Std Dev (s) |",
            "|-----------|----------|-------------|",
        ]
        for name, data in cpp.items():
            lines.append(f"| {name} | {data['mean']:.4f} | {data['std']:.4f} |")
        lines.append("")

    sizes = results.get("binary_sizes", {})
    size_keys = [k for k in sizes if not k.endswith("_path")]
    if size_keys:
        lines += [
            "## Binary Sizes",
            "",
            "| Binary | Size |",
            "|--------|------|",
        ]
        for key in size_keys:
            lines.append(f"| `{key}` | {format_bytes(sizes[key])} |")
        lines.append("")

    return "\n".join(lines)


def compare_reports(
    results_a: dict,
    label_a: str,
    results_b: dict,
    label_b: str,
) -> str:
    """Return a Markdown side-by-side comparison report.

    Speedup column = a_mean / b_mean (>1 means b is faster than a).
    Binary size reduction = (a - b) / a * 100 % (positive = b is smaller).
    """
    ts_a = results_a["metadata"]["timestamp"]
    ts_b = results_b["metadata"]["timestamp"]

    lines = [
        f"# Tycho Benchmark Comparison: {label_a} vs {label_b}",
        "",
        f"- **{label_a}**: {ts_a}",
        f"- **{label_b}**: {ts_b}",
        "",
    ]

    def _speedup_str(a: Optional[dict], b: Optional[dict]) -> str:
        if a and b and b["mean"] > 0:
            sp = a["mean"] / b["mean"]
            return f"{sp:.3f}×"
        return "—"

    def _time_str(d: Optional[dict]) -> str:
        if d:
            return f"{d['mean']:.4f} ± {d['std']:.4f}"
        return "—"

    # Compilation time
    compile_a = results_a.get("compile")
    compile_b = results_b.get("compile")
    if compile_a or compile_b:
        col = f"Speedup ({label_b}/{label_a})"
        lines += [
            "## Compilation Time",
            "",
            f"| Metric | {label_a} | {label_b} | {col} |",
            "|--------|----------|----------|---------|",
        ]
        a_mean = compile_a["mean_s"] if compile_a else None
        b_mean = compile_b["mean_s"] if compile_b else None
        a_std = compile_a["std_s"] if compile_a else None
        b_std = compile_b["std_s"] if compile_b else None

        a_str = f"{a_mean:.1f} ± {a_std:.1f}s" if a_mean is not None else "—"
        b_str = f"{b_mean:.1f} ± {b_std:.1f}s" if b_mean is not None else "—"
        if a_mean and b_mean and b_mean > 0:
            sp_str = f"{a_mean / b_mean:.3f}×"
        else:
            sp_str = "—"
        lines.append(f"| Full rebuild | {a_str} | {b_str} | {sp_str} |")
        lines.append("")

    # Python
    py_a = results_a.get("python", {})
    py_b = results_b.get("python", {})
    all_py = sorted(set(py_a) | set(py_b))
    if all_py:
        col = f"Speedup ({label_b}/{label_a})"
        lines += [
            "## Python Runtime",
            "",
            f"| Benchmark | {label_a} (s) | {label_b} (s) | {col} |",
            f"|-----------|{''.join(['-' * (len(label_a) + 5), '|', '-' * (len(label_b) + 5), '|', '-' * (len(col) + 2)])}|",
        ]
        for name in all_py:
            a, b = py_a.get(name), py_b.get(name)
            lines.append(
                f"| {name} | {_time_str(a)} | {_time_str(b)} | {_speedup_str(a, b)} |"
            )
        lines.append("")

    # C++
    cpp_a = results_a.get("cpp", {})
    cpp_b = results_b.get("cpp", {})
    all_cpp = sorted(set(cpp_a) | set(cpp_b))
    if all_cpp:
        col = f"Speedup ({label_b}/{label_a})"
        lines += [
            "## C++ Runtime",
            "",
            f"| Benchmark | {label_a} (s) | {label_b} (s) | {col} |",
            f"|-----------|{''.join(['-' * (len(label_a) + 5), '|', '-' * (len(label_b) + 5), '|', '-' * (len(col) + 2)])}|",
        ]
        for name in all_cpp:
            a, b = cpp_a.get(name), cpp_b.get(name)
            lines.append(
                f"| {name} | {_time_str(a)} | {_time_str(b)} | {_speedup_str(a, b)} |"
            )
        lines.append("")

    # Binary sizes
    sz_a = results_a.get("binary_sizes", {})
    sz_b = results_b.get("binary_sizes", {})
    all_sz = sorted(k for k in set(sz_a) | set(sz_b) if not k.endswith("_path"))
    if all_sz:
        lines += [
            "## Binary Sizes",
            "",
            f"| Binary | {label_a} | {label_b} | Reduction |",
            "|--------|----------|----------|-----------|",
        ]
        for key in all_sz:
            a_sz = sz_a.get(key)
            b_sz = sz_b.get(key)
            a_str = format_bytes(a_sz) if isinstance(a_sz, int) else "—"
            b_str = format_bytes(b_sz) if isinstance(b_sz, int) else "—"
            if isinstance(a_sz, int) and isinstance(b_sz, int) and a_sz > 0:
                pct = (a_sz - b_sz) / a_sz * 100
                red_str = f"{pct:+.1f}%"
            else:
                red_str = "—"
            lines.append(f"| `{key}` | {a_str} | {b_str} | {red_str} |")
        lines.append("")

    # Summary
    lines += [
        "## Notes",
        "",
        "- **Speedup** > 1× means the second column is faster.",
        "- **Reduction** positive % means the second column is smaller.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pybind11 vs nanobind benchmarks and generate reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per benchmark run in seconds (default: 600)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per Python benchmark (default: 10)",
    )
    parser.add_argument(
        "--cpp-runs",
        type=int,
        default=5,
        help="Number of runs per C++ benchmark (default: 5)",
    )
    parser.add_argument("--output", type=str, help="Save results as JSON")
    parser.add_argument("--report", type=str, help="Write a Markdown summary report")
    parser.add_argument(
        "--compare",
        type=str,
        help="Path to a previous JSON results file to compare against",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for this run (default: current git branch)",
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        default="baseline",
        help="Label for the comparison run (default: baseline)",
    )
    parser.add_argument(
        "--binary-sizes",
        action="store_true",
        help="Measure sizes of _tychopy extension and bench binary",
    )
    parser.add_argument(
        "--compile-json",
        type=str,
        help="Path to compile timing JSON from bench_compile.sh",
    )
    parser.add_argument(
        "--skip-python", action="store_true", help="Skip Python benchmarks"
    )
    parser.add_argument("--skip-cpp", action="store_true", help="Skip C++ benchmarks")
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Run only benchmarks in this subdirectory (e.g. basic, control, "
        "astrodynamics, low_thrust, multi_phase, aircraft)",
    )
    parser.add_argument(
        "--from-json",
        type=str,
        help="Load existing results JSON instead of running benchmarks",
    )
    args = parser.parse_args()

    label = args.label or _current_branch()

    # ------------------------------------------------------------------
    # Load existing results (--from-json short-circuits all measurements)
    # ------------------------------------------------------------------
    if args.from_json:
        from_path = Path(args.from_json)
        if not from_path.exists():
            print(f"Error: --from-json file not found: {from_path}", file=sys.stderr)
            sys.exit(1)
        with open(from_path) as f:
            results = json.load(f)
        if args.label:
            results.setdefault("metadata", {})["label"] = args.label
        label = results.get("metadata", {}).get("label", label)
        print(f"Loaded results from {from_path} (label: {label})")
    else:
        results = {
            "metadata": {
                "label": label,
                "python_runs": args.runs,
                "cpp_runs": args.cpp_runs,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "python": {},
            "cpp": {},
        }

    bench_dir = Path(__file__).parent  # bench/ directory
    build_dir = bench_dir.parent / "build"  # repo-root build dir

    # ------------------------------------------------------------------
    # Auto-discover Python benchmarks under bench/python/*/bench_*.py
    # Key format: "{suite}/{script_stem}"  e.g. "basic/bench_brachistochrone"
    # ------------------------------------------------------------------
    def discover_python_benchmarks(
        suite_filter: Optional[str],
    ) -> List[Tuple[str, Path]]:
        """Return [(key, path)] sorted by suite then name."""
        python_dir = bench_dir / "python"
        found: List[Tuple[str, Path]] = []
        if not python_dir.exists():
            return found
        for suite_dir in sorted(python_dir.iterdir()):
            if not suite_dir.is_dir():
                continue
            if suite_filter and suite_dir.name != suite_filter:
                continue
            for script in sorted(suite_dir.glob("bench_*.py")):
                key = f"{suite_dir.name}/{script.stem}"
                found.append((key, script))
        return found

    if not args.from_json and not args.skip_python:
        python_benchmarks = discover_python_benchmarks(args.suite)
        if not python_benchmarks:
            filter_msg = f" (suite={args.suite})" if args.suite else ""
            print(f"No Python benchmarks found{filter_msg}.")
        else:
            suite_note = f" [suite={args.suite}]" if args.suite else ""
            print(f"Running Python benchmarks ({args.runs} runs each){suite_note}...")
            for key, script in python_benchmarks:
                print(f"  {key}...", end=" ", flush=True)
                try:
                    mean, std = run_python_benchmark(
                        str(script), args.runs, args.timeout
                    )
                    results["python"][key] = {"mean": mean, "std": std}
                    print(f"{mean:.4f}s ± {std:.4f}s")
                except Exception as exc:
                    print(f"FAILED — {exc}")

    # ------------------------------------------------------------------
    # C++ benchmarks
    # ------------------------------------------------------------------
    cpp_benchmarks = [
        ("brachistochrone_bench", "brachistochrone/brachistochrone_bench"),
        ("gf_eval_bench", "gf_eval/gf_eval_bench"),
    ]

    if not args.from_json and not args.skip_cpp and not args.suite:
        print(f"\nRunning C++ benchmarks ({args.cpp_runs} runs each)...")
        for name, relative_path in cpp_benchmarks:
            binary = build_dir / "bench" / "cpp" / relative_path
            if not binary.exists():
                print(f"  Skipping {name} (not found: {binary})")
                continue
            print(f"  {name}...", end=" ", flush=True)
            try:
                mean, std = run_cpp_benchmark(str(binary), runs=args.cpp_runs)
                results["cpp"][name] = {"mean": mean, "std": std}
                print(f"{mean:.4f}s ± {std:.4f}s")
            except Exception as exc:
                print(f"FAILED — {exc}")

    # ------------------------------------------------------------------
    # Compile timing (from bench_compile.sh output)
    # ------------------------------------------------------------------
    if not args.from_json and args.compile_json:
        compile_path = Path(args.compile_json)
        if not compile_path.exists():
            print(f"Warning: compile JSON not found: {compile_path}", file=sys.stderr)
        else:
            with open(compile_path) as f:
                compile_data = json.load(f)
            results["compile"] = {
                "mean_s": compile_data["mean_s"],
                "std_s": compile_data["std_s"],
                "trials": compile_data.get("trials", []),
            }
            print(
                f"\nCompile timing loaded: {compile_data['mean_s']:.1f}s ± {compile_data['std_s']:.1f}s "
                f"({len(compile_data.get('trials', []))} trial(s))"
            )

    # ------------------------------------------------------------------
    # Binary sizes
    # ------------------------------------------------------------------
    if not args.from_json and args.binary_sizes:
        print("\nMeasuring binary sizes...")
        sizes = measure_binary_sizes(build_dir)
        results["binary_sizes"] = sizes
        for key, val in sizes.items():
            if not key.endswith("_path"):
                print(f"  {key}: {format_bytes(val)}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2))

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    report_text: Optional[str] = None

    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(
                f"Warning: comparison file not found: {compare_path}", file=sys.stderr
            )
        else:
            with open(compare_path) as f:
                compare_results = json.load(f)
            compare_label = args.compare_label or compare_results.get(
                "metadata", {}
            ).get("label", "baseline")
            report_text = compare_reports(
                compare_results, compare_label, results, label
            )

    if report_text is None and args.report:
        report_text = generate_report(results, label)

    if report_text and args.report:
        with open(args.report, "w") as f:
            f.write(report_text)
        print(f"Report written to {args.report}")
    elif report_text:
        print("\n" + report_text)


if __name__ == "__main__":
    main()
