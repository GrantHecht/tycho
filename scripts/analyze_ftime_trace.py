#!/usr/bin/env python3
"""Analyze -ftime-trace JSON files to identify template instantiation bottlenecks.

Usage:
    python scripts/analyze_ftime_trace.py /tmp/tycho_traces/
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_trace(path):
    """Load a Chrome trace JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    return []


def extract_base_template(name):
    """Extract the base template name (without template args) from a full type."""
    # Strip leading namespaces but keep the class name
    # e.g., "tycho::vf::GenericFunction<-1, -1>" -> "tycho::vf::GenericFunction"
    depth = 0
    result = []
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            result.append(ch)
    return "".join(result).strip()


def extract_tycho_class(name):
    """Extract the outermost tycho:: class name."""
    m = re.match(r"(tycho::\w+::\w+)", name)
    if m:
        return m.group(1)
    m = re.match(r"(tycho::\w+)", name)
    if m:
        return m.group(1)
    return None


def analyze_traces(trace_dir):
    trace_dir = Path(trace_dir)
    json_files = sorted(trace_dir.glob("*.json"))

    if not json_files:
        print(f"No .json files found in {trace_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} trace files\n")

    # Aggregate data across all TUs
    phase_time = defaultdict(float)  # phase -> total_us
    instantiate_class = defaultdict(lambda: {"total_us": 0, "count": 0, "tus": set()})
    instantiate_func = defaultdict(lambda: {"total_us": 0, "count": 0, "tus": set()})
    source_time = defaultdict(lambda: {"total_us": 0, "count": 0})
    codegen_time = defaultdict(lambda: {"total_us": 0, "count": 0})
    per_tu = {}

    for jf in json_files:
        tu_name = jf.stem  # e.g., "bench_solvers.cpp"
        events = load_trace(jf)

        tu_data = {
            "frontend_us": 0,
            "backend_us": 0,
            "source_us": 0,
            "instantiate_class_us": 0,
            "instantiate_func_us": 0,
            "codegen_us": 0,
            "optimize_us": 0,
            "total_us": 0,
            "class_count": 0,
            "func_count": 0,
        }

        for ev in events:
            if ev.get("ph") != "X":  # Duration events only
                continue
            name = ev.get("name", "")
            dur = ev.get("dur", 0)
            detail = ev.get("args", {}).get("detail", "")

            if name == "Total Frontend":
                tu_data["frontend_us"] = dur
                phase_time["Frontend"] += dur
            elif name == "Total Backend":
                tu_data["backend_us"] = dur
                phase_time["Backend"] += dur
            elif name == "Total ExecuteCompiler":
                tu_data["total_us"] = dur
            elif name == "Source":
                tu_data["source_us"] += dur
                phase_time["Source"] += dur
                source_time[detail]["total_us"] += dur
                source_time[detail]["count"] += 1
            elif name == "InstantiateClass":
                tu_data["instantiate_class_us"] += dur
                tu_data["class_count"] += 1
                phase_time["InstantiateClass"] += dur
                base = extract_base_template(detail)
                instantiate_class[base]["total_us"] += dur
                instantiate_class[base]["count"] += 1
                instantiate_class[base]["tus"].add(tu_name)
            elif name == "InstantiateFunction":
                tu_data["instantiate_func_us"] += dur
                tu_data["func_count"] += 1
                phase_time["InstantiateFunction"] += dur
                base = extract_base_template(detail)
                instantiate_func[base]["total_us"] += dur
                instantiate_func[base]["count"] += 1
                instantiate_func[base]["tus"].add(tu_name)
            elif name == "CodeGen Function":
                tu_data["codegen_us"] += dur
                phase_time["CodeGen"] += dur
                base = extract_base_template(detail)
                codegen_time[base]["total_us"] += dur
                codegen_time[base]["count"] += 1
            elif name in ("OptModule", "OptFunction"):
                tu_data["optimize_us"] += dur
                phase_time["Optimize"] += dur

        per_tu[tu_name] = tu_data

    # ============ REPORT ============

    total_all = sum(phase_time.values()) or 1

    print("=" * 90)
    print("COMPILATION PHASE BREAKDOWN (aggregate across all TUs)")
    print("=" * 90)
    for phase, us in sorted(phase_time.items(), key=lambda x: -x[1]):
        print(f"  {phase:<25} {us / 1e6:>8.1f}s  {us / total_all * 100:>5.1f}%")
    print()

    print("=" * 90)
    print("PER-TU BREAKDOWN")
    print("=" * 90)
    print(
        f"{'TU':<40} {'Total':>7} {'Front':>7} {'Back':>7} {'InstCls':>7} {'InstFn':>7} {'CdGen':>7} {'#Cls':>6} {'#Fn':>6}"
    )
    for tu, d in sorted(per_tu.items(), key=lambda x: -x[1]["total_us"]):
        print(
            f"{tu[:39]:<40} {d['total_us'] / 1e6:>6.1f}s {d['frontend_us'] / 1e6:>6.1f}s {d['backend_us'] / 1e6:>6.1f}s "
            f"{d['instantiate_class_us'] / 1e6:>6.1f}s {d['instantiate_func_us'] / 1e6:>6.1f}s {d['codegen_us'] / 1e6:>6.1f}s "
            f"{d['class_count']:>6} {d['func_count']:>6}"
        )
    print()

    print("=" * 90)
    print("TOP 30 CLASS TEMPLATE INSTANTIATIONS (by total time)")
    print("=" * 90)
    for base, data in sorted(
        instantiate_class.items(), key=lambda x: -x[1]["total_us"]
    )[:30]:
        print(
            f"  {data['total_us'] / 1e6:>7.2f}s  {data['count']:>5}x  {len(data['tus']):>2} TUs  {base[:80]}"
        )
    print()

    print("=" * 90)
    print("TOP 30 FUNCTION TEMPLATE INSTANTIATIONS (by total time)")
    print("=" * 90)
    for base, data in sorted(instantiate_func.items(), key=lambda x: -x[1]["total_us"])[
        :30
    ]:
        print(
            f"  {data['total_us'] / 1e6:>7.2f}s  {data['count']:>5}x  {len(data['tus']):>2} TUs  {base[:80]}"
        )
    print()

    print("=" * 90)
    print("TOP 20 HEADERS BY PARSE TIME")
    print("=" * 90)
    for src, data in sorted(source_time.items(), key=lambda x: -x[1]["total_us"])[:20]:
        # Shorten paths
        src_short = src.replace("/home/ghecht/Projects/tycho/", "")
        print(
            f"  {data['total_us'] / 1e6:>7.2f}s  {data['count']:>3}x  {src_short[:75]}"
        )
    print()

    print("=" * 90)
    print("TOP 20 CODE GENERATION COSTS (by total time)")
    print("=" * 90)
    for base, data in sorted(codegen_time.items(), key=lambda x: -x[1]["total_us"])[
        :20
    ]:
        print(f"  {data['total_us'] / 1e6:>7.2f}s  {data['count']:>5}x  {base[:80]}")
    print()

    # ============ DERIVATIVE MODE ANALYSIS ============
    print("=" * 90)
    print("DERIVATIVE MODE INSTANTIATIONS")
    print("=" * 90)

    deriv_modes = defaultdict(int)
    for name in list(instantiate_class.keys()) + list(instantiate_func.keys()):
        if (
            "DenseDerivativeMode" in name
            or "DenseFirstDerivatives" in name
            or "DenseSecondDerivatives" in name
        ):
            # Extract mode
            for mode in ["Analytic", "FDiffFwd", "FDiffCentArray", "AutodiffFwd"]:
                if mode in name:
                    deriv_modes[mode] += 1

    for mode, count in sorted(deriv_modes.items(), key=lambda x: -x[1]):
        print(f"  {mode:<20} {count:>5} instantiations")

    # Count DenseFirstDerivatives and DenseSecondDerivatives instantiations
    first_deriv_count = sum(
        1 for k in instantiate_class if "DenseFirstDerivatives" in k
    )
    second_deriv_count = sum(
        1 for k in instantiate_class if "DenseSecondDerivatives" in k
    )
    print(f"\n  DenseFirstDerivatives distinct types:  {first_deriv_count}")
    print(f"  DenseSecondDerivatives distinct types: {second_deriv_count}")

    # ============ TYCHO CLASS FAMILY AGGREGATION ============
    print()
    print("=" * 90)
    print("TEMPLATE INSTANTIATION TIME BY TYCHO CLASS FAMILY")
    print("=" * 90)

    family_time = defaultdict(float)
    family_count = defaultdict(int)
    for base, data in list(instantiate_class.items()) + list(instantiate_func.items()):
        cls = extract_tycho_class(base)
        if cls:
            family_time[cls] += data["total_us"]
            family_count[cls] += data["count"]

    for cls, us in sorted(family_time.items(), key=lambda x: -x[1])[:25]:
        print(f"  {us / 1e6:>7.2f}s  {family_count[cls]:>6}x  {cls}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_ftime_trace.py <trace_dir>")
        sys.exit(1)
    analyze_traces(sys.argv[1])
