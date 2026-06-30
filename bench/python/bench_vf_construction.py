#!/usr/bin/env python3
"""VectorFunction construction microbenchmark.

Measures the Python-side cost of constructing the type-erased ``GenericFunction``
wrappers that nearly every VectorFunction operation returns by value
(``f + g``, ``.normalized()``, ``.norm()``, ``vf.stack(...)``, ...). This is the
workload that nanobind 2.13.0's ``nb::pooled()`` instance recycling targets — the
C++ ``bench/cpp`` suite does not exercise the binding layer, so this is the
Python-path "ruler" for that change.

Reports, per workload: throughput as *counted construction ops/sec* — a stable
per-build proxy, NOT a literal count of every allocated object (leaf ``Arguments``
/ ``Segment`` constructions are deliberately excluded; see the per-workload
``ops`` comments). The proxy is identical across builds, so the pooled-OFF vs pooled-ON
*ratio* is exact; absolute numbers are only comparable within a single workload,
not across workloads. Also reports per-workload peak live Python allocation
(``tracemalloc``), measured in a SEPARATE untimed pass so allocation tracing never
perturbs the timing, plus a single whole-process peak RSS at the end (POSIX only).

Run the same command on a pooled-OFF and pooled-ON build and diff the output.

Usage:
    python bench/python/bench_vf_construction.py [--reps N] [--inner M] [--json]

``--reps`` = timed repetitions per workload (min wall-time is reported, the
standard robust estimator). ``--inner`` scales the per-rep construction count.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc

try:
    import resource  # POSIX-only; RSS reporting is skipped without it (e.g. Windows)
except ImportError:
    resource = None

from tychopy import vector_functions as vf

Args = vf.Arguments


# --------------------------------------------------------------------------- #
# Workloads. Each returns a count of constructed-VectorFunction operations it
# performs per call (a stable per-build proxy for throughput — NOT every allocated
# object; leaf Arguments/Segment constructions are excluded for the reasons noted
# per workload). Because the proxy is identical pooled-OFF and pooled-ON, it
# cancels in the ratio; only within-workload comparisons are meaningful.
# --------------------------------------------------------------------------- #
def w_synthetic_chain(inner: int) -> int:
    """Tight loop of ephemeral wrappers: the pooling sweet spot.

    Each iteration constructs and immediately discards several GenericFunctions.
    """
    X = Args(6)
    R = X.head(3)
    V = X.tail(3)
    ops = 0
    for _ in range(inner):
        a = R.normalized()  # normalized: 1
        b = (a + R).norm()  # add: 1, norm: 1
        c = R.cross(V).normalized()  # cross: 1, normalized: 1
        del a, b, c
        ops += 5  # 1 + (1 + 1) + (1 + 1)
    return ops


def w_rtn_tree(inner: int) -> int:
    """Realistic expression rebuild: the RTN basis tree from BettsLowThrust.

    Mirrors examples/python_examples/BettsLowThrustCentralShooting.py:77-81.
    Rebuilt ``inner`` times — models continuation/sweep loops that reconstruct
    dynamics expressions repeatedly.
    """
    ops = 0
    for _ in range(inner):
        R, V = Args(6).tolist([(0, 3), (3, 3)])
        Rhat = R.normalized()
        Nhat = R.cross(V).normalized()
        That = Nhat.cross(R).normalized()
        s = vf.stack(Rhat, That, Nhat)
        del R, V, Rhat, Nhat, That, s
        # 2 segments + 1 normalized (Rhat) + 2*(cross + normalized) + stack = 8.
        # The Args(6) leaf is not counted (see module-level note).
        ops += 8
    return ops


def w_deep_sum(inner: int) -> int:
    """Wide expression: accumulate a long sum of scalar terms in one tree.

    Stresses many GenericFunction allocations within a single construction,
    rather than many independent small constructions.
    """
    ops = 0
    for _ in range(inner):
        X = Args(8)
        elems = X.tolist()
        acc = elems[0] * elems[0]
        for e in elems[1:]:
            acc = acc + e * e  # 2 ops per term: square (e*e) + add
        r = vf.sqrt(acc)  # scalar -> scalar
        del X, elems, acc, r
        # square(1) + (n-1) terms * 2 + sqrt(1). The Args(8) leaf and the n
        # tolist() elements are not counted (see module-level note).
        n = 8
        ops += 1 + 2 * (n - 1) + 1
    return ops


WORKLOADS = {
    "synthetic_chain": w_synthetic_chain,
    "rtn_tree": w_rtn_tree,
    "deep_sum": w_deep_sum,
}


def run_one(fn, inner: int, reps: int) -> dict:
    # Warm up (let import-time / first-touch costs settle, fill the pool).
    fn(min(inner, 64))
    gc.collect()

    # Timing pass: tracemalloc OFF. Allocation tracing adds ~20% per-allocation
    # overhead that scales with allocation count — exactly the quantity pooling
    # reduces — so it must not run during timing or it attenuates the delta.
    best = float("inf")
    ops_per_call = 0
    for _ in range(reps):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            ops_per_call = fn(inner)
            t1 = time.perf_counter()
        finally:
            gc.enable()  # survive a workload exception without leaving GC off
        best = min(best, t1 - t0)

    # Memory pass: separate, untimed, with tracemalloc ON. peak_py is the peak
    # *simultaneously live* Python allocation — a per-workload neutrality check.
    tracemalloc.start()
    try:
        fn(inner)
        _, peak_py = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    objs = ops_per_call
    return {
        "best_s": best,
        "objs": objs,
        "objs_per_s": objs / best if best > 0 else 0.0,
        "ns_per_obj": (best / objs * 1e9) if objs else 0.0,
        "peak_py_bytes": peak_py,
    }


def peak_rss_mb() -> float | None:
    """Whole-process peak RSS in MB, or None if unavailable (non-POSIX).

    ``ru_maxrss`` is in kilobytes on Linux but bytes on macOS/BSD; normalize.
    This is a process-lifetime high-water mark, not per-workload.
    """
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # bytes vs KiB
    return rss * scale / 1e6


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--inner", type=int, default=20000)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--filter", default="")
    args = ap.parse_args(argv)

    if args.reps < 1 or args.inner < 1:
        print("error: --reps and --inner must be >= 1", file=sys.stderr)
        return 2

    results = {}
    for name, fn in WORKLOADS.items():
        if args.filter and args.filter not in name:
            continue
        results[name] = run_one(fn, args.inner, args.reps)

    if not results:
        print(
            f"error: --filter '{args.filter}' matched no workload; "
            f"available: {', '.join(WORKLOADS)}",
            file=sys.stderr,
        )
        return 2

    proc_rss = peak_rss_mb()
    if args.json:
        print(
            json.dumps(
                {
                    "inner": args.inner,
                    "reps": args.reps,
                    "proc_peak_rss_mb": proc_rss,
                    "results": results,
                }
            )
        )
        return 0

    print(f"VF construction benchmark  (inner={args.inner}, reps={args.reps})")
    print(f"{'workload':<18} {'ops/s':>14} {'ns/op':>10} {'peak_py_MB':>11}")
    for name, r in results.items():
        print(
            f"{name:<18} {r['objs_per_s']:>14,.0f} {r['ns_per_obj']:>10.1f} "
            f"{r['peak_py_bytes'] / 1e6:>11.3f}"
        )
    rss_str = "n/a (non-POSIX)" if proc_rss is None else f"{proc_rss:.1f} MB"
    print(f"\nwhole-process peak RSS: {rss_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
