#!/usr/bin/env python3
"""VectorFunction construction microbenchmark.

Measures the Python-side cost of constructing the type-erased ``GenericFunction``
wrappers that nearly every VectorFunction operation returns by value
(``f + g``, ``.normalized()``, ``.norm()``, ``vf.stack(...)``, ...). This is the
workload that nanobind 2.13.0's ``nb::pooled()`` instance recycling targets — the
C++ ``bench/cpp`` suite does not exercise the binding layer, so this is the
Python-path "ruler" for that change.

Reports, per workload: objects/sec (throughput), peak Python allocation
(``tracemalloc``) and peak process RSS (``getrusage``). Run the same command on a
pooled-OFF and pooled-ON build and diff the output.

Usage:
    python bench/python/bench_vf_construction.py [--reps N] [--inner M] [--json]

``--reps`` = timed repetitions per workload (min wall-time is reported, the
standard robust estimator). ``--inner`` scales the per-rep construction count.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
import tracemalloc

from tychopy import vector_functions as vf

Args = vf.Arguments


# --------------------------------------------------------------------------- #
# Workloads. Each returns the number of GenericFunction-producing operations it
# performs per call, so throughput can be reported in constructed-objects/sec.
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
        a = R.normalized()          # 1
        b = (a + R).norm()          # 2 (add) + 1 (norm) = ops below
        c = R.cross(V).normalized() # cross + normalized
        del a, b, c
        ops += 5
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
        ops += 8  # 2 segments + 3 cross/normalized pairs-ish + stack
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
            acc = acc + e * e        # 2 ops per term
        r = vf.sqrt(acc)             # scalar -> scalar
        del X, elems, acc, r
        ops += 1 + 2 * 7 + 1
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

    best = float("inf")
    ops_per_call = 0
    tracemalloc.start()
    for _ in range(reps):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        ops_per_call = fn(inner)
        t1 = time.perf_counter()
        gc.enable()
        best = min(best, t1 - t0)
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    objs = ops_per_call
    return {
        "best_s": best,
        "objs": objs,
        "objs_per_s": objs / best if best > 0 else 0.0,
        "ns_per_obj": (best / objs * 1e9) if objs else 0.0,
        "peak_py_bytes": peak_py,
        "peak_rss_kb": peak_rss_kb,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--inner", type=int, default=20000)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--filter", default="")
    args = ap.parse_args(argv)

    results = {}
    for name, fn in WORKLOADS.items():
        if args.filter and args.filter not in name:
            continue
        results[name] = run_one(fn, args.inner, args.reps)

    if args.json:
        print(json.dumps({"inner": args.inner, "reps": args.reps, "results": results}))
        return 0

    print(f"VF construction benchmark  (inner={args.inner}, reps={args.reps})")
    print(f"{'workload':<18} {'objs/s':>14} {'ns/obj':>10} "
          f"{'peak_py_MB':>11} {'peak_rss_MB':>12}")
    for name, r in results.items():
        print(f"{name:<18} {r['objs_per_s']:>14,.0f} {r['ns_per_obj']:>10.1f} "
              f"{r['peak_py_bytes'] / 1e6:>11.2f} {r['peak_rss_kb'] / 1024:>12.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
