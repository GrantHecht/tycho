#!/usr/bin/env python3
"""Sanity-check stub-file sizes after `tychopy_stubs_snapshot` regen.

nanobind's stubgen exits 0 even when `_tychopy` fails to import at stubgen
time (missing MKL on LD path, libstdc++ ABI mismatch, etc.) — producing
near-empty `.pyi` files. The committed snapshot stays consistent with the
buggy environment, so `stubs-drift.yml` would also pass. This check fails
fast when any expected stub falls below a known-safe minimum byte count.

`_tychopy/extensions.pyi` is legitimately near-empty when no user-built
extension is present; its threshold is therefore 0.

Usage:
    python scripts/check_stubs_size.py <snapshot_dir>
"""

from __future__ import annotations

import pathlib
import sys

# Minimum byte counts. Tightly under the current real sizes — a regenerated
# stub that drops below these almost certainly indicates a stubgen failure.
THRESHOLDS = {
    "_tychopy.pyi": 100,
    "_tychopy/vector_functions.pyi": 1000,
    "_tychopy/optimal_control.pyi": 1000,
    "_tychopy/solvers.pyi": 1000,
    "_tychopy/astro.pyi": 1000,
    "_tychopy/utils.pyi": 100,
    "_tychopy/extensions.pyi": 0,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <snapshot_dir>", file=sys.stderr)
        return 2
    root = pathlib.Path(argv[1]).resolve()
    failures = []
    for name, threshold in THRESHOLDS.items():
        p = root / name
        if not p.is_file():
            failures.append(f"{p}: missing")
            continue
        size = p.stat().st_size
        if size < threshold:
            failures.append(f"{p}: {size} bytes < threshold {threshold}")
    if failures:
        print(
            "error: stub snapshot is incomplete — nanobind stubgen likely "
            "failed to import _tychopy",
            file=sys.stderr,
        )
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
