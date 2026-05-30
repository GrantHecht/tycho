#!/usr/bin/env python3
"""Inspect a tycho wheel against layout invariants.

Run as ``python scripts/inspect_wheel_layout.py <wheel-path>``.

Assertions:

* Negative: no entries start with any prefix derived from
  ``[tool.scikit-build].wheel.exclude`` in ``pyproject.toml``. These are
  CMake C++ distribution prefixes (for ``find_package(tycho)`` consumers
  via ``cmake --install``) that must not leak into the Python wheel.
* Positive: exactly one ``_tychopy.cpython-*.so`` at the wheel root
  (PEP 561 layout next to the stubs).

Exit 0 on success, 1 on any assertion failure. ``::error::`` lines are
emitted in GitHub Actions workflow-command format for surfacing in PR
annotations.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import tomllib

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _load_bad_prefixes() -> tuple[str, ...]:
    with _PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    excludes = data["tool"]["scikit-build"]["wheel"]["exclude"]
    return tuple(e.removesuffix("/**").rstrip("/") + "/" for e in excludes)


BAD_PREFIXES = _load_bad_prefixes()
SO_PATTERN = re.compile(r"^_tychopy\.cpython-[^/]+\.so$")


def inspect(wheel_path: Path) -> int:
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()

    print(f"Inspecting {wheel_path}")
    print(f"Wheel contents ({len(names)} entries):")
    for name in names:
        print(f"  {name}")

    fail = False

    for prefix in BAD_PREFIXES:
        leaks = [n for n in names if n.startswith(prefix)]
        if leaks:
            print(f"::error::wheel.exclude leak — entries start with {prefix!r}:")
            for n in leaks:
                print(f"  {n}")
            fail = True

    so_matches = [n for n in names if SO_PATTERN.match(n)]
    if not so_matches:
        print("::error::wheel missing _tychopy.cpython-*.so at root")
        fail = True
    elif len(so_matches) > 1:
        print(
            f"::error::wheel has multiple _tychopy.cpython-*.so entries: {so_matches}"
        )
        fail = True

    return 1 if fail else 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <wheel-path>", file=sys.stderr)
        return 2
    return inspect(Path(argv[1]))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
