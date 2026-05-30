"""Tests for scripts/inspect_wheel_layout.py.

Synthetic wheels are built with zipfile so the test does not depend on
a real tycho build. Each test exercises one assertion in the script.
Run from the repo root with ``pytest scripts/test_inspect_wheel_layout.py``.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import tomllib

SCRIPT = Path(__file__).parent / "inspect_wheel_layout.py"
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _make_wheel(tmp_path: Path, names: list[str]) -> Path:
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for name in names:
            zf.writestr(name, b"")
    return wheel


def _run(*args: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(a) for a in args)],
        capture_output=True,
        text=True,
    )


def test_clean_wheel_passes(tmp_path: Path) -> None:
    wheel = _make_wheel(
        tmp_path,
        [
            "_tychopy.cpython-313-x86_64-linux-gnu.so",
            "_tychopy.pyi",
            "_tychopy/astro.pyi",
            "_tychopy/optimal_control.pyi",
            "tychopy/__init__.py",
        ],
    )
    result = _run(wheel)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::error::" not in result.stdout


@pytest.mark.parametrize(
    "leak_entry,prefix",
    [
        ("include/header.h", "include/"),
        ("lib/libfoo.so", "lib/"),
        ("lib64/libfoo.so", "lib64/"),
        ("share/cmake/foo.cmake", "share/"),
        ("bin/program", "bin/"),
        ("tychopy/_stubs/_tychopy.pyi", "tychopy/_stubs/"),
    ],
)
def test_leak_detected(tmp_path: Path, leak_entry: str, prefix: str) -> None:
    wheel = _make_wheel(
        tmp_path,
        [
            "_tychopy.cpython-313-x86_64-linux-gnu.so",
            leak_entry,
        ],
    )
    result = _run(wheel)
    assert result.returncode == 1
    assert (
        f"::error::wheel.exclude leak — entries start with {prefix!r}" in result.stdout
    )


def test_missing_so_detected(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, ["_tychopy.pyi", "tychopy/__init__.py"])
    result = _run(wheel)
    assert result.returncode == 1
    assert "::error::wheel missing _tychopy.cpython-*.so at root" in result.stdout


def test_multiple_so_detected(tmp_path: Path) -> None:
    wheel = _make_wheel(
        tmp_path,
        [
            "_tychopy.cpython-313-x86_64-linux-gnu.so",
            "_tychopy.cpython-314-x86_64-linux-gnu.so",
        ],
    )
    result = _run(wheel)
    assert result.returncode == 1
    assert "::error::wheel has multiple _tychopy.cpython-*.so entries" in result.stdout


def test_combined_failures_reported(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, ["include/h.h", "lib/foo.so"])
    result = _run(wheel)
    assert result.returncode == 1
    assert (
        "::error::wheel.exclude leak — entries start with 'include/'" in result.stdout
    )
    assert "::error::wheel.exclude leak — entries start with 'lib/'" in result.stdout
    assert "::error::wheel missing _tychopy.cpython-*.so at root" in result.stdout


def test_so_not_at_root_is_missing(tmp_path: Path) -> None:
    # Extension nested under a subdir does NOT satisfy the PEP-561 root layout.
    wheel = _make_wheel(tmp_path, ["tychopy/_tychopy.cpython-313-x86_64-linux-gnu.so"])
    result = _run(wheel)
    assert result.returncode == 1
    assert "::error::wheel missing _tychopy.cpython-*.so at root" in result.stdout


def test_usage_error_no_args() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_bad_prefixes_matches_pyproject() -> None:
    # Guard against silent divergence between the inspector's BAD_PREFIXES
    # and pyproject.toml's wheel.exclude. The script derives the tuple at
    # import time; this test asserts the derivation gives the expected
    # shape (so a future "I'll just hand-edit it" change breaks loudly).
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import inspect_wheel_layout
    finally:
        sys.path.pop(0)
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    excludes = data["tool"]["scikit-build"]["wheel"]["exclude"]
    expected = tuple(e.removesuffix("/**").rstrip("/") + "/" for e in excludes)
    assert inspect_wheel_layout.BAD_PREFIXES == expected
