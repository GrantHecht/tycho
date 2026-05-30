"""Tests for scripts/fix_stubs.py.

Synthetic ``.pyi`` snapshots are built in ``tmp_path`` so the test does not
depend on a real ``_tychopy`` build. Each test invokes the script as a
subprocess (the same way it is called from ``tychopy_stubs_snapshot``) and
asserts both the exit code and the post-state of the temp snapshot dir.
Run from the repo root with ``pytest scripts/test_fix_stubs.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "fix_stubs.py"

SUBMODULES = (
    "vector_functions",
    "optimal_control",
    "solvers",
    "astro",
    "utils",
    "extensions",
)


def _make_snapshot(root: Path, files: dict[str, str]) -> None:
    """Create the expected stub set in ``root``; any file listed in
    ``files`` overrides its default near-empty content."""
    (root / "_tychopy").mkdir(parents=True, exist_ok=True)
    paths = {"_tychopy.pyi": root / "_tychopy.pyi"}
    for sub in SUBMODULES:
        paths[f"_tychopy/{sub}.pyi"] = root / "_tychopy" / f"{sub}.pyi"
    for rel, p in paths.items():
        p.write_text(files.get(rel, "# stub\n"))


def _run(snapshot_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(snapshot_dir)],
        capture_output=True,
        text=True,
    )


def test_clean_snapshot_idempotent(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, {})
    first = _run(tmp_path)
    assert first.returncode == 0, first.stderr
    snapshot1 = {p: p.read_text() for p in tmp_path.rglob("*.pyi")}
    second = _run(tmp_path)
    assert second.returncode == 0, second.stderr
    snapshot2 = {p: p.read_text() for p in tmp_path.rglob("*.pyi")}
    assert snapshot1 == snapshot2


def test_bogus_self_import_stripped(tmp_path: Path) -> None:
    _make_snapshot(
        tmp_path,
        {
            "_tychopy/optimal_control.pyi": (
                "import optimal_control\nclass Phase: ...\n"
            ),
        },
    )
    result = _run(tmp_path)
    assert result.returncode == 0
    body = (tmp_path / "_tychopy" / "optimal_control.pyi").read_text()
    assert "import optimal_control" not in body
    assert "class Phase: ..." in body


def test_bare_self_qualifier_stripped(tmp_path: Path) -> None:
    _make_snapshot(
        tmp_path,
        {
            "_tychopy/optimal_control.pyi": (
                "class Phase:\n    def mode(self) -> optimal_control.ScaleModes: ...\n"
            ),
        },
    )
    result = _run(tmp_path)
    assert result.returncode == 0
    body = (tmp_path / "_tychopy" / "optimal_control.pyi").read_text()
    assert "optimal_control.ScaleModes" not in body
    assert "-> ScaleModes" in body


def test_docstring_self_qualifier_preserved(tmp_path: Path) -> None:
    # The qualifier-strip rule must not mangle prose inside docstrings.
    _make_snapshot(
        tmp_path,
        {
            "_tychopy/optimal_control.pyi": (
                "def f() -> None:\n"
                '    """See optimal_control.foo() for details."""\n'
                "    ...\n"
            ),
        },
    )
    result = _run(tmp_path)
    assert result.returncode == 0
    body = (tmp_path / "_tychopy" / "optimal_control.pyi").read_text()
    assert "optimal_control.foo()" in body


def test_typing_import_added_when_used(tmp_path: Path) -> None:
    _make_snapshot(
        tmp_path,
        {
            "_tychopy/utils.pyi": ("def f(x: Union[int, str]) -> None: ...\n"),
        },
    )
    result = _run(tmp_path)
    assert result.returncode == 0
    body = (tmp_path / "_tychopy" / "utils.pyi").read_text()
    assert "from typing import" in body
    assert "Union" in body.split("\n", 1)[0] or "Union" in body


def test_typing_import_NOT_added_for_docstring_only(tmp_path: Path) -> None:
    # Regression: a typing-name token that appears ONLY inside a docstring
    # must not trigger an `import` insertion — otherwise fix_stubs.py would
    # not be idempotent against future stubs whose docstrings mention
    # `Any`/`Union`/etc. as prose.
    _make_snapshot(
        tmp_path,
        {
            "_tychopy/utils.pyi": (
                "def f(x: int) -> int:\n"
                '    """Pass Any value; returns it unchanged."""\n'
                "    ...\n"
            ),
        },
    )
    result = _run(tmp_path)
    assert result.returncode == 0
    body = (tmp_path / "_tychopy" / "utils.pyi").read_text()
    assert "from typing import" not in body


def test_typing_import_extends_existing_line(tmp_path: Path) -> None:
    _make_snapshot(
        tmp_path,
        {
            "_tychopy/utils.pyi": (
                "from typing import List\ndef f(x: Union[int, str]) -> List[int]: ...\n"
            ),
        },
    )
    result = _run(tmp_path)
    assert result.returncode == 0
    body = (tmp_path / "_tychopy" / "utils.pyi").read_text()
    # Single typing-import line, with both names sorted in.
    typing_lines = [
        ln for ln in body.splitlines() if ln.startswith("from typing import")
    ]
    assert len(typing_lines) == 1
    assert "List" in typing_lines[0]
    assert "Union" in typing_lines[0]


def test_precondition_fail_on_missing_files(tmp_path: Path) -> None:
    # Empty dir — every expected stub file is missing.
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "expected stub file is missing" in result.stderr
    # Each missing file is reported on its own line.
    assert result.stderr.count("expected stub file is missing") >= len(SUBMODULES) + 1


def test_usage_error_no_args() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_usage_error_not_a_directory(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "nope"
    result = _run(not_a_dir)
    assert result.returncode == 1
    assert "is not a directory" in result.stderr
