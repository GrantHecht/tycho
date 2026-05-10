#!/usr/bin/env python3
"""Post-process nanobind stubgen output for the _tychopy snapshot.

nanobind's std::variant type-name composition strips the package-root prefix
when an alternative's enclosing module is a sub-package of the file's own
module. For example, in `_tychopy/optimal_control.pyi`, references to
`_tychopy.optimal_control.ScaleModes` come out as `optimal_control.ScaleModes`
(the file's submodule name leaks as a bare qualifier) and stubgen helpfully
emits a corresponding `import optimal_control` line — pointing at a nonexistent
top-level module.

The principled fix would be per-method `nb::sig()` overrides on every binding
that takes such a variant arg (~35 sites for `auto_scale` alone), or upstream
patches to nanobind's variant caster. Both are heavy. This script applies a
small, targeted text-level fix to the committed snapshot so the published
stubs parse cleanly:

  1. Drop bogus `import <submodule>` lines where `<submodule>` is the file's
     own submodule (e.g. `import optimal_control` in `_tychopy/optimal_control.pyi`).
  2. Strip the `<submodule>.` prefix from in-file type references, e.g.
     `optimal_control.ScaleModes` -> `ScaleModes`.
  3. Add `Union` (and other typing names that appear unqualified in the
     stubgen output) to the `from typing import ...` line as needed.

Usage:
    python scripts/fix_stubs.py <snapshot_dir>

Where <snapshot_dir> contains `_tychopy.pyi` and `_tychopy/*.pyi`. Edits files
in-place. Safe to re-run (idempotent).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Submodules that nanobind stubgen may leak as bare qualifiers. Kept in sync
# with _TYCHOPY_SUBMODULES in src/bindings/CMakeLists.txt.
SUBMODULES = (
    "vector_functions",
    "optimal_control",
    "solvers",
    "astro",
    "utils",
    "extensions",
)

# Typing names that may appear in stubgen output without being imported.
# nanobind's variant caster emits `Union[...]` (without `typing.` prefix)
# directly into signature strings, so the stubgen-generated `from typing
# import ...` line may miss it.
TYPING_NAMES = ("Union", "Optional", "Any", "Callable", "Tuple", "List", "Dict")


def fix_file(path: Path, own_submodule: str | None) -> bool:
    """Apply the three fixes to `path`. Returns True if anything changed.

    `own_submodule` is the submodule the file represents (e.g. `optimal_control`
    for `_tychopy/optimal_control.pyi`), or None for the top-level `_tychopy.pyi`.
    """
    original = path.read_text()
    text = original

    if own_submodule is not None:
        # Fix 1: drop bogus `import <own_submodule>` lines.
        text = re.sub(
            rf"^import {re.escape(own_submodule)}\s*$\n?",
            "",
            text,
            flags=re.MULTILINE,
        )
        # Fix 2: strip `<own_submodule>.` qualifier from type references in
        # this file. Match only when preceded by a non-identifier character
        # (or start of line) so we don't mangle longer dotted names.
        text = re.sub(
            rf"(?<![\w.]){re.escape(own_submodule)}\.",
            "",
            text,
        )

    # Fix 3: ensure typing names that appear in the body are imported.
    needed = _find_missing_typing_imports(text)
    if needed:
        text = _add_typing_imports(text, needed)

    if text != original:
        path.write_text(text)
        return True
    return False


def _find_missing_typing_imports(text: str) -> list[str]:
    """Return typing names referenced in `text` but not present in any
    `from typing import ...` line at the top of the file."""
    # Collect the names already imported from typing.
    imported: set[str] = set()
    for m in re.finditer(
        r"^from typing import\s+(.+)$", text, flags=re.MULTILINE
    ):
        for token in m.group(1).split(","):
            imported.add(token.strip())

    missing: list[str] = []
    for name in TYPING_NAMES:
        if name in imported:
            continue
        # Word-boundary match: name followed by `[` or `(` or end-of-word.
        if re.search(rf"\b{name}\[", text):
            missing.append(name)
    return missing


def _add_typing_imports(text: str, names: list[str]) -> str:
    """Add `names` to an existing `from typing import ...` line, or insert a
    new one near the top of the file."""
    # Prefer extending an existing line.
    def _extend(match: re.Match[str]) -> str:
        existing = [t.strip() for t in match.group(1).split(",")]
        merged = sorted(set(existing) | set(names))
        return f"from typing import {', '.join(merged)}"

    new_text, n = re.subn(
        r"^from typing import\s+(.+)$",
        _extend,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n:
        return new_text

    # No existing typing import — insert one after the last top-of-file
    # import / from-import line.
    import_line = f"from typing import {', '.join(sorted(names))}\n"
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            insert_at = i + 1
    lines.insert(insert_at, import_line)
    return "".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <snapshot_dir>", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    changed_any = False
    top = root / "_tychopy.pyi"
    if top.is_file() and fix_file(top, own_submodule=None):
        changed_any = True
        print(f"fix_stubs.py: rewrote {top.relative_to(root)}")

    submods_dir = root / "_tychopy"
    if submods_dir.is_dir():
        for pyi in sorted(submods_dir.glob("*.pyi")):
            submod = pyi.stem
            if fix_file(pyi, own_submodule=submod):
                changed_any = True
                print(f"fix_stubs.py: rewrote {pyi.relative_to(root)}")

    if not changed_any:
        print("fix_stubs.py: no changes (already clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
