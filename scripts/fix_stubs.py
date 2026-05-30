#!/usr/bin/env python3
"""Post-process nanobind stubgen output: drop bogus self-imports, strip
self-qualifiers, ensure typing names are imported. Idempotent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Submodules whose names nanobind stubgen may leak as bare qualifiers.
# Also the precondition list of expected snapshot files (main() asserts
# every entry exists after the stubgen step).
SUBMODULES = (
    "vector_functions",
    "optimal_control",
    "solvers",
    "astro",
    "utils",
    "extensions",
)

# nanobind's variant caster emits `Union[...]` directly into signature
# strings without the `typing.` prefix, so the stubgen-generated import line
# may miss it. Same risk for other typing names referenced from signatures.
TYPING_NAMES = ("Union", "Optional", "Any", "Callable", "Tuple", "List", "Dict")

# Triple-quoted string content is stripped before searching the body for
# typing-name tokens, so a docstring like `"""Pass Any value"""` does not
# trigger a spurious `from typing import Any` insertion (which would break
# the idempotency guarantee asserted by stubs-snapshot-sanity.yml).
_TRIPLE_QUOTE_RE = re.compile(r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')')


def _strip_string_literals(text: str) -> str:
    return _TRIPLE_QUOTE_RE.sub("", text)


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
        # Fix 2: strip `<own_submodule>.` qualifier from type references.
        # Lookahead requires a capitalized identifier so docstring text like
        # `"optimal_control.foo()"` is not mangled.
        text = re.sub(
            rf"(?<![\w.]){re.escape(own_submodule)}\.(?=[A-Z_])",
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
    """Return typing names referenced in `text` but not imported."""
    imported: set[str] = set()
    for m in re.finditer(r"^from typing import\s+(.+)$", text, flags=re.MULTILINE):
        for token in m.group(1).split(","):
            imported.add(token.strip())

    # Exclude import lines and triple-quoted strings so a typing name that
    # appears only in a docstring does not count as a body reference.
    body = "\n".join(
        ln for ln in text.splitlines() if not re.match(r"^from typing import\s", ln)
    )
    body = _strip_string_literals(body)

    missing: list[str] = []
    for name in TYPING_NAMES:
        if name in imported:
            continue
        if re.search(rf"\b{name}\b", body):
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

    # Precondition: the expected stub set must be present. A missing file
    # indicates the snapshot regen failed silently (e.g. _tychopy
    # ImportError during nanobind stubgen produced an empty output dir).
    expected = [root / "_tychopy.pyi"] + [
        root / "_tychopy" / f"{s}.pyi" for s in SUBMODULES
    ]
    missing = [p for p in expected if not p.is_file()]
    if missing:
        for p in missing:
            print(f"error: expected stub file is missing: {p}", file=sys.stderr)
        return 1

    changed_any = False
    top = root / "_tychopy.pyi"
    if fix_file(top, own_submodule=None):
        changed_any = True
        print(f"fix_stubs.py: rewrote {top.relative_to(root)}")

    for submod in SUBMODULES:
        pyi = root / "_tychopy" / f"{submod}.pyi"
        if fix_file(pyi, own_submodule=submod):
            changed_any = True
            print(f"fix_stubs.py: rewrote {pyi.relative_to(root)}")

    if not changed_any:
        print("fix_stubs.py: no changes (already clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
