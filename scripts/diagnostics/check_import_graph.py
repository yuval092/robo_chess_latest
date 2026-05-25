"""Check that production src modules do not import training or scripts modules."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_PREFIXES = ("training", "scripts")


def _import_names(node: ast.AST) -> list[str]:
    """Return top-level imported module names for an import AST node."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


def violations() -> list[tuple[Path, int, str]]:
    """Return forbidden imports found under src/."""
    found: list[tuple[Path, int, str]] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            for name in _import_names(node):
                if name.split(".")[0] in FORBIDDEN_PREFIXES:
                    found.append((path, getattr(node, "lineno", 0), name))
    return found


def main() -> None:
    """Exit non-zero if any forbidden production imports are found."""
    found = violations()
    if found:
        for path, line, name in found:
            print(f"{path}:{line}: forbidden import {name}")
        raise SystemExit(1)
    print("Import graph OK")


if __name__ == "__main__":
    main()
