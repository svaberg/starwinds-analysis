import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = ROOT / "starwinds_analysis"


def _iter_python_files():
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        yield path


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _iter_function_nodes(tree: ast.AST):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []
            self.items = []

        def generic_visit(self, node):
            self.stack.append(node)
            super().generic_visit(node)
            self.stack.pop()

        def _record(self, node):
            parents = tuple(self.stack)
            self.items.append((node, parents))

        def visit_FunctionDef(self, node):
            self._record(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self._record(node)
            self.generic_visit(node)

    v = Visitor()
    v.visit(tree)
    return v.items


def _qualname(node, parents):
    parts = []
    for p in parents:
        if isinstance(p, ast.ClassDef):
            parts.append(p.name)
        elif isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(p.name)
    parts.append(node.name)
    return ".".join(parts)


def _check_docstring(node):
    doc = ast.get_docstring(node)
    if not doc:
        return "missing docstring"

    lines = [line.strip() for line in doc.splitlines() if line.strip()]
    if len(lines) < 2:
        return "docstring should have at least 2 non-empty lines (what it does + used by)"

    has_used_by = any(line.startswith("Used by:") or line.startswith("Used in:") for line in lines)
    if not has_used_by:
        return "docstring missing 'Used by:' (or 'Used in:') line"

    # Require at least one non-'Used by' descriptive line.
    has_description = any(not (line.startswith("Used by:") or line.startswith("Used in:")) for line in lines)
    if not has_description:
        return "docstring missing descriptive line"

    return None


def test_all_functions_have_short_docstrings_with_usage():
    violations = []
    for path in _iter_python_files():
        source = path.read_text()
        tree = ast.parse(source)
        for node, parents in _iter_function_nodes(tree):
            problem = _check_docstring(node)
            if problem is None:
                continue
            violations.append(
                {
                    "file": _rel(path),
                    "line": node.lineno,
                    "name": _qualname(node, parents),
                    "problem": problem,
                }
            )

    if violations:
        lines = [
            f"Found {len(violations)} functions/methods without the required docstring format.",
            "",
            "Each function/method should have a short docstring with:",
            "- what it does",
            "- a 'Used by:' (or 'Used in:') line",
            "",
            "First violations:",
        ]
        for item in violations[:80]:
            lines.append(
                f"- {item['file']}:{item['line']} {item['name']}: {item['problem']}"
            )
        if len(violations) > 80:
            lines.append(f"- ... and {len(violations) - 80} more")
        pytest.fail("\n".join(lines))
