import ast
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("code_rules_baseline.json")
SCAN_DIRS = (ROOT / "starwinds_analysis", ROOT / "examples")


def _iter_targets():
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in {".py", ".ipynb"}:
                continue
            yield path


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


class _RuleVisitor(ast.NodeVisitor):
    def __init__(self):
        self.findings = []
        self.numpy_module_aliases = {"np", "numpy"}
        self.numpy_array_names = set()
        self.numpy_asarray_names = set()

    def _add(self, rule, lineno, detail=""):
        self.findings.append({"rule": rule, "line": int(lineno), "detail": detail})

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "numpy":
                self.numpy_module_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "numpy":
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == "array":
                    self.numpy_array_names.add(local)
                if alias.name == "asarray":
                    self.numpy_asarray_names.add(local)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name.startswith("_ensure"):
            self._add("BAN_ENSURE_FUNCTION", node.lineno, node.name)
        if node.name.startswith("resolve_"):
            self._add("BAN_RESOLVE_FUNCTION", node.lineno, node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Call(self, node):
        func_name = _dotted_name(node.func)
        is_np_array = False
        is_np_asarray = False

        if func_name is not None:
            if func_name in self.numpy_asarray_names or func_name in {"numpy.asarray"}:
                is_np_asarray = True
            elif func_name in self.numpy_array_names or func_name in {"numpy.array"}:
                is_np_array = True
            elif "." in func_name:
                head, tail = func_name.rsplit(".", 1)
                if head in self.numpy_module_aliases and tail == "asarray":
                    is_np_asarray = True
                if head in self.numpy_module_aliases and tail == "array":
                    is_np_array = True

        if is_np_asarray:
            self._add("BAN_NP_ASARRAY", node.lineno)

        if is_np_array:
            for kw in node.keywords:
                if kw.arg != "dtype":
                    continue
                if isinstance(kw.value, ast.Name) and kw.value.id == "float":
                    self._add("BAN_NP_ARRAY_DTYPE_FLOAT", node.lineno)

        self.generic_visit(node)


def _scan_python_source(source: str):
    tree = ast.parse(source)
    visitor = _RuleVisitor()
    visitor.visit(tree)
    findings = list(visitor.findings)

    for idx, line in enumerate(source.splitlines(), start=1):
        if "print('Using:', DATA_FILE)" in line or 'print("Using:", DATA_FILE)' in line:
            findings.append({"rule": "BAN_USING_DATAFILE_PRINT", "line": idx, "detail": "print('Using:', DATA_FILE)"})

    return findings


def _scan_py_file(path: Path):
    return _scan_python_source(path.read_text())


def _scan_notebook(path: Path):
    data = json.loads(path.read_text())
    findings = []
    for cell_index, cell in enumerate(data.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            source = "".join(src)
        else:
            source = str(src)
        if not source.strip():
            continue
        try:
            cell_findings = _scan_python_source(source)
            for f in cell_findings:
                findings.append(
                    {
                        "rule": f["rule"],
                        "line": f"cell {cell_index}:{f['line']}",
                        "detail": f.get("detail", ""),
                    }
                )
        except SyntaxError as exc:
            findings.append(
                {
                    "rule": "NOTEBOOK_SYNTAX_ERROR",
                    "line": f"cell {cell_index}:{exc.lineno or 1}",
                    "detail": (exc.msg or "syntax error"),
                }
            )
    return findings


def collect_findings():
    out = []
    for path in _iter_targets():
        if path.suffix == ".py":
            findings = _scan_py_file(path)
        else:
            findings = _scan_notebook(path)
        for f in findings:
            out.append(
                {
                    "file": _rel(path),
                    "rule": f["rule"],
                    "line": f["line"],
                    "detail": f.get("detail", ""),
                }
            )
    out.sort(key=lambda x: (x["file"], str(x["line"]), x["rule"], x.get("detail", "")))
    return out


def _load_baseline():
    if not BASELINE.exists():
        return []
    return json.loads(BASELINE.read_text())


def _write_baseline(findings):
    BASELINE.write_text(json.dumps(findings, indent=2) + "\n")


def test_code_rules_baseline():
    findings = collect_findings()

    if os.getenv("UPDATE_CODE_RULES_BASELINE") == "1":
        _write_baseline(findings)
        pytest.skip("Updated code rules baseline")

    baseline = _load_baseline()

    if findings != baseline:
        baseline_set = {json.dumps(x, sort_keys=True) for x in baseline}
        findings_set = {json.dumps(x, sort_keys=True) for x in findings}
        new = sorted(json.loads(x) for x in (findings_set - baseline_set))
        gone = sorted(json.loads(x) for x in (baseline_set - findings_set))

        lines = ["Code rules baseline mismatch."]
        if new:
            lines.append("")
            lines.append(f"New violations ({len(new)}):")
            for item in new[:25]:
                lines.append(f"- {item['file']}:{item['line']} {item['rule']} {item.get('detail', '')}".rstrip())
            if len(new) > 25:
                lines.append(f"- ... and {len(new) - 25} more")
        if gone:
            lines.append("")
            lines.append(f"Resolved violations not yet removed from baseline ({len(gone)}):")
            for item in gone[:25]:
                lines.append(f"- {item['file']}:{item['line']} {item['rule']} {item.get('detail', '')}".rstrip())
            if len(gone) > 25:
                lines.append(f"- ... and {len(gone) - 25} more")
        lines.append("")
        lines.append("To refresh baseline intentionally:")
        lines.append("UPDATE_CODE_RULES_BASELINE=1 pytest -q test/test_code_rules.py")
        pytest.fail("\n".join(lines))
