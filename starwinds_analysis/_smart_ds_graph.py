"""THIS FILE contains SmartDs <-> griblet graph integration internals.

It owns graph attachment, path resolution, and evaluation glue for SmartDs.
It should not contain domain physics formulas or plotting code.
"""

from __future__ import annotations

import importlib


def graph_field_names(smart_ds):
    graph = smart_ds._computation_graph
    if graph is None or not hasattr(graph, "fields"):
        return ()
    return tuple(graph.fields())


def resolve_field(smart_ds, name: str):
    """
    Resolve a field through the attached griblet computation graph.

    Returns ``(cost, tree)`` from ``griblet.DependencySolver.resolve_field``.
    """
    # TODO smartds-resolve:
    # This is graph-path resolution, not field/unit resolution. Keep the distinction
    # explicit if SmartDs grows a user-facing resolve() API that returns data + unit.
    graph = build_runtime_graph(smart_ds)
    griblet = import_griblet()
    solver = griblet.DependencySolver(graph)
    return solver.resolve_field(name)


def explain_field(smart_ds, name: str, *, return_tree: bool = False):
    cost, tree = resolve_field(smart_ds, name)
    if return_tree:
        return cost, tree

    lines: list[str] = []

    def walk(node, depth=0):
        meta = getattr(node, "recipe_metadata", {}) or {}
        desc = meta.get("description", "")
        planned = getattr(node, "cost", None)
        parts = [node.field]
        if planned is not None:
            parts.append(f"(cost={planned})")
        if desc:
            parts.append(f"- {desc}")
        lines.append("  " * depth + " ".join(parts))
        for dep in getattr(node, "deps", []):
            walk(dep, depth + 1)

    walk(tree)
    header = f"{name} total_cost={cost}"
    return "\n".join([header, *lines])


def compute_via_graph(smart_ds, name: str):
    if smart_ds._computation_graph is None:
        raise IndexError(
            f"Field '{name}' not available. Raw fields: {smart_ds._dataset.variables}. "
            f"Computed fields: {list(smart_ds._field_functions)}."
        )

    graph = build_runtime_graph(smart_ds)
    griblet = import_griblet()
    solver = griblet.DependencySolver(graph)
    _cost, tree = solver.resolve_field(name)
    return evaluate_resolved_tree(tree, graph)


def build_runtime_graph(smart_ds):
    if smart_ds._computation_graph is None:
        raise RuntimeError("No computation graph attached")
    griblet = import_griblet()
    runtime_graph = griblet.ComputationGraph()
    runtime_graph.merge(build_loader_graph(smart_ds))
    runtime_graph.merge(smart_ds._computation_graph)
    return runtime_graph


def build_loader_graph(smart_ds):
    """
    Build a zero-dependency graph exposing raw dataset variables (+ selected aux).
    """
    griblet = import_griblet()
    graph = griblet.ComputationGraph()

    for raw_name in smart_ds._dataset.variables:
        graph.add_recipe(
            field=raw_name,
            func=lambda raw_name=raw_name: smart_ds._dataset.variable(raw_name),
            deps=[],
            cost=0.0,
            metadata={"description": "Dataset raw field"},
        )

    for alias_name, candidates in smart_ds._aliases.items():
        if alias_name in smart_ds._dataset.variables:
            continue
        raw_name = next((c for c in candidates if c in smart_ds._dataset.variables), None)
        if raw_name is None:
            continue
        graph.add_recipe(
            field=alias_name,
            func=lambda raw_name=raw_name: smart_ds._dataset.variable(raw_name),
            deps=[],
            cost=0.0,
            metadata={"description": f"Alias for {raw_name}"},
        )

    if smart_ds._include_aux_in_loader:
        for key, value in smart_ds._dataset.aux.items():
            graph.add_recipe(
                field=key,
                func=lambda value=value: value,
                deps=[],
                cost=0.0,
                metadata={"description": "Dataset aux"},
            )
    return graph


def import_griblet():
    try:
        return importlib.import_module("griblet")
    except ImportError as e:
        raise ImportError(
            "griblet is required for computation-graph resolution. Install griblet "
            "or use local register_field(...) functions."
        ) from e


def evaluate_resolved_tree(node, graph):
    """
    Evaluate a griblet-resolved computation tree.

    Uses tolerant dependency matching (tuple/list) to stay compatible with the
    current griblet recipe storage/evaluator behavior.
    """
    if getattr(node, "used_primary", False):
        for recipe in graph.recipes[node.field]:
            if len(recipe["deps"]) == 0:
                return recipe["func"]()
        raise RuntimeError(f"No zero-dependency recipe for {node.field}")

    values = [evaluate_resolved_tree(dep, graph) for dep in node.deps]
    dep_fields = tuple(dep.field for dep in node.deps)
    for recipe in graph.recipes[node.field]:
        if tuple(recipe["deps"]) == dep_fields:
            return recipe["func"](*values)
    raise RuntimeError(f"No matching recipe for {node.field} with deps={dep_fields}")


__all__ = [
    "build_loader_graph",
    "build_runtime_graph",
    "compute_via_graph",
    "evaluate_resolved_tree",
    "explain_field",
    "graph_field_names",
    "import_griblet",
    "resolve_field",
]
