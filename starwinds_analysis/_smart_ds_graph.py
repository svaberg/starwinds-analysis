"""THIS FILE contains SmartDs <-> griblet graph integration internals.

It owns graph attachment, path resolution, and evaluation glue for SmartDs.
It should not contain domain physics formulas or plotting code.
"""

from __future__ import annotations

import griblet

def graph_field_names(smart_ds):
    """
    List available field names from the runtime griblet graph.
    Used by: no external call sites found
    """
    graph = smart_ds._computation_graph
    if graph is None or not hasattr(graph, "fields"):
        return ()
    return tuple(graph.fields())

def resolve_field(smart_ds, name: str):
    """
    Resolve a field through the attached griblet computation graph.
    Used by: `starwinds_analysis/_smart_ds_graph.py`
    """
    # TODO smartds-resolve:
    # This is graph-path resolution, not field/unit resolution. Keep the distinction
    # explicit if SmartDs grows a user-facing resolve() API that returns data + unit.
    graph = build_runtime_graph(smart_ds)
    solver = griblet.DependencySolver(graph)
    return solver.resolve_field(name)

def explain_field(smart_ds, name: str, *, return_tree: bool = False):
    """
    Build a human-readable explanation of the chosen graph path.
    Used by: no external call sites found
    """
    cost, tree = resolve_field(smart_ds, name)
    if return_tree:
        return cost, tree

    lines: list[str] = []

    def walk(node, depth=0):
        """
        Walk the resolved dependency tree recursively to render an explanation string.
        Used by: `explain_field` (nested helper)
        """
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
    """
    Compute a field by resolving + evaluating a griblet graph path.
    Used by: no external call sites found
    """
    if smart_ds._computation_graph is None:
        raise IndexError(
            f"Field '{name}' not available. Raw fields: {smart_ds._dataset.variables}. "
            f"Computed fields: {list(smart_ds._field_functions)}."
        )

    graph = build_runtime_graph(smart_ds)
    solver = griblet.DependencySolver(graph)
    _cost, tree = solver.resolve_field(name)
    return evaluate_resolved_tree(tree, graph)

def build_runtime_graph(smart_ds):
    """
    Merge loader graph and user graph into a runtime graph for one SmartDs instance.
    Used by: `starwinds_analysis/_smart_ds_graph.py`
    """
    if smart_ds._computation_graph is None:
        raise RuntimeError("No computation graph attached")
    runtime_graph = griblet.ComputationGraph()
    loader_graph = build_loader_graph(smart_ds)
    runtime_graph.merge(loader_graph)
    loader_fields = set(loader_graph.fields())
    for field, recipes in smart_ds._computation_graph.recipes.items():
        if field in loader_fields:
            continue
        runtime_graph.recipes.setdefault(field, []).extend(recipes)
    return runtime_graph

def build_loader_graph(smart_ds):
    """
    Build a zero-dependency graph exposing raw dataset variables (+ selected aux).
    Used by: `starwinds_analysis/_smart_ds_graph.py`
    """
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


def evaluate_resolved_tree(node, graph):
    """
    Evaluate a griblet-resolved computation tree.
    Used by: `starwinds_analysis/_smart_ds_graph.py`
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
