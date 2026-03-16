"""SmartDs <-> griblet graph integration internals.
"""

# It owns graph attachment, path resolution, and evaluation glue for SmartDs.
# It should not contain domain physics formulas or plotting code.


from __future__ import annotations

import logging

import griblet

log = logging.getLogger(__name__)


def _graph_fields(graph) -> tuple[str, ...]:
    """Return field names across griblet API variants."""
    if hasattr(graph, "list_fields"):
        return tuple(graph.list_fields())
    if hasattr(graph, "fields"):
        return tuple(graph.fields())
    return ()

def graph_field_names(smart_ds):
    """
    List available field names from the runtime griblet graph.
    Used by: no external call sites found
    """
    graph = smart_ds._computation_graph
    if graph is None:
        log.debug("graph_field_names: no graph available")
        return ()
    out = _graph_fields(graph)
    log.debug("graph_field_names: n_fields=%d", len(out))
    return out

def graph_path(smart_ds, name: str):
    """
    Return the chosen griblet dependency path for one field.
    Used by: `batwind/_smart_ds_graph.py`
    """
    log.debug("graph_path resolving field '%s'", name)
    graph = build_runtime_graph(smart_ds)
    solver = griblet.DependencySolver(graph)
    return solver.resolve_field(name)

def explain_field(smart_ds, name: str, *, return_tree: bool = False):
    """
    Build a human-readable explanation of the chosen graph path.
    Used by: no external call sites found
    """
    cost, tree = graph_path(smart_ds, name)
    log.info("explain_field resolved '%s' total_cost=%s", name, cost)
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
        log.error("compute_via_graph failed: no computation graph attached for '%s'", name)
        raise IndexError(
            f"Field '{name}' not available. Raw fields: {smart_ds._dataset.variables}. "
            "No computation graph attached."
        )

    graph = build_runtime_graph(smart_ds)
    solver = griblet.DependencySolver(graph)
    _cost, tree = solver.resolve_field(name)
    log.debug("compute_via_graph resolved '%s' cost=%s", name, _cost)
    return evaluate_resolved_tree(tree, graph)

def build_runtime_graph(smart_ds):
    """
    Merge loader graph and user graph into a runtime graph for one SmartDs instance.
    Used by: `batwind/_smart_ds_graph.py`
    """
    if smart_ds._computation_graph is None:
        log.error("build_runtime_graph failed: no computation graph attached")
        raise RuntimeError("No computation graph attached")
    runtime_graph = griblet.ComputationGraph()
    loader_graph = build_loader_graph(smart_ds)
    runtime_graph.merge(loader_graph)
    loader_fields = set(_graph_fields(loader_graph))
    for field, recipes in smart_ds._computation_graph.recipes.items():
        if field in loader_fields:
            continue
        runtime_graph.recipes.setdefault(field, []).extend(recipes)
    log.debug(
        "build_runtime_graph done loader_fields=%d runtime_fields=%d",
        len(loader_fields),
        len(_graph_fields(runtime_graph)),
    )
    return runtime_graph

def build_loader_graph(smart_ds):
    """
    Build a zero-dependency graph exposing raw dataset variables (+ selected aux).
    Used by: `batwind/_smart_ds_graph.py`
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

    if smart_ds._include_aux_in_loader:
        for key, value in smart_ds._dataset.aux.items():
            graph.add_recipe(
                field=key,
                func=lambda value=value: value,
                deps=[],
                cost=0.0,
                metadata={"description": "Dataset aux"},
            )
    log.debug(
        "build_loader_graph done raw_fields=%d aux_fields=%d",
        len(smart_ds._dataset.variables),
        len(smart_ds._dataset.aux) if smart_ds._include_aux_in_loader else 0,
    )
    return graph


def evaluate_resolved_tree(node, graph):
    """
    Evaluate a griblet-resolved computation tree.
    Used by: `batwind/_smart_ds_graph.py`
    """
    if getattr(node, "used_primary", False):
        for recipe in graph.recipes[node.field]:
            if len(recipe["deps"]) == 0:
                return recipe["func"]()
        log.error("evaluate_resolved_tree failed: no zero-dependency recipe for %s", node.field)
        raise RuntimeError(f"No zero-dependency recipe for {node.field}")

    values = [evaluate_resolved_tree(dep, graph) for dep in node.deps]
    dep_fields = tuple(dep.field for dep in node.deps)
    for recipe in graph.recipes[node.field]:
        if tuple(recipe["deps"]) == dep_fields:
            return recipe["func"](*values)
    log.error("evaluate_resolved_tree failed: no matching recipe for %s deps=%s", node.field, dep_fields)
    raise RuntimeError(f"No matching recipe for {node.field} with deps={dep_fields}")
