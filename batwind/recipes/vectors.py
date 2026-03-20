from __future__ import annotations

from collections.abc import Sequence
import logging
import re

import griblet
import numpy as np

log = logging.getLogger(__name__)

def build_vector_graph(variable_names: set[str] | Sequence[str]):
    variable_names = tuple(variable_names)
    log.info("build_vector_graph...")
    graph = griblet.ComputationGraph()

    by_prefix: dict[tuple[str, str], set[str]] = {}
    for prefix, comp, unit in _available_xyz_components(variable_names):
        by_prefix.setdefault((prefix, unit), set()).add(comp)

    n_vectors = 0
    detected_vectors: list[str] = []
    for (prefix, unit), comps in sorted(by_prefix.items()):
        if comps != {"x", "y", "z"}:
            continue
        n_vectors += 1
        detected_vectors.append(f"{prefix} [{unit}]")
        deps = [f"{prefix}_x [{unit}]", f"{prefix}_y [{unit}]", f"{prefix}_z [{unit}]"]
        graph.add_recipe(
            f"{prefix}_xyz [{unit}]",
            lambda x, y, z: np.stack([np.array(x), np.array(y), np.array(z)], axis=-1),
            deps=deps,
            cost=0.05,
            metadata={"description": f"{prefix} Cartesian vector"},
        )
        graph.add_recipe(
            f"{prefix} [{unit}]",
            lambda x, y, z: np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2),
            deps=deps,
            cost=0.1,
            metadata={"description": f"{prefix} magnitude"},
        )
    log.debug(
        "build_vector_graph vectors=%d detected=%s fields=%d",
        n_vectors,
        tuple(detected_vectors),
        len(tuple(graph.list_fields())),
    )
    return graph


def _parse_var_name(name: str):
    m = re.match(r"^(?P<base>.+?) \[(?P<unit>.+)\]$", name)
    if m:
        return m.group("base"), m.group("unit")

    if " " in name and "[" not in name and "]" not in name:
        base, unit = name.rsplit(" ", 1)
        if "/" in unit or unit.isalpha() or any(ch.isdigit() for ch in unit):
            return base, unit
    return None


def _parse_xyz_component_name(name: str):
    parsed = _parse_var_name(name)
    if parsed is None:
        return None
    base, unit = parsed
    if "_" not in base:
        return None
    prefix, comp = base.rsplit("_", 1)
    if comp not in ("x", "y", "z") or not prefix:
        return None
    return prefix, comp, unit


def _available_xyz_components(variable_names: set[str] | Sequence[str]):
    seen: set[tuple[str, str, str]] = set()
    for name in variable_names:
        parsed = _parse_xyz_component_name(name)
        if parsed is None:
            continue
        seen.add(parsed)
    return seen


__all__ = ["build_vector_graph"]
