from __future__ import annotations

from collections.abc import Sequence
import re

import griblet
import numpy as np


_UNIT_FACTORS = {
    "g/cm^3": ("kg/m^3", 1e3),
    "amu/cm^3": ("kg/m^3", 1.66053906660e-27 * 1e6),
    "km/s": ("m/s", 1e3),
    "Gauss": ("T", 1e-4),
    "G": ("T", 1e-4),
    "nT": ("T", 1e-9),
    "erg/cm^3": ("J/m^3", 1e-1),
    "dyne/cm^2": ("Pa", 1e-1),
    "nPa": ("Pa", 1e-9),
    "`mA/m^2": ("A/m^2", 1e-6),
}


def build_vector_graph(variable_names: set[str] | Sequence[str]):
    graph = griblet.ComputationGraph()

    by_prefix: dict[tuple[str, str], set[str]] = {}
    for prefix, comp, unit in _available_xyz_components(variable_names):
        by_prefix.setdefault((prefix, unit), set()).add(comp)

    for (prefix, unit), comps in sorted(by_prefix.items()):
        if comps != {"x", "y", "z"}:
            continue
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
        prefix, comp, unit = parsed
        seen.add((prefix, comp, unit))
        unit_match = _UNIT_FACTORS.get(unit)
        if unit_match is not None:
            si_unit, _factor = unit_match
            seen.add((prefix, comp, si_unit))
    return seen


__all__ = ["build_vector_graph"]
