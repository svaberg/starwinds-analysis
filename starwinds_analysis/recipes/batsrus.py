"""THIS FILE contains BATSRUS-specific normalization and derived-field recipes.

It defines SI conversion recipes and BATSRUS derived quantities for SmartDs/griblet usage.
It should keep BATSRUS naming/unit conventions localized here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import math
import re

import numpy as np


_AMU_KG = 1.66053906660e-27
_MU0 = 4.0e-7 * math.pi
_DEFAULT_GAMMA = 5.0 / 3.0


_UNIT_FACTORS = {
    "g/cm^3": ("kg/m^3", 1e3),
    "amu/cm^3": ("kg/m^3", _AMU_KG * 1e6),
    "km/s": ("m/s", 1e3),
    "Gauss": ("T", 1e-4),
    "G": ("T", 1e-4),
    "nT": ("T", 1e-9),
    "erg/cm^3": ("J/m^3", 1e-1),
    "dyne/cm^2": ("Pa", 1e-1),
    "nPa": ("Pa", 1e-9),
    "`mA/m^2": ("A/m^2", 1e-6),
}


def build_griblet_batsrus_graph(
    variable_names: Sequence[str],
    *,
    aux: Mapping[str, object] | None = None,
    body_radius_m: float | None = None,
    include_unit_normalization: bool = True,
    include_derived: bool = True,
):
    """
    Build a griblet graph for BATSRUS-style fields.

    Current scope:
    - canonical bracketed names for unbracketed unit strings
    - SI conversion recipes for common BATSRUS units
    - optional coordinate conversion X/Y/Z [R] -> [m] (requires ``body_radius_m``)
    - common derived fields: |U|, |B|, c_s, c_A, M_A
    """
    griblet = importlib.import_module("griblet")
    graph = griblet.ComputationGraph()

    vars_list = list(variable_names)
    vars_set = set(vars_list)

    if include_unit_normalization:
        graph.merge(build_griblet_unit_normalization_graph(vars_list, aux=aux, body_radius_m=body_radius_m))

    if include_derived:
        graph.merge(build_griblet_common_derived_graph(vars_set))

    return graph


def build_griblet_unit_normalization_graph(
    variable_names: Sequence[str],
    *,
    aux: Mapping[str, object] | None = None,
    body_radius_m: float | None = None,
):
    griblet = importlib.import_module("griblet")
    graph = griblet.ComputationGraph()

    for raw_name in variable_names:
        parsed = _parse_var_name(raw_name)
        if parsed is None:
            continue
        base, unit = parsed

        canonical_name = f"{base} [{unit}]"
        if canonical_name != raw_name:
            graph.add_recipe(
                canonical_name,
                lambda x: x,
                deps=[raw_name],
                cost=0.01,
                metadata={"description": f"Canonicalize unit brackets for {raw_name}"},
            )
            source_name = canonical_name
        else:
            source_name = raw_name

        match = _UNIT_FACTORS.get(unit)
        if match is None:
            continue

        si_unit, factor = match
        target_name = f"{base} [{si_unit}]"
        if target_name == source_name and factor == 1:
            continue
        graph.add_recipe(
            target_name,
            lambda x, factor=factor: factor * np.array(x),
            deps=[source_name],
            cost=0.05,
            metadata={"description": f"Unit conversion {unit}->{si_unit}"},
        )

    # Optional coordinate scale: X/Y/Z [R] -> [m]
    body_radius = _resolve_body_radius_m(aux=aux, body_radius_m=body_radius_m)
    if body_radius is not None:
        for axis in ("X", "Y", "Z"):
            source = f"{axis} [R]"
            target = f"{axis} [m]"
            graph.add_recipe(
                target,
                lambda x, scale=body_radius: scale * np.array(x),
                deps=[source],
                cost=0.05,
                metadata={"description": "Scale body-radius coordinates to meters"},
            )
        graph.add_recipe(
            "RBODY [m]",
            lambda: float(body_radius),
            deps=[],
            cost=0.0,
            metadata={"description": "Configured body radius"},
        )

    # Parse common scalar aux values into numeric fields.
    if aux is not None and "GAMMA" in aux:
        graph.add_recipe(
            "GAMMA [none]",
            _parse_float,
            deps=["GAMMA"],
            cost=0.01,
            metadata={"description": "Parse GAMMA from aux"},
        )

    return graph


def build_griblet_common_derived_graph(variable_names: set[str] | Sequence[str]):
    griblet = importlib.import_module("griblet")
    graph = griblet.ComputationGraph()
    varset = set(variable_names)

    # Vector magnitudes (generic scan)
    graph.merge(build_griblet_vector_magnitude_graph(varset))
    graph.add_recipe(
        "U [m/s]",
        lambda x, y, z: np.sqrt(np.array(x) ** 2 + np.array(y) ** 2 + np.array(z) ** 2),
        deps=["U_x [m/s]", "U_y [m/s]", "U_z [m/s]"],
        cost=0.1,
        metadata={"description": "Flow speed magnitude"},
    )
    graph.add_recipe(
        "B [T]",
        lambda x, y, z: np.sqrt(np.array(x) ** 2 + np.array(y) ** 2 + np.array(z) ** 2),
        deps=["B_x [T]", "B_y [T]", "B_z [T]"],
        cost=0.1,
        metadata={"description": "Magnetic field magnitude"},
    )

    # Sound speed c_s [m/s]
    if {"P [Pa]", "Rho [kg/m^3]"}.issubset(varset) or True:
        graph.add_recipe(
            "c_s [m/s]",
            lambda P, rho: np.sqrt(_DEFAULT_GAMMA * np.array(P) / np.array(rho)),
            deps=["P [Pa]", "Rho [kg/m^3]"],
            cost=0.25,
            metadata={"description": "Adiabatic sound speed with fallback gamma=5/3"},
        )
        graph.add_recipe(
            "c_s [m/s]",
            lambda P, rho, gamma: np.sqrt(_safe_gamma(gamma) * np.array(P) / np.array(rho)),
            deps=["P [Pa]", "Rho [kg/m^3]", "GAMMA [none]"],
            cost=0.2,
            metadata={"description": "Adiabatic sound speed using GAMMA aux"},
        )

    # Alfven speed and Alfven Mach
    graph.add_recipe(
        "c_A [m/s]",
        lambda B, rho: np.array(B) / np.sqrt(_MU0 * np.array(rho)),
        deps=["B [T]", "Rho [kg/m^3]"],
        cost=0.2,
        metadata={"description": "Alfven speed"},
    )
    graph.add_recipe(
        "M_A [none]",
        lambda U, cA: np.array(U) / np.array(cA),
        deps=["U [m/s]", "c_A [m/s]"],
        cost=0.1,
        metadata={"description": "Alfven Mach number"},
    )
    graph.add_recipe(
        "Ma [none]",
        lambda U, cs: np.array(U) / np.array(cs),
        deps=["U [m/s]", "c_s [m/s]"],
        cost=0.1,
        metadata={"description": "Sonic Mach number"},
    )
    graph.add_recipe(
        "P_b [Pa]",
        lambda B: np.array(B) ** 2 / (2.0 * _MU0),
        deps=["B [T]"],
        cost=0.12,
        metadata={"description": "Magnetic pressure"},
    )
    graph.add_recipe(
        "beta [none]",
        lambda P, Pb: np.array(P) / np.array(Pb),
        deps=["P [Pa]", "P_b [Pa]"],
        cost=0.12,
        metadata={"description": "Plasma beta"},
    )

    return graph


def build_griblet_vector_magnitude_graph(variable_names: set[str] | Sequence[str]):
    griblet = importlib.import_module("griblet")
    graph = griblet.ComputationGraph()
    names = list(variable_names)
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<comp>[xyz]) \[(?P<unit>.+)\]$")

    by_prefix: dict[tuple[str, str], set[str]] = {}
    for name in names:
        m = pattern.match(name)
        if not m:
            continue
        key = (m.group("prefix"), m.group("unit"))
        by_prefix.setdefault(key, set()).add(m.group("comp"))

    for (prefix, unit), comps in sorted(by_prefix.items()):
        if comps != {"x", "y", "z"}:
            continue
        deps = [f"{prefix}_x [{unit}]", f"{prefix}_y [{unit}]", f"{prefix}_z [{unit}]"]
        graph.add_recipe(
            f"{prefix} [{unit}]",
            lambda x, y, z: np.sqrt(np.array(x) ** 2 + np.array(y) ** 2 + np.array(z) ** 2),
            deps=deps,
            cost=0.1,
            metadata={"description": f"{prefix} magnitude"},
        )
    return graph


def _parse_var_name(name: str):
    """
    Parse BATSRUS variable names.

    Supports:
    - ``Foo [unit]``
    - ``Foo unit``  (legacy unbracketed style)
    """
    m = re.match(r"^(?P<base>.+?) \[(?P<unit>.+)\]$", name)
    if m:
        return m.group("base"), m.group("unit")

    # If there is a space and no brackets, interpret the final token as the unit.
    if " " in name and "[" not in name and "]" not in name:
        base, unit = name.rsplit(" ", 1)
        if "/" in unit or unit.isalpha() or any(ch.isdigit() for ch in unit):
            return base, unit
    return None


def _parse_float(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    return float(str(x).strip())


def _safe_gamma(gamma):
    g = _parse_float(gamma)
    if not np.isfinite(g) or g <= 0:
        return _DEFAULT_GAMMA
    return g


def _resolve_body_radius_m(*, aux: Mapping[str, object] | None, body_radius_m: float | None):
    if body_radius_m is not None:
        return float(body_radius_m)

    if aux is None:
        return None

    # Leave room for future conventions; current example files do not include SI radius.
    for key in ("RBODY_M", "RBODY[m]", "RBODY [m]", "BODY_RADIUS_M"):
        if key in aux:
            try:
                return _parse_float(aux[key])
            except Exception:
                return None
    return None


__all__ = [
    "build_griblet_batsrus_graph",
    "build_griblet_common_derived_graph",
    "build_griblet_unit_normalization_graph",
    "build_griblet_vector_magnitude_graph",
]
