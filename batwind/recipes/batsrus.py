from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re

import griblet
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


def build_batsrus_graph(
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
    graph = griblet.ComputationGraph()

    vars_list = list(variable_names)
    vars_set = set(vars_list)

    if include_unit_normalization:
        graph.merge(build_unit_normalization_graph(vars_list, aux=aux, body_radius_m=body_radius_m))

    if include_derived:
        graph.merge(build_common_derived_graph(vars_set))

    return graph


def build_unit_normalization_graph(
    variable_names: Sequence[str],
    *,
    aux: Mapping[str, object] | None = None,
    body_radius_m: float | None = None,
):
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
            lambda x, factor=factor: factor * np.asarray(x),
            deps=[source_name],
            cost=0.05,
            metadata={"description": f"Unit conversion {unit}->{si_unit}"},
        )

    if body_radius_m is not None:
        graph.merge(build_coordinate_scale_graph(body_radius_m))

    # Parse common scalar aux values into numeric fields.
    if aux is not None and "GAMMA" in aux:
        graph.add_recipe(
            "GAMMA [none]",
            float,
            deps=["GAMMA"],
            cost=0.01,
            metadata={"description": "Parse GAMMA from aux"},
        )

    return graph


def build_coordinate_scale_graph(body_radius_m: float):
    graph = griblet.ComputationGraph()
    graph.add_recipe(
        "RBODY [m]",
        lambda: float(body_radius_m),
        deps=[],
        cost=0.0,
        metadata={"description": "Configured body radius"},
    )
    for axis in ("X", "Y", "Z"):
        source = f"{axis} [R]"
        target = f"{axis} [m]"
        graph.add_recipe(
            target,
            lambda x, rbody: np.asarray(rbody) * np.asarray(x),
            deps=[source, "RBODY [m]"],
            cost=0.05,
            metadata={"description": "Scale body-radius coordinates to meters"},
        )
    graph.add_recipe(
        "R [m]",
        lambda r, rbody: np.asarray(rbody) * np.asarray(r),
        deps=["R [R]", "RBODY [m]"],
        cost=0.05,
        metadata={"description": "Scale spherical radius to meters"},
    )
    return graph


def build_common_derived_graph(variable_names: set[str] | Sequence[str]):
    graph = griblet.ComputationGraph()
    varset = set(variable_names)

    # Cartesian vector stacks and magnitudes.
    graph.merge(build_vector_cartesian_graph(varset))
    graph.merge(build_vector_magnitude_graph(varset))

    # Sound speed c_s [m/s]
    if {"P [Pa]", "Rho [kg/m^3]"}.issubset(varset) or True:
        graph.add_recipe(
            "c_s [m/s]",
            lambda P, rho: np.sqrt(_DEFAULT_GAMMA * np.asarray(P) / np.asarray(rho)),
            deps=["P [Pa]", "Rho [kg/m^3]"],
            cost=0.25,
            metadata={"description": "Adiabatic sound speed with fallback gamma=5/3"},
        )
        graph.add_recipe(
            "c_s [m/s]",
            lambda P, rho, gamma: np.sqrt(_safe_gamma(gamma) * np.asarray(P) / np.asarray(rho)),
            deps=["P [Pa]", "Rho [kg/m^3]", "GAMMA [none]"],
            cost=0.2,
            metadata={"description": "Adiabatic sound speed using GAMMA aux"},
        )

    # Alfven speed and Alfven Mach
    graph.add_recipe(
        "c_A [m/s]",
        lambda B, rho: np.asarray(B) / np.sqrt(_MU0 * np.asarray(rho)),
        deps=["B [T]", "Rho [kg/m^3]"],
        cost=0.2,
        metadata={"description": "Alfven speed"},
    )
    graph.add_recipe(
        "M_A [none]",
        lambda U, cA: np.asarray(U) / np.asarray(cA),
        deps=["U [m/s]", "c_A [m/s]"],
        cost=0.1,
        metadata={"description": "Alfven Mach number"},
    )
    graph.add_recipe(
        "Ma [none]",
        lambda U, cs: np.asarray(U) / np.asarray(cs),
        deps=["U [m/s]", "c_s [m/s]"],
        cost=0.1,
        metadata={"description": "Sonic Mach number"},
    )
    graph.add_recipe(
        "P_b [Pa]",
        lambda B: np.asarray(B) ** 2 / (2.0 * _MU0),
        deps=["B [T]"],
        cost=0.12,
        metadata={"description": "Magnetic pressure"},
    )
    graph.add_recipe(
        "magnetic_pressure [Pa]",
        lambda pb: np.asarray(pb),
        deps=["P_b [Pa]"],
        cost=0.01,
        metadata={"description": "Magnetic pressure alias"},
    )
    graph.add_recipe(
        "thermal_pressure [Pa]",
        lambda p: np.asarray(p),
        deps=["P [Pa]"],
        cost=0.01,
        metadata={"description": "Thermal pressure alias"},
    )
    graph.add_recipe(
        "ram_pressure [Pa]",
        lambda rho, U: np.asarray(rho) * (np.asarray(U) ** 2),
        deps=["Rho [kg/m^3]", "U [m/s]"],
        cost=0.12,
        metadata={"description": "Ram pressure"},
    )
    graph.add_recipe(
        "standoff_distance [m]",
        _standoff_distance_from_rho_u,
        deps=["Rho [kg/m^3]", "U [m/s]"],
        cost=0.2,
        metadata={"description": "Magnetospheric stand-off proxy from inertial ram pressure"},
    )
    graph.add_recipe(
        "beta [none]",
        lambda P, Pb: np.asarray(P) / np.asarray(Pb),
        deps=["P [Pa]", "P_b [Pa]"],
        cost=0.12,
        metadata={"description": "Plasma beta"},
    )

    graph.add_recipe(
        "mass_flux [kg/m^2/s]",
        lambda rho, ur: np.asarray(rho) * np.asarray(ur),
        deps=["Rho [kg/m^3]", "U_r [m/s]"],
        cost=0.12,
        metadata={"description": "Radial mass flux density"},
    )
    graph.add_recipe(
        "energy_flux [W/m^2]",
        lambda e, ur: np.asarray(e) * np.asarray(ur),
        deps=["E [J/m^3]", "U_r [m/s]"],
        cost=0.12,
        metadata={"description": "Radial energy flux density"},
    )

    graph.add_recipe(
        "cylindrical_radius [R]",
        lambda x, y: np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2),
        deps=["X [R]", "Y [R]"],
        cost=0.1,
        metadata={"description": "Cylindrical radius from body-radius coordinates"},
    )
    graph.add_recipe(
        "cylindrical_radius [m]",
        lambda x, y: np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2),
        deps=["X [m]", "Y [m]"],
        cost=0.1,
        metadata={"description": "Cylindrical radius from SI coordinates"},
    )

    graph.add_recipe(
        "magnetic_torque_density [N/m]",
        lambda varpi, bphi, br: -np.asarray(varpi) * np.asarray(bphi) * np.asarray(br) / _MU0,
        deps=["cylindrical_radius [m]", "B_a [T]", "B_r [T]"],
        cost=0.2,
        metadata={"description": "Magnetic z-torque density (shell form)"},
    )
    graph.add_recipe(
        "dynamic_torque_density [N/m]",
        lambda varpi, rho, uphi, ur: np.asarray(varpi) * np.asarray(rho) * np.asarray(uphi) * np.asarray(ur),
        deps=["cylindrical_radius [m]", "Rho [kg/m^3]", "U_a [m/s]", "U_r [m/s]"],
        cost=0.2,
        metadata={"description": "Dynamic z-torque density (shell form)"},
    )
    graph.add_recipe(
        "total_torque_density [N/m]",
        lambda tmag, tdyn: np.asarray(tmag) + np.asarray(tdyn),
        deps=["magnetic_torque_density [N/m]", "dynamic_torque_density [N/m]"],
        cost=0.05,
        metadata={"description": "Total z-torque density (shell form)"},
    )

    graph.add_recipe(
        "B_meridional [T]",
        lambda bp: -np.asarray(bp),
        deps=["B_p [T]"],
        cost=0.05,
        metadata={"description": "Meridional magnetic component (northward)"},
    )
    graph.add_recipe(
        "B_tangential [T]",
        lambda ba, bmer: np.sqrt(np.asarray(ba) ** 2 + np.asarray(bmer) ** 2),
        deps=["B_a [T]", "B_meridional [T]"],
        cost=0.08,
        metadata={"description": "Tangential magnetic magnitude on spherical shell"},
    )

    return graph


def build_vector_magnitude_graph(variable_names: set[str] | Sequence[str]):
    graph = griblet.ComputationGraph()
    by_prefix: dict[tuple[str, str], set[str]] = {}
    for prefix, comp, unit in _available_xyz_components(variable_names):
        by_prefix.setdefault((prefix, unit), set()).add(comp)

    for (prefix, unit), comps in sorted(by_prefix.items()):
        if comps != {"x", "y", "z"}:
            continue
        deps = [f"{prefix}_x [{unit}]", f"{prefix}_y [{unit}]", f"{prefix}_z [{unit}]"]
        graph.add_recipe(
            f"{prefix} [{unit}]",
            lambda x, y, z: np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2),
            deps=deps,
            cost=0.1,
            metadata={"description": f"{prefix} magnitude"},
        )
    return graph


def build_vector_cartesian_graph(variable_names: set[str] | Sequence[str]):
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


def _safe_gamma(gamma):
    g = float(gamma)
    if not np.isfinite(g) or g <= 0:
        return _DEFAULT_GAMMA
    return g


def _standoff_distance_from_rho_u(rho, U):
    p_ram = np.asarray(rho) * (np.asarray(U) ** 2)
    numer = (0.7e-4**2) / (2.0 * _MU0)
    return np.power(numer / p_ram, 1.0 / 6.0)


__all__ = [
    "build_batsrus_graph",
    "build_common_derived_graph",
    "build_coordinate_scale_graph",
    "build_unit_normalization_graph",
    "build_vector_cartesian_graph",
    "build_vector_magnitude_graph",
]
