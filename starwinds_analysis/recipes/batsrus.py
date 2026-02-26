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
    Used by: `starwinds_analysis/smart_ds.py`
    """
    griblet = importlib.import_module("griblet")
    graph = griblet.ComputationGraph()

    vars_list = list(variable_names)

    if include_unit_normalization:
        graph.merge(build_griblet_unit_normalization_graph(vars_list, aux=aux, body_radius_m=body_radius_m))

    if include_derived:
        from starwinds_analysis.recipes.spherical import (
            build_griblet_auto_vector_spherical_components_graph,
            build_griblet_spherical_geometry_graph,
        )

        # Include spherical geometry/components in the BATSRUS graph from the start so
        # pointwise recipes can depend on U_r/B_r/U_phi/B_phi without extra setup.
        graph.merge(build_griblet_spherical_geometry_graph(coord_fields=("X [R]", "Y [R]", "Z [R]")))
        graph.merge(
            build_griblet_auto_vector_spherical_components_graph(
                list(graph.fields()) if hasattr(graph, "fields") else vars_list,
                coord_fields=("X [R]", "Y [R]", "Z [R]"),
                prefixes=None,
                components=("r", "theta", "phi"),
            )
        )
        derived_names = set(graph.fields()) if hasattr(graph, "fields") else set(vars_list)
        graph.merge(build_griblet_common_derived_graph(derived_names))

    return graph

def build_griblet_unit_normalization_graph(
    variable_names: Sequence[str],
    *,
    aux: Mapping[str, object] | None = None,
    body_radius_m: float | None = None,
):
    """
    Add raw->SI unit conversion recipes (BATSRUS naming conventions).
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
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
                lambda x: float(x) if isinstance(x, (int, float, np.floating)) else float(str(x).strip()),
                deps=["GAMMA"],
                cost=0.01,
                metadata={"description": "Parse GAMMA from aux"},
            )

    return graph

def build_griblet_common_derived_graph(variable_names: set[str] | Sequence[str]):
    """
    Add common BATSRUS derived SI quantities (pressures, Mach numbers, fluxes, torque
      densities).
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
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
        "magnetic_pressure [Pa]",
        lambda pb: np.array(pb),
        deps=["P_b [Pa]"],
        cost=0.01,
        metadata={"description": "Magnetic pressure alias"},
    )
    graph.add_recipe(
        "ram_pressure [Pa]",
        lambda rho, U: np.array(rho) * (np.array(U) ** 2),
        deps=["Rho [kg/m^3]", "U [m/s]"],
        cost=0.12,
        metadata={"description": "Ram pressure"},
    )
    graph.add_recipe(
        "beta [none]",
        lambda P, Pb: np.array(P) / np.array(Pb),
        deps=["P [Pa]", "P_b [Pa]"],
        cost=0.12,
        metadata={"description": "Plasma beta"},
    )

    # Pointwise flux densities (depend on spherical velocity component).
    graph.add_recipe(
        "mass_flux [kg/m^2/s]",
        lambda rho, ur: np.array(rho) * np.array(ur),
        deps=["Rho [kg/m^3]", "U_r [m/s]"],
        cost=0.12,
        metadata={"description": "Radial mass flux density"},
    )
    graph.add_recipe(
        "energy_flux [W/m^2]",
        lambda e, ur: np.array(e) * np.array(ur),
        deps=["E [J/m^3]", "U_r [m/s]"],
        cost=0.12,
        metadata={"description": "Radial energy flux density"},
    )

    # Useful geometry helpers derived from coordinates.
    graph.add_recipe(
        "cylindrical_radius [R]",
        lambda x, y: np.sqrt(np.array(x) ** 2 + np.array(y) ** 2),
        deps=["X [R]", "Y [R]"],
        cost=0.1,
        metadata={"description": "Cylindrical radius from body-radius coordinates"},
    )
    graph.add_recipe(
        "cylindrical_radius [m]",
        lambda x, y: np.sqrt(np.array(x) ** 2 + np.array(y) ** 2),
        deps=["X [m]", "Y [m]"],
        cost=0.1,
        metadata={"description": "Cylindrical radius from SI coordinates"},
    )

    # Pointwise shell-style torque densities (about +z).
    graph.add_recipe(
        "magnetic_torque_density [N/m]",
        lambda varpi, bphi, br: -np.array(varpi) * np.array(bphi) * np.array(br) / _MU0,
        deps=["cylindrical_radius [m]", "B_phi [T]", "B_r [T]"],
        cost=0.2,
        metadata={"description": "Magnetic z-torque density (shell form)"},
    )
    graph.add_recipe(
        "dynamic_torque_density [N/m]",
        lambda varpi, rho, uphi, ur: np.array(varpi) * np.array(rho) * np.array(uphi) * np.array(ur),
        deps=["cylindrical_radius [m]", "Rho [kg/m^3]", "U_phi [m/s]", "U_r [m/s]"],
        cost=0.2,
        metadata={"description": "Dynamic z-torque density (shell form)"},
    )
    graph.add_recipe(
        "total_torque_density [N/m]",
        lambda tmag, tdyn: np.array(tmag) + np.array(tdyn),
        deps=["magnetic_torque_density [N/m]", "dynamic_torque_density [N/m]"],
        cost=0.05,
        metadata={"description": "Total z-torque density (shell form)"},
    )

    # Latitude-map magnetic components (common plotting quantities).
    graph.add_recipe(
        "B_meridional [T]",
        lambda btheta: -np.array(btheta),
        deps=["B_theta [T]"],
        cost=0.05,
        metadata={"description": "Meridional magnetic component (northward)"},
    )
    graph.add_recipe(
        "B_tangential [T]",
        lambda bphi, bmer: np.sqrt(np.array(bphi) ** 2 + np.array(bmer) ** 2),
        deps=["B_phi [T]", "B_meridional [T]"],
        cost=0.08,
        metadata={"description": "Tangential magnetic magnitude on spherical shell"},
    )

    return graph

def build_griblet_vector_magnitude_graph(variable_names: set[str] | Sequence[str]):
    """
    Add vector-magnitude recipes (e.g. `|U|`, `|B|`) for available Cartesian triplets.
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
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
    Used by: `starwinds_analysis/recipes/batsrus.py`
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

def _safe_gamma(gamma):
    """
    Return a physically valid adiabatic index fallback when metadata is missing/bad.
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
    if isinstance(gamma, (int, float, np.floating)):
        g = float(gamma)
    else:
        g = float(str(gamma).strip())
    if not np.isfinite(g) or g <= 0:
        return _DEFAULT_GAMMA
    return g

def _resolve_body_radius_m(*, aux: Mapping[str, object] | None, body_radius_m: float | None):
    """
    Resolve body radius in meters from explicit arg or BATSRUS aux metadata.
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
    if body_radius_m is not None:
        return float(body_radius_m)

    if aux is None:
        return None

    # Leave room for future conventions; current example files do not include SI radius.
    for key in ("RBODY_M", "RBODY[m]", "RBODY [m]", "BODY_RADIUS_M"):
        if key in aux:
            try:
                value = aux[key]
                if isinstance(value, (int, float, np.floating)):
                    return float(value)
                return float(str(value).strip())
            except Exception:
                return None
    return None
