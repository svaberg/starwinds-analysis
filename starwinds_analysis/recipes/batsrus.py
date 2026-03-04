"""THIS FILE contains BATSRUS-specific normalization and derived-field recipes.

It defines SI conversion recipes and BATSRUS derived quantities for SmartDs/griblet usage.
It should keep BATSRUS naming/unit conventions localized here.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import griblet
import math

import numpy as np

from starwinds_analysis.constants import MU0

_AMU_KG = 1.66053906660e-27
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
    body_radius: float | None = None,
    include_unit_normalization: bool = True,
    include_derived: bool = True,
):
    """
    Build a griblet graph for BATSRUS-style fields.
    Adds:
    - `raw -> SI`
    - `XYZ <-> Rpa`
    - `U_xyz -> U_rpa`
    - `B_xyz -> B_rpa`
    - `SI -> common_derived`
    Example:
    - `raw -> SI` covers conversions like `B_x [Gauss] -> B_x [T]`
    - `SI -> common_derived` covers fields like `mass_flux`, `energy_flux`,
      `magnetic_torque_density`, and `B_tangential`.
    Used by: `starwinds_analysis/smart_ds.py`
    """
    graph = griblet.ComputationGraph()

    vars_list = list(variable_names)

    if include_unit_normalization:
        graph.merge(build_griblet_unit_normalization_graph(vars_list, aux=aux, body_radius=body_radius))

    if include_derived:
        from starwinds_analysis.recipes.spherical import _vector_triplets
        from starwinds_analysis.recipes.spherical import build_griblet_spherical_geometry_graph
        from starwinds_analysis.recipes.spherical import build_griblet_vector_spherical_components_graph

        derived_input_names: set[str] = set()
        for raw_name in vars_list:
            parsed = _parse_var_name(raw_name)
            if parsed is None:
                continue
            base, unit = parsed
            match = _UNIT_FACTORS.get(unit)
            if match is None:
                derived_input_names.add(raw_name)
            else:
                si_unit, _factor = match
                derived_input_names.add(f"{base} [{si_unit}]")
        if hasattr(graph, "fields"):
            derived_input_names.update(graph.fields())

        # Include spherical geometry/components in the BATSRUS graph from the start so
        # pointwise recipes can depend on U_r/B_r/U_a/B_a without extra setup.
        graph.merge(build_griblet_spherical_geometry_graph(coord_fields=("X [R]", "Y [R]", "Z [R]")))
        for prefix, unit in _vector_triplets(sorted(derived_input_names)):
            graph.merge(
                build_griblet_vector_spherical_components_graph(
                    prefix=prefix,
                    unit=unit,
                    coord_fields=("X [R]", "Y [R]", "Z [R]"),
                )
            )
        derived_names = set(derived_input_names)
        if hasattr(graph, "fields"):
            derived_names.update(graph.fields())
        graph.merge(build_griblet_common_derived_graph(derived_names))

    return graph

def build_griblet_unit_normalization_graph(
    variable_names: Sequence[str],
    *,
    aux: Mapping[str, object] | None = None,
    body_radius: float | None = None,
):
    """
    Add raw->SI unit conversion recipes (BATSRUS naming conventions).
    Adds:
    - `raw -> SI`
    - `XYZ [R] -> XYZ [m]` (when `body_radius` is available)
    Example:
    - `B_x [Gauss] -> B_x [T]`
    - `U_x [km/s] -> U_x [m/s]`
    - `X [R] -> X [m]`
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
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

        if base == "Lat":
            graph.add_recipe(
                f"latitude [{unit}]",
                lambda x: x,
                deps=[source_name],
                cost=0.01,
                metadata={"description": "Normalize Lat to latitude"},
            )
        if base == "Lon":
            graph.add_recipe(
                f"longitude [{unit}]",
                lambda x: x,
                deps=[source_name],
                cost=0.01,
                metadata={"description": "Normalize Lon to longitude"},
            )

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
    body_radius = body_radius_from_inputs(aux=aux, body_radius=body_radius)
    if body_radius is not None:
        # Add XYZ [R] -> XYZ [m].
        for axis in ("X", "Y", "Z"):
            graph.add_recipe(
                f"{axis} [m]",
                lambda x, scale=body_radius: scale * np.array(x),
                deps=[f"{axis} [R]"],
                cost=0.05,
                metadata={"description": "Scale body-radius coordinates to meters"},
            )
        graph.add_recipe(
            "R [m]",
            lambda r, scale=body_radius: scale * np.array(r),
            deps=["R [R]"],
            cost=0.05,
            metadata={"description": "Scale spherical radius to meters"},
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

    # Add stellar params from aux/PARAM.in when available.
    for raw_name, field_name in (
        ("Star_radius_m", "star_radius [m]"),
        ("Star_mass_kg", "star_mass [kg]"),
        ("Star_rotational_period_s", "star_rotational_period [s]"),
        ("Star_rotation_rate_rad_s", "star_rotation_rate [rad/s]"),
    ):
        if aux is None or raw_name not in aux:
            continue
        graph.add_recipe(
            field_name,
            lambda x: float(x) if isinstance(x, (int, float, np.floating)) else float(str(x).strip()),
            deps=[raw_name],
            cost=0.01,
            metadata={"description": f"Parse {raw_name} from aux"},
        )

    return graph

def build_griblet_common_derived_graph(variable_names: set[str] | Sequence[str]):
    """
    Add common BATSRUS derived SI quantities (pressures, Mach numbers, fluxes, torque
      densities).
    Adds:
    - `U_xyz -> U`
    - `B_xyz -> B`
    - `Rho + U_r -> mass_flux`
    - `E + U_r -> energy_flux`
    - `varpi + B_a + B_r -> magnetic_torque_density`
    - `varpi + Rho + U_a + U_r -> dynamic_torque_density`
    Example:
    - `U_xyz -> U` means `U_x/U_y/U_z` can produce `U [m/s]`
    - `varpi + B_a + B_r -> magnetic_torque_density` means the pointwise
      shell-style magnetic torque density is built from those SI fields
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
    graph = griblet.ComputationGraph()
    varset = set(variable_names)

    # Vector magnitudes (generic scan)
    graph.merge(build_griblet_vector_cartesian_graph(varset))
    # Add explicit U_xyz -> U and B_xyz -> B.
    for prefix, unit, description in (
        ("U", "m/s", "Flow speed magnitude"),
        ("B", "T", "Magnetic field magnitude"),
    ):
        graph.add_recipe(
            f"{prefix} [{unit}]",
            lambda x, y, z: np.sqrt(np.array(x) ** 2 + np.array(y) ** 2 + np.array(z) ** 2),
            deps=[f"{prefix}_x [{unit}]", f"{prefix}_y [{unit}]", f"{prefix}_z [{unit}]"],
            cost=0.1,
            metadata={"description": description},
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
        lambda B, rho: np.array(B) / np.sqrt(MU0 * np.array(rho)),
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
        lambda B: np.array(B) ** 2 / (2.0 * MU0),
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
        "thermal_pressure [Pa]",
        lambda p: np.array(p),
        deps=["P [Pa]"],
        cost=0.01,
        metadata={"description": "Thermal pressure alias"},
    )
    graph.add_recipe(
        "ram_pressure [Pa]",
        lambda rho, U: np.array(rho) * (np.array(U) ** 2),
        deps=["Rho [kg/m^3]", "U [m/s]"],
        cost=0.12,
        metadata={"description": "Ram pressure"},
    )
    graph.add_recipe(
        "standoff_distance [m]",
        lambda rho, U: np.power(
            ((0.7e-4) ** 2 / (2.0 * MU0)) / (np.array(rho) * (np.array(U) ** 2)),
            1.0 / 6.0,
        ),
        deps=["Rho [kg/m^3]", "U [m/s]"],
        cost=0.2,
        metadata={"description": "Magnetospheric stand-off proxy from inertial ram pressure"},
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
        lambda varpi, bphi, br: -np.array(varpi) * np.array(bphi) * np.array(br) / MU0,
        deps=["cylindrical_radius [m]", "B_a [T]", "B_r [T]"],
        cost=0.2,
        metadata={"description": "Magnetic z-torque density (shell form)"},
    )
    graph.add_recipe(
        "dynamic_torque_density [N/m]",
        lambda varpi, rho, uphi, ur: np.array(varpi) * np.array(rho) * np.array(uphi) * np.array(ur),
        deps=["cylindrical_radius [m]", "Rho [kg/m^3]", "U_a [m/s]", "U_r [m/s]"],
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
        lambda bp: -np.array(bp),
        deps=["B_p [T]"],
        cost=0.05,
        metadata={"description": "Meridional magnetic component (northward)"},
    )
    graph.add_recipe(
        "B_tangential [T]",
        lambda ba, bmer: np.sqrt(np.array(ba) ** 2 + np.array(bmer) ** 2),
        deps=["B_a [T]", "B_meridional [T]"],
        cost=0.08,
        metadata={"description": "Tangential magnetic magnitude on spherical shell"},
    )

    return graph

def build_griblet_vector_cartesian_graph(variable_names: set[str] | Sequence[str]):
    """
    Add stacked Cartesian-vector and magnitude recipes for available triplets.
    Adds:
    - `prefix_x/y/z -> prefix_xyz`
    - `prefix_x/y/z -> prefix`
    Example:
    - `U_x/U_y/U_z -> U_xyz`
    - `U_x/U_y/U_z -> U`
    - `B_x/B_y/B_z -> B_xyz`
    - `B_x/B_y/B_z -> B`
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
    graph = griblet.ComputationGraph()
    names = list(variable_names)

    by_prefix: dict[tuple[str, str], set[str]] = {}
    for name in names:
        parsed = _parse_xyz_component_name(name)
        if parsed is None:
            continue
        prefix, comp, unit = parsed
        key = (prefix, unit)
        by_prefix.setdefault(key, set()).add(comp)

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
    if " [" in name and name.endswith("]"):
        base, unit = name[:-1].split(" [", 1)
        if base and unit:
            return base, unit

    # If there is a space and no brackets, interpret the final token as the unit.
    if " " in name and "[" not in name and "]" not in name:
        base, unit = name.rsplit(" ", 1)
        if "/" in unit or unit.isalpha() or any(ch.isdigit() for ch in unit):
            return base, unit
    return None


def _parse_xyz_component_name(name: str):
    """
    Parse names like ``prefix_x [unit]``.
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
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

def body_radius_from_inputs(*, aux: Mapping[str, object] | None, body_radius: float | None):
    """
    Resolve body radius in meters from explicit arg or BATSRUS aux metadata.
    Used by: `starwinds_analysis/recipes/batsrus.py`
    """
    if body_radius is not None:
        return float(body_radius)

    if aux is None:
        return None

    for key in ("Star_radius_m", "Planet_radius_m", "RBODY_M", "RBODY[m]", "RBODY [m]", "BODY_RADIUS_M"):
        if key in aux:
            try:
                value = aux[key]
                if isinstance(value, (int, float, np.floating)):
                    return float(value)
                return float(str(value).strip())
            except (TypeError, ValueError):
                return None
    return None
