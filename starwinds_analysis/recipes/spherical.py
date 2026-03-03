"""THIS FILE contains spherical field recipes built on low-level transforms.

It provides SmartDs/griblet wiring for spherical geometry and vector fields.
It should avoid owning the raw Cartesian/spherical transform math.
"""

from __future__ import annotations

from collections.abc import Sequence

import griblet
import numpy as np
from starwinds_analysis.algorithms.spherical import (
    cartesian_to_spherical_coordinates,
    cartesian_vector_to_spherical_components,
    latitude_longitude_to_polar_azimuth,
    polar_azimuth_to_latitude_longitude,
    spherical_to_cartesian_coordinates,
    spherical_vector_to_cartesian_components,
)


def register_spherical_geometry_fields(
    smart_ds,
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
):
    """
    Register local on-demand spherical coordinate fields on a SmartDs wrapper.
    Used by: `starwinds_analysis/smart_ds.py`
    """
    x_name, y_name, z_name = coord_fields
    r_name = _infer_radius_name_from_coord(x_name) or "R [unknown]"

    def _coordinates(ds):
        return cartesian_to_spherical_coordinates(ds.variable(x_name), ds.variable(y_name), ds.variable(z_name))

    smart_ds.register_field(r_name, lambda ds: _coordinates(ds)[0], overwrite=True)
    smart_ds.register_field("polar [rad]", lambda ds: _coordinates(ds)[1], overwrite=True)
    smart_ds.register_field("azimuth [rad]", lambda ds: _coordinates(ds)[2], overwrite=True)

    smart_ds.register_field(
        "latitude [rad]",
        lambda ds: polar_azimuth_to_latitude_longitude(ds.variable("polar [rad]"), ds.variable("azimuth [rad]"))[0],
        overwrite=True,
    )
    smart_ds.register_field(
        "longitude [rad]",
        lambda ds: polar_azimuth_to_latitude_longitude(ds.variable("polar [rad]"), ds.variable("azimuth [rad]"))[1],
        overwrite=True,
    )
    smart_ds.register_field(
        "latitude [deg]",
        lambda ds: np.degrees(ds.variable("latitude [rad]")),
        overwrite=True,
    )
    smart_ds.register_field(
        "longitude [deg]",
        lambda ds: np.degrees(ds.variable("longitude [rad]")),
        overwrite=True,
    )


def register_vector_spherical_components(
    smart_ds,
    *,
    prefix: str,
    unit: str,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
):
    """
    Register local on-demand spherical vector components for one Cartesian vector triplet.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    x_name, y_name, z_name = coord_fields
    vx_name = f"{prefix}_x [{unit}]"
    vy_name = f"{prefix}_y [{unit}]"
    vz_name = f"{prefix}_z [{unit}]"

    def _compute(ds):
        """
        Compute all spherical vector components once per SmartDs request.
        Used by: `register_vector_spherical_components` (nested helper)
        """
        return cartesian_vector_to_spherical_components(
            ds.variable(vx_name),
            ds.variable(vy_name),
            ds.variable(vz_name),
            ds.variable(x_name),
            ds.variable(y_name),
            ds.variable(z_name),
        )

    smart_ds.register_field(f"{prefix}_r [{unit}]", lambda ds: _compute(ds)[0], overwrite=True)
    smart_ds.register_field(f"{prefix}_p [{unit}]", lambda ds: _compute(ds)[1], overwrite=True)
    smart_ds.register_field(f"{prefix}_a [{unit}]", lambda ds: _compute(ds)[2], overwrite=True)


def _vector_triplets(
    variable_names: Sequence[str],
    *,
    prefixes: Sequence[str] | None = None,
):
    """
    Find Cartesian vector triplets named like ``prefix_x [unit]``.
    Used by: `starwinds_analysis/recipes/spherical.py`,
      `starwinds_analysis/smart_ds.py`, `starwinds_analysis/recipes/batsrus.py`
    """
    by_prefix: dict[str, dict[str, str]] = {}

    for name in variable_names:
        parsed = _parse_xyz_component_name(name)
        if parsed is None:
            continue
        prefix, comp, unit = parsed
        slot = by_prefix.setdefault(prefix, {"unit": unit})
        # Skip mixed-unit vectors for now.
        if slot["unit"] != unit:
            continue
        slot[comp] = name

    wanted = set(prefixes) if prefixes is not None else None
    triplets = []
    for prefix, info in sorted(by_prefix.items()):
        if wanted is not None and prefix not in wanted:
            continue
        if not {"x", "y", "z"}.issubset(info):
            continue
        triplets.append((prefix, info["unit"]))
    return triplets


def build_griblet_spherical_geometry_graph(
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
):
    """
    Build a griblet graph for spherical geometry fields.
    Used by: `test/test_smart_ds.py`, `starwinds_analysis/smart_ds.py`,
      `starwinds_analysis/recipes/batsrus.py`
    """
    x_name, y_name, z_name = coord_fields
    r_name = _infer_radius_name_from_coord(x_name) or "R [unknown]"

    graph = griblet.ComputationGraph()

    deps = [x_name, y_name, z_name]
    graph.add_recipe(
        r_name,
        lambda x, y, z: cartesian_to_spherical_coordinates(x, y, z)[0],
        deps=deps,
        cost=0.2,
    )
    graph.add_recipe(
        "polar [rad]",
        lambda x, y, z: cartesian_to_spherical_coordinates(x, y, z)[1],
        deps=deps,
        cost=0.2,
    )
    graph.add_recipe(
        "azimuth [rad]",
        lambda x, y, z: cartesian_to_spherical_coordinates(x, y, z)[2],
        deps=deps,
        cost=0.2,
    )
    graph.add_recipe(
        "latitude [rad]",
        lambda polar, azimuth: polar_azimuth_to_latitude_longitude(polar, azimuth)[0],
        deps=["polar [rad]", "azimuth [rad]"],
        cost=0.05,
    )
    graph.add_recipe(
        "polar [rad]",
        lambda latitude, longitude: latitude_longitude_to_polar_azimuth(latitude, longitude)[0],
        deps=["latitude [rad]", "longitude [rad]"],
        cost=0.05,
    )
    graph.add_recipe(
        "longitude [rad]",
        lambda polar, azimuth: polar_azimuth_to_latitude_longitude(polar, azimuth)[1],
        deps=["polar [rad]", "azimuth [rad]"],
        cost=0.01,
    )
    graph.add_recipe(
        "azimuth [rad]",
        lambda latitude, longitude: latitude_longitude_to_polar_azimuth(latitude, longitude)[1],
        deps=["latitude [rad]", "longitude [rad]"],
        cost=0.01,
    )
    graph.add_recipe(
        "latitude [deg]",
        lambda lat: np.degrees(lat),
        deps=["latitude [rad]"],
        cost=0.05,
    )
    graph.add_recipe(
        "longitude [deg]",
        lambda lon: np.degrees(lon),
        deps=["longitude [rad]"],
        cost=0.05,
    )
    graph.add_recipe(
        "latitude [rad]",
        lambda lat_deg: np.deg2rad(lat_deg),
        deps=["latitude [deg]"],
        cost=0.05,
    )
    graph.add_recipe(
        "longitude [rad]",
        lambda lon_deg: np.deg2rad(lon_deg),
        deps=["longitude [deg]"],
        cost=0.05,
    )
    graph.add_recipe(
        x_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[0],
        deps=[r_name, "polar [rad]", "azimuth [rad]"],
        cost=0.25,
    )
    graph.add_recipe(
        y_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[1],
        deps=[r_name, "polar [rad]", "azimuth [rad]"],
        cost=0.25,
    )
    graph.add_recipe(
        z_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[2],
        deps=[r_name, "polar [rad]", "azimuth [rad]"],
        cost=0.25,
    )

    return graph


def build_griblet_vector_spherical_components_graph(
    *,
    prefix: str,
    unit: str,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
):
    """
    Build griblet recipes for ``prefix_{r,p,a}`` from Cartesian components.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    x_name, y_name, z_name = coord_fields
    vx_name = f"{prefix}_x [{unit}]"
    vy_name = f"{prefix}_y [{unit}]"
    vz_name = f"{prefix}_z [{unit}]"
    deps = [vx_name, vy_name, vz_name, x_name, y_name, z_name]

    graph = griblet.ComputationGraph()
    graph.add_recipe(
        f"{prefix}_r [{unit}]",
        lambda vx, vy, vz, x, y, z: cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z)[0],
        deps=deps,
        cost=0.4,
    )
    graph.add_recipe(
        f"{prefix}_p [{unit}]",
        lambda vx, vy, vz, x, y, z: cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z)[1],
        deps=deps,
        cost=0.5,
    )
    graph.add_recipe(
        f"{prefix}_a [{unit}]",
        lambda vx, vy, vz, x, y, z: cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z)[2],
        deps=deps,
        cost=0.5,
    )
    return graph


def _infer_radius_name_from_coord(x_name: str) -> str | None:
    """
    Infer the matching radius field name/unit from coordinate field names.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    if x_name.startswith("X [") and x_name.endswith("]"):
        return f"R [{x_name[3:-1]}]"
    return None


def _parse_xyz_component_name(name: str) -> tuple[str, str, str] | None:
    """
    Parse names like ``prefix_x [unit]``.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    if " [" not in name or not name.endswith("]"):
        return None
    head, unit = name[:-1].split(" [", 1)
    if "_" not in head:
        return None
    prefix, comp = head.rsplit("_", 1)
    if comp not in ("x", "y", "z") or not prefix:
        return None
    return prefix, comp, unit
