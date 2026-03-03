"""THIS FILE contains spherical field recipes built on low-level transforms.

It provides SmartDs/griblet wiring for spherical geometry and vector fields.
It should avoid owning the raw Cartesian/spherical transform math.
"""

from __future__ import annotations

from collections.abc import Sequence

import griblet
import numpy as np
from starwinds_analysis.algorithms.spherical import cartesian_to_spherical_coordinates
from starwinds_analysis.algorithms.spherical import cartesian_vector_to_spherical_components
from starwinds_analysis.algorithms.spherical import latitude_longitude_to_polar_azimuth
from starwinds_analysis.algorithms.spherical import polar_azimuth_to_latitude_longitude
from starwinds_analysis.algorithms.spherical import spherical_to_cartesian_coordinates
from starwinds_analysis.algorithms.spherical import spherical_vector_to_cartesian_components


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
    rpa_names = (r_name, "polar [rad]", "azimuth [rad]")
    latlon_names = ("latitude [rad]", "longitude [rad]")
    latlon_deg_names = ("latitude [deg]", "longitude [deg]")

    def _coordinates(ds):
        return cartesian_to_spherical_coordinates(ds.variable(x_name), ds.variable(y_name), ds.variable(z_name))

    # Add XYZ -> Rpa.
    for index, field_name in enumerate(rpa_names):
        smart_ds.register_field(field_name, lambda ds, index=index: _coordinates(ds)[index], overwrite=True)

    # Add Rpa -> latlon_rad.
    for index, field_name in enumerate(latlon_names):
        smart_ds.register_field(
            field_name,
            lambda ds, index=index: polar_azimuth_to_latitude_longitude(
                ds.variable("polar [rad]"),
                ds.variable("azimuth [rad]"),
            )[index],
            overwrite=True,
        )

    # Add latlon_rad -> latlon_deg.
    for rad_name, deg_name in zip(latlon_names, latlon_deg_names):
        smart_ds.register_field(
            deg_name,
            lambda ds, rad_name=rad_name: np.degrees(ds.variable(rad_name)),
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
    Adds:
    - `XYZ <-> Rpa`
    - `pa <-> latlon`
    - `latlon_rad <-> latlon_deg`
    Example:
    - `XYZ <-> Rpa` means `X/Y/Z` can produce `R/polar/azimuth`, and
      `R/polar/azimuth` can produce `X/Y/Z`.
    Used by: `test/test_smart_ds.py`, `starwinds_analysis/smart_ds.py`,
      `starwinds_analysis/recipes/batsrus.py`
    """
    x_name, y_name, z_name = coord_fields
    r_name = _infer_radius_name_from_coord(x_name) or "R [unknown]"
    xyz_names = (x_name, y_name, z_name)
    rpa_names = (r_name, "polar [rad]", "azimuth [rad]")
    latlon_names = ("latitude [rad]", "longitude [rad]")
    latlon_deg_names = ("latitude [deg]", "longitude [deg]")

    graph = griblet.ComputationGraph()

    # Add XYZ -> Rpa.
    for index, field_name in enumerate(rpa_names):
        graph.add_recipe(
            field_name,
            lambda x, y, z, index=index: cartesian_to_spherical_coordinates(x, y, z)[index],
            deps=list(xyz_names),
            cost=0.2,
        )

    # Add Rpa -> latlon_rad.
    for index, field_name in enumerate(latlon_names):
        graph.add_recipe(
            field_name,
            lambda polar, azimuth, index=index: polar_azimuth_to_latitude_longitude(polar, azimuth)[index],
            deps=["polar [rad]", "azimuth [rad]"],
            cost=0.05 if index == 0 else 0.01,
        )

    # Add latlon_rad -> Rpa angles.
    for index, field_name in enumerate(("polar [rad]", "azimuth [rad]")):
        graph.add_recipe(
            field_name,
            lambda latitude, longitude, index=index: latitude_longitude_to_polar_azimuth(latitude, longitude)[index],
            deps=list(latlon_names),
            cost=0.05 if index == 0 else 0.01,
        )

    # Add latlon_rad -> latlon_deg.
    for rad_name, deg_name in zip(latlon_names, latlon_deg_names):
        graph.add_recipe(
            deg_name,
            lambda angle_rad: np.degrees(angle_rad),
            deps=[rad_name],
            cost=0.05,
        )

    # Add latlon_deg -> latlon_rad.
    for rad_name, deg_name in zip(latlon_names, latlon_deg_names):
        graph.add_recipe(
            rad_name,
            lambda angle_deg: np.deg2rad(angle_deg),
            deps=[deg_name],
            cost=0.05,
        )

    # Add Rpa -> XYZ.
    for index, field_name in enumerate(xyz_names):
        graph.add_recipe(
            field_name,
            lambda r, polar, azimuth, index=index: spherical_to_cartesian_coordinates(r, polar, azimuth)[index],
            deps=list(rpa_names),
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
    Adds:
    - `prefix_xyz -> prefix_rpa`
    Example:
    - `U_xyz -> U_rpa`
    - `B_xyz -> B_rpa`
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    x_name, y_name, z_name = coord_fields
    vx_name = f"{prefix}_x [{unit}]"
    vy_name = f"{prefix}_y [{unit}]"
    vz_name = f"{prefix}_z [{unit}]"
    xyz_names = (vx_name, vy_name, vz_name, x_name, y_name, z_name)
    rpa_names = (f"{prefix}_r [{unit}]", f"{prefix}_p [{unit}]", f"{prefix}_a [{unit}]")

    graph = griblet.ComputationGraph()
    # Add prefix_xyz -> prefix_rpa.
    for index, field_name in enumerate(rpa_names):
        graph.add_recipe(
            field_name,
            lambda vx, vy, vz, x, y, z, index=index: cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z)[index],
            deps=list(xyz_names),
            cost=0.4 if index == 0 else 0.5,
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
