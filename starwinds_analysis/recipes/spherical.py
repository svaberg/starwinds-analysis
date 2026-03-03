"""THIS FILE contains spherical coordinate/vector transforms and related recipes.

It provides geometry transforms plus optional recipe-graph builders for on-demand spherical fields.
It should remain backend-agnostic and avoid plotting concerns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import logging
import re

import numpy as np

log = logging.getLogger(__name__)


def cartesian_to_spherical_angles(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    r = np.sqrt(x * x + y * y + z * z)
    rho_xy = np.sqrt(x * x + y * y)

    theta = np.full_like(r, np.nan, dtype=float)
    phi = np.full_like(r, np.nan, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        mask_r = r > 0
        cos_theta = np.empty_like(r, dtype=float)
        cos_theta.fill(np.nan)
        cos_theta[mask_r] = np.clip(z[mask_r] / r[mask_r], -1.0, 1.0)
        theta[mask_r] = np.arccos(cos_theta[mask_r])

        mask_phi = rho_xy > 0
        phi[mask_phi] = np.arctan2(y[mask_phi], x[mask_phi])

    return r, theta, phi


def spherical_to_cartesian_coordinates(r, polar, azimuth):
    """
    Convert spherical coordinates into Cartesian coordinates.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    r = np.array(r)
    polar = np.array(polar)
    azimuth = np.array(azimuth)
    sin_polar = np.sin(polar)
    x = r * sin_polar * np.cos(azimuth)
    y = r * sin_polar * np.sin(azimuth)
    z = r * np.cos(polar)
    return x, y, z


def radial_component(vx, vy, vz, x, y, z):
    """
    Project a Cartesian vector onto the radial direction defined by `(x,y,z)`.
    Used by: no external call sites found
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)

    r = np.sqrt(x * x + y * y + z * z)
    out = np.full_like(r, np.nan, dtype=float)
    mask = r > 0
    out[mask] = (vx[mask] * x[mask] + vy[mask] * y[mask] + vz[mask] * z[mask]) / r[mask]
    return out


def spherical_vector_components(vx, vy, vz, x, y, z):
    """
    Return ``(v_r, v_theta, v_phi)`` using physics convention ``theta=colatitude``.
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/recipes/spherical.py`
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)

    r = np.sqrt(x * x + y * y + z * z)
    rho_xy = np.sqrt(x * x + y * y)

    v_r = np.full_like(r, np.nan, dtype=float)
    v_theta = np.full_like(r, np.nan, dtype=float)
    v_phi = np.full_like(r, np.nan, dtype=float)

    mask_r = r > 0
    v_r[mask_r] = (vx[mask_r] * x[mask_r] + vy[mask_r] * y[mask_r] + vz[mask_r] * z[mask_r]) / r[mask_r]

    mask_axis = mask_r & (rho_xy > 0)
    if np.any(mask_axis):
        rho = rho_xy[mask_axis]
        rr = r[mask_axis]
        xx = x[mask_axis]
        yy = y[mask_axis]
        zz = z[mask_axis]
        vxx = vx[mask_axis]
        vyy = vy[mask_axis]
        vzz = vz[mask_axis]

        # e_theta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
        # Derived directly in Cartesian coordinates to avoid angle singularities.
        v_theta[mask_axis] = (zz * (xx * vxx + yy * vyy) - (xx * xx + yy * yy) * vzz) / (rr * rho)
        # e_phi = (-sin(phi), cos(phi), 0)
        v_phi[mask_axis] = (-yy * vxx + xx * vyy) / rho

    return v_r, v_theta, v_phi


def register_spherical_geometry_fields(
    smart_ds,
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    r_name: str | None = None,
    polar_name: str = "polar [rad]",
    azimuth_name: str = "azimuth [rad]",
    theta_name: str = "theta [rad]",
    phi_name: str = "phi [rad]",
    latitude_name: str = "latitude [rad]",
    longitude_name: str = "longitude [rad]",
    latitude_deg_name: str = "latitude [deg]",
    longitude_deg_name: str = "longitude [deg]",
):
    """
    Register local on-demand spherical coordinate fields on a SmartDs wrapper.
    Used by: `starwinds_analysis/smart_ds.py`
    """
    x_name, y_name, z_name = coord_fields
    if r_name is None:
        r_name = _infer_radius_name_from_coord(x_name) or "R [unknown]"

    def _r(ds):
        """
        Radius from Cartesian coordinates (same unit as `coord_fields`).
        Used by: `register_spherical_geometry_fields` (nested helper)
        """
        x = ds.variable(x_name)
        y = ds.variable(y_name)
        z = ds.variable(z_name)
        r, _theta, _phi = cartesian_to_spherical_angles(x, y, z)
        return r

    def _polar(ds):
        """
        Polar angle from Cartesian coordinates.
        Used by: `register_spherical_geometry_fields` (nested helper)
        """
        x = ds.variable(x_name)
        y = ds.variable(y_name)
        z = ds.variable(z_name)
        _r, polar, _azimuth = cartesian_to_spherical_angles(x, y, z)
        return polar

    def _azimuth(ds):
        """
        Azimuth from Cartesian coordinates.
        Used by: `register_spherical_geometry_fields` (nested helper)
        """
        x = ds.variable(x_name)
        y = ds.variable(y_name)
        z = ds.variable(z_name)
        _r, _polar, azimuth = cartesian_to_spherical_angles(x, y, z)
        return azimuth

    smart_ds.register_field(r_name, _r, overwrite=True)
    smart_ds.register_field(polar_name, _polar, overwrite=True)
    smart_ds.register_field(azimuth_name, _azimuth, overwrite=True)
    if theta_name != polar_name:
        smart_ds.register_field(theta_name, lambda ds: np.array(ds.variable(polar_name)), overwrite=True)
    if phi_name != azimuth_name:
        smart_ds.register_field(phi_name, lambda ds: np.array(ds.variable(azimuth_name)), overwrite=True)

    smart_ds.register_field(
        latitude_name,
        lambda ds: (0.5 * np.pi) - np.array(ds.variable(polar_name)),
        overwrite=True,
    )
    smart_ds.register_field(
        longitude_name,
        lambda ds: np.array(ds.variable(azimuth_name)),
        overwrite=True,
    )
    smart_ds.register_field(
        latitude_deg_name,
        lambda ds: np.degrees(np.array(ds.variable(latitude_name))),
        overwrite=True,
    )
    smart_ds.register_field(
        longitude_deg_name,
        lambda ds: np.degrees(np.array(ds.variable(longitude_name))),
        overwrite=True,
    )
    registered = {
        "r": r_name,
        "polar": polar_name,
        "azimuth": azimuth_name,
        "theta": theta_name,
        "phi": phi_name,
        "latitude": latitude_name,
        "longitude": longitude_name,
        "latitude_deg": latitude_deg_name,
        "longitude_deg": longitude_deg_name,
    }
    log.debug("registered_spherical_geometry_fields %r", registered)


def register_vector_spherical_components(
    smart_ds,
    *,
    prefix: str,
    unit: str,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    component_names: Mapping[str, str] | None = None,
    register_components: Sequence[str] = ("r", "p", "a"),
):
    """
    Register local on-demand spherical vector components for one Cartesian vector triplet.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    x_name, y_name, z_name = coord_fields
    vx_name = f"{prefix}_x [{unit}]"
    vy_name = f"{prefix}_y [{unit}]"
    vz_name = f"{prefix}_z [{unit}]"

    default_component_names = {
        "r": f"{prefix}_r [{unit}]",
        "p": f"{prefix}_p [{unit}]",
        "a": f"{prefix}_a [{unit}]",
        "theta": f"{prefix}_theta [{unit}]",
        "phi": f"{prefix}_phi [{unit}]",
    }
    component_names = {
        **default_component_names,
        **({} if component_names is None else dict(component_names)),
    }

    def _compute(ds):
        """
        Compute all spherical vector components once per SmartDs request.
        Used by: `register_vector_spherical_components` (nested helper)
        """
        x = ds.variable(x_name)
        y = ds.variable(y_name)
        z = ds.variable(z_name)
        vx = ds.variable(vx_name)
        vy = ds.variable(vy_name)
        vz = ds.variable(vz_name)
        return spherical_vector_components(vx, vy, vz, x, y, z)

    if "r" in register_components:
        smart_ds.register_field(component_names["r"], lambda ds: _compute(ds)[0], overwrite=True)
    if ("p" in register_components) or ("theta" in register_components):
        smart_ds.register_field(component_names["p"], lambda ds: _compute(ds)[1], overwrite=True)
        if component_names["theta"] != component_names["p"]:
            smart_ds.register_field(component_names["theta"], lambda ds: np.array(ds.variable(component_names["p"])), overwrite=True)
    if ("a" in register_components) or ("phi" in register_components):
        smart_ds.register_field(component_names["a"], lambda ds: _compute(ds)[2], overwrite=True)
        if component_names["phi"] != component_names["a"]:
            smart_ds.register_field(component_names["phi"], lambda ds: np.array(ds.variable(component_names["a"])), overwrite=True)

    return component_names


def auto_register_vector_spherical_components(
    smart_ds,
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    prefixes: Sequence[str] | None = None,
    components: Sequence[str] = ("r", "p", "a"),
):
    """
    Auto-detect vector component triplets named like ``prefix_x [unit]``.
    Used by: `starwinds_analysis/smart_ds.py`
    """
    by_prefix: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<comp>[xyz]) \[(?P<unit>.+)\]$")

    for name in smart_ds.variables:
        m = pattern.match(name)
        if not m:
            continue
        prefix = m.group("prefix")
        comp = m.group("comp")
        unit = m.group("unit")
        slot = by_prefix.setdefault(prefix, {"unit": unit})
        # Skip mixed-unit vectors for now.
        if slot["unit"] != unit:
            continue
        slot[comp] = name

    created_prefixes = []
    wanted = set(prefixes) if prefixes is not None else None
    for prefix, info in sorted(by_prefix.items()):
        if wanted is not None and prefix not in wanted:
            continue
        if not {"x", "y", "z"}.issubset(info):
            continue
        unit = info["unit"]
        register_vector_spherical_components(
            smart_ds,
            prefix=prefix,
            unit=unit,
            coord_fields=coord_fields,
            register_components=components,
        )
        created_prefixes.append(prefix)
    log.debug("auto_registered_vector_spherical_components %r", created_prefixes)


def build_griblet_spherical_geometry_graph(
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    r_name: str | None = None,
    polar_name: str = "polar [rad]",
    azimuth_name: str = "azimuth [rad]",
    theta_name: str = "theta [rad]",
    phi_name: str = "phi [rad]",
    latitude_name: str = "latitude [rad]",
    longitude_name: str = "longitude [rad]",
    latitude_deg_name: str = "latitude [deg]",
    longitude_deg_name: str = "longitude [deg]",
):
    """
    Build a griblet graph for spherical geometry fields.
    Used by: `test/test_smart_ds.py`, `starwinds_analysis/smart_ds.py`,
      `starwinds_analysis/recipes/batsrus.py`
    """
    griblet = importlib.import_module("griblet")
    x_name, y_name, z_name = coord_fields
    if r_name is None:
        r_name = _infer_radius_name_from_coord(x_name) or "R [unknown]"

    graph = griblet.ComputationGraph()

    def _r(x, y, z):
        """
        Radius recipe from Cartesian coordinates.
        Used by: `build_griblet_spherical_geometry_graph` (nested helper)
        """
        r, _theta, _phi = cartesian_to_spherical_angles(x, y, z)
        return r

    def _polar(x, y, z):
        """
        Polar-angle recipe from Cartesian coordinates.
        Used by: `build_griblet_spherical_geometry_graph` (nested helper)
        """
        _r, polar, _azimuth = cartesian_to_spherical_angles(x, y, z)
        return polar

    def _azimuth(x, y, z):
        """
        Azimuth recipe from Cartesian coordinates.
        Used by: `build_griblet_spherical_geometry_graph` (nested helper)
        """
        _r, _polar, azimuth = cartesian_to_spherical_angles(x, y, z)
        return azimuth

    deps = [x_name, y_name, z_name]
    graph.add_recipe(r_name, _r, deps=deps, cost=0.2, metadata={"description": "Cartesian->spherical radius"})
    graph.add_recipe(polar_name, _polar, deps=deps, cost=0.2, metadata={"description": "Cartesian->spherical polar angle"})
    graph.add_recipe(azimuth_name, _azimuth, deps=deps, cost=0.2, metadata={"description": "Cartesian->spherical azimuth"})
    if theta_name != polar_name:
        graph.add_recipe(
            theta_name,
            lambda polar: np.array(polar),
            deps=[polar_name],
            cost=0.01,
            metadata={"description": "polar-angle alias"},
        )
    if phi_name != azimuth_name:
        graph.add_recipe(
            phi_name,
            lambda azimuth: np.array(azimuth),
            deps=[azimuth_name],
            cost=0.01,
            metadata={"description": "azimuth alias"},
        )

    graph.add_recipe(
        latitude_name,
        lambda polar: (0.5 * np.pi) - np.array(polar),
        deps=[polar_name],
        cost=0.05,
        metadata={"description": "polar -> latitude (radians)"},
    )
    graph.add_recipe(
        polar_name,
        lambda latitude: (0.5 * np.pi) - np.array(latitude),
        deps=[latitude_name],
        cost=0.05,
        metadata={"description": "latitude -> polar (radians)"},
    )
    graph.add_recipe(
        longitude_name,
        lambda azimuth: np.array(azimuth),
        deps=[azimuth_name],
        cost=0.01,
        metadata={"description": "azimuth -> longitude (radians)"},
    )
    graph.add_recipe(
        azimuth_name,
        lambda longitude: np.array(longitude),
        deps=[longitude_name],
        cost=0.01,
        metadata={"description": "longitude -> azimuth (radians)"},
    )
    graph.add_recipe(
        latitude_deg_name,
        lambda lat: np.degrees(np.array(lat)),
        deps=[latitude_name],
        cost=0.05,
        metadata={"description": "latitude radians -> degrees"},
    )
    graph.add_recipe(
        longitude_deg_name,
        lambda lon: np.degrees(np.array(lon)),
        deps=[longitude_name],
        cost=0.05,
        metadata={"description": "longitude radians -> degrees"},
    )
    graph.add_recipe(
        latitude_name,
        lambda lat_deg: np.deg2rad(np.array(lat_deg)),
        deps=[latitude_deg_name],
        cost=0.05,
        metadata={"description": "latitude degrees -> radians"},
    )
    graph.add_recipe(
        longitude_name,
        lambda lon_deg: np.deg2rad(np.array(lon_deg)),
        deps=[longitude_deg_name],
        cost=0.05,
        metadata={"description": "longitude degrees -> radians"},
    )
    graph.add_recipe(
        x_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[0],
        deps=[r_name, polar_name, azimuth_name],
        cost=0.25,
        metadata={"description": "Spherical->Cartesian X"},
    )
    graph.add_recipe(
        y_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[1],
        deps=[r_name, polar_name, azimuth_name],
        cost=0.25,
        metadata={"description": "Spherical->Cartesian Y"},
    )
    graph.add_recipe(
        z_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[2],
        deps=[r_name, polar_name, azimuth_name],
        cost=0.25,
        metadata={"description": "Spherical->Cartesian Z"},
    )

    return graph


def build_griblet_vector_spherical_components_graph(
    *,
    prefix: str,
    unit: str,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    register_components: Sequence[str] = ("r", "p", "a"),
):
    """
    Build griblet recipes for ``prefix_{r,p,a}`` (with ``theta/phi`` aliases) from Cartesian components.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    griblet = importlib.import_module("griblet")
    x_name, y_name, z_name = coord_fields
    vx_name = f"{prefix}_x [{unit}]"
    vy_name = f"{prefix}_y [{unit}]"
    vz_name = f"{prefix}_z [{unit}]"
    deps = [vx_name, vy_name, vz_name, x_name, y_name, z_name]

    graph = griblet.ComputationGraph()

    def _all(vx, vy, vz, x, y, z):
        """
        Compute all spherical vector components once for graph recipes.
        Used by: `build_griblet_vector_spherical_components_graph` (nested helper)
        """
        return spherical_vector_components(vx, vy, vz, x, y, z)

    if "r" in register_components:
        graph.add_recipe(
            f"{prefix}_r [{unit}]",
            lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[0],
            deps=deps,
            cost=0.4,
            metadata={"description": f"{prefix} radial component"},
        )
    if ("p" in register_components) or ("theta" in register_components):
        graph.add_recipe(
            f"{prefix}_p [{unit}]",
            lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[1],
            deps=deps,
            cost=0.5,
            metadata={"description": f"{prefix} polar component"},
        )
        graph.add_recipe(
            f"{prefix}_theta [{unit}]",
            lambda vp: np.array(vp),
            deps=[f"{prefix}_p [{unit}]"],
            cost=0.01,
            metadata={"description": f"{prefix} polar alias"},
        )
    if ("a" in register_components) or ("phi" in register_components):
        graph.add_recipe(
            f"{prefix}_a [{unit}]",
            lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[2],
            deps=deps,
            cost=0.5,
            metadata={"description": f"{prefix} azimuth component"},
        )
        graph.add_recipe(
            f"{prefix}_phi [{unit}]",
            lambda va: np.array(va),
            deps=[f"{prefix}_a [{unit}]"],
            cost=0.01,
            metadata={"description": f"{prefix} azimuth alias"},
        )
    return graph


def build_griblet_auto_vector_spherical_components_graph(
    variable_names: Sequence[str],
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    prefixes: Sequence[str] | None = None,
    components: Sequence[str] = ("r", "p", "a"),
):
    """
    Auto-detect Cartesian vector triplets in `variable_names` and build a merged spherical-
      component recipe graph.
    Used by: `starwinds_analysis/smart_ds.py`, `starwinds_analysis/recipes/batsrus.py`
    """
    griblet = importlib.import_module("griblet")
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<comp>[xyz]) \[(?P<unit>.+)\]$")

    by_prefix: dict[str, dict[str, str]] = {}
    for name in variable_names:
        m = pattern.match(name)
        if not m:
            continue
        prefix = m.group("prefix")
        comp = m.group("comp")
        unit = m.group("unit")
        slot = by_prefix.setdefault(prefix, {"unit": unit})
        if slot["unit"] != unit:
            continue
        slot[comp] = name

    wanted = set(prefixes) if prefixes is not None else None
    merged = griblet.ComputationGraph()
    for prefix, info in sorted(by_prefix.items()):
        if wanted is not None and prefix not in wanted:
            continue
        if not {"x", "y", "z"}.issubset(info):
            continue
        merged.merge(
            build_griblet_vector_spherical_components_graph(
                prefix=prefix,
                unit=info["unit"],
                coord_fields=coord_fields,
                register_components=components,
            )
        )
    return merged


def _infer_radius_name_from_coord(x_name: str) -> str | None:
    """
    Infer the matching radius field name/unit from coordinate field names.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    m = re.match(r"^X \[(.+)\]$", x_name)
    if m:
        return f"R [{m.group(1)}]"
    return None
