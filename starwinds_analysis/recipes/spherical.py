"""THIS FILE contains spherical coordinate/vector transforms and related recipes.

It provides geometry transforms plus optional recipe-graph builders for on-demand spherical fields.
It should remain backend-agnostic and avoid plotting concerns.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import logging
import re

import numpy as np

log = logging.getLogger(__name__)


def cartesian_to_spherical_coordinates(x, y, z):
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


def polar_azimuth_to_latitude_longitude(polar, azimuth):
    """
    Convert polar/azimuth coordinates to latitude/longitude.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    polar = np.array(polar)
    azimuth = np.array(azimuth)
    latitude = (0.5 * np.pi) - polar
    longitude = np.array(azimuth)
    return latitude, longitude


def latitude_longitude_to_polar_azimuth(latitude, longitude):
    """
    Convert latitude/longitude to polar/azimuth coordinates.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    polar = (0.5 * np.pi) - latitude
    azimuth = np.array(longitude)
    return polar, azimuth


def cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z):
    """
    Return ``(v_r, v_p, v_a)`` using `polar` and `azimuth` coordinates.
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
    v_p = np.full_like(r, np.nan, dtype=float)
    v_a = np.full_like(r, np.nan, dtype=float)

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

        v_p[mask_axis] = (zz * (xx * vxx + yy * vyy) - (xx * xx + yy * yy) * vzz) / (rr * rho)
        v_a[mask_axis] = (-yy * vxx + xx * vyy) / rho

    return v_r, v_p, v_a


def spherical_vector_to_cartesian_components(v_r, v_p, v_a, polar, azimuth):
    """
    Convert spherical vector components `(r, p, a)` into Cartesian components.
    Used by: `starwinds_analysis/recipes/spherical.py`
    """
    v_r = np.array(v_r)
    v_p = np.array(v_p)
    v_a = np.array(v_a)
    polar = np.array(polar)
    azimuth = np.array(azimuth)
    sin_polar = np.sin(polar)
    cos_polar = np.cos(polar)
    sin_azimuth = np.sin(azimuth)
    cos_azimuth = np.cos(azimuth)
    vx = v_r * sin_polar * cos_azimuth + v_p * cos_polar * cos_azimuth - v_a * sin_azimuth
    vy = v_r * sin_polar * sin_azimuth + v_p * cos_polar * sin_azimuth + v_a * cos_azimuth
    vz = v_r * cos_polar - v_p * sin_polar
    return vx, vy, vz


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
    smart_ds.register_field("theta [rad]", lambda ds: np.array(ds.variable("polar [rad]")), overwrite=True)
    smart_ds.register_field("phi [rad]", lambda ds: np.array(ds.variable("azimuth [rad]")), overwrite=True)

    smart_ds.register_field(
        "latitude [rad]",
        lambda ds: (0.5 * np.pi) - np.array(ds.variable("polar [rad]")),
        overwrite=True,
    )
    smart_ds.register_field(
        "longitude [rad]",
        lambda ds: np.array(ds.variable("azimuth [rad]")),
        overwrite=True,
    )
    smart_ds.register_field(
        "latitude [deg]",
        lambda ds: np.degrees(np.array(ds.variable("latitude [rad]"))),
        overwrite=True,
    )
    smart_ds.register_field(
        "longitude [deg]",
        lambda ds: np.degrees(np.array(ds.variable("longitude [rad]"))),
        overwrite=True,
    )


def register_vector_spherical_components(
    smart_ds,
    *,
    prefix: str,
    unit: str,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
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

    requested = set(register_components)
    if "theta" in requested:
        requested.add("p")
    if "phi" in requested:
        requested.add("a")

    if "r" in requested:
        smart_ds.register_field(f"{prefix}_r [{unit}]", lambda ds: _compute(ds)[0], overwrite=True)
    if "p" in requested:
        smart_ds.register_field(f"{prefix}_p [{unit}]", lambda ds: _compute(ds)[1], overwrite=True)
        smart_ds.register_field(
            f"{prefix}_theta [{unit}]",
            lambda ds: np.array(ds.variable(f"{prefix}_p [{unit}]")),
            overwrite=True,
        )
    if "a" in requested:
        smart_ds.register_field(f"{prefix}_a [{unit}]", lambda ds: _compute(ds)[2], overwrite=True)
        smart_ds.register_field(
            f"{prefix}_phi [{unit}]",
            lambda ds: np.array(ds.variable(f"{prefix}_a [{unit}]")),
            overwrite=True,
        )


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
):
    """
    Build a griblet graph for spherical geometry fields.
    Used by: `test/test_smart_ds.py`, `starwinds_analysis/smart_ds.py`,
      `starwinds_analysis/recipes/batsrus.py`
    """
    griblet = importlib.import_module("griblet")
    x_name, y_name, z_name = coord_fields
    r_name = _infer_radius_name_from_coord(x_name) or "R [unknown]"

    graph = griblet.ComputationGraph()

    deps = [x_name, y_name, z_name]
    graph.add_recipe(
        r_name,
        lambda x, y, z: cartesian_to_spherical_coordinates(x, y, z)[0],
        deps=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical radius"},
    )
    graph.add_recipe(
        "polar [rad]",
        lambda x, y, z: cartesian_to_spherical_coordinates(x, y, z)[1],
        deps=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical polar angle"},
    )
    graph.add_recipe(
        "azimuth [rad]",
        lambda x, y, z: cartesian_to_spherical_coordinates(x, y, z)[2],
        deps=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical azimuth"},
    )
    graph.add_recipe(
        "theta [rad]",
        lambda polar: np.array(polar),
        deps=["polar [rad]"],
        cost=0.01,
        metadata={"description": "polar-angle alias"},
    )
    graph.add_recipe(
        "phi [rad]",
        lambda azimuth: np.array(azimuth),
        deps=["azimuth [rad]"],
        cost=0.01,
        metadata={"description": "azimuth alias"},
    )

    graph.add_recipe(
        "latitude [rad]",
        lambda polar: (0.5 * np.pi) - np.array(polar),
        deps=["polar [rad]"],
        cost=0.05,
        metadata={"description": "polar -> latitude (radians)"},
    )
    graph.add_recipe(
        "polar [rad]",
        lambda latitude: (0.5 * np.pi) - np.array(latitude),
        deps=["latitude [rad]"],
        cost=0.05,
        metadata={"description": "latitude -> polar (radians)"},
    )
    graph.add_recipe(
        "longitude [rad]",
        lambda azimuth: np.array(azimuth),
        deps=["azimuth [rad]"],
        cost=0.01,
        metadata={"description": "azimuth -> longitude (radians)"},
    )
    graph.add_recipe(
        "azimuth [rad]",
        lambda longitude: np.array(longitude),
        deps=["longitude [rad]"],
        cost=0.01,
        metadata={"description": "longitude -> azimuth (radians)"},
    )
    graph.add_recipe(
        "latitude [deg]",
        lambda lat: np.degrees(np.array(lat)),
        deps=["latitude [rad]"],
        cost=0.05,
        metadata={"description": "latitude radians -> degrees"},
    )
    graph.add_recipe(
        "longitude [deg]",
        lambda lon: np.degrees(np.array(lon)),
        deps=["longitude [rad]"],
        cost=0.05,
        metadata={"description": "longitude radians -> degrees"},
    )
    graph.add_recipe(
        "latitude [rad]",
        lambda lat_deg: np.deg2rad(np.array(lat_deg)),
        deps=["latitude [deg]"],
        cost=0.05,
        metadata={"description": "latitude degrees -> radians"},
    )
    graph.add_recipe(
        "longitude [rad]",
        lambda lon_deg: np.deg2rad(np.array(lon_deg)),
        deps=["longitude [deg]"],
        cost=0.05,
        metadata={"description": "longitude degrees -> radians"},
    )
    graph.add_recipe(
        x_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[0],
        deps=[r_name, "polar [rad]", "azimuth [rad]"],
        cost=0.25,
        metadata={"description": "Spherical->Cartesian X"},
    )
    graph.add_recipe(
        y_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[1],
        deps=[r_name, "polar [rad]", "azimuth [rad]"],
        cost=0.25,
        metadata={"description": "Spherical->Cartesian Y"},
    )
    graph.add_recipe(
        z_name,
        lambda r, polar, azimuth: spherical_to_cartesian_coordinates(r, polar, azimuth)[2],
        deps=[r_name, "polar [rad]", "azimuth [rad]"],
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
    requested = set(register_components)
    if "theta" in requested:
        requested.add("p")
    if "phi" in requested:
        requested.add("a")

    if "r" in requested:
        graph.add_recipe(
            f"{prefix}_r [{unit}]",
            lambda vx, vy, vz, x, y, z: cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z)[0],
            deps=deps,
            cost=0.4,
            metadata={"description": f"{prefix} radial component"},
        )
    if "p" in requested:
        graph.add_recipe(
            f"{prefix}_p [{unit}]",
            lambda vx, vy, vz, x, y, z: cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z)[1],
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
    if "a" in requested:
        graph.add_recipe(
            f"{prefix}_a [{unit}]",
            lambda vx, vy, vz, x, y, z: cartesian_vector_to_spherical_components(vx, vy, vz, x, y, z)[2],
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
