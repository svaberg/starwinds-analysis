from __future__ import annotations

from collections.abc import Sequence
import re

import griblet
import numpy as np

SPHERICAL_COMPONENTS = ("r", "p", "a")


def cartesian_to_spherical_angles(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Conventions:
    - ``r >= 0``
    - ``polar`` is colatitude in ``[0, pi]`` (NaN at ``r == 0``)
    - ``azimuth`` is azimuth from ``atan2(y, x)`` in ``[-pi, pi]``
      (NaN where ``x == y == 0``)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r = np.sqrt(x * x + y * y + z * z)
    rho_xy = np.sqrt(x * x + y * y)

    polar = np.full_like(r, np.nan, dtype=float)
    azimuth = np.full_like(r, np.nan, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        mask_r = r > 0
        cos_polar = np.empty_like(r, dtype=float)
        cos_polar.fill(np.nan)
        cos_polar[mask_r] = np.clip(z[mask_r] / r[mask_r], -1.0, 1.0)
        polar[mask_r] = np.arccos(cos_polar[mask_r])

        mask_azimuth = rho_xy > 0
        azimuth[mask_azimuth] = np.arctan2(y[mask_azimuth], x[mask_azimuth])

    return r, polar, azimuth


def radial_component(vx, vy, vz, x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)

    r = np.sqrt(x * x + y * y + z * z)
    out = np.full_like(r, np.nan, dtype=float)
    mask = r > 0
    out[mask] = (vx[mask] * x[mask] + vy[mask] * y[mask] + vz[mask] * z[mask]) / r[mask]
    return out


def spherical_vector_components(vx, vy, vz, x, y, z):
    """
    Return ``(v_r, v_p, v_a)`` using physics convention ``polar=colatitude``.

    ``v_p`` and ``v_a`` are undefined on the polar axis (``x=y=0``) and are
    returned as NaN there. All components are NaN at ``r=0``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)

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

        # e_p = (cos(polar)cos(azimuth), cos(polar)sin(azimuth), -sin(polar))
        # Derived directly in Cartesian coordinates to avoid angle singularities.
        v_p[mask_axis] = (zz * (xx * vxx + yy * vyy) - (xx * xx + yy * yy) * vzz) / (rr * rho)
        # e_a = (-sin(azimuth), cos(azimuth), 0)
        v_a[mask_axis] = (-yy * vxx + xx * vyy) / rho

    return v_r, v_p, v_a


def build_griblet_spherical_geometry_graph(
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
):
    """
    Build a griblet graph for spherical geometry fields.

    This requires ``griblet``.
    """
    x_name, y_name, z_name = coord_fields
    r_name = _infer_radius_name_from_coord(x_name)
    if r_name is None:
        raise ValueError(f"could not infer radius name from coordinate field {x_name!r}")

    polar_name = "polar [rad]"
    azimuth_name = "azimuth [rad]"
    graph = griblet.ComputationGraph()

    deps = [x_name, y_name, z_name]
    graph.add_recipe(
        r_name,
        lambda x, y, z: cartesian_to_spherical_angles(x, y, z)[0],
        deps=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical radius"},
    )
    graph.add_recipe(
        polar_name,
        lambda x, y, z: cartesian_to_spherical_angles(x, y, z)[1],
        deps=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical colatitude"},
    )
    graph.add_recipe(
        azimuth_name,
        lambda x, y, z: cartesian_to_spherical_angles(x, y, z)[2],
        deps=deps,
        cost=0.2,
        metadata={"description": "Cartesian->spherical azimuth"},
    )

    return graph


def build_griblet_vector_spherical_components_graph(
    variable_names: Sequence[str],
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
):
    """
    Auto-detect Cartesian vector triplets in ``variable_names`` and build a merged
    griblet graph for their spherical components.
    """
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<comp>[xyz]) \[(?P<unit>.+)\]$")
    x_name, y_name, z_name = coord_fields
    by_prefix: dict[tuple[str, str], set[str]] = {}
    for name in variable_names:
        m = pattern.match(name)
        if not m:
            continue
        by_prefix.setdefault((m.group("prefix"), m.group("unit")), set()).add(m.group("comp"))

    merged = griblet.ComputationGraph()
    for (prefix, unit), found in sorted(by_prefix.items()):
        if found != {"x", "y", "z"}:
            continue
        deps = [f"{prefix}_x [{unit}]", f"{prefix}_y [{unit}]", f"{prefix}_z [{unit}]", x_name, y_name, z_name]

        def _all(vx, vy, vz, x, y, z):
            return spherical_vector_components(vx, vy, vz, x, y, z)

        if "r" in SPHERICAL_COMPONENTS:
            merged.add_recipe(
                f"{prefix}_r [{unit}]",
                lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[0],
                deps=deps,
                cost=0.4,
                metadata={"description": f"{prefix} radial component"},
            )
        if "p" in SPHERICAL_COMPONENTS:
            merged.add_recipe(
                f"{prefix}_p [{unit}]",
                lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[1],
                deps=deps,
                cost=0.5,
                metadata={"description": f"{prefix} polar component"},
            )
        if "a" in SPHERICAL_COMPONENTS:
            merged.add_recipe(
                f"{prefix}_a [{unit}]",
                lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[2],
                deps=deps,
                cost=0.5,
                metadata={"description": f"{prefix} azimuthal component"},
            )
    return merged


def _infer_radius_name_from_coord(x_name: str) -> str | None:
    m = re.match(r"^X \[(.+)\]$", x_name)
    if m:
        return f"R [{m.group(1)}]"
    return None
