from __future__ import annotations

from collections.abc import Sequence
import importlib
import re

import numpy as np


def cartesian_to_spherical_angles(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Conventions:
    - ``r >= 0``
    - ``theta`` is colatitude in ``[0, pi]`` (NaN at ``r == 0``)
    - ``phi`` is azimuth from ``atan2(y, x)`` in ``[-pi, pi]``
      (NaN where ``x == y == 0``)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

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
    Return ``(v_r, v_theta, v_phi)`` using physics convention ``theta=colatitude``.

    ``v_theta`` and ``v_phi`` are undefined on the polar axis (``x=y=0``) and are
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


def build_griblet_spherical_geometry_graph(
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    r_name: str | None = None,
    theta_name: str = "theta [rad]",
    phi_name: str = "phi [rad]",
):
    """
    Build a griblet graph for spherical geometry fields.

    This requires ``griblet``.
    """
    griblet = importlib.import_module("griblet")
    x_name, y_name, z_name = coord_fields
    if r_name is None:
        r_name = _infer_radius_name_from_coord(x_name) or "R [unknown]"

    graph = griblet.ComputationGraph()

    def _r(x, y, z):
        r, _theta, _phi = cartesian_to_spherical_angles(x, y, z)
        return r

    def _theta(x, y, z):
        _r, theta, _phi = cartesian_to_spherical_angles(x, y, z)
        return theta

    def _phi(x, y, z):
        _r, _theta, phi = cartesian_to_spherical_angles(x, y, z)
        return phi

    deps = [x_name, y_name, z_name]
    graph.add_recipe(r_name, _r, deps=deps, cost=0.2, metadata={"description": "Cartesian->spherical radius"})
    graph.add_recipe(theta_name, _theta, deps=deps, cost=0.2, metadata={"description": "Cartesian->spherical colatitude"})
    graph.add_recipe(phi_name, _phi, deps=deps, cost=0.2, metadata={"description": "Cartesian->spherical azimuth"})

    return graph


def build_griblet_vector_spherical_components_graph(
    *,
    prefix: str,
    unit: str,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    register_components: Sequence[str] = ("r", "theta", "phi"),
):
    """
    Build griblet recipes for ``prefix_{r,theta,phi}`` from Cartesian components.
    """
    griblet = importlib.import_module("griblet")
    x_name, y_name, z_name = coord_fields
    vx_name = f"{prefix}_x [{unit}]"
    vy_name = f"{prefix}_y [{unit}]"
    vz_name = f"{prefix}_z [{unit}]"
    deps = [vx_name, vy_name, vz_name, x_name, y_name, z_name]

    graph = griblet.ComputationGraph()

    def _all(vx, vy, vz, x, y, z):
        return spherical_vector_components(vx, vy, vz, x, y, z)

    if "r" in register_components:
        graph.add_recipe(
            f"{prefix}_r [{unit}]",
            lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[0],
            deps=deps,
            cost=0.4,
            metadata={"description": f"{prefix} radial component"},
        )
    if "theta" in register_components:
        graph.add_recipe(
            f"{prefix}_theta [{unit}]",
            lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[1],
            deps=deps,
            cost=0.5,
            metadata={"description": f"{prefix} colatitudinal component"},
        )
    if "phi" in register_components:
        graph.add_recipe(
            f"{prefix}_phi [{unit}]",
            lambda vx, vy, vz, x, y, z: _all(vx, vy, vz, x, y, z)[2],
            deps=deps,
            cost=0.5,
            metadata={"description": f"{prefix} azimuthal component"},
        )
    return graph


def build_griblet_auto_vector_spherical_components_graph(
    variable_names: Sequence[str],
    *,
    coord_fields: Sequence[str] = ("X [R]", "Y [R]", "Z [R]"),
    prefixes: Sequence[str] | None = None,
    components: Sequence[str] = ("r", "theta", "phi"),
):
    """
    Auto-detect Cartesian vector triplets in ``variable_names`` and build a merged
    griblet graph for their spherical components.
    """
    griblet = importlib.import_module("griblet")
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<comp>[xyz]) \[(?P<unit>.+)\]$")

    by_prefix: dict[tuple[str, str], dict[str, str]] = {}
    for name in variable_names:
        m = pattern.match(name)
        if not m:
            continue
        prefix = m.group("prefix")
        comp = m.group("comp")
        unit = m.group("unit")
        slot = by_prefix.setdefault((prefix, unit), {})
        slot[comp] = name

    wanted = set(prefixes) if prefixes is not None else None
    merged = griblet.ComputationGraph()
    for (prefix, unit), info in sorted(by_prefix.items()):
        if wanted is not None and prefix not in wanted:
            continue
        if not {"x", "y", "z"}.issubset(info):
            continue
        merged.merge(
            build_griblet_vector_spherical_components_graph(
                prefix=prefix,
                unit=unit,
                coord_fields=coord_fields,
                register_components=components,
            )
        )
    return merged


def _infer_radius_name_from_coord(x_name: str) -> str | None:
    m = re.match(r"^X \[(.+)\]$", x_name)
    if m:
        return f"R [{m.group(1)}]"
    return None
