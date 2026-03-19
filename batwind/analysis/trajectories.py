"""Trajectory geometry and 1D-curve sampling primitives.
"""

# It provides circular orbit paths and SmartDs resampling along those paths.
# This is analysis/sampling code (not local physics formulas).


from __future__ import annotations

import logging

import numpy as np

from batwind.data.field_names import DEFAULT_XYZ_NAMES
from batwind.recipes.vectors import build_vector_graph

log = logging.getLogger(__name__)

def trajectory_velocity(points, time, *, coordinate_scale: float = 1.0):
    """
    Velocity from explicit trajectory points and strictly increasing sample times.
    Used by: `examples/orbit_surface_revolution.ipynb`
    """
    pts = np.array(points) * float(coordinate_scale)
    t = np.array(time)
    if pts.ndim != 2 or pts.shape[1] != 3:
        log.error("trajectory_velocity failed: points shape=%s", pts.shape)
        raise ValueError("points must have shape (n, 3)")
    if t.ndim != 1 or t.shape[0] != pts.shape[0]:
        log.error("trajectory_velocity failed: time shape=%s points=%s", t.shape, pts.shape)
        raise ValueError("time must have shape (n,)")
    if pts.shape[0] < 2:
        log.warning("trajectory_velocity returned NaN: fewer than 2 points")
        return np.full_like(pts, np.nan, dtype=float)

    dt = np.diff(t)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        log.error("trajectory_velocity failed: time is not strictly increasing and finite")
        raise ValueError("time must be strictly increasing and finite")

    edge_order = 2 if pts.shape[0] > 2 else 1
    log.debug("trajectory_velocity edge_order=%d n_points=%d", edge_order, pts.shape[0])
    return np.gradient(pts, t, axis=0, edge_order=edge_order)

def circular_orbit_points(
    radius,
    *,
    n_points: int = 360,
):
    """
    Cartesian points on a circular orbit in the XY plane.
    Used by: `batwind/physics/orbit_surface.py`
    """
    radius = float(radius)
    if radius <= 0:
        log.error("circular_orbit_points failed: radius=%g", radius)
        raise ValueError("radius must be > 0")

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)
    points = np.empty((theta.size, 3), dtype=float)
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = 0.0
    log.debug("circular_orbit_points radius=%g n_points=%d", radius, points.shape[0])
    return points

def sample_curve(
    smart_ds,
    points,
    *,
    fields,
    coordinate_fields=DEFAULT_XYZ_NAMES,
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Resample `fields` onto explicit Cartesian points and return a curve SmartDs.
    Used by: `batwind/physics/orbit_surface.py`
    """
    points = np.array(points)
    log.info("sample_curve...")
    requested_fields = tuple(dict.fromkeys(fields))
    base_fields = smart_ds.source_fields(requested_fields)
    log.debug(
        "sample_curve method=%s n_points=%d requested_fields=%d source_fields=%d",
        method,
        points.shape[0],
        len(requested_fields),
        len(base_fields),
    )
    sampled_curve = smart_ds.resample(
        points,
        coordinate_fields=coordinate_fields,
        fields=base_fields,
        method=method,
        fill_value=fill_value,
        zone="orbit-samples",
    )
    log.debug("sample_curve complete")
    return sampled_curve

def sample_trajectory(
    smart_ds,
    points,
    *,
    fields,
    time,
    velocity_xyz=None,
    coordinate_fields=DEFAULT_XYZ_NAMES,
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Resample `fields` onto explicit Cartesian trajectory points and append `t` and optional `V`.
    The returned SmartDs exposes `V_xyz` via graph recipes when `velocity_xyz` is provided.
    """
    log.info("sample_trajectory...")
    sampled_curve = sample_curve(
        smart_ds,
        points,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )

    time = np.array(time)
    if time.ndim != 1 or time.shape[0] != sampled_curve.raw.points.shape[0]:
        log.error(
            "sample_trajectory failed: time shape=%s expected=(%d,)",
            time.shape,
            sampled_curve.raw.points.shape[0],
        )
        raise ValueError("time must have shape (n_points,)")

    context_fields = {"t [s]": time}
    if velocity_xyz is None:
        log.debug("sample_trajectory complete without velocity fields")
        return sampled_curve.append_fields(context_fields, zone_suffix="trajectory")

    velocity = np.array(velocity_xyz)
    if velocity.shape != (sampled_curve.raw.points.shape[0], 3):
        log.error(
            "sample_trajectory failed: velocity shape=%s expected=(%d, 3)",
            velocity.shape,
            sampled_curve.raw.points.shape[0],
        )
        raise ValueError("velocity_xyz must have shape (n_points, 3)")
    context_fields["V_x [m/s]"] = velocity[:, 0]
    context_fields["V_y [m/s]"] = velocity[:, 1]
    context_fields["V_z [m/s]"] = velocity[:, 2]
    sampled_curve = sampled_curve.append_fields(context_fields, zone_suffix="trajectory")
    sampled_curve.merge_computation_graph(build_vector_graph(sampled_curve.raw.variables))
    log.debug("sample_trajectory attached V_xyz graph from appended velocity components")
    log.debug("sample_trajectory complete with velocity fields")
    return sampled_curve
