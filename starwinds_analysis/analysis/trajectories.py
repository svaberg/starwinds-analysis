"""Trajectory geometry and 1D-curve sampling primitives.
"""

# It provides circular orbit paths and SmartDs resampling along those paths.
# This is analysis/sampling code (not local physics formulas).


from __future__ import annotations

import numpy as np

from starwinds_analysis.recipes.batsrus import build_griblet_vector_cartesian_graph

def trajectory_velocity(points, time, *, coordinate_scale: float = 1.0):
    """
    Velocity from explicit trajectory points and strictly increasing sample times.
    Used by: `examples/orbit_surface_revolution.ipynb`
    """
    pts = np.array(points) * float(coordinate_scale)
    t = np.array(time)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if t.ndim != 1 or t.shape[0] != pts.shape[0]:
        raise ValueError("time must have shape (n,)")
    if pts.shape[0] < 2:
        return np.full_like(pts, np.nan, dtype=float)

    dt = np.diff(t)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("time must be strictly increasing and finite")

    edge_order = 2 if pts.shape[0] > 2 else 1
    return np.gradient(pts, t, axis=0, edge_order=edge_order)

def circular_orbit_points(
    radius,
    *,
    n_points: int = 360,
):
    """
    Cartesian points on a circular orbit in the XY plane.
    Used by: `starwinds_analysis/physics/orbit_surface.py`
    """
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be > 0")

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)
    points = np.empty((theta.size, 3), dtype=float)
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = 0.0
    return points

def sample_curve(
    smart_ds,
    points,
    *,
    fields,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Resample `fields` onto explicit Cartesian points and return a curve SmartDs.
    Used by: `starwinds_analysis/physics/orbit_surface.py`
    """
    points = np.array(points)
    requested_fields = tuple(dict.fromkeys(fields))
    base_fields = smart_ds.base_fields_for_resample(requested_fields)
    sampled_curve = smart_ds.resample(
        points,
        coordinate_fields=coordinate_fields,
        fields=base_fields,
        method=method,
        fill_value=fill_value,
        zone="orbit-samples",
    )
    return sampled_curve

def sample_trajectory(
    smart_ds,
    points,
    *,
    fields,
    time,
    velocity_xyz=None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Resample `fields` onto explicit Cartesian trajectory points and append `t` and optional `V`.
    The returned SmartDs exposes `V_xyz` via graph recipes when `velocity_xyz` is provided.
    """
    sampled_curve = sample_curve(
        smart_ds,
        points,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )

    time = np.array(time)
    if time.ndim != 1 or time.shape[0] != sampled_curve.points.shape[0]:
        raise ValueError("time must have shape (n_points,)")

    context_fields = {"t [s]": time}
    if velocity_xyz is None:
        return sampled_curve.append_fields(context_fields, zone_suffix="trajectory")

    velocity = np.array(velocity_xyz)
    if velocity.shape != (sampled_curve.points.shape[0], 3):
        raise ValueError("velocity_xyz must have shape (n_points, 3)")
    context_fields["V_x [m/s]"] = velocity[:, 0]
    context_fields["V_y [m/s]"] = velocity[:, 1]
    context_fields["V_z [m/s]"] = velocity[:, 2]
    sampled_curve = sampled_curve.append_fields(context_fields, zone_suffix="trajectory")
    sampled_curve.set_computation_graph(
        build_griblet_vector_cartesian_graph(sampled_curve.variables),
        merge=True,
    )
    return sampled_curve

def sample_elliptic_orbit(
    smart_ds,
    semi_major_axis,
    *,
    eccentricity: float = 0.0,
    fields,
    n_points: int = 360,
    plane: str = "xy",
    angle0: float = 0.0,
    phase0: float = 0.0,
    center=(0.0, 0.0, 0.0),
    sample: str = "eccentric_anomaly",
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Sample requested fields along a circular XY orbit path and return a curve SmartDs.
    This legacy function name only supports the circular case.
    Used by: `starwinds_analysis/physics/curve.py`
    """
    if float(eccentricity) != 0.0:
        raise ValueError("sample_elliptic_orbit supports only eccentricity=0 in trajectories.py")
    if plane != "xy":
        raise ValueError("sample_elliptic_orbit supports only plane='xy' in trajectories.py")
    if float(angle0) != 0.0:
        raise ValueError("sample_elliptic_orbit supports only angle0=0 in trajectories.py")
    if float(phase0) != 0.0:
        raise ValueError("sample_elliptic_orbit supports only phase0=0 in trajectories.py")
    if tuple(center) != (0.0, 0.0, 0.0):
        raise ValueError("sample_elliptic_orbit supports only center=(0,0,0) in trajectories.py")
    if sample != "eccentric_anomaly":
        raise ValueError("sample_elliptic_orbit supports only sample='eccentric_anomaly'")

    points = circular_orbit_points(semi_major_axis, n_points=n_points)
    phase = np.arange(points.shape[0], dtype=float) / float(points.shape[0])
    time_weight = np.full(points.shape[0], 1.0 / float(points.shape[0]), dtype=float)
    sampled_curve = sample_curve(
        smart_ds,
        points,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )
    context_fields = {
        "phase [turns]": phase,
        "time_weight [none]": time_weight,
    }
    sampled_curve = sampled_curve.append_fields(
        context_fields,
        zone_suffix="circular orbit",
    )
    sampled_curve.raw.aux["orbit_kind"] = "circular"
    sampled_curve.raw.aux["orbit_plane"] = "xy"
    sampled_curve.raw.aux["orbit_radius_R"] = float(semi_major_axis)
    return sampled_curve
