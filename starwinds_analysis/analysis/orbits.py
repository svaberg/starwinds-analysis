"""THIS FILE contains orbit geometry and 1D-curve sampling primitives.

It provides ellipse-based orbit paths and SmartDs resampling along those paths.
The circular case is the `eccentricity=0` case. This is analysis/sampling code
(not local physics formulas).
"""

from __future__ import annotations

import math

import numpy as np

from starwinds_analysis.recipes.batsrus import build_griblet_vector_cartesian_graph

def _kepler_eccentric_anomaly(mean_anomaly_rad, eccentricity, *, max_iter: int = 20):
    """
    Solve `E - e sin(E) = M` for `E` with vectorized Newton iterations.
    """
    e = float(eccentricity)
    if not (0.0 <= e < 1.0):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    m = np.array(mean_anomaly_rad)
    e_anom = np.mod(m, 2.0 * math.pi)
    if e == 0.0:
        return e_anom
    for _ in range(max_iter):
        f = e_anom - e * np.sin(e_anom) - np.mod(m, 2.0 * math.pi)
        fp = 1.0 - e * np.cos(e_anom)
        step = np.divide(f, fp, out=np.zeros_like(f), where=fp != 0)
        e_anom = e_anom - step
        if np.all(np.abs(step) < 1e-12):
            break
    return e_anom

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

def elliptic_orbit_points(
    semi_major_axis,
    *,
    eccentricity: float = 0.0,
    n_points: int = 360,
    plane: str = "xy",
    angle0: float = 0.0,
    phase0: float = 0.0,
    center=(0.0, 0.0, 0.0),
    sample: str = "eccentric_anomaly",
    return_info: bool = False,
):
    """
    Cartesian points on a Kepler ellipse (same coordinate unit as `semi_major_axis`).
    The circular case is `eccentricity=0`.
    Used by: `starwinds_analysis/physics/orbit_surface.py`
    """
    a = float(semi_major_axis)
    e = float(eccentricity)
    if a <= 0:
        raise ValueError("semi_major_axis must be > 0")
    if not (0.0 <= e < 1.0):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if n_points < 8:
        raise ValueError("n_points must be >= 8")

    if sample == "eccentric_anomaly":
        e_anom = (
            np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False) + float(phase0)
        )
        mean_anom = e_anom - e * np.sin(e_anom)
        weights = 1.0 - e * np.cos(e_anom)
    elif sample == "mean_anomaly":
        mean_anom = (
            np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False) + float(phase0)
        )
        e_anom = _kepler_eccentric_anomaly(mean_anom, e)
        weights = np.ones_like(mean_anom)
    else:
        raise ValueError("sample must be 'eccentric_anomaly' or 'mean_anomaly'")

    cos_e = np.cos(e_anom)
    sin_e = np.sin(e_anom)
    b = a * math.sqrt(max(0.0, 1.0 - e * e))
    x_plane = a * (cos_e - e)
    y_plane = b * sin_e

    c0 = math.cos(float(angle0))
    s0 = math.sin(float(angle0))
    x_rot = c0 * x_plane - s0 * y_plane
    y_rot = s0 * x_plane + c0 * y_plane
    cx, cy, cz = map(float, center)
    points = np.empty((x_rot.size, 3), dtype=float)
    if plane == "xy":
        points[:, 0] = cx + x_rot
        points[:, 1] = cy + y_rot
        points[:, 2] = cz
    elif plane == "xz":
        points[:, 0] = cx + x_rot
        points[:, 1] = cy
        points[:, 2] = cz + y_rot
    elif plane == "yz":
        points[:, 0] = cx
        points[:, 1] = cy + x_rot
        points[:, 2] = cz + y_rot
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    radius = a * (1.0 - e * cos_e)
    true_anom = 2.0 * np.arctan2(
        math.sqrt(1.0 + e) * np.sin(e_anom / 2.0),
        math.sqrt(1.0 - e) * np.cos(e_anom / 2.0),
    )
    time_weight = np.array(weights)
    sw = np.sum(time_weight)
    if sw > 0:
        time_weight = time_weight / sw
    if time_weight.ndim != 1 or time_weight.size == 0:
        phase = np.array([])
    else:
        phase_weight = np.where(np.isfinite(time_weight) & (time_weight >= 0), time_weight, 0.0)
        sw_phase = float(np.sum(phase_weight))
        if sw_phase <= 0:
            phase = np.arange(phase_weight.size, dtype=float) / max(1, phase_weight.size)
        else:
            phase_weight = phase_weight / sw_phase
            phase = np.empty(phase_weight.size, dtype=float)
            phase[0] = 0.0
            if phase_weight.size > 1:
                phase[1:] = np.cumsum(phase_weight[:-1])

    if not return_info:
        return points

    return {
        "points": points,
        "phase [turns]": phase,
        "time_weight [none]": time_weight,
        "eccentric_anomaly [rad]": e_anom,
        "mean_anomaly [rad]": mean_anom,
        "true_anomaly [rad]": true_anom,
        "radius [R]": radius,
        "semi_major_axis [R]": float(a),
        "eccentricity [none]": float(e),
        "plane": plane,
        "sample": sample,
    }

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
    points = np.array(points)
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
    has_velocity = velocity_xyz is not None
    if has_velocity:
        velocity = np.array(velocity_xyz)
        if velocity.shape != (sampled_curve.points.shape[0], 3):
            raise ValueError("velocity_xyz must have shape (n_points, 3)")
        context_fields["V_x [m/s]"] = velocity[:, 0]
        context_fields["V_y [m/s]"] = velocity[:, 1]
        context_fields["V_z [m/s]"] = velocity[:, 2]
    sampled_curve = sampled_curve.append_fields(context_fields, zone_suffix="trajectory")
    if has_velocity:
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
    Sample requested fields along an elliptic orbit path and return a curve SmartDs.
    The circular case is `eccentricity=0`.
    Used by: `starwinds_analysis/physics/curve.py`
    """
    orbit_info = elliptic_orbit_points(
        semi_major_axis,
        eccentricity=eccentricity,
        n_points=n_points,
        plane=plane,
        angle0=angle0,
        phase0=phase0,
        center=center,
        sample=sample,
        return_info=True,
    )
    sampled_curve = sample_curve(
        smart_ds,
        orbit_info["points"],
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )
    context_fields = {
        "phase [turns]": orbit_info["phase [turns]"],
        "time_weight [none]": orbit_info["time_weight [none]"],
        "true_anomaly [rad]": orbit_info["true_anomaly [rad]"],
        "mean_anomaly [rad]": orbit_info["mean_anomaly [rad]"],
        "eccentric_anomaly [rad]": orbit_info["eccentric_anomaly [rad]"],
    }
    sampled_curve = sampled_curve.append_fields(
        context_fields,
        zone_suffix="elliptic orbit",
    )
    sampled_curve.raw.aux["orbit_kind"] = "elliptic"
    sampled_curve.raw.aux["orbit_plane"] = plane
    sampled_curve.raw.aux["orbit_sample_parameter"] = sample
    sampled_curve.raw.aux["orbit_semi_major_axis_R"] = float(semi_major_axis)
    sampled_curve.raw.aux["orbit_eccentricity"] = float(eccentricity)
    return sampled_curve
