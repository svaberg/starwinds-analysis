"""THIS FILE contains orbit geometry and 1D-curve sampling primitives.

It provides circular/elliptic orbit paths and SmartDs resampling along those
paths. This is analysis/sampling code (not local physics formulas).
"""

from __future__ import annotations

import math

import numpy as np

def circular_orbit_points(
    radius,
    *,
    n_points: int = 360,
    plane: str = "xy",
    phase0: float = 0.0,
    center=(0.0, 0.0, 0.0),
):
    """
    Cartesian points on a circular orbit (same coordinate unit as `radius`).
    Used by: `test/test_orbit_analysis.py`, `starwinds_analysis/analysis/orbits.py`,
      `starwinds_analysis/physics/orbit_surface.py`
    """
    return elliptic_orbit_points(
        radius,
        eccentricity=0.0,
        n_points=n_points,
        plane=plane,
        phase0=phase0,
        center=center,
        sample="eccentric_anomaly",
        return_info=False,
    )

def _kepler_eccentric_anomaly(mean_anomaly_rad, eccentricity, *, max_iter: int = 20):
    """
    Solve `E - e sin(E) = M` for `E` with vectorized Newton iterations.
    Used by: `starwinds_analysis/analysis/orbits.py`
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

def _embed_plane_coords(x, y, *, plane: str, center=(0.0, 0.0, 0.0)):
    """
    Embed 2D orbit-plane coordinates into 3D (`xy`, `xz`, `yz`) Cartesian coordinates.
    Used by: `starwinds_analysis/analysis/orbits.py`
    """
    cx, cy, cz = map(float, center)
    pts = np.empty((x.size, 3), dtype=float)
    if plane == "xy":
        pts[:, 0] = cx + x
        pts[:, 1] = cy + y
        pts[:, 2] = cz
    elif plane == "xz":
        pts[:, 0] = cx + x
        pts[:, 1] = cy
        pts[:, 2] = cz + y
    elif plane == "yz":
        pts[:, 0] = cx
        pts[:, 1] = cy + x
        pts[:, 2] = cz + y
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    return pts

def _phase_from_weights(weights):
    """
    Convert periodic sample weights into cumulative phase turns.
    Used by: `starwinds_analysis/analysis/orbits.py`
    """
    w = np.array(weights)
    if w.ndim != 1 or w.size == 0:
        return np.array([])
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    sw = float(np.sum(w))
    if sw <= 0:
        return np.arange(w.size, dtype=float) / max(1, w.size)
    w = w / sw
    phase = np.empty(w.size, dtype=float)
    phase[0] = 0.0
    if w.size > 1:
        phase[1:] = np.cumsum(w[:-1])
    return phase


def periodic_curve_velocity(points_r, phase_turns, period_s, body_radius_m):
    """
    Velocity along a periodic sampled curve from point positions and phase turns.
    Used by: `starwinds_analysis/physics/orbit_pressure.py`, `starwinds_analysis/physics/orbit_surface.py`
    """
    points = np.array(points_r) * float(body_radius_m)
    phase = np.array(phase_turns)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_r must have shape (n, 3)")
    if points.shape[0] < 3:
        return np.full_like(points, np.nan, dtype=float)
    t = phase * float(period_s)
    p_prev = np.roll(points, 1, axis=0)
    p_next = np.roll(points, -1, axis=0)
    t_prev = np.roll(t, 1)
    t_next = np.roll(t, -1)
    dt_prev = t - t_prev
    dt_next = t_next - t
    dt_prev[0] += float(period_s)
    dt_next[-1] += float(period_s)
    denom = dt_prev + dt_next
    return np.divide(
        p_next - p_prev,
        denom[:, None],
        out=np.full_like(points, np.nan, dtype=float),
        where=denom[:, None] != 0,
    )

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
    Used by: `test/test_orbit_analysis.py`, `starwinds_analysis/analysis/orbits.py`,
      `starwinds_analysis/physics/orbit_surface.py`
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
    points = _embed_plane_coords(x_rot, y_rot, plane=plane, center=center)

    radius = a * (1.0 - e * cos_e)
    true_anom = 2.0 * np.arctan2(
        math.sqrt(1.0 + e) * np.sin(e_anom / 2.0),
        math.sqrt(1.0 - e) * np.cos(e_anom / 2.0),
    )
    time_weight = np.array(weights)
    sw = np.sum(time_weight)
    if sw > 0:
        time_weight = time_weight / sw
    phase = _phase_from_weights(time_weight)

    if not return_info:
        return points

    return {
        "points": points,
        "phase [turns]": phase,
        "time_weight [none]": time_weight,
        "eccentric_anomaly [rad]": e_anom,
        "mean_anomaly [rad]": mean_anom,
        "true_anomaly [rad]": true_anom,
        "radius [orbit]": radius,
        "semi_major_axis [orbit]": float(a),
        "eccentricity [orbit]": float(e),
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
    Used by: `starwinds_analysis/analysis/orbits.py`, `starwinds_analysis/physics/orbit_surface.py`
    """
    points = np.array(points)
    sampled = smart_ds.resample(
        points,
        coordinate_fields=coordinate_fields,
        fields=tuple(dict.fromkeys(fields)),
        method=method,
        fill_value=fill_value,
        zone="orbit-samples",
    )
    return sampled.append_fields(
        {
            "X [sample]": points[:, 0],
            "Y [sample]": points[:, 1],
            "Z [sample]": points[:, 2],
            "R [sample]": np.sqrt(np.sum(points * points, axis=1)),
        },
        zone_suffix="orbit samples",
    )

def sample_trajectory(
    smart_ds,
    points,
    *,
    fields,
    time_s,
    velocity_xyz_m_s=None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Resample `fields` onto explicit Cartesian trajectory points and append `t` and optional `V_xyz`.
    Used by: no external call sites found
    """
    curve = sample_curve(
        smart_ds,
        points,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )
    time = np.array(time_s)
    if time.ndim != 1 or time.shape[0] != curve.points.shape[0]:
        raise ValueError("time_s must have shape (n_points,)")
    extra_fields = {"t [s]": time}
    if velocity_xyz_m_s is not None:
        velocity = np.array(velocity_xyz_m_s)
        if velocity.shape != (curve.points.shape[0], 3):
            raise ValueError("velocity_xyz_m_s must have shape (n_points, 3)")
        extra_fields["V_x [m/s]"] = velocity[:, 0]
        extra_fields["V_y [m/s]"] = velocity[:, 1]
        extra_fields["V_z [m/s]"] = velocity[:, 2]
    return curve.append_fields(extra_fields, zone_suffix="trajectory")

def sample_circular_orbit(
    smart_ds,
    radius,
    *,
    fields,
    n_points: int = 360,
    plane: str = "xy",
    phase0: float = 0.0,
    center=(0.0, 0.0, 0.0),
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Sample requested fields along a circular orbit path and return a curve SmartDs.
    Used by: `test/test_orbit_analysis.py`, `starwinds_analysis/physics/orbit_local.py`,
      `starwinds_analysis/physics/orbit_pressure.py`
    """
    return sample_elliptic_orbit(
        smart_ds,
        radius,
        eccentricity=0.0,
        fields=fields,
        n_points=n_points,
        plane=plane,
        angle0=0.0,
        phase0=phase0,
        center=center,
        sample="eccentric_anomaly",
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )

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
    Used by: `test/test_orbit_analysis.py`, `starwinds_analysis/physics/orbit_local.py`,
      `starwinds_analysis/physics/orbit_pressure.py`
    """
    info = elliptic_orbit_points(
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
    sampled = sample_curve(
        smart_ds,
        info["points"],
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )
    sampled = sampled.append_fields(
        {
            "phase [turns]": info["phase [turns]"],
            "time_weight [none]": info["time_weight [none]"],
            "true_anomaly [rad]": info["true_anomaly [rad]"],
            "mean_anomaly [rad]": info["mean_anomaly [rad]"],
            "eccentric_anomaly [rad]": info["eccentric_anomaly [rad]"],
        },
        zone_suffix="elliptic orbit",
    )
    sampled.raw.aux["orbit_kind"] = "elliptic"
    sampled.raw.aux["orbit_plane"] = plane
    sampled.raw.aux["orbit_sample_parameter"] = sample
    sampled.raw.aux["orbit_semi_major_axis_R"] = float(semi_major_axis)
    sampled.raw.aux["orbit_eccentricity"] = float(eccentricity)
    return sampled
