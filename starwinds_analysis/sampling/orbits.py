"""THIS FILE contains orbit geometry and orbit sampling primitives.

It provides circular/elliptic paths and SmartDs resampling along those paths.
It is a neutral sampling layer (not `analysis`, not `physics`).
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
    """
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be > 0")
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    t = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False) + float(phase0)
    c = np.cos(t)
    s = np.sin(t)
    cx, cy, cz = map(float, center)

    pts = np.empty((n_points, 3), dtype=float)
    if plane == "xy":
        pts[:, 0] = cx + r * c
        pts[:, 1] = cy + r * s
        pts[:, 2] = cz
    elif plane == "xz":
        pts[:, 0] = cx + r * c
        pts[:, 1] = cy
        pts[:, 2] = cz + r * s
    elif plane == "yz":
        pts[:, 0] = cx
        pts[:, 1] = cy + r * c
        pts[:, 2] = cz + r * s
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    return pts

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

def _embed_plane_coords(x, y, *, plane: str, center=(0.0, 0.0, 0.0)):
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

    `sample="eccentric_anomaly"` gives uniform geometric spacing and non-uniform time
    weights (`time_weight [none]`). `sample="mean_anomaly"` gives near-uniform time
    spacing with uniform weights.
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

def sample_points(
    smart_ds,
    points,
    *,
    fields,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Resample `fields` onto explicit Cartesian points.
    """
    points = np.array(points)
    out = smart_ds.resample(
        points,
        coordinate_fields=coordinate_fields,
        fields=tuple(dict.fromkeys(fields)),
        method=method,
        fill_value=fill_value,
        zone="orbit-samples",
    )
    data = {name: np.array(out.variable(name)) for name in fields}
    data["X [sample]"] = points[:, 0]
    data["Y [sample]"] = points[:, 1]
    data["Z [sample]"] = points[:, 2]
    data["R [sample]"] = np.sqrt(np.sum(points * points, axis=1))
    return data

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
    pts = circular_orbit_points(
        radius, n_points=n_points, plane=plane, phase0=phase0, center=center
    )
    sampled = sample_points(
        smart_ds,
        pts,
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )
    sampled["plane"] = plane
    sampled["phase [turns]"] = np.arange(n_points, dtype=float) / max(1, n_points)
    sampled["time_weight [none]"] = np.full(n_points, 1.0 / n_points, dtype=float)
    sampled["kind"] = "circular"
    return sampled

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
    sampled = sample_points(
        smart_ds,
        info["points"],
        fields=fields,
        coordinate_fields=coordinate_fields,
        method=method,
        fill_value=fill_value,
    )
    sampled["plane"] = plane
    sampled["kind"] = "elliptic"
    sampled["phase [turns]"] = info["phase [turns]"]
    sampled["time_weight [none]"] = info["time_weight [none]"]
    sampled["true_anomaly [rad]"] = info["true_anomaly [rad]"]
    sampled["mean_anomaly [rad]"] = info["mean_anomaly [rad]"]
    sampled["eccentric_anomaly [rad]"] = info["eccentric_anomaly [rad]"]
    sampled["semi_major_axis [sample]"] = float(semi_major_axis)
    sampled["eccentricity [sample]"] = float(eccentricity)
    sampled["sample_parameter"] = sample
    return sampled

