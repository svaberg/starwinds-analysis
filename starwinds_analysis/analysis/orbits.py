"""THIS FILE contains orbit geometry, orbit sampling, and orbit comparison workflows.

It provides circular/elliptic paths and SmartDs resampling along those paths.
It should not own low-level pressure or torque formulas.
"""

# TODO(debt): This module mixes generic orbit geometry/sampling with quantity-specific
# comparison workflows (`local_mass_loss_*`, `local_torque_*`) and imports from
# `physics`, which is a reversed layer direction under current rules.
# TODO(debt): Split reusable orbit geometry/sampling primitives from one-off or
# quantity-specific comparison workflows.

from __future__ import annotations

import math

import numpy as np
from scipy.constants import G as GRAVITATIONAL_CONSTANT

from starwinds_analysis.physics.local_estimates import (
    local_mass_loss_estimates,
    local_torque_estimates,
    summarize_samples,
)
from starwinds_analysis.physics.mass_loss import mass_loss_vs_radius
from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    resolve_batsrus_density_si,
    resolve_batsrus_vector_xyz_si,
)
from starwinds_analysis.physics.shell_torque import torque_vs_radius
from starwinds_analysis.recipes.spherical import radial_component, spherical_vector_components


def orbital_period(semi_major_axis_m, star_mass_kg):
    """
    Keplerian orbital period for a test particle around a point mass.
    """
    a = float(semi_major_axis_m)
    m = float(star_mass_kg)
    if a <= 0:
        raise ValueError("semi_major_axis_m must be > 0")
    if m <= 0:
        raise ValueError("star_mass_kg must be > 0")
    return 2.0 * math.pi * math.sqrt(a**3 / (GRAVITATIONAL_CONSTANT * m))


def orbital_velocity(radial_distance_m, star_mass_kg, semi_major_axis_m):
    """
    Vis-viva orbital speed.
    """
    r = np.array(radial_distance_m, dtype=float)
    m = float(star_mass_kg)
    a = float(semi_major_axis_m)
    if m <= 0:
        raise ValueError("star_mass_kg must be > 0")
    if a <= 0:
        raise ValueError("semi_major_axis_m must be > 0")
    with np.errstate(invalid="ignore"):
        return np.sqrt(GRAVITATIONAL_CONSTANT * m * (2.0 / r - 1.0 / a))


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
    m = np.array(mean_anomaly_rad, dtype=float)
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
    w = np.array(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        return np.array([], dtype=float)
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
    time_weight = np.array(weights, dtype=float)
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
    points = np.array(points, dtype=float)
    out = smart_ds.resample(
        points,
        coordinate_fields=coordinate_fields,
        fields=tuple(dict.fromkeys(fields)),
        method=method,
        fill_value=fill_value,
        zone="orbit-samples",
    )
    data = {name: np.array(out.variable(name), dtype=float) for name in fields}
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


def _interp_profile(radii, values, x):
    r = np.array(radii, dtype=float)
    y = np.array(values, dtype=float)
    x = np.array(x, dtype=float)
    good = np.isfinite(r) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        return np.full_like(x, np.nan, dtype=float)
    r = r[good]
    y = y[good]
    order = np.argsort(r)
    r = r[order]
    y = y[order]
    out = np.interp(x, r, y, left=np.nan, right=np.nan)
    out[(x < r[0]) | (x > r[-1])] = np.nan
    return out


def _local_mass_loss_from_orbit_sample(
    smart_ds,
    orbit,
    *,
    body_radius_m,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")

    x = orbit["X [sample]"]
    y = orbit["Y [sample]"]
    z = orbit["Z [sample]"]
    rho = rho_scale * orbit[rho_name]
    ux = u_scale * orbit[ux_name]
    uy = u_scale * orbit[uy_name]
    uz = u_scale * orbit[uz_name]
    u_r = radial_component(ux, uy, uz, x, y, z)
    r_sample_r = np.array(orbit["R [sample]"], dtype=float)
    r_m = r_sample_r * body_radius_m
    estimates = local_mass_loss_estimates(r_m, rho, u_r)
    weights = orbit.get("time_weight [none]")
    summary = summarize_samples(estimates, weights=weights)

    if shell_radii is None:
        shell = mass_loss_vs_radius(
            smart_ds,
            [float(np.nanmean(r_sample_r))],
            body_radius_m=body_radius_m,
            n_polar=shell_n_polar,
            n_azimuth=shell_n_azimuth,
            method=method,
        )
        shell_value = float(shell["mass_loss [kg/s]"][0])
        shell_interp = np.full_like(estimates, shell_value, dtype=float)
    else:
        shell_radii = np.array(shell_radii, dtype=float)
        shell = mass_loss_vs_radius(
            smart_ds,
            shell_radii,
            body_radius_m=body_radius_m,
            n_polar=shell_n_polar,
            n_azimuth=shell_n_azimuth,
            method=method,
        )
        shell_interp = _interp_profile(
            shell["radius [R]"], shell["mass_loss [kg/s]"], r_sample_r
        )
        shell_value = summarize_samples(shell_interp, weights=weights)["mean"]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to_shell = summary["mean"] / shell_value if shell_value != 0 else np.nan

    out = {
        "radius [R]": float(np.nanmean(r_sample_r)),
        "radius [m]": float(np.nanmean(r_m)),
        "u_r [m/s]": u_r,
        "rho [kg/m^3]": rho,
        "local_mass_loss [kg/s]": estimates,
        "summary": summary,
        "shell_mass_loss [kg/s]": float(shell_value),
        "mean_to_shell [none]": float(mean_to_shell),
        "orbit_samples": orbit,
        "shell_profile": shell,
    }
    if shell_radii is not None:
        out["shell_mass_loss_interp [kg/s]"] = shell_interp
    return out


def _local_torque_from_orbit_sample(
    smart_ds,
    orbit,
    *,
    body_radius_m,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")
    (bx_name, by_name, bz_name), b_scale = resolve_batsrus_vector_xyz_si(smart_ds, "B")

    x = orbit["X [sample]"]
    y = orbit["Y [sample]"]
    z = orbit["Z [sample]"]
    r_sample_r = np.array(orbit["R [sample]"], dtype=float)
    r_m = r_sample_r * body_radius_m
    rho = rho_scale * orbit[rho_name]
    ux = u_scale * orbit[ux_name]
    uy = u_scale * orbit[uy_name]
    uz = u_scale * orbit[uz_name]
    bx = b_scale * orbit[bx_name]
    by = b_scale * orbit[by_name]
    bz = b_scale * orbit[bz_name]

    u_r, _u_theta, u_phi = spherical_vector_components(ux, uy, uz, x, y, z)
    b_r, _b_theta, b_phi = spherical_vector_components(bx, by, bz, x, y, z)
    local = local_torque_estimates(r_m, rho, u_r, u_phi, b_r, b_phi)
    weights = orbit.get("time_weight [none]")

    if shell_radii is None:
        shell = torque_vs_radius(
            smart_ds,
            [float(np.nanmean(r_sample_r))],
            body_radius_m=body_radius_m,
            n_polar=shell_n_polar,
            n_azimuth=shell_n_azimuth,
            method=method,
        )
        shell_total = float(shell["total_torque [Nm]"][0])
        shell_interp = np.full_like(local["total [Nm]"], shell_total, dtype=float)
    else:
        shell_radii = np.array(shell_radii, dtype=float)
        shell = torque_vs_radius(
            smart_ds,
            shell_radii,
            body_radius_m=body_radius_m,
            n_polar=shell_n_polar,
            n_azimuth=shell_n_azimuth,
            method=method,
        )
        shell_interp = _interp_profile(
            shell["radius [R]"], shell["total_torque [Nm]"], r_sample_r
        )
        shell_total = summarize_samples(shell_interp, weights=weights)["mean"]

    summary = summarize_samples(local["total [Nm]"], weights=weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to_shell = summary["mean"] / shell_total if shell_total != 0 else np.nan

    out = {
        "radius [R]": float(np.nanmean(r_sample_r)),
        "radius [m]": float(np.nanmean(r_m)),
        "rho [kg/m^3]": rho,
        "u_r [m/s]": u_r,
        "u_phi [m/s]": u_phi,
        "b_r [T]": b_r,
        "b_phi [T]": b_phi,
        "local_magnetic_torque [Nm]": local["magnetic [Nm]"],
        "local_dynamic_torque [Nm]": local["dynamic [Nm]"],
        "local_total_torque [Nm]": local["total [Nm]"],
        "summary": summary,
        "shell_total_torque [Nm]": float(shell_total),
        "mean_to_shell [none]": float(mean_to_shell),
        "orbit_samples": orbit,
        "shell_profile": shell,
    }
    if shell_radii is not None:
        out["shell_total_torque_interp [Nm]"] = shell_interp
    return out


def local_mass_loss_on_circular_orbit(
    smart_ds,
    radius,
    *,
    body_radius_m: float | None = None,
    n_points: int = 360,
    plane: str = "xy",
    method: str = "nearest",
    shell_n_polar: int = 24,
    shell_n_azimuth: int = 48,
):
    """
    Local mass-loss estimates along a circular orbit plus shell comparison.
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(
            resolve_batsrus_density_si(smart_ds)[0],
            *resolve_batsrus_vector_xyz_si(smart_ds, "U")[0],
        ),
        n_points=n_points,
        plane=plane,
        method=method,
    )
    return _local_mass_loss_from_orbit_sample(
        smart_ds,
        orbit,
        body_radius_m=body_radius_m,
        method=method,
        shell_n_polar=shell_n_polar,
        shell_n_azimuth=shell_n_azimuth,
        shell_radii=None,
    )


def local_torque_on_circular_orbit(
    smart_ds,
    radius,
    *,
    body_radius_m: float | None = None,
    n_points: int = 360,
    plane: str = "xy",
    method: str = "nearest",
    shell_n_polar: int = 24,
    shell_n_azimuth: int = 48,
):
    """
    Local torque estimates along a circular orbit plus shell comparison.
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(
            resolve_batsrus_density_si(smart_ds)[0],
            *resolve_batsrus_vector_xyz_si(smart_ds, "U")[0],
            *resolve_batsrus_vector_xyz_si(smart_ds, "B")[0],
        ),
        n_points=n_points,
        plane=plane,
        method=method,
    )
    return _local_torque_from_orbit_sample(
        smart_ds,
        orbit,
        body_radius_m=body_radius_m,
        method=method,
        shell_n_polar=shell_n_polar,
        shell_n_azimuth=shell_n_azimuth,
        shell_radii=None,
    )


def local_mass_loss_on_elliptic_orbit(
    smart_ds,
    semi_major_axis,
    *,
    eccentricity: float = 0.0,
    body_radius_m: float | None = None,
    n_points: int = 360,
    plane: str = "xy",
    angle0: float = 0.0,
    sample: str = "eccentric_anomaly",
    method: str = "nearest",
    shell_n_polar: int = 24,
    shell_n_azimuth: int = 48,
    shell_n_radii: int = 12,
):
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    fields = (
        resolve_batsrus_density_si(smart_ds)[0],
        *resolve_batsrus_vector_xyz_si(smart_ds, "U")[0],
    )
    orbit = sample_elliptic_orbit(
        smart_ds,
        semi_major_axis,
        eccentricity=eccentricity,
        fields=fields,
        n_points=n_points,
        plane=plane,
        angle0=angle0,
        sample=sample,
        method=method,
    )
    rmin = max(0.0, float(semi_major_axis) * (1.0 - float(eccentricity)))
    rmax = float(semi_major_axis) * (1.0 + float(eccentricity))
    shell_radii = np.linspace(rmin, rmax, max(2, int(shell_n_radii)))
    out = _local_mass_loss_from_orbit_sample(
        smart_ds,
        orbit,
        body_radius_m=body_radius_m,
        method=method,
        shell_n_polar=shell_n_polar,
        shell_n_azimuth=shell_n_azimuth,
        shell_radii=shell_radii,
    )
    out["semi_major_axis [R]"] = float(semi_major_axis)
    out["eccentricity [none]"] = float(eccentricity)
    return out


def local_torque_on_elliptic_orbit(
    smart_ds,
    semi_major_axis,
    *,
    eccentricity: float = 0.0,
    body_radius_m: float | None = None,
    n_points: int = 360,
    plane: str = "xy",
    angle0: float = 0.0,
    sample: str = "eccentric_anomaly",
    method: str = "nearest",
    shell_n_polar: int = 24,
    shell_n_azimuth: int = 48,
    shell_n_radii: int = 12,
):
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    fields = (
        resolve_batsrus_density_si(smart_ds)[0],
        *resolve_batsrus_vector_xyz_si(smart_ds, "U")[0],
        *resolve_batsrus_vector_xyz_si(smart_ds, "B")[0],
    )
    orbit = sample_elliptic_orbit(
        smart_ds,
        semi_major_axis,
        eccentricity=eccentricity,
        fields=fields,
        n_points=n_points,
        plane=plane,
        angle0=angle0,
        sample=sample,
        method=method,
    )
    rmin = max(0.0, float(semi_major_axis) * (1.0 - float(eccentricity)))
    rmax = float(semi_major_axis) * (1.0 + float(eccentricity))
    shell_radii = np.linspace(rmin, rmax, max(2, int(shell_n_radii)))
    out = _local_torque_from_orbit_sample(
        smart_ds,
        orbit,
        body_radius_m=body_radius_m,
        method=method,
        shell_n_polar=shell_n_polar,
        shell_n_azimuth=shell_n_azimuth,
        shell_radii=shell_radii,
    )
    out["semi_major_axis [R]"] = float(semi_major_axis)
    out["eccentricity [none]"] = float(eccentricity)
    return out


__all__ = [
    "orbital_period",
    "orbital_velocity",
    "circular_orbit_points",
    "elliptic_orbit_points",
    "sample_points",
    "sample_circular_orbit",
    "sample_elliptic_orbit",
    "local_mass_loss_on_circular_orbit",
    "local_mass_loss_on_elliptic_orbit",
    "local_torque_on_circular_orbit",
    "local_torque_on_elliptic_orbit",
]
