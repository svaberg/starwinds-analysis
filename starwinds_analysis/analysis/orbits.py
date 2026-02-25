from __future__ import annotations

import math

import numpy as np

from starwinds_analysis.analysis.local_estimates import (
    local_mass_loss_estimates,
    local_torque_estimates,
    summarize_samples,
)
from starwinds_analysis.analysis.mass_loss import mass_loss_vs_radius
from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    resolve_batsrus_density_si,
    resolve_batsrus_vector_xyz_si,
)
from starwinds_analysis.analysis.torque import torque_vs_radius
from starwinds_analysis.recipes.spherical import radial_component, spherical_vector_components


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
    points = np.asarray(points, dtype=float)
    out = smart_ds.resample(
        points,
        coordinate_fields=coordinate_fields,
        fields=tuple(dict.fromkeys(fields)),
        method=method,
        fill_value=fill_value,
        zone="orbit-samples",
    )
    data = {name: np.asarray(out.variable(name), dtype=float) for name in fields}
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
    return sampled


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
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")

    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(rho_name, ux_name, uy_name, uz_name),
        n_points=n_points,
        plane=plane,
        method=method,
    )

    x = orbit["X [sample]"]
    y = orbit["Y [sample]"]
    z = orbit["Z [sample]"]
    rho = rho_scale * orbit[rho_name]
    ux = u_scale * orbit[ux_name]
    uy = u_scale * orbit[uy_name]
    uz = u_scale * orbit[uz_name]
    u_r = radial_component(ux, uy, uz, x, y, z)
    r_m = orbit["R [sample]"] * body_radius_m

    estimates = local_mass_loss_estimates(r_m, rho, u_r)
    summary = summarize_samples(estimates)

    shell = mass_loss_vs_radius(
        smart_ds,
        [float(radius)],
        body_radius_m=body_radius_m,
        n_polar=shell_n_polar,
        n_azimuth=shell_n_azimuth,
        method=method,
    )
    shell_value = float(shell["mass_loss [kg/s]"][0])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to_shell = summary["mean"] / shell_value if shell_value != 0 else np.nan

    return {
        "radius [R]": float(radius),
        "radius [m]": float(np.nanmean(r_m)),
        "u_r [m/s]": u_r,
        "rho [kg/m^3]": rho,
        "local_mass_loss [kg/s]": estimates,
        "summary": summary,
        "shell_mass_loss [kg/s]": shell_value,
        "mean_to_shell [none]": float(mean_to_shell),
        "orbit_samples": orbit,
        "shell_profile": shell,
    }


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
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")
    (bx_name, by_name, bz_name), b_scale = resolve_batsrus_vector_xyz_si(smart_ds, "B")

    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(rho_name, ux_name, uy_name, uz_name, bx_name, by_name, bz_name),
        n_points=n_points,
        plane=plane,
        method=method,
    )

    x = orbit["X [sample]"]
    y = orbit["Y [sample]"]
    z = orbit["Z [sample]"]
    r_m = orbit["R [sample]"] * body_radius_m
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

    shell = torque_vs_radius(
        smart_ds,
        [float(radius)],
        body_radius_m=body_radius_m,
        n_polar=shell_n_polar,
        n_azimuth=shell_n_azimuth,
        method=method,
    )
    shell_total = float(shell["total_torque [Nm]"][0])
    summary = summarize_samples(local["total [Nm]"])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to_shell = summary["mean"] / shell_total if shell_total != 0 else np.nan

    return {
        "radius [R]": float(radius),
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
        "shell_total_torque [Nm]": shell_total,
        "mean_to_shell [none]": float(mean_to_shell),
        "orbit_samples": orbit,
        "shell_profile": shell,
    }


__all__ = [
    "circular_orbit_points",
    "sample_points",
    "sample_circular_orbit",
    "local_mass_loss_on_circular_orbit",
    "local_torque_on_circular_orbit",
]

