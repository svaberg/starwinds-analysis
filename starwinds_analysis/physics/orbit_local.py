"""THIS FILE contains local orbit diagnostics built from sampled orbit points.

It compares local mass-loss/torque estimates against shell profiles along circular
or elliptic orbits. This is a workflow-heavy module and remains debt while the
library is being split into stricter primitives.
"""

# TODO(debt): This file is workflow-heavy and quantity-specific (`local_mass_loss_*`,
# `local_torque_*`) in a deep layer. Keep moving shared pieces into neutral
# primitives and request SI quantities directly from SmartDs/griblet where feasible.

from __future__ import annotations

import numpy as np

from starwinds_analysis.sampling.orbits import sample_circular_orbit, sample_elliptic_orbit
from starwinds_analysis.analysis.shells import infer_body_radius_m
from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.physics.local_estimates import (
    local_mass_loss_estimates,
    local_torque_estimates,
)
from starwinds_analysis.physics.mass_loss import mass_loss_vs_radius
from starwinds_analysis.physics.shell_torque import torque_vs_radius
from starwinds_analysis.recipes.spherical import radial_component, spherical_vector_components

def _ensure_batsrus_orbit_fields(smart_ds, *, body_radius_m: float, need_b: bool) -> None:
    needed = {
        "Rho [kg/m^3]",
        "U_x [m/s]",
        "U_y [m/s]",
        "U_z [m/s]",
    }
    if need_b:
        needed.update({"B_x [T]", "B_y [T]", "B_z [T]"})
    if all(smart_ds.has_field(name) for name in needed):
        return
    smart_ds.add_batsrus_graph(body_radius_m=float(body_radius_m))

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
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"

    x = orbit["X [sample]"]
    y = orbit["Y [sample]"]
    z = orbit["Z [sample]"]
    rho = orbit[rho_name]
    ux = orbit[ux_name]
    uy = orbit[uy_name]
    uz = orbit[uz_name]
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
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"
    bx_name, by_name, bz_name = "B_x [T]", "B_y [T]", "B_z [T]"

    x = orbit["X [sample]"]
    y = orbit["Y [sample]"]
    z = orbit["Z [sample]"]
    r_sample_r = np.array(orbit["R [sample]"], dtype=float)
    r_m = r_sample_r * body_radius_m
    rho = orbit[rho_name]
    ux = orbit[ux_name]
    uy = orbit[uy_name]
    uz = orbit[uz_name]
    bx = orbit[bx_name]
    by = orbit[by_name]
    bz = orbit[bz_name]

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
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    _ensure_batsrus_orbit_fields(smart_ds, body_radius_m=body_radius_m, need_b=False)
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
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
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    _ensure_batsrus_orbit_fields(smart_ds, body_radius_m=body_radius_m, need_b=True)
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
            "B_x [T]",
            "B_y [T]",
            "B_z [T]",
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
    _ensure_batsrus_orbit_fields(smart_ds, body_radius_m=body_radius_m, need_b=False)
    fields = (
        "Rho [kg/m^3]",
        "U_x [m/s]",
        "U_y [m/s]",
        "U_z [m/s]",
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
    _ensure_batsrus_orbit_fields(smart_ds, body_radius_m=body_radius_m, need_b=True)
    fields = (
        "Rho [kg/m^3]",
        "U_x [m/s]",
        "U_y [m/s]",
        "U_z [m/s]",
        "B_x [T]",
        "B_y [T]",
        "B_z [T]",
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

