"""THIS FILE contains local orbit diagnostics built from sampled orbit points.

It compares local mass-loss/torque estimates against shell profiles along circular
or elliptic orbits. This is a workflow-heavy module and remains debt while the
library is being split into stricter primitives.
"""

# TODO(debt): This file is workflow-heavy and quantity-specific (`local_mass_loss_*`,
# `local_torque_*`) in a deep layer. It now uses generic shell primitives, but the
# orbit comparison workflow still needs to move upward or be reduced further.

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis.orbits import sample_circular_orbit
from starwinds_analysis.analysis.orbits import sample_elliptic_orbit
from starwinds_analysis.analysis.shells import integrate_shell_scalar
from starwinds_analysis.analysis.shells import sample_shell_field
from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.physics.torque import local_torque_estimates

def interp_profile(radii, values, x):
    """
    1D interpolate a shell profile onto orbit sample radii (with NaNs outside range).
    """
    r = np.array(radii)
    y = np.array(values)
    x = np.array(x)
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

def local_mass_loss_from_orbit_sample(
    smart_ds,
    orbit,
    *,
    body_radius_m,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    """
    Compute local mass-loss estimates on one sampled orbit and compare to shell profile values.
    Used by: `starwinds_analysis/physics/orbit_local.py`
    """
    mass_flux = np.array(orbit("mass_flux [kg/m^2/s]"))
    r_sample_r = np.array(orbit("R [sample]"))
    r_m = r_sample_r * body_radius_m
    estimates = 4.0 * np.pi * np.square(r_m) * mass_flux
    weights = orbit.get("time_weight [none]")
    stats = summarize_samples(estimates, weights=weights)

    _, shell_mass_flux, shell_area, shell_profile_radii = sample_shell_field(
        smart_ds,
        [float(np.nanmean(r_sample_r))] if shell_radii is None else shell_radii,
        source_fields=("Rho [kg/m^3]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"),
        shell_field="mass_flux [kg/m^2/s]",
        body_radius_m=body_radius_m,
        n_polar=shell_n_polar,
        n_azimuth=shell_n_azimuth,
        method=method,
    )
    shell_values, _ = integrate_shell_scalar(shell_mass_flux, shell_area)
    if shell_radii is None:
        shell_value = float(shell_values[0])
        shell_interp = np.full_like(estimates, shell_value, dtype=float)
    else:
        shell_interp = interp_profile(shell_profile_radii, shell_values, r_sample_r)
        shell_value = summarize_samples(shell_interp, weights=weights)["mean"]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to_shell = stats["mean"] / shell_value if shell_value != 0 else np.nan

    out = {
        "radius [R]": float(np.nanmean(r_sample_r)),
        "radius [m]": float(np.nanmean(r_m)),
        "mass_flux [kg/m^2/s]": mass_flux,
        "local_mass_loss [kg/s]": estimates,
        "local_mass_loss_mean [kg/s]": float(stats["mean"]),
        "local_mass_loss_std [kg/s]": float(stats["std"]),
        "shell_mass_loss [kg/s]": float(shell_value),
        "mean_to_shell [none]": float(mean_to_shell),
        "orbit_samples": orbit,
    }
    if shell_radii is not None:
        out["shell_mass_loss_interp [kg/s]"] = shell_interp
    return out

def local_torque_from_orbit_sample(
    smart_ds,
    orbit,
    *,
    body_radius_m,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    """
    Compute local torque estimates on one sampled orbit and compare to shell torque profile
      values.
    Used by: `starwinds_analysis/physics/orbit_local.py`
    """
    r_sample_r = np.array(orbit("R [sample]"))
    r_m = r_sample_r * body_radius_m
    orbit_magnetic_density = np.array(orbit("magnetic_torque_density [N/m]"))
    orbit_dynamic_density = np.array(orbit("dynamic_torque_density [N/m]"))
    local_magnetic, local_dynamic, local_total = local_torque_estimates(
        r_m,
        orbit_magnetic_density,
        orbit_dynamic_density,
    )
    weights = orbit.get("time_weight [none]")

    torque_shells, shell_magnetic_density, shell_area, shell_profile_radii = sample_shell_field(
        smart_ds,
        [float(np.nanmean(r_sample_r))] if shell_radii is None else shell_radii,
        source_fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
            "B_x [T]",
            "B_y [T]",
            "B_z [T]",
        ),
        shell_field="magnetic_torque_density [N/m]",
        body_radius_m=body_radius_m,
        n_polar=shell_n_polar,
        n_azimuth=shell_n_azimuth,
        method=method,
    )
    shell_dynamic_density = np.array(torque_shells("dynamic_torque_density [N/m]"))
    shell_magnetic, _ = integrate_shell_scalar(shell_magnetic_density, shell_area)
    shell_dynamic, _ = integrate_shell_scalar(shell_dynamic_density, shell_area)
    shell_values = shell_magnetic + shell_dynamic
    if shell_radii is None:
        shell_total = float(shell_values[0])
        shell_interp = np.full_like(local_total, shell_total, dtype=float)
    else:
        shell_interp = interp_profile(shell_profile_radii, shell_values, r_sample_r)
        shell_total = summarize_samples(shell_interp, weights=weights)["mean"]

    stats = summarize_samples(local_total, weights=weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to_shell = stats["mean"] / shell_total if shell_total != 0 else np.nan

    out = {
        "radius [R]": float(np.nanmean(r_sample_r)),
        "radius [m]": float(np.nanmean(r_m)),
        "magnetic_torque_density [N/m]": orbit_magnetic_density,
        "dynamic_torque_density [N/m]": orbit_dynamic_density,
        "local_magnetic_torque [Nm]": local_magnetic,
        "local_dynamic_torque [Nm]": local_dynamic,
        "local_total_torque [Nm]": local_total,
        "local_total_torque_mean [Nm]": float(stats["mean"]),
        "local_total_torque_std [Nm]": float(stats["std"]),
        "shell_total_torque [Nm]": float(shell_total),
        "mean_to_shell [none]": float(mean_to_shell),
        "orbit_samples": orbit,
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
    Sample a circular orbit and compute local-vs-shell mass-loss comparisons.
    Used by: `test/test_orbit_analysis.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if body_radius_m is None:
        body_radius_m = float(smart_ds("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(
            "mass_flux [kg/m^2/s]",
        ),
        n_points=n_points,
        plane=plane,
        method=method,
    )
    return local_mass_loss_from_orbit_sample(
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
    Sample a circular orbit and compute local-vs-shell torque comparisons.
    Used by: `test/test_orbit_analysis.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if body_radius_m is None:
        body_radius_m = float(smart_ds("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(
            "magnetic_torque_density [N/m]",
            "dynamic_torque_density [N/m]",
        ),
        n_points=n_points,
        plane=plane,
        method=method,
    )
    return local_torque_from_orbit_sample(
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
    """
    Sample an elliptic orbit and compute local-vs-shell mass-loss comparisons.
    Used by: `test/test_orbit_analysis.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if body_radius_m is None:
        body_radius_m = float(smart_ds("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    fields = (
        "mass_flux [kg/m^2/s]",
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
    out = local_mass_loss_from_orbit_sample(
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
    """
    Sample an elliptic orbit and compute local-vs-shell torque comparisons.
    Used by: `test/test_orbit_analysis.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if body_radius_m is None:
        body_radius_m = float(smart_ds("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    fields = (
        "magnetic_torque_density [N/m]",
        "dynamic_torque_density [N/m]",
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
    out = local_torque_from_orbit_sample(
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
