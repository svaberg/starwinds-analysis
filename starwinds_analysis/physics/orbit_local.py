"""THIS FILE contains local diagnostics built from sampled curves.

It compares local mass-loss/torque estimates against shell profiles along sampled
curves. This is still workflow-heavy and remains debt while the library is being
split into stricter primitives.
"""

# TODO(debt): This file is workflow-heavy and quantity-specific in a deep layer.
# It now consumes sampled curves instead of constructing orbit geometry, but the
# comparison workflow and bundle-shaped outputs still need to move upward or shrink.

from __future__ import annotations

import numpy as np

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

def local_mass_loss_from_curve(
    smart_ds,
    curve,
    *,
    body_radius_m,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    """
    Compute local mass-loss estimates on one sampled curve and compare to shell profile values.
    """
    if body_radius_m is None:
        body_radius_m = float(curve("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    mass_flux = np.array(curve("mass_flux [kg/m^2/s]"))
    r_sample_r = np.array(curve("R [sample]"))
    r_m = r_sample_r * body_radius_m
    estimates = 4.0 * np.pi * np.square(r_m) * mass_flux
    weights = curve.get("time_weight [none]")
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
        "curve_samples": curve,
    }
    if shell_radii is not None:
        out["shell_mass_loss_interp [kg/s]"] = shell_interp
    return out

def local_torque_from_curve(
    smart_ds,
    curve,
    *,
    body_radius_m,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    """
    Compute local torque estimates on one sampled curve and compare to shell torque profile
      values.
    """
    if body_radius_m is None:
        body_radius_m = float(curve("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    r_sample_r = np.array(curve("R [sample]"))
    r_m = r_sample_r * body_radius_m
    orbit_magnetic_density = np.array(curve("magnetic_torque_density [N/m]"))
    orbit_dynamic_density = np.array(curve("dynamic_torque_density [N/m]"))
    local_magnetic, local_dynamic, local_total = local_torque_estimates(
        r_m,
        orbit_magnetic_density,
        orbit_dynamic_density,
    )
    weights = curve.get("time_weight [none]")

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
        "curve_samples": curve,
    }
    if shell_radii is not None:
        out["shell_total_torque_interp [Nm]"] = shell_interp
    return out
