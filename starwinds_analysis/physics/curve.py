"""THIS FILE contains diagnostics evaluated on sampled curves.

It operates on already sampled curve `SmartDs` objects. Curve geometry belongs
in `analysis/orbits.py`. Pressure formulas belong in `pressure.py`.
"""

from __future__ import annotations

import logging

import numpy as np

from starwinds_analysis.analysis.orbits import periodic_curve_velocity
from starwinds_analysis.analysis.shells import integrate_shell_scalar
from starwinds_analysis.analysis.shells import sample_shell_field
from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.physics.pressure import magnetospheric_standoff_distance
from starwinds_analysis.physics.pressure import ram_pressure
from starwinds_analysis.physics.torque import local_torque_estimates

log = logging.getLogger(__name__)

def curve_context(curve, body_radius_m):
    """Return the shared body-radius and time-weight context for one sampled curve."""
    if body_radius_m is None:
        body_radius_m = float(curve("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    weights = curve.get("time_weight [none]")
    return body_radius_m, weights


def pressure_components_from_curve(
    curve,
    *,
    body_radius_m: float | None = None,
    period_s: float | None = None,
    include_relative_ram: bool = True,
    standoff_b0_t: float = 0.7e-4,
):
    """Assemble pressure components and standoff proxies from a sampled curve."""
    body_radius_m, weights = curve_context(curve, body_radius_m)
    rho = np.array(curve("Rho [kg/m^3]"))
    log.debug(
        "pressure_components_from_curve: n_points=%d, include_relative=%s",
        rho.size,
        include_relative_ram,
    )
    U_xyz = np.array(curve("U_xyz [m/s]"))

    V_xyz = None
    if include_relative_ram and period_s is not None and np.isfinite(period_s):
        phase = np.array(curve.get("phase [turns]"))
        if phase.shape == (len(rho),):
            points_r = np.column_stack(
                [curve("X [sample]"), curve("Y [sample]"), curve("Z [sample]")]
            )
            V_xyz = periodic_curve_velocity(points_r, phase, float(period_s), body_radius_m)

    comps = {
        "U [m/s]": np.array(curve("U [m/s]")),
        "B [T]": np.array(curve("B [T]")),
        "magnetic_pressure [Pa]": np.array(curve("magnetic_pressure [Pa]")),
        "ram_pressure [Pa]": np.array(curve("ram_pressure [Pa]")),
        "thermal_pressure [Pa]": np.array(curve("thermal_pressure [Pa]")),
        "standoff_distance [m]": np.array(curve("standoff_distance [m]")),
    }

    # TODO(griblet): Relative-speed/relative-ram and standoff quantities still use
    # local workflow logic because they depend on the trajectory velocity.
    if V_xyz is not None:
        U_minus_V = U_xyz - V_xyz
        U_minus_V_m_s = np.sqrt(np.sum(U_minus_V * U_minus_V, axis=-1))
        comps["V [m/s]"] = np.sqrt(np.sum(V_xyz * V_xyz, axis=-1))
        comps["U_minus_V [m/s]"] = U_minus_V_m_s
        comps["relative_ram_pressure [Pa]"] = ram_pressure(rho, U_minus_V_m_s)
        comps["standoff_distance [m]"] = magnetospheric_standoff_distance(
            rho,
            U_minus_V_m_s,
            b0_t=standoff_b0_t,
        )
        log.debug("pressure_components_from_curve: using relative velocity for standoff")

    summary = {}
    for key, value in comps.items():
        arr = np.array(value)
        if arr.ndim == 1:
            summary[key] = summarize_samples(arr, weights=weights)
    return {
        "rho [kg/m^3]": rho,
        "curve_samples": curve,
        **comps,
        "summary": summary,
    }


def interpolate_profile(radii, values, x):
    """Interpolate one shell profile onto curve radii, using NaN outside range."""
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


def mass_loss_from_curve(
    smart_ds,
    curve,
    *,
    body_radius_m,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    """Compute local mass-loss estimates on one sampled curve."""
    body_radius_m, weights = curve_context(curve, body_radius_m)
    mass_flux = np.array(curve("mass_flux [kg/m^2/s]"))
    r_sample_r = np.array(curve("R [sample]"))
    r_m = r_sample_r * body_radius_m
    estimates = 4.0 * np.pi * np.square(r_m) * mass_flux
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
        shell_interp = interpolate_profile(shell_profile_radii, shell_values, r_sample_r)
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


def torque_from_curve(
    smart_ds,
    curve,
    *,
    body_radius_m,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    """Compute local torque estimates on one sampled curve."""
    body_radius_m, weights = curve_context(curve, body_radius_m)
    r_sample_r = np.array(curve("R [sample]"))
    r_m = r_sample_r * body_radius_m
    orbit_magnetic_density = np.array(curve("magnetic_torque_density [N/m]"))
    orbit_dynamic_density = np.array(curve("dynamic_torque_density [N/m]"))
    local_magnetic, local_dynamic, local_total = local_torque_estimates(
        r_m,
        orbit_magnetic_density,
        orbit_dynamic_density,
    )
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
        shell_interp = interpolate_profile(shell_profile_radii, shell_values, r_sample_r)
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
