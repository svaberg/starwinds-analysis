"""THIS FILE contains pressure diagnostics evaluated on orbit samples.

It orchestrates orbit sampling with pressure formulas and includes temporary BATSRUS field-resolution glue.
Pressure formulas themselves belong in pressure.py.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis.local_estimates import summarize_samples
from starwinds_analysis.analysis.orbits import (
    orbital_period,
    sample_circular_orbit,
    sample_elliptic_orbit,
)
from starwinds_analysis.analysis.pressure import (
    magnetospheric_standoff_distance,
    pressure_components,
)
from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    resolve_batsrus_density_si,
    resolve_batsrus_vector_xyz_si,
    resolve_field_with_scale,
)


#
# TODO smartds-resolve:
# This BATSRUS pressure resolver should move into SmartDs so analysis code can ask
# SmartDs for SI-ready pressure data (with units parsed from bracketed field names)
# instead of maintaining local fallback logic.
#
def resolve_batsrus_pressure_si(smart_ds):
    return resolve_field_with_scale(
        smart_ds,
        [
            ("P [Pa]", 1.0),
            ("P [dyne/cm^2]", 0.1),
        ],
    )


def _periodic_orbit_velocity(points_r, phase_turns, period_s, body_radius_m):
    points = np.asarray(points_r, dtype=float) * float(body_radius_m)
    phase = np.asarray(phase_turns, dtype=float)
    n = points.shape[0]
    if n < 3:
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


def _summaries_from_arrays(data, *, weights=None):
    out = {}
    for key, value in data.items():
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1:
            continue
        out[key] = summarize_samples(arr, weights=weights)
    return out


def pressure_components_from_orbit_sample(
    smart_ds,
    orbit,
    *,
    body_radius_m: float,
    star_mass_kg: float | None = None,
    semi_major_axis_r: float | None = None,
    include_relative_ram: bool = True,
    standoff_b0_t: float = 0.7e-4,
):
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")
    (bx_name, by_name, bz_name), b_scale = resolve_batsrus_vector_xyz_si(smart_ds, "B")
    p_name, p_scale = resolve_batsrus_pressure_si(smart_ds)

    rho = rho_scale * np.asarray(orbit[rho_name], dtype=float)
    u_xyz = u_scale * np.column_stack([orbit[ux_name], orbit[uy_name], orbit[uz_name]])
    b_xyz = b_scale * np.column_stack([orbit[bx_name], orbit[by_name], orbit[bz_name]])
    p_therm = p_scale * np.asarray(orbit[p_name], dtype=float)

    object_velocity = None
    if (
        include_relative_ram
        and star_mass_kg is not None
        and semi_major_axis_r is not None
        and np.isfinite(semi_major_axis_r)
    ):
        phase = np.asarray(orbit.get("phase [turns]"), dtype=float)
        if phase.shape == (len(rho),):
            period_s = orbital_period(float(semi_major_axis_r) * body_radius_m, star_mass_kg)
            points_r = np.column_stack(
                [orbit["X [sample]"], orbit["Y [sample]"], orbit["Z [sample]"]]
            )
            object_velocity = _periodic_orbit_velocity(points_r, phase, period_s, body_radius_m)

    comps = pressure_components(
        rho,
        u_xyz,
        b_xyz,
        thermal_pressure_pa=p_therm,
        object_velocity_xyz_m_s=object_velocity,
    )

    speed_for_standoff = comps.get("relative_speed [m/s]", comps["U [m/s]"])
    comps["standoff_distance [m]"] = magnetospheric_standoff_distance(
        rho,
        speed_for_standoff,
        b0_t=standoff_b0_t,
    )

    weights = orbit.get("time_weight [none]")
    return {
        "rho [kg/m^3]": rho,
        "orbit_samples": orbit,
        **comps,
        "summary": _summaries_from_arrays(comps, weights=weights),
    }


def pressure_components_on_circular_orbit(
    smart_ds,
    radius,
    *,
    body_radius_m: float | None = None,
    n_points: int = 360,
    plane: str = "xy",
    method: str = "nearest",
    star_mass_kg: float | None = None,
    include_relative_ram: bool = True,
):
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name = resolve_batsrus_density_si(smart_ds)[0]
    p_name = resolve_batsrus_pressure_si(smart_ds)[0]
    u_xyz = resolve_batsrus_vector_xyz_si(smart_ds, "U")[0]
    b_xyz = resolve_batsrus_vector_xyz_si(smart_ds, "B")[0]
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(rho_name, *u_xyz, *b_xyz, p_name),
        n_points=n_points,
        plane=plane,
        method=method,
    )
    out = pressure_components_from_orbit_sample(
        smart_ds,
        orbit,
        body_radius_m=body_radius_m,
        star_mass_kg=star_mass_kg,
        semi_major_axis_r=float(radius),
        include_relative_ram=include_relative_ram,
    )
    out["radius [R]"] = float(radius)
    out["radius [m]"] = float(radius) * body_radius_m
    return out


def pressure_components_on_elliptic_orbit(
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
    star_mass_kg: float | None = None,
    include_relative_ram: bool = True,
):
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name = resolve_batsrus_density_si(smart_ds)[0]
    p_name = resolve_batsrus_pressure_si(smart_ds)[0]
    u_xyz = resolve_batsrus_vector_xyz_si(smart_ds, "U")[0]
    b_xyz = resolve_batsrus_vector_xyz_si(smart_ds, "B")[0]
    orbit = sample_elliptic_orbit(
        smart_ds,
        semi_major_axis,
        eccentricity=eccentricity,
        fields=(rho_name, *u_xyz, *b_xyz, p_name),
        n_points=n_points,
        plane=plane,
        angle0=angle0,
        sample=sample,
        method=method,
    )
    out = pressure_components_from_orbit_sample(
        smart_ds,
        orbit,
        body_radius_m=body_radius_m,
        star_mass_kg=star_mass_kg,
        semi_major_axis_r=float(semi_major_axis),
        include_relative_ram=include_relative_ram,
    )
    out["semi_major_axis [R]"] = float(semi_major_axis)
    out["eccentricity [none]"] = float(eccentricity)
    out["radius [R]"] = float(np.nanmean(np.asarray(orbit["R [sample]"], dtype=float)))
    out["radius [m]"] = out["radius [R]"] * body_radius_m
    return out


__all__ = [
    "resolve_batsrus_pressure_si",
    "pressure_components_from_orbit_sample",
    "pressure_components_on_circular_orbit",
    "pressure_components_on_elliptic_orbit",
]
