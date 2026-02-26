"""THIS FILE contains pressure diagnostics evaluated on orbit samples.

It orchestrates orbit sampling with pressure formulas and includes temporary BATSRUS field-resolution glue.
Pressure formulas themselves belong in pressure.py.
"""

# TODO(debt): This file is an orbit workflow/pipeline (sampling + field resolution +
# summaries) but currently lives in `physics`.
# TODO(debt): It is a workflow-heavy module in `physics`; keep moving shared
# orbit/sampling pieces into neutral primitives and use SmartDs/griblet SI requests
# internally.

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.physics.orbits import orbital_period
from starwinds_analysis.sampling.orbits import sample_circular_orbit, sample_elliptic_orbit
from starwinds_analysis.physics.pressure import (
    magnetospheric_standoff_distance,
    ram_pressure,
)
from starwinds_analysis.analysis.shells import infer_body_radius_m

def _pressure_field_name_and_scale(smart_ds):
    """
    Choose a thermal-pressure field name and conversion scale from available orbit/surface
      sample fields.
    Used by: `starwinds_analysis/physics/orbit_surface.py`,
      `starwinds_analysis/physics/orbit_pressure.py`
    """
    if smart_ds.has_field("P [Pa]"):
        return "P [Pa]", 1.0
    if smart_ds.has_field("P [dyne/cm^2]"):
        return "P [dyne/cm^2]", 0.1
    raise KeyError("Could not find pressure field in SI or cgs form")

def _periodic_orbit_velocity(points_r, phase_turns, period_s, body_radius_m):
    """
    Compute periodic orbit-frame velocity components from sampled points/phase for relative-
      speed calculations.
    Used by: `starwinds_analysis/physics/orbit_surface.py`,
      `starwinds_analysis/physics/orbit_pressure.py`
    """
    points = np.array(points_r) * float(body_radius_m)
    phase = np.array(phase_turns)
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
    """
    Build weighted summary dicts (mean/std/quantiles) for a dict of arrays.
    Used by: `starwinds_analysis/physics/orbit_pressure.py`
    """
    out = {}
    for key, value in data.items():
        arr = np.array(value)
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
    """
    Assemble orbit-sampled pressure components and standoff proxies from sampled fields.
    Used by: `starwinds_analysis/physics/orbit_pressure.py`
    """
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"
    p_name, p_scale = _pressure_field_name_and_scale(smart_ds)

    rho = np.array(orbit[rho_name])
    u_xyz = np.column_stack([orbit[ux_name], orbit[uy_name], orbit[uz_name]])
    p_therm = p_scale * np.array(orbit[p_name])

    object_velocity = None
    if (
        include_relative_ram
        and star_mass_kg is not None
        and semi_major_axis_r is not None
        and np.isfinite(semi_major_axis_r)
    ):
        phase = np.array(orbit.get("phase [turns]"))
        if phase.shape == (len(rho),):
            period_s = orbital_period(float(semi_major_axis_r) * body_radius_m, star_mass_kg)
            points_r = np.column_stack(
                [orbit["X [sample]"], orbit["Y [sample]"], orbit["Z [sample]"]]
            )
            object_velocity = _periodic_orbit_velocity(points_r, phase, period_s, body_radius_m)

    comps = {
        "U [m/s]": np.array(orbit["U [m/s]"]),
        "B [T]": np.array(orbit["B [T]"]),
        "magnetic_pressure [Pa]": np.array(orbit["magnetic_pressure [Pa]"]),
        "ram_pressure [Pa]": np.array(orbit["ram_pressure [Pa]"]),
        "thermal_pressure [Pa]": p_therm,
    }

    # TODO(griblet): Relative-speed/relative-ram and standoff quantities still use
    # local workflow logic because they depend on the orbit-derived object velocity.
    if object_velocity is not None:
        v_obj = object_velocity
        rel = u_xyz - v_obj
        rel_speed = np.sqrt(np.sum(rel * rel, axis=-1))
        comps["object_speed [m/s]"] = np.sqrt(np.sum(v_obj * v_obj, axis=-1))
        comps["relative_speed [m/s]"] = rel_speed
        comps["relative_ram_pressure [Pa]"] = ram_pressure(rho, rel_speed)

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
    """
    Sample a circular orbit and compute pressure-component diagnostics.
    Used by: `test/test_orbit_pressure.py`, `starwinds_analysis/quicklook2d.py`
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    p_name = _pressure_field_name_and_scale(smart_ds)[0]
    u_xyz = ("U_x [m/s]", "U_y [m/s]", "U_z [m/s]")
    b_xyz = ("B_x [T]", "B_y [T]", "B_z [T]")
    derived = ("U [m/s]", "B [T]", "magnetic_pressure [Pa]", "ram_pressure [Pa]")
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(rho_name, *u_xyz, *b_xyz, *derived, p_name),
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
    """
    Sample an elliptic orbit and compute pressure-component diagnostics.
    Used by: `test/test_orbit_pressure.py`, `starwinds_analysis/quicklook2d.py`
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    p_name = _pressure_field_name_and_scale(smart_ds)[0]
    u_xyz = ("U_x [m/s]", "U_y [m/s]", "U_z [m/s]")
    b_xyz = ("B_x [T]", "B_y [T]", "B_z [T]")
    derived = ("U [m/s]", "B [T]", "magnetic_pressure [Pa]", "ram_pressure [Pa]")
    orbit = sample_elliptic_orbit(
        smart_ds,
        semi_major_axis,
        eccentricity=eccentricity,
        fields=(rho_name, *u_xyz, *b_xyz, *derived, p_name),
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
    out["radius [R]"] = float(np.nanmean(np.array(orbit["R [sample]"])))
    out["radius [m]"] = out["radius [R]"] * body_radius_m
    return out
