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

import logging

import numpy as np

from starwinds_analysis.analysis.orbits import periodic_curve_velocity
from starwinds_analysis.analysis.orbits import sample_circular_orbit
from starwinds_analysis.analysis.orbits import sample_elliptic_orbit
from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.analysis.shells import infer_body_radius_m
from starwinds_analysis.physics.pressure import magnetospheric_standoff_distance
from starwinds_analysis.physics.orbits import orbital_period
from starwinds_analysis.physics.pressure import ram_pressure

log = logging.getLogger(__name__)

def pressure_components_from_orbit_sample(
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
    rho = np.array(orbit("Rho [kg/m^3]"))
    log.debug(
        "pressure_components_from_orbit_sample: n_points=%d, include_relative=%s",
        rho.size,
        include_relative_ram,
    )
    u_xyz = np.column_stack(
        [orbit("U_x [m/s]"), orbit("U_y [m/s]"), orbit("U_z [m/s]")]
    )

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
                [orbit("X [sample]"), orbit("Y [sample]"), orbit("Z [sample]")]
            )
            object_velocity = periodic_curve_velocity(points_r, phase, period_s, body_radius_m)

    comps = {
        "U [m/s]": np.array(orbit("U [m/s]")),
        "B [T]": np.array(orbit("B [T]")),
        "magnetic_pressure [Pa]": np.array(orbit("magnetic_pressure [Pa]")),
        "ram_pressure [Pa]": np.array(orbit("ram_pressure [Pa]")),
        "thermal_pressure [Pa]": np.array(orbit("thermal_pressure [Pa]")),
        "standoff_distance [m]": np.array(orbit("standoff_distance [m]")),
    }

    # TODO(griblet): Relative-speed/relative-ram and standoff quantities still use
    # local workflow logic because they depend on the orbit-derived object velocity.
    if object_velocity is not None:
        V_xyz = object_velocity
        U_minus_V = u_xyz - V_xyz
        U_minus_V_m_s = np.sqrt(np.sum(U_minus_V * U_minus_V, axis=-1))
        comps["V [m/s]"] = np.sqrt(np.sum(V_xyz * V_xyz, axis=-1))
        comps["U_minus_V [m/s]"] = U_minus_V_m_s
        comps["relative_ram_pressure [Pa]"] = ram_pressure(rho, U_minus_V_m_s)
        comps["standoff_distance [m]"] = magnetospheric_standoff_distance(
            rho,
            U_minus_V_m_s,
            b0_t=standoff_b0_t,
        )
        log.debug("pressure_components_from_orbit_sample: using relative velocity for standoff")

    weights = orbit.get("time_weight [none]")
    summary = {}
    for key, value in comps.items():
        arr = np.array(value)
        if arr.ndim == 1:
            summary[key] = summarize_samples(arr, weights=weights)
    return {
        "rho [kg/m^3]": rho,
        "orbit_samples": orbit,
        **comps,
        "summary": summary,
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
    Used by: `test/test_orbit_pressure.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    log.info(
        "pressure_components_on_circular_orbit start: radius=%s, n_points=%d, method=%s, plane=%s",
        radius,
        n_points,
        method,
        plane,
    )
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    u_xyz = ("U_x [m/s]", "U_y [m/s]", "U_z [m/s]")
    derived = (
        "U [m/s]",
        "B [T]",
        "magnetic_pressure [Pa]",
        "ram_pressure [Pa]",
        "thermal_pressure [Pa]",
        "standoff_distance [m]",
    )
    orbit = sample_circular_orbit(
        smart_ds,
        radius,
        fields=(rho_name, *u_xyz, *derived),
        n_points=n_points,
        plane=plane,
        method=method,
    )
    out = pressure_components_from_orbit_sample(
        orbit,
        body_radius_m=body_radius_m,
        star_mass_kg=star_mass_kg,
        semi_major_axis_r=float(radius),
        include_relative_ram=include_relative_ram,
    )
    out["radius [R]"] = float(radius)
    out["radius [m]"] = float(radius) * body_radius_m
    log.info(
        "pressure_components_on_circular_orbit done: finite_ram=%d",
        np.count_nonzero(np.isfinite(out["ram_pressure [Pa]"])),
    )
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
    Used by: `test/test_orbit_pressure.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    log.info(
        "pressure_components_on_elliptic_orbit start: a=%s, e=%s, n_points=%d, method=%s, plane=%s",
        semi_major_axis,
        eccentricity,
        n_points,
        method,
        plane,
    )
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    u_xyz = ("U_x [m/s]", "U_y [m/s]", "U_z [m/s]")
    derived = (
        "U [m/s]",
        "B [T]",
        "magnetic_pressure [Pa]",
        "ram_pressure [Pa]",
        "thermal_pressure [Pa]",
        "standoff_distance [m]",
    )
    orbit = sample_elliptic_orbit(
        smart_ds,
        semi_major_axis,
        eccentricity=eccentricity,
        fields=(rho_name, *u_xyz, *derived),
        n_points=n_points,
        plane=plane,
        angle0=angle0,
        sample=sample,
        method=method,
    )
    out = pressure_components_from_orbit_sample(
        orbit,
        body_radius_m=body_radius_m,
        star_mass_kg=star_mass_kg,
        semi_major_axis_r=float(semi_major_axis),
        include_relative_ram=include_relative_ram,
    )
    out["semi_major_axis [R]"] = float(semi_major_axis)
    out["eccentricity [none]"] = float(eccentricity)
    out["radius [R]"] = float(np.nanmean(np.array(orbit("R [sample]"))))
    out["radius [m]"] = out["radius [R]"] * body_radius_m
    log.info(
        "pressure_components_on_elliptic_orbit done: finite_ram=%d",
        np.count_nonzero(np.isfinite(out["ram_pressure [Pa]"])),
    )
    return out
