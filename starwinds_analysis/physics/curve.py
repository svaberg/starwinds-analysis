"""THIS FILE contains diagnostics evaluated on sampled curves.

It operates on already sampled curve `SmartDs` objects. Curve geometry belongs
in `analysis/orbits.py`. Pressure formulas belong in `pressure.py`.
"""

from __future__ import annotations

import logging

import numpy as np

from starwinds_analysis.analysis.orbits import periodic_curve_velocity
from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.physics.pressure import magnetospheric_standoff_distance
from starwinds_analysis.physics.pressure import ram_pressure
from starwinds_analysis.physics.torque import local_torque_estimates

log = logging.getLogger(__name__)


def mass_loss_from_curve(curve, *, body_radius_m: float | None = None):
    """Compute local mass-loss estimates along a sampled curve."""
    if body_radius_m is None:
        body_radius_m = float(curve("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    mass_flux = np.array(curve("mass_flux [kg/m^2/s]"))
    r_m = np.array(curve("R [sample]")) * body_radius_m
    return 4.0 * np.pi * np.square(r_m) * mass_flux


def torque_from_curve(curve, *, body_radius_m: float | None = None):
    """Compute local magnetic, dynamic, and total torque estimates along a curve."""
    if body_radius_m is None:
        body_radius_m = float(curve("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    r_m = np.array(curve("R [sample]")) * body_radius_m
    magnetic_torque_density = np.array(curve("magnetic_torque_density [N/m]"))
    dynamic_torque_density = np.array(curve("dynamic_torque_density [N/m]"))
    return local_torque_estimates(
        r_m,
        magnetic_torque_density,
        dynamic_torque_density,
    )


def pressure_components_from_curve(
    curve,
    *,
    body_radius_m: float | None = None,
    period_s: float | None = None,
    include_relative_ram: bool = True,
    standoff_b0_t: float = 0.7e-4,
):
    """Assemble pressure components and standoff proxies from a sampled curve."""
    if body_radius_m is None:
        body_radius_m = float(curve("star_radius [m]"))
    else:
        body_radius_m = float(body_radius_m)
    weights = curve.get("time_weight [none]")
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
