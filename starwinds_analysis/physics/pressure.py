"""THIS FILE contains core pressure and standoff formulas.

These are quantity definitions (ram, magnetic, component combinations, standoff distance).
It should stay as pure math on arrays/scalars, not dataset-specific logic.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.physics.constants import MU0

def magnetic_pressure(b_t_or_mag):
    """
    Magnetic pressure `B^2 / (2 mu0)` in Pa.
    """
    return np.square(b_t_or_mag) / (2.0 * MU0)

def ram_pressure(rho_kg_m3, speed_m_s):
    """
    Ram pressure `rho * u^2` in Pa.
    """
    return rho_kg_m3 * np.square(speed_m_s)

def pressure_components(
    rho_kg_m3,
    u_xyz_m_s,
    b_xyz_t,
    *,
    thermal_pressure_pa=None,
    object_velocity_xyz_m_s=None,
):
    """
    Compute thermal/magnetic/ram pressure components from local samples.
    """
    # TODO(griblet): The derived quantities assembled here (`|U|`, `|B|`,
    # magnetic/ram/relative pressures) should come from SmartDs/griblet requests in
    # SI units instead of being bundled/computed here.
    u = u_xyz_m_s
    b = b_xyz_t
    u_shape = np.shape(u)
    b_shape = np.shape(b)
    if not u_shape or not b_shape or u_shape[-1] != 3 or b_shape[-1] != 3:
        raise ValueError("u_xyz_m_s and b_xyz_t must have shape (..., 3)")

    speed = np.sqrt(np.sum(np.square(u), axis=-1))
    bmag = np.sqrt(np.sum(np.square(b), axis=-1))
    out = {
        "U [m/s]": speed,
        "B [T]": bmag,
        "magnetic_pressure [Pa]": magnetic_pressure(bmag),
        "ram_pressure [Pa]": ram_pressure(rho_kg_m3, speed),
    }

    if thermal_pressure_pa is not None:
        out["thermal_pressure [Pa]"] = thermal_pressure_pa

    if object_velocity_xyz_m_s is not None:
        v_obj = object_velocity_xyz_m_s
        if np.shape(v_obj) != u_shape:
            raise ValueError("object_velocity_xyz_m_s must match u_xyz_m_s shape")
        rel = np.subtract(u, v_obj)
        rel_speed = np.sqrt(np.sum(np.square(rel), axis=-1))
        out["object_speed [m/s]"] = np.sqrt(np.sum(np.square(v_obj), axis=-1))
        out["relative_speed [m/s]"] = rel_speed
        out["relative_ram_pressure [Pa]"] = ram_pressure(rho_kg_m3, rel_speed)
    return out

def magnetospheric_standoff_distance(rho_kg_m3, speed_m_s, *, b0_t: float = 0.7e-4):
    """
    Vidotto-style stand-off distance proxy from pressure balance.

    The default `b0_t` matches the old batplotlib helper.
    """
    # TODO(griblet): If this proxy is kept as a reusable quantity, expose it through
    # SmartDs/griblet so orbit/surface diagnostics can request it directly in SI.
    p_ram = ram_pressure(rho_kg_m3, speed_m_s)
    numer = (float(b0_t) ** 2) / (2.0 * MU0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.power(numer / p_ram, 1.0 / 6.0)
