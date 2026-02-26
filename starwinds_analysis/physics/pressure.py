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
    # TODO(griblet): `magnetic_pressure [Pa]` should be a SmartDs/griblet quantity
    # so callers can request it directly in SI units.
    b = np.array(b_t_or_mag, dtype=float)
    return (b * b) / (2.0 * MU0)

def ram_pressure(rho_kg_m3, speed_m_s):
    """
    Ram pressure `rho * u^2` in Pa.
    """
    # TODO(griblet): `ram_pressure [Pa]` should be a SmartDs/griblet quantity so
    # callers do not recompute it outside the field graph.
    rho = np.array(rho_kg_m3, dtype=float)
    u = np.array(speed_m_s, dtype=float)
    return rho * u * u

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
    rho = np.array(rho_kg_m3, dtype=float)
    u = np.array(u_xyz_m_s, dtype=float)
    b = np.array(b_xyz_t, dtype=float)
    if u.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("u_xyz_m_s and b_xyz_t must have shape (..., 3)")

    speed = np.sqrt(np.sum(u * u, axis=-1))
    bmag = np.sqrt(np.sum(b * b, axis=-1))
    out = {
        "U [m/s]": speed,
        "B [T]": bmag,
        "magnetic_pressure [Pa]": magnetic_pressure(bmag),
        "ram_pressure [Pa]": ram_pressure(rho, speed),
    }

    if thermal_pressure_pa is not None:
        out["thermal_pressure [Pa]"] = np.array(thermal_pressure_pa, dtype=float)

    if object_velocity_xyz_m_s is not None:
        v_obj = np.array(object_velocity_xyz_m_s, dtype=float)
        if v_obj.shape != u.shape:
            raise ValueError("object_velocity_xyz_m_s must match u_xyz_m_s shape")
        rel = u - v_obj
        rel_speed = np.sqrt(np.sum(rel * rel, axis=-1))
        out["object_speed [m/s]"] = np.sqrt(np.sum(v_obj * v_obj, axis=-1))
        out["relative_speed [m/s]"] = rel_speed
        out["relative_ram_pressure [Pa]"] = ram_pressure(rho, rel_speed)
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

