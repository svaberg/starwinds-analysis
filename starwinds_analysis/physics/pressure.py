"""THIS FILE contains core pressure and standoff formulas.

These are quantity definitions (ram, magnetic, component combinations, standoff distance).
It should stay as pure math on arrays/scalars, not dataset-specific logic.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.constants import MU0


def ram_pressure(rho_kg_m3, V_m_s):
    """
    Ram pressure `rho * V^2` in Pa.
    Used by: `test/test_orbit_pressure.py`, `starwinds_analysis/physics/pressure.py`,
      `starwinds_analysis/physics/orbit_pressure.py`
    """
    return rho_kg_m3 * np.square(V_m_s)


def pressure_components(
    rho_kg_m3,
    U_xyz_m_s,
    B_xyz_t,
    *,
    thermal_pressure_pa=None,
    V_xyz_m_s=None,
):
    """
    Compute thermal/magnetic/ram pressure components from local samples.
    Used by: `test/test_orbit_pressure.py`, `starwinds_analysis/physics/orbit_surface.py`
    """
    # TODO(griblet): The derived quantities assembled here (`|U|`, `|B|`,
    # magnetic/ram/relative pressures) should come from SmartDs/griblet requests in
    # SI units instead of being bundled/computed here.
    U = U_xyz_m_s
    B = B_xyz_t
    U_shape = np.shape(U)
    B_shape = np.shape(B)
    if not U_shape or not B_shape or U_shape[-1] != 3 or B_shape[-1] != 3:
        raise ValueError("U_xyz_m_s and B_xyz_t must have shape (..., 3)")

    U_m_s = np.sqrt(np.sum(np.square(U), axis=-1))
    B_t = np.sqrt(np.sum(np.square(B), axis=-1))
    out = {
        "U [m/s]": U_m_s,
        "B [T]": B_t,
        "magnetic_pressure [Pa]": np.square(B_t) / (2.0 * MU0),
        "ram_pressure [Pa]": ram_pressure(rho_kg_m3, U_m_s),
    }

    if thermal_pressure_pa is not None:
        out["thermal_pressure [Pa]"] = thermal_pressure_pa

    if V_xyz_m_s is not None:
        V = V_xyz_m_s
        if np.shape(V) != U_shape:
            raise ValueError("V_xyz_m_s must match U_xyz_m_s shape")
        U_minus_V = np.subtract(U, V)
        V_m_s = np.sqrt(np.sum(np.square(V), axis=-1))
        U_minus_V_m_s = np.sqrt(np.sum(np.square(U_minus_V), axis=-1))
        out["V [m/s]"] = V_m_s
        out["U_minus_V [m/s]"] = U_minus_V_m_s
        out["relative_ram_pressure [Pa]"] = ram_pressure(rho_kg_m3, U_minus_V_m_s)
    return out


def magnetospheric_standoff_distance(rho_kg_m3, V_m_s, *, b0_t: float = 0.7e-4):
    """
    Vidotto-style stand-off distance proxy from pressure balance.
    Used by: `test/test_orbit_pressure.py`, `starwinds_analysis/physics/orbit_surface.py`,
      `starwinds_analysis/physics/orbit_pressure.py`
    """
    # TODO(griblet): If this proxy is kept as a reusable quantity, expose it through
    # SmartDs/griblet so orbit/surface diagnostics can request it directly in SI.
    p_ram = ram_pressure(rho_kg_m3, V_m_s)
    numer = (float(b0_t) ** 2) / (2.0 * MU0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.power(numer / p_ram, 1.0 / 6.0)
