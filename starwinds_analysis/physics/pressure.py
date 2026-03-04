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
    Used by: `starwinds_analysis/physics/curve.py`, `starwinds_analysis/physics/orbit_surface.py`
    """
    return rho_kg_m3 * np.square(V_m_s)


def magnetospheric_standoff_distance(rho_kg_m3, V_m_s, *, b0_t: float = 0.7e-4):
    """
    Vidotto-style stand-off distance proxy from pressure balance.
    Used by: `starwinds_analysis/physics/curve.py`, `starwinds_analysis/physics/orbit_surface.py`
    """
    # TODO(griblet): If this proxy is kept as a reusable quantity, expose it through
    # SmartDs/griblet so orbit/surface diagnostics can request it directly in SI.
    p_ram = ram_pressure(rho_kg_m3, V_m_s)
    numer = (float(b0_t) ** 2) / (2.0 * MU0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.power(numer / p_ram, 1.0 / 6.0)
