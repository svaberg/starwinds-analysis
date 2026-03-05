"""Core pressure and standoff formulas.
"""

# These are quantity definitions (ram, magnetic, component combinations, standoff distance).
# It should stay as pure math on arrays/scalars, not dataset-specific logic.


from __future__ import annotations

import logging

import numpy as np

from starwinds_analysis.constants import MU0

log = logging.getLogger(__name__)


def ram_pressure(rho, V):
    """
    Ram pressure `rho * V^2` in Pa.
    Used by: `starwinds_analysis/physics/curve.py`, `starwinds_analysis/physics/orbit_surface.py`
    """
    out = rho * np.square(V)
    non_finite = int(np.count_nonzero(~np.isfinite(out)))
    if non_finite > 0:
        log.warning("ram_pressure output has %d/%d non-finite values", non_finite, int(np.size(out)))
    return out


def magnetospheric_standoff_distance(rho, V, *, b0: float = 0.7e-4):
    """
    Vidotto-style stand-off distance proxy from pressure balance.
    Used by: `starwinds_analysis/physics/curve.py`, `starwinds_analysis/physics/orbit_surface.py`
    """
    # TODO(griblet): If this proxy is kept as a reusable quantity, expose it through
    # SmartDs/griblet so orbit/surface diagnostics can request it directly in SI.
    p_ram = ram_pressure(rho, V)
    numer = (float(b0) ** 2) / (2.0 * MU0)
    out = np.power(numer / p_ram, 1.0 / 6.0)
    non_finite = int(np.count_nonzero(~np.isfinite(out)))
    if non_finite > 0:
        log.warning(
            "magnetospheric_standoff_distance output has %d/%d non-finite values",
            non_finite,
            int(np.size(out)),
        )
    return out
