"""THIS FILE contains local flux-density formulas.

It defines pointwise flux-density combinations on arrays (no sampling/integration).
"""

from __future__ import annotations

import numpy as np


def radial_advective_flux_density(density_like, u_radial_m_s):
    """
    Advective radial flux density `q * u_r`.

    Examples:
    - `q = rho [kg/m^3]` -> mass flux density `[kg/m^2/s]`
    - `q = E [J/m^3]`    -> energy flux density `[W/m^2]`
    """
    q = np.array(density_like, dtype=float)
    u_r = np.array(u_radial_m_s, dtype=float)
    return q * u_r


__all__ = ["radial_advective_flux_density"]
