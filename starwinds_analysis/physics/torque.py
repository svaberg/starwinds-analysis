"""THIS FILE contains local torque-density formulas.

It defines pointwise torque-density terms and constants, without shell/surface sampling
or integration orchestration.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.physics.constants import MU0


def spherical_wind_torque_density_terms(
    *,
    rho_kg_m3,
    u_radial_m_s,
    u_azimuthal_m_s,
    b_radial_t,
    b_azimuthal_t,
    cylindrical_radius_m,
):
    """
    Spherical-shell wind torque-density terms about +z.

    Returns `(magnetic_density, dynamic_density)` with units `N/m`.
    """
    # TODO(griblet): These local spherical torque-density terms should be available
    # via SmartDs/griblet for SI fields, instead of being recomputed in shell/orbit
    # diagnostics.
    rho = np.array(rho_kg_m3, dtype=float)
    u_r = np.array(u_radial_m_s, dtype=float)
    u_phi = np.array(u_azimuthal_m_s, dtype=float)
    b_r = np.array(b_radial_t, dtype=float)
    b_phi = np.array(b_azimuthal_t, dtype=float)
    varpi = np.array(cylindrical_radius_m, dtype=float)
    magnetic = -varpi * b_phi * b_r / MU0
    dynamic = varpi * rho * u_phi * u_r
    return magnetic, dynamic


__all__ = ["MU0", "spherical_wind_torque_density_terms"]
