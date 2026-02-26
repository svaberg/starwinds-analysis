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
    magnetic = -cylindrical_radius_m * b_azimuthal_t * b_radial_t / MU0
    dynamic = cylindrical_radius_m * rho_kg_m3 * u_azimuthal_m_s * u_radial_m_s
    return magnetic, dynamic
