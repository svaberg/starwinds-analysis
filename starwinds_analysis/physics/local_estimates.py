"""THIS FILE contains local analytic estimators for mass loss and torque.

These are pointwise formulas (no resampling, no plotting, no summaries).
"""

from __future__ import annotations

import math

import numpy as np

from starwinds_analysis.physics.constants import MU0

def local_mass_loss_estimates(radius_m, rho_kg_m3, u_radial_m_s):
    """
    Pointwise local mass-loss estimates using `4*pi*r^2*rho*u_r`.
    """
    # TODO(griblet): This local estimate should be a griblet/SmartDs quantity when
    # the required SI inputs (`R`, `rho`, `U_r`) are available on samples.
    r = np.array(radius_m, dtype=float)
    rho = np.array(rho_kg_m3, dtype=float)
    u_r = np.array(u_radial_m_s, dtype=float)
    return 4.0 * math.pi * r * r * rho * u_r

def local_torque_estimates(radius_m, rho_kg_m3, u_radial_m_s, u_phi_m_s, b_r_t, b_phi_t):
    """
    Pointwise local torque estimates using the spherical-shell scaling from old quicklook.

    This mirrors the local approximation idea in the Tecplot quicklook path:
    - magnetic torque density term without the cylindrical factor
    - dynamic torque density term without the cylindrical factor
    - multiply by `∫ C dS = pi^2 r^3` for a sphere of radius `r`
    """
    # TODO(griblet): This local torque estimate bundle should move behind SmartDs/
    # griblet field requests once the required SI inputs (`R`, `U_r`, `U_phi`,
    # `B_r`, `B_phi`) are available as derived quantities.
    # TODO is said NO CASTING!
    r = np.array(radius_m, dtype=float)
    rho = np.array(rho_kg_m3, dtype=float)
    u_r = np.array(u_radial_m_s, dtype=float)
    u_phi = np.array(u_phi_m_s, dtype=float)
    b_r = np.array(b_r_t, dtype=float)
    b_phi = np.array(b_phi_t, dtype=float)

    rest_integral = (math.pi**2) * r**3
    magnetic = (-b_phi * b_r / MU0) * rest_integral
    dynamic = (u_phi * u_r * rho) * rest_integral
    total = magnetic + dynamic
    return {"magnetic [Nm]": magnetic, "dynamic [Nm]": dynamic, "total [Nm]": total}

