"""THIS FILE contains local analytic estimators for mass loss and torque.

These are pointwise formulas and sample summaries, not shell/surface integrators.
It should stay free of resampling and plotting orchestration.
"""

from __future__ import annotations

import math

import numpy as np

from starwinds_analysis.analysis.stats import weighted_mean_std, weighted_quantile
from starwinds_analysis.physics.torque import MU0


def local_mass_loss_estimates(radius_m, rho_kg_m3, u_radial_m_s):
    """
    Pointwise local mass-loss estimates using `4*pi*r^2*rho*u_r`.
    """
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


def summarize_samples(values, *, quantiles=(0.0, 0.25, 0.5, 0.75, 1.0), weights=None):
    """
    Small helper mirroring the old quicklook habit of logging quantiles + mean/std.
    """
    v = np.array(values, dtype=float)
    qv = weighted_quantile(v, quantiles, weights=weights)
    mean, std = weighted_mean_std(v, weights=weights)
    return {
        "quantiles": np.array(quantiles, dtype=float),
        "values": np.array(qv, dtype=float),
        "mean": float(mean),
        "std": float(std),
    }


__all__ = [
    "local_mass_loss_estimates",
    "local_torque_estimates",
    "summarize_samples",
]
