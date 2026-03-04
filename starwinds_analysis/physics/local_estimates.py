"""THIS FILE contains local analytic estimators for mass loss and torque.

These are pointwise formulas (no resampling, no plotting, no summaries).
"""

from __future__ import annotations

import math

import numpy as np

def local_mass_loss_estimates(radius_m, mass_flux_kg_m2_s):
    """
    Pointwise local mass-loss estimates using `4*pi*r^2*mass_flux`.
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/physics/orbit_local.py`
    """
    return 4.0 * math.pi * np.square(radius_m) * mass_flux_kg_m2_s

def local_torque_estimates(radius_m, magnetic_torque_density_n_m, dynamic_torque_density_n_m):
    """
    Pointwise local torque estimates using torque-density fields and the old quicklook scaling.
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/physics/orbit_local.py`
    """
    rest_integral = (math.pi**2) * np.power(radius_m, 3)
    magnetic = magnetic_torque_density_n_m * rest_integral
    dynamic = dynamic_torque_density_n_m * rest_integral
    total = magnetic + dynamic
    return magnetic, dynamic, total
