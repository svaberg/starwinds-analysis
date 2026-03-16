"""Diagnostics evaluated on sampled curves and trajectories.
"""

# It operates on already sampled curve `SmartDs` objects. Curve geometry belongs
# in `analysis/trajectories.py`. Pressure formulas belong in `pressure.py`.


from __future__ import annotations

import logging

import numpy as np

from batwind.physics.pressure import magnetospheric_standoff_distance
from batwind.physics.pressure import ram_pressure

log = logging.getLogger(__name__)


def mass_loss_from_curve(curve):
    """Compute local mass-loss estimates along a sampled curve."""
    mass_flux = np.array(curve("mass_flux [kg/m^2/s]"))
    shell_area = 4.0 * np.pi * np.square(np.array(curve("R [m]")))
    out = shell_area * mass_flux
    log.info("mass_loss_from_curve done n=%d", out.size)
    return out


def torque_from_curve(curve):
    """Compute local magnetic, dynamic, and total torque estimates along a curve."""
    shell_area = 4.0 * np.pi * np.square(np.array(curve("R [m]")))
    magnetic_torque_density = np.array(curve("magnetic_torque_density [N/m]"))
    dynamic_torque_density = np.array(curve("dynamic_torque_density [N/m]"))
    magnetic = shell_area * magnetic_torque_density
    dynamic = shell_area * dynamic_torque_density
    log.info("torque_from_curve done n=%d", magnetic.size)
    return magnetic, dynamic, magnetic + dynamic


def relative_ram_pressure_from_trajectory(
    trajectory,
    *,
    standoff_b0: float = 0.7e-4,
):
    """Compute trajectory-frame ram pressure and standoff distance."""
    log.info("relative_ram_pressure_from_trajectory start")
    rho = np.array(trajectory("Rho [kg/m^3]"))
    U_xyz = np.array(trajectory("U_xyz [m/s]"))
    V_xyz = np.array(trajectory("V_xyz [m/s]"))
    if U_xyz.shape != V_xyz.shape:
        log.error("relative_ram_pressure_from_trajectory failed: U_xyz shape=%s V_xyz shape=%s", U_xyz.shape, V_xyz.shape)
        raise ValueError("U_xyz and V_xyz must have matching shapes")
    U_minus_V = U_xyz - V_xyz
    U_minus_V_speed = np.sqrt(np.sum(U_minus_V * U_minus_V, axis=-1))
    relative_ram_pressure = ram_pressure(rho, U_minus_V_speed)
    standoff_distance = magnetospheric_standoff_distance(
        rho,
        U_minus_V_speed,
        b0=standoff_b0,
    )
    log.info("relative_ram_pressure_from_trajectory done n=%d", relative_ram_pressure.size)
    return relative_ram_pressure, standoff_distance
