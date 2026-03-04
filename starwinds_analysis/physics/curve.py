"""THIS FILE contains diagnostics evaluated on sampled curves and trajectories.

It operates on already sampled curve `SmartDs` objects. Curve geometry belongs
in `analysis/orbits.py`. Pressure formulas belong in `pressure.py`.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.physics.pressure import magnetospheric_standoff_distance
from starwinds_analysis.physics.pressure import ram_pressure


def mass_loss_from_curve(curve):
    """Compute local mass-loss estimates along a sampled curve."""
    mass_flux = np.array(curve("mass_flux [kg/m^2/s]"))
    shell_area = 4.0 * np.pi * np.square(np.array(curve("R [m]")))
    return shell_area * mass_flux


def torque_from_curve(curve):
    """Compute local magnetic, dynamic, and total torque estimates along a curve."""
    shell_area = 4.0 * np.pi * np.square(np.array(curve("R [m]")))
    magnetic_torque_density = np.array(curve("magnetic_torque_density [N/m]"))
    dynamic_torque_density = np.array(curve("dynamic_torque_density [N/m]"))
    magnetic = shell_area * magnetic_torque_density
    dynamic = shell_area * dynamic_torque_density
    return magnetic, dynamic, magnetic + dynamic


def relative_ram_pressure_from_trajectory(
    trajectory,
    *,
    standoff_b0: float = 0.7e-4,
):
    """Compute trajectory-frame ram pressure and standoff distance."""
    rho = np.array(trajectory("Rho [kg/m^3]"))
    U_xyz = np.array(trajectory("U_xyz [m/s]"))
    V_xyz = np.array(trajectory("V_xyz [m/s]"))
    U_minus_V = U_xyz - V_xyz
    U_minus_V_speed = np.sqrt(np.sum(U_minus_V * U_minus_V, axis=-1))
    relative_ram_pressure = ram_pressure(rho, U_minus_V_speed)
    standoff_distance = magnetospheric_standoff_distance(
        rho,
        U_minus_V_speed,
        b0=standoff_b0,
    )
    return relative_ram_pressure, standoff_distance
