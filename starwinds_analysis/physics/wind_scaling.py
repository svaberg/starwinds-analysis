"""THIS FILE contains wind-scaling formulas.

It defines local/scalar array formulas (for example escape speed and open-wind
magnetisation) without profile-bundle orchestration.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.constants as c

# TODO why is this not taken from scipy.constants? 
MU0 = 4.0e-7 * math.pi


def surface_escape_speed(star_mass_kg, star_radius_m):
    """
    Surface escape speed `sqrt(2GM/R)`.
    """
    m = np.asarray(star_mass_kg, dtype=float)
    r = np.asarray(star_radius_m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(2.0 * c.G * m / r)


def open_wind_magnetisation(open_flux_wb, mass_loss_kg_s, star_mass_kg, star_radius_m):
    """
    Reville-style open wind magnetisation used in the old quicklook (`Upsilon_open`).
    """
    phi = np.asarray(open_flux_wb, dtype=float)
    dotm = np.asarray(mass_loss_kg_s, dtype=float)
    r = np.asarray(star_radius_m, dtype=float)
    vesc = surface_escape_speed(star_mass_kg, star_radius_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (4.0 * math.pi / MU0) * phi * phi / (r * r * dotm * vesc)


__all__ = [
    "MU0",
    "open_wind_magnetisation",
    "surface_escape_speed",
]
