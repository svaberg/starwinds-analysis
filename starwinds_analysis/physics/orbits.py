"""THIS FILE contains Kepler orbit kinematics primitives.

It defines reusable local/scalar formulas (period, vis-viva speed) without orbit
sampling workflows or dataset access.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.constants import G as GRAVITATIONAL_CONSTANT

def orbital_period(semi_major_axis_m, star_mass_kg):
    """
    Keplerian orbital period for a test particle around a point mass.
    """
    a = float(semi_major_axis_m)
    m = float(star_mass_kg)
    if a <= 0:
        raise ValueError("semi_major_axis_m must be > 0")
    if m <= 0:
        raise ValueError("star_mass_kg must be > 0")
    return 2.0 * math.pi * math.sqrt(a**3 / (GRAVITATIONAL_CONSTANT * m))

def orbital_velocity(radial_distance_m, star_mass_kg, semi_major_axis_m):
    """
    Vis-viva orbital speed.
    """
    r = np.array(radial_distance_m)
    m = float(star_mass_kg)
    a = float(semi_major_axis_m)
    if m <= 0:
        raise ValueError("star_mass_kg must be > 0")
    if a <= 0:
        raise ValueError("semi_major_axis_m must be > 0")
    with np.errstate(invalid="ignore"):
        return np.sqrt(GRAVITATIONAL_CONSTANT * m * (2.0 / r - 1.0 / a))

