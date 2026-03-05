"""Kepler orbit kinematics primitives.
"""

# It defines reusable local/scalar formulas without orbit sampling workflows or
# dataset access.


from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.constants import G as GRAVITATIONAL_CONSTANT
from scipy.constants import au as AU_M


@dataclass(frozen=True)
class PlanetOrbitElements:
    """Named Kepler elements for common Solar-system reference orbits."""
    semi_major_axis_m: float
    eccentricity: float
    argument_of_periapsis_deg: float = 0.0
    inclination_deg: float = 0.0


SOLAR_SYSTEM_PLANETS: dict[str, PlanetOrbitElements] = {
    "Mercury": PlanetOrbitElements(0.387098 * AU_M, 0.205630, 0.0, 3.38),
    "Venus": PlanetOrbitElements(0.723332 * AU_M, 0.006772, 0.0, 3.86),
    "Earth": PlanetOrbitElements(1.00000102 * AU_M, 0.0167086, 288.1, 7.155),
    "Mars": PlanetOrbitElements(1.523679 * AU_M, 0.0934, 0.0, 5.65),
}

def orbital_period(semi_major_axis_m, star_mass_kg):
    """
    Keplerian orbital period for a test particle around a point mass.
    Used by: `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/physics/orbit_surface.py`,
      `/Users/dagfev/Documents/starwinds/starwinds-analysis/starwinds_analysis/physics/curve.py`
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
    Used by: `test/test_orbit_analysis.py`
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
