"""Kepler orbit kinematics primitives.
"""

# It defines reusable local/scalar formulas without orbit sampling workflows or
# dataset access.


from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.constants import G as GRAVITATIONAL_CONSTANT
from scipy.constants import au as AU_M

log = logging.getLogger(__name__)


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
    Used by: `/Users/dagfev/Documents/starwinds/batwind/batwind/physics/orbit_surface.py`,
      `/Users/dagfev/Documents/starwinds/batwind/batwind/physics/curve.py`
    """
    a = float(semi_major_axis_m)
    m = float(star_mass_kg)
    if a <= 0:
        log.error("orbital_period failed: semi_major_axis_m=%g", a)
        raise ValueError("semi_major_axis_m must be > 0")
    if m <= 0:
        log.error("orbital_period failed: star_mass_kg=%g", m)
        raise ValueError("star_mass_kg must be > 0")
    log.debug("orbital_period semi_major_axis_m=%g star_mass_kg=%g", a, m)
    period = 2.0 * math.pi * math.sqrt(a**3 / (GRAVITATIONAL_CONSTANT * m))
    log.debug("orbital_period computed period=%g", period)
    return period


def orbital_velocity(radial_distance_m, star_mass_kg, semi_major_axis_m):
    """
    Vis-viva orbital speed.
    Used by: `test/test_orbit_analysis.py`
    """
    r = np.array(radial_distance_m)
    m = float(star_mass_kg)
    a = float(semi_major_axis_m)
    if m <= 0:
        log.error("orbital_velocity failed: star_mass_kg=%g", m)
        raise ValueError("star_mass_kg must be > 0")
    if a <= 0:
        log.error("orbital_velocity failed: semi_major_axis_m=%g", a)
        raise ValueError("semi_major_axis_m must be > 0")
    log.debug(
        "orbital_velocity radial_shape=%s star_mass_kg=%g semi_major_axis_m=%g",
        r.shape,
        m,
        a,
    )
    with np.errstate(invalid="ignore"):
        v = np.sqrt(GRAVITATIONAL_CONSTANT * m * (2.0 / r - 1.0 / a))
    non_finite = int(np.count_nonzero(~np.isfinite(v)))
    if non_finite > 0:
        log.warning("orbital_velocity output has %d/%d non-finite values", non_finite, v.size)
    log.debug("orbital_velocity complete shape=%s", v.shape)
    return v
