"""THIS FILE contains named planetary orbital constants only.

It is a data/constants layer and should not contain sampling or convenience wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from scipy.constants import au as AU_M

@dataclass(frozen=True)
class PlanetOrbitElements:
    semi_major_axis_m: float
    eccentricity: float
    argument_of_periapsis_deg: float = 0.0
    inclination_deg: float = 0.0

# Values copied from old batplotlib elliptic_orbit.py presets.
SOLAR_SYSTEM_PLANETS: dict[str, PlanetOrbitElements] = {
    "Mercury": PlanetOrbitElements(0.387098 * AU_M, 0.205630, 0.0, 3.38),
    "Venus": PlanetOrbitElements(0.723332 * AU_M, 0.006772, 0.0, 3.86),
    "Earth": PlanetOrbitElements(1.00000102 * AU_M, 0.0167086, 288.1, 7.155),
    "Mars": PlanetOrbitElements(1.523679 * AU_M, 0.0934, 0.0, 5.65),
}
