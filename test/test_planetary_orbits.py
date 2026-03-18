import numpy as np
from scipy import constants as const

from batwind.physics.orbits import PlanetOrbitElements
from batwind.physics.orbits import SOLAR_SYSTEM_PLANETS
from batwind.physics.orbits import orbital_period


def test_planet_orbit_period_is_reasonable():
    expected_years = {"Mercury": 0.240846, "Venus": 0.615, "Earth": 1.0, "Mars": 1.881}
    for name, expected in expected_years.items():
        elem = SOLAR_SYSTEM_PLANETS[name]
        assert isinstance(elem, PlanetOrbitElements)
        p = orbital_period(elem.semi_major_axis_m, 1.98847e30)
        assert np.isclose(p / const.year, expected, rtol=5e-3), name
