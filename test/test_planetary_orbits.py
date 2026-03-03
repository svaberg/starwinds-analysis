import numpy as np
from scipy import constants as const

from starwinds_analysis.physics.planetary_orbits import PlanetOrbitElements
from starwinds_analysis.physics.planetary_orbits import SOLAR_SYSTEM_PLANETS
from starwinds_analysis.physics.orbits import orbital_period


def test_planet_table_has_known_names():
    for name in ("Mercury", "Venus", "Earth", "Mars"):
        elem = SOLAR_SYSTEM_PLANETS[name]
        assert isinstance(elem, PlanetOrbitElements)
        assert elem.semi_major_axis_m > 0
        assert 0 <= elem.eccentricity < 1


def test_planet_orbit_period_is_reasonable():
    # Mirrors the intent of the old batplotlib test_period using the canonical
    # Kepler primitive + planet constants table.
    expected_years = {"Mercury": 0.240846, "Venus": 0.615, "Earth": 1.0, "Mars": 1.881}
    for name, expected in expected_years.items():
        p = orbital_period(SOLAR_SYSTEM_PLANETS[name].semi_major_axis_m, 1.98847e30)
        assert np.isclose(p / const.year, expected, rtol=5e-3), name
