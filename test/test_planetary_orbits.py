import numpy as np
from scipy import constants as const

from starwinds_analysis.physics.orbits import orbital_period


def test_planet_orbit_period_is_reasonable():
    # Mirrors the intent of the old batplotlib test_period using canonical
    # Kepler primitive + local semi-major axes (in AU).
    expected_years = {"Mercury": 0.240846, "Venus": 0.615, "Earth": 1.0, "Mars": 1.881}
    semi_major_axes_au = {"Mercury": 0.387098, "Venus": 0.723332, "Earth": 1.00000102, "Mars": 1.523679}
    for name, expected in expected_years.items():
        p = orbital_period(semi_major_axes_au[name] * const.au, 1.98847e30)
        assert np.isclose(p / const.year, expected, rtol=5e-3), name
