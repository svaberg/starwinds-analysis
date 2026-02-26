import numpy as np
from scipy import constants as const

from starwinds_analysis.physics.planetary_orbits import (
    SOLAR_SYSTEM_PLANETS,
    get_planet_orbit_elements,
    planet_orbit_period,
    planet_orbit_spec,
)


def test_get_planet_orbit_elements_known_names():
    for name in ("Mercury", "Venus", "Earth", "Mars"):
        elem = get_planet_orbit_elements(name)
        assert elem.semi_major_axis_m > 0
        assert 0 <= elem.eccentricity < 1


def test_planet_orbit_spec_converts_to_stellar_radii():
    star_radius_m = 6.957e8
    spec = planet_orbit_spec("Earth", star_radius_m=star_radius_m, n_points=96)
    expected = SOLAR_SYSTEM_PLANETS["Earth"].semi_major_axis_m / star_radius_m
    assert spec["label"] == "Earth"
    assert spec["n_points"] == 96
    assert np.isclose(spec["semi_major_axis"], expected)
    assert np.isclose(spec["eccentricity"], SOLAR_SYSTEM_PLANETS["Earth"].eccentricity)


def test_planet_orbit_period_is_reasonable():
    # Mirrors the intent of the old batplotlib test_period.
    expected_years = {"Mercury": 0.240846, "Venus": 0.615, "Earth": 1.0, "Mars": 1.881}
    for name, expected in expected_years.items():
        p = planet_orbit_period(name, star_mass_kg=1.98847e30)
        assert np.isclose(p / const.year, expected, rtol=5e-3), name
