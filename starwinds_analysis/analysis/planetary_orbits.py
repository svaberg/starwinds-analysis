"""THIS FILE contains named planetary orbital constants and convenience orbit specs.

It is a data/constants layer plus lightweight helpers.
It should not contain SmartDs access, resampling, or plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.constants import au as AU_M

from starwinds_analysis.analysis.orbits import orbital_period


@dataclass(frozen=True)
class PlanetOrbitElements:
    semi_major_axis_m: float
    eccentricity: float
    argument_of_periapsis_deg: float = 0.0
    inclination_deg: float = 0.0


# Values copied from the old batplotlib elliptic_orbit.py presets.
SOLAR_SYSTEM_PLANETS: dict[str, PlanetOrbitElements] = {
    "Mercury": PlanetOrbitElements(0.387098 * AU_M, 0.205630, 0.0, 3.38),
    "Venus": PlanetOrbitElements(0.723332 * AU_M, 0.006772, 0.0, 3.86),
    "Earth": PlanetOrbitElements(1.00000102 * AU_M, 0.0167086, 288.1, 7.155),
    "Mars": PlanetOrbitElements(1.523679 * AU_M, 0.0934, 0.0, 5.65),
}


def get_planet_orbit_elements(name: str) -> PlanetOrbitElements:
    try:
        return SOLAR_SYSTEM_PLANETS[str(name)]
    except KeyError as exc:
        raise KeyError(f"Unknown planet '{name}'. Available: {sorted(SOLAR_SYSTEM_PLANETS)}") from exc


def planet_orbit_spec(
    name: str,
    *,
    star_radius_m: float,
    label: str | None = None,
    n_points: int = 180,
    sample: str = "eccentric_anomaly",
    plane: str = "xy",
    include_orientation: bool = False,
):
    """
    Build a `run_quicklook2d` orbit spec dict for a named planet, in stellar radii.

    By default the orbit is projected into the requested Cartesian plane for 2D/3D
    sampling workflows. `include_orientation=True` maps the old argument of periapsis
    to `angle0` within that plane (inclination is not yet applied).
    """
    star_radius_m = float(star_radius_m)
    if star_radius_m <= 0:
        raise ValueError("star_radius_m must be > 0")
    elem = get_planet_orbit_elements(name)
    spec = {
        "label": str(name) if label is None else str(label),
        "semi_major_axis": float(elem.semi_major_axis_m / star_radius_m),
        "eccentricity": float(elem.eccentricity),
        "n_points": int(n_points),
        "sample": str(sample),
        "plane": str(plane),
    }
    if include_orientation:
        spec["angle0"] = float(elem.argument_of_periapsis_deg) * 3.141592653589793 / 180.0
    return spec


def planet_orbit_period(name: str, *, star_mass_kg: float):
    """
    Keplerian period for a named planet around a star of mass `star_mass_kg`.
    """
    elem = get_planet_orbit_elements(name)
    return orbital_period(elem.semi_major_axis_m, star_mass_kg)


__all__ = [
    "AU_M",
    "PlanetOrbitElements",
    "SOLAR_SYSTEM_PLANETS",
    "get_planet_orbit_elements",
    "planet_orbit_spec",
    "planet_orbit_period",
]

