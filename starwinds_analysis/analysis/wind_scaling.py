"""THIS FILE contains wind-scaling helper formulas.

It computes derived scaling quantities (for example escape speed and open-wind magnetisation) from inputs/profiles.
It should stay formula-focused and not perform dataset sampling.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.constants as c


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


def open_wind_magnetisation_from_profiles(
    diagnostics,
    *,
    star_mass_kg,
    star_radius_m,
):
    """
    Compute `Upsilon_open` along shell radii using diagnostics bundle entries.
    """
    if "open_flux" not in diagnostics or "mass_loss" not in diagnostics:
        raise KeyError("diagnostics must include 'open_flux' and 'mass_loss'")

    phi = np.asarray(diagnostics["open_flux"]["open_flux [Wb]"], dtype=float)
    dotm = np.asarray(diagnostics["mass_loss"]["mass_loss [kg/s]"], dtype=float)
    y = open_wind_magnetisation(phi, dotm, star_mass_kg, star_radius_m)
    return {
        "radius [R]": np.asarray(diagnostics["mass_loss"]["radius [R]"], dtype=float),
        "height [R]": np.asarray(diagnostics["mass_loss"]["height [R]"], dtype=float),
        "Upsilon_open [none]": np.asarray(y, dtype=float),
    }


__all__ = [
    "MU0",
    "open_wind_magnetisation",
    "open_wind_magnetisation_from_profiles",
    "surface_escape_speed",
]

