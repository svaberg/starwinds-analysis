"""THIS FILE contains wind-scaling pipeline helpers.

It maps diagnostics/profile bundles onto local wind-scaling formulas.
The formulas themselves belong in `starwinds_analysis.physics.wind_scaling`.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.physics.wind_scaling import open_wind_magnetisation


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


__all__ = ["open_wind_magnetisation_from_profiles"]
