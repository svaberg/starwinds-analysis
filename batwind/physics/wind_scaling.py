"""Wind-scaling formulas.
"""

# It defines local/scalar array formulas (for example escape speed and open-wind
# magnetisation) without profile-bundle orchestration.


# DONE(debt): Profile-bundle/orchestration helper was removed; keep only local formulas
# in this module.
# DONE(debt): `MU0` is imported from `batwind.constants` (single shared source).

from __future__ import annotations

import logging
import math

import numpy as np
import scipy.constants as c

from batwind.constants import MU0

log = logging.getLogger(__name__)

# DONE(debt): Reuse the shared `MU0` constant from `batwind.constants`.


def surface_escape_speed(star_mass_kg, star_radius_m):
    """
    Surface escape speed `sqrt(2GM/R)`.
    Used by: `test/test_shell_analysis.py`, `batwind/physics/wind_scaling.py`
    """
    # TODO(griblet): If stellar mass/radius are carried as SmartDs/aux quantities,
    # this should be requestable as a derived SI quantity instead of recomputed.
    m = np.array(star_mass_kg)
    r = np.array(star_radius_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(2.0 * c.G * m / r)
    non_finite = int(np.count_nonzero(~np.isfinite(out)))
    if non_finite > 0:
        log.warning("surface_escape_speed output has %d/%d non-finite values", non_finite, int(np.size(out)))
    return out


def open_wind_magnetisation(open_flux_wb, mass_loss_kg_s, star_mass_kg, star_radius_m):
    """
    Reville-style open wind magnetisation used in the old quicklook (`Upsilon_open`).
    Used by: `test/test_shell_analysis.py`, `batwind/pipelines/slice.py`, `batwind/pipelines/volume.py`
    """
    # TODO(griblet): `Upsilon_open` is a derived physical quantity and should be
    # requestable via SmartDs/griblet when its SI inputs are present.
    phi = np.array(open_flux_wb)
    dotm = np.array(mass_loss_kg_s)
    r = np.array(star_radius_m)
    vesc = surface_escape_speed(star_mass_kg, star_radius_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (4.0 * math.pi / MU0) * phi * phi / (r * r * dotm * vesc)
    non_finite = int(np.count_nonzero(~np.isfinite(out)))
    if non_finite > 0:
        log.warning("open_wind_magnetisation output has %d/%d non-finite values", non_finite, int(np.size(out)))
    return out
