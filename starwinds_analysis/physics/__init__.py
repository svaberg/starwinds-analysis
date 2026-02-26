"""THIS FILE contains the public re-export surface for local physics formulas.

It groups pointwise/domain formulas (no sampling, no plotting, no notebook orchestration).
It should not depend on SmartDs or analysis pipelines.
"""

# DONE(debt): Keep the `physics` re-export surface limited to local formulas and
# constants; profile-derived helper exports were removed.

from .constants import MU0
from .pressure import (
    magnetic_pressure,
    magnetospheric_standoff_distance,
    pressure_components,
    ram_pressure,
)
from .wind_scaling import (
    open_wind_magnetisation,
    surface_escape_speed,
)

__all__ = [
    "MU0",
    "magnetic_pressure",
    "magnetospheric_standoff_distance",
    "pressure_components",
    "ram_pressure",
    "surface_escape_speed",
    "open_wind_magnetisation",
]
