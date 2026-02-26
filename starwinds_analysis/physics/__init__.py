"""THIS FILE contains the public re-export surface for local physics formulas.

It groups pointwise/domain formulas (no sampling, no plotting, no notebook orchestration).
It should not depend on SmartDs or analysis pipelines.
"""

# TODO(debt): This re-export surface currently includes non-local helpers (for example
# profile-derived wind-scaling helpers). Keep the `physics` API small and limited to
# local formulas/quantities.

from .pressure import (
    MU0,
    magnetic_pressure,
    magnetospheric_standoff_distance,
    pressure_components,
    ram_pressure,
)
from .wind_scaling import (
    open_wind_magnetisation,
    open_wind_magnetisation_from_profiles,
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
    "open_wind_magnetisation_from_profiles",
]
