"""THIS FILE contains the public re-export surface for pipeline workflows.

Pipelines orchestrate reusable primitives into multi-step workflows.
They should stay smaller than notebooks/examples and avoid one-off convenience glue.
"""

from .orbit_pressure import (
    pressure_components_from_orbit_sample,
    pressure_components_on_circular_orbit,
    pressure_components_on_elliptic_orbit,
    resolve_batsrus_pressure_si,
)
from .orbit_surface import (
    pressure_components_on_orbit_surface,
    sample_orbit_surface_revolution,
    surface_point_normals_and_areas,
    surface_of_revolution_from_path,
    torque_components_on_orbit_surface,
)
from .wind_scaling import open_wind_magnetisation_from_profiles

__all__ = [
    "resolve_batsrus_pressure_si",
    "pressure_components_from_orbit_sample",
    "pressure_components_on_circular_orbit",
    "pressure_components_on_elliptic_orbit",
    "surface_of_revolution_from_path",
    "surface_point_normals_and_areas",
    "sample_orbit_surface_revolution",
    "pressure_components_on_orbit_surface",
    "torque_components_on_orbit_surface",
    "open_wind_magnetisation_from_profiles",
]
