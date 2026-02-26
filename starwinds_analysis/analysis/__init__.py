"""THIS FILE contains the public re-export surface for analysis helpers.

It groups stable analysis functions in one import location.
It should not implement analysis logic itself.
"""

from .fluxes import (
    axisymmetric_open_flux_vs_radius,
    energy_flux_vs_radius,
    open_magnetic_flux_vs_radius,
)
from ..physics.local_estimates import (
    local_mass_loss_estimates,
    local_torque_estimates,
    summarize_samples,
)
from .orbits import (
    circular_orbit_points,
    elliptic_orbit_points,
    local_mass_loss_on_circular_orbit,
    local_mass_loss_on_elliptic_orbit,
    local_torque_on_circular_orbit,
    local_torque_on_elliptic_orbit,
    orbital_period,
    orbital_velocity,
    sample_circular_orbit,
    sample_elliptic_orbit,
    sample_points,
)
from ..physics.pressure import (
    magnetic_pressure,
    magnetospheric_standoff_distance,
    pressure_components,
    ram_pressure,
)
from ..physics.mass_loss import (
    ShellMassFluxMap,
    mass_loss_vs_radius,
    sample_shell_mass_flux_map,
)
from .shells import (
    SphericalShellSamples,
    infer_body_radius_m,
    integrate_shell_scalar,
    sample_spherical_shells,
    sample_spherical_shells_fibonacci,
)
from .shell_summary import (
    boxcar_shell_weights,
    summarize_shell_diagnostics_band,
    summarize_shell_series,
)
from .slices import infer_range, resample_structured_xz_slice, structured_quad_corners
from .stats import weighted_mean_std, weighted_quantile
from ..physics.shell_torque import torque_vs_radius
from ..physics.surface_torque import (
    normalize_surface_normals,
    radial_surface_normals,
    rotational_frame_velocity,
    surface_torque_density_terms,
)
from .surface_torque import (
    integrate_surface_torque_terms,
    surface_torque_terms_on_shell_samples,
    surface_torque_vs_radius,
)
from ..physics.wind_scaling import (
    open_wind_magnetisation,
    surface_escape_speed,
)

__all__ = [
    "SphericalShellSamples",
    "infer_body_radius_m",
    "integrate_shell_scalar",
    "sample_spherical_shells",
    "sample_spherical_shells_fibonacci",
    "structured_quad_corners",
    "infer_range",
    "resample_structured_xz_slice",
    "boxcar_shell_weights",
    "summarize_shell_series",
    "summarize_shell_diagnostics_band",
    "ShellMassFluxMap",
    "sample_shell_mass_flux_map",
    "weighted_mean_std",
    "weighted_quantile",
    "open_magnetic_flux_vs_radius",
    "axisymmetric_open_flux_vs_radius",
    "energy_flux_vs_radius",
    "circular_orbit_points",
    "elliptic_orbit_points",
    "sample_points",
    "sample_circular_orbit",
    "sample_elliptic_orbit",
    "local_mass_loss_on_circular_orbit",
    "local_mass_loss_on_elliptic_orbit",
    "local_torque_on_circular_orbit",
    "local_torque_on_elliptic_orbit",
    "orbital_period",
    "orbital_velocity",
    "local_mass_loss_estimates",
    "local_torque_estimates",
    "summarize_samples",
    "magnetic_pressure",
    "ram_pressure",
    "pressure_components",
    "magnetospheric_standoff_distance",
    "surface_escape_speed",
    "open_wind_magnetisation",
    "mass_loss_vs_radius",
    "torque_vs_radius",
    "rotational_frame_velocity",
    "radial_surface_normals",
    "surface_torque_density_terms",
    "integrate_surface_torque_terms",
    "surface_torque_terms_on_shell_samples",
    "surface_torque_vs_radius",
]
