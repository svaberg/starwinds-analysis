"""THIS FILE contains the public re-export surface for analysis helpers.

It groups stable analysis functions in one import location.
It should not implement analysis logic itself.
"""

from .fluxes import (
    axisymmetric_open_flux_vs_radius,
    energy_flux_vs_radius,
    open_magnetic_flux_vs_radius,
    plot_energy_flux_profile,
    plot_open_flux_profile,
)
from .local_estimates import (
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
from .orbit_pressure import (
    pressure_components_from_orbit_sample,
    pressure_components_on_circular_orbit,
    pressure_components_on_elliptic_orbit,
    resolve_batsrus_pressure_si,
)
from .planetary_orbits import (
    AU_M,
    PlanetOrbitElements,
    SOLAR_SYSTEM_PLANETS,
    get_planet_orbit_elements,
    planet_orbit_period,
    planet_orbit_spec,
)
from .orbit_surface import (
    pressure_components_on_orbit_surface,
    sample_orbit_surface_revolution,
    surface_point_normals_and_areas,
    surface_of_revolution_from_path,
    torque_components_on_orbit_surface,
)
from ..physics.pressure import (
    magnetic_pressure,
    magnetospheric_standoff_distance,
    pressure_components,
    ram_pressure,
)
from .mass_loss import mass_loss_vs_radius, plot_mass_loss_profile
from .mass_loss import (
    ShellMassFluxMap,
    plot_shell_mass_flux_lonlat,
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
from .shell_magnetic import (
    ShellMagneticFieldMap,
    plot_magnetic_zdi_triplet,
    plot_shell_scalar_lonlat,
    plot_shell_tangential_vectors_lonlat,
    sample_shell_magnetic_field_map,
    summarize_shell_magnetic_field_map,
    style_shell_lonlat_axes,
)
from .slices import infer_range, resample_structured_xz_slice, structured_quad_corners
from .stats import weighted_mean_std, weighted_quantile
from .torque import plot_torque_profile, torque_vs_radius
from .surface_torque import (
    integrate_surface_torque_terms,
    radial_surface_normals,
    rotational_frame_velocity,
    surface_torque_density_terms,
    surface_torque_terms_on_shell_samples,
    surface_torque_vs_radius,
)
from .wind_scaling import (
    open_wind_magnetisation,
    open_wind_magnetisation_from_profiles,
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
    "ShellMagneticFieldMap",
    "ShellMassFluxMap",
    "sample_shell_magnetic_field_map",
    "sample_shell_mass_flux_map",
    "style_shell_lonlat_axes",
    "plot_shell_scalar_lonlat",
    "plot_shell_mass_flux_lonlat",
    "plot_magnetic_zdi_triplet",
    "plot_shell_tangential_vectors_lonlat",
    "summarize_shell_magnetic_field_map",
    "weighted_mean_std",
    "weighted_quantile",
    "open_magnetic_flux_vs_radius",
    "axisymmetric_open_flux_vs_radius",
    "energy_flux_vs_radius",
    "plot_open_flux_profile",
    "plot_energy_flux_profile",
    "circular_orbit_points",
    "elliptic_orbit_points",
    "sample_points",
    "sample_circular_orbit",
    "sample_elliptic_orbit",
    "resolve_batsrus_pressure_si",
    "pressure_components_from_orbit_sample",
    "pressure_components_on_circular_orbit",
    "pressure_components_on_elliptic_orbit",
    "surface_of_revolution_from_path",
    "surface_point_normals_and_areas",
    "sample_orbit_surface_revolution",
    "pressure_components_on_orbit_surface",
    "torque_components_on_orbit_surface",
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
    "open_wind_magnetisation_from_profiles",
    "AU_M",
    "PlanetOrbitElements",
    "SOLAR_SYSTEM_PLANETS",
    "get_planet_orbit_elements",
    "planet_orbit_spec",
    "planet_orbit_period",
    "mass_loss_vs_radius",
    "plot_mass_loss_profile",
    "torque_vs_radius",
    "plot_torque_profile",
    "rotational_frame_velocity",
    "radial_surface_normals",
    "surface_torque_density_terms",
    "integrate_surface_torque_terms",
    "surface_torque_terms_on_shell_samples",
    "surface_torque_vs_radius",
]
