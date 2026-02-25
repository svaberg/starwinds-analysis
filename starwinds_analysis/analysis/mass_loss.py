from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    integrate_shell_scalar,
    resolve_batsrus_density_si,
    resolve_batsrus_vector_xyz_si,
    sample_spherical_shells_by_strategy,
    shell_profile_radius_height,
)
from starwinds_analysis.recipes.spherical import spherical_vector_components


def mass_loss_vs_radius(
    smart_ds,
    radii,
    *,
    body_radius_m: float | None = None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_polar: int = 24,
    n_azimuth: int = 48,
    sampling: str = "fibonacci",
    fibonacci_randomize: bool = False,
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Wind mass-loss profile on spherical shells.

    Returns a dict with SI mass-loss values and shell coverage fractions.
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")

    shells = sample_spherical_shells_by_strategy(
        smart_ds,
        radii,
        fields=(rho_name, ux_name, uy_name, uz_name),
        coordinate_fields=coordinate_fields,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        sampling=sampling,
        fibonacci_randomize=fibonacci_randomize,
        method=method,
        fill_value=fill_value,
        length_unit_to_m=body_radius_m,
    )

    rho = rho_scale * shells.fields[rho_name]
    ux = u_scale * shells.fields[ux_name]
    uy = u_scale * shells.fields[uy_name]
    uz = u_scale * shells.fields[uz_name]

    u_r, _u_theta, _u_phi = spherical_vector_components(ux, uy, uz, shells.x, shells.y, shells.z)
    mass_flux = rho * u_r  # kg / m^2 / s

    mass_loss, coverage = integrate_shell_scalar(mass_flux, shells.area)
    return {
        **shell_profile_radius_height(shells),
        "mass_loss [kg/s]": np.asarray(mass_loss, dtype=float),
        "coverage [none]": np.asarray(coverage, dtype=float),
        "shell_samples": shells,
    }


def plot_mass_loss_profile(ax, profile, *, show_negative=True):
    radii = np.asarray(profile["radius [R]"], dtype=float)
    values = np.asarray(profile["mass_loss [kg/s]"], dtype=float)
    x = radii - 1.0
    ax.plot(x, values, ".-", color="C0", label="mass loss")
    if show_negative:
        ax.plot(x, -values, ".--", color="C0", fillstyle="none")
    ax.set_xlabel("Height over surface [R]")
    ax.set_ylabel("Mass flux [kg/s]")
    return ax


__all__ = ["mass_loss_vs_radius", "plot_mass_loss_profile"]
