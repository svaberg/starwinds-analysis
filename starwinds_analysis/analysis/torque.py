"""THIS FILE contains spherical-shell torque diagnostics and profile plotting wrappers.

It builds shell torque profiles from shell/surface primitives and returns SI outputs.
It should not duplicate the core explicit-surface torque definitions.
"""

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis._profile_plotting import (
    plot_shell_height_series,
    shell_profile_height,
)
from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    integrate_shell_scalar,
    resolve_batsrus_density_si,
    resolve_batsrus_vector_xyz_si,
    sample_spherical_shells_by_strategy,
    shell_profile_radius_height,
)
from starwinds_analysis.physics.torque import MU0, spherical_wind_torque_density_terms
from starwinds_analysis.recipes.spherical import spherical_vector_components


def torque_vs_radius(
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
    Spherical-shell wind torque profile (magnetic + dynamic + total).

    Uses the same shell-form terms as the old Tecplot-based `calculate_torque()`:
    - magnetic: `-varpi * B_phi * B_r / mu0`
    - dynamic:  ` varpi * rho * U_phi * U_r`
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")
    (bx_name, by_name, bz_name), b_scale = resolve_batsrus_vector_xyz_si(smart_ds, "B")

    shells = sample_spherical_shells_by_strategy(
        smart_ds,
        radii,
        fields=(rho_name, ux_name, uy_name, uz_name, bx_name, by_name, bz_name),
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
    bx = b_scale * shells.fields[bx_name]
    by = b_scale * shells.fields[by_name]
    bz = b_scale * shells.fields[bz_name]

    u_r, _u_theta, u_phi = spherical_vector_components(ux, uy, uz, shells.x, shells.y, shells.z)
    b_r, _b_theta, b_phi = spherical_vector_components(bx, by, bz, shells.x, shells.y, shells.z)

    varpi = np.sqrt(shells.x * shells.x + shells.y * shells.y) * body_radius_m

    magnetic_density, dynamic_density = spherical_wind_torque_density_terms(
        rho_kg_m3=rho,
        u_radial_m_s=u_r,
        u_azimuthal_m_s=u_phi,
        b_radial_t=b_r,
        b_azimuthal_t=b_phi,
        cylindrical_radius_m=varpi,
    )

    magnetic, cov_mag = integrate_shell_scalar(magnetic_density, shells.area)
    dynamic, cov_dyn = integrate_shell_scalar(dynamic_density, shells.area)
    total = magnetic + dynamic
    coverage = np.minimum(cov_mag, cov_dyn)

    return {
        **shell_profile_radius_height(shells),
        "magnetic_torque [Nm]": np.asarray(magnetic, dtype=float),
        "dynamic_torque [Nm]": np.asarray(dynamic, dtype=float),
        "total_torque [Nm]": np.asarray(total, dtype=float),
        "coverage [none]": np.asarray(coverage, dtype=float),
        "shell_samples": shells,
    }


def plot_torque_profile(ax, profile, *, show_negative=True):
    h = shell_profile_height(profile)
    mag = np.asarray(profile["magnetic_torque [Nm]"], dtype=float)
    dyn = np.asarray(profile["dynamic_torque [Nm]"], dtype=float)

    plot_shell_height_series(
        ax,
        profile,
        "total_torque [Nm]",
        label="total",
        ylabel="Torque [Nm]",
        color="C0",
        show_negative=show_negative,
    )
    ax.plot(h, mag, ".-", color="C1", label="magnetic")
    ax.plot(h, dyn, ".-", color="C2", label="dynamic")
    return ax


__all__ = ["MU0", "plot_torque_profile", "torque_vs_radius"]
