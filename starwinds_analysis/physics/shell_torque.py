"""THIS FILE contains spherical-shell wind torque diagnostics.

It computes shell torque profiles (magnetic/dynamic/total) without plotting wrappers.
"""

# TODO(debt): This is a quantity-specific shell profile wrapper (`torque_vs_radius`)
# in a deep layer and depends on `analysis.shells`. Keep local torque-density
# formulas in `physics` and push orchestration upward.

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    integrate_shell_scalar,
    sample_spherical_shells_by_strategy,
    shell_profile_radius_height,
)
from starwinds_analysis.physics.torque import MU0, spherical_wind_torque_density_terms
from starwinds_analysis.recipes.spherical import spherical_vector_components


def _ensure_batsrus_si_fields(smart_ds, *, body_radius_m: float) -> None:
    needed = (
        "Rho [kg/m^3]",
        "U_x [m/s]",
        "U_y [m/s]",
        "U_z [m/s]",
        "B_x [T]",
        "B_y [T]",
        "B_z [T]",
    )
    if all(smart_ds.has_field(name) for name in needed):
        return
    smart_ds.add_batsrus_graph(body_radius_m=float(body_radius_m))


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
    _ensure_batsrus_si_fields(smart_ds, body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"
    bx_name, by_name, bz_name = "B_x [T]", "B_y [T]", "B_z [T]"

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

    rho = shells.fields[rho_name]
    ux = shells.fields[ux_name]
    uy = shells.fields[uy_name]
    uz = shells.fields[uz_name]
    bx = shells.fields[bx_name]
    by = shells.fields[by_name]
    bz = shells.fields[bz_name]

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
        "magnetic_torque [Nm]": np.array(magnetic, dtype=float),
        "dynamic_torque [Nm]": np.array(dynamic, dtype=float),
        "total_torque [Nm]": np.array(total, dtype=float),
        "coverage [none]": np.array(coverage, dtype=float),
        "shell_samples": shells,
    }
__all__ = ["MU0", "torque_vs_radius"]
