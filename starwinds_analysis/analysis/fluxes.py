from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis._profile_plotting import plot_shell_height_series
from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    integrate_shell_scalar,
    resolve_batsrus_vector_xyz_si,
    resolve_field_with_scale,
    sample_spherical_shells_by_strategy,
    shell_profile_radius_height,
)
from starwinds_analysis.recipes.spherical import spherical_vector_components


def open_magnetic_flux_vs_radius(
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
    Signed/unsigned magnetic flux on spherical shells.
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    (bx_name, by_name, bz_name), b_scale = resolve_batsrus_vector_xyz_si(smart_ds, "B")

    shells = sample_spherical_shells_by_strategy(
        smart_ds,
        radii,
        fields=(bx_name, by_name, bz_name),
        coordinate_fields=coordinate_fields,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        sampling=sampling,
        fibonacci_randomize=fibonacci_randomize,
        method=method,
        fill_value=fill_value,
        length_unit_to_m=body_radius_m,
    )

    bx = b_scale * shells.fields[bx_name]
    by = b_scale * shells.fields[by_name]
    bz = b_scale * shells.fields[bz_name]
    b_r, _b_theta, _b_phi = spherical_vector_components(bx, by, bz, shells.x, shells.y, shells.z)

    signed_flux, cov_signed = integrate_shell_scalar(b_r, shells.area)
    open_flux, cov_open = integrate_shell_scalar(np.abs(b_r), shells.area)

    r_norm = np.sqrt(shells.x * shells.x + shells.y * shells.y + shells.z * shells.z)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx = shells.x / r_norm
        ny = shells.y / r_norm
        nz = shells.z / r_norm
    bdotn = bx * nx + by * ny + bz * nz
    signed_flux_from_vector, cov_vec = integrate_shell_scalar(bdotn, shells.area)

    coverage = np.minimum(np.minimum(cov_signed, cov_open), cov_vec)
    return {
        **shell_profile_radius_height(shells),
        "signed_flux [Wb]": np.asarray(signed_flux, dtype=float),
        "signed_flux_from_vector [Wb]": np.asarray(signed_flux_from_vector, dtype=float),
        "open_flux [Wb]": np.asarray(open_flux, dtype=float),
        "coverage [none]": np.asarray(coverage, dtype=float),
        "shell_samples": shells,
    }


def axisymmetric_open_flux_vs_radius(
    smart_ds,
    radii,
    *,
    body_radius_m: float | None = None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_polar: int = 24,
    n_azimuth: int = 48,
    sampling: str = "grid",
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Axisymmetric open magnetic flux and fraction using shell-sampled B_r.

    Axisymmetry is defined here as the azimuthal mean of `B_r` at each `(r, theta)`.
    """
    if sampling != "grid":
        raise ValueError("axisymmetric_open_flux_vs_radius currently requires sampling='grid'")
    prof = open_magnetic_flux_vs_radius(
        smart_ds,
        radii,
        body_radius_m=body_radius_m,
        coordinate_fields=coordinate_fields,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        sampling="grid",
        method=method,
        fill_value=fill_value,
    )
    shells = prof["shell_samples"]
    # Reconstruct B_r from the cached shell samples in SI.
    (bx_raw, by_raw, bz_raw), b_scale = resolve_batsrus_vector_xyz_si(smart_ds, "B")
    bx = b_scale * shells.fields[bx_raw]
    by = b_scale * shells.fields[by_raw]
    bz = b_scale * shells.fields[bz_raw]
    b_r, _b_theta, _b_phi = spherical_vector_components(bx, by, bz, shells.x, shells.y, shells.z)

    with np.errstate(invalid="ignore"):
        b_r_axi_theta = np.nanmean(b_r, axis=-1, keepdims=True)
    b_r_axi = np.broadcast_to(b_r_axi_theta, b_r.shape)
    axi_open_flux, cov_axi = integrate_shell_scalar(np.abs(b_r_axi), shells.area)

    open_flux = np.asarray(prof["open_flux [Wb]"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.divide(
            axi_open_flux,
            open_flux,
            out=np.full_like(axi_open_flux, np.nan, dtype=float),
            where=open_flux != 0,
        )

    prof = dict(prof)
    prof["axisymmetric_open_flux [Wb]"] = np.asarray(axi_open_flux, dtype=float)
    prof["axisymmetric_open_flux_fraction [none]"] = np.asarray(fraction, dtype=float)
    prof["coverage [none]"] = np.minimum(np.asarray(prof["coverage [none]"]), cov_axi)
    return prof


def energy_flux_vs_radius(
    smart_ds,
    radii,
    *,
    body_radius_m: float | None = None,
    energy_field_candidates=(
        ("E [J/m^3]", 1.0),
        ("E [erg/cm^3]", 1e-1),
    ),
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_polar: int = 24,
    n_azimuth: int = 48,
    sampling: str = "fibonacci",
    fibonacci_randomize: bool = False,
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Radial energy flux profile using `E * U_r`.
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    e_name, e_scale = resolve_field_with_scale(smart_ds, energy_field_candidates)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")

    shells = sample_spherical_shells_by_strategy(
        smart_ds,
        radii,
        fields=(e_name, ux_name, uy_name, uz_name),
        coordinate_fields=coordinate_fields,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        sampling=sampling,
        fibonacci_randomize=fibonacci_randomize,
        method=method,
        fill_value=fill_value,
        length_unit_to_m=body_radius_m,
    )

    e = e_scale * shells.fields[e_name]
    ux = u_scale * shells.fields[ux_name]
    uy = u_scale * shells.fields[uy_name]
    uz = u_scale * shells.fields[uz_name]
    u_r, _u_theta, _u_phi = spherical_vector_components(ux, uy, uz, shells.x, shells.y, shells.z)

    energy_flux_density = e * u_r  # W / m^2
    energy_flux, coverage = integrate_shell_scalar(energy_flux_density, shells.area)
    return {
        **shell_profile_radius_height(shells),
        "energy_flux [W]": np.asarray(energy_flux, dtype=float),
        "coverage [none]": np.asarray(coverage, dtype=float),
        "shell_samples": shells,
    }


def plot_open_flux_profile(ax, profile):
    return plot_shell_height_series(
        ax,
        profile,
        "open_flux [Wb]",
        label="open flux",
        ylabel="Open magnetic flux [Wb]",
        color="C0",
        show_negative=False,
    )


def plot_energy_flux_profile(ax, profile, *, show_negative=True):
    return plot_shell_height_series(
        ax,
        profile,
        "energy_flux [W]",
        label="energy flux",
        ylabel="Energy flux [W]",
        color="C0",
        show_negative=show_negative,
    )


__all__ = [
    "axisymmetric_open_flux_vs_radius",
    "energy_flux_vs_radius",
    "open_magnetic_flux_vs_radius",
    "plot_energy_flux_profile",
    "plot_open_flux_profile",
]
