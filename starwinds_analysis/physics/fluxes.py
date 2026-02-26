"""THIS FILE contains shell-integrated flux diagnostics.

It computes quantity-specific flux profiles from shell samples.
Flux plotting helpers are implemented in `starwinds_analysis.visualisation.profile_plots`.
"""

# TODO(debt): This file is quantity-specific (`fluxes`) and should eventually be
# expressed as local quantity definitions + generic shell reduction primitives.
# TODO(debt): This module now requests SI fields (including spherical components)
# through SmartDs/griblet, but still computes some local diagnostics in code
# (`B·n`, axisymmetric reductions, `E*U_r`).

from __future__ import annotations

import numpy as np

from starwinds_analysis.analysis.shells import (
    infer_body_radius_m,
    integrate_shell_scalar,
    sample_spherical_shells_by_strategy,
    shell_profile_radius_height,
)
from starwinds_analysis.physics.flux_density import radial_advective_flux_density


def _ensure_batsrus_si_fields(smart_ds, *, body_radius_m: float, include_energy: bool = False) -> None:
    needed = {"B_x [T]", "B_y [T]", "B_z [T]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"}
    if include_energy:
        needed.add("E [J/m^3]")
    if all(smart_ds.has_field(name) for name in needed):
        return
    smart_ds.add_batsrus_graph(body_radius_m=float(body_radius_m))


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
    _ensure_batsrus_si_fields(smart_ds, body_radius_m=body_radius_m)
    bx_name, by_name, bz_name = "B_x [T]", "B_y [T]", "B_z [T]"
    x_name, y_name, z_name = coordinate_fields
    area_name = "dA [m^2]"

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

    bx = np.array(shells(bx_name), dtype=float)
    by = np.array(shells(by_name), dtype=float)
    bz = np.array(shells(bz_name), dtype=float)
    x = np.array(shells(x_name), dtype=float)
    y = np.array(shells(y_name), dtype=float)
    z = np.array(shells(z_name), dtype=float)
    area = np.array(shells(area_name), dtype=float)
    b_r = np.array(shells("B_r [T]"), dtype=float)

    signed_flux, cov_signed = integrate_shell_scalar(b_r, area)
    open_flux, cov_open = integrate_shell_scalar(np.abs(b_r), area)

    r_norm = np.sqrt(x * x + y * y + z * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx = x / r_norm
        ny = y / r_norm
        nz = z / r_norm
    bdotn = bx * nx + by * ny + bz * nz
    signed_flux_from_vector, cov_vec = integrate_shell_scalar(bdotn, area)

    coverage = np.minimum(np.minimum(cov_signed, cov_open), cov_vec)
    return {
        **shell_profile_radius_height(shells),
        "signed_flux [Wb]": np.array(signed_flux, dtype=float),
        "signed_flux_from_vector [Wb]": np.array(signed_flux_from_vector, dtype=float),
        "open_flux [Wb]": np.array(open_flux, dtype=float),
        "coverage [none]": np.array(coverage, dtype=float),
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
    area = np.array(shells("dA [m^2]"), dtype=float)
    b_r = np.array(shells("B_r [T]"), dtype=float)

    with np.errstate(invalid="ignore"):
        b_r_axi_theta = np.nanmean(b_r, axis=-1, keepdims=True)
    b_r_axi = np.broadcast_to(b_r_axi_theta, b_r.shape)
    axi_open_flux, cov_axi = integrate_shell_scalar(np.abs(b_r_axi), area)

    open_flux = np.array(prof["open_flux [Wb]"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.divide(
            axi_open_flux,
            open_flux,
            out=np.full_like(axi_open_flux, np.nan, dtype=float),
            where=open_flux != 0,
        )

    prof = dict(prof)
    prof["axisymmetric_open_flux [Wb]"] = np.array(axi_open_flux, dtype=float)
    prof["axisymmetric_open_flux_fraction [none]"] = np.array(fraction, dtype=float)
    prof["coverage [none]"] = np.minimum(np.array(prof["coverage [none]"]), cov_axi)
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
    _ensure_batsrus_si_fields(smart_ds, body_radius_m=body_radius_m, include_energy=False)
    if smart_ds.has_field("E [J/m^3]"):
        e_name, e_scale = "E [J/m^3]", 1.0
    elif smart_ds.has_field("E [erg/cm^3]"):
        e_name, e_scale = "E [erg/cm^3]", 1e-1
    else:
        # Fall back to the historical candidate list without using `resolve_*`.
        e_name = e_scale = None
        for cand_name, cand_scale in energy_field_candidates:
            if smart_ds.has_field(cand_name):
                e_name, e_scale = cand_name, float(cand_scale)
                break
        if e_name is None:
            names = ", ".join(name for name, _ in energy_field_candidates)
            raise KeyError(f"Could not find any energy field candidate: {names}")
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"
    area_name = "dA [m^2]"

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

    e = e_scale * np.array(shells(e_name), dtype=float)
    area = np.array(shells(area_name), dtype=float)
    u_r = np.array(shells("U_r [m/s]"), dtype=float)

    # TODO(griblet): Request energy-flux density directly from SmartDs/griblet in SI
    # (e.g. `energy_flux [W/m^2]`) instead of recomputing `E * U_r` here.
    energy_flux_density = radial_advective_flux_density(e, u_r)  # W / m^2
    energy_flux, coverage = integrate_shell_scalar(energy_flux_density, area)
    return {
        **shell_profile_radius_height(shells),
        "energy_flux [W]": np.array(energy_flux, dtype=float),
        "coverage [none]": np.array(coverage, dtype=float),
        "shell_samples": shells,
    }

__all__ = [
    "axisymmetric_open_flux_vs_radius",
    "energy_flux_vs_radius",
    "open_magnetic_flux_vs_radius",
]
