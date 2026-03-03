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

import logging

import numpy as np

from starwinds_analysis.analysis.shells import infer_body_radius_m
from starwinds_analysis.analysis.shells import integrate_shell_scalar
from starwinds_analysis.analysis.shells import sample_spherical_shells_by_strategy

log = logging.getLogger(__name__)

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
    log_result: bool = True,
):
    """
    Signed/unsigned magnetic flux on spherical shells.
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`,
      `starwinds_analysis/physics/fluxes.py`
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
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
    shells.add_batsrus_graph(body_radius_m=body_radius_m, merge=False)

    bx = np.array(shells(bx_name))
    by = np.array(shells(by_name))
    bz = np.array(shells(bz_name))
    x = np.array(shells(x_name))
    y = np.array(shells(y_name))
    z = np.array(shells(z_name))
    area = np.array(shells(area_name))
    b_r = np.array(shells("B_r [T]"))

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
    r_field = np.array(shells("R [R]"))
    radii_profile = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
    if log_result:
        finite = np.isfinite(radii_profile) & np.isfinite(open_flux)
        if np.any(finite):
            idx = np.where(finite)[0][-1]
            log.info(
                "open_flux_wb radius=%g value=%g",
                float(radii_profile[idx]),
                float(open_flux[idx]),
            )
    return {
        "radius [R]": radii_profile,
        "height [R]": radii_profile - 1.0,
        "signed_flux [Wb]": np.array(signed_flux),
        "signed_flux_from_vector [Wb]": np.array(signed_flux_from_vector),
        "open_flux [Wb]": np.array(open_flux),
        "coverage [none]": np.array(coverage),
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
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
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
        log_result=False,
    )
    shells = prof["shell_samples"]
    # Reconstruct B_r from the cached shell samples in SI.
    area = np.array(shells("dA [m^2]"))
    b_r = np.array(shells("B_r [T]"))

    with np.errstate(invalid="ignore"):
        b_r_axi_theta = np.nanmean(b_r, axis=-1, keepdims=True)
    b_r_axi = np.broadcast_to(b_r_axi_theta, b_r.shape)
    axi_open_flux, cov_axi = integrate_shell_scalar(np.abs(b_r_axi), area)

    open_flux = np.array(prof["open_flux [Wb]"])
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.divide(
            axi_open_flux,
            open_flux,
            out=np.full_like(axi_open_flux, np.nan, dtype=float),
            where=open_flux != 0,
        )

    prof = dict(prof)
    prof["axisymmetric_open_flux [Wb]"] = np.array(axi_open_flux)
    prof["axisymmetric_open_flux_fraction [none]"] = np.array(fraction)
    prof["coverage [none]"] = np.minimum(np.array(prof["coverage [none]"]), cov_axi)
    radii_profile = np.array(prof["radius [R]"])
    frac_profile = np.array(prof["axisymmetric_open_flux_fraction [none]"])
    finite = np.isfinite(radii_profile) & np.isfinite(frac_profile)
    if np.any(finite):
        idx = np.where(finite)[0][-1]
        log.info(
            "axisymmetric_open_flux_fraction radius=%g value=%g",
            float(radii_profile[idx]),
            float(frac_profile[idx]),
        )
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
    Used by: `test/test_shell_analysis.py`, `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    if smart_ds.has_field("E [J/m^3]"):
        e_name = "E [J/m^3]"
    elif smart_ds.has_field("E [erg/cm^3]"):
        e_name = "E [erg/cm^3]"
    else:
        # Fall back to the historical candidate list without using `resolve_*`.
        e_name = None
        for cand_name, cand_scale in energy_field_candidates:
            if smart_ds.has_field(cand_name):
                e_name = cand_name
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
    shells.add_batsrus_graph(body_radius_m=body_radius_m, merge=False)

    area = np.array(shells(area_name))

    energy_flux_density = np.array(shells("energy_flux [W/m^2]"))
    energy_flux, coverage = integrate_shell_scalar(energy_flux_density, area)
    r_field = np.array(shells("R [R]"))
    radii_profile = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
    finite = np.isfinite(radii_profile) & np.isfinite(energy_flux)
    if np.any(finite):
        idx = np.where(finite)[0][-1]
        log.info(
            "energy_flux_w radius=%g value=%g",
            float(radii_profile[idx]),
            float(energy_flux[idx]),
        )
    return {
        "radius [R]": radii_profile,
        "height [R]": radii_profile - 1.0,
        "energy_flux [W]": np.array(energy_flux),
        "coverage [none]": np.array(coverage),
        "shell_samples": shells,
    }
