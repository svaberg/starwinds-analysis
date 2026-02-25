from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.analysis._profile_plotting import plot_shell_height_series
from starwinds_analysis.analysis.shell_magnetic import plot_shell_scalar_lonlat
from starwinds_analysis.analysis.shells import (
    SphericalShellSamples,
    infer_body_radius_m,
    integrate_shell_scalar,
    resolve_batsrus_density_si,
    resolve_batsrus_vector_xyz_si,
    sample_spherical_shells_by_strategy,
    shell_profile_radius_height,
)
from starwinds_analysis.recipes.spherical import spherical_vector_components


@dataclass
class ShellMassFluxMap:
    radius: float
    shell_samples: SphericalShellSamples
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    mass_flux_kg_m2_s: np.ndarray

    def integrate(self):
        integral, coverage = integrate_shell_scalar(
            self.mass_flux_kg_m2_s[None, ...],
            self.shell_samples.area[:1],
        )
        return float(integral[0]), float(coverage[0])

    def summary(self):
        arr = np.asarray(self.mass_flux_kg_m2_s, dtype=float)
        finite = arr[np.isfinite(arr)]
        out = {
            "finite_cells": int(finite.size),
            "total_cells": int(arr.size),
            "nonpositive_cells": int(np.count_nonzero(np.isfinite(arr) & (arr <= 0.0))),
            "min [kg/m^2/s]": np.nan,
            "max [kg/m^2/s]": np.nan,
        }
        if finite.size:
            out["min [kg/m^2/s]"] = float(np.nanmin(finite))
            out["max [kg/m^2/s]"] = float(np.nanmax(finite))
        return out


def sample_shell_mass_flux_map(
    smart_ds,
    radius: float,
    *,
    body_radius_m: float | None = None,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    n_polar: int = 48,
    n_azimuth: int = 96,
    method: str = "nearest",
    fill_value: float = np.nan,
):
    """
    Sample shell mass flux (`rho * U_r`) on a single spherical shell in SI units.

    Uses grid sampling so the result is directly plottable on a lon/lat mesh.
    """
    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    rho_name, rho_scale = resolve_batsrus_density_si(smart_ds)
    (ux_name, uy_name, uz_name), u_scale = resolve_batsrus_vector_xyz_si(smart_ds, "U")

    shells = sample_spherical_shells_by_strategy(
        smart_ds,
        [float(radius)],
        fields=(rho_name, ux_name, uy_name, uz_name),
        coordinate_fields=coordinate_fields,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        sampling="grid",
        method=method,
        fill_value=fill_value,
        length_unit_to_m=body_radius_m,
    )

    rho = rho_scale * np.asarray(shells.fields[rho_name], dtype=float)
    ux = u_scale * np.asarray(shells.fields[ux_name], dtype=float)
    uy = u_scale * np.asarray(shells.fields[uy_name], dtype=float)
    uz = u_scale * np.asarray(shells.fields[uz_name], dtype=float)
    u_r, _u_theta, _u_phi = spherical_vector_components(ux, uy, uz, shells.x, shells.y, shells.z)
    mass_flux = np.asarray(rho * u_r, dtype=float)

    return ShellMassFluxMap(
        radius=float(radius),
        shell_samples=shells,
        lon_deg=np.degrees(np.asarray(shells.phi, dtype=float)),
        lat_deg=90.0 - np.degrees(np.asarray(shells.theta, dtype=float)),
        mass_flux_kg_m2_s=np.asarray(mass_flux[0], dtype=float),
    )


def plot_shell_mass_flux_lonlat(
    shell_map: ShellMassFluxMap,
    *,
    scale: str = "log",
    figsize=(9, 4.5),
    cmap: str = "viridis",
):
    """
    Convenience lon/lat plot for a shell mass-flux map.

    `scale` may be `"log"` (positive log, under-color highlights inflow) or `"linear"`.
    """
    mode = {"log": "positive_log", "linear": "linear"}.get(str(scale).lower())
    if mode is None:
        raise ValueError("scale must be 'log' or 'linear'")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _img, _cbar, extra = plot_shell_scalar_lonlat(
        ax,
        shell_map.lon_deg,
        shell_map.lat_deg,
        shell_map.mass_flux_kg_m2_s,
        title=f"Mass flux on shell at R={shell_map.radius:g}",
        cbar_label="Mass flux [kg/m^2/s]",
        cmap=cmap,
        scale=mode,
        under_color=("magenta" if mode == "positive_log" else None),
    )
    return fig, ax, extra


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
    return plot_shell_height_series(
        ax,
        profile,
        "mass_loss [kg/s]",
        label="mass loss",
        ylabel="Mass flux [kg/s]",
        color="C0",
        show_negative=show_negative,
    )

__all__ = [
    "ShellMassFluxMap",
    "mass_loss_vs_radius",
    "plot_mass_loss_profile",
    "plot_shell_mass_flux_lonlat",
    "sample_shell_mass_flux_map",
]
