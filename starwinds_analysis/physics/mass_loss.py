"""THIS FILE contains mass-loss shell diagnostics and shell mass-flux products.

It defines reusable mass-loss computations (sampling + shell integration), without
plotting wrappers.
"""

# TODO(debt): This module mixes local quantity logic with shell sampling/integration
# orchestration and depends on `analysis.shells` (reversed layer direction).
# TODO(debt): `ShellMassFluxMap` is a workflow-specific custom container; prefer
# shared abstractions (structured SmartDs + explicit metadata fields).
# TODO(debt): `sample_shell_mass_flux_map` / `mass_loss_vs_radius` are quantity-
# specific pipeline wrappers; keep only generic primitives at deep layers.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from starwinds_analysis.physics.flux_density import radial_advective_flux_density
from starwinds_analysis.recipes.spherical import spherical_vector_components

if TYPE_CHECKING:
    from starwinds_analysis.analysis.shells import SphericalShellSamples


def _ensure_batsrus_si_fields(smart_ds, *, body_radius_m: float) -> None:
    """
    Ensure common BATSRUS SI fields are requestable from `SmartDs`.
    """
    needed = ("Rho [kg/m^3]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]")
    if all(smart_ds.has_field(name) for name in needed):
        return
    smart_ds.add_batsrus_graph(body_radius_m=float(body_radius_m))


@dataclass
class ShellMassFluxMap:
    radius: float
    shell_samples: SphericalShellSamples
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    mass_flux_kg_m2_s: np.ndarray

    def integrate(self):
        from starwinds_analysis.analysis.shells import integrate_shell_scalar

        integral, coverage = integrate_shell_scalar(
            self.mass_flux_kg_m2_s[None, ...],
            self.shell_samples.area[:1],
        )
        return float(integral[0]), float(coverage[0])

    def summary(self):
        arr = np.array(self.mass_flux_kg_m2_s, dtype=float)
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
    from starwinds_analysis.analysis.shells import (
        infer_body_radius_m,
        sample_spherical_shells_by_strategy,
    )

    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    _ensure_batsrus_si_fields(smart_ds, body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"

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

    rho = np.array(shells.fields[rho_name], dtype=float)
    ux = np.array(shells.fields[ux_name], dtype=float)
    uy = np.array(shells.fields[uy_name], dtype=float)
    uz = np.array(shells.fields[uz_name], dtype=float)
    # TODO(griblet): Request `U_r [m/s]` from SmartDs/griblet on the shell sample
    # instead of recomputing spherical components here.
    u_r, _u_theta, _u_phi = spherical_vector_components(ux, uy, uz, shells.x, shells.y, shells.z)
    # TODO(griblet): Request mass-flux density directly from SmartDs/griblet in SI
    # (e.g. `mass_flux [kg/m^2/s]`) instead of recomputing `rho * U_r` here.
    mass_flux = np.array(radial_advective_flux_density(rho, u_r), dtype=float)

    return ShellMassFluxMap(
        radius=float(radius),
        shell_samples=shells,
        lon_deg=np.degrees(np.array(shells.phi, dtype=float)),
        lat_deg=90.0 - np.degrees(np.array(shells.theta, dtype=float)),
        mass_flux_kg_m2_s=np.array(mass_flux[0], dtype=float),
    )
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
    from starwinds_analysis.analysis.shells import (
        infer_body_radius_m,
        integrate_shell_scalar,
        sample_spherical_shells_by_strategy,
        shell_profile_radius_height,
    )

    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    _ensure_batsrus_si_fields(smart_ds, body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"

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

    rho = shells.fields[rho_name]
    ux = shells.fields[ux_name]
    uy = shells.fields[uy_name]
    uz = shells.fields[uz_name]

    # TODO(griblet): Request `U_r [m/s]` from SmartDs/griblet on the shell sample
    # instead of recomputing spherical components here.
    u_r, _u_theta, _u_phi = spherical_vector_components(ux, uy, uz, shells.x, shells.y, shells.z)
    # TODO(griblet): Request mass-flux density directly from SmartDs/griblet in SI
    # (e.g. `mass_flux [kg/m^2/s]`) instead of recomputing `rho * U_r` here.
    mass_flux = radial_advective_flux_density(rho, u_r)  # kg / m^2 / s

    mass_loss, coverage = integrate_shell_scalar(mass_flux, shells.area)
    return {
        **shell_profile_radius_height(shells),
        "mass_loss [kg/s]": np.array(mass_loss, dtype=float),
        "coverage [none]": np.array(coverage, dtype=float),
        "shell_samples": shells,
    }
__all__ = [
    "ShellMassFluxMap",
    "mass_loss_vs_radius",
    "sample_shell_mass_flux_map",
]
