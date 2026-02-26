"""THIS FILE contains mass-loss shell diagnostics and shell mass-flux products.

It defines reusable mass-loss computations (sampling + shell integration), without
plotting wrappers.
"""

# TODO(debt): This module mixes local quantity logic with shell sampling/integration
# orchestration and depends on `analysis.shells` (reversed layer direction).
# TODO(debt): `mass_loss_vs_radius` is a quantity-specific pipeline wrapper; keep
# only generic shell reduction primitives at deep layers.

from __future__ import annotations

import numpy as np

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
    Used by: `test/test_shell_analysis.py`, `examples/smartds_shell_mass_flux.ipynb`,
      `starwinds_analysis/quicklook2d.py`, `starwinds_analysis/physics/orbit_local.py`
    """
    from starwinds_analysis.analysis.shells import (
        infer_body_radius_m,
        integrate_shell_scalar,
        sample_spherical_shells_by_strategy,
        shell_profile_radius_height,
    )

    body_radius_m = infer_body_radius_m(smart_ds, body_radius_m=body_radius_m)
    smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
    rho_name = "Rho [kg/m^3]"
    ux_name, uy_name, uz_name = "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"
    area_name = "dA [m^2]"

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

    area = np.array(shells(area_name))

    mass_flux = np.array(shells("mass_flux [kg/m^2/s]"))

    mass_loss, coverage = integrate_shell_scalar(mass_flux, area)
    return {
        **shell_profile_radius_height(shells),
        "mass_loss [kg/s]": np.array(mass_loss),
        "coverage [none]": np.array(coverage),
        "shell_samples": shells,
    }
