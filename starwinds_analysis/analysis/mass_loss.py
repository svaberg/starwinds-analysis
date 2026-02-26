"""THIS FILE contains mass-loss plotting helpers.

It plots mass-loss diagnostics and shell mass-flux maps computed elsewhere.
Mass-loss computations live in `starwinds_analysis.physics.mass_loss`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from starwinds_analysis.analysis._profile_plotting import plot_shell_height_series
from starwinds_analysis.analysis.shell_magnetic import plot_shell_scalar_lonlat
from starwinds_analysis.physics.mass_loss import ShellMassFluxMap


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


__all__ = ["plot_mass_loss_profile", "plot_shell_mass_flux_lonlat"]
