"""THIS FILE contains mass-loss plotting helpers.

It plots mass-loss diagnostics and shell mass-flux maps computed elsewhere.
Mass-loss computations live in `starwinds_analysis.physics.mass_loss`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np

from starwinds_analysis.analysis._profile_plotting import plot_shell_height_series
from starwinds_analysis.physics.mass_loss import ShellMassFluxMap


def _style_shell_lonlat_axes(ax, *, title: str | None = None):
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    if title is not None:
        ax.set_title(title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.xaxis.set_minor_locator(MultipleLocator(30))
    ax.yaxis.set_major_locator(MultipleLocator(45))
    ax.yaxis.set_minor_locator(MultipleLocator(15))
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=3)
    ax.grid(which="major", alpha=0.15, linewidth=0.5)
    return ax


def _positive_log_plot_values(values: np.ndarray):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    positive = finite & (arr > 0.0)
    if not np.any(positive):
        return np.ma.masked_invalid(arr), None, int(np.count_nonzero(finite & (arr <= 0.0)))
    norm = LogNorm()
    norm.autoscale_None(arr[positive])
    under_value = max(np.nextafter(float(norm.vmin), 0.0), np.finfo(float).tiny)
    plot_values = np.array(arr, dtype=float, copy=True)
    plot_values[finite & (plot_values <= 0.0)] = under_value
    return plot_values, norm, int(np.count_nonzero(finite & (arr <= 0.0)))


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
    values = np.asarray(shell_map.mass_flux_kg_m2_s, dtype=float)
    cmap_obj = plt.get_cmap(cmap).copy()
    norm = None
    extend = "neither"
    extra = {}
    plot_values = values
    if mode == "positive_log":
        cmap_obj.set_under("magenta")
        plot_values, norm, n_nonpositive = _positive_log_plot_values(values)
        extend = "min"
        extra["n_nonpositive"] = n_nonpositive

    img = ax.pcolormesh(
        shell_map.lon_deg,
        shell_map.lat_deg,
        plot_values,
        shading="nearest",
        cmap=cmap_obj,
        norm=norm,
    )
    _style_shell_lonlat_axes(ax, title=f"Mass flux on shell at R={shell_map.radius:g}")
    cbar = fig.colorbar(img, ax=ax, extend=extend)
    cbar.set_label("Mass flux [kg/m^2/s]")
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
