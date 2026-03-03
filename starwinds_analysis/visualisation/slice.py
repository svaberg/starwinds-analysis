"""THIS FILE contains reusable 2D slice plotting helpers.

Pipelines and examples should call these helpers directly instead of carrying
plot implementation details in pipeline modules.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from starwinds_analysis.utils import auto_coords, triangles


def _resolve_field(ds, var: str | None) -> str:
    """
    Resolve a default display field for slice plots when `var` is not provided.
    Used by: `starwinds_analysis/visualisation/slice.py`
    """
    if var is not None:
        return str(var)
    for candidate in ("Rho [kg/m^3]", "Rho [g/cm^3]", "Rho [amu/cm^3]"):
        try:
            ds.variable(candidate)
            return candidate
        except Exception:
            continue
    return str(ds.variables[0])


def _apply_field_scale(ax_left, ax_bottom, norm) -> None:
    """
    Make the marginal field axes follow the main color normalization.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if norm is None:
        ax_left.xaxis.set_minor_locator(NullLocator())
        ax_bottom.yaxis.set_minor_locator(NullLocator())
        return
    if isinstance(norm, LogNorm):
        ax_left.set_xscale("log")
        ax_bottom.set_yscale("log")
        ax_left.xaxis.set_minor_locator(NullLocator())
        ax_bottom.yaxis.set_minor_locator(NullLocator())
    if isinstance(norm, SymLogNorm):
        base = getattr(getattr(norm, "_scale", None), "base", 10)
        ax_left.set_xscale("symlog", linthresh=norm.linthresh, base=base)
        ax_bottom.set_yscale("symlog", linthresh=norm.linthresh, base=base)
        ax_left.xaxis.set_minor_locator(NullLocator())
        ax_bottom.yaxis.set_minor_locator(NullLocator())


def _match_field_ticks_to_colorbar(ax_left, ax_bottom, cbar) -> None:
    """
    Reuse the colorbar field ticks on the marginal field axes.
    Used by: `starwinds_analysis/visualisation/slice.py`
    """
    cbar.update_ticks()
    ticks = cbar.get_ticks()
    ax_left.xaxis.set_major_locator(FixedLocator(ticks))
    ax_bottom.yaxis.set_major_locator(FixedLocator(ticks))
    formatter = cbar.ax.yaxis.get_major_formatter()
    ax_left.xaxis.set_major_formatter(formatter)
    ax_bottom.yaxis.set_major_formatter(formatter)
def plot_xz_slice_tripcolor_with_marginals(ds, *, var: str | None = None, **kwargs):
    """
    Tripcolor slice plot with compact marginal companion axes.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    field = _resolve_field(ds, var)
    x_name, y_name = auto_coords(ds)
    tri = triangles(ds, x_name, y_name)
    c = ds.variable(field)
    figsize = kwargs.pop("figsize", (8, 7))
    cmap = kwargs.pop("cmap", "viridis")
    norm = kwargs.pop("norm", None)
    shading = kwargs.pop("shading", "flat")

    fig, ax_main = plt.subplots(figsize=figsize)
    divider = make_axes_locatable(ax_main)
    ax_left = divider.append_axes("left", size="25%", pad=0.03, sharey=ax_main)
    ax_bottom = divider.append_axes("bottom", size="25%", pad=0.03, sharex=ax_main)
    ax_cbar = divider.append_axes("right", size="4%", pad=0.03)

    image = ax_main.tripcolor(tri, c, shading=shading, cmap=cmap, norm=norm)
    ax_main.set_aspect("equal")
    ax_main.set_title(field)

    x = ds.variable(x_name)
    y = ds.variable(y_name)
    ax_left.plot(c, y, ",", alpha=0.4)
    ax_bottom.plot(x, c, ",", alpha=0.4)
    ax_bottom.tick_params(axis="y", labelright=True, right=True, labelleft=False, left=False)
    ax_left.set_ylim(ax_main.get_ylim())
    ax_bottom.set_xlim(ax_main.get_xlim())
    _apply_field_scale(ax_left, ax_bottom, norm)

    cbar = fig.colorbar(image, cax=ax_cbar, label=field)
    _match_field_ticks_to_colorbar(ax_left, ax_bottom, cbar)
    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_tripcolor_with_cross_quantiles(ds, *, var: str | None = None, **kwargs):
    """
    Tripcolor slice plot in the cross-quantile style.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    field = _resolve_field(ds, var)
    x_name, y_name = auto_coords(ds)
    tri = triangles(ds, x_name, y_name)
    c = ds.variable(field)
    figsize = kwargs.pop("figsize", (8, 7))
    cmap = kwargs.pop("cmap", "viridis")
    norm = kwargs.pop("norm", None)
    shading = kwargs.pop("shading", "flat")

    fig, ax_main = plt.subplots(figsize=figsize)
    divider = make_axes_locatable(ax_main)
    ax_left = divider.append_axes("left", size="25%", pad=0.03, sharey=ax_main)
    ax_bottom = divider.append_axes("bottom", size="25%", pad=0.03, sharex=ax_main)
    ax_cbar = divider.append_axes("right", size="4%", pad=0.03)

    image = ax_main.tripcolor(tri, c, shading=shading, cmap=cmap, norm=norm)
    ax_main.set_aspect("equal")
    ax_main.set_title(field)

    x = ds.variable(x_name)
    y = ds.variable(y_name)
    ax_left.plot(c, y, ",", alpha=0.4)
    ax_bottom.plot(x, c, ",", alpha=0.4)
    ax_bottom.tick_params(axis="y", labelright=True, right=True, labelleft=False, left=False)
    ax_left.set_ylim(ax_main.get_ylim())
    ax_bottom.set_xlim(ax_main.get_xlim())
    _apply_field_scale(ax_left, ax_bottom, norm)

    cbar = fig.colorbar(image, cax=ax_cbar, label=field)
    _match_field_ticks_to_colorbar(ax_left, ax_bottom, cbar)
    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_with_marginal_points(ds, *, var: str | None = None, **kwargs):
    """
    Tripcolor slice plot with point-style marginal companions.
    Used by: `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """
    field = _resolve_field(ds, var)
    x_name, y_name = auto_coords(ds)
    tri = triangles(ds, x_name, y_name)
    c = ds.variable(field)
    figsize = kwargs.pop("figsize", (8, 7))
    cmap = kwargs.pop("cmap", "viridis")
    norm = kwargs.pop("norm", None)
    shading = kwargs.pop("shading", "flat")

    fig, ax_main = plt.subplots(figsize=figsize)
    divider = make_axes_locatable(ax_main)
    ax_left = divider.append_axes("left", size="25%", pad=0.03, sharey=ax_main)
    ax_bottom = divider.append_axes("bottom", size="25%", pad=0.03, sharex=ax_main)
    ax_cbar = divider.append_axes("right", size="4%", pad=0.03)

    image = ax_main.tripcolor(tri, c, shading=shading, cmap=cmap, norm=norm)
    ax_main.set_aspect("equal")
    ax_main.set_title(field)

    x = ds.variable(x_name)
    y = ds.variable(y_name)
    ax_left.plot(c, y, ",", alpha=0.4)
    ax_bottom.plot(x, c, ",", alpha=0.4)
    ax_bottom.tick_params(axis="y", labelright=True, right=True, labelleft=False, left=False)
    ax_left.set_ylim(ax_main.get_ylim())
    ax_bottom.set_xlim(ax_main.get_xlim())
    _apply_field_scale(ax_left, ax_bottom, norm)

    cbar = fig.colorbar(image, cax=ax_cbar, label=field)
    _match_field_ticks_to_colorbar(ax_left, ax_bottom, cbar)
    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords(
    ds,
    *,
    var: str | None = None,
    **kwargs,
):
    """
    Tripcolor slice plot with unique-coordinate marginal quantile style.
    Used by: `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """
    field = _resolve_field(ds, var)
    x_name, y_name = auto_coords(ds)
    tri = triangles(ds, x_name, y_name)
    c = ds.variable(field)
    figsize = kwargs.pop("figsize", (8, 7))
    cmap = kwargs.pop("cmap", "viridis")
    norm = kwargs.pop("norm", None)
    shading = kwargs.pop("shading", "flat")

    fig, ax_main = plt.subplots(figsize=figsize)
    divider = make_axes_locatable(ax_main)
    ax_left = divider.append_axes("left", size="25%", pad=0.03, sharey=ax_main)
    ax_bottom = divider.append_axes("bottom", size="25%", pad=0.03, sharex=ax_main)
    ax_cbar = divider.append_axes("right", size="4%", pad=0.03)

    image = ax_main.tripcolor(tri, c, shading=shading, cmap=cmap, norm=norm)
    ax_main.set_aspect("equal")
    ax_main.set_title(field)

    x = ds.variable(x_name)
    y = ds.variable(y_name)
    ax_left.plot(c, y, ",", alpha=0.4)
    ax_bottom.plot(x, c, ",", alpha=0.4)
    ax_bottom.tick_params(axis="y", labelright=True, right=True, labelleft=False, left=False)
    ax_left.set_ylim(ax_main.get_ylim())
    ax_bottom.set_xlim(ax_main.get_xlim())
    _apply_field_scale(ax_left, ax_bottom, norm)

    cbar = fig.colorbar(image, cax=ax_cbar, label=field)
    _match_field_ticks_to_colorbar(ax_left, ax_bottom, cbar)
    return fig, (ax_main, ax_left, ax_bottom), cbar
