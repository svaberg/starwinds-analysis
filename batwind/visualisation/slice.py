"""Reusable 2D slice plotting helpers.
"""

# Pipelines and examples should call these helpers directly instead of carrying
# plot implementation details in pipeline modules.


from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

log = logging.getLogger(__name__)


def auto_coords(ds, names=None):
    """
    Detect the two varying coordinates in a nominal 2D slice dataset.
    Used by: `/Users/dagfev/Documents/starwinds/batwind/examples/smartds_2d_xy_points.ipynb`,
      `/Users/dagfev/Documents/starwinds/batwind/examples/planet.py`,
      `/Users/dagfev/Documents/starwinds/batwind/examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """
    if names is None:
        names = ("X [R]", "Y [R]", "Z [R]")

    if np.allclose(ds.variable("X [R]"), 0):
        return "Y [R]", "Z [R]"
    if np.allclose(ds.variable("Y [R]"), 0):
        return "X [R]", "Z [R]"
    if np.allclose(ds.variable("Z [R]"), 0):
        return "X [R]", "Y [R]"

    spread = [np.nanmax(np.abs(np.array(ds.variable(name)))) for name in names]
    i, j = np.argsort(spread)[-2:]
    log.debug("auto_coords fallback selected %s and %s", names[i], names[j])
    return names[i], names[j]


def triangles(ds, uname=None, vname=None):
    """
    Build a Matplotlib triangulation from 2D quad-cell connectivity.
    Used by: `/Users/dagfev/Documents/starwinds/batwind/examples/planet.py`,
      `/Users/dagfev/Documents/starwinds/batwind/examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """
    if uname is None and vname is None:
        uname, vname = auto_coords(ds)

    pu = ds.variable(uname)
    pv = ds.variable(vname)

    if ds.corners.shape[1] != 4:
        log.error("triangles failed: expected 4 corners per element, got %d", ds.corners.shape[1])
        raise ValueError("Can only triangulate a 2D dataset with 4 corners per element")

    faces = np.vstack((ds.corners[:, [0, 1, 2]], ds.corners[:, [2, 3, 0]]))
    return tri.Triangulation(pu, pv, faces)


def default_slice_field(ds, var: str | None) -> str:
    """
    Resolve a default display field for slice plots when `var` is not provided.
    Used by: `batwind/visualisation/slice.py`
    """
    if var is not None:
        return str(var)
    for candidate in ("Rho [kg/m^3]", "Rho [g/cm^3]", "Rho [amu/cm^3]"):
        if ds.has_field(candidate):
            return candidate
    log.warning("default_slice_field falling back to first variable")
    return str(ds.variables[0])


def _apply_field_scale(ax_left, ax_bottom, norm) -> None:
    """
    Make the marginal field axes follow the main color normalization.
    Used by: `batwind/pipelines/slice.py`, `batwind/pipelines/volume.py`
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
    Used by: `batwind/visualisation/slice.py`
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
    Used by: `batwind/pipelines/slice.py`, `batwind/pipelines/volume.py`
    """
    field = default_slice_field(ds, var)
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
    log.info("plot_xz_slice_tripcolor_with_marginals done field=%s", field)
    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_tripcolor_with_cross_quantiles(ds, *, var: str | None = None, **kwargs):
    """
    Tripcolor slice plot in the cross-quantile style.
    Used by: `batwind/pipelines/slice.py`, `batwind/pipelines/volume.py`
    """
    field = default_slice_field(ds, var)
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
    log.info("plot_xz_slice_tripcolor_with_cross_quantiles done field=%s", field)
    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_with_marginal_points(ds, *, var: str | None = None, **kwargs):
    """
    Tripcolor slice plot with point-style marginal companions.
    Used by: `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """
    field = default_slice_field(ds, var)
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
    log.info("plot_xz_slice_with_marginal_points done field=%s", field)
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
    field = default_slice_field(ds, var)
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
    log.info("plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords done field=%s", field)
    return fig, (ax_main, ax_left, ax_bottom), cbar
