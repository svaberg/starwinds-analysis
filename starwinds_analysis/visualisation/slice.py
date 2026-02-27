"""THIS FILE contains reusable 2D slice plotting helpers.

Pipelines and examples should call these helpers directly instead of carrying
plot implementation details in pipeline modules.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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


def _make_slice_tripcolor_figure(
    ds,
    *,
    var: str | None = None,
    figsize=(8, 7),
    cmap: str = "viridis",
    norm=None,
    shading: str = "flat",
):
    """
    Create a 2D tripcolor figure with simple marginal companion axes.
    Used by: `starwinds_analysis/visualisation/slice.py`
    """
    field = _resolve_field(ds, var)
    x_name, y_name = auto_coords(ds)
    tri = triangles(ds, x_name, y_name)
    c = ds.variable(field)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 4.0),
        height_ratios=(4.0, 1.0),
        wspace=0.05,
        hspace=0.05,
    )
    ax_main = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_corner = fig.add_subplot(gs[1, 0])
    ax_corner.axis("off")

    image = ax_main.tripcolor(tri, c, shading=shading, cmap=cmap, norm=norm)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel(x_name)
    ax_main.set_ylabel(y_name)
    ax_main.set_title(field)

    x = ds.variable(x_name)
    y = ds.variable(y_name)
    ax_left.plot(c, y, ",", alpha=0.4)
    ax_left.set_xlabel(field)
    ax_left.tick_params(labelleft=False)
    ax_bottom.plot(x, c, ",", alpha=0.4)
    ax_bottom.set_xlabel(x_name)
    ax_bottom.set_ylabel(field)

    cbar = fig.colorbar(image, ax=ax_main, label=field)
    return fig, (ax_main, ax_left, ax_bottom), cbar


def plot_xz_slice_tripcolor_with_marginals(ds, *, var: str | None = None, **kwargs):
    """
    Tripcolor slice plot with compact marginal companion axes.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    return _make_slice_tripcolor_figure(ds, var=var, **kwargs)


def plot_xz_slice_tripcolor_with_cross_quantiles(ds, *, var: str | None = None, **kwargs):
    """
    Tripcolor slice plot in the cross-quantile style.
    Used by: `starwinds_analysis/pipelines/slice.py`, `starwinds_analysis/pipelines/volume.py`
    """
    return _make_slice_tripcolor_figure(ds, var=var, **kwargs)


def plot_xz_slice_with_marginal_points(ds, *, var: str | None = None, **kwargs):
    """
    Tripcolor slice plot with point-style marginal companions.
    Used by: `examples/planet.py`, `examples/earth-xuv-neutrals/earth-xuv-neutrals.py`
    """
    return _make_slice_tripcolor_figure(ds, var=var, **kwargs)


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
    return _make_slice_tripcolor_figure(ds, var=var, **kwargs)
