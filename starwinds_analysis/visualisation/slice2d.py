from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
import numpy as np

from starwinds_analysis.utils import auto_coords, triangles as ds_triangles


@dataclass
class TriangulatedSliceGeometry:
    x_field: str
    y_field: str
    x: np.ndarray
    y: np.ndarray
    triangulation: mtri.Triangulation


def triangulated_slice_geometry(smart_ds, *, x_field: str | None = None, y_field: str | None = None):
    """
    Build a triangulated geometry for an already-2D slice dataset (no resampling).

    Uses library helpers `auto_coords(...)` and `utils.triangles(...)` when available.
    """
    if x_field is None and y_field is None:
        x_field, y_field = auto_coords(smart_ds)
    elif x_field is None or y_field is None:
        raise ValueError("x_field and y_field must both be provided or both omitted")

    x = np.asarray(smart_ds.variable(x_field), dtype=float)
    y = np.asarray(smart_ds.variable(y_field), dtype=float)

    try:
        tris = ds_triangles(smart_ds, x_field, y_field)
    except Exception:
        corners = np.asarray(smart_ds.corners, dtype=int)
        if corners.ndim != 2 or corners.shape[1] not in (3, 4):
            raise
        tri_idx = (
            np.vstack((corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]))
            if corners.shape[1] == 4
            else corners
        )
        tris = mtri.Triangulation(x, y, tri_idx)

    return TriangulatedSliceGeometry(
        x_field=str(x_field),
        y_field=str(y_field),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        triangulation=tris,
    )


def _tripcolor_scale(values, *, scale: str, vmin=None, vmax=None):
    arr = np.asarray(values, dtype=float)
    arr_plot = np.ma.masked_invalid(arr)

    if scale == "auto":
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmin(finite)) > 0.0:
            scale = "positive_log"
        else:
            scale = "linear"

    if scale == "linear":
        return arr, arr_plot, None, "linear"

    if scale != "positive_log":
        raise ValueError("scale must be 'auto', 'linear', or 'positive_log'")

    arr_plot = np.ma.masked_less_equal(arr_plot, 0.0)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if positive.size == 0:
        # Be forgiving in examples: show something instead of erroring.
        return arr, np.ma.masked_invalid(arr), None, "linear"

    if vmin is None:
        vmin = float(np.nanmin(positive))
    if vmax is None:
        vmax = float(np.nanmax(positive))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0.0 or vmax <= vmin:
        return arr, np.ma.masked_invalid(arr), None, "linear"

    return arr, arr_plot, LogNorm(vmin=float(vmin), vmax=float(vmax)), "positive_log"


def plot_slice_tripcolor(
    smart_ds,
    field_name: str,
    *,
    geometry: TriangulatedSliceGeometry | None = None,
    ax=None,
    figsize=(7, 6),
    cmap: str = "viridis",
    scale: str = "auto",
    vmin=None,
    vmax=None,
    outside_colors: bool = False,
    show_mesh: bool = True,
    mesh_color: str = "k",
    mesh_linewidth: float = 0.15,
    mesh_alpha: float = 0.15,
    title: str | None = None,
    cbar_label: str | None = None,
):
    """
    Generic flat-shaded triangulated slice plot on the native 2D mesh.

    This is intentionally parameterized by field/label/scale/cmap instead of
    creating quantity-specific plotting functions.
    """
    geom = triangulated_slice_geometry(smart_ds) if geometry is None else geometry
    raw_values = np.asarray(smart_ds.variable(field_name), dtype=float)
    values, values_plot, norm, scale_used = _tripcolor_scale(raw_values, scale=scale, vmin=vmin, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap).copy()
    if outside_colors and scale_used == "positive_log":
        cmap_obj.set_under(cmap_obj(0.0))
        cmap_obj.set_over(cmap_obj(1.0))

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    img = ax.tripcolor(geom.triangulation, values_plot, shading="flat", cmap=cmap_obj, norm=norm)
    if show_mesh:
        ax.triplot(
            geom.triangulation,
            color=mesh_color,
            linewidth=mesh_linewidth,
            alpha=mesh_alpha,
        )

    ax.set_aspect("equal")
    ax.set_xlabel(geom.x_field)
    ax.set_ylabel(geom.y_field)
    ax.set_title(str(field_name) if title is None else str(title))
    extend = "both" if (outside_colors and scale_used == "positive_log") else "neither"
    cbar = fig.colorbar(img, ax=ax, label=(str(field_name) if cbar_label is None else str(cbar_label)), extend=extend)
    return fig, ax, {
        "image": img,
        "colorbar": cbar,
        "scale": scale_used,
        "geometry": geom,
        "values": values,
    }


def add_slice_contours(
    smart_ds,
    field_name: str,
    *,
    levels,
    geometry: TriangulatedSliceGeometry | None = None,
    ax=None,
    require_all_levels_in_range: bool = False,
    **kwargs,
):
    """
    Add contours for a field on a triangulated native 2D slice.

    Returns `(contour_set, drawn)` where `drawn` is `False` if no requested contour
    level is present in the finite data range.
    """
    if ax is None:
        raise ValueError("ax is required")
    geom = triangulated_slice_geometry(smart_ds) if geometry is None else geometry
    values = np.asarray(smart_ds.variable(field_name), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, False

    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    inside = (levels >= lo) & (levels <= hi)
    if require_all_levels_in_range:
        if not np.all(inside):
            return None, False
    elif not np.any(inside):
        return None, False

    cs = ax.tricontour(geom.triangulation, values, levels=levels, **kwargs)
    return cs, True


__all__ = [
    "TriangulatedSliceGeometry",
    "add_slice_contours",
    "plot_slice_tripcolor",
    "triangulated_slice_geometry",
]
