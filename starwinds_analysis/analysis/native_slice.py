from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
import numpy as np

from starwinds_analysis.utils import auto_coords, triangles as ds_triangles


@dataclass
class NativeSlice2DGeometry:
    x_field: str
    y_field: str
    x: np.ndarray
    y: np.ndarray
    triangulation: mtri.Triangulation


def native_slice_geometry(smart_ds, *, x_field: str | None = None, y_field: str | None = None):
    """
    Native 2D slice geometry from an already-2D dataset (no resampling).

    Uses library helpers `auto_coords(...)` and `utils.triangles(...)` when possible.
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

    return NativeSlice2DGeometry(
        x_field=str(x_field),
        y_field=str(y_field),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        triangulation=tris,
    )


def _tripcolor_values(values, *, scale: str):
    arr = np.asarray(values, dtype=float)
    arr_plot = np.ma.masked_invalid(arr)

    if scale == "linear":
        return arr, arr_plot, None, "linear"

    if scale == "positive_log":
        arr_plot = np.ma.masked_less_equal(arr_plot, 0.0)
        return arr, arr_plot, LogNorm(), "positive_log"

    if scale != "auto":
        raise ValueError("scale must be 'auto', 'linear', or 'positive_log'")

    finite = arr[np.isfinite(arr)]
    use_log = bool(finite.size) and float(np.nanmin(finite)) > 0.0
    if use_log:
        arr_plot = np.ma.masked_less_equal(arr_plot, 0.0)
        return arr, arr_plot, LogNorm(), "positive_log"
    return arr, arr_plot, None, "linear"


def plot_native_slice_tripcolor(
    smart_ds,
    field_name: str,
    *,
    geometry: NativeSlice2DGeometry | None = None,
    ax=None,
    figsize=(7, 6),
    cmap: str = "viridis",
    scale: str = "auto",
    show_mesh: bool = True,
    mesh_color: str = "k",
    mesh_linewidth: float = 0.15,
    mesh_alpha: float = 0.15,
):
    """
    Flat-shaded native 2D slice plot (no resampling, no notebook-side masking logic).
    """
    geom = native_slice_geometry(smart_ds) if geometry is None else geometry
    raw_values = np.asarray(smart_ds.variable(field_name), dtype=float)
    values, values_plot, norm, scale_used = _tripcolor_values(raw_values, scale=scale)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    img = ax.tripcolor(geom.triangulation, values_plot, shading="flat", cmap=cmap, norm=norm)
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
    ax.set_title(str(field_name))
    cbar = fig.colorbar(img, ax=ax, label=str(field_name))
    return fig, ax, {"image": img, "colorbar": cbar, "scale": scale_used, "geometry": geom, "values": values}


def plot_alfven_mach_slice(
    smart_ds,
    *,
    geometry: NativeSlice2DGeometry | None = None,
    ax=None,
    figsize=(7, 6),
    ma_field: str = "M_A [none]",
    ensure_batsrus_graph: bool = True,
    graph_kwargs: dict | None = None,
    cmap: str = "cividis",
    vmin: float = 1e-2,
    vmax: float = 1e2,
    contour_level: float = 1.0,
    contour_color: str = "crimson",
    contour_linewidth: float = 1.5,
):
    """
    Fixed-range log plot of Alfvén Mach number with optional `M_A=1` contour.

    Values outside `[vmin, vmax]` use the colormap under/over colors.
    """
    if ensure_batsrus_graph and not smart_ds.has_field(ma_field):
        kwargs = {} if graph_kwargs is None else dict(graph_kwargs)
        smart_ds.add_batsrus_graph(**kwargs)

    geom = native_slice_geometry(smart_ds) if geometry is None else geometry
    ma = np.asarray(smart_ds.variable(ma_field), dtype=float)
    ma_plot = np.ma.masked_invalid(ma)
    ma_plot = np.ma.masked_less_equal(ma_plot, 0.0)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_under(cmap_obj(0.0))
    cmap_obj.set_over(cmap_obj(1.0))
    norm = LogNorm(vmin=float(vmin), vmax=float(vmax))
    img = ax.tripcolor(geom.triangulation, ma_plot, shading="flat", cmap=cmap_obj, norm=norm)

    finite = ma[np.isfinite(ma)]
    contour_drawn = False
    contour = None
    if finite.size and (float(np.nanmin(finite)) <= contour_level <= float(np.nanmax(finite))):
        contour = ax.tricontour(
            geom.triangulation,
            ma,
            levels=[float(contour_level)],
            colors=contour_color,
            linewidths=float(contour_linewidth),
        )
        contour_drawn = True

    ax.set_aspect("equal")
    ax.set_xlabel(geom.x_field)
    ax.set_ylabel(geom.y_field)
    ax.set_title("Alfvén Mach number")
    cbar = fig.colorbar(img, ax=ax, label="M_A", extend="both")
    return fig, ax, {
        "image": img,
        "colorbar": cbar,
        "geometry": geom,
        "contour": contour,
        "contour_drawn": contour_drawn,
    }


__all__ = [
    "NativeSlice2DGeometry",
    "native_slice_geometry",
    "plot_native_slice_tripcolor",
    "plot_alfven_mach_slice",
]
