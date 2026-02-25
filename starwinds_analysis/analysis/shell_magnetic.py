"""THIS FILE contains magnetic-field shell sampling products and convenience shell map plotting.

It builds reusable shell magnetic component maps/summaries for examples and diagnostics.
It should reuse shell sampling and spherical transforms, and avoid redefining magnetic quantities elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np

from starwinds_analysis.analysis.shells import (
    SphericalShellSamples,
    integrate_shell_scalar,
    resolve_batsrus_vector_xyz_si,
    sample_spherical_shells,
)
from starwinds_analysis.recipes.spherical import spherical_vector_components


def _magnetic_unit_scale(unit: str) -> tuple[float, str]:
    key = str(unit).strip()
    table = {
        "T": (1.0, "T"),
        "Tesla": (1.0, "T"),
        "G": (1e4, "G"),
        "Gauss": (1e4, "G"),
        "nT": (1e9, "nT"),
    }
    if key not in table:
        raise ValueError(f"Unsupported magnetic display unit '{unit}'")
    return table[key]


@dataclass
class ShellMagneticFieldMap:
    radius: float
    theta: np.ndarray
    phi: np.ndarray
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    shell_samples: SphericalShellSamples
    b_r_T: np.ndarray
    b_theta_T: np.ndarray
    b_phi_T: np.ndarray
    b_meridional_T: np.ndarray
    b_tangential_T: np.ndarray

    def component(self, name: str, *, unit: str = "T") -> np.ndarray:
        scale, _unit_label = _magnetic_unit_scale(unit)
        key = str(name).lower()
        if key in {"radial", "r", "b_r"}:
            arr = self.b_r_T
        elif key in {"theta", "b_theta"}:
            arr = self.b_theta_T
        elif key in {"azimuthal", "phi", "b_phi"}:
            arr = self.b_phi_T
        elif key in {"meridional", "b_meridional", "north"}:
            arr = self.b_meridional_T
        elif key in {"tangential", "tan", "b_tangential"}:
            arr = self.b_tangential_T
        else:
            raise KeyError(f"Unknown magnetic component '{name}'")
        return scale * np.asarray(arr, dtype=float)

    def summary(self, *, unit: str = "G"):
        return summarize_shell_magnetic_field_map(self, unit=unit)


# TODO this function is retarded. WHY IS THIS EVEN A FILE
def sample_shell_magnetic_field_map(
    smart_ds,
    radius: float,
    *,
    n_polar: int = 48,
    n_azimuth: int = 96,
    coordinate_fields=("X [R]", "Y [R]", "Z [R]"),
    length_unit_to_m: float | None = None,
    method: str = "nearest",
):
    """
    Sample magnetic field on a spherical shell and return ZDI-style components.

    Components are stored internally in SI (Tesla). `b_meridional_T` is the
    northward tangent component, i.e. `-b_theta_T` for latitude-based maps.
    """
    (bx_name, by_name, bz_name), b_scale = resolve_batsrus_vector_xyz_si(smart_ds, "B")
    shell = sample_spherical_shells(
        smart_ds,
        [float(radius)],
        fields=(bx_name, by_name, bz_name),
        coordinate_fields=coordinate_fields,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        length_unit_to_m=length_unit_to_m,
        method=method,
    )

    bx = b_scale * np.asarray(shell.fields[bx_name][0], dtype=float)
    by = b_scale * np.asarray(shell.fields[by_name][0], dtype=float)
    bz = b_scale * np.asarray(shell.fields[bz_name][0], dtype=float)
    x = np.asarray(shell.x[0], dtype=float)
    y = np.asarray(shell.y[0], dtype=float)
    z = np.asarray(shell.z[0], dtype=float)

    b_r, b_theta, b_phi = spherical_vector_components(bx, by, bz, x, y, z)
    b_meridional = -b_theta
    b_tangential = np.sqrt(b_phi * b_phi + b_meridional * b_meridional)

    theta = np.asarray(shell.theta, dtype=float)
    phi = np.asarray(shell.phi, dtype=float)
    return ShellMagneticFieldMap(
        radius=float(radius),
        theta=theta,
        phi=phi,
        lon_deg=np.degrees(phi),
        lat_deg=90.0 - np.degrees(theta),
        shell_samples=shell,
        b_r_T=np.asarray(b_r, dtype=float),
        b_theta_T=np.asarray(b_theta, dtype=float),
        b_phi_T=np.asarray(b_phi, dtype=float),
        b_meridional_T=np.asarray(b_meridional, dtype=float),
        b_tangential_T=np.asarray(b_tangential, dtype=float),
    )


def style_shell_lonlat_axes(ax, *, title: str | None = None):
    """Dress a lon/lat axis so full-sphere coverage is visually obvious."""
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
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    if not np.any(positive):
        return np.ma.masked_invalid(values), None, int(np.count_nonzero(finite & (values <= 0.0)))

    norm = LogNorm()
    norm.autoscale_None(values[positive])
    under_value = max(np.nextafter(float(norm.vmin), 0.0), np.finfo(float).tiny)

    plot_values = np.array(values, dtype=float, copy=True)
    plot_values[finite & (plot_values <= 0.0)] = under_value
    return plot_values, norm, int(np.count_nonzero(finite & (values <= 0.0)))


def plot_shell_scalar_lonlat(
    ax,
    lon_deg,
    lat_deg,
    values,
    *,
    title: str | None = None,
    cbar_label: str | None = None,
    cmap: str = "viridis",
    symmetric: bool = False,
    vabs: float | None = None,
    scale: str = "linear",
    under_color: str | None = None,
):
    """
    Plot a shell scalar field on a longitude/latitude grid.

    `scale` may be `"linear"` or `"positive_log"`. In positive-log mode, non-positive
    values can be highlighted with the colormap under-color.
    """
    arr = np.asarray(values, dtype=float)
    cmap_obj = plt.get_cmap(cmap).copy()
    if under_color is not None:
        cmap_obj.set_under(under_color)

    norm = None
    extend = "neither"
    plot_values = arr
    extra = {}

    if scale == "positive_log":
        plot_values, norm, n_nonpos = _positive_log_plot_values(arr)
        extend = "min" if under_color is not None else "neither"
        extra["n_nonpositive"] = n_nonpos
    elif scale != "linear":
        raise ValueError("scale must be 'linear' or 'positive_log'")

    kwargs = {}
    if symmetric:
        if vabs is None:
            finite = arr[np.isfinite(arr)]
            vabs = float(np.max(np.abs(finite))) if finite.size else 1.0
        kwargs["vmin"] = -float(vabs)
        kwargs["vmax"] = float(vabs)

    img = ax.pcolormesh(
        lon_deg,
        lat_deg,
        plot_values,
        shading="nearest",
        cmap=cmap_obj,
        norm=norm,
        **kwargs,
    )
    style_shell_lonlat_axes(ax, title=title)
    cbar = ax.figure.colorbar(img, ax=ax, extend=extend)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    return img, cbar, extra


def plot_magnetic_zdi_triplet(
    shell_map: ShellMagneticFieldMap,
    *,
    unit: str = "G",
    figsize=(10, 10),
    cmap: str = "RdBu_r",
    share_scale: bool = True,
):
    """
    Plot radial/azimuthal/meridional shell magnetic components in a ZDI-like 3x1 layout.
    """
    _scale, unit_label = _magnetic_unit_scale(unit)
    components = [
        ("radial", "Radial", shell_map.component("radial", unit=unit)),
        ("azimuthal", "Azimuthal", shell_map.component("azimuthal", unit=unit)),
        ("meridional", "Meridional", shell_map.component("meridional", unit=unit)),
    ]

    shared_vabs = None
    if share_scale:
        vals = np.concatenate(
            [np.ravel(v[np.isfinite(v)]) for _key, _label, v in components if np.any(np.isfinite(v))]
        )
        shared_vabs = float(np.max(np.abs(vals))) if vals.size else 1.0

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    for ax, (_key, label, arr) in zip(np.ravel(axes), components):
        plot_shell_scalar_lonlat(
            ax,
            shell_map.lon_deg,
            shell_map.lat_deg,
            arr,
            title=f"{label} field at R={shell_map.radius:g}",
            cbar_label=f"{label} [{unit_label}]",
            cmap=cmap,
            symmetric=True,
            vabs=shared_vabs,
        )
    return fig, np.asarray(axes)


def plot_shell_tangential_vectors_lonlat(
    shell_map: ShellMagneticFieldMap,
    *,
    unit: str = "G",
    figsize=(10, 4.8),
    background: str = "tangential",
    background_scale: str = "linear",
    arrow_stride: tuple[int, int] = (3, 4),
    normalize_arrows: bool = True,
    arrow_length_deg: float = 8.0,
    arrow_color: str = "white",
    overlay_radial_zero_contour: bool = False,
    radial_zero_contour_color: str = "black",
    radial_zero_contour_alpha: float = 0.7,
    radial_zero_contour_linewidth: float = 0.8,
):
    """
    Plot a tangential magnetic field vector map on longitude/latitude axes.

    The background defaults to `|B_tan|`. Arrows show tangent direction. When
    `normalize_arrows=True`, arrow length is fixed in plot coordinates and does not
    encode magnitude (magnitude is shown by the background).
    """
    _scale, unit_label = _magnetic_unit_scale(unit)
    lon_deg = shell_map.lon_deg
    lat_deg = shell_map.lat_deg

    if background == "tangential":
        bg_values = shell_map.component("tangential", unit=unit)
        bg_name = "|B_tan|"
        bg_cmap = "viridis"
        bg_under = "magenta" if background_scale == "positive_log" else None
    elif background == "radial":
        bg_values = shell_map.component("radial", unit=unit)
        bg_name = "B_r"
        bg_cmap = "RdBu_r"
        bg_under = None
    else:
        raise ValueError("background must be 'tangential' or 'radial'")

    b_phi = shell_map.component("azimuthal", unit=unit)
    b_mer = shell_map.component("meridional", unit=unit)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _img, _cbar, extra = plot_shell_scalar_lonlat(
        ax,
        lon_deg,
        lat_deg,
        bg_values,
        title=f"Tangential vectors at R={shell_map.radius:g}",
        cbar_label=f"{bg_name} [{unit_label}]",
        cmap=bg_cmap,
        scale=background_scale,
        under_color=bg_under,
    )

    i_step = max(1, int(arrow_stride[0]))
    j_step = max(1, int(arrow_stride[1]))
    lon_q = lon_deg[::i_step, ::j_step]
    lat_q = lat_deg[::i_step, ::j_step]
    b_phi_q = b_phi[::i_step, ::j_step]
    b_mer_q = b_mer[::i_step, ::j_step]

    # Project eastward component into lon/lat map coordinates. Longitude spacing
    # shrinks with cos(latitude), so divide by cos(lat) for direction on the map.
    cos_lat = np.cos(np.deg2rad(lat_q))
    u = np.full_like(b_phi_q, np.nan, dtype=float)
    mask_lon = np.abs(cos_lat) > 1e-6
    u[mask_lon] = b_phi_q[mask_lon] / cos_lat[mask_lon]
    v = np.asarray(b_mer_q, dtype=float)

    finite_vec = np.isfinite(lon_q) & np.isfinite(lat_q) & np.isfinite(u) & np.isfinite(v)
    if normalize_arrows:
        mag = np.sqrt(u * u + v * v)
        good = finite_vec & (mag > 0)
        u_plot = np.full_like(u, np.nan, dtype=float)
        v_plot = np.full_like(v, np.nan, dtype=float)
        u_plot[good] = arrow_length_deg * u[good] / mag[good]
        v_plot[good] = arrow_length_deg * v[good] / mag[good]
    else:
        u_plot = u
        v_plot = v

    qmask = np.isfinite(lon_q) & np.isfinite(lat_q) & np.isfinite(u_plot) & np.isfinite(v_plot)
    q = ax.quiver(
        lon_q[qmask],
        lat_q[qmask],
        u_plot[qmask],
        v_plot[qmask],
        color=arrow_color,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
        pivot="mid",
    )

    zero_contour = None
    if overlay_radial_zero_contour:
        b_r = shell_map.component("radial", unit=unit)
        zero_contour = ax.contour(
            shell_map.lon_deg,
            shell_map.lat_deg,
            b_r,
            levels=[0.0],
            colors=radial_zero_contour_color,
            linewidths=float(radial_zero_contour_linewidth),
            alpha=float(radial_zero_contour_alpha),
        )
    return fig, ax, {"quiver": q, "radial_zero_contour": zero_contour, **extra}


def _finite_rms(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(finite * finite)))


def summarize_shell_magnetic_field_map(shell_map: ShellMagneticFieldMap, *, unit: str = "G"):
    """
    Compact summary for notebook/examples: fluxes, coverage, finite count, RMS components.
    """
    signed_flux, signed_cov = integrate_shell_scalar(
        shell_map.b_r_T[None, ...],
        shell_map.shell_samples.area[:1],
    )
    unsigned_flux, unsigned_cov = integrate_shell_scalar(
        np.abs(shell_map.b_r_T)[None, ...],
        shell_map.shell_samples.area[:1],
    )
    return {
        "finite_B_r_cells": int(np.count_nonzero(np.isfinite(shell_map.b_r_T))),
        "total_cells": int(shell_map.b_r_T.size),
        "signed_radial_flux [Wb]": float(signed_flux[0]),
        "signed_flux_coverage [none]": float(signed_cov[0]),
        "unsigned_radial_flux [Wb]": float(unsigned_flux[0]),
        "unsigned_flux_coverage [none]": float(unsigned_cov[0]),
        f"rms_B_r [{unit}]": _finite_rms(shell_map.component("radial", unit=unit)),
        f"rms_B_azimuthal [{unit}]": _finite_rms(shell_map.component("azimuthal", unit=unit)),
        f"rms_B_meridional [{unit}]": _finite_rms(shell_map.component("meridional", unit=unit)),
        f"rms_|B_tan| [{unit}]": _finite_rms(shell_map.component("tangential", unit=unit)),
    }


__all__ = [
    "ShellMagneticFieldMap",
    "plot_magnetic_zdi_triplet",
    "plot_shell_scalar_lonlat",
    "plot_shell_tangential_vectors_lonlat",
    "sample_shell_magnetic_field_map",
    "summarize_shell_magnetic_field_map",
    "style_shell_lonlat_axes",
]
