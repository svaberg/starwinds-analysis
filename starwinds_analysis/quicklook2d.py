from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.analysis.fluxes import (
    axisymmetric_open_flux_vs_radius,
    energy_flux_vs_radius,
    open_magnetic_flux_vs_radius,
    plot_energy_flux_profile,
    plot_open_flux_profile,
)
from starwinds_analysis.analysis.mass_loss import mass_loss_vs_radius, plot_mass_loss_profile
from starwinds_analysis.analysis.torque import plot_torque_profile, torque_vs_radius
from starwinds_analysis.utils import triangles
from starwinds_analysis.visualisation.slice import (
    plot_xz_slice_tripcolor_with_cross_quantiles,
    plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords,
    plot_xz_slice_tripcolor_with_marginals,
    plot_xz_slice_with_marginal_points,
)


@dataclass(frozen=True)
class SlicePreset:
    field_candidates: tuple[str, ...]
    overlays: tuple[tuple[str, float, str], ...] = ()


SLICE_PRESETS: dict[str, SlicePreset] = {
    "rho": SlicePreset(("Rho [kg/m^3]", "Rho [g/cm^3]", "Rho [amu/cm^3]")),
    "b_r": SlicePreset(
        ("B_r [T]", "B_r [Gauss]", "B_r [G]"),
        overlays=(
            ("B_r [T]", 0.0, "k"),
            ("B_r [Gauss]", 0.0, "k"),
            ("Ma [none]", 1.0, "C0"),
            ("M_A [none]", 1.0, "C2"),
            ("beta [none]", 1.0, "C3"),
        ),
    ),
    "u_r": SlicePreset(
        ("U_r [m/s]", "U_r [km/s]"),
        overlays=(
            ("U_r [m/s]", 0.0, "C3"),
            ("U_r [km/s]", 0.0, "C3"),
            ("B_r [T]", 0.0, "C2"),
            ("B_r [Gauss]", 0.0, "C2"),
            ("M_A [none]", 1.0, "C0"),
            ("beta [none]", 1.0, "C4"),
        ),
    ),
    "ti": SlicePreset(("ti [K]",)),
    "te": SlicePreset(("te [K]",)),
    "ma": SlicePreset(("Ma [none]",), overlays=(("Ma [none]", 1.0, "k"),)),
    "m_a": SlicePreset(("M_A [none]",), overlays=(("M_A [none]", 1.0, "k"), ("beta [none]", 1.0, "C3"))),
    "beta": SlicePreset(("beta [none]",), overlays=(("beta [none]", 1.0, "k"),)),
}


_SLICE_STYLES = {
    "marginals": plot_xz_slice_tripcolor_with_marginals,
    "cross_quantiles": plot_xz_slice_tripcolor_with_cross_quantiles,
    "marginal_points": plot_xz_slice_with_marginal_points,
    "unique_quantiles": plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords,
}


def _has_field(ds, name: str) -> bool:
    if hasattr(ds, "has_field"):
        try:
            return bool(ds.has_field(name))
        except Exception:
            return False
    try:
        ds.variable(name)
    except Exception:
        return False
    return True


def _resolve_first_field(ds, candidates):
    for name in candidates:
        if _has_field(ds, name):
            return name
    return None


def _normalize_overlays(ds, overlays):
    out = []
    for item in overlays or ():
        if len(item) == 2:
            field, level = item
            color = "k"
        else:
            field, level, color = item
        if _has_field(ds, field):
            out.append((field, float(level), color))
    return out


def plot_slice_quicklook(
    ds,
    *,
    preset: str | None = None,
    field: str | None = None,
    style: str = "cross_quantiles",
    overlays=None,
    contour_kwargs=None,
    **slice_kwargs,
):
    """
    Thin 2D quicklook wrapper over existing slice plotting helpers.
    """
    if field is None:
        if preset is None:
            raise ValueError("Provide either field=... or preset=...")
        if preset not in SLICE_PRESETS:
            raise KeyError(f"Unknown preset '{preset}'")
        preset_cfg = SLICE_PRESETS[preset]
        field = _resolve_first_field(ds, preset_cfg.field_candidates)
        if field is None:
            joined = ", ".join(preset_cfg.field_candidates)
            raise KeyError(f"None of the preset fields are available for '{preset}': {joined}")
        if overlays is None:
            overlays = preset_cfg.overlays

    if style not in _SLICE_STYLES:
        raise KeyError(f"Unknown style '{style}'. Valid styles: {sorted(_SLICE_STYLES)}")

    fig, axes, cbar = _SLICE_STYLES[style](ds, var=field, **slice_kwargs)
    ax_main = axes[0]

    contour_kwargs = dict(contour_kwargs or {})
    tris = triangles(ds)
    for overlay_field, level, color in _normalize_overlays(ds, overlays):
        values = np.asarray(ds.variable(overlay_field))
        kwargs = {"levels": [level], "colors": [color], "linewidths": 1.0}
        kwargs.update(contour_kwargs)
        ax_main.tricontour(tris, values, **kwargs)

    return fig, axes, cbar


def compute_shell_diagnostics(
    smart_ds,
    radii,
    *,
    body_radius_m: float,
    n_polar: int = 24,
    n_azimuth: int = 48,
    method: str = "nearest",
    include=("mass_loss", "torque", "open_flux", "energy", "axisymmetric_open_flux"),
):
    """
    Compute a bundle of shell diagnostics used by 2D quicklook summaries.
    """
    include = tuple(include)
    out = {}
    common = dict(
        body_radius_m=body_radius_m,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        method=method,
    )
    if "mass_loss" in include:
        out["mass_loss"] = mass_loss_vs_radius(smart_ds, radii, **common)
    if "torque" in include:
        out["torque"] = torque_vs_radius(smart_ds, radii, **common)
    if "open_flux" in include:
        out["open_flux"] = open_magnetic_flux_vs_radius(smart_ds, radii, **common)
    if "energy" in include:
        out["energy"] = energy_flux_vs_radius(smart_ds, radii, **common)
    if "axisymmetric_open_flux" in include:
        out["axisymmetric_open_flux"] = axisymmetric_open_flux_vs_radius(smart_ds, radii, **common)
    return out


def plot_shell_diagnostics(diagnostics, *, figsize=(12, 8)):
    """
    Plot a compact shell-diagnostics summary figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axs = np.asarray(axs)

    if "mass_loss" in diagnostics:
        plot_mass_loss_profile(axs[0, 0], diagnostics["mass_loss"])
        axs[0, 0].set_title("Wind Mass Loss")
        axs[0, 0].set_yscale("symlog", linthresh=1e-3)

    if "torque" in diagnostics:
        plot_torque_profile(axs[0, 1], diagnostics["torque"])
        axs[0, 1].set_title("Wind Torque")
        axs[0, 1].set_yscale("symlog", linthresh=1e-3)
        axs[0, 1].legend(loc="best")

    if "open_flux" in diagnostics:
        plot_open_flux_profile(axs[1, 0], diagnostics["open_flux"])
        axs[1, 0].set_title("Open Magnetic Flux")
        if "axisymmetric_open_flux" in diagnostics:
            p = diagnostics["axisymmetric_open_flux"]
            h = np.asarray(p["height [R]"])
            frac = np.asarray(p["axisymmetric_open_flux_fraction [none]"])
            ax2 = axs[1, 0].twinx()
            ax2.plot(h, frac, ".-", color="C3", label="axisymmetric fraction")
            ax2.set_ylabel("Axisymmetric fraction [none]")
            ax2.set_ylim(0, 1.05)

    if "energy" in diagnostics:
        plot_energy_flux_profile(axs[1, 1], diagnostics["energy"])
        axs[1, 1].set_title("Energy Flux")
        axs[1, 1].set_yscale("symlog", linthresh=1e-3)

    for ax in axs.ravel():
        ax.grid(True, alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
        ax.set_xscale("symlog", linthresh=1e-2)

    return fig, axs


def quicklook_shell_figure(
    smart_ds,
    radii,
    *,
    body_radius_m: float,
    n_polar: int = 24,
    n_azimuth: int = 48,
    method: str = "nearest",
    include=("mass_loss", "torque", "open_flux", "energy", "axisymmetric_open_flux"),
    figsize=(12, 8),
):
    diagnostics = compute_shell_diagnostics(
        smart_ds,
        radii,
        body_radius_m=body_radius_m,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        method=method,
        include=include,
    )
    fig, axs = plot_shell_diagnostics(diagnostics, figsize=figsize)
    return fig, axs, diagnostics


__all__ = [
    "SLICE_PRESETS",
    "SlicePreset",
    "compute_shell_diagnostics",
    "plot_shell_diagnostics",
    "plot_slice_quicklook",
    "quicklook_shell_figure",
]

