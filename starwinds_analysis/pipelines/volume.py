"""THIS FILE contains high-level non-VTK quicklook orchestration and convenience wrappers.

It assembles analysis results, plots, and exports for quicklook-style workflows.
Core quantity definitions and sampling primitives should live in analysis modules instead.
"""

# TODO(debt): This is a high-level orchestration/wrapper module inside the library and
# contains quantity-specific presets/workflows. Re-evaluate this API against the
# library-purity rule and keep one-off example composition in notebooks/scripts.

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.physics.planetary_orbits import SOLAR_SYSTEM_PLANETS
from starwinds_analysis.physics.orbit_pressure import pressure_components_on_circular_orbit
from starwinds_analysis.physics.orbit_pressure import pressure_components_on_elliptic_orbit
from starwinds_analysis.physics.orbit_surface import pressure_components_on_orbit_surface
from starwinds_analysis.physics.orbit_surface import torque_components_on_orbit_surface
from starwinds_analysis.analysis.shell_summary import summarize_shell_diagnostics_band
from starwinds_analysis.analysis.slices import resample_structured_xz_slice
from starwinds_analysis.physics.mass_loss import mass_loss_vs_radius
from starwinds_analysis.physics.orbit_local import local_mass_loss_on_circular_orbit
from starwinds_analysis.physics.orbit_local import local_mass_loss_on_elliptic_orbit
from starwinds_analysis.physics.orbit_local import local_torque_on_circular_orbit
from starwinds_analysis.physics.orbit_local import local_torque_on_elliptic_orbit
from starwinds_analysis.physics.fluxes import axisymmetric_open_flux_vs_radius
from starwinds_analysis.physics.fluxes import energy_flux_vs_radius
from starwinds_analysis.physics.fluxes import open_magnetic_flux_vs_radius
from starwinds_analysis.visualisation.profile_plots import plot_shell_height_series
from starwinds_analysis.visualisation.profile_plots import shell_profile_height
from starwinds_analysis.physics.torque import torque_vs_radius
from starwinds_analysis.physics.wind_scaling import open_wind_magnetisation
from starwinds_analysis.utils import triangles
from starwinds_analysis.visualisation.slice import plot_xz_slice_tripcolor_with_cross_quantiles
from starwinds_analysis.visualisation.slice import plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords
from starwinds_analysis.visualisation.slice import plot_xz_slice_tripcolor_with_marginals
from starwinds_analysis.visualisation.slice import plot_xz_slice_with_marginal_points
from starwinds_analysis.visualisation.histograms import plot_binned_vs_radius
from starwinds_analysis.visualisation.histograms import plot_cumulative_hists
from starwinds_analysis.visualisation.histograms import plot_radial_hist2d
from starwinds_analysis.visualisation.histograms import plot_vs_radius
from starwinds_analysis.pipelines.orchestration_helpers import array_summary as _array_summary
from starwinds_analysis.pipelines.orchestration_helpers import is_2d_input
from starwinds_analysis.pipelines.orchestration_helpers import log_pipeline_event
from starwinds_analysis.pipelines.orchestration_helpers import prepare_smartds
from starwinds_analysis.pipelines.orchestration_helpers import resolve_quicklook_prefix as _resolve_quicklook_prefix
from starwinds_analysis.pipelines.orchestration_helpers import slug_key as _slug_key
from starwinds_analysis.smart_ds import SmartDs

log = logging.getLogger(__name__)
pipeline_log = log.getChild("pipeline")
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
DEFAULT_STAR_RADIUS_M = 6.957e8
DEFAULT_QUICKLOOK_RADII_R = (2.0, 4.0, 8.0, 16.0)

@dataclass(frozen=True)
class SlicePreset:
    field_candidates: tuple[str, ...]
    overlays: tuple[tuple[str, float, str], ...] = ()
    intent: str = "si_diagnostic"

SLICE_PRESETS_SI_DIAGNOSTIC: dict[str, SlicePreset] = {
    "rho": SlicePreset(("Rho [kg/m^3]",), intent="si_diagnostic"),
    "b_r": SlicePreset(
        ("B_r [T]",),
        overlays=(
            ("B_r [T]", 0.0, "k"),
            ("Ma [none]", 1.0, "C0"),
            ("M_A [none]", 1.0, "C2"),
            ("beta [none]", 1.0, "C3"),
        ),
        intent="si_diagnostic",
    ),
    "u_r": SlicePreset(
        ("U_r [m/s]",),
        overlays=(
            ("U_r [m/s]", 0.0, "C3"),
            ("B_r [T]", 0.0, "C2"),
            ("M_A [none]", 1.0, "C0"),
            ("beta [none]", 1.0, "C4"),
        ),
        intent="si_diagnostic",
    ),
    "ti": SlicePreset(("ti [K]",), intent="si_diagnostic"),
    "te": SlicePreset(("te [K]",), intent="si_diagnostic"),
    "ma": SlicePreset(("Ma [none]",), overlays=(("Ma [none]", 1.0, "k"),), intent="si_diagnostic"),
    "m_a": SlicePreset(
        ("M_A [none]",),
        overlays=(("M_A [none]", 1.0, "k"), ("beta [none]", 1.0, "C3")),
        intent="si_diagnostic",
    ),
    "beta": SlicePreset(("beta [none]",), overlays=(("beta [none]", 1.0, "k"),), intent="si_diagnostic"),
}

SLICE_PRESETS_RAW_DISPLAY: dict[str, SlicePreset] = {
    "rho_raw": SlicePreset(("Rho [g/cm^3]", "Rho [amu/cm^3]"), intent="raw_display"),
    "b_r_raw": SlicePreset(
        ("B_r [Gauss]", "B_r [G]"),
        overlays=(("B_r [Gauss]", 0.0, "k"), ("B_r [G]", 0.0, "k")),
        intent="raw_display",
    ),
    "u_r_raw": SlicePreset(
        ("U_r [km/s]",),
        overlays=(("U_r [km/s]", 0.0, "C3"), ("B_r [Gauss]", 0.0, "C2"), ("B_r [G]", 0.0, "C2")),
        intent="raw_display",
    ),
}

SLICE_PRESETS: dict[str, SlicePreset] = {
    **SLICE_PRESETS_SI_DIAGNOSTIC,
    **SLICE_PRESETS_RAW_DISPLAY,
}

RADIAL_SUMMARY_PRESETS_SI_DIAGNOSTIC: dict[str, tuple[str, ...]] = {
    "wind_basic": (
        "Rho [kg/m^3]",
        "U [m/s]",
        "B [T]",
        "P [Pa]",
    ),
}

RADIAL_SUMMARY_PRESETS_RAW_DISPLAY: dict[str, tuple[str, ...]] = {
    "wind_raw": (
        "Rho [g/cm^3]",
        "U_x [km/s]",
        "B_x [Gauss]",
        "P [dyne/cm^2]",
    ),
}

RADIAL_SUMMARY_PRESETS: dict[str, tuple[str, ...]] = {
    **RADIAL_SUMMARY_PRESETS_SI_DIAGNOSTIC,
    **RADIAL_SUMMARY_PRESETS_RAW_DISPLAY,
}

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
    Used by: `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if field is None:
        if preset is None:
            raise ValueError("Provide either field=... or preset=...")
        if preset not in SLICE_PRESETS:
            raise KeyError(f"Unknown preset '{preset}'")
        preset_cfg = SLICE_PRESETS[preset]
        field = None
        for name in preset_cfg.field_candidates:
            try:
                ds.variable(name)
            except Exception:
                continue
            field = name
            break
        if field is None:
            joined = ", ".join(preset_cfg.field_candidates)
            raise KeyError(f"None of the preset fields are available for '{preset}': {joined}")
        if overlays is None:
            overlays = preset_cfg.overlays

    if style == "marginals":
        fig, axes, cbar = plot_xz_slice_tripcolor_with_marginals(ds, var=field, **slice_kwargs)
    elif style == "cross_quantiles":
        fig, axes, cbar = plot_xz_slice_tripcolor_with_cross_quantiles(ds, var=field, **slice_kwargs)
    elif style == "marginal_points":
        fig, axes, cbar = plot_xz_slice_with_marginal_points(ds, var=field, **slice_kwargs)
    elif style == "unique_quantiles":
        fig, axes, cbar = plot_xz_slice_tripcolor_with_marginal_quantiles_by_unique_coords(ds, var=field, **slice_kwargs)
    else:
        raise KeyError("Unknown style '%s'. Valid styles: %s" % (style, ["cross_quantiles", "marginal_points", "marginals", "unique_quantiles"]))
    ax_main = axes[0]
    tris = triangles(ds)
    for item in overlays or ():
        if len(item) == 2:
            overlay_field, level = item
            color = "k"
        else:
            overlay_field, level, color = item
        try:
            values = np.array(ds.variable(overlay_field))
        except Exception:
            continue
        kwargs = {"levels": [level], "colors": [color], "linewidths": 1.0}
        if contour_kwargs:
            kwargs.update(contour_kwargs)
        ax_main.tricontour(tris, values, **kwargs)

    return fig, axes, cbar

def plot_radius_quicklook(
    ds,
    *,
    fields=None,
    preset: str | None = None,
    mode: str = "binned",
    ncols: int = 2,
    figsize=None,
    **plot_kwargs,
):
    """
    Radius/scatter/cumulative/hist2d quicklook wrapper over `visualisation.histograms`.
    Used by: `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if fields is None:
        if preset is None:
            raise ValueError("Provide either fields=... or preset=...")
        if preset not in RADIAL_SUMMARY_PRESETS:
            raise KeyError(f"Unknown radial preset '{preset}'")
        resolved_fields = []
        for candidate in RADIAL_SUMMARY_PRESETS[preset]:
            try:
                ds.variable(candidate)
            except Exception:
                continue
            resolved_fields.append(candidate)
        fields = tuple(resolved_fields)
        if not fields:
            raise KeyError(f"No fields from preset '{preset}' are available")
    else:
        fields = tuple(fields)

    n = len(fields)
    if n == 0:
        raise ValueError("No fields requested")
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 3.0 * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axs = np.array(axs).ravel()

    if mode == "binned":
        plot_binned_vs_radius(ds, axs, fields=fields, **plot_kwargs)
    elif mode == "scatter":
        plot_vs_radius(ds, axs, fields=fields, **plot_kwargs)
    elif mode in ("hist2d", "monster"):
        plot_radial_hist2d(ds, axs, fields=fields, **plot_kwargs)
    elif mode in ("cdf", "cumulative"):
        plot_cumulative_hists(ds, axs, fields=fields, **plot_kwargs)
    else:
        raise KeyError("mode must be 'binned', 'scatter', 'hist2d', or 'cdf'")

    for ax in axs[n:]:
        ax.set_visible(False)
    return fig, axs[:n]

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
    Used by: `starwinds_analysis/pipelines/volume.py`
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
    Used by: `starwinds_analysis/pipelines/volume.py`
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axs = np.array(axs)

    if "mass_loss" in diagnostics:
        plot_shell_height_series(
            axs[0, 0],
            diagnostics["mass_loss"],
            "mass_loss [kg/s]",
            label="mass loss",
            ylabel="Mass flux [kg/s]",
            color="C0",
            show_negative=True,
        )
        axs[0, 0].set_title("Wind Mass Loss")
        axs[0, 0].set_yscale("symlog", linthresh=1e-3)

    if "torque" in diagnostics:
        plot_shell_height_series(
            axs[0, 1],
            diagnostics["torque"],
            "total_torque [Nm]",
            label="total",
            ylabel="Torque [Nm]",
            color="C0",
            show_negative=True,
        )
        h = shell_profile_height(diagnostics["torque"])
        axs[0, 1].plot(h, np.array(diagnostics["torque"]["magnetic_torque [Nm]"]), ".-", color="C1", label="magnetic")
        axs[0, 1].plot(h, np.array(diagnostics["torque"]["dynamic_torque [Nm]"]), ".-", color="C2", label="dynamic")
        axs[0, 1].set_title("Wind Torque")
        axs[0, 1].set_yscale("symlog", linthresh=1e-3)
        axs[0, 1].legend(loc="best")

    if "open_flux" in diagnostics:
        plot_shell_height_series(
            axs[1, 0],
            diagnostics["open_flux"],
            "open_flux [Wb]",
            label="open flux",
            ylabel="Open magnetic flux [Wb]",
            color="C0",
            show_negative=False,
        )
        axs[1, 0].set_title("Open Magnetic Flux")
        ax2 = None
        if "axisymmetric_open_flux" in diagnostics:
            p = diagnostics["axisymmetric_open_flux"]
            h = np.array(p["height [R]"])
            frac = np.array(p["axisymmetric_open_flux_fraction [none]"])
            ax2 = axs[1, 0].twinx()
            ax2.plot(h, frac, ".-", color="C3", label="axisymmetric fraction")
            ax2.set_ylabel("Axisymmetric fraction [none]")
            ax2.set_ylim(0, 1.05)
        if "wind_scaling" in diagnostics:
            p = diagnostics["wind_scaling"]
            h = np.array(p["height [R]"])
            y = np.array(p["Upsilon_open [none]"])
            if ax2 is None:
                ax2 = axs[1, 0].twinx()
            ax2.plot(h, y, ".-", color="C4", label="Upsilon_open")
            ax2.set_ylabel("Axisymmetric fraction / Upsilon_open")
            finite = np.isfinite(y) & (y > 0)
            if np.any(finite):
                ax2.set_yscale("log")

    if "energy" in diagnostics:
        plot_shell_height_series(
            axs[1, 1],
            diagnostics["energy"],
            "energy_flux [W]",
            label="energy flux",
            ylabel="Energy flux [W]",
            color="C0",
            show_negative=True,
        )
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
    star_mass_kg: float | None = None,
    figsize=(12, 8),
):
    """
    Convenience shell-diagnostics figure builder used by quicklook and tests.
    Used by: `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/volume.py`
    """
    diagnostics = compute_shell_diagnostics(
        smart_ds,
        radii,
        body_radius_m=body_radius_m,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        method=method,
        include=include,
    )
    if star_mass_kg is not None:
        try:
            phi = np.array(diagnostics["open_flux"]["open_flux [Wb]"])
            dotm = np.array(diagnostics["mass_loss"]["mass_loss [kg/s]"])
            y = open_wind_magnetisation(phi, dotm, star_mass_kg, body_radius_m)
            diagnostics["wind_scaling"] = {
                "radius [R]": np.array(diagnostics["mass_loss"]["radius [R]"]),
                "height [R]": np.array(diagnostics["mass_loss"]["height [R]"]),
                "Upsilon_open [none]": np.array(y),
            }
        except Exception:
            pass
    fig, axs = plot_shell_diagnostics(diagnostics, figsize=figsize)
    return fig, axs, diagnostics

def plot_orbit_mass_loss_comparison(ax, result):
    """
    Plot local-vs-shell mass-loss comparison for one orbit result bundle.
    Used by: `starwinds_analysis/pipelines/volume.py`
    """
    y = np.array(result["local_mass_loss [kg/s]"])
    phase = _orbit_phase(result, y.size)
    shell_interp = result.get("shell_mass_loss_interp [kg/s]")
    shell = float(result["shell_mass_loss [kg/s]"])
    mean = float(result["summary"]["mean"])

    ax.plot(phase, y, ",", color="C0", alpha=0.6, label="local estimate")
    if shell_interp is not None:
        ax.plot(
            phase,
            np.array(shell_interp),
            "-",
            color="C1",
            alpha=0.9,
            label="shell (interp)",
        )
    else:
        ax.axhline(shell, color="C1", linestyle="-", label="shell")
    ax.axhline(mean, color="C0", linestyle="--", label="local mean")
    ax.axhline(-shell, color="C1", linestyle=":")
    ax.set_xlabel("Orbit phase [turns]")
    ax.set_ylabel("Mass loss [kg/s]")
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        ax.set_title(
            f"Mass Loss @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    else:
        ax.set_title(f"Mass Loss @ r={float(result['radius [R]']):.3g} R")
    return ax

def plot_orbit_torque_comparison(ax, result, *, show_components: bool = True):
    """
    Plot local-vs-shell torque comparison for one orbit result bundle.
    Used by: `starwinds_analysis/pipelines/volume.py`
    """
    tot = np.array(result["local_total_torque [Nm]"])
    phase = _orbit_phase(result, tot.size)
    shell_interp = result.get("shell_total_torque_interp [Nm]")
    shell = float(result["shell_total_torque [Nm]"])
    mean = float(result["summary"]["mean"])

    ax.plot(phase, tot, ",", color="C0", alpha=0.6, label="local total")
    if show_components:
        ax.plot(phase, np.array(result["local_magnetic_torque [Nm]"]), ",", color="C1", alpha=0.35, label="local mag")
        ax.plot(phase, np.array(result["local_dynamic_torque [Nm]"]), ",", color="C2", alpha=0.35, label="local dyn")

    if shell_interp is not None:
        ax.plot(
            phase,
            np.array(shell_interp),
            "-",
            color="k",
            alpha=0.9,
            label="shell total (interp)",
        )
    else:
        ax.axhline(shell, color="k", linestyle="-", label="shell total")
    ax.axhline(mean, color="C0", linestyle="--", label="local mean")
    ax.axhline(-shell, color="k", linestyle=":")
    ax.set_xlabel("Orbit phase [turns]")
    ax.set_ylabel("Torque [Nm]")
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        ax.set_title(
            f"Torque @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    else:
        ax.set_title(f"Torque @ r={float(result['radius [R]']):.3g} R")
    return ax

def _orbit_phase(result, n):
    """
    Extract orbit phase array from result bundles with a safe empty fallback.
    Used by: `starwinds_analysis/pipelines/volume.py`
    """
    try:
        phase = np.array(result.get("orbit_samples", {}).get("phase [turns]"))
    except Exception:
        phase = np.array([])
    if phase.shape == (n,):
        return phase
    return np.arange(n, dtype=float) / max(1, n)

def plot_orbit_pressure_components(ax, result, *, include_relative: bool = True):
    """
    Plot orbit pressure-component series and derived standoff proxy.
    Used by: `starwinds_analysis/pipelines/volume.py`
    """
    phase = _orbit_phase(result, len(np.array(result["ram_pressure [Pa]"])))
    for key, label, color in (
        ("thermal_pressure [Pa]", "thermal", "C0"),
        ("magnetic_pressure [Pa]", "magnetic", "C1"),
        ("ram_pressure [Pa]", "ram", "C2"),
    ):
        if key in result:
            ax.plot(phase, np.array(result[key]), ",", color=color, alpha=0.65, label=label)
    if include_relative and "relative_ram_pressure [Pa]" in result:
        ax.plot(
            phase,
            np.array(result["relative_ram_pressure [Pa]"]),
            ",",
            color="C3",
            alpha=0.65,
            label="relative ram",
        )
    ax.set_xlabel("Orbit phase [turns]")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_yscale("log")
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        ax.set_title(
            f"Orbit Pressures @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    else:
        ax.set_title(f"Orbit Pressures @ r={float(result['radius [R]']):.3g} R")
    return ax

def orbit_pressure_figure(
    smart_ds,
    radius,
    *,
    body_radius_m: float,
    n_points: int = 360,
    plane: str = "xy",
    method: str = "nearest",
    star_mass_kg: float | None = None,
    figsize=(12, 4),
):
    """
    Orbit pressure quicklook (thermal/magnetic/ram and stand-off proxy).
    Used by: `test/test_quicklook2d.py`
    """
    if isinstance(radius, dict):
        spec = dict(radius)
        kind = str(spec.pop("kind", "kepler")).lower()
        if kind not in {"kepler", "elliptic", "ellipse"}:
            raise ValueError(f"Unsupported orbit kind: {kind}")
        a = float(spec.pop("semi_major_axis", spec.pop("a", np.nan)))
        if not np.isfinite(a):
            raise ValueError("orbit spec requires 'semi_major_axis' (or 'a')")
        e = float(spec.pop("eccentricity", 0.0))
        this_plane = str(spec.pop("plane", plane))
        this_n_points = int(spec.pop("n_points", n_points))
        angle0 = float(spec.pop("angle0", 0.0))
        sample = str(spec.pop("sample", "eccentric_anomaly"))
        spec.pop("label", None)
        if spec:
            raise ValueError(f"Unknown orbit spec keys: {sorted(spec)}")
        result = pressure_components_on_elliptic_orbit(
            smart_ds,
            a,
            eccentricity=e,
            body_radius_m=body_radius_m,
            n_points=this_n_points,
            plane=this_plane,
            angle0=angle0,
            sample=sample,
            method=method,
            star_mass_kg=star_mass_kg,
        )
    else:
        result = pressure_components_on_circular_orbit(
            smart_ds,
            radius,
            body_radius_m=body_radius_m,
            n_points=n_points,
            plane=plane,
            method=method,
            star_mass_kg=star_mass_kg,
        )

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    plot_orbit_pressure_components(axs[0], result)
    phase = _orbit_phase(result, len(np.array(result["standoff_distance [m]"])))
    axs[1].plot(phase, np.array(result["standoff_distance [m]"]), ",", color="C4", alpha=0.7)
    axs[1].set_xlabel("Orbit phase [turns]")
    axs[1].set_ylabel("Stand-off Proxy [m]")
    axs[1].set_yscale("log")
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        axs[1].set_title(
            f"Stand-off Proxy @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    else:
        axs[1].set_title(f"Stand-off Proxy @ r={float(result['radius [R]']):.3g} R")
    for ax in np.ravel(axs):
        ax.grid(True, alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
    axs[0].legend(loc="best")
    return fig, axs, result

def _plot_phase_quantile_band(ax, phase_profile, *, label, color, q_low=0.25, q_med=0.5, q_high=0.75):
    """
    Plot filled phase-quantile bands for orbit-surface diagnostics.
    Used by: `starwinds_analysis/pipelines/volume.py`
    """
    phase = np.array(phase_profile["phase [turns]"])
    qs = np.array(phase_profile["quantiles [none]"])
    vals = np.array(phase_profile["values"])
    if vals.ndim != 2 or vals.shape[0] != phase.size:
        return ax

    def _pick(qtarget):
        """
        Pick the nearest stored quantile column for plotting.
        Used by: `_plot_phase_quantile_band` (nested helper)
        """
        idx = int(np.argmin(np.abs(qs - qtarget)))
        return vals[:, idx]

    y_lo = _pick(q_low)
    y_md = _pick(q_med)
    y_hi = _pick(q_high)
    ax.fill_between(phase, y_lo, y_hi, color=color, alpha=0.15)
    ax.plot(phase, y_md, "-", color=color, label=label)
    return ax

def orbit_surface_pressure_figure(
    smart_ds,
    orbit,
    *,
    body_radius_m: float,
    n_longitudes: int = 199,
    method: str = "nearest",
    star_mass_kg: float | None = None,
    figsize=(12, 6),
):
    """
    Surface-of-revolution orbit pressure quicklook (pure NumPy/SciPy resampling).
    Used by: `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/volume.py`
    """
    result = pressure_components_on_orbit_surface(
        smart_ds,
        orbit,
        body_radius_m=body_radius_m,
        n_longitudes=n_longitudes,
        method=method,
        star_mass_kg=star_mass_kg,
    )
    pq = result["phase_quantiles"]

    fig, axs = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)
    axs = np.array(axs)

    for key, label, color in (
        ("thermal_pressure [Pa]", "thermal", "C0"),
        ("magnetic_pressure [Pa]", "magnetic", "C1"),
        ("ram_pressure [Pa]", "ram", "C2"),
    ):
        if key in pq:
            _plot_phase_quantile_band(axs[0], pq[key], label=label, color=color)
    if "relative_ram_pressure [Pa]" in pq:
        _plot_phase_quantile_band(axs[0], pq["relative_ram_pressure [Pa]"], label="relative ram", color="C3")
    axs[0].set_ylabel("Pressure [Pa]")
    axs[0].set_yscale("log")
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        axs[0].set_title(
            f"Orbit-Surface Pressures @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    else:
        axs[0].set_title(f"Orbit-Surface Pressures @ r={float(result['radius [R]']):.3g} R")
    axs[0].legend(loc="best")

    if "standoff_distance [m]" in pq:
        _plot_phase_quantile_band(axs[1], pq["standoff_distance [m]"], label="stand-off proxy", color="C4")
    axs[1].set_xlabel("Orbit phase [turns]")
    axs[1].set_ylabel("Stand-off Proxy [m]")
    axs[1].set_yscale("log")
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        axs[1].set_title(
            f"Orbit-Surface Stand-off @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    else:
        axs[1].set_title(f"Orbit-Surface Stand-off @ r={float(result['radius [R]']):.3g} R")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
    return fig, axs, result

def orbit_surface_torque_figure(
    smart_ds,
    orbit,
    *,
    body_radius_m: float,
    n_longitudes: int = 199,
    method: str = "nearest",
    angvel_rad_s: float = 0.0,
    figsize=(12, 6),
):
    """
    Surface-of-revolution torque quicklook (`T1..T4` + total), non-VTK.
    Used by: `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/volume.py`
    """
    result = torque_components_on_orbit_surface(
        smart_ds,
        orbit,
        body_radius_m=body_radius_m,
        n_longitudes=n_longitudes,
        method=method,
        angvel_rad_s=angvel_rad_s,
    )

    fig, axs = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)
    axs = np.array(axs)

    for key, label, color in (
        ("T1_magnetic", "T1 magnetic", "C1"),
        ("T2_pressure", "T2 pressure", "C3"),
        ("T3_corotation", "T3 corotation", "C4"),
        ("T4_dynamic", "T4 dynamic", "C2"),
        ("total", "total", "C0"),
    ):
        p = result["phase_integrals"][key]
        axs[0].plot(
            np.array(p["phase [turns]"]),
            np.array(p["integral [Nm]"]),
            ".-",
            color=color,
            alpha=0.8,
            label=label,
        )
    axs[0].set_ylabel("Phase Ring Integral [Nm]")
    axs[0].set_yscale("symlog", linthresh=1e-3)
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        axs[0].set_title(
            f"Orbit-Surface Torque Terms @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    else:
        axs[0].set_title(f"Orbit-Surface Torque Terms @ r={float(result['radius [R]']):.3g} R")
    axs[0].legend(loc="best", ncol=2)

    if "total" in result["phase_quantiles"]:
        _plot_phase_quantile_band(
            axs[1],
            {
                "phase [turns]": result["phase_quantiles"]["total"]["phase [turns]"],
                "quantiles [none]": result["phase_quantiles"]["total"]["quantiles [none]"],
                "values": result["phase_quantiles"]["total"]["values [N/m]"],
            },
            label="total density (median/IQR)",
            color="C0",
        )
    if "T1_magnetic" in result["phase_quantiles"]:
        _plot_phase_quantile_band(
            axs[1],
            {
                "phase [turns]": result["phase_quantiles"]["T1_magnetic"]["phase [turns]"],
                "quantiles [none]": result["phase_quantiles"]["T1_magnetic"]["quantiles [none]"],
                "values": result["phase_quantiles"]["T1_magnetic"]["values [N/m]"],
            },
            label="T1 density",
            color="C1",
        )
    if "T4_dynamic" in result["phase_quantiles"]:
        _plot_phase_quantile_band(
            axs[1],
            {
                "phase [turns]": result["phase_quantiles"]["T4_dynamic"]["phase [turns]"],
                "quantiles [none]": result["phase_quantiles"]["T4_dynamic"]["quantiles [none]"],
                "values": result["phase_quantiles"]["T4_dynamic"]["values [N/m]"],
            },
            label="T4 density",
            color="C2",
        )
    axs[1].set_xlabel("Orbit phase [turns]")
    axs[1].set_ylabel("Torque Density [N/m]")
    axs[1].set_yscale("symlog", linthresh=1e-6)
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        axs[1].set_title(
            f"Orbit-Surface Torque Density Quantiles @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    else:
        axs[1].set_title(f"Orbit-Surface Torque Density Quantiles @ r={float(result['radius [R]']):.3g} R")
    axs[1].legend(loc="best")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
    return fig, axs, result

def orbit_local_comparison_figure(
    smart_ds,
    radius,
    *,
    body_radius_m: float,
    n_points: int = 360,
    plane: str = "xy",
    method: str = "nearest",
    shell_n_polar: int = 24,
    shell_n_azimuth: int = 48,
    figsize=(12, 4),
):
    """
    Compute and plot local-vs-shell comparisons for mass loss and torque on one orbit.
    Used by: `test/test_quicklook2d.py`, `starwinds_analysis/pipelines/volume.py`
    """
    if isinstance(radius, dict):
        spec = dict(radius)
        kind = str(spec.pop("kind", "kepler")).lower()
        if kind not in {"kepler", "elliptic", "ellipse"}:
            raise ValueError(f"Unsupported orbit kind: {kind}")
        a = float(spec.pop("semi_major_axis", spec.pop("a", np.nan)))
        if not np.isfinite(a):
            raise ValueError("orbit spec requires 'semi_major_axis' (or 'a')")
        e = float(spec.pop("eccentricity", 0.0))
        this_plane = str(spec.pop("plane", plane))
        this_n_points = int(spec.pop("n_points", n_points))
        angle0 = float(spec.pop("angle0", 0.0))
        sample = str(spec.pop("sample", "eccentric_anomaly"))
        shell_n_radii = int(spec.pop("shell_n_radii", 12))
        spec.pop("label", None)
        if spec:
            raise ValueError(f"Unknown orbit spec keys: {sorted(spec)}")
        mass = local_mass_loss_on_elliptic_orbit(
            smart_ds,
            a,
            eccentricity=e,
            body_radius_m=body_radius_m,
            n_points=this_n_points,
            plane=this_plane,
            angle0=angle0,
            sample=sample,
            method=method,
            shell_n_polar=shell_n_polar,
            shell_n_azimuth=shell_n_azimuth,
            shell_n_radii=shell_n_radii,
        )
        torque = local_torque_on_elliptic_orbit(
            smart_ds,
            a,
            eccentricity=e,
            body_radius_m=body_radius_m,
            n_points=this_n_points,
            plane=this_plane,
            angle0=angle0,
            sample=sample,
            method=method,
            shell_n_polar=shell_n_polar,
            shell_n_azimuth=shell_n_azimuth,
            shell_n_radii=shell_n_radii,
        )
    else:
        mass = local_mass_loss_on_circular_orbit(
            smart_ds,
            radius,
            body_radius_m=body_radius_m,
            n_points=n_points,
            plane=plane,
            method=method,
            shell_n_polar=shell_n_polar,
            shell_n_azimuth=shell_n_azimuth,
        )
        torque = local_torque_on_circular_orbit(
            smart_ds,
            radius,
            body_radius_m=body_radius_m,
            n_points=n_points,
            plane=plane,
            method=method,
            shell_n_polar=shell_n_polar,
            shell_n_azimuth=shell_n_azimuth,
        )

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    plot_orbit_mass_loss_comparison(axs[0], mass)
    plot_orbit_torque_comparison(axs[1], torque)
    for ax in np.ravel(axs):
        ax.grid(True, alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
        ax.set_yscale("symlog", linthresh=1e-3)
    axs[1].legend(loc="best")
    return fig, axs, {"mass_loss": mass, "torque": torque}

def summarize_shell_diagnostics(
    diagnostics,
    *,
    band_radius_range=None,
    include_band_summary: bool = True,
    star_mass_kg: float | None = None,
    star_radius_m: float | None = None,
):
    """
    JSON-friendly summary (stats only) of shell diagnostics.
    Used by: `starwinds_analysis/pipelines/volume.py`
    """
    out = {}
    for name, profile in diagnostics.items():
        if not isinstance(profile, dict):
            continue
        pdata = {}
        for key, value in profile.items():
            if key == "shell_samples":
                continue
            arr = np.array(value)
            if arr.ndim == 0:
                try:
                    pdata[key] = float(arr)
                except Exception:
                    pdata[key] = str(value)
                continue
            pdata[key] = _array_summary(arr)
        out[name] = pdata

    if include_band_summary:
        rmin = rmax = None
        if band_radius_range is not None:
            rmin, rmax = band_radius_range
        out["_band_summary"] = summarize_shell_diagnostics_band(
            diagnostics,
            rmin=rmin,
            rmax=rmax,
        )

    if star_mass_kg is not None and star_radius_m is not None:
        try:
            phi = np.array(diagnostics["open_flux"]["open_flux [Wb]"])
            dotm = np.array(diagnostics["mass_loss"]["mass_loss [kg/s]"])
            y = open_wind_magnetisation(phi, dotm, star_mass_kg, star_radius_m)
            out["_wind_scaling"] = {
                "Upsilon_open [none]": _array_summary(y)
            }
        except Exception:
            pass
    return out

def shell_profile_payload(diagnostics):
    """
    JSON/NPZ-friendly shell-profile payload with 1D numeric series.
    Used by: `starwinds_analysis/pipelines/volume.py`
    """
    out = {}
    for name, profile in diagnostics.items():
        if not isinstance(profile, dict):
            continue
        pdata = {}
        for key, value in profile.items():
            if key == "shell_samples":
                continue
            arr = np.array(value)
            if arr.ndim == 0:
                try:
                    pdata[key] = arr.item()
                except Exception:
                    pdata[key] = str(value)
                continue
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                pdata[key] = arr
        if pdata:
            out[name] = pdata
    return out

def run_quicklook2d(
    smart_ds,
    *,
    body_radius_m: float,
    radii,
    slice_presets=("rho", "b_r", "u_r"),
    slice_style: str = "cross_quantiles",
    radius_modes=("binned",),
    radius_fields=None,
    radius_preset: str = "wind_raw",
    slice_ds=None,
    slice_grid: dict | None = None,
    orbit_radii=(),
    orbit_specs=(),
    orbit_planets=(),
    orbit_surface_specs=(),
    orbit_surface_planets=(),
    orbit_surface_modes=("pressure", "torque"),
    orbit_surface_n_longitudes: int = 199,
    orbit_plane: str = "xy",
    orbit_n_points: int = 180,
    n_polar: int = 24,
    n_azimuth: int = 48,
    method: str = "nearest",
    output_dir=None,
    prefix: str | None = None,
    input_file=None,
    band_radius_range=None,
    star_mass_kg: float | None = None,
):
    """
    End-to-end non-3D quicklook runner (figures + shell diagnostics + optional save).
    Used by: `test/test_quicklook2d.py`
    """
    try:
        n_radii = len(radii)
    except TypeError:
        n_radii = None
    log_pipeline_event(pipeline_log, 
        "quicklook.run.start",
        n_radii=n_radii,
        output_dir=None if output_dir is None else str(output_dir),
        prefix=_resolve_quicklook_prefix(prefix=prefix, input_file=input_file),
    )
    prepare_smartds(smart_ds, body_radius_m=body_radius_m)

    if orbit_planets:
        planet_specs = []
        for name in orbit_planets:
            if name not in SOLAR_SYSTEM_PLANETS:
                raise KeyError(
                    f"Unknown planet '{name}'. Available: {sorted(SOLAR_SYSTEM_PLANETS)}"
                )
            elem = SOLAR_SYSTEM_PLANETS[name]
            planet_specs.append(
                {
                    "label": str(name),
                    "semi_major_axis": float(elem.semi_major_axis_m / float(body_radius_m)),
                    "eccentricity": float(elem.eccentricity),
                    "n_points": int(orbit_n_points),
                    "sample": "eccentric_anomaly",
                    "plane": str(orbit_plane),
                }
            )
        orbit_specs = tuple(orbit_specs) + tuple(planet_specs)
    if orbit_surface_planets:
        planet_surface_specs = []
        for name in orbit_surface_planets:
            if name not in SOLAR_SYSTEM_PLANETS:
                raise KeyError(
                    f"Unknown planet '{name}'. Available: {sorted(SOLAR_SYSTEM_PLANETS)}"
                )
            elem = SOLAR_SYSTEM_PLANETS[name]
            planet_surface_specs.append(
                {
                    "label": str(name),
                    "semi_major_axis": float(elem.semi_major_axis_m / float(body_radius_m)),
                    "eccentricity": float(elem.eccentricity),
                    "n_points": int(orbit_n_points),
                    "sample": "eccentric_anomaly",
                    "plane": str(orbit_plane),
                }
            )
        orbit_surface_specs = tuple(orbit_surface_specs) + tuple(planet_surface_specs)

    if slice_ds is None and slice_presets:
        slice_input = smart_ds
        try:
            corners = getattr(smart_ds.raw, "corners", None)
            if corners is None or corners.ndim != 2 or corners.shape[1] != 4:
                slice_input = resample_structured_xz_slice(
                    smart_ds,
                    **dict(slice_grid or {}),
                )
        except Exception:
            slice_input = smart_ds
    else:
        slice_input = slice_ds if slice_ds is not None else smart_ds

    slice_figs = {}
    for preset in slice_presets:
        fig, _axes, _cbar = plot_slice_quicklook(
            slice_input, preset=preset, style=slice_style
        )
        slice_figs[preset] = fig
    log_pipeline_event(pipeline_log, "quicklook.run.slices", count=len(slice_figs))

    shell_fig, shell_axs, diagnostics = quicklook_shell_figure(
        smart_ds,
        radii,
        body_radius_m=body_radius_m,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        method=method,
        star_mass_kg=star_mass_kg,
    )

    radius_figs = {}
    for mode in radius_modes:
        fig, _axs = plot_radius_quicklook(
            smart_ds,
            fields=radius_fields,
            preset=None if radius_fields is not None else radius_preset,
            mode=mode,
        )
        radius_figs[mode] = fig
    log_pipeline_event(pipeline_log, "quicklook.run.radius", count=len(radius_figs))

    orbit_figs = {}
    orbit_results = {}
    for radius in orbit_radii:
        fig, _axs, result = orbit_local_comparison_figure(
            smart_ds,
            radius,
            body_radius_m=body_radius_m,
            n_points=orbit_n_points,
            plane=orbit_plane,
            method=method,
            shell_n_polar=n_polar,
            shell_n_azimuth=n_azimuth,
        )
        key = f"r{float(radius):g}_{orbit_plane}"
        orbit_figs[key] = fig
        orbit_results[key] = result
    for spec in orbit_specs:
        fig, _axs, result = orbit_local_comparison_figure(
            smart_ds,
            spec,
            body_radius_m=body_radius_m,
            n_points=orbit_n_points,
            plane=orbit_plane,
            method=method,
            shell_n_polar=n_polar,
            shell_n_azimuth=n_azimuth,
        )
        if isinstance(spec, dict):
            label = spec.get("label")
            if label:
                key = str(label)
            else:
                a = float(spec.get("semi_major_axis", spec.get("a", np.nan)))
                e = float(spec.get("eccentricity", 0.0))
                p = str(spec.get("plane", orbit_plane))
                key = f"a{a:g}_e{e:g}_{p}"
        else:
            key = f"orbit_{len(orbit_figs)}"
        orbit_figs[key] = fig
        orbit_results[key] = result
    for spec in orbit_surface_specs:
        if isinstance(spec, dict):
            spec_dict = dict(spec)
            label = spec_dict.pop("label", None)
        else:
            spec_dict = spec
            label = None
        if label is None:
            if isinstance(spec, dict):
                a = float(spec.get("semi_major_axis", spec.get("a", np.nan)))
                e = float(spec.get("eccentricity", 0.0))
                label = f"a{a:g}_e{e:g}_surface"
            else:
                label = f"r{float(spec):g}_surface"
        groups = orbit_results.setdefault(str(label), {})
        if "pressure" in orbit_surface_modes:
            fig, _axs, result = orbit_surface_pressure_figure(
                smart_ds,
                spec_dict,
                body_radius_m=body_radius_m,
                n_longitudes=orbit_surface_n_longitudes,
                method=method,
                star_mass_kg=star_mass_kg,
            )
            orbit_figs[f"{label}_surface_pressure"] = fig
            groups["surface_pressure"] = result
        if "torque" in orbit_surface_modes:
            fig, _axs, result = orbit_surface_torque_figure(
                smart_ds,
                spec_dict,
                body_radius_m=body_radius_m,
                n_longitudes=orbit_surface_n_longitudes,
                method=method,
            )
            orbit_figs[f"{label}_surface_torque"] = fig
            groups["surface_torque"] = result
    log_pipeline_event(pipeline_log, "quicklook.run.orbits", figures=len(orbit_figs), result_groups=len(orbit_results))

    out = {
        "slice_figures": slice_figs,
        "shell_figure": shell_fig,
        "shell_diagnostics": diagnostics,
        "radius_figures": radius_figs,
        "orbit_figures": orbit_figs,
        "orbit_results": orbit_results,
    }

    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        resolved_prefix = _resolve_quicklook_prefix(prefix=prefix, input_file=input_file)
        saved = []
        if shell_fig is not None:
            path = outdir / f"{resolved_prefix}.shells.png"
            shell_fig.savefig(path)
            saved.append(path)
            add_record("quicklook_shell_png %r", str(path))
        for group_name, figures in (("slices", slice_figs), ("radius", radius_figs), ("orbits", orbit_figs)):
            for key, fig in figures.items():
                path = outdir / f"{resolved_prefix}.{group_name}.{_slug_key(str(key))}.png"
                fig.savefig(path)
                saved.append(path)
                add_record("quicklook_png %r", str(path))
        add_record("shell_summary %r", summarize_shell_diagnostics(
            diagnostics,
            band_radius_range=band_radius_range,
            star_mass_kg=star_mass_kg,
            star_radius_m=body_radius_m,
        ))
        if orbit_results:
            add_record("orbit_results %r", orbit_results)
        out["saved"] = tuple(saved)
    log_pipeline_event(pipeline_log, "quicklook.run.done", saved=output_dir is not None)
    return out


def process_plt_file(file_path: str | Path) -> None:
    """
    Per-file volume pipeline step for `sw-pipe`.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`
    """
    path = Path(file_path)
    output_dir = path.parent / "volume"
    log.info("%s", path.name)
    smart_ds = SmartDs.from_file(path)
    if is_2d_input(smart_ds):
        log.info("skip file=%s reason=non_3d_input", path.name)
        add_record("volume_status %r", "skipped_non_3d")
        return
    out = run_quicklook2d(
        smart_ds,
        body_radius_m=DEFAULT_STAR_RADIUS_M,
        radii=DEFAULT_QUICKLOOK_RADII_R,
        slice_presets=(),
        radius_modes=(),
        orbit_radii=(),
        orbit_specs=(),
        orbit_planets=(),
        orbit_surface_specs=(),
        orbit_surface_planets=(),
        output_dir=output_dir,
        input_file=path.name,
    )
    diagnostics = out.get("shell_diagnostics", {})
    mass_loss_profile = diagnostics.get("mass_loss", {})
    radii = np.array(mass_loss_profile.get("radius [R]", []))
    mass_loss = np.array(mass_loss_profile.get("mass_loss [kg/s]", []))
    mass_loss_ref = np.nan
    radius_ref = np.nan
    if radii.ndim == 1 and mass_loss.ndim == 1 and radii.shape == mass_loss.shape and radii.size > 0:
        add_record(
            "mass_loss_profile_kg_s %r",
            [
                {"radius_R": float(r), "mass_loss_kg_s": float(m)}
                for r, m in zip(radii, mass_loss)
            ],
        )
        finite = np.isfinite(radii) & np.isfinite(mass_loss)
        if np.any(finite):
            idx = np.where(finite)[0][-1]
            r_ref = float(radii[idx])
            m_ref = float(mass_loss[idx])
            add_record("mass_loss_radius_R %r", r_ref)
            add_record("mass_loss_value_kg_s %r", m_ref)
            radius_ref = r_ref
            mass_loss_ref = m_ref
    total_torque_ref = np.nan
    total_torque = np.array(diagnostics.get("torque", {}).get("total_torque [Nm]", []))
    if radii.ndim == 1 and total_torque.ndim == 1 and radii.shape == total_torque.shape and radii.size > 0:
        finite = np.isfinite(radii) & np.isfinite(total_torque)
        if np.any(finite):
            idx = np.where(finite)[0][-1]
            add_record("total_torque_radius_R %r", float(radii[idx]))
            total_torque_ref = float(total_torque[idx])
            add_record("total_torque_value_nm %r", total_torque_ref)
    open_flux_ref = np.nan
    open_flux = np.array(diagnostics.get("open_flux", {}).get("open_flux [Wb]", []))
    if radii.ndim == 1 and open_flux.ndim == 1 and radii.shape == open_flux.shape and radii.size > 0:
        finite = np.isfinite(radii) & np.isfinite(open_flux)
        if np.any(finite):
            idx = np.where(finite)[0][-1]
            add_record("open_flux_radius_R %r", float(radii[idx]))
            open_flux_ref = float(open_flux[idx])
            add_record("open_flux_value_wb %r", open_flux_ref)
    energy_flux_ref = np.nan
    energy_flux = np.array(diagnostics.get("energy", {}).get("energy_flux [W]", []))
    if radii.ndim == 1 and energy_flux.ndim == 1 and radii.shape == energy_flux.shape and radii.size > 0:
        finite = np.isfinite(radii) & np.isfinite(energy_flux)
        if np.any(finite):
            idx = np.where(finite)[0][-1]
            add_record("energy_flux_radius_R %r", float(radii[idx]))
            energy_flux_ref = float(energy_flux[idx])
            add_record("energy_flux_value_w %r", energy_flux_ref)
    if np.isfinite(radius_ref):
        log.info(
            "result file=%s radius=%gR mass_loss_kg_s=%g total_torque_nm=%g open_flux_wb=%g energy_flux_w=%g",
            path.name,
            radius_ref,
            mass_loss_ref,
            total_torque_ref,
            open_flux_ref,
            energy_flux_ref,
        )
    add_record(
        "shell_summary %r",
        summarize_shell_diagnostics(diagnostics),
    )
    add_record(
        "shell_profiles %r",
        shell_profile_payload(diagnostics),
    )

    for fig in out.get("slice_figures", {}).values():
        plt.close(fig)
    shell_fig = out.get("shell_figure")
    if shell_fig is not None:
        plt.close(shell_fig)
    for fig in out.get("radius_figures", {}).values():
        plt.close(fig)
    for fig in out.get("orbit_figures", {}).values():
        plt.close(fig)
