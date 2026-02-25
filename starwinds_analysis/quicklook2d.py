from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

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
from starwinds_analysis.analysis.orbits import (
    local_mass_loss_on_circular_orbit,
    local_mass_loss_on_elliptic_orbit,
    local_torque_on_circular_orbit,
    local_torque_on_elliptic_orbit,
)
from starwinds_analysis.analysis.shell_summary import summarize_shell_diagnostics_band
from starwinds_analysis.analysis.slices import resample_structured_xz_slice
from starwinds_analysis.analysis.torque import plot_torque_profile, torque_vs_radius
from starwinds_analysis.analysis.wind_scaling import open_wind_magnetisation_from_profiles
from starwinds_analysis.utils import triangles
from starwinds_analysis.visualisation.histograms import (
    plot_binned_vs_radius,
    plot_cumulative_hists,
    plot_vs_radius,
)
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

RADIAL_SUMMARY_PRESETS: dict[str, tuple[str, ...]] = {
    "wind_basic": (
        "Rho [kg/m^3]",
        "U [m/s]",
        "B [T]",
        "P [Pa]",
    ),
    "wind_raw": (
        "Rho [g/cm^3]",
        "U_x [km/s]",
        "B_x [Gauss]",
        "P [dyne/cm^2]",
    ),
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
    Radius/scatter/cumulative quicklook wrapper over `visualisation.histograms`.
    """
    if fields is None:
        if preset is None:
            raise ValueError("Provide either fields=... or preset=...")
        if preset not in RADIAL_SUMMARY_PRESETS:
            raise KeyError(f"Unknown radial preset '{preset}'")
        fields = tuple(f for f in RADIAL_SUMMARY_PRESETS[preset] if _has_field(ds, f))
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
    axs = np.asarray(axs).ravel()

    if mode == "binned":
        plot_binned_vs_radius(ds, axs, fields=fields, **plot_kwargs)
    elif mode == "scatter":
        plot_vs_radius(ds, axs, fields=fields, **plot_kwargs)
    elif mode in ("cdf", "cumulative"):
        plot_cumulative_hists(ds, axs, fields=fields, **plot_kwargs)
    else:
        raise KeyError("mode must be 'binned', 'scatter', or 'cdf'")

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
        ax2 = None
        if "axisymmetric_open_flux" in diagnostics:
            p = diagnostics["axisymmetric_open_flux"]
            h = np.asarray(p["height [R]"])
            frac = np.asarray(p["axisymmetric_open_flux_fraction [none]"])
            ax2 = axs[1, 0].twinx()
            ax2.plot(h, frac, ".-", color="C3", label="axisymmetric fraction")
            ax2.set_ylabel("Axisymmetric fraction [none]")
            ax2.set_ylim(0, 1.05)
        if "wind_scaling" in diagnostics:
            p = diagnostics["wind_scaling"]
            h = np.asarray(p["height [R]"])
            y = np.asarray(p["Upsilon_open [none]"])
            if ax2 is None:
                ax2 = axs[1, 0].twinx()
            ax2.plot(h, y, ".-", color="C4", label="Upsilon_open")
            ax2.set_ylabel("Axisymmetric fraction / Upsilon_open")
            finite = np.isfinite(y) & (y > 0)
            if np.any(finite):
                ax2.set_yscale("log")

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
    star_mass_kg: float | None = None,
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
    if star_mass_kg is not None:
        try:
            diagnostics["wind_scaling"] = open_wind_magnetisation_from_profiles(
                diagnostics,
                star_mass_kg=star_mass_kg,
                star_radius_m=body_radius_m,
            )
        except Exception:
            pass
    fig, axs = plot_shell_diagnostics(diagnostics, figsize=figsize)
    return fig, axs, diagnostics


def plot_orbit_mass_loss_comparison(ax, result):
    y = np.asarray(result["local_mass_loss [kg/s]"], dtype=float)
    phase = _orbit_phase(result, y.size)
    shell_interp = result.get("shell_mass_loss_interp [kg/s]")
    shell = float(result["shell_mass_loss [kg/s]"])
    mean = float(result["summary"]["mean"])

    ax.plot(phase, y, ",", color="C0", alpha=0.6, label="local estimate")
    if shell_interp is not None:
        ax.plot(
            phase,
            np.asarray(shell_interp, dtype=float),
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
    ax.set_title(_orbit_result_title("Mass Loss", result))
    return ax


def plot_orbit_torque_comparison(ax, result, *, show_components: bool = True):
    tot = np.asarray(result["local_total_torque [Nm]"], dtype=float)
    phase = _orbit_phase(result, tot.size)
    shell_interp = result.get("shell_total_torque_interp [Nm]")
    shell = float(result["shell_total_torque [Nm]"])
    mean = float(result["summary"]["mean"])

    ax.plot(phase, tot, ",", color="C0", alpha=0.6, label="local total")
    if show_components:
        ax.plot(phase, np.asarray(result["local_magnetic_torque [Nm]"]), ",", color="C1", alpha=0.35, label="local mag")
        ax.plot(phase, np.asarray(result["local_dynamic_torque [Nm]"]), ",", color="C2", alpha=0.35, label="local dyn")

    if shell_interp is not None:
        ax.plot(
            phase,
            np.asarray(shell_interp, dtype=float),
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
    ax.set_title(_orbit_result_title("Torque", result))
    return ax


def _orbit_phase(result, n):
    try:
        phase = np.asarray(result.get("orbit_samples", {}).get("phase [turns]"), dtype=float)
    except Exception:
        phase = np.array([], dtype=float)
    if phase.shape == (n,):
        return phase
    return np.arange(n, dtype=float) / max(1, n)


def _orbit_result_title(prefix, result):
    if "semi_major_axis [R]" in result and "eccentricity [none]" in result:
        return (
            f"{prefix} @ a={float(result['semi_major_axis [R]']):.3g} R, "
            f"e={float(result['eccentricity [none]']):.3g}"
        )
    return f"{prefix} @ r={float(result['radius [R]']):.3g} R"


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

    `radius` may be a float (circular orbit radius in `R`) or a dict describing a
    Kepler ellipse, e.g. `{"semi_major_axis": 10.0, "eccentricity": 0.2}`.
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
    """
    out = {}
    for name, profile in diagnostics.items():
        if not isinstance(profile, dict):
            continue
        pdata = {}
        for key, value in profile.items():
            if key == "shell_samples":
                continue
            arr = np.asarray(value)
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
            y = open_wind_magnetisation_from_profiles(
                diagnostics,
                star_mass_kg=star_mass_kg,
                star_radius_m=star_radius_m,
            )
            out["_wind_scaling"] = {
                "Upsilon_open [none]": _array_summary(y["Upsilon_open [none]"])
            }
        except Exception:
            pass
    return out


def flatten_shell_diagnostics_arrays(diagnostics):
    """
    Flatten shell diagnostic arrays for `np.savez`.
    """
    arrays = {}
    for name, profile in diagnostics.items():
        if not isinstance(profile, dict):
            continue
        for key, value in profile.items():
            if key == "shell_samples":
                continue
            arr = np.asarray(value)
            if arr.ndim == 0:
                continue
            flat_key = f"{name}__{_slug_key(key)}"
            arrays[flat_key] = arr
    return arrays


def summarize_orbit_results(orbit_results):
    """
    JSON-friendly summary of orbit local-vs-shell comparison results.
    """
    out = {}
    for orbit_key, groups in (orbit_results or {}).items():
        if not isinstance(groups, dict):
            continue
        orbit_out = {}
        for group_name, result in groups.items():
            if not isinstance(result, dict):
                continue
            group_out = {}
            for key, value in result.items():
                if key in {"orbit_samples", "shell_profile"}:
                    continue
                if key == "summary" and isinstance(value, dict):
                    summary_out = {}
                    for sk, sv in value.items():
                        arr = np.asarray(sv)
                        if arr.ndim == 0:
                            try:
                                summary_out[sk] = float(arr)
                            except Exception:
                                summary_out[sk] = str(sv)
                        else:
                            summary_out[sk] = np.asarray(sv, dtype=float).tolist()
                    group_out[key] = summary_out
                    continue
                arr = np.asarray(value)
                if arr.ndim == 0:
                    try:
                        group_out[key] = float(arr)
                    except Exception:
                        group_out[key] = str(value)
                else:
                    group_out[key] = _array_summary(arr)
            orbit_out[group_name] = group_out
        out[str(orbit_key)] = orbit_out
    return out


def flatten_orbit_results_arrays(orbit_results):
    """
    Flatten selected orbit result arrays for `np.savez`.
    """
    arrays = {}
    for orbit_key, groups in (orbit_results or {}).items():
        if not isinstance(groups, dict):
            continue
        for group_name, result in groups.items():
            if not isinstance(result, dict):
                continue
            for key, value in result.items():
                if key in {"orbit_samples", "shell_profile", "summary"}:
                    continue
                arr = np.asarray(value)
                if arr.ndim == 0:
                    continue
                flat_key = f"{_slug_key(orbit_key)}__{_slug_key(group_name)}__{_slug_key(key)}"
                arrays[flat_key] = arr
    return arrays


def save_shell_diagnostics_json(
    path,
    diagnostics,
    *,
    band_radius_range=None,
    star_mass_kg: float | None = None,
    star_radius_m: float | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize_shell_diagnostics(
        diagnostics,
        band_radius_range=band_radius_range,
        star_mass_kg=star_mass_kg,
        star_radius_m=star_radius_m,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def save_shell_diagnostics_npz(path, diagnostics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = flatten_shell_diagnostics_arrays(diagnostics)
    np.savez(path, **arrays)
    return path


def save_orbit_results_json(path, orbit_results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize_orbit_results(orbit_results)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def save_orbit_results_npz(path, orbit_results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = flatten_orbit_results_arrays(orbit_results)
    np.savez(path, **arrays)
    return path


def save_quicklook2d_bundle(
    output_dir,
    *,
    shell_fig=None,
    diagnostics=None,
    orbit_results=None,
    slice_figures=None,
    radius_figures=None,
    orbit_figures=None,
    prefix: str = "quicklook2d",
    band_radius_range=None,
    star_mass_kg: float | None = None,
    star_radius_m: float | None = None,
):
    """
    Save figures and shell summaries (JSON/NPZ) as a small quicklook bundle.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = {"figures": {}, "files": {}}

    if shell_fig is not None:
        p = outdir / f"{prefix}.shells.png"
        shell_fig.savefig(p)
        saved["figures"]["shells"] = p

    for group_name, figs in (("slices", slice_figures), ("radius", radius_figures), ("orbits", orbit_figures)):
        if not figs:
            continue
        for key, fig in figs.items():
            p = outdir / f"{prefix}.{group_name}.{_slug_key(str(key))}.png"
            fig.savefig(p)
            saved["figures"][f"{group_name}:{key}"] = p

    if diagnostics is not None:
        saved["files"]["shells_json"] = save_shell_diagnostics_json(
            outdir / f"{prefix}.shells.json",
            diagnostics,
            band_radius_range=band_radius_range,
            star_mass_kg=star_mass_kg,
            star_radius_m=star_radius_m,
        )
        saved["files"]["shells_npz"] = save_shell_diagnostics_npz(
            outdir / f"{prefix}.shells.npz", diagnostics
        )

    if orbit_results:
        saved["files"]["orbits_json"] = save_orbit_results_json(
            outdir / f"{prefix}.orbits.json", orbit_results
        )
        saved["files"]["orbits_npz"] = save_orbit_results_npz(
            outdir / f"{prefix}.orbits.npz", orbit_results
        )

    return saved


def _slug_key(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "item"


def _array_summary(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    return {
        "shape": list(arr.shape),
        "n": int(arr.size),
        "n_finite": int(np.count_nonzero(finite)),
        "min": float(np.nanmin(arr)) if np.any(finite) else np.nan,
        "max": float(np.nanmax(arr)) if np.any(finite) else np.nan,
        "mean": float(np.nanmean(arr)) if np.any(finite) else np.nan,
    }


def prepare_smartds_for_quicklook(smart_ds, *, body_radius_m: float | None = None):
    """
    Best-effort setup of common BATSRUS + spherical derived fields.
    """
    if hasattr(smart_ds, "add_batsrus_graph"):
        try:
            smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
        except Exception:
            pass
    if hasattr(smart_ds, "add_spherical_graph"):
        try:
            smart_ds.add_spherical_graph(vectors=("B", "U"))
            return smart_ds
        except Exception:
            pass
    if hasattr(smart_ds, "add_spherical_fields"):
        smart_ds.add_spherical_fields(vectors=("B", "U"))
    return smart_ds


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
    orbit_plane: str = "xy",
    orbit_n_points: int = 180,
    n_polar: int = 24,
    n_azimuth: int = 48,
    method: str = "nearest",
    output_dir=None,
    prefix: str = "quicklook2d",
    band_radius_range=None,
    star_mass_kg: float | None = None,
):
    """
    End-to-end non-3D quicklook runner (figures + shell diagnostics + optional save).
    """
    prepare_smartds_for_quicklook(smart_ds, body_radius_m=body_radius_m)

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

    out = {
        "slice_figures": slice_figs,
        "shell_figure": shell_fig,
        "shell_axes": shell_axs,
        "diagnostics": diagnostics,
        "radius_figures": radius_figs,
        "orbit_figures": orbit_figs,
        "orbit_results": orbit_results,
    }

    if output_dir is not None:
        out["saved"] = save_quicklook2d_bundle(
            output_dir,
            shell_fig=shell_fig,
            diagnostics=diagnostics,
            orbit_results=orbit_results,
            slice_figures=slice_figs,
            radius_figures=radius_figs,
            orbit_figures=orbit_figs,
            prefix=prefix,
            band_radius_range=band_radius_range,
            star_mass_kg=star_mass_kg,
            star_radius_m=body_radius_m,
        )
    return out


__all__ = [
    "RADIAL_SUMMARY_PRESETS",
    "SLICE_PRESETS",
    "SlicePreset",
    "compute_shell_diagnostics",
    "flatten_shell_diagnostics_arrays",
    "flatten_orbit_results_arrays",
    "plot_shell_diagnostics",
    "plot_radius_quicklook",
    "plot_slice_quicklook",
    "plot_orbit_mass_loss_comparison",
    "plot_orbit_torque_comparison",
    "prepare_smartds_for_quicklook",
    "quicklook_shell_figure",
    "orbit_local_comparison_figure",
    "run_quicklook2d",
    "save_quicklook2d_bundle",
    "save_orbit_results_json",
    "save_orbit_results_npz",
    "save_shell_diagnostics_json",
    "save_shell_diagnostics_npz",
    "summarize_orbit_results",
    "summarize_shell_diagnostics",
]
