"""Per-file 2D slice pipeline for `sw-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.pipelines.orchestration_helpers import resolve_quicklook_prefix as _resolve_quicklook_prefix
from starwinds_analysis.smart_ds import SmartDs
from starwinds_analysis.visualisation.slice import plot_xz_slice_tripcolor_with_cross_quantiles

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
SLICE_FORCE_3D_ENV = "STARWINDS_SLICE_FORCE_3D"
DEFAULT_STAR_RADIUS_M = 6.957e8


def _has_field(ds, name: str) -> bool:
    """Return True when a SmartDs field exists."""
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


def plot_slice_quicklook(ds, *, preset: str, style: str = "cross_quantiles"):
    """Plot one native-mesh slice for preset `rho`, `u`, or `b`."""
    if style != "cross_quantiles":
        raise KeyError("Only 'cross_quantiles' style is supported in slice pipeline")
    field_map = {
        "rho": "Rho [kg/m^3]",
        "u": "U [m/s]",
        "b": "B [T]",
    }
    if preset not in field_map:
        raise KeyError(f"Unknown preset '{preset}'")
    field = field_map[preset]
    if not _has_field(ds, field):
        raise KeyError(f"Missing required field '{field}' for preset '{preset}'")
    return plot_xz_slice_tripcolor_with_cross_quantiles(ds, var=field)


def _prepare_smartds(smart_ds, *, body_radius_m: float) -> None:
    """Best-effort graph setup for BATSRUS + spherical quantities."""
    if hasattr(smart_ds, "add_batsrus_graph"):
        try:
            smart_ds.add_batsrus_graph(body_radius_m=body_radius_m)
        except Exception:
            pass
    if hasattr(smart_ds, "add_spherical_graph"):
        try:
            smart_ds.add_spherical_graph(vectors=("B", "U"))
            return
        except Exception:
            pass
    if hasattr(smart_ds, "add_spherical_fields"):
        try:
            smart_ds.add_spherical_fields(vectors=("B", "U"))
        except Exception:
            pass


def _is_2d_input(smart_ds) -> bool:
    """Detect whether input behaves like a 2D slice dataset."""
    corners = getattr(smart_ds, "corners", None)
    if getattr(corners, "ndim", 0) == 2:
        if corners.shape[1] == 4:
            return True
        if corners.shape[1] >= 8:
            return False

    constant_axes = 0
    for name in ("X [R]", "Y [R]", "Z [R]"):
        try:
            values = np.ravel(smart_ds(name))
        except Exception:
            continue
        finite = np.isfinite(values)
        if not np.any(finite):
            constant_axes += 1
            continue
        finite_values = values[finite]
        vmin = np.min(finite_values)
        vmax = np.max(finite_values)
        scale = max(abs(vmin), abs(vmax), 1.0)
        if abs(vmax - vmin) <= (1.0e-12 + 1.0e-10 * scale):
            constant_axes += 1
    return constant_axes >= 1 or (constant_axes == 0 and not hasattr(smart_ds, "corners"))


def process_plt_file(file_path: str | Path, *, force_3d: bool | None = None) -> None:
    """Process one `.plt` file into 2D rho/u/b slice PNGs."""
    path = Path(file_path)
    output_dir = path.parent / "slice"
    log.info("%s", path.name)

    if force_3d is None:
        text = os.getenv(SLICE_FORCE_3D_ENV, "").strip().lower()
        force_3d = text in {"1", "true", "yes", "on"}

    smart_ds = SmartDs.from_file(path)
    if not _is_2d_input(smart_ds) and not force_3d:
        log.info("skip file=%s reason=3d_input", path.name)
        add_record("slice_status %r", "skipped_3d")
        return

    _prepare_smartds(smart_ds, body_radius_m=DEFAULT_STAR_RADIUS_M)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _resolve_quicklook_prefix(prefix=None, input_file=path.name)

    saved = {}
    for preset in ("rho", "u", "b"):
        try:
            fig, _axes, _cbar = plot_slice_quicklook(smart_ds, preset=preset, style="cross_quantiles")
        except Exception:
            continue
        out_path = output_dir / f"{prefix}.slices.{preset}.png"
        fig.savefig(out_path)
        plt.close(fig)
        saved[preset] = str(out_path.relative_to(path.parent))
        add_record("slice_%s_png %r", preset, saved[preset])

    add_record("slice_status %r", "processed")
    add_record("slice_figure_count %r", len(saved))
    add_record("slice_output_dir %r", str(output_dir.relative_to(path.parent)))
    log.info("result file=%s figures=%d", path.name, len(saved))
