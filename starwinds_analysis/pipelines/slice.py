"""THIS FILE contains the per-file 2D slice pipeline entrypoint for `sw-pipe`."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.pipelines.orchestration_helpers import resolve_quicklook_prefix as _resolve_quicklook_prefix
from starwinds_analysis.pipelines.quicklook_core import (
    DEFAULT_STAR_RADIUS_M,
    _has_field,
    orbit_local_comparison_figure,
    orbit_pressure_figure,
    orbit_surface_pressure_figure,
    orbit_surface_torque_figure,
    plot_radius_quicklook,
    plot_slice_quicklook,
    prepare_smartds_for_quicklook,
    quicklook_shell_figure,
    run_quicklook2d,
    save_quicklook2d_bundle,
)
from starwinds_analysis.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
SLICE_FORCE_3D_ENV = "STARWINDS_SLICE_FORCE_3D"


def _is_2d_input(smart_ds) -> bool:
    """
    Detect whether a dataset should be treated as 2D slice input.
    Used by: `starwinds_analysis/pipelines/slice.py`
    """
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
    """
    Process one `.plt` input with the lightweight 2D slice quicklook pipeline.
    Used by: `starwinds_analysis/pipelines/sw_pipe.py`, `test/test_quicklook2d.py`
    """
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

    prepare_smartds_for_quicklook(smart_ds, body_radius_m=DEFAULT_STAR_RADIUS_M)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _resolve_quicklook_prefix(prefix=None, input_file=path.name)

    saved = {}
    for preset in ("rho", "u", "b"):
        if not any(_has_field(smart_ds, name) for name in {
            "rho": ("Rho [kg/m^3]",),
            "u": ("U [m/s]",),
            "b": ("B [T]",),
        }[preset]):
            continue
        fig, _axes, _cbar = plot_slice_quicklook(smart_ds, preset=preset, style="cross_quantiles")
        out_path = output_dir / f"{prefix}.slices.{preset}.png"
        fig.savefig(out_path)
        plt.close(fig)
        saved[preset] = str(out_path.relative_to(path.parent))
        add_record("slice_%s_png %r", preset, saved[preset])

    add_record("slice_status %r", "processed")
    add_record("slice_figure_count %r", len(saved))
    add_record("slice_output_dir %r", str(output_dir.relative_to(path.parent)))
    log.info("result file=%s figures=%d", path.name, len(saved))
