"""Per-file 2D slice pipeline for `sw-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt

from starwinds_analysis.pipelines.orchestration_helpers import is_2d_input
from starwinds_analysis.pipelines.orchestration_helpers import prepare_smartds
from starwinds_analysis.pipelines.orchestration_helpers import resolve_output_prefix as _resolve_output_prefix
from starwinds_analysis.smart_ds import SmartDs
from starwinds_analysis.visualisation.slice import plot_xz_slice_tripcolor_with_cross_quantiles

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
SLICE_FORCE_3D_ENV = "STARWINDS_SLICE_FORCE_3D"
DEFAULT_STAR_RADIUS_M = 6.957e8


def process_plt_file(file_path: str | Path, *, force_3d: bool | None = None) -> None:
    """Process one `.plt` file into 2D rho/u/b slice PNGs."""
    # Start: resolve input/output paths and log the file being processed.
    path = Path(file_path)
    output_dir = path.parent / "slice"
    log.info("%s", path.name)

    # Start: decide whether this file should be handled by the slice pipeline.
    if force_3d is None:
        text = os.getenv(SLICE_FORCE_3D_ENV, "").strip().lower()
        force_3d = text in {"1", "true", "yes", "on"}

    # Start: load and prepare the dataset for direct plotting.
    smart_ds = SmartDs.from_file(path)
    if not is_2d_input(smart_ds) and not force_3d:
        log.info("skip file=%s reason=3d_input", path.name)
        add_record("slice_status %r", "skipped_3d")
        return

    prepare_smartds(smart_ds, body_radius_m=DEFAULT_STAR_RADIUS_M)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _resolve_output_prefix(prefix=None, input_file=path.name)

    # Start: make, save, and record the density slice.
    saved_count = 0
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(smart_ds, var="Rho [kg/m^3]")
    out_path = output_dir / f"{prefix}.slices.rho.png"
    fig.savefig(out_path)
    plt.close(fig)
    saved_count += 1
    add_record("slice_rho_png %r", str(out_path.relative_to(path.parent)))

    # Start: make, save, and record the speed slice.
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(smart_ds, var="U [m/s]")
    out_path = output_dir / f"{prefix}.slices.u.png"
    fig.savefig(out_path)
    plt.close(fig)
    saved_count += 1
    add_record("slice_u_png %r", str(out_path.relative_to(path.parent)))

    # Start: make, save, and record the magnetic-field slice.
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(smart_ds, var="B [T]")
    out_path = output_dir / f"{prefix}.slices.b.png"
    fig.savefig(out_path)
    plt.close(fig)
    saved_count += 1
    add_record("slice_b_png %r", str(out_path.relative_to(path.parent)))

    # Start: record the final pipeline summary.
    add_record("slice_status %r", "processed")
    add_record("slice_figure_count %r", saved_count)
    add_record("slice_output_dir %r", str(output_dir.relative_to(path.parent)))
    log.info("result file=%s figures=%d", path.name, saved_count)
