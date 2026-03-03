"""Per-file 2D slice pipeline for `sw-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from starwinds_analysis.pipelines.orchestration_helpers import resolve_output_prefix as _resolve_output_prefix
from starwinds_analysis.smart_ds import prepare_smartds
from starwinds_analysis.smart_ds import SmartDs
from starwinds_analysis.visualisation.slice import plot_xz_slice_tripcolor_with_cross_quantiles

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
DEFAULT_STAR_RADIUS_M = 6.957e8


def process_plt_file(file_path: str | Path) -> None:
    """Process one `.plt` file into 2D rho/u/b slice PNGs."""
    # Start: resolve input/output paths and log the file being processed.
    log.debug("Resolving slice pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "slice"
    log.info("%s", path.name)
    log.info("Resolving slice pipeline paths complete.")

    # Start: load and prepare the dataset for direct plotting.
    log.debug("Loading and preparing slice dataset...")
    smart_ds = SmartDs.from_file(path)
    prepare_smartds(smart_ds, body_radius_m=DEFAULT_STAR_RADIUS_M)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _resolve_output_prefix(prefix=None, input_file=path.name)
    log.info("Loading and preparing slice dataset complete.")

    # Start: make, save, and record the density slice.
    log.debug("Computing density slice...")
    saved_count = 0
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(smart_ds, var="Rho [kg/m^3]")
    out_path = output_dir / f"{prefix}.slices.rho.png"
    fig.savefig(out_path)
    plt.close(fig)
    saved_count += 1
    add_record("slice_rho_png %r", str(out_path.relative_to(path.parent)))
    log.info("Computing density slice complete.")

    # Start: make, save, and record the speed slice.
    log.debug("Computing speed slice...")
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(smart_ds, var="U [m/s]")
    out_path = output_dir / f"{prefix}.slices.u.png"
    fig.savefig(out_path)
    plt.close(fig)
    saved_count += 1
    add_record("slice_u_png %r", str(out_path.relative_to(path.parent)))
    log.info("Computing speed slice complete.")

    # Start: make, save, and record the magnetic-field slice.
    log.debug("Computing magnetic-field slice...")
    fig, _axes, _cbar = plot_xz_slice_tripcolor_with_cross_quantiles(smart_ds, var="B [T]")
    out_path = output_dir / f"{prefix}.slices.b.png"
    fig.savefig(out_path)
    plt.close(fig)
    saved_count += 1
    add_record("slice_b_png %r", str(out_path.relative_to(path.parent)))
    log.info("Computing magnetic-field slice complete.")

    # Start: record the final pipeline summary.
    log.debug("Recording slice pipeline summary...")
    add_record("slice_status %r", "processed")
    add_record("slice_figure_count %r", saved_count)
    add_record("slice_output_dir %r", str(output_dir.relative_to(path.parent)))
    log.info("Recording slice pipeline summary complete.")
    log.info("result file=%s figures=%d", path.name, saved_count)
