"""Per-file 3D volume pipeline for `sw-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt

from starwinds_analysis.physics.fluxes import energy_flux_vs_radius
from starwinds_analysis.physics.fluxes import open_magnetic_flux_vs_radius
from starwinds_analysis.physics.mass_loss import mass_loss_vs_radius
from starwinds_analysis.physics.torque import torque_vs_radius
from starwinds_analysis.pipelines.orchestration_helpers import is_2d_input
from starwinds_analysis.pipelines.orchestration_helpers import prepare_smartds
from starwinds_analysis.pipelines.orchestration_helpers import resolve_output_prefix as _resolve_output_prefix
from starwinds_analysis.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
DEFAULT_STAR_RADIUS_M = 6.957e8
DEFAULT_QUICKLOOK_RADII_R = (2.0, 4.0, 8.0, 16.0)


def process_plt_file(file_path: str | Path) -> None:
    """Process one 3D `.plt` file into a shell-summary PNG and recorded diagnostics."""
    # Start: resolve input/output paths and log the file being processed.
    path = Path(file_path)
    output_dir = path.parent / "volume"
    prefix = _resolve_output_prefix(prefix=None, input_file=path.name)
    log.info("%s", path.name)

    # Start: load the dataset and reject non-3D inputs.
    smart_ds = SmartDs.from_file(path)
    if is_2d_input(smart_ds):
        log.info("skip file=%s reason=non_3d_input", path.name)
        add_record("volume_status %r", "skipped_non_3d")
        return

    # Start: prepare the dataset and create the output figure canvas.
    prepare_smartds(smart_ds, body_radius_m=DEFAULT_STAR_RADIUS_M)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    radii = DEFAULT_QUICKLOOK_RADII_R

    # Start: compute, plot, and record wind mass loss.
    mass_loss_radius_ref = float("nan")
    mass_loss_value_ref = float("nan")
    mass_loss = mass_loss_vs_radius(
        smart_ds,
        radii,
        body_radius_m=DEFAULT_STAR_RADIUS_M,
        n_polar=24,
        n_azimuth=48,
        method="nearest",
    )
    axes[0, 0].plot(mass_loss["height [R]"], mass_loss["mass_loss [kg/s]"], ".-", color="C0")
    axes[0, 0].set_title("Wind Mass Loss")
    axes[0, 0].set_ylabel("Mass flux [kg/s]")
    axes[0, 0].set_xlabel("Height [R]")
    axes[0, 0].grid(True, alpha=0.3)
    add_record("radius_R %r", mass_loss["radius [R]"])
    add_record("mass_loss_kg_s %r", mass_loss["mass_loss [kg/s]"])
    for radius_value, mass_loss_value in zip(mass_loss["radius [R]"], mass_loss["mass_loss [kg/s]"]):
        if isfinite(radius_value) and isfinite(mass_loss_value):
            mass_loss_radius_ref = float(radius_value)
            mass_loss_value_ref = float(mass_loss_value)
    if isfinite(mass_loss_radius_ref):
        add_record("mass_loss_radius_R %r", mass_loss_radius_ref)
        add_record("mass_loss_value_kg_s %r", mass_loss_value_ref)

    # Start: compute, plot, and record wind torque.
    torque_radius_ref = float("nan")
    torque_value_ref = float("nan")
    torque = torque_vs_radius(
        smart_ds,
        radii,
        body_radius_m=DEFAULT_STAR_RADIUS_M,
        n_polar=24,
        n_azimuth=48,
        method="nearest",
    )
    axes[0, 1].plot(torque["height [R]"], torque["total_torque [Nm]"], ".-", color="C1")
    axes[0, 1].set_title("Wind Torque")
    axes[0, 1].set_ylabel("Torque [Nm]")
    axes[0, 1].set_xlabel("Height [R]")
    axes[0, 1].grid(True, alpha=0.3)
    add_record("total_torque_nm %r", torque["total_torque [Nm]"])
    for radius_value, torque_value in zip(torque["radius [R]"], torque["total_torque [Nm]"]):
        if isfinite(radius_value) and isfinite(torque_value):
            torque_radius_ref = float(radius_value)
            torque_value_ref = float(torque_value)
    if isfinite(torque_radius_ref):
        add_record("total_torque_radius_R %r", torque_radius_ref)
        add_record("total_torque_value_nm %r", torque_value_ref)

    # Start: compute, plot, and record open magnetic flux.
    open_flux_radius_ref = float("nan")
    open_flux_value_ref = float("nan")
    open_flux = open_magnetic_flux_vs_radius(
        smart_ds,
        radii,
        body_radius_m=DEFAULT_STAR_RADIUS_M,
        n_polar=24,
        n_azimuth=48,
        method="nearest",
    )
    axes[1, 0].plot(open_flux["height [R]"], open_flux["open_flux [Wb]"], ".-", color="C2")
    axes[1, 0].set_title("Open Magnetic Flux")
    axes[1, 0].set_ylabel("Open flux [Wb]")
    axes[1, 0].set_xlabel("Height [R]")
    axes[1, 0].grid(True, alpha=0.3)
    add_record("open_flux_wb %r", open_flux["open_flux [Wb]"])
    for radius_value, open_flux_value in zip(open_flux["radius [R]"], open_flux["open_flux [Wb]"]):
        if isfinite(radius_value) and isfinite(open_flux_value):
            open_flux_radius_ref = float(radius_value)
            open_flux_value_ref = float(open_flux_value)
    if isfinite(open_flux_radius_ref):
        add_record("open_flux_radius_R %r", open_flux_radius_ref)
        add_record("open_flux_value_wb %r", open_flux_value_ref)

    # Start: compute, plot, and record energy flux.
    energy_flux_radius_ref = float("nan")
    energy_flux_value_ref = float("nan")
    energy_flux = energy_flux_vs_radius(
        smart_ds,
        radii,
        body_radius_m=DEFAULT_STAR_RADIUS_M,
        n_polar=24,
        n_azimuth=48,
        method="nearest",
    )
    axes[1, 1].plot(energy_flux["height [R]"], energy_flux["energy_flux [W]"], ".-", color="C3")
    axes[1, 1].set_title("Energy Flux")
    axes[1, 1].set_ylabel("Energy flux [W]")
    axes[1, 1].set_xlabel("Height [R]")
    axes[1, 1].grid(True, alpha=0.3)
    add_record("energy_flux_w %r", energy_flux["energy_flux [W]"])
    for radius_value, energy_flux_value in zip(energy_flux["radius [R]"], energy_flux["energy_flux [W]"]):
        if isfinite(radius_value) and isfinite(energy_flux_value):
            energy_flux_radius_ref = float(radius_value)
            energy_flux_value_ref = float(energy_flux_value)
    if isfinite(energy_flux_radius_ref):
        add_record("energy_flux_radius_R %r", energy_flux_radius_ref)
        add_record("energy_flux_value_w %r", energy_flux_value_ref)

    # Start: save the figure and record the output artifact.
    shell_png = output_dir / f"{prefix}.shells.png"
    fig.savefig(shell_png)
    plt.close(fig)

    # Start: record the final pipeline summary.
    add_record("volume_status %r", "processed")
    add_record("volume_shell_png %r", str(shell_png.relative_to(path.parent)))
    if isfinite(mass_loss_radius_ref):
        log.info(
            "result file=%s radius=%gR mass_loss_kg_s=%g total_torque_nm=%g open_flux_wb=%g energy_flux_w=%g",
            path.name,
            mass_loss_radius_ref,
            mass_loss_value_ref,
            torque_value_ref,
            open_flux_value_ref,
            energy_flux_value_ref,
        )
