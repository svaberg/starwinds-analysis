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
from starwinds_analysis.pipelines.orchestration_helpers import resolve_output_prefix as _resolve_output_prefix
from starwinds_analysis.smart_ds import prepare_smartds
from starwinds_analysis.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
DEFAULT_STAR_RADIUS_M = 6.957e8
DEFAULT_QUICKLOOK_RADII_R = (2.0, 4.0, 8.0, 16.0)


def process_plt_file(file_path: str | Path) -> None:
    """Process one 3D `.plt` file into a shell PNG and recorded diagnostics."""
    # Start: resolve input/output paths and log the file being processed.
    log.debug("Resolving volume pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "volume"
    prefix = _resolve_output_prefix(prefix=None, input_file=path.name)
    log.info("%s", path.name)
    log.info("Resolving volume pipeline paths complete.")

    # Start: load the dataset.
    log.debug("Loading volume dataset...")
    smart_ds = SmartDs.from_file(path)
    log.info("Loading volume dataset complete.")

    # Start: prepare the dataset and create the output figure canvas.
    log.debug("Preparing volume dataset and figure canvas...")
    prepare_smartds(smart_ds, body_radius_m=DEFAULT_STAR_RADIUS_M)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    radii = DEFAULT_QUICKLOOK_RADII_R
    log.info("Preparing volume dataset and figure canvas complete.")

    # Start: compute, plot, and record wind mass loss.
    log.debug("Computing wind mass loss...")
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
    log.info("Computing wind mass loss complete.")

    # Start: compute, plot, and record wind torque.
    log.debug("Computing wind torque...")
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
    log.info("Computing wind torque complete.")

    # Start: compute, plot, and record open magnetic flux.
    log.debug("Computing open magnetic flux...")
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
    log.info("Computing open magnetic flux complete.")

    # Start: compute, plot, and record energy flux.
    log.debug("Computing energy flux...")
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
    log.info("Computing energy flux complete.")

    # Start: save the figure and record the output artifact.
    log.debug("Saving volume figure...")
    shell_png = output_dir / f"{prefix}.shells.png"
    fig.savefig(shell_png)
    plt.close(fig)
    add_record("volume_shell_png %r", str(shell_png.relative_to(path.parent)))
    log.info("Saving volume figure complete.")
