"""Per-file 3D volume pipeline for `batwind-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from batwind.constants import DEFAULT_QUICKLOOK_RADII_R
from batwind.analysis.shells import integrate_shell_scalar
from batwind.analysis.shells import sample_spherical_shells_fibonacci
from batwind.data.field_names import DEFAULT_XYZ_NAMES
from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
LOS_GRID_N = 128


def process_plt_file(file_path: str | Path) -> None:
    """Process one 3D `.plt` file into a shell PNG and recorded diagnostics."""
    # Start: resolve input/output paths and log the file being processed.
    log.debug("Resolving volume pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "volume"
    prefix = output_prefix_from_input_file(path.name)
    log.info("%s", path.name)
    log.info("Resolving volume pipeline paths complete.")

    # Start: load the dataset.
    log.debug("Loading volume dataset...")
    smart_ds = SmartDs.from_file(path, batsrus=True, spherical=True)
    log.info("Loading volume dataset complete.")

    # Start: create the output figure canvas.
    log.debug("Preparing volume dataset and figure canvas...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    radii = DEFAULT_QUICKLOOK_RADII_R
    log.info("Preparing volume dataset and figure canvas complete.")

    # Start: sample shells once for all diagnostics.
    log.debug("Sampling shell grid once for all diagnostics...")
    energy_source = "E [J/m^3]"
    shared_source_fields = [
        "Rho [kg/m^3]",
        "U_x [m/s]",
        "U_y [m/s]",
        "U_z [m/s]",
        "B_x [T]",
        "B_y [T]",
        "B_z [T]",
    ]
    has_energy_source = energy_source in smart_ds
    if has_energy_source:
        shared_source_fields.append(energy_source)
    body_radius = float(smart_ds["RBODY [m]"])
    shells = sample_spherical_shells_fibonacci(
        smart_ds,
        radii,
        fields=shared_source_fields,
        n_points=24 * 48,
        method="octree",
        length_unit_to_m=body_radius,
    )
    mass_flux = np.array(shells["mass_flux [kg/m^2/s]"])
    shell_area = np.array(shells["dA [m^2]"])
    r_field = np.array(shells["R [R]"])
    shell_radii = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
    log.info("Sampling shell grid once for all diagnostics complete.")

    # Start: compute, plot, and record wind mass loss.
    log.debug("Computing wind mass loss...")
    mass_loss_radius_ref = float("nan")
    mass_loss_value_ref = float("nan")
    mass_loss_values, mass_loss_coverage = integrate_shell_scalar(mass_flux, shell_area)
    axes[0, 0].plot(shell_radii - 1.0, mass_loss_values, ".-", color="C0")
    axes[0, 0].set_title("Wind Mass Loss")
    axes[0, 0].set_ylabel("Mass flux [kg/s]")
    axes[0, 0].set_xlabel("Height [R]")
    axes[0, 0].grid(True, alpha=0.3)
    add_record("radius_R %r", shell_radii)
    add_record("mass_loss_kg_s %r", mass_loss_values)
    add_record("mass_loss_coverage %r", mass_loss_coverage)
    for radius_value, mass_loss_value in zip(shell_radii, mass_loss_values):
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
    magnetic_density = shells["magnetic_torque_density [N/m]"]
    dynamic_density = shells["dynamic_torque_density [N/m]"]
    magnetic_torque, torque_coverage_mag = integrate_shell_scalar(magnetic_density, shell_area)
    dynamic_torque, torque_coverage_dyn = integrate_shell_scalar(dynamic_density, shell_area)
    total_torque = magnetic_torque + dynamic_torque
    torque_coverage = np.minimum(torque_coverage_mag, torque_coverage_dyn)
    axes[0, 1].plot(shell_radii - 1.0, total_torque, ".-", color="C1")
    axes[0, 1].set_title("Wind Torque")
    axes[0, 1].set_ylabel("Torque [Nm]")
    axes[0, 1].set_xlabel("Height [R]")
    axes[0, 1].grid(True, alpha=0.3)
    add_record("magnetic_torque_nm %r", magnetic_torque)
    add_record("dynamic_torque_nm %r", dynamic_torque)
    add_record("total_torque_nm %r", total_torque)
    add_record("total_torque_coverage %r", torque_coverage)
    for radius_value, torque_value in zip(shell_radii, total_torque):
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
    b_r = shells["B_r [T]"]
    open_flux_values, open_flux_coverage = integrate_shell_scalar(np.abs(b_r), shell_area)
    axes[1, 0].plot(shell_radii - 1.0, open_flux_values, ".-", color="C2")
    axes[1, 0].set_title("Open Magnetic Flux")
    axes[1, 0].set_ylabel("Open flux [Wb]")
    axes[1, 0].set_xlabel("Height [R]")
    axes[1, 0].grid(True, alpha=0.3)
    add_record("open_flux_wb %r", open_flux_values)
    add_record("open_flux_coverage %r", open_flux_coverage)
    for radius_value, open_flux_value in zip(shell_radii, open_flux_values):
        if isfinite(radius_value) and isfinite(open_flux_value):
            open_flux_radius_ref = float(radius_value)
            open_flux_value_ref = float(open_flux_value)
    if isfinite(open_flux_radius_ref):
        add_record("open_flux_radius_R %r", open_flux_radius_ref)
        add_record("open_flux_value_wb %r", open_flux_value_ref)
    log.info("Computing open magnetic flux complete.")

    # Start: compute, plot, and record energy flux.
    log.debug("Computing energy flux...")
    if has_energy_source:
        energy_flux_radius_ref = float("nan")
        energy_flux_value_ref = float("nan")
        energy_flux_density = shells["energy_flux [W/m^2]"]
        energy_flux_values, energy_flux_coverage = integrate_shell_scalar(energy_flux_density, shell_area)
        axes[1, 1].plot(shell_radii - 1.0, energy_flux_values, ".-", color="C3")
        axes[1, 1].set_title("Energy Flux")
        axes[1, 1].set_ylabel("Energy flux [W]")
        axes[1, 1].set_xlabel("Height [R]")
        axes[1, 1].grid(True, alpha=0.3)
        add_record("energy_flux_w %r", energy_flux_values)
        add_record("energy_flux_coverage %r", energy_flux_coverage)
        for radius_value, energy_flux_value in zip(shell_radii, energy_flux_values):
            if isfinite(radius_value) and isfinite(energy_flux_value):
                energy_flux_radius_ref = float(radius_value)
                energy_flux_value_ref = float(energy_flux_value)
        if isfinite(energy_flux_radius_ref):
            add_record("energy_flux_radius_R %r", energy_flux_radius_ref)
            add_record("energy_flux_value_w %r", energy_flux_value_ref)
    else:
        axes[1, 1].set_title("Energy Flux")
        axes[1, 1].text(0.5, 0.5, "E [J/m^3] unavailable", ha="center", va="center")
        axes[1, 1].set_axis_off()
    log.info("Computing energy flux complete.")

    # Start: resample onto a regular 3D cube and make a fake LOS rho^2 image.
    log.debug("Computing fake LOS rho^2 image...")
    x = np.asarray(smart_ds["X [R]"], dtype=float)
    y = np.asarray(smart_ds["Y [R]"], dtype=float)
    z = np.asarray(smart_ds["Z [R]"], dtype=float)
    cube_n = LOS_GRID_N
    x_lin = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), cube_n)
    y_lin = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), cube_n)
    z_lin = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), cube_n)
    grid_x, grid_y, grid_z = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
    cube_points = np.stack([grid_x, grid_y, grid_z], axis=-1)
    los_cube = smart_ds.resample(
        cube_points,
        coordinate_fields=DEFAULT_XYZ_NAMES,
        fields=smart_ds.source_fields(("Rho [kg/m^3]",)),
        method="octree",
    )
    rho = np.asarray(los_cube["Rho [kg/m^3]"], dtype=float)
    rho_sq_los = np.nansum(rho**2, axis=2)
    positive = rho_sq_los[np.isfinite(rho_sq_los) & (rho_sq_los > 0.0)]
    los_fig, los_ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    los_norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive))) if positive.size else None
    image = los_ax.imshow(
        rho_sq_los.T,
        origin="lower",
        extent=(x_lin[0], x_lin[-1], y_lin[0], y_lin[-1]),
        cmap="magma",
        norm=los_norm,
        aspect="equal",
    )
    los_ax.set_title(r"Fake LOS $\rho^2$")
    los_ax.set_xlabel("X [R]")
    los_ax.set_ylabel("Y [R]")
    los_fig.colorbar(image, ax=los_ax, label=r"$\sum \rho^2$")
    los_png = output_dir / f"{prefix}.rho2_los.png"
    los_fig.savefig(los_png)
    plt.close(los_fig)
    add_record("volume_rho2_los_png %r", str(los_png.relative_to(path.parent)))
    add_record("volume_rho2_los_grid_n %r", cube_n)
    log.info("Computing fake LOS rho^2 image complete.")

    # Start: save the figure and record the output artifact.
    log.debug("Saving volume figure...")
    shell_png = output_dir / f"{prefix}.shells.png"
    fig.savefig(shell_png)
    plt.close(fig)
    add_record("volume_shell_png %r", str(shell_png.relative_to(path.parent)))
    log.info("Saving volume figure complete.")
