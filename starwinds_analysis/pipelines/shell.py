"""Per-file shell pipeline for `sw-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.algorithms.sphere_sampling import PolarAzimuthalGrid
from starwinds_analysis.pipelines.orchestration_helpers import prepare_smartds
from starwinds_analysis.pipelines.orchestration_helpers import resolve_output_prefix as _resolve_output_prefix
from starwinds_analysis.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
DEFAULT_STAR_RADIUS_M = 6.957e8


def process_plt_file(file_path: str | Path) -> None:
    """Process one shell-like `.plt` file into shell maps, profiles, and recorded diagnostics."""
    # Start: resolve input/output paths and log the file being processed.
    log.debug("Resolving shell pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "shell"
    prefix = _resolve_output_prefix(prefix=None, input_file=path.name)
    log.info("%s", path.name)
    log.info("Resolving shell pipeline paths complete.")

    # Start: load the dataset and reconstruct shell geometry from native theta/phi points.
    log.debug("Loading shell dataset and reconstructing shell geometry...")
    smart_ds = SmartDs.from_file(path)
    prepare_smartds(smart_ds, body_radius_m=DEFAULT_STAR_RADIUS_M)
    output_dir.mkdir(parents=True, exist_ok=True)

    r_all = np.ravel(smart_ds("R [R]"))
    theta_all = np.ravel(smart_ds("theta [rad]"))
    phi_all = np.ravel(smart_ds("phi [rad]"))
    finite = np.isfinite(r_all) & np.isfinite(theta_all) & np.isfinite(phi_all)

    shell_masks = []
    shell_areas_m2 = []
    shell_radii_r = []
    plot_mask = None
    plot_theta_edges = None
    plot_phi_edges = None
    plot_theta_index = None
    plot_phi_index = None

    for radius_r in np.unique(np.round(r_all[finite], 10)):
        shell_mask = finite & np.isclose(r_all, float(radius_r), rtol=0.0, atol=1.0e-10)
        if np.count_nonzero(shell_mask) < 8:
            continue

        theta_levels = np.unique(np.round(theta_all[shell_mask], 10))
        phi_levels = np.unique(np.round(phi_all[shell_mask], 10))
        if theta_levels.size < 2 or phi_levels.size < 2:
            continue
        if theta_levels.size * phi_levels.size != np.count_nonzero(shell_mask):
            continue

        theta_edges = np.empty(theta_levels.size + 1, dtype=float)
        theta_edges[1:-1] = 0.5 * (theta_levels[:-1] + theta_levels[1:])
        theta_edges[0] = max(0.0, theta_levels[0] - 0.5 * (theta_levels[1] - theta_levels[0]))
        theta_edges[-1] = min(math.pi, theta_levels[-1] + 0.5 * (theta_levels[-1] - theta_levels[-2]))

        phi_edges = np.empty(phi_levels.size + 1, dtype=float)
        phi_edges[1:-1] = 0.5 * (phi_levels[:-1] + phi_levels[1:])
        phi_edges[0] = phi_levels[0] - 0.5 * (phi_levels[1] - phi_levels[0])
        phi_edges[-1] = phi_levels[-1] + 0.5 * (phi_levels[-1] - phi_levels[-2])

        theta_index = np.searchsorted(theta_levels, np.round(theta_all[shell_mask], 10))
        phi_index = np.searchsorted(phi_levels, np.round(phi_all[shell_mask], 10))
        radius_m = float(radius_r) * DEFAULT_STAR_RADIUS_M
        area_grid_m2 = PolarAzimuthalGrid(theta_edges, phi_edges).cell_area(radius_m)

        shell_masks.append(shell_mask)
        shell_areas_m2.append(area_grid_m2[theta_index, phi_index])
        shell_radii_r.append(float(radius_r))
        plot_mask = shell_mask
        plot_theta_edges = theta_edges
        plot_phi_edges = phi_edges
        plot_theta_index = theta_index
        plot_phi_index = phi_index

    if not shell_masks:
        log.info("skip file=%s reason=not_shell_like", path.name)
        add_record("shell_status %r", "skipped_non_shell")
        return

    height_r = [radius_r - 1.0 for radius_r in shell_radii_r]
    log.info("Loading shell dataset and reconstructing shell geometry complete.")

    # Start: create the figure canvases used by the shell tasks.
    log.debug("Preparing shell plotting canvases...")
    map_fig, map_axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    profile_fig, profile_axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for axis in map_axes.ravel():
        axis.set_xlabel("Azimuth [rad]")
        axis.set_ylabel("Polar [rad]")

    for axis in profile_axes.ravel():
        axis.set_xlabel("Height [R]")
        axis.grid(True, alpha=0.25)

    log.info("Preparing shell plotting canvases complete.")

    # Start: compute, plot, and record shell wind mass flux.
    log.debug("Computing shell wind mass flux...")
    mass_flux = np.ravel(smart_ds("mass_flux [kg/m^2/s]"))
    mass_loss_kg_s = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        mass_loss_kg_s.append(float(np.sum(mass_flux[shell_mask] * area_m2)))

    mass_flux_map = np.full((plot_theta_edges.size - 1, plot_phi_edges.size - 1), np.nan)
    mass_flux_map[plot_theta_index, plot_phi_index] = mass_flux[plot_mask]
    image = map_axes[0, 0].pcolormesh(plot_phi_edges, plot_theta_edges, mass_flux_map, shading="flat", cmap="viridis")
    map_axes[0, 0].set_title("Wind Mass Flux")
    map_fig.colorbar(image, ax=map_axes[0, 0], label="Mass flux [kg/m^2/s]")
    profile_axes[0, 0].plot(height_r, mass_loss_kg_s, ".-", color="C0")
    profile_axes[0, 0].set_title("Wind Mass Loss")
    profile_axes[0, 0].set_ylabel("Mass loss [kg/s]")
    add_record("shell_radius_R %r", shell_radii_r)
    add_record("shell_mass_loss_kg_s %r", mass_loss_kg_s)
    add_record("shell_mass_loss_value_kg_s %r", mass_loss_kg_s[-1])
    log.info("Computing shell wind mass flux complete.")

    # Start: compute, plot, and record shell angular momentum flux.
    log.debug("Computing shell angular momentum flux...")
    torque_density = np.ravel(smart_ds("total_torque_density [N/m]"))
    total_torque_nm = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        total_torque_nm.append(float(np.sum(torque_density[shell_mask] * area_m2)))

    torque_map = np.full((plot_theta_edges.size - 1, plot_phi_edges.size - 1), np.nan)
    torque_map[plot_theta_index, plot_phi_index] = torque_density[plot_mask]
    image = map_axes[0, 1].pcolormesh(plot_phi_edges, plot_theta_edges, torque_map, shading="flat", cmap="cividis")
    map_axes[0, 1].set_title("Angular Momentum Flux")
    map_fig.colorbar(image, ax=map_axes[0, 1], label="Torque density [N/m]")
    profile_axes[0, 1].plot(height_r, total_torque_nm, ".-", color="C1")
    profile_axes[0, 1].set_title("Wind Torque")
    profile_axes[0, 1].set_ylabel("Torque [Nm]")
    add_record("shell_total_torque_nm %r", total_torque_nm)
    add_record("shell_total_torque_value_nm %r", total_torque_nm[-1])
    log.info("Computing shell angular momentum flux complete.")

    # Start: compute, plot, and record shell energy flux.
    log.debug("Computing shell energy flux...")
    energy_flux = np.ravel(smart_ds("energy_flux [W/m^2]"))
    energy_flow_w = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        energy_flow_w.append(float(np.sum(energy_flux[shell_mask] * area_m2)))

    energy_map = np.full((plot_theta_edges.size - 1, plot_phi_edges.size - 1), np.nan)
    energy_map[plot_theta_index, plot_phi_index] = energy_flux[plot_mask]
    image = map_axes[1, 0].pcolormesh(plot_phi_edges, plot_theta_edges, energy_map, shading="flat", cmap="plasma")
    map_axes[1, 0].set_title("Energy Flux")
    map_fig.colorbar(image, ax=map_axes[1, 0], label="Energy flux [W/m^2]")
    profile_axes[1, 0].plot(height_r, energy_flow_w, ".-", color="C2")
    profile_axes[1, 0].set_title("Shell Energy Flow")
    profile_axes[1, 0].set_ylabel("Energy flow [W]")
    add_record("shell_energy_flow_w %r", energy_flow_w)
    add_record("shell_energy_flow_value_w %r", energy_flow_w[-1])
    log.info("Computing shell energy flux complete.")

    # Start: compute, plot, and record shell open magnetic flux.
    log.debug("Computing shell open magnetic flux...")
    open_flux_density = np.abs(np.ravel(smart_ds("B_r [T]")))
    open_flux_wb = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        open_flux_wb.append(float(np.sum(open_flux_density[shell_mask] * area_m2)))

    open_flux_map = np.full((plot_theta_edges.size - 1, plot_phi_edges.size - 1), np.nan)
    open_flux_map[plot_theta_index, plot_phi_index] = open_flux_density[plot_mask]
    image = map_axes[1, 1].pcolormesh(plot_phi_edges, plot_theta_edges, open_flux_map, shading="flat", cmap="magma")
    map_axes[1, 1].set_title("Open Magnetic Flux Density")
    map_fig.colorbar(image, ax=map_axes[1, 1], label="|B_r| [T]")
    profile_axes[1, 1].plot(height_r, open_flux_wb, ".-", color="C3")
    profile_axes[1, 1].set_title("Open Magnetic Flux")
    profile_axes[1, 1].set_ylabel("Open flux [Wb]")
    add_record("shell_open_flux_wb %r", open_flux_wb)
    add_record("shell_open_flux_value_wb %r", open_flux_wb[-1])
    log.info("Computing shell open magnetic flux complete.")

    # Start: save the figures and record the output artifacts.
    log.debug("Saving shell figures...")
    shell_maps_png = output_dir / f"{prefix}.shell_maps.png"
    shell_profiles_png = output_dir / f"{prefix}.shell_profiles.png"
    map_fig.savefig(shell_maps_png)
    profile_fig.savefig(shell_profiles_png)
    plt.close(map_fig)
    plt.close(profile_fig)
    add_record("shell_maps_png %r", str(shell_maps_png.relative_to(path.parent)))
    add_record("shell_profiles_png %r", str(shell_profiles_png.relative_to(path.parent)))
    log.info("Saving shell figures complete.")

    # Start: record the final shell pipeline summary.
    log.debug("Recording shell pipeline summary...")
    add_record("shell_status %r", "processed")
    add_record("shell_count %r", len(shell_masks))
    log.info("Recording shell pipeline summary complete.")
    log.info(
        "result file=%s shells=%d radius=%gR mass_loss_kg_s=%g total_torque_nm=%g energy_flow_w=%g open_flux_wb=%g",
        path.name,
        len(shell_masks),
        shell_radii_r[-1],
        mass_loss_kg_s[-1],
        total_torque_nm[-1],
        energy_flow_w[-1],
        open_flux_wb[-1],
    )
