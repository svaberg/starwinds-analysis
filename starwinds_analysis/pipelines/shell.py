"""Per-file shell pipeline for `sw-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.physics.constants import MU0
from starwinds_analysis.pipelines.orchestration_helpers import resolve_output_prefix as _resolve_output_prefix
from starwinds_analysis.smart_ds import prepare_smartds
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

    # Start: load the dataset and prepare the native shell grid.
    log.debug("Loading shell dataset and preparing native shell grid...")
    smart_ds = SmartDs.from_file(path)
    prepare_smartds(smart_ds, body_radius_m=DEFAULT_STAR_RADIUS_M)
    output_dir.mkdir(parents=True, exist_ok=True)

    r_all = np.ravel(smart_ds("R [R]"))
    lon_all = np.ravel(smart_ds("Lon [deg]"))
    lat_all = np.ravel(smart_ds("Lat [deg]"))
    lon_rad = np.deg2rad(lon_all)
    lat_rad = np.deg2rad(lat_all)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    shell_radii_r = [float(radius_r) for radius_r in np.unique(np.round(r_all, 10))]
    lon_levels = np.unique(np.round(lon_all, 10))
    if lon_levels.size > 1 and np.isclose(lon_levels[0], 0.0) and np.isclose(lon_levels[-1], 360.0):
        lon_levels = lon_levels[:-1]
    lat_levels = np.unique(np.round(lat_all, 10))

    lon_edges = np.empty(lon_levels.size + 1, dtype=float)
    lon_edges[1:-1] = 0.5 * (lon_levels[:-1] + lon_levels[1:])
    lon_edges[0] = lon_levels[0] - 0.5 * (lon_levels[1] - lon_levels[0])
    lon_edges[-1] = lon_levels[-1] + 0.5 * (lon_levels[-1] - lon_levels[-2])

    lat_edges = np.empty(lat_levels.size + 1, dtype=float)
    lat_edges[1:-1] = 0.5 * (lat_levels[:-1] + lat_levels[1:])
    lat_edges[0] = -90.0
    lat_edges[-1] = 90.0

    solid_angle = (
        np.sin(np.deg2rad(lat_edges[1:]))[:, None] - np.sin(np.deg2rad(lat_edges[:-1]))[:, None]
    ) * np.deg2rad(np.diff(lon_edges))[None, :]

    seam_mask = lon_all < lon_edges[-1]
    plot_mask = np.isclose(r_all, shell_radii_r[-1], rtol=0.0, atol=1.0e-10) & seam_mask
    plot_lat_index = np.searchsorted(lat_levels, np.round(lat_all[plot_mask], 10))
    plot_lon_index = np.searchsorted(lon_levels, np.round(lon_all[plot_mask], 10))
    shell_masks = []
    shell_areas_m2 = []

    for radius_r in shell_radii_r:
        shell_mask = np.isclose(r_all, radius_r, rtol=0.0, atol=1.0e-10) & seam_mask
        lat_index = np.searchsorted(lat_levels, np.round(lat_all[shell_mask], 10))
        lon_index = np.searchsorted(lon_levels, np.round(lon_all[shell_mask], 10))
        shell_masks.append(shell_mask)
        shell_areas_m2.append((float(radius_r) * DEFAULT_STAR_RADIUS_M) ** 2 * solid_angle[lat_index, lon_index])

    height_r = [radius_r - 1.0 for radius_r in shell_radii_r]
    log.info("Loading shell dataset and preparing native shell grid complete.")

    # Start: create the figure canvases used by the shell tasks.
    log.debug("Preparing shell plotting canvases...")
    map_fig, map_axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    profile_fig, profile_axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for axis in map_axes.ravel():
        axis.set_xlabel("Longitude [deg]")
        axis.set_ylabel("Latitude [deg]")

    for axis in profile_axes.ravel():
        axis.set_xlabel("Height [R]")
        axis.grid(True, alpha=0.25)

    log.info("Preparing shell plotting canvases complete.")

    # Start: compute, plot, and record shell wind mass flux.
    log.debug("Computing shell wind mass flux...")
    rho = np.ravel(smart_ds("Rho [kg/m^3]"))
    u_x = np.ravel(smart_ds("U_x [m/s]"))
    u_y = np.ravel(smart_ds("U_y [m/s]"))
    u_z = np.ravel(smart_ds("U_z [m/s]"))
    u_r = u_x * cos_lat * cos_lon + u_y * cos_lat * sin_lon + u_z * sin_lat
    mass_flux = rho * u_r
    mass_loss_kg_s = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        mass_loss_kg_s.append(float(np.sum(mass_flux[shell_mask] * area_m2)))

    mass_flux_map = np.full((lat_edges.size - 1, lon_edges.size - 1), np.nan)
    mass_flux_map[plot_lat_index, plot_lon_index] = mass_flux[plot_mask]
    image = map_axes[0, 0].pcolormesh(lon_edges, lat_edges, mass_flux_map, shading="flat", cmap="viridis")
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
    b_x = np.ravel(smart_ds("B_x [T]"))
    b_y = np.ravel(smart_ds("B_y [T]"))
    b_z = np.ravel(smart_ds("B_z [T]"))
    b_r = b_x * cos_lat * cos_lon + b_y * cos_lat * sin_lon + b_z * sin_lat
    u_phi = -u_x * sin_lon + u_y * cos_lon
    b_phi = -b_x * sin_lon + b_y * cos_lon
    varpi_m = r_all * DEFAULT_STAR_RADIUS_M * cos_lat
    torque_density = varpi_m * (rho * u_phi * u_r - (b_phi * b_r / MU0))
    total_torque_nm = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        total_torque_nm.append(float(np.sum(torque_density[shell_mask] * area_m2)))

    torque_map = np.full((lat_edges.size - 1, lon_edges.size - 1), np.nan)
    torque_map[plot_lat_index, plot_lon_index] = torque_density[plot_mask]
    image = map_axes[0, 1].pcolormesh(lon_edges, lat_edges, torque_map, shading="flat", cmap="cividis")
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
    energy_density = np.ravel(smart_ds("E [J/m^3]"))
    energy_flux = energy_density * u_r
    energy_flow_w = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        energy_flow_w.append(float(np.sum(energy_flux[shell_mask] * area_m2)))

    energy_map = np.full((lat_edges.size - 1, lon_edges.size - 1), np.nan)
    energy_map[plot_lat_index, plot_lon_index] = energy_flux[plot_mask]
    image = map_axes[1, 0].pcolormesh(lon_edges, lat_edges, energy_map, shading="flat", cmap="plasma")
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
    open_flux_density = np.abs(b_r)
    open_flux_wb = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        open_flux_wb.append(float(np.sum(open_flux_density[shell_mask] * area_m2)))

    open_flux_map = np.full((lat_edges.size - 1, lon_edges.size - 1), np.nan)
    open_flux_map[plot_lat_index, plot_lon_index] = open_flux_density[plot_mask]
    image = map_axes[1, 1].pcolormesh(lon_edges, lat_edges, open_flux_map, shading="flat", cmap="magma")
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
