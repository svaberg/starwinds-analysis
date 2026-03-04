"""Per-file shell pipeline for `sw-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.constants import MU0
from starwinds_analysis.pipelines.utils import output_prefix_from_input_file
from starwinds_analysis.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug


def shell_spherical_components(
    vector,
    *,
    lon_deg,
    lat_deg,
):
    """Project a stacked Cartesian shell vector onto radial/polar/azimuth directions."""
    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    x_component = vector[0]
    y_component = vector[1]
    z_component = vector[2]
    radial = x_component * cos_lat * cos_lon + y_component * cos_lat * sin_lon + z_component * sin_lat
    polar = -x_component * sin_lat * cos_lon - y_component * sin_lat * sin_lon + z_component * cos_lat
    azimuth = -x_component * sin_lon + y_component * cos_lon
    return radial, polar, azimuth


def shell_cell_values(
    values,
    *,
    shell_mask,
    lon_deg,
    lat_deg,
    lon_nodes,
    lat_nodes,
):
    """Convert nodal shell values on corner nodes into explicit cell values."""
    node_values = np.full((lat_nodes.size, lon_nodes.size), np.nan)
    lat_index = np.searchsorted(lat_nodes, np.round(lat_deg[shell_mask], 10))
    lon_index = np.searchsorted(lon_nodes, np.round(lon_deg[shell_mask], 10))
    node_values[lat_index, lon_index] = values[shell_mask]
    return 0.25 * (
        node_values[:-1, :-1]
        + node_values[1:, :-1]
        + node_values[:-1, 1:]
        + node_values[1:, 1:]
    )


def process_plt_file(file_path: str | Path) -> None:
    """Process one shell-like `.plt` file into shell maps, profiles, and recorded diagnostics."""
    # Start: resolve input/output paths and log the file being processed.
    log.debug("Resolving shell pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "shell"
    prefix = output_prefix_from_input_file(path.name)
    log.info("%s", path.name)
    log.info("Resolving shell pipeline paths complete.")

    # Start: load the dataset and prepare the native shell grid.
    log.debug("Loading shell dataset and preparing native shell grid...")
    smart_ds = SmartDs.from_file(path)
    smart_ds.prepare()
    output_dir.mkdir(parents=True, exist_ok=True)

    r_all = np.ravel(smart_ds("R [R]"))
    star_radius_m = float(smart_ds("star_radius [m]"))
    lon_all = np.ravel(smart_ds("Lon [deg]"))
    lat_all = np.ravel(smart_ds("Lat [deg]"))
    shell_radii_r = [float(radius_r) for radius_r in np.unique(np.round(r_all, 10))]
    lon_nodes = np.unique(np.round(lon_all, 10))
    lat_nodes = np.unique(np.round(lat_all, 10))

    solid_angle = (
        np.sin(np.deg2rad(lat_nodes[1:]))[:, None] - np.sin(np.deg2rad(lat_nodes[:-1]))[:, None]
    ) * np.deg2rad(np.diff(lon_nodes))[None, :]

    shell_masks = []
    shell_areas_m2 = []

    for radius_r in shell_radii_r:
        shell_mask = np.isclose(r_all, radius_r, rtol=0.0, atol=1.0e-10)
        shell_masks.append(shell_mask)
        shell_areas_m2.append((float(radius_r) * star_radius_m) ** 2 * solid_angle)

    height_r = [radius_r - 1.0 for radius_r in shell_radii_r]
    log.info("Loading shell dataset and preparing native shell grid complete.")

    # Start: compute, plot, and record shell wind mass flux.
    log.debug("Computing shell wind mass flux...")
    rho = np.ravel(smart_ds("Rho [kg/m^3]"))
    u_vector = np.stack(
        (
            np.ravel(smart_ds("U_x [m/s]")),
            np.ravel(smart_ds("U_y [m/s]")),
            np.ravel(smart_ds("U_z [m/s]")),
        ),
        axis=0,
    )
    u_r, _u_polar, u_phi = shell_spherical_components(
        u_vector,
        lon_deg=lon_all,
        lat_deg=lat_all,
    )
    mass_flux = rho * u_r
    mass_loss_kg_s = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        mass_flux_cells = shell_cell_values(
            mass_flux,
            shell_mask=shell_mask,
            lon_deg=lon_all,
            lat_deg=lat_all,
            lon_nodes=lon_nodes,
            lat_nodes=lat_nodes,
        )
        mass_loss_kg_s.append(float(np.sum(mass_flux_cells * area_m2)))

    mass_flux_map = shell_cell_values(
        mass_flux,
        shell_mask=shell_masks[-1],
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    mass_flux_fig, mass_flux_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = mass_flux_ax.pcolormesh(lon_nodes, lat_nodes, mass_flux_map, shading="flat", cmap="viridis")
    mass_flux_ax.set_xlabel("Longitude [deg]")
    mass_flux_ax.set_ylabel("Latitude [deg]")
    mass_flux_ax.set_title("Wind Mass Flux")
    mass_flux_fig.colorbar(image, ax=mass_flux_ax, label="Mass flux [kg/m^2/s]")
    mass_flux_png = output_dir / f"{prefix}.mass_flux_map.png"
    mass_flux_fig.savefig(mass_flux_png)
    plt.close(mass_flux_fig)
    add_record("shell_mass_flux_map_png %r", str(mass_flux_png.relative_to(path.parent)))

    mass_loss_fig, mass_loss_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    mass_loss_ax.plot(height_r, mass_loss_kg_s, ".-", color="C0")
    mass_loss_ax.set_xlabel("Height [R]")
    mass_loss_ax.set_ylabel("Mass loss [kg/s]")
    mass_loss_ax.set_title("Wind Mass Loss")
    mass_loss_ax.grid(True, alpha=0.25)
    mass_loss_png = output_dir / f"{prefix}.mass_loss_profile.png"
    mass_loss_fig.savefig(mass_loss_png)
    plt.close(mass_loss_fig)
    add_record("shell_mass_loss_profile_png %r", str(mass_loss_png.relative_to(path.parent)))
    add_record("shell_radius_R %r", shell_radii_r)
    add_record("shell_mass_loss_kg_s %r", mass_loss_kg_s)
    add_record("shell_mass_loss_value_kg_s %r", mass_loss_kg_s[-1])
    log.info("Computing shell wind mass flux complete.")

    # Start: compute, plot, and record shell angular momentum flux.
    log.debug("Computing shell angular momentum flux...")
    b_vector = np.stack(
        (
            np.ravel(smart_ds("B_x [T]")),
            np.ravel(smart_ds("B_y [T]")),
            np.ravel(smart_ds("B_z [T]")),
        ),
        axis=0,
    )
    b_r, _b_polar, b_phi = shell_spherical_components(
        b_vector,
        lon_deg=lon_all,
        lat_deg=lat_all,
    )
    varpi_m = r_all * star_radius_m * np.cos(np.deg2rad(lat_all))
    torque_density = varpi_m * (rho * u_phi * u_r - (b_phi * b_r / MU0))
    total_torque_nm = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        torque_cells = shell_cell_values(
            torque_density,
            shell_mask=shell_mask,
            lon_deg=lon_all,
            lat_deg=lat_all,
            lon_nodes=lon_nodes,
            lat_nodes=lat_nodes,
        )
        total_torque_nm.append(float(np.sum(torque_cells * area_m2)))

    torque_map = shell_cell_values(
        torque_density,
        shell_mask=shell_masks[-1],
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    torque_map_fig, torque_map_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = torque_map_ax.pcolormesh(lon_nodes, lat_nodes, torque_map, shading="flat", cmap="cividis")
    torque_map_ax.set_xlabel("Longitude [deg]")
    torque_map_ax.set_ylabel("Latitude [deg]")
    torque_map_ax.set_title("Angular Momentum Flux")
    torque_map_fig.colorbar(image, ax=torque_map_ax, label="Torque density [N/m]")
    torque_map_png = output_dir / f"{prefix}.torque_map.png"
    torque_map_fig.savefig(torque_map_png)
    plt.close(torque_map_fig)
    add_record("shell_torque_map_png %r", str(torque_map_png.relative_to(path.parent)))

    torque_profile_fig, torque_profile_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    torque_profile_ax.plot(height_r, total_torque_nm, ".-", color="C1")
    torque_profile_ax.set_xlabel("Height [R]")
    torque_profile_ax.set_ylabel("Torque [Nm]")
    torque_profile_ax.set_title("Wind Torque")
    torque_profile_ax.grid(True, alpha=0.25)
    torque_profile_png = output_dir / f"{prefix}.torque_profile.png"
    torque_profile_fig.savefig(torque_profile_png)
    plt.close(torque_profile_fig)
    add_record("shell_torque_profile_png %r", str(torque_profile_png.relative_to(path.parent)))
    add_record("shell_total_torque_nm %r", total_torque_nm)
    add_record("shell_total_torque_value_nm %r", total_torque_nm[-1])
    log.info("Computing shell angular momentum flux complete.")

    # Start: compute, plot, and record shell energy flux.
    log.debug("Computing shell energy flux...")
    energy_density = np.ravel(smart_ds("E [J/m^3]"))
    energy_flux = energy_density * u_r
    energy_flow_w = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        energy_cells = shell_cell_values(
            energy_flux,
            shell_mask=shell_mask,
            lon_deg=lon_all,
            lat_deg=lat_all,
            lon_nodes=lon_nodes,
            lat_nodes=lat_nodes,
        )
        energy_flow_w.append(float(np.sum(energy_cells * area_m2)))

    energy_map = shell_cell_values(
        energy_flux,
        shell_mask=shell_masks[-1],
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    energy_map_fig, energy_map_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = energy_map_ax.pcolormesh(lon_nodes, lat_nodes, energy_map, shading="flat", cmap="plasma")
    energy_map_ax.set_xlabel("Longitude [deg]")
    energy_map_ax.set_ylabel("Latitude [deg]")
    energy_map_ax.set_title("Energy Flux")
    energy_map_fig.colorbar(image, ax=energy_map_ax, label="Energy flux [W/m^2]")
    energy_map_png = output_dir / f"{prefix}.energy_flux_map.png"
    energy_map_fig.savefig(energy_map_png)
    plt.close(energy_map_fig)
    add_record("shell_energy_flux_map_png %r", str(energy_map_png.relative_to(path.parent)))

    energy_profile_fig, energy_profile_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    energy_profile_ax.plot(height_r, energy_flow_w, ".-", color="C2")
    energy_profile_ax.set_xlabel("Height [R]")
    energy_profile_ax.set_ylabel("Energy flow [W]")
    energy_profile_ax.set_title("Shell Energy Flow")
    energy_profile_ax.grid(True, alpha=0.25)
    energy_profile_png = output_dir / f"{prefix}.energy_flow_profile.png"
    energy_profile_fig.savefig(energy_profile_png)
    plt.close(energy_profile_fig)
    add_record("shell_energy_flow_profile_png %r", str(energy_profile_png.relative_to(path.parent)))
    add_record("shell_energy_flow_w %r", energy_flow_w)
    add_record("shell_energy_flow_value_w %r", energy_flow_w[-1])
    log.info("Computing shell energy flux complete.")

    # Start: compute, plot, and record shell open magnetic flux.
    log.debug("Computing shell open magnetic flux...")
    open_flux_density = np.abs(b_r)
    open_flux_wb = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        open_flux_cells = shell_cell_values(
            open_flux_density,
            shell_mask=shell_mask,
            lon_deg=lon_all,
            lat_deg=lat_all,
            lon_nodes=lon_nodes,
            lat_nodes=lat_nodes,
        )
        open_flux_wb.append(float(np.sum(open_flux_cells * area_m2)))

    open_flux_map = shell_cell_values(
        open_flux_density,
        shell_mask=shell_masks[-1],
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    open_flux_map_fig, open_flux_map_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = open_flux_map_ax.pcolormesh(lon_nodes, lat_nodes, open_flux_map, shading="flat", cmap="magma")
    open_flux_map_ax.set_xlabel("Longitude [deg]")
    open_flux_map_ax.set_ylabel("Latitude [deg]")
    open_flux_map_ax.set_title("Open Magnetic Flux Density")
    open_flux_map_fig.colorbar(image, ax=open_flux_map_ax, label="|B_r| [T]")
    open_flux_map_png = output_dir / f"{prefix}.open_flux_map.png"
    open_flux_map_fig.savefig(open_flux_map_png)
    plt.close(open_flux_map_fig)
    add_record("shell_open_flux_map_png %r", str(open_flux_map_png.relative_to(path.parent)))

    open_flux_profile_fig, open_flux_profile_ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    open_flux_profile_ax.plot(height_r, open_flux_wb, ".-", color="C3")
    open_flux_profile_ax.set_xlabel("Height [R]")
    open_flux_profile_ax.set_ylabel("Open flux [Wb]")
    open_flux_profile_ax.set_title("Open Magnetic Flux")
    open_flux_profile_ax.grid(True, alpha=0.25)
    open_flux_profile_png = output_dir / f"{prefix}.open_flux_profile.png"
    open_flux_profile_fig.savefig(open_flux_profile_png)
    plt.close(open_flux_profile_fig)
    add_record("shell_open_flux_profile_png %r", str(open_flux_profile_png.relative_to(path.parent)))
    add_record("shell_open_flux_wb %r", open_flux_wb)
    add_record("shell_open_flux_value_wb %r", open_flux_wb[-1])
    log.info("Computing shell open magnetic flux complete.")
