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


def load_shell_grid(smart_ds: SmartDs):
    """Load the native shell grid and precompute shell masks, areas, and plot geometry."""
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
    return (
        r_all,
        star_radius_m,
        lon_all,
        lat_all,
        shell_radii_r,
        lon_nodes,
        lat_nodes,
        shell_masks,
        shell_areas_m2,
        height_r,
    )


def shell_map_and_profile(
    values,
    *,
    shell_masks,
    shell_areas_m2,
    lon_deg,
    lat_deg,
    lon_nodes,
    lat_nodes,
):
    """Build the outer-shell cell map and the integrated radial profile for one shell field."""
    integrated_values = []
    for shell_mask, area_m2 in zip(shell_masks, shell_areas_m2):
        shell_cells = shell_cell_values(
            values,
            shell_mask=shell_mask,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            lon_nodes=lon_nodes,
            lat_nodes=lat_nodes,
        )
        integrated_values.append(float(np.sum(shell_cells * area_m2)))
    outer_map = shell_cell_values(
        values,
        shell_mask=shell_masks[-1],
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    return outer_map, integrated_values


def save_shell_map(
    output_dir: Path,
    prefix: str,
    stem: str,
    values,
    lon_nodes,
    lat_nodes,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
):
    """Save one shell longitude-latitude map figure and return the output path."""
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.pcolormesh(lon_nodes, lat_nodes, values, shading="flat", cmap=cmap)
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    png_path = output_dir / f"{prefix}.{stem}.png"
    figure.savefig(png_path)
    plt.close(figure)
    return png_path


def save_shell_profile(
    output_dir: Path,
    prefix: str,
    stem: str,
    height_r,
    values,
    *,
    title: str,
    y_label: str,
    color: str,
):
    """Save one shell radial profile figure and return the output path."""
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(height_r, values, ".-", color=color)
    axis.set_xlabel("Height [R]")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    png_path = output_dir / f"{prefix}.{stem}.png"
    figure.savefig(png_path)
    plt.close(figure)
    return png_path


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
    (
        r_all,
        star_radius_m,
        lon_all,
        lat_all,
        shell_radii_r,
        lon_nodes,
        lat_nodes,
        shell_masks,
        shell_areas_m2,
        height_r,
    ) = load_shell_grid(smart_ds)
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
    mass_flux_map, mass_loss_kg_s = shell_map_and_profile(
        mass_flux,
        shell_masks=shell_masks,
        shell_areas_m2=shell_areas_m2,
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    mass_flux_png = save_shell_map(
        output_dir,
        prefix,
        "mass_flux_map",
        mass_flux_map,
        lon_nodes,
        lat_nodes,
        title="Wind Mass Flux",
        colorbar_label="Mass flux [kg/m^2/s]",
        cmap="viridis",
    )
    add_record("shell_mass_flux_map_png %r", str(mass_flux_png.relative_to(path.parent)))
    mass_loss_png = save_shell_profile(
        output_dir,
        prefix,
        "mass_loss_profile",
        height_r,
        mass_loss_kg_s,
        title="Wind Mass Loss",
        y_label="Mass loss [kg/s]",
        color="C0",
    )
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
    torque_map, total_torque_nm = shell_map_and_profile(
        torque_density,
        shell_masks=shell_masks,
        shell_areas_m2=shell_areas_m2,
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    torque_map_png = save_shell_map(
        output_dir,
        prefix,
        "torque_map",
        torque_map,
        lon_nodes,
        lat_nodes,
        title="Angular Momentum Flux",
        colorbar_label="Torque density [N/m]",
        cmap="cividis",
    )
    add_record("shell_torque_map_png %r", str(torque_map_png.relative_to(path.parent)))
    torque_profile_png = save_shell_profile(
        output_dir,
        prefix,
        "torque_profile",
        height_r,
        total_torque_nm,
        title="Wind Torque",
        y_label="Torque [Nm]",
        color="C1",
    )
    add_record("shell_torque_profile_png %r", str(torque_profile_png.relative_to(path.parent)))
    add_record("shell_total_torque_nm %r", total_torque_nm)
    add_record("shell_total_torque_value_nm %r", total_torque_nm[-1])
    log.info("Computing shell angular momentum flux complete.")

    # Start: compute, plot, and record shell energy flux.
    log.debug("Computing shell energy flux...")
    energy_density = np.ravel(smart_ds("E [J/m^3]"))
    energy_flux = energy_density * u_r
    energy_map, energy_flow_w = shell_map_and_profile(
        energy_flux,
        shell_masks=shell_masks,
        shell_areas_m2=shell_areas_m2,
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    energy_map_png = save_shell_map(
        output_dir,
        prefix,
        "energy_flux_map",
        energy_map,
        lon_nodes,
        lat_nodes,
        title="Energy Flux",
        colorbar_label="Energy flux [W/m^2]",
        cmap="plasma",
    )
    add_record("shell_energy_flux_map_png %r", str(energy_map_png.relative_to(path.parent)))
    energy_profile_png = save_shell_profile(
        output_dir,
        prefix,
        "energy_flow_profile",
        height_r,
        energy_flow_w,
        title="Shell Energy Flow",
        y_label="Energy flow [W]",
        color="C2",
    )
    add_record("shell_energy_flow_profile_png %r", str(energy_profile_png.relative_to(path.parent)))
    add_record("shell_energy_flow_w %r", energy_flow_w)
    add_record("shell_energy_flow_value_w %r", energy_flow_w[-1])
    log.info("Computing shell energy flux complete.")

    # Start: compute, plot, and record shell open magnetic flux.
    log.debug("Computing shell open magnetic flux...")
    open_flux_density = np.abs(b_r)
    open_flux_map, open_flux_wb = shell_map_and_profile(
        open_flux_density,
        shell_masks=shell_masks,
        shell_areas_m2=shell_areas_m2,
        lon_deg=lon_all,
        lat_deg=lat_all,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
    )
    open_flux_map_png = save_shell_map(
        output_dir,
        prefix,
        "open_flux_map",
        open_flux_map,
        lon_nodes,
        lat_nodes,
        title="Open Magnetic Flux Density",
        colorbar_label="|B_r| [T]",
        cmap="magma",
    )
    add_record("shell_open_flux_map_png %r", str(open_flux_map_png.relative_to(path.parent)))
    open_flux_profile_png = save_shell_profile(
        output_dir,
        prefix,
        "open_flux_profile",
        height_r,
        open_flux_wb,
        title="Open Magnetic Flux",
        y_label="Open flux [Wb]",
        color="C3",
    )
    add_record("shell_open_flux_profile_png %r", str(open_flux_profile_png.relative_to(path.parent)))
    add_record("shell_open_flux_wb %r", open_flux_wb)
    add_record("shell_open_flux_value_wb %r", open_flux_wb[-1])
    log.info("Computing shell open magnetic flux complete.")
