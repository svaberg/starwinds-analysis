"""Per-file 3D volume pipeline for `batwind-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from math import isfinite
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from batcamp import camera_rays
from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer

from batwind.constants import DEFAULT_QUICKLOOK_RADII_R
from batwind.analysis.shells import integrate_shell_scalar
from batwind.analysis.shells import sample_spherical_shells_fibonacci
from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
LOS_GRID_N = 512


def build_rho2_los_renderer(smart_ds: SmartDs) -> tuple[OctreeRayTracer, OctreeInterpolator, tuple[float, float, float, float, float, float]]:
    """
    Build the shared octree LOS renderer state for `rho^2`.
    """
    x = np.asarray(smart_ds["X [R]"], dtype=float)
    y = np.asarray(smart_ds["Y [R]"], dtype=float)
    z = np.asarray(smart_ds["Z [R]"], dtype=float)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    tree = Octree.from_ds(smart_ds.raw)
    rho = np.asarray(smart_ds["Rho [kg/m^3]"], dtype=float)
    interp = OctreeInterpolator(tree, rho**2)
    tracer = OctreeRayTracer(tree)
    return tracer, interp, (x_min, x_max, y_min, y_max, z_min, z_max)


def render_rho2_los_image(
    tracer: OctreeRayTracer,
    interp: OctreeInterpolator,
    bounds_r: tuple[float, float, float, float, float, float],
    *,
    body_radius_m: float,
    image_n: int,
    view_axis: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
    """
    Render one physical-unit LOS image of `rho^2` through the octree.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds_r
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    x_span = x_max - x_min
    z_span = z_max - z_min
    x_pad = max(1.0e-6 * x_span, 1.0e-6)
    z_pad = max(1.0e-6 * z_span, 1.0e-6)

    if view_axis == "+Z":
        origins, directions = camera_rays(
            origin=(x_center, y_center, z_min - z_pad),
            target=(x_center, y_center, z_max + z_pad),
            up=(0.0, 1.0, 0.0),
            nx=image_n,
            ny=image_n,
            width=x_max - x_min,
            height=y_max - y_min,
            projection="parallel",
        )
        extent = (x_min, x_max, y_min, y_max)
    elif view_axis == "+X":
        origins, directions = camera_rays(
            origin=(x_min - x_pad, y_center, z_center),
            target=(x_max + x_pad, y_center, z_center),
            up=(0.0, 0.0, 1.0),
            nx=image_n,
            ny=image_n,
            width=y_max - y_min,
            height=z_max - z_min,
            projection="parallel",
        )
        extent = (y_min, y_max, z_min, z_max)
    else:
        raise ValueError(f"Unsupported LOS view_axis {view_axis!r}")

    image_r_units, counts = tracer.trilinear_image(interp, origins, directions)
    image_m_units = np.asarray(image_r_units, dtype=float) * float(body_radius_m)
    return image_m_units, extent, counts


def process_plt_file(file_path: str | Path) -> None:
    """Process one 3D `.plt` file into a shell PNG and recorded diagnostics."""
    # Start: resolve input/output paths and log the file being processed.
    stage_start = perf_counter()
    log.info("Resolving volume pipeline paths...")
    path = Path(file_path)
    output_dir = path.parent / "volume"
    prefix = output_prefix_from_input_file(path.name)
    log.info("%s", path.name)
    log.debug("Resolving volume pipeline paths complete in %.2f s.", perf_counter() - stage_start)

    # Start: load the dataset.
    stage_start = perf_counter()
    log.info("Loading volume dataset...")
    smart_ds = SmartDs.from_file(path, batsrus=True, spherical=True)
    log.debug("Loading volume dataset complete in %.2f s.", perf_counter() - stage_start)

    # Start: create the output figure canvas.
    stage_start = perf_counter()
    log.info("Preparing volume dataset and figure canvas...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    radii = DEFAULT_QUICKLOOK_RADII_R
    log.debug("Preparing volume dataset and figure canvas complete in %.2f s.", perf_counter() - stage_start)

    # Start: sample shells once for all diagnostics.
    stage_start = perf_counter()
    log.info("Sampling shell grid once for all diagnostics...")
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
    log.debug("Sampling shell grid once for all diagnostics complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record wind mass loss.
    stage_start = perf_counter()
    log.info("Computing wind mass loss...")
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
    log.debug("Computing wind mass loss complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record wind torque.
    stage_start = perf_counter()
    log.info("Computing wind torque...")
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
    log.debug("Computing wind torque complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record open magnetic flux.
    stage_start = perf_counter()
    log.info("Computing open magnetic flux...")
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
    log.debug("Computing open magnetic flux complete in %.2f s.", perf_counter() - stage_start)

    # Start: compute, plot, and record energy flux.
    stage_start = perf_counter()
    log.info("Computing energy flux...")
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
    log.debug("Computing energy flux complete in %.2f s.", perf_counter() - stage_start)

    # Start: build LOS renderer once, then render the face-on and side views.
    stage_start = perf_counter()
    log.info("Rendering LOS rho^2 images...")
    image_n = LOS_GRID_N
    tracer, interp, bounds_r = build_rho2_los_renderer(smart_ds)
    rho_sq_los, los_extent, los_counts = render_rho2_los_image(
        tracer,
        interp,
        bounds_r,
        body_radius_m=body_radius,
        image_n=image_n,
        view_axis="+Z",
    )
    rho_sq_los_side, los_side_extent, los_side_counts = render_rho2_los_image(
        tracer,
        interp,
        bounds_r,
        body_radius_m=body_radius,
        image_n=image_n,
        view_axis="+X",
    )
    positive = rho_sq_los[np.isfinite(rho_sq_los) & (rho_sq_los > 0.0)]
    los_fig, los_ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    los_norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive))) if positive.size else None
    image = los_ax.imshow(
        rho_sq_los,
        origin="lower",
        extent=los_extent,
        cmap="magma",
        norm=los_norm,
        aspect="equal",
    )
    los_ax.set_title(r"LOS $\int \rho^2\,dl$")
    los_ax.set_xlabel("X [R]")
    los_ax.set_ylabel("Y [R]")
    los_fig.colorbar(image, ax=los_ax, label=r"$\int \rho^2\,dl$ [kg$^2$/m$^5$]")
    los_png = output_dir / f"{prefix}.rho2_los.png"
    los_fig.savefig(los_png)
    plt.close(los_fig)
    positive_side = rho_sq_los_side[np.isfinite(rho_sq_los_side) & (rho_sq_los_side > 0.0)]
    los_side_fig, los_side_ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    los_side_norm = (
        LogNorm(vmin=float(np.nanmin(positive_side)), vmax=float(np.nanmax(positive_side)))
        if positive_side.size
        else None
    )
    image_side = los_side_ax.imshow(
        rho_sq_los_side,
        origin="lower",
        extent=los_side_extent,
        cmap="magma",
        norm=los_side_norm,
        aspect="equal",
    )
    los_side_ax.set_title(r"Side LOS $\int \rho^2\,dl$")
    los_side_ax.set_xlabel("Y [R]")
    los_side_ax.set_ylabel("Z [R]")
    los_side_fig.colorbar(image_side, ax=los_side_ax, label=r"$\int \rho^2\,dl$ [kg$^2$/m$^5$]")
    los_side_png = output_dir / f"{prefix}.rho2_los_side.png"
    los_side_fig.savefig(los_side_png)
    plt.close(los_side_fig)
    add_record("volume_rho2_los_png %r", str(los_png.relative_to(path.parent)))
    add_record("volume_rho2_los_side_png %r", str(los_side_png.relative_to(path.parent)))
    add_record("volume_rho2_los_image_n %r", image_n)
    add_record("volume_rho2_los_view_axis %r", "+Z")
    add_record("volume_rho2_los_side_view_axis %r", "+X")
    add_record("volume_rho2_los_unit %r", "kg^2/m^5")
    add_record("volume_rho2_los_nonempty_rays %r", int(np.count_nonzero(np.asarray(los_counts) > 0)))
    add_record("volume_rho2_los_side_nonempty_rays %r", int(np.count_nonzero(np.asarray(los_side_counts) > 0)))
    log.debug("Rendering LOS rho^2 images complete in %.2f s.", perf_counter() - stage_start)

    # Start: save the figure and record the output artifact.
    stage_start = perf_counter()
    log.info("Saving volume figure...")
    shell_png = output_dir / f"{prefix}.shells.png"
    fig.savefig(shell_png)
    plt.close(fig)
    add_record("volume_shell_png %r", str(shell_png.relative_to(path.parent)))
    log.debug("Saving volume figure complete in %.2f s.", perf_counter() - stage_start)
