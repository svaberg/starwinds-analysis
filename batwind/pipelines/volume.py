"""Per-file 3D volume pipeline for `batwind-pipe` (minimal, user-serviceable)."""

from __future__ import annotations

import logging
from math import isfinite
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import ticker
from batcamp import camera_rays
from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer

from batwind.algorithms.octree_integration import cumulative_radius_exact_rpa
from batwind.algorithms.octree_integration import radial_emission_profile_exact_rpa
from batwind.constants import DEFAULT_QUICKLOOK_RADII_R
from batwind.analysis.shells import integrate_shell_scalar
from batwind.analysis.shells import sample_spherical_shells_fibonacci
from batwind.physics.xray import band_emissivity_from_response_table_legacy
from batwind.physics.xray import DEFAULT_RESPONSE_FUNCTION_PATH
from batwind.physics.xray import point_unblocked_solid_angle_sr
from batwind.pipelines.utils import output_prefix_from_input_file
from batwind.smart_ds import SmartDs

log = logging.getLogger(__name__)
# Method for recording structured, machine-ingested pipeline payloads.
add_record = logging.getLogger(f"recorder.{__name__}").debug
LOS_GRID_N = 512
LOS_EXAMPLE_GRID_N = 192
LOS_EXAMPLE_SIDE_LENGTH_R = 2.0
X_RAY_BAND_LABELS = {
    "hard": r"Hard X-ray band intensity [$10^{-26}$]",
    "rosat": r"ROSAT band intensity [$10^{-26}$]",
    "euv": r"EUV band intensity [$10^{-26}$]",
}
X_RAY_EMISSION_TOTAL_UNIT = r"$10^{-26}$ cm$^2$"
X_RAY_DIRECTIONAL_EMISSIVITY_UNIT = r"$10^{-26}$ cm$^{-1}$"
X_RAY_SINGLE_DIRECTION_VIEW_AXIS = "+Y"
X_RAY_TOTALS_IMAGE_N = 512


def build_los_geometry(smart_ds: SmartDs) -> tuple[Octree, OctreeRayTracer, tuple[float, float, float, float, float, float]]:
    """
    Build the shared octree LOS geometry state.
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
    tracer = OctreeRayTracer(tree)
    return tree, tracer, (x_min, x_max, y_min, y_max, z_min, z_max)


def build_los_interpolator(tree: Octree, point_values: np.ndarray) -> OctreeInterpolator:
    """
    Build an octree interpolator from one point-valued scalar field.
    """
    return OctreeInterpolator(tree, np.asarray(point_values, dtype=float))


def build_rho2_los_renderer(
    smart_ds: SmartDs,
) -> tuple[OctreeRayTracer, OctreeInterpolator, tuple[float, float, float, float, float, float]]:
    """
    Build the shared octree LOS renderer state for `rho^2`.
    """
    tree, tracer, bounds_r = build_los_geometry(smart_ds)
    rho = np.asarray(smart_ds["Rho [kg/m^3]"], dtype=float)
    interp = build_los_interpolator(tree, rho**2)
    return tracer, interp, bounds_r


def integrate_image_total(
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    body_radius_cm: float,
) -> float:
    """
    Integrate one LOS image over its image-plane area.
    """
    x_min, x_max, y_min, y_max = extent
    pixel_area_cm2 = (
        (float(x_max - x_min) * body_radius_cm) * (float(y_max - y_min) * body_radius_cm) / float(image.size)
    )
    return float(np.nansum(np.asarray(image, dtype=float)) * pixel_area_cm2)


def plot_xray_radial_summary(
    radial_png_path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stats: dict[str, dict[str, float]],
) -> None:
    """
    Plot per-band radial emission and cumulative-fraction summaries.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.5), constrained_layout=True, sharex=True)
    colors = {"hard": "C0", "rosat": "C1", "euv": "C2"}
    for band_name, (radius_r, shell_emission, cumulative_fraction) in profiles.items():
        positive_shell_emission = np.where(np.asarray(shell_emission, dtype=float) > 0.0, shell_emission, np.nan)
        axes[0].plot(
            radius_r,
            positive_shell_emission,
            color=colors[band_name],
            label=(
                f"{band_name}: "
                f"r90={stats[band_name]['r90_r']:.2f} R*, "
                f"r99={stats[band_name]['r99_r']:.2f} R*"
            ),
        )
        axes[1].plot(radius_r, cumulative_fraction, color=colors[band_name], label=band_name)
        axes[1].axvline(stats[band_name]["r90_r"], color=colors[band_name], linestyle="--", alpha=0.5)
        axes[1].axvline(stats[band_name]["r99_r"], color=colors[band_name], linestyle=":", alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel(f"Shell emission [{X_RAY_EMISSION_TOTAL_UNIT}]")
    axes[1].set_ylabel("Cumulative fraction [-]")
    axes[1].set_xlabel(r"Radius [$R_\star$]")
    axes[0].set_title("X-ray Band Emission by Radius")
    axes[1].set_title("Cumulative Emission Fraction")
    axes[1].axhline(0.90, color="0.3", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.99, color="0.3", linestyle=":", linewidth=0.8)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.savefig(radial_png_path)
    plt.close(fig)


def plot_xray_unit_summary(
    summary_png_path: Path,
    stats: dict[str, dict[str, float]],
    *,
    view_axis: str,
) -> None:
    """
    Plot a unit-carrying summary of the X-ray totals.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    ax.axis("off")
    lines = [
        "X-ray emission unit trace",
        r"$G(T)$: response table values [cm$^5$/10$^{26}$]",
        r"$N_e$: electron density [cm$^{-3}$]",
        rf"$\epsilon = N_e^2 G(T)$ [{X_RAY_DIRECTIONAL_EMISSIVITY_UNIT}]",
        r"$I_{\mathrm{dir}} = \int \epsilon \, dl$ [$10^{-26}$]",
        rf"$L_{{\mathrm{{dir,{view_axis}}}}} = \int I_{{\mathrm{{dir}}}} \, dA$ [{X_RAY_EMISSION_TOTAL_UNIT}]",
        rf"$L_{{\Omega}} = \int \epsilon \, \Omega_{{\mathrm{{unblocked}}}} \, dV$ [{X_RAY_EMISSION_TOTAL_UNIT}]",
        rf"$L_{{4\pi}} = \int \epsilon \, 4\pi \, dV$ [{X_RAY_EMISSION_TOTAL_UNIT}]",
        "",
        "Band totals",
    ]
    for band_name in ("hard", "rosat", "euv"):
        band_stats = stats[band_name]
        lines.extend(
            [
                (
                    f"{band_name}: "
                    f"L_dir={band_stats['directional_total']:.3e}, "
                    f"L_Ω={band_stats['unblocked_total']:.3e}, "
                    f"L_4π={band_stats['four_pi_total']:.3e}"
                ),
                (
                    f"      r90={band_stats['r90_r']:.2f} R*, "
                    f"r99={band_stats['r99_r']:.2f} R*, "
                    f"L_Ω/L_4π={band_stats['unblocked_over_four_pi']:.4f}"
                ),
            ]
        )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    fig.savefig(summary_png_path)
    plt.close(fig)


def render_rho2_los_image(
    tracer: OctreeRayTracer,
    interp: OctreeInterpolator,
    bounds_r: tuple[float, float, float, float, float, float],
    *,
    path_length_scale: float,
    image_n: int,
    view_axis: str,
    width: float | None = None,
    height: float | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
    """
    Render one LOS image through the octree with an explicit path-length scale.
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
        width = (x_max - x_min) if width is None else float(width)
        height = (y_max - y_min) if height is None else float(height)
        origins, directions = camera_rays(
            origin=(x_center, y_center, z_min - z_pad),
            target=(x_center, y_center, z_max + z_pad),
            up=(0.0, 1.0, 0.0),
            nx=image_n,
            ny=image_n,
            width=width,
            height=height,
            projection="parallel",
        )
        extent = (x_center - 0.5 * width, x_center + 0.5 * width, y_center - 0.5 * height, y_center + 0.5 * height)
    elif view_axis == "+X":
        width = (y_max - y_min) if width is None else float(width)
        height = (z_max - z_min) if height is None else float(height)
        origins, directions = camera_rays(
            origin=(x_min - x_pad, y_center, z_center),
            target=(x_max + x_pad, y_center, z_center),
            up=(0.0, 0.0, 1.0),
            nx=image_n,
            ny=image_n,
            width=width,
            height=height,
            projection="parallel",
        )
        extent = (y_center - 0.5 * width, y_center + 0.5 * width, z_center - 0.5 * height, z_center + 0.5 * height)
    elif view_axis == "+Y":
        width = (x_max - x_min) if width is None else float(width)
        height = (z_max - z_min) if height is None else float(height)
        origins, directions = camera_rays(
            origin=(x_center, y_min - x_pad, z_center),
            target=(x_center, y_max + x_pad, z_center),
            up=(0.0, 0.0, 1.0),
            nx=image_n,
            ny=image_n,
            width=width,
            height=height,
            projection="parallel",
        )
        extent = (x_center - 0.5 * width, x_center + 0.5 * width, z_center - 0.5 * height, z_center + 0.5 * height)
    else:
        raise ValueError(f"Unsupported LOS view_axis {view_axis!r}")

    image_r_units, counts = tracer.trilinear_image(interp, origins, directions)
    image_scaled = np.asarray(image_r_units, dtype=float) * float(path_length_scale)
    return image_scaled, extent, counts


def save_los_colormesh_npz(
    npz_path: Path,
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    counts: np.ndarray,
    *,
    view_axis: str,
) -> None:
    """
    Save one LOS image as a reusable colormesh product.
    """
    x_min, x_max, y_min, y_max = extent
    y_n, x_n = image.shape
    x = np.linspace(x_min, x_max, x_n)
    y = np.linspace(y_min, y_max, y_n)
    if view_axis == "+Z":
        xlabel = "X [R]"
        ylabel = "Y [R]"
        title = r"LOS $\int \rho^2\,dl$"
    elif view_axis == "+X":
        xlabel = "Y [R]"
        ylabel = "Z [R]"
        title = r"Side LOS $\int \rho^2\,dl$"
    elif view_axis == "+Y":
        xlabel = "X [R]"
        ylabel = "Z [R]"
        title = r"Example LOS $\int \rho^2\,dl$"
    else:
        raise ValueError(f"Unsupported LOS view_axis {view_axis!r}")
    np.savez_compressed(
        npz_path,
        x=x,
        y=y,
        image=np.asarray(image, dtype=float),
        counts=np.asarray(counts),
        view_axis=np.asarray(view_axis),
        xlabel=np.asarray(xlabel),
        ylabel=np.asarray(ylabel),
        title=np.asarray(title),
        colorbar_label=np.asarray(r"$\int \rho^2\,dl$ [kg$^2$/m$^5$]"),
        unit=np.asarray("kg^2/m^5"),
    )


def save_example_los_colormesh_npz(
    source_npz_path: Path,
    example_npz_path: Path,
    *,
    side_length_r: float,
    colorbar_label: str,
    unit: str,
) -> None:
    """
    Save a cropped example-panel LOS colormesh product.
    """
    with np.load(source_npz_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        image = np.asarray(data["image"], dtype=float)
        counts = np.asarray(data["counts"])
        view_axis = str(data["view_axis"])
    x_mask = np.abs(x) <= side_length_r
    y_mask = np.abs(y) <= side_length_r
    x_crop = x[x_mask]
    y_crop = y[y_mask]
    image_crop = image[np.ix_(y_mask, x_mask)]
    counts_crop = counts[np.ix_(y_mask, x_mask)]
    np.savez_compressed(
        example_npz_path,
        x=x_crop,
        y=y_crop,
        image=np.asarray(image_crop, dtype=float),
        counts=counts_crop,
        view_axis=np.asarray(view_axis),
        xlabel=np.asarray(r"$x$ $(R_\star)$"),
        ylabel=np.asarray(r"$z$ $(R_\star)$"),
        colorbar_label=np.asarray(colorbar_label),
        unit=np.asarray(unit),
        side_length_r=np.asarray(float(side_length_r)),
    )


def overlay_sphere_graticule(
    ax: plt.Axes,
    *,
    radius: float = 1.0,
    central_lon_deg: float = 0.0,
    central_lat_deg: float = 40.0,
    color: str = "black",
    linestyle: str = "dotted",
    linewidth: float = 0.5,
) -> None:
    """
    Overlay a simple orthographic globe graticule in data coordinates.
    """
    lon0 = np.deg2rad(central_lon_deg)
    lat0 = np.deg2rad(central_lat_deg)

    def project(lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
        x = radius * np.cos(lat) * np.sin(lon - lon0)
        y = radius * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0))
        return x, y, cos_c >= 0.0

    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=color, linestyle=linestyle, linewidth=linewidth)

    longitudes = np.arange(-180.0, 180.0, 45.0)
    latitudes = np.arange(-60.0, 61.0, 30.0)
    lat_line = np.linspace(-90.0, 90.0, 721)
    lon_line = np.linspace(-180.0, 180.0, 721)

    for lon_deg in longitudes:
        lon = np.full_like(lat_line, lon_deg)
        x, y, visible = project(lon, lat_line)
        ax.plot(x[visible], y[visible], color=color, linestyle=linestyle, linewidth=linewidth)

    for lat_deg in latitudes:
        lat = np.full_like(lon_line, lat_deg)
        x, y, visible = project(lon_line, lat)
        ax.plot(x[visible], y[visible], color=color, linestyle=linestyle, linewidth=linewidth)


def plot_los_colormesh_npz(npz_path: Path, png_path: Path) -> None:
    """
    Plot one saved LOS colormesh product.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        image = np.asarray(data["image"], dtype=float)
        xlabel = str(data["xlabel"])
        ylabel = str(data["ylabel"])
        title = str(data["title"])
        colorbar_label = str(data["colorbar_label"])
    x_mesh, y_mesh = np.meshgrid(x, y)
    x_span = float(x[-1] - x[0]) if x.size else 0.0
    y_span = float(y[-1] - y[0]) if y.size else 0.0
    positive = image[np.isfinite(image) & (image > 0.0)]
    norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive))) if positive.size else None
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    mesh = ax.pcolormesh(
        x_mesh,
        y_mesh,
        image,
        cmap="viridis",
        norm=norm,
        shading="gouraud",
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    if x_span <= 10.0:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    if y_span <= 10.0:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    fig.savefig(png_path)
    plt.close(fig)


def plot_example_los_colormesh_npz(npz_path: Path, png_path: Path) -> None:
    """
    Plot one cropped example-panel LOS colormesh product in the old notebook style.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        image = np.asarray(data["image"], dtype=float)
        xlabel = str(data["xlabel"])
        ylabel = str(data["ylabel"])
        colorbar_label = str(data["colorbar_label"])
        side_length_r = float(data["side_length_r"])
    positive = image[np.isfinite(image) & (image > 0.0)]
    norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive))) if positive.size else None
    fig = plt.figure(figsize=(4.2, 4.3), constrained_layout=True)
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(x, y, image, cmap="viridis", norm=norm, shading="nearest", rasterized=True)
    ax.set_xlim(-side_length_r, side_length_r)
    ax.set_ylim(-side_length_r, side_length_r)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.grid(False)
    overlay_sphere_graticule(ax, color="0.2", linewidth=0.35)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.02, fraction=0.06, location="top")
    colorbar.set_label(colorbar_label)
    fig.savefig(png_path)
    plt.close(fig)


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
    tree, tracer, bounds_r = build_los_geometry(smart_ds)
    interp = build_los_interpolator(tree, np.asarray(smart_ds["Rho [kg/m^3]"], dtype=float) ** 2)
    rho_sq_los, los_extent, los_counts = render_rho2_los_image(
        tracer,
        interp,
        bounds_r,
        path_length_scale=body_radius,
        image_n=image_n,
        view_axis="+Z",
    )
    rho_sq_los_side, los_side_extent, los_side_counts = render_rho2_los_image(
        tracer,
        interp,
        bounds_r,
        path_length_scale=body_radius,
        image_n=image_n,
        view_axis="+X",
    )
    rho_sq_los_example, los_example_extent, los_example_counts = render_rho2_los_image(
        tracer,
        interp,
        bounds_r,
        path_length_scale=body_radius,
        image_n=LOS_EXAMPLE_GRID_N,
        view_axis="+Y",
        width=2.0 * LOS_EXAMPLE_SIDE_LENGTH_R,
        height=2.0 * LOS_EXAMPLE_SIDE_LENGTH_R,
    )
    los_npz = output_dir / f"{prefix}.rho2_los.npz"
    los_png = output_dir / f"{prefix}.rho2_los.png"
    save_los_colormesh_npz(
        los_npz,
        rho_sq_los,
        los_extent,
        los_counts,
        view_axis="+Z",
    )
    plot_los_colormesh_npz(los_npz, los_png)
    los_side_npz = output_dir / f"{prefix}.rho2_los_side.npz"
    los_side_png = output_dir / f"{prefix}.rho2_los_side.png"
    save_los_colormesh_npz(
        los_side_npz,
        rho_sq_los_side,
        los_side_extent,
        los_side_counts,
        view_axis="+X",
    )
    plot_los_colormesh_npz(los_side_npz, los_side_png)
    los_example_npz_full = output_dir / f"{prefix}.rho2_los_example_full.npz"
    save_los_colormesh_npz(
        los_example_npz_full,
        rho_sq_los_example,
        los_example_extent,
        los_example_counts,
        view_axis="+Y",
    )
    los_example_npz = output_dir / f"{prefix}.rho2_los_example.npz"
    save_example_los_colormesh_npz(
        los_example_npz_full,
        los_example_npz,
        side_length_r=LOS_EXAMPLE_SIDE_LENGTH_R,
        colorbar_label="Proxy LOS intensity",
        unit="kg^2/m^5",
    )
    los_example_png = output_dir / f"{prefix}.rho2_los_example.png"
    plot_example_los_colormesh_npz(los_example_npz, los_example_png)

    response_path = DEFAULT_RESPONSE_FUNCTION_PATH
    body_radius_cm = 1.0e2 * body_radius
    path_length_scale_cgs = 1.0e-26 * body_radius_cm
    point_unblocked_solid_angle = point_unblocked_solid_angle_sr(smart_ds)
    raw_band_emissivities = {
        band_name: band_emissivity_from_response_table_legacy(
            smart_ds,
            band_name,
            response_path=response_path,
        )
        for band_name in ("hard", "rosat", "euv")
    }
    band_emissivities = {
        band_name: np.asarray(
            raw_band_emissivity * point_unblocked_solid_angle,
            dtype=float,
        )
        for band_name, raw_band_emissivity in raw_band_emissivities.items()
    }
    xray_band_stats = {}
    xray_band_profiles = {}
    for band_name, emissivity in band_emissivities.items():
        band_interp = build_los_interpolator(tree, emissivity)
        band_image, band_extent, band_counts = render_rho2_los_image(
            tracer,
            band_interp,
            bounds_r,
            path_length_scale=path_length_scale_cgs,
            image_n=LOS_EXAMPLE_GRID_N,
            view_axis="+Y",
            width=2.0 * LOS_EXAMPLE_SIDE_LENGTH_R,
            height=2.0 * LOS_EXAMPLE_SIDE_LENGTH_R,
        )
        band_npz_full = output_dir / f"{prefix}.{band_name}_los_example_full.npz"
        save_los_colormesh_npz(
            band_npz_full,
            band_image,
            band_extent,
            band_counts,
            view_axis="+Y",
        )
        band_npz = output_dir / f"{prefix}.{band_name}_los_example.npz"
        save_example_los_colormesh_npz(
            band_npz_full,
            band_npz,
            side_length_r=LOS_EXAMPLE_SIDE_LENGTH_R,
            colorbar_label=X_RAY_BAND_LABELS[band_name],
            unit="band-intensity",
        )
        band_png = output_dir / f"{prefix}.{band_name}_los_example.png"
        plot_example_los_colormesh_npz(band_npz, band_png)
        add_record(f"volume_{band_name}_los_example_npz %r", str(band_npz.relative_to(path.parent)))
        add_record(f"volume_{band_name}_los_example_png %r", str(band_png.relative_to(path.parent)))
        add_record(f"volume_{band_name}_los_example_response %r", str(response_path))

        directional_image, directional_extent, _ = render_rho2_los_image(
            tracer,
            build_los_interpolator(tree, raw_band_emissivities[band_name]),
            bounds_r,
            path_length_scale=path_length_scale_cgs,
            image_n=X_RAY_TOTALS_IMAGE_N,
            view_axis=X_RAY_SINGLE_DIRECTION_VIEW_AXIS,
        )
        directional_total = integrate_image_total(directional_image, directional_extent, body_radius_cm)
        point_unblocked_luminosity_density = raw_band_emissivities[band_name] * point_unblocked_solid_angle
        point_four_pi_luminosity_density = raw_band_emissivities[band_name] * (4.0 * np.pi)
        radius_r, unblocked_shell_total, unblocked_cumulative_fraction = radial_emission_profile_exact_rpa(
            tree,
            point_unblocked_luminosity_density,
            length_scale=body_radius_cm,
        )
        unblocked_shell_total = 1.0e-26 * unblocked_shell_total
        unblocked_total = float(np.sum(unblocked_shell_total))
        four_pi_total = float(
            np.sum(
                1.0e-26
                * radial_emission_profile_exact_rpa(
                    tree,
                    point_four_pi_luminosity_density,
                    length_scale=body_radius_cm,
                )[1]
            )
        )
        r90_r = cumulative_radius_exact_rpa(
            tree,
            point_unblocked_luminosity_density,
            0.90,
            length_scale=body_radius_cm,
        )
        r99_r = cumulative_radius_exact_rpa(
            tree,
            point_unblocked_luminosity_density,
            0.99,
            length_scale=body_radius_cm,
        )
        xray_band_stats[band_name] = {
            "directional_total": directional_total,
            "unblocked_total": unblocked_total,
            "four_pi_total": four_pi_total,
            "unblocked_over_four_pi": unblocked_total / four_pi_total if four_pi_total > 0.0 else float("nan"),
            "r90_r": r90_r,
            "r99_r": r99_r,
        }
        xray_band_profiles[band_name] = (radius_r, unblocked_shell_total, unblocked_cumulative_fraction)
        add_record(f"volume_{band_name}_directional_total_1e_minus26_cm2 %r", directional_total)
        add_record(f"volume_{band_name}_unblocked_total_1e_minus26_cm2 %r", unblocked_total)
        add_record(f"volume_{band_name}_four_pi_total_1e_minus26_cm2 %r", four_pi_total)
        add_record(f"volume_{band_name}_r90_R %r", r90_r)
        add_record(f"volume_{band_name}_r99_R %r", r99_r)
        add_record(f"volume_{band_name}_emissivity_unit %r", X_RAY_DIRECTIONAL_EMISSIVITY_UNIT)
        add_record(f"volume_{band_name}_total_unit %r", X_RAY_EMISSION_TOTAL_UNIT)
    xray_summary_npz = output_dir / f"{prefix}.xray_summary.npz"
    np.savez_compressed(
        xray_summary_npz,
        bands=np.asarray(["hard", "rosat", "euv"]),
        directional_total_1e_minus26_cm2=np.asarray(
            [xray_band_stats[name]["directional_total"] for name in ("hard", "rosat", "euv")]
        ),
        unblocked_total_1e_minus26_cm2=np.asarray(
            [xray_band_stats[name]["unblocked_total"] for name in ("hard", "rosat", "euv")]
        ),
        four_pi_total_1e_minus26_cm2=np.asarray(
            [xray_band_stats[name]["four_pi_total"] for name in ("hard", "rosat", "euv")]
        ),
        r90_r=np.asarray([xray_band_stats[name]["r90_r"] for name in ("hard", "rosat", "euv")]),
        r99_r=np.asarray([xray_band_stats[name]["r99_r"] for name in ("hard", "rosat", "euv")]),
        directional_total_unit=np.asarray(X_RAY_EMISSION_TOTAL_UNIT),
        total_emission_unit=np.asarray(X_RAY_EMISSION_TOTAL_UNIT),
        emissivity_unit=np.asarray(X_RAY_DIRECTIONAL_EMISSIVITY_UNIT),
        radius_unit=np.asarray(r"$R_\star$"),
        view_axis=np.asarray(X_RAY_SINGLE_DIRECTION_VIEW_AXIS),
    )
    xray_radial_png = output_dir / f"{prefix}.xray_radial_summary.png"
    plot_xray_radial_summary(xray_radial_png, xray_band_profiles, xray_band_stats)
    xray_units_png = output_dir / f"{prefix}.xray_unit_summary.png"
    plot_xray_unit_summary(xray_units_png, xray_band_stats, view_axis=X_RAY_SINGLE_DIRECTION_VIEW_AXIS)
    add_record("volume_xray_summary_npz %r", str(xray_summary_npz.relative_to(path.parent)))
    add_record("volume_xray_radial_summary_png %r", str(xray_radial_png.relative_to(path.parent)))
    add_record("volume_xray_unit_summary_png %r", str(xray_units_png.relative_to(path.parent)))
    add_record("volume_rho2_los_npz %r", str(los_npz.relative_to(path.parent)))
    add_record("volume_rho2_los_png %r", str(los_png.relative_to(path.parent)))
    add_record("volume_rho2_los_side_npz %r", str(los_side_npz.relative_to(path.parent)))
    add_record("volume_rho2_los_side_png %r", str(los_side_png.relative_to(path.parent)))
    add_record("volume_rho2_los_example_npz %r", str(los_example_npz.relative_to(path.parent)))
    add_record("volume_rho2_los_example_png %r", str(los_example_png.relative_to(path.parent)))
    add_record("volume_rho2_los_image_n %r", image_n)
    add_record("volume_rho2_los_view_axis %r", "+Z")
    add_record("volume_rho2_los_side_view_axis %r", "+X")
    add_record("volume_rho2_los_example_view_axis %r", "+Y")
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
