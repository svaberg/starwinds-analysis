from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.analysis.shells import integrate_shell_scalar, sample_spherical_shells
from starwinds_analysis.data.samples import get_sample
from starwinds_analysis.physics.flux_density import radial_advective_flux_density
from starwinds_analysis.physics.magnetic import (
    magnetic_field_unit_scale,
    magnetic_shell_components_from_cartesian,
)
from starwinds_analysis.recipes.spherical import spherical_vector_components
from starwinds_analysis.smart_ds import SmartDs


def _example_3d():
    for name in ("3d__var_1_n00060000.plt", "3d__var_3_n00060000.plt"):
        try:
            return Path(get_sample(name))
        except FileNotFoundError:
            pass
    raise FileNotFoundError("No suitable 3d sample found in sample_data")


EXAMPLE_3D = _example_3d()
SUN_RADIUS_M = 6.957e8


def _sample_shell_magnetic_components(*, n_polar=12, n_azimuth=24):
    sds = SmartDs.from_file(str(EXAMPLE_3D))
    sds.add_batsrus_graph(body_radius_m=SUN_RADIUS_M)
    shell = sample_spherical_shells(
        sds,
        [1.0],
        fields=("B_x [T]", "B_y [T]", "B_z [T]"),
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        length_unit_to_m=SUN_RADIUS_M,
    )
    bx = np.array(shell("B_x [T]"), dtype=float)
    by = np.array(shell("B_y [T]"), dtype=float)
    bz = np.array(shell("B_z [T]"), dtype=float)
    x = np.array(shell("X [R]"), dtype=float)
    y = np.array(shell("Y [R]"), dtype=float)
    z = np.array(shell("Z [R]"), dtype=float)
    comps = magnetic_shell_components_from_cartesian(bx, by, bz, x, y, z)
    return shell, comps


def test_magnetic_shell_components_from_cartesian_shapes():
    shell, comps = _sample_shell_magnetic_components(n_polar=12, n_azimuth=24)

    assert np.array(shell("theta [rad]"), dtype=float).shape == (1, 12, 24)
    assert np.array(shell("phi [rad]"), dtype=float).shape == (1, 12, 24)
    assert comps["B_r [T]"].shape == (1, 12, 24)
    assert comps["B_phi [T]"].shape == (1, 12, 24)
    assert comps["B_meridional [T]"].shape == (1, 12, 24)
    assert comps["B_tangential [T]"].shape == (1, 12, 24)
    assert np.isfinite(comps["B_r [T]"]).any()


def test_magnetic_field_unit_scale_smoke():
    g_scale, g_label = magnetic_field_unit_scale("G")
    t_scale, t_label = magnetic_field_unit_scale("T")
    assert (g_scale, g_label) == (1e4, "G")
    assert (t_scale, t_label) == (1.0, "T")


def test_direct_zdi_style_plots_smoke():
    shell, comps = _sample_shell_magnetic_components(n_polar=10, n_azimuth=20)
    lon_deg = np.degrees(np.array(shell("phi [rad]"), dtype=float)[0])
    lat_deg = 90.0 - np.degrees(np.array(shell("theta [rad]"), dtype=float)[0])
    b_r = np.array(comps["B_r [T]"][0], dtype=float) * 1e4
    b_phi = np.array(comps["B_phi [T]"][0], dtype=float) * 1e4
    b_mer = np.array(comps["B_meridional [T]"][0], dtype=float) * 1e4

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    try:
        for ax, arr in zip(np.ravel(axes), (b_r, b_phi, b_mer)):
            ax.pcolormesh(lon_deg, lat_deg, arr, shading="nearest", cmap="RdBu_r")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
        assert len(np.ravel(axes)) == 3
    finally:
        plt.close(fig)


def test_direct_tangential_vector_plot_smoke():
    shell, comps = _sample_shell_magnetic_components(n_polar=12, n_azimuth=24)
    lon_deg = np.degrees(np.array(shell("phi [rad]"), dtype=float)[0])
    lat_deg = 90.0 - np.degrees(np.array(shell("theta [rad]"), dtype=float)[0])
    b_r = np.array(comps["B_r [T]"][0], dtype=float) * 1e4
    b_phi = np.array(comps["B_phi [T]"][0], dtype=float) * 1e4
    b_mer = np.array(comps["B_meridional [T]"][0], dtype=float) * 1e4
    b_tan = np.array(comps["B_tangential [T]"][0], dtype=float) * 1e4

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        ax.pcolormesh(lon_deg, lat_deg, b_tan, shading="nearest", cmap="viridis")
        i_step, j_step = 2, 3
        lon_q = lon_deg[::i_step, ::j_step]
        lat_q = lat_deg[::i_step, ::j_step]
        u = b_phi[::i_step, ::j_step] / np.cos(np.deg2rad(lat_q))
        v = b_mer[::i_step, ::j_step]
        ax.quiver(lon_q, lat_q, u, v, color="white", angles="xy", scale_units="xy", scale=1.0)
        ax.contour(lon_deg, lat_deg, b_r, levels=[0.0], colors="k", linewidths=0.8)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        assert ax.get_xlim()[0] <= -180
        assert ax.get_xlim()[1] >= 180
    finally:
        plt.close(fig)


def test_shell_magnetic_flux_summary_primitives_smoke():
    shell, comps = _sample_shell_magnetic_components(n_polar=12, n_azimuth=24)
    b_r = np.array(comps["B_r [T]"], dtype=float)
    area = np.array(shell("dA [m^2]"), dtype=float)
    signed_flux, signed_cov = integrate_shell_scalar(b_r, area)
    unsigned_flux, unsigned_cov = integrate_shell_scalar(np.abs(b_r), area)
    assert np.isfinite(signed_flux[0])
    assert np.isfinite(unsigned_flux[0])
    assert signed_cov[0] > 0.0
    assert unsigned_cov[0] > 0.0


def test_direct_shell_mass_flux_lonlat_plot_smoke():
    sds = SmartDs.from_file(str(EXAMPLE_3D))
    sds.add_batsrus_graph(body_radius_m=SUN_RADIUS_M)
    shell = sample_spherical_shells(
        sds,
        [2.0],
        fields=("Rho [kg/m^3]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"),
        n_polar=10,
        n_azimuth=20,
        length_unit_to_m=SUN_RADIUS_M,
    )
    rho = np.array(shell("Rho [kg/m^3]"), dtype=float)
    ux = np.array(shell("U_x [m/s]"), dtype=float)
    uy = np.array(shell("U_y [m/s]"), dtype=float)
    uz = np.array(shell("U_z [m/s]"), dtype=float)
    x = np.array(shell("X [R]"), dtype=float)
    y = np.array(shell("Y [R]"), dtype=float)
    z = np.array(shell("Z [R]"), dtype=float)
    u_r, _u_theta, _u_phi = spherical_vector_components(ux, uy, uz, x, y, z)
    mass_flux = radial_advective_flux_density(rho, u_r)
    lon_deg = np.degrees(np.array(shell("phi [rad]"), dtype=float)[0])
    lat_deg = 90.0 - np.degrees(np.array(shell("theta [rad]"), dtype=float)[0])
    fig, ax = plt.subplots(figsize=(6, 3))
    try:
        img = ax.pcolormesh(
            lon_deg,
            lat_deg,
            np.array(mass_flux[0], dtype=float),
            shading="nearest",
            cmap="viridis",
        )
        fig.colorbar(img, ax=ax)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        assert ax.get_xlim()[0] <= -180
        assert ax.get_xlim()[1] >= 180
    finally:
        plt.close(fig)
