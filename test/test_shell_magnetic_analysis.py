from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.analysis.shells import integrate_shell_scalar
from starwinds_analysis.analysis.shells import sample_spherical_shells
from starwinds_analysis.constants import SOLAR_RADIUS_M
from starwinds_analysis.data.samples import get_sample
from starwinds_analysis.smart_ds import SmartDs


def _example_3d():
    return Path(get_sample("3d__var_4_n00000000.plt"))


EXAMPLE_3D = _example_3d()


def magnetic_field_unit_scale(unit: str) -> tuple[float, str]:
    key = str(unit).strip()
    table = {
        "T": (1.0, "T"),
        "Tesla": (1.0, "T"),
        "G": (1e4, "G"),
        "Gauss": (1e4, "G"),
        "nT": (1e9, "nT"),
    }
    if key not in table:
        raise ValueError(f"Unsupported magnetic display unit '{unit}'")
    return table[key]


def _sample_shell_magnetic_components(*, n_polar=12, n_azimuth=24):
    sds = SmartDs.from_file(str(EXAMPLE_3D))
    sds.add_batsrus_graph(body_radius_m=SOLAR_RADIUS_M)
    shell = sample_spherical_shells(
        sds,
        [1.0],
        fields=("B_x [T]", "B_y [T]", "B_z [T]"),
        n_polar=n_polar,
        n_azimuth=n_azimuth,
        length_unit_to_m=SOLAR_RADIUS_M,
    )
    comps = {
        "B_r [T]": np.array(shell("B_r [T]"), dtype=float),
        "B_p [T]": np.array(shell("B_p [T]"), dtype=float),
        "B_a [T]": np.array(shell("B_a [T]"), dtype=float),
        "B_meridional [T]": np.array(shell("B_meridional [T]"), dtype=float),
        "B_tangential [T]": np.array(shell("B_tangential [T]"), dtype=float),
    }
    return shell, comps


def test_shell_magnetic_component_fields_via_griblet_shapes():
    shell, comps = _sample_shell_magnetic_components(n_polar=12, n_azimuth=24)

    assert np.array(shell("polar [rad]"), dtype=float).shape == (1, 12, 24)
    assert np.array(shell("azimuth [rad]"), dtype=float).shape == (1, 12, 24)
    assert comps["B_r [T]"].shape == (1, 12, 24)
    assert comps["B_a [T]"].shape == (1, 12, 24)
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
    lon_deg = np.degrees(np.array(shell("azimuth [rad]"), dtype=float)[0])
    lat_deg = 90.0 - np.degrees(np.array(shell("polar [rad]"), dtype=float)[0])
    b_r = np.array(comps["B_r [T]"][0], dtype=float) * 1e4
    b_a = np.array(comps["B_a [T]"][0], dtype=float) * 1e4
    b_mer = np.array(comps["B_meridional [T]"][0], dtype=float) * 1e4

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    try:
        for ax, arr in zip(np.ravel(axes), (b_r, b_a, b_mer)):
            ax.pcolormesh(lon_deg, lat_deg, arr, shading="nearest", cmap="RdBu_r")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
        assert len(np.ravel(axes)) == 3
    finally:
        plt.close(fig)


def test_direct_tangential_vector_plot_smoke():
    shell, comps = _sample_shell_magnetic_components(n_polar=12, n_azimuth=24)
    lon_deg = np.degrees(np.array(shell("azimuth [rad]"), dtype=float)[0])
    lat_deg = 90.0 - np.degrees(np.array(shell("polar [rad]"), dtype=float)[0])
    b_r = np.array(comps["B_r [T]"][0], dtype=float) * 1e4
    b_a = np.array(comps["B_a [T]"][0], dtype=float) * 1e4
    b_mer = np.array(comps["B_meridional [T]"][0], dtype=float) * 1e4
    b_tan = np.array(comps["B_tangential [T]"][0], dtype=float) * 1e4

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        ax.pcolormesh(lon_deg, lat_deg, b_tan, shading="nearest", cmap="viridis")
        i_step, j_step = 2, 3
        lon_q = lon_deg[::i_step, ::j_step]
        lat_q = lat_deg[::i_step, ::j_step]
        u = b_a[::i_step, ::j_step] / np.cos(np.deg2rad(lat_q))
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
    sds.add_batsrus_graph(body_radius_m=SOLAR_RADIUS_M)
    shell = sample_spherical_shells(
        sds,
        [2.0],
        fields=("Rho [kg/m^3]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"),
        n_polar=10,
        n_azimuth=20,
        length_unit_to_m=SOLAR_RADIUS_M,
    )
    rho = np.array(shell("Rho [kg/m^3]"), dtype=float)
    mass_flux = rho * np.array(shell("U_r [m/s]"), dtype=float)
    lon_deg = np.degrees(np.array(shell("azimuth [rad]"), dtype=float)[0])
    lat_deg = 90.0 - np.degrees(np.array(shell("polar [rad]"), dtype=float)[0])
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
