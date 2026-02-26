from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.data.samples import get_sample
from starwinds_analysis.smart_ds import SmartDs
from starwinds_analysis.analysis.mass_loss import plot_shell_mass_flux_lonlat
from starwinds_analysis.physics.mass_loss import sample_shell_mass_flux_map
from starwinds_analysis.analysis.shell_magnetic import (
    plot_magnetic_zdi_triplet,
    plot_shell_tangential_vectors_lonlat,
    sample_shell_magnetic_field_map,
    summarize_shell_magnetic_field_map,
)


def _example_3d():
    for name in ("3d__var_1_n00060000.plt", "3d__var_1_n00060000.plt"):
        try:
            return Path(get_sample(name))
        except FileNotFoundError:
            pass
    raise FileNotFoundError("No suitable 3d__var_1 sample found in sample_data")


EXAMPLE_3D = _example_3d()
SUN_RADIUS_M = 6.957e8


def test_sample_shell_magnetic_field_map_shapes():
    sds = SmartDs.from_file(str(EXAMPLE_3D))
    shell_map = sample_shell_magnetic_field_map(sds, 1.0, n_polar=12, n_azimuth=24)

    assert shell_map.lon_deg.shape == (12, 24)
    assert shell_map.lat_deg.shape == (12, 24)
    assert shell_map.b_r_T.shape == (12, 24)
    assert shell_map.b_phi_T.shape == (12, 24)
    assert shell_map.b_meridional_T.shape == (12, 24)
    assert shell_map.b_tangential_T.shape == (12, 24)
    assert np.isfinite(shell_map.component("radial", unit="G")).any()


def test_plot_magnetic_zdi_triplet_smoke():
    sds = SmartDs.from_file(str(EXAMPLE_3D))
    shell_map = sample_shell_magnetic_field_map(sds, 1.0, n_polar=10, n_azimuth=20)

    fig, axes = plot_magnetic_zdi_triplet(shell_map, unit="G", figsize=(8, 8))
    try:
        assert len(np.ravel(axes)) == 3
        for ax in np.ravel(axes):
            assert ax.get_xlim()[0] <= -180
            assert ax.get_xlim()[1] >= 180
    finally:
        plt.close(fig)


def test_plot_shell_tangential_vectors_lonlat_smoke():
    sds = SmartDs.from_file(str(EXAMPLE_3D))
    shell_map = sample_shell_magnetic_field_map(sds, 1.0, n_polar=12, n_azimuth=24)

    fig, ax, extra = plot_shell_tangential_vectors_lonlat(
        shell_map,
        unit="G",
        background_scale="positive_log",
        arrow_stride=(2, 3),
        overlay_radial_zero_contour=True,
    )
    try:
        assert ax.get_xlim()[0] <= -180
        assert ax.get_xlim()[1] >= 180
        assert "quiver" in extra
        assert "radial_zero_contour" in extra
    finally:
        plt.close(fig)


def test_shell_magnetic_summary_smoke():
    sds = SmartDs.from_file(str(EXAMPLE_3D))
    shell_map = sample_shell_magnetic_field_map(sds, 1.0, n_polar=12, n_azimuth=24)
    summary = summarize_shell_magnetic_field_map(shell_map, unit="G")
    assert summary["finite_B_r_cells"] > 0
    assert summary["total_cells"] == 12 * 24
    assert np.isfinite(summary["signed_radial_flux [Wb]"])
    assert "rms_B_r [G]" in summary


def test_plot_shell_mass_flux_lonlat_smoke():
    sds = SmartDs.from_file(str(EXAMPLE_3D))
    shell_map = sample_shell_mass_flux_map(
        sds,
        2.0,
        body_radius_m=SUN_RADIUS_M,
        n_polar=10,
        n_azimuth=20,
    )
    fig, ax, extra = plot_shell_mass_flux_lonlat(shell_map, scale="log", figsize=(6, 3))
    try:
        assert ax.get_xlim()[0] <= -180
        assert ax.get_xlim()[1] >= 180
        assert "n_nonpositive" in extra
    finally:
        plt.close(fig)
