from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from starwinds_analysis.data.samples import get_sample
from starwinds_analysis.smart_ds import SmartDs
from starwinds_analysis.analysis.shell_magnetic import (
    plot_magnetic_zdi_triplet,
    plot_shell_tangential_vectors_lonlat,
    sample_shell_magnetic_field_map,
)


EXAMPLE_3D = Path(get_sample("3d__var_1_n00000000.plt"))


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
    )
    try:
        assert ax.get_xlim()[0] <= -180
        assert ax.get_xlim()[1] >= 180
        assert "quiver" in extra
    finally:
        plt.close(fig)
