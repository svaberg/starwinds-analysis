from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from starwinds_readplt.dataset import Dataset

from starwinds_analysis.quicklook2d import plot_slice_quicklook, quicklook_shell_figure
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")
SUN_RADIUS_M = 6.957e8


def make_slice_dataset():
    # 2x2 quad (XZ plane, Y=0) with enough fields for quicklook overlays.
    variables = [
        "X [R]",
        "Y [R]",
        "Z [R]",
        "Rho [g/cm^3]",
        "U_x [km/s]",
        "U_y [km/s]",
        "U_z [km/s]",
        "B_x [Gauss]",
        "B_y [Gauss]",
        "B_z [Gauss]",
        "P [dyne/cm^2]",
    ]
    points = np.array(
        [
            [-1.0, 0.0, -1.0, 1e-16, 300.0, 20.0, 50.0, 5.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, -1.0, 2e-16, 350.0, 10.0, 20.0, -5.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 3e-16, 400.0, -5.0, -20.0, -5.0, 0.0, -1.0, 3.0],
            [-1.0, 0.0, 1.0, 4e-16, 450.0, 0.0, -50.0, 5.0, 0.0, -1.0, 4.0],
        ],
        dtype=float,
    )
    corners = np.array([[0, 1, 2, 3]], dtype=int)
    return Dataset(
        points,
        corners,
        aux={"GAMMA": "1.666667"},
        title="slice-demo",
        variables=variables,
        zone="slice",
    )


def test_plot_slice_quicklook_with_preset_and_overlays():
    sds = SmartDs(make_slice_dataset())
    sds.add_spherical_fields(vectors=("B", "U"))
    sds.add_batsrus_graph()

    fig, axes, cbar = plot_slice_quicklook(sds, preset="b_r", style="marginals")
    assert fig is not None
    assert len(axes) == 3
    assert cbar is not None
    plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_quicklook_shell_figure_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    fig, axs, diagnostics = quicklook_shell_figure(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        body_radius_m=SUN_RADIUS_M,
        n_polar=12,
        n_azimuth=24,
        method="nearest",
    )

    assert fig is not None
    assert np.asarray(axs).shape == (2, 2)
    assert "mass_loss" in diagnostics
    assert "torque" in diagnostics
    assert "open_flux" in diagnostics
    assert "energy" in diagnostics
    plt.close(fig)

