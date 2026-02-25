from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from starwinds_readplt.dataset import Dataset

from starwinds_analysis.quicklook2d import (
    orbit_local_comparison_figure,
    plot_radius_quicklook,
    plot_slice_quicklook,
    quicklook_shell_figure,
    run_quicklook2d,
    save_quicklook2d_bundle,
)
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


def test_plot_radius_quicklook_modes():
    sds = SmartDs(make_slice_dataset())
    sds.add_batsrus_graph()

    fields = ("Rho [g/cm^3]", "U_x [km/s]", "B_x [Gauss]", "P [dyne/cm^2]")
    figs = []
    for mode in ("binned", "scatter", "cdf"):
        fig, axes = plot_radius_quicklook(sds, fields=fields, mode=mode, ncols=2)
        assert fig is not None
        assert len(axes) == 4
        figs.append(fig)

    for fig in figs:
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


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_orbit_local_comparison_figure_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    fig, axs, out = orbit_local_comparison_figure(
        sds,
        10.0,
        body_radius_m=SUN_RADIUS_M,
        n_points=96,
        shell_n_polar=12,
        shell_n_azimuth=24,
        method="nearest",
    )
    assert fig is not None
    assert np.asarray(axs).shape == (2,)
    assert "mass_loss" in out and "torque" in out
    plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_save_quicklook2d_bundle_writes_figures_and_summaries(tmp_path):
    sds = SmartDs.from_file(str(EXAMPLE_PLT))

    shell_fig, _axs, diagnostics = quicklook_shell_figure(
        sds,
        [2.0, 4.0, 8.0],
        body_radius_m=SUN_RADIUS_M,
        n_polar=12,
        n_azimuth=24,
        method="nearest",
    )
    radius_fig, _axes = plot_radius_quicklook(
        sds,
        fields=("Rho [g/cm^3]", "U_x [km/s]", "B_x [Gauss]", "P [dyne/cm^2]"),
        mode="binned",
        ncols=2,
    )

    saved = save_quicklook2d_bundle(
        tmp_path,
        shell_fig=shell_fig,
        diagnostics=diagnostics,
        radius_figures={"binned": radius_fig},
        prefix="demo",
        band_radius_range=(2.0, 8.0),
        star_mass_kg=1.98847e30,
        star_radius_m=SUN_RADIUS_M,
    )

    shell_png = tmp_path / "demo.shells.png"
    json_path = tmp_path / "demo.shells.json"
    npz_path = tmp_path / "demo.shells.npz"
    radius_png = tmp_path / "demo.radius.binned.png"

    assert shell_png.exists()
    assert json_path.exists()
    assert npz_path.exists()
    assert radius_png.exists()
    assert "figures" in saved and "files" in saved

    payload = json.loads(json_path.read_text())
    assert "mass_loss" in payload
    assert "torque" in payload
    assert "_band_summary" in payload
    assert "mass_loss" in payload["_band_summary"]
    assert "_wind_scaling" in payload

    with np.load(npz_path) as data:
        keys = set(data.files)
    assert any(k.startswith("mass_loss__") for k in keys)
    assert any(k.startswith("torque__") for k in keys)

    plt.close(shell_fig)
    plt.close(radius_fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_run_quicklook2d_end_to_end_writes_bundle(tmp_path):
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = run_quicklook2d(
        sds,
        body_radius_m=SUN_RADIUS_M,
        radii=[2.0, 4.0, 8.0],
        slice_presets=("rho", "b_r"),
        slice_grid={"nx": 32, "nz": 24, "method": "nearest", "symmetric_ranges": True},
        radius_modes=("binned",),
        orbit_radii=(10.0,),
        orbit_n_points=96,
        n_polar=12,
        n_azimuth=24,
        method="nearest",
        output_dir=tmp_path,
        prefix="e2e",
    )

    assert "diagnostics" in out
    assert "saved" in out
    assert "slice_figures" in out and len(out["slice_figures"]) == 2
    assert "radius_figures" in out and "binned" in out["radius_figures"]
    assert "orbit_figures" in out and len(out["orbit_figures"]) == 1

    assert (tmp_path / "e2e.shells.png").exists()
    assert (tmp_path / "e2e.shells.json").exists()
    assert (tmp_path / "e2e.shells.npz").exists()
    assert (tmp_path / "e2e.slices.rho.png").exists()
    assert (tmp_path / "e2e.slices.b_r.png").exists()
    assert (tmp_path / "e2e.radius.binned.png").exists()
    assert any(p.name.startswith("e2e.orbits.") and p.suffix == ".png" for p in tmp_path.iterdir())

    for fig in out["slice_figures"].values():
        plt.close(fig)
    plt.close(out["shell_figure"])
    for fig in out["radius_figures"].values():
        plt.close(fig)
    for fig in out["orbit_figures"].values():
        plt.close(fig)
