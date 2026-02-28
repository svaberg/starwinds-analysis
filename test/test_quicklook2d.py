from pathlib import Path
import importlib.util
import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from starwinds_readplt.dataset import Dataset

from starwinds_analysis.pipelines.slice import process_plt_file
from starwinds_analysis.pipelines.volume import orbit_local_comparison_figure
from starwinds_analysis.pipelines.volume import orbit_pressure_figure
from starwinds_analysis.pipelines.volume import orbit_surface_pressure_figure
from starwinds_analysis.pipelines.volume import orbit_surface_torque_figure
from starwinds_analysis.pipelines.volume import plot_radius_quicklook
from starwinds_analysis.pipelines.volume import plot_slice_quicklook
from starwinds_analysis.pipelines.volume import quicklook_shell_figure
from starwinds_analysis.pipelines.volume import run_quicklook2d
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00060000.plt")
SUN_RADIUS_M = 6.957e8
SLICE_PLOTTING_AVAILABLE = importlib.util.find_spec("starwinds_analysis.visualisation.slice") is not None


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


def test_quicklook_presets_work_for_si_and_raw_paths():
    sds = SmartDs(make_slice_dataset())
    sds.add_spherical_fields(vectors=("B", "U"))
    sds.add_batsrus_graph()

    fig, axes, cbar = plot_slice_quicklook(sds, preset="b_r", style="marginals")
    assert fig is not None
    assert len(axes) == 3
    assert cbar is not None
    plt.close(fig)

    fig, axes, cbar = plot_slice_quicklook(sds, preset="b_r_raw", style="marginals")
    assert fig is not None
    assert len(axes) == 3
    assert cbar is not None
    plt.close(fig)

    fig, axes = plot_radius_quicklook(sds, preset="wind_raw", mode="binned", ncols=2)
    assert fig is not None
    assert len(axes) == 4
    plt.close(fig)


@pytest.mark.skipif(not SLICE_PLOTTING_AVAILABLE, reason="slice plotting module not available on this branch")
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
    for mode in ("binned", "scatter", "cdf", "hist2d"):
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
        star_mass_kg=1.98847e30,
    )

    assert fig is not None
    assert np.array(axs).shape == (2, 2)
    assert "mass_loss" in diagnostics
    assert "torque" in diagnostics
    assert "open_flux" in diagnostics
    assert "energy" in diagnostics
    assert "wind_scaling" in diagnostics
    assert "Upsilon_open [none]" in diagnostics["wind_scaling"]
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
    assert np.array(axs).shape == (2,)
    assert "mass_loss" in out and "torque" in out
    plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_orbit_local_comparison_figure_accepts_kepler_spec():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    fig, axs, out = orbit_local_comparison_figure(
        sds,
        {"semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 96, "shell_n_radii": 8},
        body_radius_m=SUN_RADIUS_M,
        shell_n_polar=12,
        shell_n_azimuth=24,
        method="nearest",
    )
    assert fig is not None
    assert np.array(axs).shape == (2,)
    assert "mass_loss" in out and "torque" in out
    assert "semi_major_axis [R]" in out["mass_loss"]
    assert "shell_mass_loss_interp [kg/s]" in out["mass_loss"]
    assert "shell_total_torque_interp [Nm]" in out["torque"]
    plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_orbit_pressure_figure_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    fig, axs, out = orbit_pressure_figure(
        sds,
        10.0,
        body_radius_m=SUN_RADIUS_M,
        n_points=96,
        method="nearest",
        star_mass_kg=1.98847e30,
    )
    assert fig is not None
    assert np.array(axs).shape == (2,)
    assert "ram_pressure [Pa]" in out
    assert "standoff_distance [m]" in out
    plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_orbit_pressure_figure_accepts_kepler_spec():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    fig, axs, out = orbit_pressure_figure(
        sds,
        {"semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 96},
        body_radius_m=SUN_RADIUS_M,
        method="nearest",
        star_mass_kg=1.98847e30,
    )
    assert fig is not None
    assert np.array(axs).shape == (2,)
    assert "relative_ram_pressure [Pa]" in out
    assert "semi_major_axis [R]" in out
    plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_orbit_surface_pressure_figure_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    fig, axs, out = orbit_surface_pressure_figure(
        sds,
        {"semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 64},
        body_radius_m=SUN_RADIUS_M,
        n_longitudes=48,
        method="nearest",
        star_mass_kg=1.98847e30,
    )
    assert fig is not None
    assert np.array(axs).shape == (2,)
    assert "phase_quantiles" in out
    assert "ram_pressure [Pa]" in out["phase_quantiles"]
    assert "standoff_distance [m]" in out
    plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_orbit_surface_torque_figure_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    fig, axs, out = orbit_surface_torque_figure(
        sds,
        {"semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 64},
        body_radius_m=SUN_RADIUS_M,
        n_longitudes=48,
        method="nearest",
        angvel_rad_s=0.0,
    )
    assert fig is not None
    assert np.array(axs).shape == (2,)
    assert "phase_integrals" in out
    assert "total" in out["phase_integrals"]
    assert "total [Nm]" in out
    plt.close(fig)


def test_process_plt_file_runs_per_file_quicklook_and_closes_figures(tmp_path, monkeypatch):
    file_path = tmp_path / "alpha.plt"
    file_path.write_text("")
    from_file_calls: list[Path] = []
    sentinel_ds = type("Fake2DSmartDs", (), {"corners": np.zeros((1, 4), dtype=int)})()

    class FakeSmartDs:
        @classmethod
        def from_file(cls, path):
            from_file_calls.append(Path(path))
            return sentinel_ds

    created_figs: list[plt.Figure] = []

    plot_calls: list[str] = []

    def fake_plot_xz(_smart_ds, *, var=None, **_kwargs):
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0], ",")
        created_figs.append(fig)
        plot_calls.append(str(var))
        return fig, (ax,), None

    monkeypatch.setattr("starwinds_analysis.pipelines.slice.SmartDs", FakeSmartDs)
    monkeypatch.setattr(
        "starwinds_analysis.pipelines.slice.plot_xz_slice_tripcolor_with_cross_quantiles",
        fake_plot_xz,
    )
    process_plt_file(file_path)

    assert from_file_calls == [file_path]
    assert plot_calls == ["Rho [kg/m^3]", "U [m/s]", "B [T]"]
    assert (tmp_path / "slice" / "alpha.slices.rho.png").exists()
    assert (tmp_path / "slice" / "alpha.slices.u.png").exists()
    assert (tmp_path / "slice" / "alpha.slices.b.png").exists()
    assert all(not plt.fignum_exists(fig.number) for fig in created_figs)


def test_process_plt_file_skips_3d_inputs_by_default(tmp_path, monkeypatch, caplog):
    file_path = tmp_path / "alpha.plt"
    file_path.write_text("")

    class Fake3DDataset:
        corners = np.zeros((1, 8), dtype=int)

    class FakeSmartDs:
        @classmethod
        def from_file(cls, _path):
            return Fake3DDataset()

    monkeypatch.setattr("starwinds_analysis.pipelines.slice.SmartDs", FakeSmartDs)
    with caplog.at_level(logging.INFO, logger="starwinds_analysis.pipelines.slice"):
        process_plt_file(file_path)

    messages = [record.getMessage() for record in caplog.records]
    assert any("skip file=alpha.plt reason=3d_input" in message for message in messages)


def test_process_plt_file_can_force_3d_inputs(tmp_path, monkeypatch):
    file_path = tmp_path / "alpha.plt"
    file_path.write_text("")
    calls: list[str] = []

    class Fake3DDataset:
        corners = np.zeros((1, 8), dtype=int)

    class FakeSmartDs:
        @classmethod
        def from_file(cls, _path):
            return Fake3DDataset()

    def fake_plot_xz(_smart_ds, *, var=None, **_kwargs):
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0], ",")
        calls.append(str(var))
        return fig, (ax,), None

    monkeypatch.setattr("starwinds_analysis.pipelines.slice.SmartDs", FakeSmartDs)
    monkeypatch.setattr(
        "starwinds_analysis.pipelines.slice.plot_xz_slice_tripcolor_with_cross_quantiles",
        fake_plot_xz,
    )
    process_plt_file(file_path, force_3d=True)

    assert calls == ["Rho [kg/m^3]", "U [m/s]", "B [T]"]
    assert (tmp_path / "slice" / "alpha.slices.rho.png").exists()


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
@pytest.mark.skipif(not SLICE_PLOTTING_AVAILABLE, reason="slice plotting module not available on this branch")
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
        star_mass_kg=1.98847e30,
    )

    assert "shell_diagnostics" in out
    assert "saved" in out
    assert "slice_figures" in out and len(out["slice_figures"]) == 2
    assert "radius_figures" in out and "binned" in out["radius_figures"]
    assert "orbit_figures" in out and len(out["orbit_figures"]) == 1
    assert "wind_scaling" in out["shell_diagnostics"]

    assert (tmp_path / "e2e.shells.png").exists()
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


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_run_quicklook2d_supports_kepler_orbit_specs(tmp_path):
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = run_quicklook2d(
        sds,
        body_radius_m=SUN_RADIUS_M,
        radii=[4.0, 8.0],
        slice_presets=(),
        radius_modes=(),
        orbit_specs=(
            {"label": "ecc_orbit", "semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 96, "shell_n_radii": 8},
        ),
        n_polar=12,
        n_azimuth=24,
        method="nearest",
        output_dir=tmp_path,
        prefix="ecc",
    )
    assert "ecc_orbit" in out["orbit_figures"]
    assert "ecc_orbit" in out["orbit_results"]
    assert (tmp_path / "ecc.orbits.ecc_orbit.png").exists()
    plt.close(out["shell_figure"])
    for fig in out["orbit_figures"].values():
        plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_run_quicklook2d_supports_orbit_surface_specs_and_exports(tmp_path):
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = run_quicklook2d(
        sds,
        body_radius_m=SUN_RADIUS_M,
        radii=[4.0, 8.0],
        slice_presets=(),
        radius_modes=(),
        orbit_surface_specs=(
            {"label": "orbittube", "semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 48},
        ),
        orbit_surface_modes=("pressure", "torque"),
        orbit_surface_n_longitudes=32,
        n_polar=12,
        n_azimuth=24,
        method="nearest",
        output_dir=tmp_path,
        prefix="surface",
        star_mass_kg=1.98847e30,
    )

    assert any(k.endswith("_surface_pressure") for k in out["orbit_figures"])
    assert any(k.endswith("_surface_torque") for k in out["orbit_figures"])
    assert "orbittube" in out["orbit_results"]
    assert "surface_pressure" in out["orbit_results"]["orbittube"]
    assert "surface_torque" in out["orbit_results"]["orbittube"]

    assert any(p.name.startswith("surface.orbits.orbittube_surface_pressure") for p in tmp_path.iterdir())
    assert any(p.name.startswith("surface.orbits.orbittube_surface_torque") for p in tmp_path.iterdir())

    plt.close(out["shell_figure"])
    for fig in out["orbit_figures"].values():
        plt.close(fig)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_run_quicklook2d_supports_named_planet_orbit_surface(tmp_path):
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = run_quicklook2d(
        sds,
        body_radius_m=SUN_RADIUS_M,
        radii=[4.0, 8.0],
        slice_presets=(),
        radius_modes=(),
        orbit_surface_planets=("Earth",),
        orbit_surface_modes=("pressure",),
        orbit_surface_n_longitudes=24,
        orbit_n_points=48,
        n_polar=12,
        n_azimuth=24,
        method="nearest",
        output_dir=tmp_path,
        prefix="planet",
        star_mass_kg=1.98847e30,
    )
    assert "Earth" in out["orbit_results"]
    assert "surface_pressure" in out["orbit_results"]["Earth"]
    assert any("Earth_surface_pressure" in k for k in out["orbit_figures"])
    plt.close(out["shell_figure"])
    for fig in out["orbit_figures"].values():
        plt.close(fig)
