from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.pipelines.orbit_surface import (
    pressure_components_on_orbit_surface,
    sample_orbit_surface_revolution,
    surface_of_revolution_from_path,
    surface_point_normals_and_areas,
    torque_components_on_orbit_surface,
)
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00060000.plt")
SUN_RADIUS_M = 6.957e8
SUN_MASS_KG = 1.98847e30


def test_surface_of_revolution_from_path_preserves_cyl_radius_and_z():
    path = np.array(
        [
            [2.0, 0.0, -1.0],
            [0.0, 3.0, 0.5],
            [-4.0, 0.0, 1.5],
        ],
        dtype=float,
    )
    out = surface_of_revolution_from_path(path, n_longitudes=16)
    pts = out["points"]
    assert pts.shape == (3, 16, 3)
    cyl = np.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
    np.testing.assert_allclose(cyl[:, 0], np.sqrt(path[:, 0] ** 2 + path[:, 1] ** 2))
    np.testing.assert_allclose(pts[:, :, 2], np.repeat(path[:, 2][:, None], 16, axis=1))


def test_surface_point_normals_and_areas_on_cylinder_like_surface_are_finite():
    path = np.column_stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
            np.linspace(-1.0, 1.0, 16, endpoint=False),
        ]
    )
    surf = surface_of_revolution_from_path(path, n_longitudes=24)
    normals, area = surface_point_normals_and_areas(surf["points"])
    assert normals.shape == surf["points"].shape
    assert area.shape == surf["points"].shape[:2]
    assert np.count_nonzero(np.isfinite(area)) > 0
    nmag = np.sqrt(np.sum(normals * normals, axis=-1))
    finite = np.isfinite(nmag)
    assert np.allclose(nmag[finite], 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_sample_orbit_surface_revolution_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = sample_orbit_surface_revolution(
        sds,
        fields=("Rho [g/cm^3]", "U_x [km/s]", "B_x [Gauss]"),
        orbit={"semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 64},
        n_longitudes=32,
        method="nearest",
    )
    assert out["Rho [g/cm^3]"].shape == (64, 32)
    assert out["U_x [km/s]"].shape == (64, 32)
    assert out["phase [turns]"].shape == (64,)
    assert out["time_weight [none]"].shape == (64,)
    assert np.isclose(np.sum(out["time_weight [none]"]), 1.0)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_pressure_components_on_orbit_surface_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = pressure_components_on_orbit_surface(
        sds,
        {"semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 64},
        body_radius_m=SUN_RADIUS_M,
        n_longitudes=48,
        method="nearest",
        star_mass_kg=SUN_MASS_KG,
    )
    for key in (
        "ram_pressure [Pa]",
        "magnetic_pressure [Pa]",
        "thermal_pressure [Pa]",
        "standoff_distance [m]",
    ):
        arr = np.asarray(out[key], dtype=float)
        assert arr.shape == (64, 48)
        assert np.count_nonzero(np.isfinite(arr)) > 0
        assert key in out["summary"]
        assert key in out["phase_quantiles"]
    assert "relative_ram_pressure [Pa]" in out
    assert out["phase_quantiles"]["ram_pressure [Pa]"]["values"].shape[0] == 64


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_torque_components_on_orbit_surface_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = torque_components_on_orbit_surface(
        sds,
        {"semi_major_axis": 10.0, "eccentricity": 0.2, "n_points": 64},
        body_radius_m=SUN_RADIUS_M,
        n_longitudes=48,
        method="nearest",
        angvel_rad_s=0.0,
    )
    for key in (
        "T1_magnetic [Nm]",
        "T2_pressure [Nm]",
        "T3_corotation [Nm]",
        "T4_dynamic [Nm]",
        "total [Nm]",
        "coverage [none]",
    ):
        arr = np.asarray(out[key], dtype=float)
        assert arr.shape == ()
        assert np.isfinite(arr)
    assert np.isclose(out["total [Nm]"], out["T1_magnetic [Nm]"] + out["T2_pressure [Nm]"] + out["T3_corotation [Nm]"] + out["T4_dynamic [Nm]"])
    assert np.count_nonzero(np.isfinite(np.asarray(out["surface_area [m^2]"]))) > 0
    assert "phase_integrals" in out and "phase_quantiles" in out
    assert out["phase_integrals"]["total"]["integral [Nm]"].shape[0] == 64
    assert out["phase_quantiles"]["total"]["values [N/m]"].shape[0] == 64
