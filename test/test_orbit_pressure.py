from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.analysis.orbit_pressure import (
    pressure_components_on_circular_orbit,
    pressure_components_on_elliptic_orbit,
)
from starwinds_analysis.physics.pressure import (
    magnetic_pressure,
    magnetospheric_standoff_distance,
    pressure_components,
    ram_pressure,
)
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00060000.plt")
SUN_RADIUS_M = 6.957e8
SUN_MASS_KG = 1.98847e30


def test_pressure_components_static_and_relative_ram():
    rho = np.array([1.0, 2.0])
    u = np.array([[3.0, 0.0, 4.0], [0.0, 5.0, 0.0]])
    b = np.array([[1e-4, 0.0, 0.0], [0.0, 2e-4, 0.0]])
    p = np.array([1.5, 2.5])
    v_obj = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])

    out = pressure_components(
        rho, u, b, thermal_pressure_pa=p, object_velocity_xyz_m_s=v_obj
    )

    np.testing.assert_allclose(out["U [m/s]"], [5.0, 5.0])
    np.testing.assert_allclose(out["ram_pressure [Pa]"], ram_pressure(rho, [5.0, 5.0]))
    np.testing.assert_allclose(
        out["magnetic_pressure [Pa]"], magnetic_pressure(np.array([1e-4, 2e-4]))
    )
    np.testing.assert_allclose(out["thermal_pressure [Pa]"], p)
    assert np.all(out["relative_ram_pressure [Pa]"] >= 0)
    assert out["relative_ram_pressure [Pa]"][1] < out["ram_pressure [Pa]"][1]


def test_magnetospheric_standoff_distance_decreases_with_speed():
    rho = 1e-16
    r1 = magnetospheric_standoff_distance(rho, 1e5)
    r2 = magnetospheric_standoff_distance(rho, 2e5)
    assert np.isfinite(r1)
    assert np.isfinite(r2)
    assert r2 < r1


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_pressure_components_on_circular_orbit_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = pressure_components_on_circular_orbit(
        sds,
        10.0,
        body_radius_m=SUN_RADIUS_M,
        n_points=96,
        method="nearest",
        star_mass_kg=SUN_MASS_KG,
    )

    for key in (
        "magnetic_pressure [Pa]",
        "ram_pressure [Pa]",
        "thermal_pressure [Pa]",
        "standoff_distance [m]",
    ):
        arr = np.asarray(out[key], dtype=float)
        assert arr.shape == (96,)
        assert np.count_nonzero(np.isfinite(arr)) > 0

    assert "relative_ram_pressure [Pa]" in out
    assert "summary" in out
    assert "ram_pressure [Pa]" in out["summary"]
    assert np.isfinite(out["summary"]["ram_pressure [Pa]"]["mean"])


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_pressure_components_on_elliptic_orbit_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = pressure_components_on_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.2,
        body_radius_m=SUN_RADIUS_M,
        n_points=96,
        method="nearest",
        star_mass_kg=SUN_MASS_KG,
    )

    assert np.isclose(out["semi_major_axis [R]"], 10.0)
    assert np.isclose(out["eccentricity [none]"], 0.2)
    assert "relative_ram_pressure [Pa]" in out
    assert "object_speed [m/s]" in out
    assert out["orbit_samples"]["time_weight [none]"].shape == (96,)
    assert np.isclose(np.sum(out["orbit_samples"]["time_weight [none]"]), 1.0)
    assert np.count_nonzero(np.isfinite(out["relative_ram_pressure [Pa]"])) > 0
