from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.analysis.orbits import sample_elliptic_orbit
from starwinds_analysis.constants import SOLAR_RADIUS_M
from starwinds_analysis.physics.orbit_pressure import pressure_components_from_curve
from starwinds_analysis.physics.orbits import orbital_period
from starwinds_analysis.physics.pressure import magnetospheric_standoff_distance
from starwinds_analysis.physics.pressure import ram_pressure
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00060000.plt")
SUN_MASS_KG = 1.98847e30

def test_magnetospheric_standoff_distance_decreases_with_speed():
    rho = 1e-16
    r1 = magnetospheric_standoff_distance(rho, 1e5)
    r2 = magnetospheric_standoff_distance(rho, 2e5)
    assert np.isfinite(r1)
    assert np.isfinite(r2)
    assert r2 < r1


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_pressure_components_from_curve_runs_on_circular_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius_m=SOLAR_RADIUS_M)
    curve = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.0,
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
            "U [m/s]",
            "B [T]",
            "magnetic_pressure [Pa]",
            "ram_pressure [Pa]",
            "thermal_pressure [Pa]",
            "standoff_distance [m]",
        ),
        n_points=96,
        method="nearest",
    )
    out = pressure_components_from_curve(
        curve,
        body_radius_m=SOLAR_RADIUS_M,
        period_s=orbital_period(10.0 * SOLAR_RADIUS_M, SUN_MASS_KG),
        include_relative_ram=True,
    )

    for key in (
        "magnetic_pressure [Pa]",
        "ram_pressure [Pa]",
        "thermal_pressure [Pa]",
        "standoff_distance [m]",
    ):
        arr = np.array(out[key], dtype=float)
        assert arr.shape == (96,)
        assert np.count_nonzero(np.isfinite(arr)) > 0

    assert "relative_ram_pressure [Pa]" in out
    assert "summary" in out
    assert "ram_pressure [Pa]" in out["summary"]
    assert np.isfinite(out["summary"]["ram_pressure [Pa]"]["mean"])


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_pressure_components_from_curve_runs_on_elliptic_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius_m=SOLAR_RADIUS_M)
    curve = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.2,
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
            "U [m/s]",
            "B [T]",
            "magnetic_pressure [Pa]",
            "ram_pressure [Pa]",
            "thermal_pressure [Pa]",
            "standoff_distance [m]",
        ),
        n_points=96,
        method="nearest",
    )
    out = pressure_components_from_curve(
        curve,
        body_radius_m=SOLAR_RADIUS_M,
        period_s=orbital_period(10.0 * SOLAR_RADIUS_M, SUN_MASS_KG),
        include_relative_ram=True,
    )

    assert "relative_ram_pressure [Pa]" in out
    assert "V [m/s]" in out
    assert "U_minus_V [m/s]" in out
    assert np.array(out["curve_samples"]("time_weight [none]")).shape == (96,)
    assert np.isclose(np.sum(out["curve_samples"]("time_weight [none]")), 1.0)
    assert np.count_nonzero(np.isfinite(out["relative_ram_pressure [Pa]"])) > 0
