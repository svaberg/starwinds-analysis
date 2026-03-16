from pathlib import Path

import numpy as np
import pytest

from batwind.analysis.trajectories import circular_orbit_points
from batwind.analysis.trajectories import sample_trajectory
from batwind.analysis.trajectories import trajectory_velocity
from batwind.constants import SOLAR_RADIUS_M
from batwind.physics.curve import relative_ram_pressure_from_trajectory
from batwind.physics.orbits import orbital_period
from batwind.physics.pressure import magnetospheric_standoff_distance
from batwind.physics.pressure import ram_pressure
from batwind.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_4_n00000000.plt")
SUN_MASS_KG = 1.98847e30

def test_magnetospheric_standoff_distance_decreases_with_speed():
    rho = 1e-16
    r1 = magnetospheric_standoff_distance(rho, 1e5)
    r2 = magnetospheric_standoff_distance(rho, 2e5)
    assert np.isfinite(r1)
    assert np.isfinite(r2)
    assert r2 < r1


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_relative_ram_pressure_from_trajectory_runs_on_zero_eccentricity_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    period_s = orbital_period(10.0 * SOLAR_RADIUS_M, SUN_MASS_KG)
    points = circular_orbit_points(10.0, n_points=96)
    phase = np.arange(points.shape[0], dtype=float) / float(points.shape[0])
    time = phase * period_s
    velocity = trajectory_velocity(
        points,
        time,
        coordinate_scale=SOLAR_RADIUS_M,
    )
    trajectory = sample_trajectory(
        sds,
        points,
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
        ),
        time=time,
        velocity_xyz=velocity,
        method="nearest",
    )
    relative_ram_pressure, standoff_distance = relative_ram_pressure_from_trajectory(trajectory)

    for value in (relative_ram_pressure, standoff_distance):
        arr = np.array(value, dtype=float)
        assert arr.shape == (96,)
        assert np.count_nonzero(np.isfinite(arr)) > 0
    assert np.isfinite(np.nanmean(relative_ram_pressure))


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_relative_ram_pressure_from_trajectory_runs_on_circular_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    period_s = orbital_period(10.0 * SOLAR_RADIUS_M, SUN_MASS_KG)
    points = circular_orbit_points(10.0, n_points=96)
    phase = np.arange(points.shape[0], dtype=float) / float(points.shape[0])
    time = phase * period_s
    velocity = trajectory_velocity(
        points,
        time,
        coordinate_scale=SOLAR_RADIUS_M,
    )
    trajectory = sample_trajectory(
        sds,
        points,
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
        ),
        time=time,
        velocity_xyz=velocity,
        method="nearest",
    )
    relative_ram_pressure, standoff_distance = relative_ram_pressure_from_trajectory(trajectory)

    assert np.array(trajectory("t [s]")).shape == (96,)
    assert np.array(trajectory("V_x [m/s]")).shape == (96,)
    assert np.count_nonzero(np.isfinite(relative_ram_pressure)) > 0
    assert np.count_nonzero(np.isfinite(standoff_distance)) > 0
