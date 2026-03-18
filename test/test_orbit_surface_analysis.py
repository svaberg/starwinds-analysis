from pathlib import Path

import numpy as np

from batread.dataset import Dataset

from batwind.analysis.trajectories import circular_orbit_points
from batwind.analysis.trajectories import trajectory_velocity
from batwind.constants import SOLAR_RADIUS_M
from batwind.physics.orbits import orbital_period
from batwind.physics.orbit_surface import pressure_components_on_surface
from batwind.physics.orbit_surface import sample_surface_revolution
from batwind.physics.orbit_surface import surface_of_revolution_from_trajectory
from batwind.physics.orbit_surface import surface_point_normals_and_areas
from batwind.physics.orbit_surface import torque_components_on_surface
from batwind.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")
SUN_MASS_KG = 1.98847e30


def test_surface_of_revolution_from_trajectory_preserves_cyl_radius_and_z():
    path = np.array(
        [
            [2.0, 0.0, -1.0],
            [0.0, 3.0, 0.5],
            [-4.0, 0.0, 1.5],
        ],
        dtype=float,
    )
    out = surface_of_revolution_from_trajectory(path, n_longitudes=16)
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
    surf = surface_of_revolution_from_trajectory(path, n_longitudes=24)
    normals, area = surface_point_normals_and_areas(surf["points"])
    assert normals.shape == surf["points"].shape
    assert area.shape == surf["points"].shape[:2]
    assert np.count_nonzero(np.isfinite(area)) > 0
    nmag = np.sqrt(np.sum(normals * normals, axis=-1))
    finite = np.isfinite(nmag)
    assert np.allclose(nmag[finite], 1.0, rtol=1e-6, atol=1e-6)


def test_pressure_components_on_surface_skips_relative_when_no_trajectory_velocity():
    n_phase = 5
    n_lon = 7
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    rho = np.full((n_phase, n_lon), 1e-13, dtype=float)
    u_xyz = np.zeros((n_phase, n_lon, 3), dtype=float)
    u_xyz[..., 0] = 3.0e5
    b_xyz = np.zeros((n_phase, n_lon, 3), dtype=float)
    b_xyz[..., 2] = 5.0e-5

    azimuth = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    xx = np.repeat(np.cos(azimuth)[None, :], n_phase, axis=0)
    yy = np.repeat(np.sin(azimuth)[None, :], n_phase, axis=0)
    zz = np.repeat((phase - 0.5)[:, None], n_lon, axis=1)
    points = np.stack(
        [
            xx,
            yy,
            zz,
            rho,
            u_xyz[..., 0],
            u_xyz[..., 1],
            u_xyz[..., 2],
            b_xyz[..., 0],
            b_xyz[..., 1],
            b_xyz[..., 2],
            np.full((n_phase, n_lon), 1.1e-3, dtype=float),
            np.repeat(phase[:, None], n_lon, axis=1),
            np.repeat(np.full(n_phase, 1.0 / n_phase, dtype=float)[:, None], n_lon, axis=1),
        ],
        axis=-1,
    )
    sampled = SmartDs(
        Dataset(
            points,
            np.empty((0, 0), dtype=int),
            {"RBODY [m]": SOLAR_RADIUS_M},
            "surface-demo",
            [
                "X [R]",
                "Y [R]",
                "Z [R]",
                "Rho [kg/m^3]",
                "U_x [m/s]",
                "U_y [m/s]",
                "U_z [m/s]",
                "B_x [T]",
                "B_y [T]",
                "B_z [T]",
                "P [Pa]",
                "phase [turns]",
                "time_weight [none]",
            ],
            "surface",
        )
    ).prepare(body_radius=SOLAR_RADIUS_M)

    out = pressure_components_on_surface(sampled)
    assert "relative_ram_pressure [Pa]" not in out
    assert "U_minus_V [m/s]" not in out
    assert "V [m/s]" not in out
    assert out["phase_quantiles"]["ram_pressure [Pa]"]["values"].shape == (n_phase, 5)
    assert np.isfinite(out["summary"]["ram_pressure [Pa]"]["mean"])


def test_sample_surface_revolution_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    points = circular_orbit_points(10.0, n_points=64)
    phase = np.arange(points.shape[0], dtype=float) / float(points.shape[0])
    time_weight = np.full(points.shape[0], 1.0 / float(points.shape[0]), dtype=float)
    out = sample_surface_revolution(
        sds,
        fields=("Rho [g/cm^3]", "U_x [km/s]", "B_x [Gauss]"),
        trajectory_points=points,
        phase=phase,
        time_weight=time_weight,
        trajectory_meta={
            "semi_major_axis [R]": 10.0,
            "eccentricity [none]": 0.0,
        },
        n_longitudes=32,
        method="nearest",
    )
    assert np.array(out("Rho [g/cm^3]")).shape == (64, 32)
    assert np.array(out("U_x [km/s]")).shape == (64, 32)
    assert np.array(out("phase [turns]")).shape == (64, 32)
    assert np.array(out("time_weight [none]")).shape == (64, 32)
    assert np.isclose(np.sum(np.array(out("time_weight [none]"))[:, 0]), 1.0)


def test_pressure_components_on_surface_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    points = circular_orbit_points(10.0, n_points=64)
    phase = np.arange(points.shape[0], dtype=float) / float(points.shape[0])
    time_weight = np.full(points.shape[0], 1.0 / float(points.shape[0]), dtype=float)
    period_s = orbital_period(10.0 * SOLAR_RADIUS_M, SUN_MASS_KG)
    time = phase * period_s
    velocity = trajectory_velocity(
        points,
        time,
        coordinate_scale=SOLAR_RADIUS_M,
    )
    sampled = sample_surface_revolution(
        sds,
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
            "B_x [T]",
            "B_y [T]",
            "B_z [T]",
            "U [m/s]",
            "B [T]",
            "magnetic_pressure [Pa]",
            "ram_pressure [Pa]",
            "thermal_pressure [Pa]",
            "standoff_distance [m]",
        ),
        trajectory_points=points,
        phase=phase,
        time=time,
        time_weight=time_weight,
        velocity_xyz=velocity,
        trajectory_meta={
            "semi_major_axis [R]": 10.0,
            "eccentricity [none]": 0.0,
        },
        n_longitudes=48,
        method="nearest",
    )
    out = pressure_components_on_surface(sampled)
    for key in (
        "ram_pressure [Pa]",
        "magnetic_pressure [Pa]",
        "thermal_pressure [Pa]",
        "standoff_distance [m]",
    ):
        arr = np.array(out[key], dtype=float)
        assert arr.shape == (64, 48)
        assert np.count_nonzero(np.isfinite(arr)) > 0
        assert key in out["summary"]
        assert key in out["phase_quantiles"]
    assert "relative_ram_pressure [Pa]" in out
    assert out["phase_quantiles"]["ram_pressure [Pa]"]["values"].shape[0] == 64


def test_torque_components_on_surface_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    points = circular_orbit_points(10.0, n_points=64)
    phase = np.arange(points.shape[0], dtype=float) / float(points.shape[0])
    time_weight = np.full(points.shape[0], 1.0 / float(points.shape[0]), dtype=float)
    sampled = sample_surface_revolution(
        sds,
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
            "B_x [T]",
            "B_y [T]",
            "B_z [T]",
            "thermal_pressure [Pa]",
        ),
        trajectory_points=points,
        phase=phase,
        time_weight=time_weight,
        trajectory_meta={
            "semi_major_axis [R]": 10.0,
            "eccentricity [none]": 0.0,
        },
        n_longitudes=48,
        method="nearest",
    )
    out = torque_components_on_surface(
        sampled,
        body_radius=SOLAR_RADIUS_M,
        angvel=0.0,
    )
    for key in (
        "T1_magnetic [Nm]",
        "T2_pressure [Nm]",
        "T3_corotation [Nm]",
        "T4_dynamic [Nm]",
        "total [Nm]",
        "coverage [none]",
    ):
        arr = np.array(out[key], dtype=float)
        assert arr.shape == ()
        assert np.isfinite(arr)
    assert np.isclose(out["total [Nm]"], out["T1_magnetic [Nm]"] + out["T2_pressure [Nm]"] + out["T3_corotation [Nm]"] + out["T4_dynamic [Nm]"])
    assert np.count_nonzero(np.isfinite(np.array(out["surface_area [m^2]"]))) > 0
    assert "phase_integrals" in out and "phase_quantiles" in out
    assert out["phase_integrals"]["total"]["integral [Nm]"].shape[0] == 64
    assert out["phase_quantiles"]["total"]["values [N/m]"].shape[0] == 64
