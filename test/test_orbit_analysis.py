from pathlib import Path

import numpy as np
import pytest
from scipy import constants as const

from starwinds_analysis.analysis.orbits import (
    circular_orbit_points,
    elliptic_orbit_points,
    local_mass_loss_on_circular_orbit,
    local_mass_loss_on_elliptic_orbit,
    local_torque_on_circular_orbit,
    local_torque_on_elliptic_orbit,
    orbital_period,
    orbital_velocity,
    sample_circular_orbit,
    sample_elliptic_orbit,
)
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")
SUN_RADIUS_M = 6.957e8


def test_circular_orbit_points_constant_radius():
    pts = circular_orbit_points(3.0, n_points=64, plane="xy")
    r = np.sqrt(np.sum(pts * pts, axis=1))
    np.testing.assert_allclose(r, 3.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pts[:, 2], 0.0)


def test_elliptic_orbit_points_radius_range_matches_a_e():
    a = 10.0
    e = 0.2
    info = elliptic_orbit_points(a, eccentricity=e, n_points=256, return_info=True)
    pts = info["points"]
    r = np.sqrt(np.sum(pts * pts, axis=1))

    np.testing.assert_allclose(np.nanmin(r), a * (1 - e), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.nanmax(r), a * (1 + e), rtol=0, atol=1e-10)
    assert np.isclose(np.sum(info["time_weight [none]"]), 1.0)
    assert np.max(info["time_weight [none]"]) > np.min(info["time_weight [none]"])


def test_orbital_velocity_is_constant_for_circular_case():
    # Adapted from the old batplotlib elliptic-orbit test: circular speed is constant.
    r = np.full(64, 1.0 * const.au)
    v = orbital_velocity(r, 1.98847e30, 1.0 * const.au)
    np.testing.assert_allclose(v, v[0], rtol=1e-12, atol=0)


def test_orbital_period_is_approximately_one_year_for_1au_solar_mass():
    p = orbital_period(1.0 * const.au, 1.98847e30)
    assert np.isclose(p / const.year, 1.0, rtol=5e-3)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_sample_circular_orbit_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = sample_circular_orbit(
        sds,
        10.0,
        fields=("Rho [g/cm^3]", "U_x [km/s]", "B_x [Gauss]"),
        n_points=72,
        method="nearest",
    )

    assert out["R [sample]"].shape == (72,)
    assert out["Rho [g/cm^3]"].shape == (72,)
    assert np.allclose(out["R [sample]"], 10.0, rtol=0, atol=1e-12)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_sample_elliptic_orbit_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.2,
        fields=("Rho [g/cm^3]", "U_x [km/s]", "B_x [Gauss]"),
        n_points=96,
        method="nearest",
    )

    assert out["R [sample]"].shape == (96,)
    assert out["Rho [g/cm^3]"].shape == (96,)
    assert out["phase [turns]"].shape == (96,)
    assert out["time_weight [none]"].shape == (96,)
    assert np.isclose(np.sum(out["time_weight [none]"]), 1.0)
    assert np.nanmin(out["R [sample]"]) < 10.0
    assert np.nanmax(out["R [sample]"]) > 10.0


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_local_mass_loss_on_circular_orbit_runs_and_compares_to_shell():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = local_mass_loss_on_circular_orbit(
        sds,
        10.0,
        body_radius_m=SUN_RADIUS_M,
        n_points=96,
        method="nearest",
        shell_n_polar=12,
        shell_n_azimuth=24,
    )

    local_vals = np.asarray(out["local_mass_loss [kg/s]"])
    assert local_vals.shape == (96,)
    assert np.count_nonzero(np.isfinite(local_vals)) > 0
    assert np.isfinite(out["summary"]["mean"])
    assert np.isfinite(out["shell_mass_loss [kg/s]"])
    assert np.isfinite(out["mean_to_shell [none]"])


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_local_torque_on_circular_orbit_runs_and_compares_to_shell():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = local_torque_on_circular_orbit(
        sds,
        10.0,
        body_radius_m=SUN_RADIUS_M,
        n_points=96,
        method="nearest",
        shell_n_polar=12,
        shell_n_azimuth=24,
    )

    tot = np.asarray(out["local_total_torque [Nm]"])
    mag = np.asarray(out["local_magnetic_torque [Nm]"])
    dyn = np.asarray(out["local_dynamic_torque [Nm]"])

    np.testing.assert_allclose(tot, mag + dyn, rtol=1e-12, atol=1e-12)
    assert np.isfinite(out["summary"]["mean"])
    assert np.isfinite(out["shell_total_torque [Nm]"])
    assert np.isfinite(out["mean_to_shell [none]"])


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_local_mass_loss_on_elliptic_orbit_runs_and_compares_to_shell_profile():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = local_mass_loss_on_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.2,
        body_radius_m=SUN_RADIUS_M,
        n_points=96,
        method="nearest",
        shell_n_polar=12,
        shell_n_azimuth=24,
        shell_n_radii=8,
    )

    local_vals = np.asarray(out["local_mass_loss [kg/s]"])
    shell_vals = np.asarray(out["shell_mass_loss_interp [kg/s]"])
    assert local_vals.shape == (96,)
    assert shell_vals.shape == (96,)
    assert np.count_nonzero(np.isfinite(local_vals)) > 0
    assert np.count_nonzero(np.isfinite(shell_vals)) > 0
    assert np.isfinite(out["summary"]["mean"])
    assert np.isfinite(out["shell_mass_loss [kg/s]"])
    assert np.isfinite(out["mean_to_shell [none]"])
    assert np.isclose(out["semi_major_axis [R]"], 10.0)
    assert np.isclose(out["eccentricity [none]"], 0.2)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_local_torque_on_elliptic_orbit_runs_and_compares_to_shell_profile():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    out = local_torque_on_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.15,
        body_radius_m=SUN_RADIUS_M,
        n_points=96,
        method="nearest",
        shell_n_polar=12,
        shell_n_azimuth=24,
        shell_n_radii=8,
    )

    tot = np.asarray(out["local_total_torque [Nm]"])
    mag = np.asarray(out["local_magnetic_torque [Nm]"])
    dyn = np.asarray(out["local_dynamic_torque [Nm]"])
    shell_tot = np.asarray(out["shell_total_torque_interp [Nm]"])

    np.testing.assert_allclose(tot, mag + dyn, rtol=1e-12, atol=1e-12)
    assert shell_tot.shape == (96,)
    assert np.count_nonzero(np.isfinite(shell_tot)) > 0
    assert np.isfinite(out["summary"]["mean"])
    assert np.isfinite(out["shell_total_torque [Nm]"])
    assert np.isfinite(out["mean_to_shell [none]"])
