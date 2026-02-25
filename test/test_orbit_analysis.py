from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.analysis.orbits import (
    circular_orbit_points,
    local_mass_loss_on_circular_orbit,
    local_torque_on_circular_orbit,
    sample_circular_orbit,
)
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")
SUN_RADIUS_M = 6.957e8


def test_circular_orbit_points_constant_radius():
    pts = circular_orbit_points(3.0, n_points=64, plane="xy")
    r = np.sqrt(np.sum(pts * pts, axis=1))
    np.testing.assert_allclose(r, 3.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pts[:, 2], 0.0)


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

