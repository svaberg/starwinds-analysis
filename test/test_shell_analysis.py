from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.analysis.mass_loss import mass_loss_vs_radius
from starwinds_analysis.analysis.shells import integrate_shell_scalar, sample_spherical_shells
from starwinds_analysis.analysis.stats import weighted_mean_std, weighted_quantile
from starwinds_analysis.analysis.torque import torque_vs_radius
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")
SUN_RADIUS_M = 6.957e8


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_sample_spherical_shells_area_matches_sphere():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    radii = np.array([2.0, 5.0, 10.0])

    shells = sample_spherical_shells(
        sds,
        radii,
        fields=(),
        n_polar=12,
        n_azimuth=24,
        method="nearest",
        length_unit_to_m=SUN_RADIUS_M,
    )

    area_total = np.sum(shells.area, axis=(-2, -1))
    expected = 4.0 * np.pi * (radii * SUN_RADIUS_M) ** 2
    np.testing.assert_allclose(area_total, expected, rtol=2e-2, atol=0.0)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_mass_loss_profile_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    profile = mass_loss_vs_radius(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        body_radius_m=SUN_RADIUS_M,
        n_polar=12,
        n_azimuth=24,
        method="nearest",
    )

    m = np.asarray(profile["mass_loss [kg/s]"])
    c = np.asarray(profile["coverage [none]"])

    assert m.shape == (4,)
    assert c.shape == (4,)
    assert np.all(np.isfinite(c))
    assert np.all((c > 0.95) & (c <= 1.0 + 1e-12))
    assert np.count_nonzero(np.isfinite(m)) == 4
    assert np.any(np.abs(m) > 0)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_torque_profile_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    profile = torque_vs_radius(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        body_radius_m=SUN_RADIUS_M,
        n_polar=12,
        n_azimuth=24,
        method="nearest",
    )

    mag = np.asarray(profile["magnetic_torque [Nm]"])
    dyn = np.asarray(profile["dynamic_torque [Nm]"])
    tot = np.asarray(profile["total_torque [Nm]"])
    cov = np.asarray(profile["coverage [none]"])

    assert mag.shape == dyn.shape == tot.shape == (4,)
    np.testing.assert_allclose(tot, mag + dyn, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(cov))
    assert np.all((cov > 0.90) & (cov <= 1.0 + 1e-12))
    assert np.any(np.isfinite(tot))


def test_integrate_shell_scalar_reports_coverage():
    area = np.ones((2, 3, 4))
    values = np.ones_like(area)
    values[0, 0, 0] = np.nan

    integral, coverage = integrate_shell_scalar(values, area)
    np.testing.assert_allclose(integral, [11.0, 12.0])
    np.testing.assert_allclose(coverage, [11 / 12, 1.0])


def test_weighted_stats_helpers():
    mean, std = weighted_mean_std([1, 2, 3], [1, 1, 2])
    assert np.isclose(mean, 2.25)
    assert std > 0

    qs = weighted_quantile([1, 2, 3, 4], [0.0, 0.5, 1.0], [1, 1, 1, 1])
    np.testing.assert_allclose(qs, [1, 2, 4])

