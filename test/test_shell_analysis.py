from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.analysis.fluxes import (
    axisymmetric_open_flux_vs_radius,
    energy_flux_vs_radius,
    open_magnetic_flux_vs_radius,
)
from starwinds_analysis.analysis.local_estimates import (
    local_mass_loss_estimates,
    local_torque_estimates,
    summarize_samples,
)
from starwinds_analysis.analysis.mass_loss import mass_loss_vs_radius
from starwinds_analysis.analysis.shell_summary import (
    boxcar_shell_weights,
    summarize_shell_diagnostics_band,
    summarize_shell_series,
)
from starwinds_analysis.analysis.shells import (
    infer_cartesian_axis_radii,
    integrate_shell_scalar,
    sample_spherical_shells,
    sample_spherical_shells_fibonacci,
)
from starwinds_analysis.analysis.stats import weighted_mean_std, weighted_quantile
from starwinds_analysis.analysis.torque import torque_vs_radius
from starwinds_analysis.analysis.wind_scaling import (
    open_wind_magnetisation,
    open_wind_magnetisation_from_profiles,
    surface_escape_speed,
)
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00000000.plt")
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
def test_infer_cartesian_axis_radii_returns_sorted_positive_values():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    radii = infer_cartesian_axis_radii(sds, axis="x", r_min=1.0)
    assert radii.ndim == 1
    assert radii.size > 10
    assert np.all(np.isfinite(radii))
    assert np.all(radii > 0)
    assert np.all(np.diff(radii) >= 0)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_sample_spherical_shells_fibonacci_area_matches_sphere():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    radii = np.array([2.0, 5.0, 10.0])

    shells = sample_spherical_shells_fibonacci(
        sds,
        radii,
        fields=(),
        n_points=12 * 24,
        method="nearest",
        length_unit_to_m=SUN_RADIUS_M,
    )

    area_total = np.sum(shells.area, axis=(-2, -1))
    expected = 4.0 * np.pi * (radii * SUN_RADIUS_M) ** 2
    np.testing.assert_allclose(area_total, expected, rtol=1e-12, atol=0.0)
    assert shells.x.shape[-1] == 1


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
    assert profile["shell_samples"].x.shape[-1] == 1  # Fibonacci default


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
    assert profile["shell_samples"].x.shape[-1] == 1  # Fibonacci default


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_unsigned_magnetic_flux_profile_runs_on_example():
    # Adapted from batplotlib's test_unsigned_magnetic_flux: compare signed flux from
    # B_r with signed flux from B·n, and compute unsigned/open flux.
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    profile = open_magnetic_flux_vs_radius(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        body_radius_m=SUN_RADIUS_M,
        n_polar=16,
        n_azimuth=32,
        method="nearest",
    )

    signed_scalar = np.asarray(profile["signed_flux [Wb]"])
    signed_vector = np.asarray(profile["signed_flux_from_vector [Wb]"])
    open_flux = np.asarray(profile["open_flux [Wb]"])
    cov = np.asarray(profile["coverage [none]"])

    np.testing.assert_allclose(signed_scalar, signed_vector, rtol=1e-10, atol=1e-10)
    assert np.all(open_flux >= np.abs(signed_scalar) - 1e-12)
    assert np.all((cov > 0.95) & (cov <= 1.0 + 1e-12))


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_axisymmetric_open_flux_fraction_is_bounded():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    profile = axisymmetric_open_flux_vs_radius(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        body_radius_m=SUN_RADIUS_M,
        n_polar=16,
        n_azimuth=32,
        method="nearest",
    )

    axi = np.asarray(profile["axisymmetric_open_flux [Wb]"])
    total = np.asarray(profile["open_flux [Wb]"])
    frac = np.asarray(profile["axisymmetric_open_flux_fraction [none]"])
    finite = np.isfinite(frac)

    assert np.any(finite)
    assert np.all(axi[finite] >= 0)
    assert np.all(total[finite] >= 0)
    assert np.all(frac[finite] >= -1e-12)
    assert np.all(frac[finite] <= 1.0 + 1e-12)
    assert profile["shell_samples"].x.shape[-1] > 1  # grid sampler retained for axisymmetry


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_energy_flux_profile_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    profile = energy_flux_vs_radius(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        body_radius_m=SUN_RADIUS_M,
        n_polar=12,
        n_azimuth=24,
        method="nearest",
    )

    y = np.asarray(profile["energy_flux [W]"])
    c = np.asarray(profile["coverage [none]"])
    assert y.shape == (4,)
    assert np.count_nonzero(np.isfinite(y)) == 4
    assert np.all((c > 0.95) & (c <= 1.0 + 1e-12))


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


def test_local_mass_loss_estimate_formula():
    r = np.array([2.0, 3.0])
    rho = np.array([1.0, 2.0])
    u_r = np.array([10.0, -5.0])
    got = local_mass_loss_estimates(r, rho, u_r)
    expected = 4 * np.pi * r**2 * rho * u_r
    np.testing.assert_allclose(got, expected)


def test_local_torque_estimate_formula_and_summary():
    r = np.array([2.0, 3.0, 4.0])
    rho = np.array([1.0, 2.0, 3.0])
    u_r = np.array([10.0, 20.0, 30.0])
    u_phi = np.array([1.0, 2.0, 3.0])
    b_r = np.array([1e-3, 2e-3, 3e-3])
    b_phi = np.array([2e-3, 1e-3, -1e-3])

    out = local_torque_estimates(r, rho, u_r, u_phi, b_r, b_phi)
    np.testing.assert_allclose(out["total [Nm]"], out["magnetic [Nm]"] + out["dynamic [Nm]"])

    summary = summarize_samples(out["total [Nm]"])
    assert np.isfinite(summary["mean"])
    assert np.isfinite(summary["std"])
    assert summary["values"].shape == (5,)


def test_shell_band_summary_helpers():
    radii = np.array([2.0, 4.0, 8.0, 16.0])
    values = np.array([1.0, 3.0, 9.0, 27.0])
    coverage = np.array([1.0, 0.5, 1.0, 1.0])
    weights = boxcar_shell_weights(radii, rmin=3.0, rmax=9.0)
    np.testing.assert_allclose(weights, [0, 1, 1, 0])

    s = summarize_shell_series(radii, values, coverage=coverage, rmin=3.0, rmax=9.0)
    assert s["rmin [R]"] == 4.0
    assert s["rmax [R]"] == 8.0
    assert s["n_active"] == 2
    assert isinstance(s["quantiles"], list)
    assert isinstance(s["values"], list)
    assert s["mean"] > 0

    diagnostics = {
        "mass_loss": {
            "radius [R]": radii,
            "height [R]": radii - 1,
            "mass_loss [kg/s]": values,
            "coverage [none]": coverage,
        }
    }
    band = summarize_shell_diagnostics_band(diagnostics, rmin=3.0, rmax=9.0)
    assert "mass_loss" in band
    assert "mass_loss [kg/s]" in band["mass_loss"]
    assert band["mass_loss"]["mass_loss [kg/s]"]["n_active"] == 2


def test_wind_scaling_helpers_formula():
    m = 1.0e30
    r = 1.0e9
    vesc = surface_escape_speed(m, r)
    assert np.isfinite(vesc) and vesc > 0

    phi = np.array([1e14, 2e14])
    dotm = np.array([1e9, 2e9])
    ups = open_wind_magnetisation(phi, dotm, m, r)
    assert ups.shape == (2,)
    assert np.all(np.isfinite(ups))

    diagnostics = {
        "mass_loss": {
            "radius [R]": np.array([2.0, 4.0]),
            "height [R]": np.array([1.0, 3.0]),
            "mass_loss [kg/s]": dotm,
        },
        "open_flux": {
            "radius [R]": np.array([2.0, 4.0]),
            "height [R]": np.array([1.0, 3.0]),
            "open_flux [Wb]": phi,
        },
    }
    prof = open_wind_magnetisation_from_profiles(diagnostics, star_mass_kg=m, star_radius_m=r)
    np.testing.assert_allclose(prof["Upsilon_open [none]"], ups)
