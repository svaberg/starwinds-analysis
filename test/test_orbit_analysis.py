from pathlib import Path

import numpy as np
import pytest
from scipy import constants as const

from starwinds_analysis.analysis.trajectories import elliptic_orbit_points
from starwinds_analysis.analysis.trajectories import sample_elliptic_orbit
from starwinds_analysis.analysis.trajectories import trajectory_velocity
from starwinds_analysis.analysis.shells import integrate_shell_scalar
from starwinds_analysis.analysis.shells import sample_shell_field
from starwinds_analysis.analysis.stats import summarize_samples
from starwinds_analysis.constants import SOLAR_RADIUS_M
from starwinds_analysis.physics.curve import mass_loss_from_curve
from starwinds_analysis.physics.curve import torque_from_curve
from starwinds_analysis.physics.orbits import orbital_period
from starwinds_analysis.physics.orbits import orbital_velocity
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_4_n00000000.plt")


def interpolate_profile(radii, values, x):
    """Interpolate one shell profile onto curve radii, using NaN outside range."""
    r = np.array(radii)
    y = np.array(values)
    x = np.array(x)
    good = np.isfinite(r) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        return np.full_like(x, np.nan, dtype=float)
    r = r[good]
    y = y[good]
    order = np.argsort(r)
    r = r[order]
    y = y[order]
    out = np.interp(x, r, y, left=np.nan, right=np.nan)
    out[(x < r[0]) | (x > r[-1])] = np.nan
    return out


def compare_curve_mass_loss_to_shell(
    smart_ds,
    curve,
    *,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    """Compare local curve mass-loss estimates against shell-integrated values."""
    body_radius = float(curve("star_radius [m]"))
    weights = curve.get("time_weight [none]")
    mass_flux = np.array(curve("mass_flux [kg/m^2/s]"))
    radius = np.array(curve("R [m]"))
    radius_r = radius / body_radius
    estimates = mass_loss_from_curve(curve)
    stats = summarize_samples(estimates, weights=weights)

    _, shell_mass_flux, shell_area, shell_profile_radii = sample_shell_field(
        smart_ds,
        [float(np.nanmean(radius_r))] if shell_radii is None else shell_radii,
        source_fields=("Rho [kg/m^3]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"),
        shell_field="mass_flux [kg/m^2/s]",
        body_radius=body_radius,
        n_polar=shell_n_polar,
        n_azimuth=shell_n_azimuth,
        method=method,
    )
    shell_values, _ = integrate_shell_scalar(shell_mass_flux, shell_area)
    if shell_radii is None:
        shell_value = float(shell_values[0])
        shell_interp = np.full_like(estimates, shell_value, dtype=float)
    else:
        shell_interp = interpolate_profile(shell_profile_radii, shell_values, radius_r)
        shell_value = summarize_samples(shell_interp, weights=weights)["mean"]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to_shell = stats["mean"] / shell_value if shell_value != 0 else np.nan

    out = {
        "radius [R]": float(np.nanmean(radius_r)),
        "radius [m]": float(np.nanmean(radius)),
        "mass_flux [kg/m^2/s]": mass_flux,
        "local_mass_loss [kg/s]": estimates,
        "local_mass_loss_mean [kg/s]": float(stats["mean"]),
        "local_mass_loss_std [kg/s]": float(stats["std"]),
        "shell_mass_loss [kg/s]": float(shell_value),
        "mean_to_shell [none]": float(mean_to_shell),
        "curve_samples": curve,
    }
    if shell_radii is not None:
        out["shell_mass_loss_interp [kg/s]"] = shell_interp
    return out


def compare_curve_torque_to_shell(
    smart_ds,
    curve,
    *,
    method: str,
    shell_n_polar: int,
    shell_n_azimuth: int,
    shell_radii=None,
):
    """Compare local curve torque estimates against shell-integrated values."""
    body_radius = float(curve("star_radius [m]"))
    weights = curve.get("time_weight [none]")
    radius = np.array(curve("R [m]"))
    radius_r = radius / body_radius
    curve_magnetic_density = np.array(curve("magnetic_torque_density [N/m]"))
    curve_dynamic_density = np.array(curve("dynamic_torque_density [N/m]"))
    local_magnetic, local_dynamic, local_total = torque_from_curve(curve)
    torque_shells, shell_magnetic_density, shell_area, shell_profile_radii = sample_shell_field(
        smart_ds,
        [float(np.nanmean(radius_r))] if shell_radii is None else shell_radii,
        source_fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
            "B_x [T]",
            "B_y [T]",
            "B_z [T]",
        ),
        shell_field="magnetic_torque_density [N/m]",
        body_radius=body_radius,
        n_polar=shell_n_polar,
        n_azimuth=shell_n_azimuth,
        method=method,
    )
    shell_dynamic_density = np.array(torque_shells("dynamic_torque_density [N/m]"))
    shell_magnetic, _ = integrate_shell_scalar(shell_magnetic_density, shell_area)
    shell_dynamic, _ = integrate_shell_scalar(shell_dynamic_density, shell_area)
    shell_values = shell_magnetic + shell_dynamic
    if shell_radii is None:
        shell_total = float(shell_values[0])
        shell_interp = np.full_like(local_total, shell_total, dtype=float)
    else:
        shell_interp = interpolate_profile(shell_profile_radii, shell_values, radius_r)
        shell_total = summarize_samples(shell_interp, weights=weights)["mean"]

    stats = summarize_samples(local_total, weights=weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to_shell = stats["mean"] / shell_total if shell_total != 0 else np.nan

    out = {
        "radius [R]": float(np.nanmean(radius_r)),
        "radius [m]": float(np.nanmean(radius)),
        "magnetic_torque_density [N/m]": curve_magnetic_density,
        "dynamic_torque_density [N/m]": curve_dynamic_density,
        "local_magnetic_torque [Nm]": local_magnetic,
        "local_dynamic_torque [Nm]": local_dynamic,
        "local_total_torque [Nm]": local_total,
        "local_total_torque_mean [Nm]": float(stats["mean"]),
        "local_total_torque_std [Nm]": float(stats["std"]),
        "shell_total_torque [Nm]": float(shell_total),
        "mean_to_shell [none]": float(mean_to_shell),
        "curve_samples": curve,
    }
    if shell_radii is not None:
        out["shell_total_torque_interp [Nm]"] = shell_interp
    return out


def test_zero_eccentricity_orbit_points_constant_radius():
    pts = elliptic_orbit_points(3.0, eccentricity=0.0, n_points=64, plane="xy")
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


def test_trajectory_velocity_matches_linear_motion():
    points = np.column_stack(
        [
            2.0 + 0.1 * np.arange(6, dtype=float),
            -3.0 + 0.2 * np.arange(6, dtype=float),
            5.0 - 0.4 * np.arange(6, dtype=float),
        ]
    )
    time = 10.0 + np.arange(6, dtype=float)
    velocity = trajectory_velocity(points, time, coordinate_scale=2.0)
    expected = np.array([0.2, 0.4, -0.8], dtype=float)
    np.testing.assert_allclose(velocity, np.repeat(expected[None, :], 6, axis=0), rtol=1e-12, atol=1e-12)


def test_trajectory_velocity_rejects_nonincreasing_time():
    points = np.column_stack(
        [
            np.arange(4, dtype=float),
            np.arange(4, dtype=float),
            np.arange(4, dtype=float),
        ]
    )
    time = np.array([0.0, 1.0, 1.0, 2.0], dtype=float)
    with pytest.raises(ValueError, match="strictly increasing"):
        trajectory_velocity(points, time, coordinate_scale=1.0)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_sample_zero_eccentricity_orbit_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    out = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.0,
        fields=("Rho [g/cm^3]", "U_x [km/s]", "B_x [Gauss]"),
        n_points=72,
        method="nearest",
    )

    x = np.array(out("X [R]"))
    y = np.array(out("Y [R]"))
    z = np.array(out("Z [R]"))
    radius = np.array(out("R [m]"))
    assert np.array(out("Rho [g/cm^3]")).shape == (72,)
    np.testing.assert_allclose(np.sqrt(x * x + y * y + z * z), 10.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(radius, SOLAR_RADIUS_M * 10.0, rtol=1e-12, atol=0.0)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_sample_elliptic_orbit_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    out = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.2,
        fields=("Rho [g/cm^3]", "U_x [km/s]", "B_x [Gauss]"),
        n_points=96,
        method="nearest",
    )

    x = np.array(out("X [R]"))
    y = np.array(out("Y [R]"))
    z = np.array(out("Z [R]"))
    radius_r = np.sqrt(x * x + y * y + z * z)
    radius = np.array(out("R [m]"))
    assert np.array(out("Rho [g/cm^3]")).shape == (96,)
    assert np.array(out("phase [turns]")).shape == (96,)
    assert np.array(out("time_weight [none]")).shape == (96,)
    assert np.isclose(np.sum(out("time_weight [none]")), 1.0)
    assert np.nanmin(radius_r) < 10.0
    assert np.nanmax(radius_r) > 10.0
    np.testing.assert_allclose(radius, radius_r * SOLAR_RADIUS_M, rtol=1e-12, atol=0.0)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_mass_loss_from_curve_runs():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    curve = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.1,
        fields=("mass_flux [kg/m^2/s]",),
        n_points=96,
        method="nearest",
    )
    values = np.array(mass_loss_from_curve(curve))
    assert values.shape == (96,)
    assert np.count_nonzero(np.isfinite(values)) > 0


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_torque_from_curve_runs():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    curve = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.1,
        fields=(
            "magnetic_torque_density [N/m]",
            "dynamic_torque_density [N/m]",
        ),
        n_points=96,
        method="nearest",
    )
    magnetic, dynamic, total = torque_from_curve(curve)
    magnetic = np.array(magnetic)
    dynamic = np.array(dynamic)
    total = np.array(total)
    assert magnetic.shape == (96,)
    assert dynamic.shape == (96,)
    np.testing.assert_allclose(total, magnetic + dynamic, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_compare_curve_mass_loss_to_shell_runs():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    curve = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.0,
        fields=("mass_flux [kg/m^2/s]",),
        n_points=96,
        method="nearest",
    )
    out = compare_curve_mass_loss_to_shell(
        sds,
        curve,
        shell_n_polar=12,
        shell_n_azimuth=24,
        method="nearest",
    )

    local_vals = np.array(out["local_mass_loss [kg/s]"])
    assert local_vals.shape == (96,)
    assert np.count_nonzero(np.isfinite(local_vals)) > 0
    assert np.isfinite(out["local_mass_loss_mean [kg/s]"])
    assert np.isfinite(out["shell_mass_loss [kg/s]"])
    assert np.isfinite(out["mean_to_shell [none]"])


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_compare_curve_torque_to_shell_runs():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    curve = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.0,
        fields=(
            "magnetic_torque_density [N/m]",
            "dynamic_torque_density [N/m]",
        ),
        n_points=96,
        method="nearest",
    )
    out = compare_curve_torque_to_shell(
        sds,
        curve,
        shell_n_polar=12,
        shell_n_azimuth=24,
        method="nearest",
    )

    tot = np.array(out["local_total_torque [Nm]"])
    mag = np.array(out["local_magnetic_torque [Nm]"])
    dyn = np.array(out["local_dynamic_torque [Nm]"])

    np.testing.assert_allclose(tot, mag + dyn, rtol=1e-12, atol=1e-12)
    assert np.isfinite(out["local_total_torque_mean [Nm]"])
    assert np.isfinite(out["shell_total_torque [Nm]"])
    assert np.isfinite(out["mean_to_shell [none]"])


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_compare_curve_mass_loss_to_shell_profile_runs():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    curve = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.2,
        fields=("mass_flux [kg/m^2/s]",),
        n_points=96,
        method="nearest",
    )
    shell_radii = np.linspace(8.0, 12.0, 8)
    out = compare_curve_mass_loss_to_shell(
        sds,
        curve,
        method="nearest",
        shell_n_polar=12,
        shell_n_azimuth=24,
        shell_radii=shell_radii,
    )

    local_vals = np.array(out["local_mass_loss [kg/s]"])
    shell_vals = np.array(out["shell_mass_loss_interp [kg/s]"])
    assert local_vals.shape == (96,)
    assert shell_vals.shape == (96,)
    assert np.count_nonzero(np.isfinite(local_vals)) > 0
    assert np.count_nonzero(np.isfinite(shell_vals)) > 0
    assert np.isfinite(out["local_mass_loss_mean [kg/s]"])
    assert np.isfinite(out["shell_mass_loss [kg/s]"])
    assert np.isfinite(out["mean_to_shell [none]"])


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_compare_curve_torque_to_shell_profile_runs():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    curve = sample_elliptic_orbit(
        sds,
        10.0,
        eccentricity=0.15,
        fields=(
            "magnetic_torque_density [N/m]",
            "dynamic_torque_density [N/m]",
        ),
        n_points=96,
        method="nearest",
    )
    shell_radii = np.linspace(8.5, 11.5, 8)
    out = compare_curve_torque_to_shell(
        sds,
        curve,
        method="nearest",
        shell_n_polar=12,
        shell_n_azimuth=24,
        shell_radii=shell_radii,
    )

    tot = np.array(out["local_total_torque [Nm]"])
    mag = np.array(out["local_magnetic_torque [Nm]"])
    dyn = np.array(out["local_dynamic_torque [Nm]"])
    shell_tot = np.array(out["shell_total_torque_interp [Nm]"])

    np.testing.assert_allclose(tot, mag + dyn, rtol=1e-12, atol=1e-12)
    assert shell_tot.shape == (96,)
    assert np.isfinite(out["local_total_torque_mean [Nm]"])
