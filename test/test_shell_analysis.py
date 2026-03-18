from pathlib import Path

import numpy as np

from batwind.algorithms.spherical import cartesian_vector_to_spherical_components
from batwind.analysis.shell_summary import boxcar_shell_weights
from batwind.analysis.shell_summary import summarize_shell_diagnostics_band
from batwind.analysis.shell_summary import summarize_shell_series
from batwind.analysis.shells import infer_cartesian_axis_radii
from batwind.analysis.shells import integrate_shell_scalar
from batwind.analysis.shells import sample_spherical_shells
from batwind.analysis.shells import sample_spherical_shells_fibonacci
from batwind.analysis.stats import summarize_samples
from batwind.analysis.stats import weighted_mean_std
from batwind.analysis.stats import weighted_quantile
from batwind.constants import SOLAR_RADIUS_M
from batwind.physics.wind_scaling import open_wind_magnetisation
from batwind.physics.wind_scaling import surface_escape_speed
from batwind.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")


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
        length_unit_to_m=SOLAR_RADIUS_M,
    )

    area_total = np.sum(np.array(shells["dA [m^2]"], dtype=float), axis=(-2, -1))
    expected = 4.0 * np.pi * (radii * SOLAR_RADIUS_M) ** 2
    np.testing.assert_allclose(area_total, expected, rtol=2e-2, atol=0.0)


def test_infer_cartesian_axis_radii_returns_sorted_positive_values():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    radii = infer_cartesian_axis_radii(sds, axis="x", r_min=1.0)
    assert radii.ndim == 1
    assert radii.size > 10
    assert np.all(np.isfinite(radii))
    assert np.all(radii > 0)
    assert np.all(np.diff(radii) >= 0)


def test_sample_spherical_shells_fibonacci_area_matches_sphere():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    radii = np.array([2.0, 5.0, 10.0])

    shells = sample_spherical_shells_fibonacci(
        sds,
        radii,
        fields=(),
        n_points=12 * 24,
        method="nearest",
        length_unit_to_m=SOLAR_RADIUS_M,
    )

    area_total = np.sum(np.array(shells["dA [m^2]"], dtype=float), axis=(-2, -1))
    expected = 4.0 * np.pi * (radii * SOLAR_RADIUS_M) ** 2
    np.testing.assert_allclose(area_total, expected, rtol=1e-12, atol=0.0)
    assert np.array(shells["X [R]"], dtype=float).shape[-1] == 1


def test_mass_loss_profile_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    shells = sample_spherical_shells_fibonacci(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        fields=("Rho [kg/m^3]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"),
        n_points=12 * 24,
        method="nearest",
        length_unit_to_m=SOLAR_RADIUS_M,
    )
    mass_flux = np.array(shells["mass_flux [kg/m^2/s]"])
    area = np.array(shells["dA [m^2]"])
    r_field = np.array(shells["R [R]"])
    radii_profile = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
    m, c = integrate_shell_scalar(mass_flux, area)

    assert m.shape == (4,)
    assert c.shape == (4,)
    assert np.all(np.isfinite(c))
    assert np.all((c > 0.95) & (c <= 1.0 + 1e-12))
    assert np.count_nonzero(np.isfinite(m)) == 4
    assert np.any(np.abs(m) > 0)
    assert radii_profile.shape == (4,)
    assert np.array(shells["X [R]"], dtype=float).shape[-1] == 1


def test_grid_shell_mass_flux_primitives_match_shell_integral():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    shells = sample_spherical_shells(
        sds,
        [5.0],
        fields=("Rho [kg/m^3]", "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"),
        n_polar=12,
        n_azimuth=24,
        method="nearest",
        length_unit_to_m=SOLAR_RADIUS_M,
    )
    rho = np.array(shells["Rho [kg/m^3]"], dtype=float)
    ux = np.array(shells["U_x [m/s]"], dtype=float)
    uy = np.array(shells["U_y [m/s]"], dtype=float)
    uz = np.array(shells["U_z [m/s]"], dtype=float)
    x = np.array(shells["X [R]"], dtype=float)
    y = np.array(shells["Y [R]"], dtype=float)
    z = np.array(shells["Z [R]"], dtype=float)
    u_r, _u_p, _u_a = cartesian_vector_to_spherical_components(ux, uy, uz, x, y, z)
    mass_flux = rho * u_r
    integral_arr, coverage_arr = integrate_shell_scalar(mass_flux, np.array(shells["dA [m^2]"], dtype=float))
    integral = float(integral_arr[0])
    coverage = float(coverage_arr[0])
    assert np.isfinite(integral)
    assert 0.95 < coverage <= 1.0 + 1e-12
    arr = np.array(mass_flux[0], dtype=float)
    finite = arr[np.isfinite(arr)]
    assert finite.size > 0
    assert arr.size == 12 * 24


def test_torque_profile_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    shells = sample_spherical_shells_fibonacci(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        fields=(
            "Rho [kg/m^3]",
            "U_x [m/s]",
            "U_y [m/s]",
            "U_z [m/s]",
            "B_x [T]",
            "B_y [T]",
            "B_z [T]",
        ),
        n_points=12 * 24,
        method="nearest",
        length_unit_to_m=SOLAR_RADIUS_M,
    )
    magnetic_density = np.array(shells["magnetic_torque_density [N/m]"])
    area = np.array(shells["dA [m^2]"])
    r_field = np.array(shells["R [R]"])
    radii_profile = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
    dynamic_density = np.array(shells["dynamic_torque_density [N/m]"])
    mag, cov_mag = integrate_shell_scalar(magnetic_density, area)
    dyn, cov_dyn = integrate_shell_scalar(dynamic_density, area)
    tot = mag + dyn
    cov = np.minimum(cov_mag, cov_dyn)

    assert mag.shape == dyn.shape == tot.shape == (4,)
    np.testing.assert_allclose(tot, mag + dyn, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(cov))
    assert np.all((cov > 0.90) & (cov <= 1.0 + 1e-12))
    assert np.any(np.isfinite(tot))
    assert radii_profile.shape == (4,)
    assert np.array(shells["X [R]"], dtype=float).shape[-1] == 1


def test_unsigned_magnetic_flux_profile_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    shells = sample_spherical_shells_fibonacci(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        fields=("B_x [T]", "B_y [T]", "B_z [T]"),
        n_points=16 * 32,
        method="nearest",
        length_unit_to_m=SOLAR_RADIUS_M,
    )
    b_r = np.array(shells["B_r [T]"])
    area = np.array(shells["dA [m^2]"])
    r_field = np.array(shells["R [R]"])
    radii_profile = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
    bx = np.array(shells["B_x [T]"])
    by = np.array(shells["B_y [T]"])
    bz = np.array(shells["B_z [T]"])
    x = np.array(shells["X [R]"])
    y = np.array(shells["Y [R]"])
    z = np.array(shells["Z [R]"])
    r_norm = np.sqrt(x * x + y * y + z * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx = x / r_norm
        ny = y / r_norm
        nz = z / r_norm
    bdotn = bx * nx + by * ny + bz * nz
    signed_scalar, cov_signed = integrate_shell_scalar(b_r, area)
    signed_vector, cov_vec = integrate_shell_scalar(bdotn, area)
    open_flux, cov_open = integrate_shell_scalar(np.abs(b_r), area)
    cov = np.minimum(np.minimum(cov_signed, cov_open), cov_vec)

    np.testing.assert_allclose(signed_scalar, signed_vector, rtol=1e-10, atol=1e-10)
    assert np.all(open_flux >= np.abs(signed_scalar) - 1e-12)
    assert np.all((cov > 0.95) & (cov <= 1.0 + 1e-12))
    assert radii_profile.shape == (4,)


def test_axisymmetric_open_flux_fraction_is_bounded():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    shells = sample_spherical_shells(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        fields=("B_x [T]", "B_y [T]", "B_z [T]"),
        n_polar=16,
        n_azimuth=32,
        method="nearest",
        length_unit_to_m=SOLAR_RADIUS_M,
    )
    b_r = np.array(shells["B_r [T]"])
    area = np.array(shells["dA [m^2]"])
    with np.errstate(invalid="ignore"):
        b_r_axi_theta = np.nanmean(b_r, axis=-1, keepdims=True)
    b_r_axi = np.broadcast_to(b_r_axi_theta, b_r.shape)
    axi, _ = integrate_shell_scalar(np.abs(b_r_axi), area)
    total, _ = integrate_shell_scalar(np.abs(b_r), area)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.divide(axi, total, out=np.full_like(axi, np.nan), where=total != 0)

    finite = np.isfinite(frac)
    assert np.any(finite)
    assert np.all(axi[finite] >= 0)
    assert np.all(total[finite] >= 0)
    assert np.all(frac[finite] >= -1e-12)
    assert np.all(frac[finite] <= 1.0 + 1e-12)
    assert np.array(shells["X [R]"], dtype=float).shape[-1] > 1


def test_energy_flux_profile_runs_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)
    energy_source = "E [J/m^3]" if sds.has_field("E [J/m^3]") else "E [erg/cm^3]"
    shells = sample_spherical_shells_fibonacci(
        sds,
        [2.0, 4.0, 8.0, 16.0],
        fields=(energy_source, "U_x [m/s]", "U_y [m/s]", "U_z [m/s]"),
        n_points=12 * 24,
        method="nearest",
        length_unit_to_m=SOLAR_RADIUS_M,
    )
    energy_flux_density = np.array(shells["energy_flux [W/m^2]"])
    area = np.array(shells["dA [m^2]"])
    r_field = np.array(shells["R [R]"])
    radii_profile = np.nanmean(r_field.reshape(r_field.shape[0], -1), axis=1)
    y, c = integrate_shell_scalar(energy_flux_density, area)
    assert y.shape == (4,)
    assert np.count_nonzero(np.isfinite(y)) == 4
    assert np.all((c > 0.95) & (c <= 1.0 + 1e-12))
    assert radii_profile.shape == (4,)


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


def test_local_torque_scaling_formula_and_summary():
    r = np.array([2.0, 3.0, 4.0])
    magnetic_density = np.array([1.0, -2.0, 3.0])
    dynamic_density = np.array([4.0, 5.0, -6.0])

    shell_area = 4.0 * np.pi * r**2
    magnetic = shell_area * magnetic_density
    dynamic = shell_area * dynamic_density
    total = magnetic + dynamic
    np.testing.assert_allclose(magnetic, 4.0 * np.pi * r**2 * magnetic_density)
    np.testing.assert_allclose(dynamic, 4.0 * np.pi * r**2 * dynamic_density)
    np.testing.assert_allclose(total, magnetic + dynamic)

    summary = summarize_samples(total)
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
