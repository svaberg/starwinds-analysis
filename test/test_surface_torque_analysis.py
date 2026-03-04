from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.algorithms.sphere_sampling import fibonacci_sphere
from starwinds_analysis.constants import MU0
from starwinds_analysis.constants import SOLAR_RADIUS_M
from starwinds_analysis.analysis.shells import integrate_shell_scalar
from starwinds_analysis.analysis.shells import sample_shell_field
from starwinds_analysis.physics.torque import integrate_surface_torque_terms
from starwinds_analysis.physics.torque import surface_torque_density_terms
from starwinds_analysis.physics.torque import surface_torque_vs_radius
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00060000.plt")


def _cart_from_spherical_components(v_r, v_phi, xyz):
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx = np.divide(x, r, out=np.zeros_like(x), where=r > 0)
        ny = np.divide(y, r, out=np.zeros_like(y), where=r > 0)
        nz = np.divide(z, r, out=np.zeros_like(z), where=r > 0)
        cyl = np.sqrt(x * x + y * y)
        ex_phi = np.divide(-y, cyl, out=np.zeros_like(y), where=cyl > 0)
        ey_phi = np.divide(x, cyl, out=np.zeros_like(x), where=cyl > 0)
    v = np.zeros_like(xyz)
    v[..., 0] = v_r * nx + v_phi * ex_phi
    v[..., 1] = v_r * ny + v_phi * ey_phi
    v[..., 2] = v_r * nz
    return v


def test_surface_torque_density_terms_matches_analytic_sphere_integral():
    n_points = 4096
    r_m = 3.0
    xyz = fibonacci_sphere(n_points).reshape(1, n_points, 1, 3) * r_m
    normals = xyz / r_m
    area = np.full((1, n_points, 1), 4.0 * np.pi * r_m * r_m / n_points)

    rho = np.full((1, n_points, 1), 2.0)
    u_r = 5.0
    u_phi = 7.0
    b_r = 2.0e-4
    b_phi = -4.0e-4
    u = _cart_from_spherical_components(u_r, u_phi, xyz)
    b = _cart_from_spherical_components(b_r, b_phi, xyz)
    p = np.full((1, n_points, 1), 3.0)

    terms = surface_torque_density_terms(
        xyz_m=xyz,
        normals_xyz=normals,
        area_m2=area,
        rho_kg_m3=rho,
        u_xyz_m_s=u,
        b_xyz_t=b,
        pressure_pa=p,
        angvel_rad_s=0.0,
        use_rotating_frame=True,
    )
    ints = integrate_surface_torque_terms(terms)

    rest = (np.pi**2) * (r_m**3)
    expected_t1 = -(b_phi * b_r / MU0) * rest
    expected_t4 = (u_phi * u_r * float(rho[0, 0, 0])) * rest
    expected_total = expected_t1 + expected_t4

    assert np.isclose(ints["T2_pressure [Nm]"][0], 0.0, rtol=0, atol=1e-8)
    assert np.isclose(ints["T3_corotation [Nm]"][0], 0.0, rtol=0, atol=1e-8)
    assert np.isclose(ints["T1_magnetic [Nm]"][0], expected_t1, rtol=3e-3, atol=0)
    assert np.isclose(ints["T4_dynamic [Nm]"][0], expected_t4, rtol=3e-3, atol=0)
    assert np.isclose(ints["total [Nm]"][0], expected_total, rtol=3e-3, atol=0)
    assert np.isclose(ints["coverage [none]"][0], 1.0, rtol=0, atol=1e-12)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_surface_torque_vs_radius_matches_shell_torque_on_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius_m=SOLAR_RADIUS_M)
    radii = [2.0, 4.0, 8.0, 16.0]

    shells, magnetic_density, area, _ = sample_shell_field(
        sds,
        radii,
        body_radius_m=SOLAR_RADIUS_M,
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
        n_polar=12,
        n_azimuth=24,
        sampling="fibonacci",
        method="nearest",
    )
    dynamic_density = np.array(shells("dynamic_torque_density [N/m]"))
    shell_magnetic, _ = integrate_shell_scalar(magnetic_density, area)
    shell_dynamic, _ = integrate_shell_scalar(dynamic_density, area)
    shell_total = shell_magnetic + shell_dynamic
    surf = surface_torque_vs_radius(
        sds,
        radii,
        body_radius_m=SOLAR_RADIUS_M,
        n_polar=12,
        n_azimuth=24,
        sampling="fibonacci",
        method="nearest",
        include_pressure_term=True,
        angvel_rad_s=0.0,
    )

    np.testing.assert_allclose(
        surf["T1_magnetic [Nm]"],
        shell_magnetic,
        rtol=2e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        surf["T4_dynamic [Nm]"],
        shell_dynamic,
        rtol=2e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        surf["total [Nm]"],
        shell_total,
        rtol=2e-10,
        atol=1e-10,
    )
    assert np.all(np.abs(np.array(surf["T2_pressure [Nm]"])) < 1e-6 * np.nanmax(np.abs(surf["total [Nm]"])) + 1e-12)
    assert np.all(np.abs(np.array(surf["T3_corotation [Nm]"])) < 1e-12)
