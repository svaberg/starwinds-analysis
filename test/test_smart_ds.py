import importlib.util
from pathlib import Path

import numpy as np
import pytest

from starwinds_readplt.dataset import Dataset

from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_1_n00060000.plt")
SUN_RADIUS_M = 6.96e8


def make_dataset_2d():
    # q = x + y on a unit square, useful for interpolation checks.
    variables = ["X [R]", "Y [R]", "Q [none]"]
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    corners = np.empty((0, 0), dtype=int)
    return Dataset(points, corners, aux={"demo": 1}, title="demo", variables=variables, zone="z0")


def make_dataset_3d_vectors():
    variables = [
        "X [R]",
        "Y [R]",
        "Z [R]",
        "B_x [T]",
        "B_y [T]",
        "B_z [T]",
    ]
    # Three points away from axis/origin for well-defined all spherical components.
    points = np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0, 4.0, 5.0, 6.0],
            [1.0, 1.0, 1.0, -1.0, 2.0, -3.0],
        ],
        dtype=float,
    )
    corners = np.empty((0, 0), dtype=int)
    return Dataset(points, corners, aux={}, title="demo3d", variables=variables, zone="z3d")


def test_passthrough_raw_field():
    sds = SmartDs(make_dataset_2d())

    np.testing.assert_allclose(sds.variable("Q [none]"), [0.0, 1.0, 1.0, 2.0])
    assert sds.has_field("Q [none]")
    assert "Q [none]" in sds


def test_lazy_registered_field_is_cached():
    sds = SmartDs(make_dataset_2d())
    calls = {"n": 0}

    def q_squared(ds):
        calls["n"] += 1
        q = ds.variable("Q [none]")
        return q**2

    sds.register_field("Q2 [none]", q_squared)

    first = sds.variable("Q2 [none]")
    second = sds.variable("Q2 [none]")

    np.testing.assert_allclose(first, [0.0, 1.0, 1.0, 4.0])
    np.testing.assert_allclose(second, first)
    assert calls["n"] == 1


def test_call_uses_smartds_resolution_but_getitem_is_raw_only():
    sds = SmartDs(make_dataset_2d())
    sds.register_field("Q2 [none]", lambda ds: np.array(ds.variable("Q [none]")) ** 2)

    # `()` routes through SmartDs.variable(...) and can resolve computed fields.
    np.testing.assert_allclose(sds("Q2 [none]"), [0.0, 1.0, 1.0, 4.0])

    # `[]` is a raw Dataset passthrough (base fields only).
    np.testing.assert_allclose(sds["Q [none]"], [0.0, 1.0, 1.0, 2.0])
    with pytest.raises(IndexError):
        _ = sds["Q2 [none]"]


def test_alias_passthrough_to_existing_raw_field():
    sds = SmartDs(make_dataset_2d(), aliases={"q": "Q [none]"})
    np.testing.assert_allclose(sds.variable("q"), [0.0, 1.0, 1.0, 2.0])


@pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="scipy is required for SmartDs.resample()",
)
def test_resample_returns_new_wrapped_dataset_nearest():
    sds = SmartDs(make_dataset_2d())
    target = np.array([[0.1, 0.2], [0.95, 0.85]])

    out = sds.resample(
        target,
        coordinate_fields=("X [R]", "Y [R]"),
        fields=["Q [none]"],
        method="nearest",
    )

    assert isinstance(out, SmartDs)
    assert out is not sds
    assert out.raw is not sds.raw
    assert list(out.raw.variables) == ["X [R]", "Y [R]", "Q [none]"]

    np.testing.assert_allclose(out.variable("X [R]"), target[:, 0])
    np.testing.assert_allclose(out.variable("Y [R]"), target[:, 1])
    np.testing.assert_allclose(out.variable("Q [none]"), [0.0, 2.0])


@pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="scipy is required for SmartDs.resample()",
)
def test_resample_linear_interpolates_inside_hull():
    sds = SmartDs(make_dataset_2d())
    target = np.array([[0.25, 0.50], [0.60, 0.10]])

    out = sds.resample(
        target,
        coordinate_fields=("X [R]", "Y [R]"),
        fields=["Q [none]"],
        method="linear",
    )

    np.testing.assert_allclose(out.variable("Q [none]"), [0.75, 0.70], rtol=0, atol=1e-12)


def test_add_spherical_fields_computes_geometry_and_vector_components():
    sds = SmartDs(make_dataset_3d_vectors()).add_spherical_fields(vectors=("B",))

    r = sds.variable("R [R]")
    theta = sds.variable("theta [rad]")
    phi = sds.variable("phi [rad]")
    b_r = sds.variable("B_r [T]")
    b_theta = sds.variable("B_theta [T]")
    b_phi = sds.variable("B_phi [T]")

    assert r.shape == (3,)
    assert theta.shape == (3,)
    assert phi.shape == (3,)
    assert b_r.shape == (3,)
    assert b_theta.shape == (3,)
    assert b_phi.shape == (3,)

    # First point is on +x axis.
    np.testing.assert_allclose(r[0], 1.0)
    np.testing.assert_allclose(theta[0], np.pi / 2)
    np.testing.assert_allclose(phi[0], 0.0)
    np.testing.assert_allclose(b_r[0], 1.0)
    np.testing.assert_allclose(b_theta[0], -3.0)
    np.testing.assert_allclose(b_phi[0], 2.0)

    lat = sds.variable("latitude [rad]")
    lon = sds.variable("longitude [rad]")
    lat_deg = sds.variable("latitude [deg]")
    lon_deg = sds.variable("longitude [deg]")
    np.testing.assert_allclose(lat, (np.pi / 2) - theta)
    np.testing.assert_allclose(lon, phi)
    np.testing.assert_allclose(lat_deg, np.degrees(lat))
    np.testing.assert_allclose(lon_deg, np.degrees(lon))


def test_spherical_fields_are_available_by_default_for_xyz_vector_datasets():
    sds = SmartDs(make_dataset_3d_vectors())

    # No explicit `add_spherical_fields()` call.
    r = np.array(sds("R [R]"))
    theta = np.array(sds("theta [rad]"))
    phi = np.array(sds("phi [rad]"))
    b_r = np.array(sds("B_r [T]"))
    lat_deg = np.array(sds("latitude [deg]"))
    lon_deg = np.array(sds("longitude [deg]"))

    assert r.shape == (3,)
    assert theta.shape == (3,)
    assert phi.shape == (3,)
    assert b_r.shape == (3,)
    np.testing.assert_allclose(lat_deg, np.degrees((np.pi / 2) - theta))
    np.testing.assert_allclose(lon_deg, np.degrees(phi))


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_add_spherical_fields_on_real_example_data():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.add_spherical_fields(vectors=("B", "U"))

    x = np.array(sds.variable("X [R]"))
    y = np.array(sds.variable("Y [R]"))
    z = np.array(sds.variable("Z [R]"))

    r = np.array(sds.variable("R [R]"))
    theta = np.array(sds.variable("theta [rad]"))
    phi = np.array(sds.variable("phi [rad]"))

    assert r.shape == x.shape
    assert theta.shape == x.shape
    assert phi.shape == x.shape

    finite_r = np.isfinite(r)
    assert np.any(finite_r)
    assert np.nanmin(r) >= 0.0

    finite_theta = np.isfinite(theta)
    assert np.all((theta[finite_theta] >= 0.0) & (theta[finite_theta] <= np.pi))

    finite_phi = np.isfinite(phi)
    assert np.all((phi[finite_phi] >= -np.pi) & (phi[finite_phi] <= np.pi))

    # Check B_r against direct projection for all non-singular points.
    bx = np.array(sds.variable("B_x [Gauss]"))
    by = np.array(sds.variable("B_y [Gauss]"))
    bz = np.array(sds.variable("B_z [Gauss]"))
    b_r = np.array(sds.variable("B_r [Gauss]"))

    mask = np.isfinite(r) & (r > 0)
    direct = np.full_like(r, np.nan, dtype=float)
    direct[mask] = (bx[mask] * x[mask] + by[mask] * y[mask] + bz[mask] * z[mask]) / r[mask]
    np.testing.assert_allclose(b_r[mask], direct[mask], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("griblet") is None,
    reason="griblet not installed in this environment",
)
def test_griblet_graph_resolution_and_explain():
    import griblet
    from starwinds_analysis.recipes.spherical import build_griblet_spherical_geometry_graph

    sds = SmartDs(make_dataset_3d_vectors())
    graph = build_griblet_spherical_geometry_graph(coord_fields=("X [R]", "Y [R]", "Z [R]"))
    sds.set_computation_graph(graph)

    r = sds.variable("R [R]")
    np.testing.assert_allclose(r, np.sqrt(np.sum(sds.points[:, :3] ** 2, axis=1)))

    lat = sds.variable("latitude [rad]")
    lon = sds.variable("longitude [rad]")
    lat_deg = sds.variable("latitude [deg]")
    lon_deg = sds.variable("longitude [deg]")
    theta = sds.variable("theta [rad]")
    phi = sds.variable("phi [rad]")
    np.testing.assert_allclose(lat, (np.pi / 2) - theta)
    np.testing.assert_allclose(lon, phi)
    np.testing.assert_allclose(lat_deg, np.degrees(lat))
    np.testing.assert_allclose(lon_deg, np.degrees(lon))

    explanation = sds.explain("theta [rad]")
    assert "theta [rad]" in explanation
    assert "X [R]" in explanation


@pytest.mark.skipif(
    importlib.util.find_spec("griblet") is None,
    reason="griblet not installed in this environment",
)
@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_griblet_add_spherical_graph_on_real_example_data():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.add_spherical_graph(vectors=("B",))

    theta = np.array(sds.variable("theta [rad]"))
    b_r = np.array(sds.variable("B_r [Gauss]"))

    assert theta.shape == sds.variable("X [R]").shape
    assert b_r.shape == sds.variable("B_x [Gauss]").shape

    finite_theta = np.isfinite(theta)
    assert np.all((theta[finite_theta] >= 0.0) & (theta[finite_theta] <= np.pi))

    expl = sds.explain("B_r [Gauss]")
    assert "B_r [Gauss]" in expl
    assert "B_x [Gauss]" in expl


@pytest.mark.skipif(
    importlib.util.find_spec("griblet") is None,
    reason="griblet not installed in this environment",
)
@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_griblet_batsrus_si_normalization_and_derived_fields():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.add_batsrus_graph(body_radius_m=SUN_RADIUS_M)

    # Unit normalization examples
    bx_g = np.array(sds.variable("B_x [Gauss]"))
    bx_t = np.array(sds.variable("B_x [T]"))
    rho_cgs = np.array(sds.variable("Rho [g/cm^3]"))
    rho_si = np.array(sds.variable("Rho [kg/m^3]"))
    p_cgs = np.array(sds.variable("P [dyne/cm^2]"))
    p_si = np.array(sds.variable("P [Pa]"))

    np.testing.assert_allclose(bx_t, bx_g * 1e-4, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(rho_si, rho_cgs * 1e3, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(p_si, p_cgs * 1e-1, rtol=1e-12, atol=0.0)

    # Canonicalize unbracketed units
    qrad_raw = np.array(sds.variable("qrad J/m^3/s"))
    qrad_canonical = np.array(sds.variable("qrad [J/m^3/s]"))
    np.testing.assert_allclose(qrad_canonical, qrad_raw)

    # Derived examples
    gamma = float(sds.variable("GAMMA [none]"))
    c_s = np.array(sds.variable("c_s [m/s]"))
    c_s_direct = np.sqrt(gamma * p_si / rho_si)
    np.testing.assert_allclose(c_s, c_s_direct, rtol=1e-10, atol=1e-10)

    u = np.array(sds.variable("U [m/s]"))
    u_direct = np.sqrt(
        np.array(sds.variable("U_x [m/s]")) ** 2
        + np.array(sds.variable("U_y [m/s]")) ** 2
        + np.array(sds.variable("U_z [m/s]")) ** 2
    )
    np.testing.assert_allclose(u, u_direct, rtol=1e-12, atol=1e-12)

    b = np.array(sds.variable("B [T]"))
    c_a = np.array(sds.variable("c_A [m/s]"))
    c_a_direct = b / np.sqrt((4e-7 * np.pi) * rho_si)
    np.testing.assert_allclose(c_a, c_a_direct, rtol=1e-10, atol=1e-10)

    m_a = np.array(sds.variable("M_A [none]"))
    np.testing.assert_allclose(m_a, u / c_a, rtol=1e-10, atol=1e-10)

    ma = np.array(sds.variable("Ma [none]"))
    np.testing.assert_allclose(ma, u / c_s, rtol=1e-10, atol=1e-10)

    p_b = np.array(sds.variable("P_b [Pa]"))
    np.testing.assert_allclose(p_b, b**2 / (2 * (4e-7 * np.pi)), rtol=1e-10, atol=1e-10)
    magnetic_pressure = np.array(sds.variable("magnetic_pressure [Pa]"))
    np.testing.assert_allclose(magnetic_pressure, p_b, rtol=1e-12, atol=1e-12)

    ram_pressure = np.array(sds.variable("ram_pressure [Pa]"))
    np.testing.assert_allclose(ram_pressure, rho_si * u**2, rtol=1e-10, atol=1e-10)

    beta = np.array(sds.variable("beta [none]"))
    np.testing.assert_allclose(beta, p_si / p_b, rtol=1e-10, atol=1e-10)

    # Pointwise spherical/flux/torque quantities (via batsrus graph + spherical graph).
    u_r = np.array(sds.variable("U_r [m/s]"))
    mass_flux = np.array(sds.variable("mass_flux [kg/m^2/s]"))
    np.testing.assert_allclose(mass_flux, rho_si * u_r, rtol=1e-10, atol=1e-10)

    b_r = np.array(sds.variable("B_r [T]"))
    b_phi = np.array(sds.variable("B_phi [T]"))
    u_phi = np.array(sds.variable("U_phi [m/s]"))
    varpi = np.array(sds.variable("cylindrical_radius [m]"))
    tmag = np.array(sds.variable("magnetic_torque_density [N/m]"))
    tdyn = np.array(sds.variable("dynamic_torque_density [N/m]"))
    ttot = np.array(sds.variable("total_torque_density [N/m]"))
    np.testing.assert_allclose(tmag, -varpi * b_phi * b_r / (4e-7 * np.pi), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(tdyn, varpi * rho_si * u_phi * u_r, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ttot, tmag + tdyn, rtol=1e-10, atol=1e-10)

    b_mer = np.array(sds.variable("B_meridional [T]"))
    b_theta = np.array(sds.variable("B_theta [T]"))
    b_tan = np.array(sds.variable("B_tangential [T]"))
    np.testing.assert_allclose(b_mer, -b_theta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b_tan, np.sqrt(np.array(sds.variable("B_phi [T]")) ** 2 + b_mer**2), rtol=1e-10, atol=1e-10)

    expl = sds.explain("M_A [none]")
    assert "M_A [none]" in expl
    assert "c_A [m/s]" in expl
    assert "U [m/s]" in expl

    assert "beta [none]" in sds.explain("beta [none]")
    assert "mass_flux [kg/m^2/s]" in sds.explain("mass_flux [kg/m^2/s]")
