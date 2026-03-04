import importlib.util
from pathlib import Path

import numpy as np
import pytest

from starwinds_readplt.dataset import Dataset

from starwinds_analysis.constants import MU0
from starwinds_analysis.constants import SOLAR_RADIUS_M
from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("sample_data/3d__var_4_n00000000.plt")
DAT_EQUIV_STEMS = (
    "3d__var_4_n00000000",
    "x=0_var_2_n00000000",
    "z=0_var_1_n00000000",
)


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


def make_dataset_spherical_coords():
    variables = ["R [R]", "polar [rad]", "azimuth [rad]"]
    points = np.array(
        [
            [2.0, np.pi / 2.0, 0.0],
            [3.0, np.pi / 2.0, np.pi / 2.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    corners = np.empty((0, 0), dtype=int)
    return Dataset(points, corners, aux={}, title="demo-spherical", variables=variables, zone="zsph")


def test_passthrough_raw_field():
    sds = SmartDs(make_dataset_2d())

    np.testing.assert_allclose(sds.variable("Q [none]"), [0.0, 1.0, 1.0, 2.0])
    assert sds.has_field("Q [none]")
    assert "Q [none]" in sds



def test_call_uses_smartds_resolution():
    sds = SmartDs(make_dataset_2d())
    sds.add_batsrus_graph(include_unit_normalization=False, include_derived=False)

    with pytest.raises(IndexError):
        sds("Q2 [none]")


def test_one_off_field_can_be_added_directly_to_griblet_graph():
    points = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=float,
    )
    dataset = Dataset(
        points,
        np.empty((0, 0), dtype=int),
        aux={},
        title="demo-graph",
        variables=["B_x [T]", "U_x [m/s]"],
        zone="z-graph",
    )
    sds = SmartDs(dataset)

    import griblet

    graph = griblet.ComputationGraph()
    graph.add_recipe(
        "Bx_plus_Ux [mixed]",
        lambda bx, ux: np.array(bx) + np.array(ux),
        deps=["B_x [T]", "U_x [m/s]"],
        cost=0.1,
    )
    sds.set_computation_graph(graph)

    np.testing.assert_allclose(sds("Bx_plus_Ux [mixed]"), [11.0, 22.0, 33.0])


def test_smartds_field_access_is_name_only():
    sds = SmartDs(make_dataset_2d())

    with pytest.raises(TypeError):
        sds.variable(0)

    with pytest.raises(TypeError):
        sds(0)

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


def test_add_spherical_graph_computes_geometry_and_vector_components():
    sds = SmartDs(make_dataset_3d_vectors()).add_spherical_graph(vectors=("B",))

    r = sds.variable("R [R]")
    polar = sds.variable("polar [rad]")
    azimuth = sds.variable("azimuth [rad]")
    b_r = sds.variable("B_r [T]")
    b_p = sds.variable("B_p [T]")
    b_a = sds.variable("B_a [T]")
    b_rpa = sds.variable("B_rpa [T]")

    assert r.shape == (3,)
    assert polar.shape == (3,)
    assert azimuth.shape == (3,)
    assert b_r.shape == (3,)
    assert b_p.shape == (3,)
    assert b_a.shape == (3,)
    assert b_rpa.shape == (3, 3)

    # First point is on +x axis.
    np.testing.assert_allclose(r[0], 1.0)
    np.testing.assert_allclose(polar[0], np.pi / 2)
    np.testing.assert_allclose(azimuth[0], 0.0)
    np.testing.assert_allclose(b_r[0], 1.0)
    np.testing.assert_allclose(b_p[0], -3.0)
    np.testing.assert_allclose(b_a[0], 2.0)
    np.testing.assert_allclose(b_rpa[:, 0], b_r)
    np.testing.assert_allclose(b_rpa[:, 1], b_p)
    np.testing.assert_allclose(b_rpa[:, 2], b_a)

    lat = sds.variable("latitude [rad]")
    lon = sds.variable("longitude [rad]")
    lat_deg = sds.variable("latitude [deg]")
    lon_deg = sds.variable("longitude [deg]")
    np.testing.assert_allclose(lat, (np.pi / 2) - polar)
    np.testing.assert_allclose(lon, azimuth)
    np.testing.assert_allclose(lat_deg, np.degrees(lat))
    np.testing.assert_allclose(lon_deg, np.degrees(lon))


def test_add_spherical_graph_exposes_polar_azimuth_and_compact_vector_names():
    sds = SmartDs(make_dataset_3d_vectors()).add_spherical_graph(vectors=("B",))

    polar = sds.variable("polar [rad]")
    azimuth = sds.variable("azimuth [rad]")
    b_p = sds.variable("B_p [T]")
    b_a = sds.variable("B_a [T]")

    with pytest.raises(IndexError):
        sds.variable("theta [rad]")
    with pytest.raises(IndexError):
        sds.variable("phi [rad]")
    with pytest.raises(IndexError):
        sds.variable("B_theta [T]")
    with pytest.raises(IndexError):
        sds.variable("B_phi [T]")


@pytest.mark.skipif(
    importlib.util.find_spec("griblet") is None,
    reason="griblet is required for graph-based spherical inverse recipes",
)
def test_add_spherical_graph_can_recover_cartesian_from_r_polar_azimuth():
    sds = SmartDs(make_dataset_spherical_coords())
    sds.add_spherical_graph(coord_fields=("X [R]", "Y [R]", "Z [R]"))

    x = np.array(sds.variable("X [R]"))
    y = np.array(sds.variable("Y [R]"))
    z = np.array(sds.variable("Z [R]"))

    np.testing.assert_allclose(x, [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(y, [0.0, 3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(z, [0.0, 0.0, 4.0], atol=1e-12)


def test_add_spherical_graph_registers_xyz_geometry_and_vectors():
    sds = SmartDs(make_dataset_3d_vectors())
    sds.add_spherical_graph(vectors=("B",))

    r = np.array(sds("R [R]"))
    polar = np.array(sds("polar [rad]"))
    azimuth = np.array(sds("azimuth [rad]"))
    b_r = np.array(sds("B_r [T]"))
    lat_deg = np.array(sds("latitude [deg]"))
    lon_deg = np.array(sds("longitude [deg]"))

    assert r.shape == (3,)
    assert polar.shape == (3,)
    assert azimuth.shape == (3,)
    assert b_r.shape == (3,)


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_batsrus_graph_fields_resolve_on_prepared_example():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.prepare(body_radius=SOLAR_RADIUS_M)

    expected_fields = [
        "X [m]",
        "Y [m]",
        "Z [m]",
        "R [R]",
        "R [m]",
        "RBODY [m]",
        "polar [rad]",
        "azimuth [rad]",
        "latitude [rad]",
        "longitude [rad]",
        "latitude [deg]",
        "longitude [deg]",
        "U [m/s]",
        "B [T]",
        "U_xyz [m/s]",
        "B_xyz [T]",
        "U_r [m/s]",
        "U_p [m/s]",
        "U_a [m/s]",
        "B_r [T]",
        "B_p [T]",
        "B_a [T]",
        "c_s [m/s]",
        "c_A [m/s]",
        "M_A [none]",
        "Ma [none]",
        "P_b [Pa]",
        "magnetic_pressure [Pa]",
        "thermal_pressure [Pa]",
        "ram_pressure [Pa]",
        "standoff_distance [m]",
        "beta [none]",
        "mass_flux [kg/m^2/s]",
        "energy_flux [W/m^2]",
        "cylindrical_radius [R]",
        "cylindrical_radius [m]",
        "magnetic_torque_density [N/m]",
        "dynamic_torque_density [N/m]",
        "total_torque_density [N/m]",
        "B_meridional [T]",
        "B_tangential [T]",
    ]

    if Path("sample_data/PARAM.in").exists():
        expected_fields.extend(
            [
                "star_radius [m]",
                "star_mass [kg]",
                "star_rotational_period [s]",
                "star_rotation_rate [rad/s]",
            ]
        )

    unit_factors = {
        "g/cm^3": "kg/m^3",
        "amu/cm^3": "kg/m^3",
        "km/s": "m/s",
        "Gauss": "T",
        "G": "T",
        "nT": "T",
        "erg/cm^3": "J/m^3",
        "dyne/cm^2": "Pa",
        "nPa": "Pa",
        "`mA/m^2": "A/m^2",
    }
    by_prefix: dict[tuple[str, str], set[str]] = {}
    for name in sds.raw.variables:
        if " [" not in name or not name.endswith("]"):
            continue
        base, unit = name[:-1].split(" [", 1)
        if "_" not in base:
            continue
        prefix, comp = base.rsplit("_", 1)
        if comp not in {"x", "y", "z"}:
            continue
        si_unit = unit_factors.get(unit, unit)
        by_prefix.setdefault((prefix, si_unit), set()).add(comp)

    for (prefix, unit), comps in by_prefix.items():
        if comps != {"x", "y", "z"}:
            continue
        expected_fields.extend(
            [
                f"{prefix}_xyz [{unit}]",
                f"{prefix} [{unit}]",
                f"{prefix}_r [{unit}]",
                f"{prefix}_p [{unit}]",
                f"{prefix}_a [{unit}]",
                f"{prefix}_rpa [{unit}]",
            ]
        )

    n_points = len(sds.points)
    seen: set[str] = set()
    for field in expected_fields:
        if field in seen:
            continue
        seen.add(field)
        value = np.array(sds(field))
        if value.ndim == 0:
            continue
        assert value.shape[0] == n_points, field


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_add_spherical_graph_on_real_example_data():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.add_spherical_graph(vectors=("B", "U"))

    x = np.array(sds.variable("X [R]"))
    y = np.array(sds.variable("Y [R]"))
    z = np.array(sds.variable("Z [R]"))

    r = np.array(sds.variable("R [R]"))
    polar = np.array(sds.variable("polar [rad]"))
    azimuth = np.array(sds.variable("azimuth [rad]"))

    assert r.shape == x.shape
    assert polar.shape == x.shape
    assert azimuth.shape == x.shape

    finite_r = np.isfinite(r)
    assert np.any(finite_r)
    assert np.nanmin(r) >= 0.0

    finite_polar = np.isfinite(polar)
    assert np.all((polar[finite_polar] >= 0.0) & (polar[finite_polar] <= np.pi))

    finite_azimuth = np.isfinite(azimuth)
    assert np.all((azimuth[finite_azimuth] >= -np.pi) & (azimuth[finite_azimuth] <= np.pi))

    # Check B_r against direct projection for all non-singular points.
    bx = np.array(sds.variable("B_x [Gauss]"))
    by = np.array(sds.variable("B_y [Gauss]"))
    bz = np.array(sds.variable("B_z [Gauss]"))
    b_r = np.array(sds.variable("B_r [Gauss]"))

    mask = np.isfinite(r) & (r > 0)
    direct = np.full_like(r, np.nan, dtype=float)
    direct[mask] = (bx[mask] * x[mask] + by[mask] * y[mask] + bz[mask] * z[mask]) / r[mask]
    np.testing.assert_allclose(b_r[mask], direct[mask], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("stem", DAT_EQUIV_STEMS)
def test_dat_and_plt_pairs_load_to_nearly_identical_smartds(stem):
    plt_path = Path("sample_data") / f"{stem}.plt"
    dat_path = Path("sample_data") / f"{stem}.dat"

    if not plt_path.exists() or not dat_path.exists():
        pytest.skip("matching .plt/.dat pair not present")

    sds_plt = SmartDs.from_file(plt_path)
    sds_dat = SmartDs.from_file(dat_path)

    assert sds_plt.title == sds_dat.title
    assert sds_plt.zone == sds_dat.zone
    assert len(sds_plt.points) == len(sds_dat.points)
    assert getattr(sds_plt.corners, "shape", None) == getattr(sds_dat.corners, "shape", None)
    assert tuple(sds_plt.variables) == tuple(sds_dat.variables)

    for name in sds_plt.variables:
        np.testing.assert_allclose(
            np.ravel(sds_plt(name)),
            np.ravel(sds_dat(name)),
            rtol=0.0,
            atol=2.0e-4,
            err_msg=name,
        )


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
    polar = sds.variable("polar [rad]")
    azimuth = sds.variable("azimuth [rad]")
    np.testing.assert_allclose(lat, (np.pi / 2) - polar)
    np.testing.assert_allclose(lon, azimuth)
    np.testing.assert_allclose(lat_deg, np.degrees(lat))
    np.testing.assert_allclose(lon_deg, np.degrees(lon))

    explanation = sds.explain("polar [rad]")
    assert "polar [rad]" in explanation
    assert "X [R]" in explanation


@pytest.mark.skipif(
    importlib.util.find_spec("griblet") is None,
    reason="griblet not installed in this environment",
)
@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_griblet_add_spherical_graph_on_real_example_data():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.add_spherical_graph(vectors=("B",))

    polar = np.array(sds.variable("polar [rad]"))
    b_r = np.array(sds.variable("B_r [Gauss]"))

    assert polar.shape == sds.variable("X [R]").shape
    assert b_r.shape == sds.variable("B_x [Gauss]").shape

    finite_polar = np.isfinite(polar)
    assert np.all((polar[finite_polar] >= 0.0) & (polar[finite_polar] <= np.pi))

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
    sds.add_batsrus_graph()

    # Unit normalization examples
    bx_g = np.array(sds.variable("B_x [Gauss]"))
    bx = np.array(sds.variable("B_x [T]"))
    rho_cgs = np.array(sds.variable("Rho [g/cm^3]"))
    rho_si = np.array(sds.variable("Rho [kg/m^3]"))
    p_cgs = np.array(sds.variable("P [dyne/cm^2]"))
    p_si = np.array(sds.variable("P [Pa]"))

    np.testing.assert_allclose(bx, bx_g * 1e-4, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(rho_si, rho_cgs * 1e3, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(p_si, p_cgs * 1e-1, rtol=1e-12, atol=0.0)

    assert sds.aux["Star_name"] == "tau Boötis"
    assert np.isfinite(float(sds("star_radius [m]")))
    assert np.isfinite(float(sds("star_mass [kg]")))
    assert np.isfinite(float(sds("star_rotational_period [s]")))
    assert np.isfinite(float(sds("star_rotation_rate [rad/s]")))

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
    c_a_direct = b / np.sqrt(MU0 * rho_si)
    np.testing.assert_allclose(c_a, c_a_direct, rtol=1e-10, atol=1e-10)

    m_a = np.array(sds.variable("M_A [none]"))
    np.testing.assert_allclose(m_a, u / c_a, rtol=1e-10, atol=1e-10)

    ma = np.array(sds.variable("Ma [none]"))
    np.testing.assert_allclose(ma, u / c_s, rtol=1e-10, atol=1e-10)

    p_b = np.array(sds.variable("P_b [Pa]"))
    np.testing.assert_allclose(p_b, b**2 / (2 * MU0), rtol=1e-10, atol=1e-10)
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
    b_a = np.array(sds.variable("B_a [T]"))
    u_a = np.array(sds.variable("U_a [m/s]"))
    varpi = np.array(sds.variable("cylindrical_radius [m]"))
    tmag = np.array(sds.variable("magnetic_torque_density [N/m]"))
    tdyn = np.array(sds.variable("dynamic_torque_density [N/m]"))
    ttot = np.array(sds.variable("total_torque_density [N/m]"))
    np.testing.assert_allclose(tmag, -varpi * b_a * b_r / MU0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(tdyn, varpi * rho_si * u_a * u_r, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ttot, tmag + tdyn, rtol=1e-10, atol=1e-10)

    b_mer = np.array(sds.variable("B_meridional [T]"))
    b_p = np.array(sds.variable("B_p [T]"))
    b_tan = np.array(sds.variable("B_tangential [T]"))
    np.testing.assert_allclose(b_mer, -b_p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b_tan, np.sqrt(np.array(sds.variable("B_a [T]")) ** 2 + b_mer**2), rtol=1e-10, atol=1e-10)

    expl = sds.explain("M_A [none]")
    assert "M_A [none]" in expl
    assert "c_A [m/s]" in expl
    assert "U [m/s]" in expl

    assert "beta [none]" in sds.explain("beta [none]")
    assert "mass_flux [kg/m^2/s]" in sds.explain("mass_flux [kg/m^2/s]")
