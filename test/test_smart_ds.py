from pathlib import Path

import griblet
import numpy as np
import pytest

from batread.dataset import Dataset

from batwind.recipes.batsrus import build_griblet_batsrus_graph
from batwind.recipes.spherical import build_griblet_spherical_graph
from batwind.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")


def explain_field(sds: SmartDs, name: str) -> str:
    cost, tree = sds._resolve_field(name)
    lines: list[str] = []

    def walk(node, depth=0):
        meta = getattr(node, "recipe_metadata", {}) or {}
        desc = meta.get("description", "")
        planned = getattr(node, "cost", None)
        parts = [node.field]
        if planned is not None:
            parts.append(f"(cost={planned})")
        if desc:
            parts.append(f"- {desc}")
        lines.append("  " * depth + " ".join(parts))
        for dep in getattr(node, "deps", []):
            walk(dep, depth + 1)

    walk(tree)
    return "\n".join([f"{name} total_cost={cost}", *lines])


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

    np.testing.assert_allclose(sds["Q [none]"], [0.0, 1.0, 1.0, 2.0])
    assert sds.has_field("Q [none]")


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

    np.testing.assert_allclose(out["X [R]"], target[:, 0])
    np.testing.assert_allclose(out["Y [R]"], target[:, 1])
    np.testing.assert_allclose(out["Q [none]"], [0.0, 2.0])


def test_resample_linear_interpolates_inside_hull():
    sds = SmartDs(make_dataset_2d())
    target = np.array([[0.25, 0.50], [0.60, 0.10]])

    out = sds.resample(
        target,
        coordinate_fields=("X [R]", "Y [R]"),
        fields=["Q [none]"],
        method="linear",
    )

    np.testing.assert_allclose(out["Q [none]"], [0.75, 0.70], rtol=0, atol=1e-12)


def test_spherical_graph_computes_geometry_and_vector_components():
    sds = SmartDs(make_dataset_3d_vectors())
    sds.merge_computation_graph(build_griblet_spherical_graph(sds.keys()))

    r = sds["R [R]"]
    polar = sds["polar [rad]"]
    azimuth = sds["azimuth [rad]"]
    b_r = sds["B_r [T]"]
    b_p = sds["B_p [T]"]
    b_a = sds["B_a [T]"]

    assert r.shape == (3,)
    assert polar.shape == (3,)
    assert azimuth.shape == (3,)
    assert b_r.shape == (3,)
    assert b_p.shape == (3,)
    assert b_a.shape == (3,)

    # First point is on +x axis.
    np.testing.assert_allclose(r[0], 1.0)
    np.testing.assert_allclose(polar[0], np.pi / 2)
    np.testing.assert_allclose(azimuth[0], 0.0)
    np.testing.assert_allclose(b_r[0], 1.0)
    np.testing.assert_allclose(b_p[0], -3.0)
    np.testing.assert_allclose(b_a[0], 2.0)


def test_spherical_graph_on_real_example_data():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.merge_computation_graph(build_griblet_spherical_graph(sds.keys()))

    x = np.asarray(sds["X [R]"])
    y = np.asarray(sds["Y [R]"])
    z = np.asarray(sds["Z [R]"])

    r = np.asarray(sds["R [R]"])
    polar = np.asarray(sds["polar [rad]"])
    azimuth = np.asarray(sds["azimuth [rad]"])

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
    bx = np.asarray(sds["B_x [Gauss]"])
    by = np.asarray(sds["B_y [Gauss]"])
    bz = np.asarray(sds["B_z [Gauss]"])
    b_r = np.asarray(sds["B_r [Gauss]"])

    mask = np.isfinite(r) & (r > 0)
    direct = np.full_like(r, np.nan, dtype=float)
    direct[mask] = (bx[mask] * x[mask] + by[mask] * y[mask] + bz[mask] * z[mask]) / r[mask]
    np.testing.assert_allclose(b_r[mask], direct[mask], rtol=1e-5, atol=1e-6)


def test_griblet_graph_resolution_and_explain():
    sds = SmartDs(make_dataset_3d_vectors())
    graph = build_griblet_spherical_graph(sds.keys(), coord_fields=("X [R]", "Y [R]", "Z [R]"))
    sds.merge_computation_graph(graph)

    r = sds["R [R]"]
    np.testing.assert_allclose(r, np.sqrt(np.sum(sds.points[:, :3] ** 2, axis=1)))

    explanation = explain_field(sds, "polar [rad]")
    assert "polar [rad]" in explanation
    assert "X [R]" in explanation


def test_smartds_graph_is_never_none():
    sds = SmartDs(make_dataset_2d())

    assert isinstance(sds.computation_graph, griblet.ComputationGraph)
    assert sds.computation_graph.list_fields() == {"X [R]", "Y [R]", "Q [none]", "demo"}

    graph = griblet.ComputationGraph()
    graph.add_recipe("A [none]", lambda: np.array([1.0]), deps=[], cost=0.0)
    sds.merge_computation_graph(graph)
    assert "A [none]" in sds.keys()

    sds.clear_computation_graph()
    assert sds.computation_graph.list_fields() == {"X [R]", "Y [R]", "Q [none]", "demo"}
    assert "A [none]" not in sds.keys()


def test_griblet_spherical_graph_on_real_example_data():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.merge_computation_graph(build_griblet_spherical_graph(sds.keys()))

    polar = np.asarray(sds["polar [rad]"])
    b_r = np.asarray(sds["B_r [Gauss]"])

    assert polar.shape == sds["X [R]"].shape
    assert b_r.shape == sds["B_x [Gauss]"].shape

    finite_polar = np.isfinite(polar)
    assert np.all((polar[finite_polar] >= 0.0) & (polar[finite_polar] <= np.pi))

    expl = explain_field(sds, "B_r [Gauss]")
    assert "B_r [Gauss]" in expl
    assert "B_x [Gauss]" in expl


def test_griblet_batsrus_si_normalization_and_derived_fields():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.merge_computation_graph(build_griblet_batsrus_graph(sds.variables, aux=sds.aux))

    # Unit normalization examples
    bx_g = np.asarray(sds["B_x [Gauss]"])
    bx_t = np.asarray(sds["B_x [T]"])
    rho_cgs = np.asarray(sds["Rho [g/cm^3]"])
    rho_si = np.asarray(sds["Rho [kg/m^3]"])
    p_cgs = np.asarray(sds["P [dyne/cm^2]"])
    p_si = np.asarray(sds["P [Pa]"])

    np.testing.assert_allclose(bx_t, bx_g * 1e-4, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(rho_si, rho_cgs * 1e3, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(p_si, p_cgs * 1e-1, rtol=1e-12, atol=0.0)

    # Canonicalize unbracketed units
    qrad_raw = np.asarray(sds["qrad J/m^3/s"])
    qrad_canonical = np.asarray(sds["qrad [J/m^3/s]"])
    np.testing.assert_allclose(qrad_canonical, qrad_raw)

    # Derived examples
    gamma = float(sds["GAMMA [none]"])
    c_s = np.asarray(sds["c_s [m/s]"])
    c_s_direct = np.sqrt(gamma * p_si / rho_si)
    np.testing.assert_allclose(c_s, c_s_direct, rtol=1e-10, atol=1e-10)

    u = np.asarray(sds["U [m/s]"])
    u_direct = np.sqrt(
        np.asarray(sds["U_x [m/s]"]) ** 2
        + np.asarray(sds["U_y [m/s]"]) ** 2
        + np.asarray(sds["U_z [m/s]"]) ** 2
    )
    np.testing.assert_allclose(u, u_direct, rtol=1e-12, atol=1e-12)

    b = np.asarray(sds["B [T]"])
    c_a = np.asarray(sds["c_A [m/s]"])
    c_a_direct = b / np.sqrt((4e-7 * np.pi) * rho_si)
    np.testing.assert_allclose(c_a, c_a_direct, rtol=1e-10, atol=1e-10)

    m_a = np.asarray(sds["M_A [none]"])
    np.testing.assert_allclose(m_a, u / c_a, rtol=1e-10, atol=1e-10)

    ma = np.asarray(sds["Ma [none]"])
    np.testing.assert_allclose(ma, u / c_s, rtol=1e-10, atol=1e-10)

    p_b = np.asarray(sds["P_b [Pa]"])
    np.testing.assert_allclose(p_b, b**2 / (2 * (4e-7 * np.pi)), rtol=1e-10, atol=1e-10)

    beta = np.asarray(sds["beta [none]"])
    np.testing.assert_allclose(beta, p_si / p_b, rtol=1e-10, atol=1e-10)

    expl = explain_field(sds, "M_A [none]")
    assert "M_A [none]" in expl
    assert "c_A [m/s]" in expl
    assert "U [m/s]" in expl

    assert "beta [none]" in explain_field(sds, "beta [none]")
