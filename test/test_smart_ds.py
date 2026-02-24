import importlib.util
from pathlib import Path

import numpy as np
import pytest

from starwinds_readplt.dataset import Dataset

from starwinds_analysis.smart_ds import SmartDs


EXAMPLE_PLT = Path("examples/3d__var_1_n00000000.plt")


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


@pytest.mark.skipif(not EXAMPLE_PLT.exists(), reason="example BATSRUS file not present")
def test_add_spherical_fields_on_real_example_data():
    sds = SmartDs.from_file(str(EXAMPLE_PLT))
    sds.add_spherical_fields(vectors=("B", "U"))

    x = np.asarray(sds.variable("X [R]"))
    y = np.asarray(sds.variable("Y [R]"))
    z = np.asarray(sds.variable("Z [R]"))

    r = np.asarray(sds.variable("R [R]"))
    theta = np.asarray(sds.variable("theta [rad]"))
    phi = np.asarray(sds.variable("phi [rad]"))

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
    bx = np.asarray(sds.variable("B_x [Gauss]"))
    by = np.asarray(sds.variable("B_y [Gauss]"))
    bz = np.asarray(sds.variable("B_z [Gauss]"))
    b_r = np.asarray(sds.variable("B_r [Gauss]"))

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

    explanation = sds.explain("theta [rad]")
    assert "theta [rad]" in explanation
    assert "X [R]" in explanation
