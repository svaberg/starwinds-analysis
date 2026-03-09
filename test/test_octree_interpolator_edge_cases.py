from __future__ import annotations

import math

import numpy as np
import pytest

from starwinds_analysis.octree import Octree
from starwinds_analysis.octree import OctreeInterpolator
from starwinds_analysis.octree import OctreeRayInterpolator


class _FakeDataset:
    """Minimal dataset-like object used for interpolator edge-case tests."""

    def __init__(
        self,
        points: np.ndarray,
        corners: np.ndarray | None,
        variables: dict[str, np.ndarray],
    ) -> None:
        """Store geometry/fields with a `Dataset`-compatible API."""
        self.points = points
        self.corners = corners
        self._variables = variables
        self.variables = list(variables.keys())
        self.aux: dict[str, str] = {}

    def variable(self, name: str) -> np.ndarray:
        """Return one variable array by name."""
        return self._variables[name]


def _build_fake_dataset(
    *,
    nr: int = 1,
    ntheta: int = 2,
    nphi: int = 4,
) -> _FakeDataset:
    """Build a small regular spherical hexahedral dataset."""
    r_edges = np.linspace(1.0, 2.0, nr + 1)
    theta_edges = np.linspace(0.0, math.pi, ntheta + 1)
    phi_edges = np.linspace(0.0, 2.0 * math.pi, nphi + 1)

    node_index = -np.ones((nr + 1, ntheta + 1, nphi + 1), dtype=np.int64)
    xyz_list: list[tuple[float, float, float]] = []
    node_id = 0
    for ir in range(nr + 1):
        r = float(r_edges[ir])
        for it in range(ntheta + 1):
            theta = float(theta_edges[it])
            st = math.sin(theta)
            ct = math.cos(theta)
            for ip in range(nphi + 1):
                phi = float(phi_edges[ip])
                x = r * st * math.cos(phi)
                y = r * st * math.sin(phi)
                z = r * ct
                xyz_list.append((x, y, z))
                node_index[ir, it, ip] = node_id
                node_id += 1

    corners: list[list[int]] = []
    for ir in range(nr):
        for it in range(ntheta):
            for ip in range(nphi):
                corners.append(
                    [
                        int(node_index[ir, it, ip]),
                        int(node_index[ir + 1, it, ip]),
                        int(node_index[ir, it + 1, ip]),
                        int(node_index[ir + 1, it + 1, ip]),
                        int(node_index[ir, it, ip + 1]),
                        int(node_index[ir + 1, it, ip + 1]),
                        int(node_index[ir, it + 1, ip + 1]),
                        int(node_index[ir + 1, it + 1, ip + 1]),
                    ]
                )

    points = np.array(xyz_list)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = 3.0 * x - 2.0 * y + 0.5 * z + 1.0
    return _FakeDataset(
        points=points,
        corners=np.array(corners, dtype=np.int64),
        variables={
            "X [R]": x,
            "Y [R]": y,
            "Z [R]": z,
            "Scalar": scalar,
        },
    )


def _first_resolvable_center(tree: Octree) -> np.ndarray:
    """Return first cell center that successfully resolves via lookup."""
    for c in np.array(tree.lookup._cell_centers):
        hit = tree.lookup_point(np.array(c, dtype=float), space="xyz")
        if hit is not None:
            return np.array(c, dtype=float)
    raise AssertionError("No resolvable center found in fake dataset.")


def _first_resolvable_rpa(tree: Octree) -> tuple[float, float, float]:
    """Return one interior spherical point that resolves in lookup."""
    lookup = tree.lookup
    for cid in range(int(lookup._corners.shape[0])):
        r = 0.5 * (float(lookup._cell_r_min[cid]) + float(lookup._cell_r_max[cid]))
        polar = 0.5 * (float(lookup._cell_theta_min[cid]) + float(lookup._cell_theta_max[cid]))
        azimuth = (float(lookup._cell_phi_start[cid]) + 0.4 * float(lookup._cell_phi_width[cid])) % (
            2.0 * math.pi
        )
        hit = tree.lookup_point(np.array([r, polar, azimuth], dtype=float), space="rpa")
        if hit is not None:
            return r, polar, azimuth
    raise AssertionError("No resolvable interior rpa point found in fake dataset.")


def test_interpolator_constructor_rejects_invalid_query_space() -> None:
    """Constructor should reject invalid query_space values."""
    ds = _build_fake_dataset()
    with pytest.raises(ValueError, match="query_space must be 'xyz' or 'rpa'"):
        OctreeInterpolator(ds, "Scalar", query_space="bad")


def test_interpolator_constructor_rejects_missing_corners() -> None:
    """Constructor should fail when dataset has no corner connectivity."""
    ds = _build_fake_dataset()
    ds_bad = _FakeDataset(ds.points, None, ds._variables)
    with pytest.raises(ValueError, match="Dataset has no cell connectivity"):
        OctreeInterpolator(ds_bad, "Scalar", query_space="xyz")


def test_interpolator_constructor_rejects_value_length_mismatch() -> None:
    """Constructor should require point-centered values with matching length."""
    ds = _build_fake_dataset()
    bad_values = np.ones(ds.points.shape[0] - 1)
    with pytest.raises(ValueError, match="values length"):
        OctreeInterpolator(ds, bad_values, query_space="xyz")


def test_interpolator_call_rejects_invalid_query_space_override() -> None:
    """Runtime call should reject invalid query_space override."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, "Scalar", query_space="xyz")
    with pytest.raises(ValueError, match="query_space must be 'xyz' or 'rpa'"):
        interp(np.array([[1.0, 0.0, 0.0]]), query_space="bad")


def test_prepare_queries_validation_errors() -> None:
    """`prepare_queries` should enforce valid tuple/shape conventions."""
    with pytest.raises(ValueError, match="Tuple input must have exactly 3 arrays"):
        OctreeInterpolator.prepare_queries((np.array([1.0]), np.array([2.0])))
    with pytest.raises(ValueError, match="1D xi must have length 3"):
        OctreeInterpolator.prepare_queries(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="xi must have shape"):
        OctreeInterpolator.prepare_queries(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="Call with xi or with x1, x2, x3"):
        OctreeInterpolator.prepare_queries(np.array([1.0]), np.array([2.0]))


def test_sample_ray_xyz_rejects_bad_arguments() -> None:
    """Ray sampling should reject non-positive sample count and zero direction."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, "Scalar", query_space="xyz")
    with pytest.raises(ValueError, match="n_samples must be positive"):
        OctreeRayInterpolator(interp).sample(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0, 1.0, 0)
    with pytest.raises(ValueError, match="direction_xyz must be finite and non-zero"):
        OctreeRayInterpolator(interp).sample(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.0, 1.0, 10)


def test_ray_linear_pieces_rejects_zero_direction() -> None:
    """Piecewise linear ray decomposition should reject zero direction."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, "Scalar", query_space="xyz")
    with pytest.raises(ValueError, match="direction_xyz must be finite and non-zero"):
        OctreeRayInterpolator(interp).linear_pieces(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.0, 1.0)


def test_interpolator_outside_queries_use_fill_value_and_minus_one_cell_id() -> None:
    """Outside-domain queries should return fill values and `cell_id=-1`."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, "Scalar", query_space="xyz", fill_value=-77.0)
    q = np.array(
        [
            [1e6, 0.0, 0.0],
            [-1e6, 0.0, 0.0],
        ]
    )
    vals, cids = interp(q, return_cell_ids=True)
    assert np.all(cids == -1)
    assert np.allclose(vals, -77.0, atol=0.0, rtol=0.0)


def test_interpolator_vector_fill_value_is_applied_outside_domain() -> None:
    """Vector-valued fill should broadcast correctly for outside-domain queries."""
    ds = _build_fake_dataset()
    scalar = ds.variable("Scalar")
    vec = np.column_stack((scalar, 2.0 * scalar + 3.0))
    fill = np.array([-5.0, 8.0])
    interp = OctreeInterpolator(ds, vec, query_space="xyz", fill_value=fill)
    q = np.array([[1e6, 0.0, 0.0]])
    vals, cids = interp(q, return_cell_ids=True)
    assert vals.shape == (1, 2)
    assert int(cids[0]) == -1
    assert np.allclose(vals[0], fill, atol=0.0, rtol=0.0)


def test_interpolator_rpa_wrap_equivalence_on_resolvable_point() -> None:
    """`rpa` interpolation should treat azimuth `phi` and `phi + 2pi` equivalently."""
    ds = _build_fake_dataset()
    tree = Octree.from_dataset(ds, coord_system="rpa")
    interp = OctreeInterpolator(ds, "Scalar", query_space="rpa", tree=tree)
    r, polar, azimuth = _first_resolvable_rpa(tree)
    q0 = np.array([[r, polar, azimuth]])
    q1 = np.array([[r, polar, azimuth + 2.0 * math.pi]])
    v0, c0 = interp(q0, return_cell_ids=True)
    v1, c1 = interp(q1, return_cell_ids=True)
    assert np.array_equal(c0, c1)
    assert np.allclose(v0, v1, atol=1e-12, rtol=0.0)
