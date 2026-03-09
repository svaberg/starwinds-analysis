from __future__ import annotations

import math

import numpy as np
import pytest

from starwinds_analysis.octree import Octree
from starwinds_analysis.octree import OctreeInterpolator


class _SyntheticDataset:
    """Minimal dataset-like object supporting octree build/interpolation tests."""

    def __init__(self, points: np.ndarray, corners: np.ndarray, variables: dict[str, np.ndarray]) -> None:
        """Store geometry and field arrays with a `Dataset`-compatible API surface."""
        self.points = points
        self.corners = corners
        self._variables = variables
        self.variables = list(variables.keys())
        self.aux: dict[str, str] = {}

    def variable(self, name: str) -> np.ndarray:
        """Return one named variable array."""
        return self._variables[name]


def _build_uniform_spherical_hex_dataset(
    *,
    nr: int = 2,
    ntheta: int = 4,
    nphi: int = 8,
) -> tuple[_SyntheticDataset, np.ndarray, tuple[float, float, float, float]]:
    """Build a synthetic full-sphere structured dataset with hexahedral cells."""
    r_edges = np.linspace(1.0, 3.0, nr + 1)
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
                c000 = int(node_index[ir, it, ip])
                c100 = int(node_index[ir + 1, it, ip])
                c010 = int(node_index[ir, it + 1, ip])
                c110 = int(node_index[ir + 1, it + 1, ip])
                c001 = int(node_index[ir, it, ip + 1])
                c101 = int(node_index[ir + 1, it, ip + 1])
                c011 = int(node_index[ir, it + 1, ip + 1])
                c111 = int(node_index[ir + 1, it + 1, ip + 1])
                corners.append([c000, c100, c010, c110, c001, c101, c011, c111])

    points = np.array(xyz_list)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r_nodes = np.sqrt(x * x + y * y + z * z)
    theta_nodes = np.arccos(np.clip(z / np.maximum(r_nodes, np.finfo(float).tiny), -1.0, 1.0))
    phi_nodes = np.mod(np.arctan2(y, x), 2.0 * math.pi)

    a, b, c, d = (1.7, -0.45, 0.3, 2.1)
    linear_field = a * r_nodes + b * theta_nodes + c * phi_nodes + d

    ds = _SyntheticDataset(
        points=points,
        corners=np.array(corners, dtype=np.int64),
        variables={
            "X [R]": x,
            "Y [R]": y,
            "Z [R]": z,
            "LinField": linear_field,
        },
    )
    return ds, linear_field, (a, b, c, d)


@pytest.fixture(scope="module")
def synthetic_context() -> tuple[_SyntheticDataset, Octree, np.ndarray, tuple[float, float, float, float]]:
    """Return synthetic dataset, built tree, linear nodal field and coefficients."""
    ds, linear_field, coeffs = _build_uniform_spherical_hex_dataset()
    tree = Octree.from_dataset(ds, coord_system="rpa")
    return ds, tree, linear_field, coeffs


def _sample_inside_cells(tree: Octree, cids: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate one interior xyz and matching rpa point per requested cell id."""
    lookup = tree.lookup
    xyz_list: list[np.ndarray] = []
    rpa_list: list[np.ndarray] = []
    for cid in cids.tolist():
        r0 = float(lookup._cell_r_min[cid])
        r1 = float(lookup._cell_r_max[cid])
        t0 = float(lookup._cell_theta_min[cid])
        t1 = float(lookup._cell_theta_max[cid])
        p0 = float(lookup._cell_phi_start[cid])
        pw = float(lookup._cell_phi_width[cid])

        u = float(rng.uniform(0.15, 0.85))
        v = float(rng.uniform(0.15, 0.85))
        w = float(rng.uniform(0.15, 0.85))
        r = r0 + u * (r1 - r0)
        theta = t0 + v * (t1 - t0)
        phi = (p0 + w * pw) % (2.0 * math.pi)

        st = math.sin(theta)
        xyz = np.array([r * st * math.cos(phi), r * st * math.sin(phi), r * math.cos(theta)])
        xyz_list.append(xyz)
        rpa_list.append(np.array([r, theta, phi]))
    return np.array(xyz_list), np.array(rpa_list)


def _interpolation_valid_cells(
    tree: Octree,
    *,
    interp: OctreeInterpolator | None = None,
) -> np.ndarray:
    """Return cells suitable for stable interpolation checks."""
    lookup = tree.lookup
    phi_end = lookup._cell_phi_start + lookup._cell_phi_width
    ids = np.flatnonzero(
        (lookup._cell_theta_min > 1e-6)
        & (lookup._cell_theta_max < (math.pi - 1e-6))
        & (phi_end < (2.0 * math.pi - 1e-8))
    )
    if interp is None:
        return ids
    good = [int(cid) for cid in ids.tolist() if np.unique(interp._bin_to_corner[int(cid)]).size == 8]
    return np.array(good, dtype=np.int64)


def test_synthetic_lookup_hits_its_own_cell_centers(synthetic_context) -> None:
    """Lookup of each synthetic cell center should return the corresponding cell id."""
    _ds, tree, _field, _coeffs = synthetic_context
    centers = np.array(tree.lookup._cell_centers)
    for cid in range(centers.shape[0]):
        q = centers[cid]
        hit = tree.lookup_point(q, space="xyz")
        assert hit is not None
        assert int(hit.cell_id) == int(cid)


def test_synthetic_lookup_xyz_and_rpa_match_for_random_interior_points(synthetic_context) -> None:
    """xyz and rpa lookups should agree on synthetic interior random points."""
    _ds, tree, _field, _coeffs = synthetic_context
    rng = np.random.default_rng(11)
    valid = _interpolation_valid_cells(tree)
    choose = rng.choice(valid, size=min(120, valid.size), replace=False)
    xyz, rpa = _sample_inside_cells(tree, choose, rng)

    for i in range(xyz.shape[0]):
        hit_xyz = tree.lookup_point(xyz[i], space="xyz")
        hit_rpa = tree.lookup_point(rpa[i], space="rpa")
        assert hit_xyz is not None
        assert hit_rpa is not None
        assert int(hit_xyz.cell_id) == int(hit_rpa.cell_id)


def test_synthetic_interpolation_matches_linear_field_in_xyz(synthetic_context) -> None:
    """xyz interpolation should reconstruct an exactly linear spherical field."""
    ds, tree, _field, coeffs = synthetic_context
    a, b, c, d = coeffs
    interp = OctreeInterpolator(ds, "LinField", query_space="xyz", tree=tree)
    rng = np.random.default_rng(22)
    valid = _interpolation_valid_cells(tree, interp=interp)
    assert valid.size > 0
    choose = rng.choice(valid, size=min(200, valid.size), replace=False)
    xyz, rpa = _sample_inside_cells(tree, choose, rng)
    vals, cell_ids = interp(xyz, return_cell_ids=True)
    expected = a * rpa[:, 0] + b * rpa[:, 1] + c * rpa[:, 2] + d

    assert np.array_equal(cell_ids, choose)
    assert np.allclose(vals, expected, atol=1e-9, rtol=0.0)


def test_synthetic_interpolation_matches_linear_field_in_rpa_with_wrap(synthetic_context) -> None:
    """rpa interpolation should normalize wrapped azimuth and match linear field."""
    ds, tree, _field, coeffs = synthetic_context
    a, b, c, d = coeffs
    interp = OctreeInterpolator(ds, "LinField", query_space="rpa", tree=tree)
    rng = np.random.default_rng(33)
    valid = _interpolation_valid_cells(tree, interp=interp)
    assert valid.size > 0
    choose = rng.choice(valid, size=min(200, valid.size), replace=False)
    _xyz, rpa = _sample_inside_cells(tree, choose, rng)
    wrapped = np.array(rpa, copy=True)
    wrapped[:, 2] += 2.0 * math.pi
    vals, cell_ids = interp(wrapped, return_cell_ids=True)
    expected = a * rpa[:, 0] + b * rpa[:, 1] + c * rpa[:, 2] + d

    assert np.array_equal(cell_ids, choose)
    assert np.allclose(vals, expected, atol=1e-9, rtol=0.0)


def test_synthetic_vector_interpolation_returns_expected_shape_and_values(synthetic_context) -> None:
    """Vector-valued interpolation should preserve trailing dims and nodal exactness."""
    ds, tree, linear_field, coeffs = synthetic_context
    a, b, c, d = coeffs
    vec = np.column_stack((linear_field, 2.0 * linear_field + 1.0, np.full_like(linear_field, 5.0)))
    interp = OctreeInterpolator(ds, vec, query_space="xyz", tree=tree)

    rng = np.random.default_rng(44)
    valid = _interpolation_valid_cells(tree, interp=interp)
    assert valid.size > 0
    choose = rng.choice(valid, size=min(120, valid.size), replace=False)
    xyz, rpa = _sample_inside_cells(tree, choose, rng)
    vals = interp(xyz)

    scalar = a * rpa[:, 0] + b * rpa[:, 1] + c * rpa[:, 2] + d
    expected = np.column_stack((scalar, 2.0 * scalar + 1.0, np.full_like(scalar, 5.0)))

    assert vals.shape == expected.shape
    assert np.allclose(vals, expected, atol=1e-9, rtol=0.0)


def test_synthetic_outside_points_use_fill_value_and_negative_cell_id(synthetic_context) -> None:
    """Outside-domain synthetic points should return fill value and cell_id=-1."""
    ds, tree, _field, _coeffs = synthetic_context
    fill = -999.0
    interp = OctreeInterpolator(ds, "LinField", query_space="xyz", tree=tree, fill_value=fill)

    inside = np.array(tree.lookup._cell_centers[0]).reshape(1, 3)
    outside = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    q = np.vstack((inside, outside))

    vals, cell_ids = interp(q, return_cell_ids=True)
    assert int(cell_ids[0]) >= 0
    assert int(cell_ids[1]) == -1
    assert int(cell_ids[2]) == -1
    assert np.isclose(float(vals[1]), fill)
    assert np.isclose(float(vals[2]), fill)
