from __future__ import annotations

import math

import numpy as np
import pytest

from starwinds_analysis.octree_interpolator import Octree
from starwinds_analysis.octree_interpolator import CartesianOctree
from starwinds_analysis.octree_interpolator import OctreeInterpolator
from starwinds_analysis.octree_interpolator import OctreeBuilder
from starwinds_analysis.octree_interpolator import build_octree


class _FakeDataset:
    """Minimal dataset-like object for octree-builder edge tests."""

    def __init__(
        self,
        points: np.ndarray,
        corners: np.ndarray | None,
        variables: dict[str, np.ndarray],
        *,
        aux: dict[str, str] | None = None,
    ) -> None:
        """Store geometry/fields with Dataset-compatible attribute names."""
        self.points = points
        self.corners = corners
        self._variables = variables
        self.variables = list(variables.keys())
        self.aux = {} if aux is None else dict(aux)

    def variable(self, name: str) -> np.ndarray:
        """Return one named variable array."""
        return self._variables[name]


def _build_regular_dataset(
    *,
    nr: int = 2,
    ntheta: int = 4,
    nphi: int = 8,
) -> _FakeDataset:
    """Build a small regular spherical hexahedral dataset."""
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
    scalar = 1.5 * x - 0.7 * y + 0.2 * z + 3.0
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


def _build_regular_xyz_dataset(
    *,
    nx: int = 4,
    ny: int = 3,
    nz: int = 2,
) -> _FakeDataset:
    """Build a small regular Cartesian hexahedral dataset."""
    x_edges = np.linspace(-2.0, 2.0, nx + 1)
    y_edges = np.linspace(-1.5, 1.5, ny + 1)
    z_edges = np.linspace(-1.0, 1.0, nz + 1)

    node_index = -np.ones((nx + 1, ny + 1, nz + 1), dtype=np.int64)
    xyz_list: list[tuple[float, float, float]] = []
    node_id = 0
    for ix in range(nx + 1):
        x = float(x_edges[ix])
        for iy in range(ny + 1):
            y = float(y_edges[iy])
            for iz in range(nz + 1):
                z = float(z_edges[iz])
                xyz_list.append((x, y, z))
                node_index[ix, iy, iz] = node_id
                node_id += 1

    corners: list[list[int]] = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                corners.append(
                    [
                        int(node_index[ix, iy, iz]),
                        int(node_index[ix + 1, iy, iz]),
                        int(node_index[ix, iy + 1, iz]),
                        int(node_index[ix + 1, iy + 1, iz]),
                        int(node_index[ix, iy, iz + 1]),
                        int(node_index[ix + 1, iy, iz + 1]),
                        int(node_index[ix, iy + 1, iz + 1]),
                        int(node_index[ix + 1, iy + 1, iz + 1]),
                    ]
                )

    points = np.array(xyz_list)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = 1.2 * x - 0.3 * y + 0.8 * z + 0.5
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


@pytest.fixture(scope="module")
def cartesian_octree_context() -> tuple[_FakeDataset, CartesianOctree, OctreeInterpolator]:
    """Build one reusable Cartesian octree/interpolator context for xyz-path tests."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, coord_system="xyz")
    assert isinstance(tree, CartesianOctree)
    interp = OctreeInterpolator(ds, "Scalar", query_space="xyz", tree=tree)
    return ds, tree, interp


def test_cartesian_fixture_builds_xyz_tree(cartesian_octree_context) -> None:
    """Fixture should provide a bound Cartesian tree and xyz interpolator."""
    _ds, tree, interp = cartesian_octree_context
    assert isinstance(tree, CartesianOctree)
    assert tree.coord_system == "xyz"
    assert interp.tree is tree


def test_cartesian_lookup_hits_cell_centers(cartesian_octree_context) -> None:
    """Cartesian lookup should resolve each cell center to its own cell id."""
    _ds, tree, _interp = cartesian_octree_context
    centers = np.array(tree.lookup._cell_centers, dtype=float)
    for cid in range(centers.shape[0]):
        q = centers[cid]
        hit = tree.lookup_point(q, space="xyz")
        assert hit is not None
        assert int(hit.cell_id) == int(cid)


def test_cartesian_interpolation_matches_linear_xyz_field(cartesian_octree_context) -> None:
    """Cartesian trilinear interpolation should reconstruct the synthetic linear xyz field."""
    _ds, tree, interp = cartesian_octree_context
    lookup = tree.lookup
    rng = np.random.default_rng(42)
    n_cells = int(lookup._cell_centers.shape[0])
    choose = rng.choice(n_cells, size=min(20, n_cells), replace=False)

    q = np.empty((choose.size, 3), dtype=float)
    expected = np.empty(choose.size, dtype=float)
    for i, cid in enumerate(choose.tolist()):
        u, v, w = rng.uniform(0.1, 0.9, size=3)
        x = float(lookup._cell_x_min[cid] + u * (lookup._cell_x_max[cid] - lookup._cell_x_min[cid]))
        y = float(lookup._cell_y_min[cid] + v * (lookup._cell_y_max[cid] - lookup._cell_y_min[cid]))
        z = float(lookup._cell_z_min[cid] + w * (lookup._cell_z_max[cid] - lookup._cell_z_min[cid]))
        q[i] = (x, y, z)
        expected[i] = 1.2 * x - 0.3 * y + 0.8 * z + 0.5

    values, cell_ids = interp(q, return_cell_ids=True)
    assert np.array_equal(np.array(cell_ids, dtype=np.int64), np.array(choose, dtype=np.int64))
    assert np.allclose(np.array(values, dtype=float), expected, atol=1e-12, rtol=0.0)


def test_builder_build_rejects_missing_corners() -> None:
    """Builder should fail fast when dataset has no corners."""
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    ds = _FakeDataset(
        points=points,
        corners=None,
        variables={"X [R]": points[:, 0], "Y [R]": points[:, 1], "Z [R]": points[:, 2]},
    )
    with pytest.raises(ValueError, match="Dataset has no corners"):
        OctreeBuilder().build(ds)


def test_builder_build_rejects_unknown_coord_system() -> None:
    """Builder should reject unsupported coordinate-system identifiers."""
    ds = _build_regular_dataset()
    with pytest.raises(ValueError, match="Unsupported coord_system"):
        OctreeBuilder().build(ds, coord_system="foo")


def test_builder_build_xyz_returns_cartesian_octree() -> None:
    """Builder should construct Cartesian octree when coord_system='xyz'."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, coord_system="xyz")
    assert isinstance(tree, CartesianOctree)
    assert tree.coord_system == "xyz"


def test_builder_build_default_returns_cartesian_octree_subclass() -> None:
    """Default build path should return the Cartesian octree specialization."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds)
    assert isinstance(tree, CartesianOctree)


def test_builder_compute_phi_levels_rejects_missing_phi_source() -> None:
    """Phi-level computation should reject datasets lacking phi source fields."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    corners = np.array([[0, 1, 2]], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={"X [R]": points[:, 0], "Z [R]": points[:, 2]},
    )
    with pytest.raises(ValueError, match="Could not determine phi"):
        OctreeBuilder().compute_phi_levels(ds)


def test_builder_compute_phi_levels_rejects_bad_corner_rank() -> None:
    """Phi-level computation should reject non-2D corner arrays."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    corners = np.array([0, 1, 2], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={"X [R]": points[:, 0], "Y [R]": points[:, 1], "Z [R]": points[:, 2]},
    )
    with pytest.raises(ValueError, match="Expected 2D corner array"):
        OctreeBuilder().compute_phi_levels(ds)


def test_builder_compute_phi_levels_rejects_too_few_corners_per_cell() -> None:
    """Phi-level computation should reject cells with fewer than 3 corners."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    corners = np.array([[0, 1]], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={"X [R]": points[:, 0], "Y [R]": points[:, 1], "Z [R]": points[:, 2]},
    )
    with pytest.raises(ValueError, match="Need at least 3 corners per cell"):
        OctreeBuilder().compute_phi_levels(ds)


def test_builder_infer_levels_marks_non_dyadic_span_invalid() -> None:
    """Non-dyadic delta-phi spans should map to level -1."""
    levels = OctreeBuilder().infer_levels_from_delta_phi(np.array([1.0, 0.5, 0.3]))
    assert np.array_equal(levels, np.array([0, 1, -1], dtype=np.int64))


def test_builder_build_tree_rejects_all_invalid_levels() -> None:
    """Tree construction should fail when all provided levels are invalid."""
    ds = _build_regular_dataset()
    builder = OctreeBuilder()
    delta_phi, _center_phi, _cell_levels, _expected, _coarse = builder.compute_phi_levels(ds)
    all_invalid = np.full(delta_phi.shape, -1, dtype=np.int64)
    with pytest.raises(ValueError, match="No valid \\(>=0\\) levels available to infer octree"):
        builder.build_tree(ds, ds.corners, delta_phi, cell_levels=all_invalid)


def test_builder_handles_incompatible_blocks_aux_without_block_tree() -> None:
    """Incompatible BLOCKS metadata should not crash and should skip block-tree fields."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "7 3x5x9"
    tree = OctreeBuilder().build(ds, coord_system="rpa")
    assert tree.block_shape is None
    assert tree.block_cell_shape is None
    assert tree.block_root_shape is None
    assert tree.block_level_counts is None


def test_octree_depth_for_level_rejects_unsupported_level() -> None:
    """Depth mapping should reject levels too far below supported range."""
    tree = OctreeBuilder().build(_build_regular_dataset(), coord_system="rpa")
    bad_level = int(tree.min_level - tree.depth - 1)
    with pytest.raises(ValueError, match="Derived negative tree depth"):
        tree.depth_for_level(bad_level)


def test_octree_trace_ray_returns_empty_for_non_increasing_interval() -> None:
    """Ray trace should return empty when `t_end <= t_start`."""
    tree = OctreeBuilder().build(_build_regular_dataset(), coord_system="rpa")
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    assert tree.trace_ray(origin, direction, 1.0, 1.0) == []
    assert tree.trace_ray(origin, direction, 2.0, 1.0) == []


def test_build_octree_helper_returns_unbound_tree_until_bind() -> None:
    """`build_octree` helper should return unbound tree requiring explicit bind."""
    ds = _build_regular_dataset()
    builder = OctreeBuilder()
    delta_phi, center_phi, cell_levels, expected, coarse = builder.compute_phi_levels(ds)
    tree = build_octree(
        ds,
        ds.corners,
        delta_phi,
        coord_system="rpa",
        cell_levels=cell_levels,
    )
    tree.center_phi = center_phi
    tree.expected_delta_phi = expected
    tree.coarse_delta_phi = float(coarse)
    with pytest.raises(ValueError, match="not bound to a dataset"):
        tree.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), space="xyz")
    tree.bind(ds)
    assert tree.corners is not None


def test_build_octree_helper_stores_coord_system_metadata() -> None:
    """Helper should store requested coordinate-system metadata in the tree."""
    ds = _build_regular_dataset()
    builder = OctreeBuilder()
    delta_phi, _center_phi, cell_levels, _expected, _coarse = builder.compute_phi_levels(ds)
    tree = build_octree(
        ds,
        ds.corners,
        delta_phi,
        coord_system="xyz",
        cell_levels=cell_levels,
    )
    assert isinstance(tree, CartesianOctree)
    assert tree.coord_system == "xyz"


def test_lookup_runs_for_xyz_coord_system() -> None:
    """Lookup APIs should run when the tree is tagged as Cartesian."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, coord_system="xyz")
    hit_xyz = tree.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), space="xyz")
    assert hit_xyz is not None
    assert not hasattr(tree, "lookup_rpa")
