from __future__ import annotations

import math

import numpy as np
import pytest

from starwinds_analysis.octree import Octree
from starwinds_analysis.octree import OctreeInterpolator
from starwinds_analysis.octree import OctreeRayTracer


class _FakeDataset:
    """Minimal dataset-like object for edge-case tests."""

    def __init__(
        self,
        points: np.ndarray,
        corners: np.ndarray | None,
        variables: dict[str, np.ndarray],
    ) -> None:
        """Store geometry/field arrays with a `Dataset`-compatible API."""
        self.points = points
        self.corners = corners
        self._variables = variables
        self.variables = list(variables.keys())
        self.aux: dict[str, str] = {}

    def variable(self, name: str) -> np.ndarray:
        """Return one variable array by name."""
        return self._variables[name]


def _build_regular_fake_dataset(
    *,
    nr: int = 1,
    ntheta: int = 2,
    nphi: int = 4,
) -> _FakeDataset:
    """Build a small regular spherical dataset suitable for lookup/interp tests."""
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
    scalar = 2.0 * x - 1.0 * y + 0.5 * z + 7.0
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


def _build_axis_only_fake_dataset() -> _FakeDataset:
    """Build an intentionally invalid dataset where all corners are on axis."""
    z = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    points = np.column_stack((x, y, z))
    corners = np.arange(8, dtype=np.int64).reshape(1, 8)
    scalar = 0.25 * z + 1.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            "X [R]": x,
            "Y [R]": y,
            "Z [R]": z,
            "Scalar": scalar,
        },
    )


def test_fake_axis_only_dataset_is_rejected() -> None:
    """Axis-only cells should fail octree build because no valid phi levels exist."""
    ds = _build_axis_only_fake_dataset()
    with pytest.raises(ValueError, match="No valid \\(>=0\\) levels"):
        Octree.from_dataset(ds, coord_system="rpa")


def test_fake_lookup_rejects_invalid_queries() -> None:
    """Lookup should return None for non-finite or invalid-angle queries."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, coord_system="rpa")

    assert tree.lookup_point(np.array([float("nan"), 0.0, 0.0], dtype=float), space="xyz") is None
    assert tree.lookup_point(np.array([float("inf"), 0.0, 0.0], dtype=float), space="xyz") is None
    assert tree.lookup_point(np.array([1.5, -1e-6, 0.0], dtype=float), space="rpa") is None
    assert tree.lookup_point(np.array([1.5, math.pi + 1e-6, 0.0], dtype=float), space="rpa") is None
    assert tree.lookup_point(np.array([float("nan"), 1.0, 0.0], dtype=float), space="rpa") is None


def test_fake_trace_ray_zero_direction_raises() -> None:
    """Ray trace should reject zero-length direction vectors."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, coord_system="rpa")
    with pytest.raises(ValueError, match="direction_xyz must be finite and non-zero"):
        OctreeRayTracer(tree).trace(
            origin_xyz=np.array([1.0, 0.0, 0.0]),
            direction_xyz=np.array([0.0, 0.0, 0.0]),
            t_start=0.0,
            t_end=1.0,
        )


def test_fake_interpolator_fill_for_invalid_points() -> None:
    """Interpolator should emit fill value and cell_id=-1 for invalid queries."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, coord_system="rpa")
    interp = OctreeInterpolator(ds, ["Scalar"], query_space="xyz", tree=tree, fill_value=-123.0)

    invalid = np.array(
        [
            [float("nan"), 0.0, 0.0],
            [float("inf"), 0.0, 0.0],
            [-float("inf"), 0.0, 0.0],
        ]
    )
    q = invalid

    vals, cids = interp(q, return_cell_ids=True)
    assert np.all(cids == -1)
    assert np.allclose(vals, -123.0, atol=0.0, rtol=0.0)


def test_fake_bind_without_corners_is_rejected() -> None:
    """Binding a tree to a dataset with missing corners should fail clearly."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, coord_system="rpa")
    ds_no_corners = _FakeDataset(ds.points, None, ds._variables)

    with pytest.raises(ValueError, match="Dataset has no corners"):
        tree.bind(ds_no_corners)
