from __future__ import annotations

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from starwinds_analysis.data.samples import data_file
from starwinds_analysis.octree import Octree
from starwinds_analysis.octree import OctreeInterpolator
from starwinds_analysis.octree import OctreeRayTracer
from starwinds_analysis.octree import SphericalOctree


@pytest.fixture(scope="module")
def advanced_context() -> tuple[Dataset, Octree]:
    """Build reusable dataset/tree pair for advanced octree behavior checks."""
    input_file = data_file("difflevels-3d__var_1_n00000000.dat")
    assert input_file.exists(), f"Missing sample file: {input_file}"
    ds = Dataset.from_file(str(input_file))
    tree = Octree.from_dataset(ds, coord_system="rpa")
    return ds, tree


def _select_center_queries(tree: Octree, *, n_query: int, seed: int) -> np.ndarray:
    """Pick deterministic random cell centers as robust inside-domain query points."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(tree.lookup._cell_centers, dtype=float)
    n = min(int(n_query), int(centers.shape[0]))
    idx = rng.choice(centers.shape[0], size=n, replace=False)
    return centers[idx]


@pytest.mark.slow
def test_lookup_xyz_rpa_consistency_many_points(advanced_context) -> None:
    """Many interior points should map to the same cell in xyz and rpa lookup spaces."""
    _ds, tree = advanced_context
    queries = _select_center_queries(tree, n_query=64, seed=1)

    for q in queries:
        hit_xyz = tree.lookup_point(q, space="xyz")
        assert hit_xyz is not None
        r, polar, azimuth = SphericalOctree.xyz_to_rpa(q)
        hit_rpa = tree.lookup_point(np.array([r, polar, azimuth], dtype=float), space="rpa")
        assert hit_rpa is not None
        assert int(hit_xyz.cell_id) == int(hit_rpa.cell_id)


def test_trace_ray_segments_are_ordered_and_inside_cells(advanced_context) -> None:
    """Ray traversal segments must be monotone and contain their midpoint sample."""
    _ds, tree = advanced_context
    centers = np.asarray(tree.lookup._cell_centers, dtype=float)
    center_r = np.linalg.norm(centers, axis=1)
    start_id = int(np.argmin(np.abs(center_r - 1.0)))
    origin = np.asarray(centers[start_id], dtype=float)
    direction = np.array([1.0, 0.32, 0.11], dtype=float)
    t_start = 0.0
    t_end = 6.5

    segments = OctreeRayTracer(tree).trace(origin, direction, t_start, t_end)
    assert segments, "Expected at least one traversed segment."

    ray_dir = direction / np.linalg.norm(direction)
    prev_exit = float(t_start)
    for seg in segments:
        assert float(seg.t_exit) >= float(seg.t_enter)
        assert float(seg.t_enter) >= prev_exit - 1e-6
        prev_exit = float(seg.t_exit)
        mid_t = 0.5 * (float(seg.t_enter) + float(seg.t_exit))
        p_mid = origin + mid_t * ray_dir
        assert tree.contains_cell(int(seg.cell_id), p_mid, space="xyz", tol=1e-6)
    assert float(segments[0].t_enter) >= float(t_start) - 1e-8
    assert float(segments[-1].t_exit) <= float(t_end) + 1e-6


@pytest.mark.slow
def test_loaded_tree_matches_original_ray_walk(advanced_context, tmp_path) -> None:
    """Persisted/reloaded tree should produce equivalent ray traversal segments."""
    _ds, tree = advanced_context
    path = tmp_path / "advanced_ray_tree.npz"
    tree.save(path)
    loaded = Octree.load(path, ds=tree.ds)

    centers = np.asarray(tree.lookup._cell_centers, dtype=float)
    center_r = np.linalg.norm(centers, axis=1)
    start_id = int(np.argmin(np.abs(center_r - 1.0)))
    origin = np.asarray(centers[start_id], dtype=float)
    direction = np.array([1.0, 0.32, 0.11], dtype=float)
    t_start = 0.0
    t_end = 6.5

    seg_a = OctreeRayTracer(tree).trace(origin, direction, t_start, t_end)
    seg_b = OctreeRayTracer(loaded).trace(origin, direction, t_start, t_end)
    assert len(seg_a) == len(seg_b)
    for a, b in zip(seg_a, seg_b):
        assert int(a.cell_id) == int(b.cell_id)
        assert np.isclose(float(a.t_enter), float(b.t_enter), atol=1e-8, rtol=0.0)
        assert np.isclose(float(a.t_exit), float(b.t_exit), atol=1e-8, rtol=0.0)


@pytest.mark.slow
def test_interpolator_matches_when_using_loaded_tree(advanced_context, tmp_path) -> None:
    """Interpolator outputs should be equal when using original vs loaded tree."""
    ds, tree = advanced_context
    path = tmp_path / "advanced_interp_tree.npz"
    tree.save(path)
    loaded = Octree.load(path, ds=ds)

    interp_a = OctreeInterpolator(ds, "Rho [g/cm^3]", query_space="xyz", tree=tree)
    interp_b = OctreeInterpolator(ds, "Rho [g/cm^3]", query_space="xyz", tree=loaded)

    queries = _select_center_queries(tree, n_query=64, seed=7)
    vals_a, cids_a = interp_a(queries, return_cell_ids=True)
    vals_b, cids_b = interp_b(queries, return_cell_ids=True)

    assert np.array_equal(cids_a, cids_b)
    assert np.allclose(vals_a, vals_b, atol=0.0, rtol=0.0, equal_nan=True)
