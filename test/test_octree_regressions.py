from __future__ import annotations

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from starwinds_analysis.data.samples import data_file
from starwinds_analysis.octree_interpolator import Octree
from starwinds_analysis.octree_interpolator import SphericalOctree


@pytest.fixture(scope="module")
def regression_context() -> tuple[Dataset, Octree]:
    """Build one dataset/tree pair used for regression checks."""
    input_file = data_file("difflevels-3d__var_1_n00000000.dat")
    assert input_file.exists(), f"Missing sample file: {input_file}"
    ds = Dataset.from_file(str(input_file))
    tree = Octree.from_dataset(ds, coord_system="rpa")
    return ds, tree


def test_regression_xyz_to_rpa_is_stable_and_finite() -> None:
    """Regression: xyz->rpa conversion should be finite and non-recursive."""
    q = np.array([1.0, 0.0, 0.0], dtype=float)
    r, polar, azimuth = SphericalOctree.xyz_to_rpa(q)
    assert np.isfinite(r)
    assert np.isfinite(polar)
    assert np.isfinite(azimuth)
    assert np.isclose(r, 1.0, rtol=0.0, atol=1e-15)
    assert np.isclose(polar, np.pi / 2.0, rtol=0.0, atol=1e-15)
    assert np.isclose(azimuth, 0.0, rtol=0.0, atol=1e-15)


def test_regression_lookup_outside_domain_returns_none(regression_context) -> None:
    """Regression: lookup outside radial domain should not snap to nearest cell."""
    _ds, tree = regression_context
    r_max = float(tree.lookup._r_max)
    q = np.array([r_max + 50.0, 0.0, 0.0], dtype=float)
    hit = tree.lookup_point(q, space="xyz")
    assert hit is None


def test_regression_trace_ray_from_outside_returns_empty(regression_context) -> None:
    """Regression: ray trace started outside the domain should return no segments."""
    _ds, tree = regression_context
    r_max = float(tree.lookup._r_max)
    origin = np.array([r_max + 25.0, 0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    segments = tree.trace_ray(origin, direction, 0.0, 10.0)
    assert segments == []


def test_regression_load_without_corners_uses_dataset_corners(tmp_path, regression_context) -> None:
    """Regression: loading no-corner files with ds must bind via ds.corners."""
    ds, tree = regression_context
    path = tmp_path / "tree_no_corners_regression.npz"
    tree.save(path, include_corners=False)

    loaded = Octree.load(path, ds=ds)
    assert loaded.corners is not None

    # Ensure lookups are functional without an explicit corners argument.
    q = np.array([1.0, 0.0, 0.0], dtype=float)
    hit = loaded.lookup_point(q, space="xyz")
    assert hit is not None
