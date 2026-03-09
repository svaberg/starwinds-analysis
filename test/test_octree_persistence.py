from __future__ import annotations

import json

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from starwinds_analysis.data.samples import data_file
from starwinds_analysis.octree_interpolator import OCTREE_FILE_VERSION
from starwinds_analysis.octree_interpolator import Octree
from starwinds_analysis.octree_interpolator import SphericalOctree


@pytest.fixture(scope="module")
def tree_dataset_pair() -> tuple[Octree, Dataset]:
    """Return one built octree and source dataset for persistence tests."""
    input_file = data_file("difflevels-3d__var_1_n00000000.dat")
    assert input_file.exists(), f"Missing sample file: {input_file}"
    ds = Dataset.from_file(str(input_file))
    tree = Octree.from_dataset(ds, coord_system="rpa")
    return tree, ds


def test_octree_save_load_roundtrip_preserves_core_arrays(tree_dataset_pair, tmp_path) -> None:
    """Round-trip save/load preserves core octree metadata arrays."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "persist" / "tree_roundtrip.npz"
    tree.save(path)

    loaded = Octree.load(path, ds=ds)
    assert isinstance(loaded, SphericalOctree)
    assert loaded.leaf_shape == tree.leaf_shape
    assert loaded.root_shape == tree.root_shape
    assert loaded.level_counts == tree.level_counts
    assert loaded.depth == tree.depth
    assert loaded.coord_system == tree.coord_system

    assert loaded.cell_levels is not None and tree.cell_levels is not None
    assert loaded.center_phi is not None and tree.center_phi is not None
    assert loaded.delta_phi is not None and tree.delta_phi is not None
    assert loaded.expected_delta_phi is not None and tree.expected_delta_phi is not None
    assert np.array_equal(loaded.cell_levels, tree.cell_levels)
    assert np.allclose(loaded.center_phi, tree.center_phi)
    assert np.allclose(loaded.delta_phi, tree.delta_phi)
    assert np.allclose(loaded.expected_delta_phi, tree.expected_delta_phi)


def test_octree_load_without_dataset_requires_bind_for_lookup(tree_dataset_pair, tmp_path) -> None:
    """Loading without dataset keeps metadata but blocks lookup until bound."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "tree_unbound.npz"
    tree.save(path)

    loaded = Octree.load(path)
    assert loaded.ds is None
    assert loaded.corners is not None

    with pytest.raises(ValueError, match="not bound to a dataset"):
        loaded.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), space="xyz")

    loaded.bind(ds)
    hit = loaded.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), space="xyz")
    assert hit is not None


def test_octree_save_without_corners_still_loads_and_binds(tree_dataset_pair, tmp_path) -> None:
    """`include_corners=False` persists a lighter file that can still be rebound."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "tree_no_corners.npz"
    tree.save(path, include_corners=False)
    loaded = Octree.load(path)

    assert loaded.corners is None
    loaded.bind(ds)
    hit = loaded.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), space="xyz")
    assert hit is not None


def test_octree_load_rejects_unsupported_file_version(tree_dataset_pair, tmp_path) -> None:
    """Loader rejects persisted files with unknown serialization version."""
    tree, _ds = tree_dataset_pair
    good_path = tmp_path / "tree_good.npz"
    bad_path = tmp_path / "tree_bad_version.npz"
    tree.save(good_path)

    payload: dict[str, np.ndarray] = {}
    with np.load(good_path, allow_pickle=False) as data:
        for key in data.files:
            payload[key] = np.array(data[key], copy=True)
    meta = json.loads(str(payload["meta_json"]))
    meta["version"] = int(OCTREE_FILE_VERSION) + 100
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(bad_path, **payload)

    with pytest.raises(ValueError, match="Unsupported octree file version"):
        Octree.load(bad_path)


def test_octree_load_rejects_unsupported_coord_system(tree_dataset_pair, tmp_path) -> None:
    """Loader should reject serialized metadata with unknown coordinate-system tags."""
    tree, _ds = tree_dataset_pair
    good_path = tmp_path / "tree_good_coords.npz"
    bad_path = tmp_path / "tree_bad_coords.npz"
    tree.save(good_path)

    payload: dict[str, np.ndarray] = {}
    with np.load(good_path, allow_pickle=False) as data:
        for key in data.files:
            payload[key] = np.array(data[key], copy=True)
    meta = json.loads(str(payload["meta_json"]))
    meta["coord_system"] = "bad_coords"
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(bad_path, **payload)

    with pytest.raises(ValueError, match="Unsupported coord_system"):
        Octree.load(bad_path)
