from __future__ import annotations

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from starwinds_analysis.data.samples import data_file
from starwinds_analysis.octree import DEFAULT_AXIS_RHO_TOL
from starwinds_analysis.octree import DEFAULT_MIN_VALID_CELL_FRACTION
from starwinds_analysis.octree import Octree
from starwinds_analysis.octree import OctreeBuilder
from starwinds_analysis.octree import SphericalOctree
from starwinds_analysis.octree import format_histogram
from starwinds_analysis.octree import point_refinement_levels
from starwinds_analysis.octree import valid_cell_fraction


@pytest.fixture(scope="module")
def octree_context() -> dict[str, object]:
    """Build reusable octree refinement context for the mixed-level 3D sample file."""
    input_file = data_file("difflevels-3d__var_1_n00000000.dat")
    assert input_file.exists(), f"Missing sample file: {input_file}"

    ds = Dataset.from_file(str(input_file))
    assert ds.corners is not None

    corners = np.asarray(ds.corners, dtype=np.int64)
    tree = Octree.from_dataset(
        ds,
        coord_system="rpa",
        axis_rho_tol=DEFAULT_AXIS_RHO_TOL,
        level_rtol=1e-4,
        level_atol=1e-9,
    )
    delta_phi, center_phi, _levels, expected, coarse = OctreeBuilder(
        level_rtol=1e-4,
        level_atol=1e-9,
    ).compute_phi_levels(ds, axis_rho_tol=DEFAULT_AXIS_RHO_TOL)
    assert tree.cell_levels is not None
    cell_levels = tree.cell_levels

    point_levels = point_refinement_levels(
        n_points=ds.points.shape[0],
        corners=corners,
        cell_levels=cell_levels,
    )
    lookup = tree.lookup

    return {
        "ds": ds,
        "corners": corners,
        "delta_phi": delta_phi,
        "center_phi": center_phi,
        "cell_levels": cell_levels,
        "expected": expected,
        "coarse": coarse,
        "point_levels": point_levels,
        "tree": tree,
        "lookup": lookup,
    }


def test_compute_phi_levels_shapes(octree_context: dict[str, object]) -> None:
    """Per-cell level arrays have expected sizes and finite coarse spacing."""
    corners = octree_context["corners"]
    delta_phi = octree_context["delta_phi"]
    center_phi = octree_context["center_phi"]
    cell_levels = octree_context["cell_levels"]
    expected = octree_context["expected"]
    coarse = octree_context["coarse"]

    assert delta_phi.shape[0] == corners.shape[0]
    assert center_phi.shape[0] == corners.shape[0]
    assert cell_levels.shape[0] == corners.shape[0]
    assert expected.shape[0] == corners.shape[0]
    assert np.isfinite(coarse)


def test_refinement_fraction_and_histograms(octree_context: dict[str, object]) -> None:
    """Valid-level fraction and histogram utilities behave on the sample file."""
    corners = octree_context["corners"]
    ds = octree_context["ds"]
    cell_levels = octree_context["cell_levels"]
    point_levels = octree_context["point_levels"]

    valid, total, frac_valid = valid_cell_fraction(cell_levels)
    assert total == corners.shape[0]
    assert valid > 0
    assert frac_valid >= DEFAULT_MIN_VALID_CELL_FRACTION

    assert point_levels.shape[0] == ds.points.shape[0]
    cell_hist = format_histogram(cell_levels)
    point_hist = format_histogram(point_levels)
    assert cell_hist
    assert point_hist


def test_tree_build_caches_metadata(octree_context: dict[str, object]) -> None:
    """Tree stores lookup-level metadata needed for downstream construction."""
    cell_levels = octree_context["cell_levels"]
    tree = octree_context["tree"]

    assert tree.max_level >= tree.min_level
    assert tree.max_level > tree.min_level
    assert tree.cell_levels is not None
    assert tree.cell_levels.shape == cell_levels.shape


def test_lookup_xyz_and_rpa_agree(octree_context: dict[str, object]) -> None:
    """Cartesian and spherical lookup spaces resolve the same leaf cell."""
    lookup = octree_context["lookup"]

    q_xyz = np.array([1.0, 0.0, 0.0], dtype=float)
    hit_xyz = lookup.lookup_point(q_xyz, space="xyz")
    assert hit_xyz is not None

    hit_rpa = lookup.lookup_point(np.array(SphericalOctree.xyz_to_rpa(q_xyz), dtype=float), space="rpa")
    assert hit_rpa is not None
    assert hit_xyz.cell_id == hit_rpa.cell_id


def test_octree_save_load_roundtrip(octree_context: dict[str, object], tmp_path) -> None:
    """Saved octree can be loaded and produce matching lookup hits when rebound."""
    tree = octree_context["tree"]
    ds = octree_context["ds"]
    q_xyz = np.array([1.0, 0.0, 0.0], dtype=float)

    path = tmp_path / "octree_roundtrip.npz"
    tree.save(path)
    loaded = Octree.load(path, ds=ds)

    assert loaded.leaf_shape == tree.leaf_shape
    assert loaded.root_shape == tree.root_shape
    assert loaded.depth == tree.depth
    assert loaded.level_counts == tree.level_counts
    assert loaded.cell_levels is not None
    assert tree.cell_levels is not None
    assert np.array_equal(loaded.cell_levels, tree.cell_levels)

    hit_tree = tree.lookup_point(q_xyz, space="xyz")
    hit_loaded = loaded.lookup_point(q_xyz, space="xyz")
    assert hit_tree is not None
    assert hit_loaded is not None
    assert hit_tree.cell_id == hit_loaded.cell_id
