#!/usr/bin/env python3
"""Octree orchestration builder and coordinate-agnostic utilities."""

from __future__ import annotations

from collections import Counter
from typing import TypeAlias

import numpy as np
from starwinds_readplt.dataset import Dataset

from .base import CoordSystem
from .base import DEFAULT_AXIS_RHO_TOL
from .base import DEFAULT_COORD_SYSTEM
from .base import GridShape
from .base import LevelCountTable
from .base import Octree
from .base import SUPPORTED_COORD_SYSTEMS
from .base import octree_class_for_coord
from .builder_cartesian import CartesianOctreeBuilder
from .builder_spherical import SphericalOctreeBuilder

LevelShapeStatsRow: TypeAlias = tuple[int, int, float, float, int]
"""Tuple `(n_axis1, n_axis2, d_axis1, d_axis2, n_cells_at_level)`."""

LevelShapeStatsMap: TypeAlias = dict[int, LevelShapeStatsRow]
"""Map `level -> LevelShapeStatsRow`."""


def point_refinement_levels(
    n_points: int,
    corners: np.ndarray,
    cell_levels: np.ndarray,
) -> np.ndarray:
    """Assign each mesh point the finest adjacent valid cell level.

    Consumes:
    - Number of points, corner connectivity, and per-cell levels.
    Returns:
    - Point-level array of length `n_points` (`-1` where unknown).
    """
    out = np.full(n_points, -1, dtype=np.int64)
    for cell_id, nodes in enumerate(corners):
        level = int(cell_levels[cell_id])
        if level < 0:
            continue
        out[nodes] = np.maximum(out[nodes], level)
    return out


def format_histogram(levels: np.ndarray) -> str:
    """Format level histogram text as `level:count` pairs."""
    counts = Counter(int(v) for v in levels.tolist())
    return ", ".join(f"{lvl}:{counts[lvl]}" for lvl in sorted(counts))


def valid_cell_fraction(levels: np.ndarray) -> tuple[int, int, float]:
    """Compute valid-level fraction statistics for `levels >= 0`."""
    total = int(levels.size)
    valid = int(np.count_nonzero(levels >= 0))
    frac = float(valid / total) if total > 0 else 0.0
    return valid, total, frac


class OctreeBuilder:
    """Build `Octree` objects from dataset geometry using coord strategies."""

    def __init__(
        self,
        *,
        level_rtol: float = 1e-4,
        level_atol: float = 1e-9,
    ) -> None:
        """Configure tolerances used for dyadic level inference."""
        self.level_rtol = float(level_rtol)
        self.level_atol = float(level_atol)
        self._rpa_builder = SphericalOctreeBuilder(level_rtol=level_rtol, level_atol=level_atol)
        self._xyz_builder = CartesianOctreeBuilder(level_rtol=level_rtol, level_atol=level_atol)

    def infer_levels_from_delta_phi(self, delta_phi: np.ndarray) -> np.ndarray:
        """Infer dyadic refinement levels from per-cell `delta_phi` spans."""
        return self._rpa_builder.infer_levels_from_delta_phi(delta_phi)

    def compute_phi_levels(
        self,
        ds: Dataset,
        *,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute per-cell azimuth spans and dyadic levels from dataset geometry."""
        return self._rpa_builder.compute_phi_levels(ds, axis_rho_tol=axis_rho_tol)

    @staticmethod
    def _twos_factor(n: int) -> int:
        """Compute the exponent of the largest power of two dividing `n`."""
        k = 0
        while n > 0 and (n % 2 == 0):
            n //= 2
            k += 1
        return k

    @staticmethod
    def _full_tree_counts(leaf_shape: GridShape) -> tuple[LevelCountTable, GridShape, int]:
        """Compute full-tree counts, root shape, and depth from finest leaf shape."""
        depth = min(
            OctreeBuilder._twos_factor(leaf_shape[0]),
            OctreeBuilder._twos_factor(leaf_shape[1]),
            OctreeBuilder._twos_factor(leaf_shape[2]),
        )
        root_shape = (
            leaf_shape[0] >> depth,
            leaf_shape[1] >> depth,
            leaf_shape[2] >> depth,
        )
        base = int(root_shape[0] * root_shape[1] * root_shape[2])
        counts = tuple((level, base * (8**level), base * (8**level)) for level in range(depth + 1))
        return counts, root_shape, depth

    @staticmethod
    def _infer_leaf_shape_from_levels(
        level_shapes: LevelShapeStatsMap,
    ) -> tuple[GridShape, int, int]:
        """Infer finest leaf shape from per-level shape/count statistics."""
        max_level = max(level_shapes)
        n_axis1_f = int(level_shapes[max_level][0])
        n_axis2_f = int(level_shapes[max_level][1])
        weighted_cells = 0
        for level, (_n_axis1, _n_axis2, _d_axis1, _d_axis2, count) in level_shapes.items():
            weighted_cells += int(count) * (8 ** int(max_level - level))

        denom = int(n_axis1_f * n_axis2_f)
        if denom <= 0:
            raise ValueError("Invalid finest angular denominator while inferring n_axis0.")
        n_axis0_float = weighted_cells / float(denom)
        n_axis0 = int(round(n_axis0_float))
        if not np.isclose(n_axis0_float, float(n_axis0), rtol=0.0, atol=1e-9):
            raise ValueError(
                "Could not infer integer finest n_axis0 from weighted cell counts: "
                f"weighted={weighted_cells}, n_axis1={n_axis1_f}, n_axis2={n_axis2_f}."
            )
        return (n_axis0, n_axis1_f, n_axis2_f), int(weighted_cells), max_level

    def build_tree(
        self,
        ds: Dataset,
        corners: np.ndarray,
        *,
        coord_system: CoordSystem = DEFAULT_COORD_SYSTEM,
        cell_levels: np.ndarray | None = None,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> Octree:
        """Build an `Octree` from dataset geometry and optional level hints."""
        tree_cls = octree_class_for_coord(coord_system)
        if coord_system == "rpa":
            level_shapes, levels, min_level, max_level = self._rpa_builder.infer_level_shapes(
                ds,
                corners,
                cell_levels=cell_levels,
                axis_rho_tol=axis_rho_tol,
            )
        elif coord_system == "xyz":
            level_shapes, levels, min_level, max_level = self._xyz_builder.infer_level_shapes(
                ds,
                corners,
                cell_levels=cell_levels,
            )
        else:
            raise ValueError(
                f"Unsupported coord_system '{coord_system}'; expected one of {SUPPORTED_COORD_SYSTEMS}."
            )

        leaf_shape, weighted_cells, _max_level = self._infer_leaf_shape_from_levels(level_shapes)
        _counts_full, root_shape, _depth = self._full_tree_counts(leaf_shape)
        level_counts = tuple(
            (
                int(level),
                int(level_shapes[level][4]),
                int(level_shapes[level][4] * (8 ** int(max_level - level))),
            )
            for level in sorted(level_shapes)
        )
        is_full = (
            int(np.count_nonzero(levels >= 0)) == int(levels.size)
            and int(sum(item[2] for item in level_counts)) == int(np.prod(leaf_shape))
            and int(weighted_cells) == int(np.prod(leaf_shape))
        )

        return tree_cls(
            leaf_shape=leaf_shape,
            root_shape=root_shape,
            is_full=bool(is_full),
            level_counts=level_counts,
            min_level=min_level,
            max_level=max_level,
            coord_system=coord_system,
            cell_levels=levels,
        )

    def build(
        self,
        ds: Dataset,
        *,
        coord_system: CoordSystem = DEFAULT_COORD_SYSTEM,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> Octree:
        """Build and bind an `Octree` directly from dataset geometry."""
        if coord_system not in SUPPORTED_COORD_SYSTEMS:
            raise ValueError(
                f"Unsupported coord_system '{coord_system}'; "
                f"expected one of {SUPPORTED_COORD_SYSTEMS}."
            )
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot build octree.")
        corners = np.asarray(ds.corners, dtype=np.int64)
        tree = self.build_tree(
            ds,
            corners,
            coord_system=coord_system,
            cell_levels=None,
            axis_rho_tol=axis_rho_tol,
        )
        tree.bind(ds, axis_rho_tol=axis_rho_tol)
        return tree


def build_octree(
    ds: Dataset,
    corners: np.ndarray,
    *,
    coord_system: CoordSystem = DEFAULT_COORD_SYSTEM,
    cell_levels: np.ndarray | None = None,
) -> Octree:
    """Build an octree from precomputed metadata without binding."""
    if coord_system not in SUPPORTED_COORD_SYSTEMS:
        raise ValueError(
            f"Unsupported coord_system '{coord_system}'; "
            f"expected one of {SUPPORTED_COORD_SYSTEMS}."
        )
    return OctreeBuilder().build_tree(
        ds,
        corners,
        coord_system=coord_system,
        cell_levels=cell_levels,
    )
