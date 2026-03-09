#!/usr/bin/env python3
"""Core octree data structures and shared lookup/ray utilities."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
import math
from pathlib import Path
from typing import ClassVar
from typing import Literal
from typing import TypeAlias

import numpy as np
from starwinds_readplt.dataset import Dataset

DEFAULT_MIN_VALID_CELL_FRACTION = 0.5
DEFAULT_AXIS_RHO_TOL = 1e-12
OCTREE_FILE_VERSION = 1
SUPPORTED_COORD_SYSTEMS = ("rpa", "xyz")
DEFAULT_COORD_SYSTEM = "xyz"

CoordSystem: TypeAlias = Literal["rpa", "xyz"]
"""Coordinate-system tag used by octree builder/lookup dispatch."""

GridShape: TypeAlias = tuple[int, int, int]
"""Grid extents `(n_axis0, n_axis1, n_axis2)`."""

GridIndex: TypeAlias = tuple[int, int, int]
"""Discrete cell/bin index triplet `(i_axis0, i_axis1, i_axis2)`."""

GridPath: TypeAlias = tuple[GridIndex, ...]
"""Root-to-leaf sequence of `GridIndex` entries."""

TetCornerIndex: TypeAlias = tuple[int, int, int, int]
"""Four corner ids selecting one tetrahedron from an 8-corner hexahedron."""

TriFaceIndex: TypeAlias = GridIndex
"""Three local tetrahedron-corner ids selecting one triangular face."""

LevelCountRow: TypeAlias = tuple[int, int, int]
"""Tuple meaning `(level, leaf_count, fine_equivalent_count)`."""

LevelCountTable: TypeAlias = tuple[LevelCountRow, ...]
"""Sorted collection of per-level count rows."""

LevelToDepthMap: TypeAlias = dict[int, int]
"""Map `level -> tree depth index`."""

LevelToShapeMap: TypeAlias = dict[int, GridShape]
"""Map `level -> grid shape`."""

LevelToSpacingMap: TypeAlias = dict[int, float]
"""Map `level -> axis spacing` (`d_axis1` or `d_axis2`)."""

LevelAngularShapeRow: TypeAlias = tuple[int, int, float, float, int]
"""Tuple `(n_axis1, n_axis2, d_axis1, d_axis2, n_cells_at_level)`."""

LevelAngularShapeMap: TypeAlias = dict[int, LevelAngularShapeRow]
"""Map `level -> LevelAngularShapeRow`."""

BlockAux: TypeAlias = tuple[int, GridShape]
"""Parsed BLOCKS aux tuple `(n_blocks, cells_per_block_xyz)`."""

BlockShapeInference: TypeAlias = tuple[GridShape, GridShape]
"""Inferred `(cells_per_block, block_grid_shape)`."""

ScoredBlockShapeCandidate: TypeAlias = tuple[GridShape, GridShape, int]
"""Block-shape candidate `(cells_per_block, block_grid_shape, score)`."""

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi
_LOOKUP_CONTAIN_TOL = 1e-10
_DEFAULT_LOOKUP_MAX_RADIUS = 2


def _normalize_direction(direction_xyz: np.ndarray) -> np.ndarray:
    """Normalize one Cartesian ray direction.

    Consumes:
    - `direction_xyz`: array-like with 3 Cartesian components.
    Returns:
    - Unit-length `(3,)` direction vector.
    """
    d = np.array(direction_xyz, dtype=float).reshape(3)
    dn = float(np.linalg.norm(d))
    if not math.isfinite(dn) or dn == 0.0:
        raise ValueError("direction_xyz must be finite and non-zero.")
    return d / dn


class _CellLookup:
    """Minimal lookup interface shared by concrete lookup backends."""

    _cell_centers: np.ndarray

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        space: str,
        tol: float = 1e-10,
    ) -> bool:
        """Containment test of one query point against one leaf cell."""
        raise NotImplementedError

    def contains_xyz_cell(self, cell_id: int, x: float, y: float, z: float, *, tol: float = 1e-10) -> bool:
        """Containment test of one Cartesian query point against one leaf cell."""
        raise NotImplementedError

    def cell_step_hint(self, cell_id: int) -> float:
        """Return one characteristic step size used for ray marching."""
        raise NotImplementedError

    def lookup_cell_id(
        self,
        point: np.ndarray,
        *,
        space: str,
    ) -> int:
        """Resolve one query point to a leaf `cell_id` (or `-1`)."""
        raise NotImplementedError

    def lookup_point(
        self,
        point: np.ndarray,
        *,
        space: str,
    ) -> "LookupHit | None":
        """Lookup one query point and materialize `LookupHit` metadata."""
        q = np.array(point, dtype=float).reshape(3)
        chosen = self.lookup_cell_id(q, space=str(space))
        return self._hit_from_chosen(chosen)

    def _hit_from_chosen(self, chosen: int, *, allow_invalid_depth: bool = False) -> "LookupHit | None":
        """Materialize lookup metadata from one chosen cell id."""
        raise NotImplementedError


@dataclass
class Octree:
    """Adaptive octree summary plus bound lookup/ray-query entrypoints.

    `level_counts` and `block_level_counts` rows are
    `(level, leaf_count, fine_equivalent_count)`.
    """

    COORD_SYSTEM: ClassVar[str | None] = None

    leaf_shape: GridShape
    root_shape: GridShape
    depth: int
    is_full: bool
    level_counts: LevelCountTable
    min_level: int
    max_level: int
    coarse_axis1_step: float | None
    coarse_axis2_step: float | None
    block_cell_shape: GridShape | None
    block_shape: GridShape | None
    block_root_shape: GridShape | None
    block_depth: int | None
    block_level_counts: LevelCountTable | None
    coord_system: CoordSystem = DEFAULT_COORD_SYSTEM
    cell_levels: np.ndarray | None = None
    axis2_center: np.ndarray | None = None
    axis2_span: np.ndarray | None = None
    expected_axis2_span: np.ndarray | None = None
    coarse_axis2_span: float | None = None
    ds: Dataset | None = field(default=None, repr=False)
    corners: np.ndarray | None = field(default=None, repr=False)
    axis_rho_tol: float = field(default=DEFAULT_AXIS_RHO_TOL, repr=False)
    _lookup_cache: "_CellLookup | None" = field(default=None, init=False, repr=False)

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        *,
        coord_system: CoordSystem = DEFAULT_COORD_SYSTEM,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
        level_rtol: float = 1e-4,
        level_atol: float = 1e-9,
    ) -> "Octree":
        """Build and bind an octree directly from a plain dataset.

        Consumes:
        - `ds`: source dataset with points/corners.
        - Coordinate-system and level-inference tolerance parameters.
        Returns:
        - Built and bound octree instance.
        """
        if cls is not Octree and cls.COORD_SYSTEM is not None:
            if coord_system == DEFAULT_COORD_SYSTEM and cls.COORD_SYSTEM != DEFAULT_COORD_SYSTEM:
                coord_system = cls.COORD_SYSTEM
            if coord_system != cls.COORD_SYSTEM:
                raise ValueError(
                    f"{cls.__name__} requires coord_system='{cls.COORD_SYSTEM}', got '{coord_system}'."
                )
        from .builder import OctreeBuilder

        builder = OctreeBuilder(level_rtol=level_rtol, level_atol=level_atol)
        return builder.build(ds, coord_system=coord_system, axis_rho_tol=axis_rho_tol)

    @property
    def levels(self) -> tuple[int, ...]:
        """Expose sorted refinement levels present in this tree.

        Consumes:
        - `self.level_counts`.
        Returns:
        - Tuple of level integers.
        """
        return tuple(int(level) for level, _count, _expected in self.level_counts)

    @property
    def is_uniform(self) -> bool:
        """Report whether the tree has one refinement level.

        Consumes:
        - `self.min_level`, `self.max_level`.
        Returns:
        - `True` when min/max levels are equal, else `False`.
        """
        return int(self.min_level) == int(self.max_level)

    def bind(
        self,
        ds: Dataset,
        corners: np.ndarray | None = None,
        *,
        axis_rho_tol: float | None = None,
    ) -> None:
        """Bind this octree to dataset geometry for lookups/ray queries.

        Consumes:
        - `ds`: dataset with point variables.
        - Optional `corners` override and `axis_rho_tol`.
        Returns:
        - `None`; binds geometry and invalidates cached lookup state.
        """
        resolved_corners = corners
        if resolved_corners is None:
            if ds.corners is None:
                raise ValueError("Dataset has no corners; cannot bind octree lookup.")
            resolved_corners = np.array(ds.corners, dtype=np.int64)
        self.ds = ds
        self.corners = np.array(resolved_corners, dtype=np.int64)
        if axis_rho_tol is not None:
            self.axis_rho_tol = float(axis_rho_tol)
        self._lookup_cache = None

    def save(self, path: str | Path, *, include_corners: bool = True) -> None:
        """Save octree metadata to a compressed `.npz` file.

        Consumes:
        - Output `path` and optional `include_corners` flag.
        Returns:
        - `None`; writes one `.npz` persistence file.
        """
        from .persistence import OctreeArrayState
        from .persistence import OctreePersistenceState

        state = OctreePersistenceState.from_octree(self)
        arrays = OctreeArrayState.from_tree(self)
        out_path = Path(path)
        state.save_npz(
            out_path,
            arrays=arrays,
            corners=self.corners,
            include_corners=include_corners,
        )
        logger.info("Saved octree to %s", str(out_path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        ds: Dataset | None = None,
        corners: np.ndarray | None = None,
        axis_rho_tol: float | None = None,
    ) -> "Octree":
        """Load octree metadata from `.npz` and optionally bind geometry.

        Consumes:
        - Input file `path`, optional `ds`, optional `corners` override.
        Returns:
        - Loaded `Octree` (or subclass) instance.
        """
        from .persistence import OctreePersistenceState

        in_path = Path(path)
        state, array_state, saved_corners = OctreePersistenceState.load_npz(in_path)
        coord_system = str(state.coord_system)
        tree_cls = _octree_class_for_coord(coord_system)
        if cls is not Octree and cls.COORD_SYSTEM is not None and cls.COORD_SYSTEM != coord_system:
            raise ValueError(
                f"{cls.__name__} cannot load coord_system='{coord_system}'."
            )
        if cls is not Octree:
            tree_cls = cls
        tree = state.instantiate_tree(tree_cls, arrays=array_state)

        resolved_corners = corners if corners is not None else saved_corners
        if ds is not None:
            tree.bind(ds, resolved_corners, axis_rho_tol=axis_rho_tol)
        elif resolved_corners is not None:
            tree.corners = np.array(resolved_corners, dtype=np.int64)
        if axis_rho_tol is not None:
            tree.axis_rho_tol = float(axis_rho_tol)
        logger.info("Loaded octree from %s", str(in_path))
        return tree

    def depth_for_level(self, level: int) -> int:
        """Map refinement level to tree depth index.

        Consumes:
        - `level`: refinement level id.
        Returns:
        - Integer depth index for that level.
        """
        lvl = int(level)
        depth = int(self.depth - (int(self.max_level) - lvl))
        if depth < 0:
            raise ValueError(
                f"Derived negative tree depth for level {lvl}; "
                f"tree.depth={self.depth}, max_level={self.max_level}."
            )
        return depth

    def shape_for_level(self, level: int) -> GridShape:
        """Return grid shape at one refinement level.

        Consumes:
        - `level`: refinement level id.
        Returns:
        - `(nr, ntheta, nphi)` grid shape.
        """
        depth = self.depth_for_level(level)
        return (
            int(self.root_shape[0] * (1 << depth)),
            int(self.root_shape[1] * (1 << depth)),
            int(self.root_shape[2] * (1 << depth)),
        )

    def axis_spacing_for_level(self, level: int) -> tuple[float, float]:
        """Return axis spacing at one refinement level.

        Consumes:
        - `level`: refinement level id.
        Returns:
        - `(d_axis1, d_axis2)` spacings.
        """
        _n0, n1, n2 = self.shape_for_level(level)
        return math.pi / float(n1), (2.0 * math.pi) / float(n2)

    def level_maps(
        self,
        levels: np.ndarray,
    ) -> tuple[LevelToDepthMap, LevelToShapeMap, LevelToSpacingMap, LevelToSpacingMap]:
        """Build per-level depth, shape, and spacing lookup maps.

        Consumes:
        - `levels`: per-cell level array (non-negative entries are used).
        Returns:
        - Tuple of maps: `(level_to_depth, shape_by_level, axis1_by_level, axis2_by_level)`.
        """
        uniq = sorted(set(int(v) for v in np.array(levels, dtype=np.int64).tolist() if int(v) >= 0))
        if not uniq:
            raise ValueError("No valid (>=0) levels available to build level maps.")
        level_to_depth: LevelToDepthMap = {}
        shape_by_level: LevelToShapeMap = {}
        axis1_by_level: LevelToSpacingMap = {}
        axis2_by_level: LevelToSpacingMap = {}
        for level in uniq:
            depth = self.depth_for_level(level)
            shape = self.shape_for_level(level)
            d_axis1, d_axis2 = self.axis_spacing_for_level(level)
            level_to_depth[level] = depth
            shape_by_level[level] = shape
            axis1_by_level[level] = d_axis1
            axis2_by_level[level] = d_axis2
        return level_to_depth, shape_by_level, axis1_by_level, axis2_by_level

    def summary(self) -> str:
        """Return compact summary text for this tree.

        Consumes:
        - Octree summary fields on `self`.
        Returns:
        - Single summary string.
        """
        return format_octree_summary(self)

    def _build_lookup(
        self,
    ) -> "_CellLookup":
        """Construct a bound cell-lookup object for this tree.

        Consumes:
        - Bound tree geometry.
        Returns:
        - Concrete `_CellLookup` implementation.
        """
        raise NotImplementedError("Lookup must be implemented by concrete octree subclasses.")

    def _require_lookup(self) -> "_CellLookup":
        """Return cached lookup state, building it lazily if needed.

        Consumes:
        - Bound dataset/corners state on `self`.
        Returns:
        - Bound `_CellLookup` instance.
        """
        if self._lookup_cache is not None:
            return self._lookup_cache
        if self.ds is None or self.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call Octree.from_dataset(...) or bind(...).")
        self._lookup_cache = self._build_lookup()
        return self._lookup_cache

    @property
    def lookup(self) -> "_CellLookup":
        """Expose the bound lookup object used by octree queries.

        Consumes:
        - Cached/bound lookup state on `self`.
        Returns:
        - `_CellLookup` instance.
        """
        return self._require_lookup()

    def lookup_point(
        self,
        point: np.ndarray,
        *,
        space: CoordSystem,
    ) -> "LookupHit | None":
        """Lookup one query point in the requested coordinate space.

        Consumes:
        - `point`: query coordinate triple.
        - `space`: `"xyz"` or `"rpa"`.
        Returns:
        - `LookupHit` if a cell is resolved, else `None`.
        """
        q = np.array(point, dtype=float).reshape(3)
        resolved_space = str(space)
        if resolved_space not in SUPPORTED_COORD_SYSTEMS:
            raise ValueError(
                f"Unsupported lookup space '{resolved_space}'; expected one of {SUPPORTED_COORD_SYSTEMS}."
            )
        return self.lookup.lookup_point(q, space=resolved_space)

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        space: CoordSystem,
        tol: float = 1e-10,
    ) -> bool:
        """Containment test of one query point against one leaf cell.

        Consumes:
        - `cell_id`: integer leaf cell id.
        - `point`: query coordinate triple.
        - `space`: `"xyz"` or `"rpa"`.
        - Optional containment tolerance `tol`.
        Returns:
        - `True` when the point lies inside/on the cell bounds, else `False`.
        """
        q = np.array(point, dtype=float).reshape(3)
        resolved_space = str(space)
        if resolved_space not in SUPPORTED_COORD_SYSTEMS:
            raise ValueError(
                f"Unsupported lookup space '{resolved_space}'; expected one of {SUPPORTED_COORD_SYSTEMS}."
            )
        return bool(
            self.lookup.contains_cell(
                int(cell_id),
                q,
                space=resolved_space,
                tol=float(tol),
            )
        )

    def hit_from_cell_id(self, cell_id: int) -> "LookupHit":
        """Materialize `LookupHit` metadata from a known cell id.

        Consumes:
        - `cell_id`: integer leaf-cell id.
        Returns:
        - `LookupHit` for that id, or raises `ValueError` when invalid.
        """
        cid = int(cell_id)
        lookup = self.lookup
        n_cells = int(lookup._cell_centers.shape[0])
        if cid < 0 or cid >= n_cells:
            raise ValueError(f"Invalid cell_id {cid}; expected [0, {n_cells - 1}].")
        hit = lookup._hit_from_chosen(cid, allow_invalid_depth=True)
        if hit is None:
            raise ValueError(f"Invalid cell_id {cid}; cannot materialize LookupHit.")
        return hit

    def _lookup_local(self, xyz: np.ndarray, near_cid: int | None = None) -> "LookupHit | None":
        """Lookup in xyz using a nearby-cell hint, then fallback to full lookup.

        Consumes:
        - Cartesian query `xyz` and optional nearby `cell_id` hint.
        Returns:
        - `LookupHit` if resolved, else `None`.
        """
        q = np.array(xyz, dtype=float)
        x = float(q[0])
        y = float(q[1])
        z = float(q[2])
        if near_cid is not None and int(near_cid) >= 0:
            near = int(near_cid)
            if self.contains_cell(near, q, space="xyz"):
                return self.hit_from_cell_id(near)
        return self.lookup.lookup_point(np.array([x, y, z], dtype=float), space="xyz")

    def trace_ray(
        self,
        origin_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        t_start: float,
        t_end: float,
        *,
        max_steps: int = 100000,
        bisect_iters: int = 48,
        boundary_tol: float = 1e-9,
    ) -> list["RaySegment"]:
        """Trace a ray into contiguous per-cell segments.

        This delegates to `octree.ray.OctreeRayTracer` so ray traversal logic
        is isolated from the core octree container class.
        Consumes:
        - Ray origin/direction, `t` bounds, and tracing controls.
        Returns:
        - Ordered list of `RaySegment` intervals.
        """
        from .ray import OctreeRayTracer

        return OctreeRayTracer(self).trace(
            origin_xyz,
            direction_xyz,
            t_start,
            t_end,
            max_steps=max_steps,
            bisect_iters=bisect_iters,
            boundary_tol=boundary_tol,
        )


def _octree_class_for_coord(coord_system: str) -> type[Octree]:
    """Resolve coordinate-system tag to the concrete octree class.

    Consumes:
    - `coord_system` tag (`"rpa"` or `"xyz"`).
    Returns:
    - Matching `Octree` subclass type.
    """
    from .cartesian import CartesianOctree
    from .spherical import SphericalOctree

    if coord_system == "rpa":
        return SphericalOctree
    if coord_system == "xyz":
        return CartesianOctree
    raise ValueError(
        f"Unsupported coord_system '{coord_system}'; expected one of {SUPPORTED_COORD_SYSTEMS}."
    )


@dataclass(frozen=True)
class LookupHit:
    """Resolved lookup metadata for one query point."""

    cell_id: int
    level: int
    i0: int
    i1: int
    i2: int
    path: GridPath
    center_xyz: tuple[float, float, float]

def format_octree_summary(tree: Octree) -> str:
    """Format one-line summary text for an `Octree` object.

    Consumes:
    - `tree`: octree summary object.
    Returns:
    - Single formatted summary string.
    """
    leaf_levels = ", ".join(
        f"L{level}:{count} (fine-equiv {expected})"
        for level, count, expected in tree.level_counts
    )
    shape_kind = "uniform" if tree.is_uniform else "adaptive"
    out = (
        f"Octree ({shape_kind}): "
        f"coord_system={tree.coord_system}, "
        f"finest_leaf_grid={tree.leaf_shape}, root_grid={tree.root_shape}, "
        f"depth={tree.depth}, full={tree.is_full}, "
        f"levels={tree.min_level}..{tree.max_level}; leaf_levels[{leaf_levels}]"
    )
    if tree.block_shape is not None and tree.block_level_counts is not None:
        block_levels = ", ".join(
            f"L{level}:{count}/{expected}" for level, count, expected in tree.block_level_counts
        )
        out += (
            "; block_tree{"
            f"cells_per_block={tree.block_cell_shape}, block_grid={tree.block_shape}, "
            f"root_grid={tree.block_root_shape}, depth={tree.block_depth}, "
            f"levels[{block_levels}]"
            "}"
        )
    return out

@dataclass(frozen=True)
class RaySegment:
    """Ray interval `[t_enter, t_exit]` that remains inside one leaf cell."""

    cell_id: int
    t_enter: float
    t_exit: float


@dataclass(frozen=True)
class RayLinearPiece:
    """Linear function f(t) = slope*t + intercept over [t_start, t_end]."""

    t_start: float
    t_end: float
    cell_id: int
    tet_id: int
    slope: np.ndarray
