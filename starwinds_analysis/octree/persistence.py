#!/usr/bin/env python3
"""Octree persistence dataclasses and `.npz` save/load helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import CoordSystem
from .base import GridShape
from .base import LevelCountTable
from .base import OCTREE_FILE_VERSION
from .base import Octree
from .base import SUPPORTED_COORD_SYSTEMS


@dataclass(frozen=True)
class OctreeArrayState:
    """Array payload persisted alongside octree core metadata."""

    cell_levels: np.ndarray

    @classmethod
    def from_tree(cls, tree: Octree) -> "OctreeArrayState":
        """Capture array payload from one in-memory tree."""
        if tree.cell_levels is None:
            raise ValueError("Cannot persist octree without cell_levels.")
        return cls(
            cell_levels=np.asarray(tree.cell_levels, dtype=np.int64),
        )


@dataclass(frozen=True)
class OctreePersistenceState:
    """Versioned octree core metadata used by save/load operations."""

    leaf_shape: GridShape
    root_shape: GridShape
    is_full: bool
    level_counts: LevelCountTable
    min_level: int
    max_level: int
    coord_system: CoordSystem
    axis_rho_tol: float

    @classmethod
    def from_octree(cls, tree: Octree) -> "OctreePersistenceState":
        """Capture persistence-safe core metadata from one octree object."""
        coord_system = str(tree.coord_system)
        if coord_system not in SUPPORTED_COORD_SYSTEMS:
            raise ValueError(
                f"Unsupported coord_system '{coord_system}'; expected one of {SUPPORTED_COORD_SYSTEMS}."
            )
        return cls(
            leaf_shape=tuple(int(v) for v in tree.leaf_shape),
            root_shape=tuple(int(v) for v in tree.root_shape),
            is_full=bool(tree.is_full),
            level_counts=tuple(tuple(int(v) for v in row) for row in tree.level_counts),
            min_level=int(tree.min_level),
            max_level=int(tree.max_level),
            coord_system=coord_system,
            axis_rho_tol=float(tree.axis_rho_tol),
        )

    def save_npz(self, path: Path, *, arrays: OctreeArrayState) -> None:
        """Persist one octree snapshot to a compressed `.npz` file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            version=int(OCTREE_FILE_VERSION),
            coord_system=str(self.coord_system),
            leaf_shape=np.asarray(self.leaf_shape, dtype=np.int64),
            root_shape=np.asarray(self.root_shape, dtype=np.int64),
            is_full=bool(self.is_full),
            level_counts=np.asarray(self.level_counts, dtype=np.int64),
            min_level=int(self.min_level),
            max_level=int(self.max_level),
            axis_rho_tol=float(self.axis_rho_tol),
            cell_levels=np.asarray(arrays.cell_levels, dtype=np.int64),
        )

    @classmethod
    def load_npz(cls, path: Path) -> tuple["OctreePersistenceState", OctreeArrayState]:
        """Load one persisted octree snapshot and array payload."""
        required = (
            "version",
            "coord_system",
            "leaf_shape",
            "root_shape",
            "is_full",
            "level_counts",
            "min_level",
            "max_level",
            "axis_rho_tol",
            "cell_levels",
        )
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"Missing required octree fields: {missing}.")

            version = int(data["version"])
            if version != OCTREE_FILE_VERSION:
                raise ValueError(
                    f"Unsupported octree file version {version}; expected {OCTREE_FILE_VERSION}."
                )

            coord_system = str(data["coord_system"])
            if coord_system not in SUPPORTED_COORD_SYSTEMS:
                raise ValueError(
                    f"Unsupported coord_system '{coord_system}' in octree file; "
                    f"expected one of {SUPPORTED_COORD_SYSTEMS}."
                )

            state = cls(
                leaf_shape=tuple(int(v) for v in np.asarray(data["leaf_shape"], dtype=np.int64).tolist()),
                root_shape=tuple(int(v) for v in np.asarray(data["root_shape"], dtype=np.int64).tolist()),
                is_full=bool(data["is_full"]),
                level_counts=tuple(
                    tuple(int(v) for v in row)
                    for row in np.asarray(data["level_counts"], dtype=np.int64).tolist()
                ),
                min_level=int(data["min_level"]),
                max_level=int(data["max_level"]),
                coord_system=coord_system,
                axis_rho_tol=float(data["axis_rho_tol"]),
            )
            arrays = OctreeArrayState(cell_levels=np.asarray(data["cell_levels"], dtype=np.int64))
            return state, arrays

    def instantiate_tree(
        self,
        tree_cls: type[Octree],
        *,
        arrays: OctreeArrayState,
    ) -> Octree:
        """Instantiate one octree object from loaded metadata."""
        return tree_cls(
            leaf_shape=self.leaf_shape,
            root_shape=self.root_shape,
            is_full=self.is_full,
            level_counts=self.level_counts,
            min_level=self.min_level,
            max_level=self.max_level,
            coord_system=self.coord_system,
            cell_levels=arrays.cell_levels,
            axis_rho_tol=self.axis_rho_tol,
        )
