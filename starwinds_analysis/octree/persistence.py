#!/usr/bin/env python3
"""Octree persistence dataclasses and `.npz` save/load helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .base import CoordSystem
from .base import DEFAULT_AXIS_RHO_TOL
from .base import DEFAULT_COORD_SYSTEM
from .base import GridShape
from .base import LevelCountTable
from .base import OCTREE_FILE_VERSION
from .base import Octree
from .base import SUPPORTED_COORD_SYSTEMS


@dataclass(frozen=True)
class OctreeArrayState:
    """Typed optional ndarray payload persisted alongside octree metadata.

    This keeps array payload flow typed inside Python code; only file/JSON
    boundaries use dict-based representations.
    """

    cell_levels: np.ndarray | None = None
    axis2_center: np.ndarray | None = None
    axis2_span: np.ndarray | None = None
    expected_axis2_span: np.ndarray | None = None

    @classmethod
    def from_tree(cls, tree: "Octree") -> "OctreeArrayState":
        """Capture optional ndarray payload from one in-memory tree.

        Consumes:
        - `tree`: source octree instance.
        Returns:
        - Typed `OctreeArrayState` with copied/normalized arrays.
        """
        return cls(
            cell_levels=None if tree.cell_levels is None else np.array(tree.cell_levels, dtype=np.int64),
            axis2_center=None if tree.axis2_center is None else np.array(tree.axis2_center, dtype=float),
            axis2_span=None if tree.axis2_span is None else np.array(tree.axis2_span, dtype=float),
            expected_axis2_span=(
                None
                if tree.expected_axis2_span is None
                else np.array(tree.expected_axis2_span, dtype=float)
            ),
        )

    @classmethod
    def from_npz(cls, data: np.lib.npyio.NpzFile) -> "OctreeArrayState":
        """Load optional ndarray payload from one opened `.npz` file handle.

        Consumes:
        - `data`: opened `np.load(..., allow_pickle=False)` handle.
        Returns:
        - Typed `OctreeArrayState` populated from available array keys.
        """
        return cls(
            cell_levels=None if "cell_levels" not in data else np.array(data["cell_levels"], dtype=np.int64),
            axis2_center=(
                np.array(data["axis2_center"], dtype=float)
                if "axis2_center" in data
                else (np.array(data["center_phi"], dtype=float) if "center_phi" in data else None)
            ),
            axis2_span=(
                np.array(data["axis2_span"], dtype=float)
                if "axis2_span" in data
                else (np.array(data["delta_phi"], dtype=float) if "delta_phi" in data else None)
            ),
            expected_axis2_span=(
                np.array(data["expected_axis2_span"], dtype=float)
                if "expected_axis2_span" in data
                else (np.array(data["expected_delta_phi"], dtype=float) if "expected_delta_phi" in data else None)
            ),
        )

@dataclass(frozen=True)
class OctreePersistenceState:
    """Typed, versioned core state used by octree save/load persistence.

    This object stores the main structural fields required to reconstruct an
    `Octree`. Optional axis-2 payloads (`axis2_center`, etc.) are
    persisted separately in the `.npz` file.
    """

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
    coord_system: CoordSystem
    coarse_axis2_span: float | None
    axis_rho_tol: float

    @classmethod
    def from_octree(cls, tree: "Octree") -> "OctreePersistenceState":
        """Capture persistence state from one in-memory tree.

        Consumes:
        - `tree`: source octree instance.
        Returns:
        - `OctreePersistenceState` snapshot with serializable core metadata.
        """
        coord_system = str(tree.coord_system)
        if coord_system not in SUPPORTED_COORD_SYSTEMS:
            raise ValueError(
                f"Unsupported coord_system '{coord_system}'; expected one of {SUPPORTED_COORD_SYSTEMS}."
            )
        return cls(
            leaf_shape=tuple(int(v) for v in tree.leaf_shape),
            root_shape=tuple(int(v) for v in tree.root_shape),
            depth=int(tree.depth),
            is_full=bool(tree.is_full),
            level_counts=tuple(tuple(int(v) for v in row) for row in tree.level_counts),
            min_level=int(tree.min_level),
            max_level=int(tree.max_level),
            coarse_axis1_step=(
                None if tree.coarse_axis1_step is None else float(tree.coarse_axis1_step)
            ),
            coarse_axis2_step=(
                None if tree.coarse_axis2_step is None else float(tree.coarse_axis2_step)
            ),
            block_cell_shape=(
                None if tree.block_cell_shape is None else tuple(int(v) for v in tree.block_cell_shape)
            ),
            block_shape=None if tree.block_shape is None else tuple(int(v) for v in tree.block_shape),
            block_root_shape=(
                None if tree.block_root_shape is None else tuple(int(v) for v in tree.block_root_shape)
            ),
            block_depth=None if tree.block_depth is None else int(tree.block_depth),
            block_level_counts=(
                None
                if tree.block_level_counts is None
                else tuple(tuple(int(v) for v in row) for row in tree.block_level_counts)
            ),
            coord_system=coord_system,
            coarse_axis2_span=None if tree.coarse_axis2_span is None else float(tree.coarse_axis2_span),
            axis_rho_tol=float(tree.axis_rho_tol),
        )

    def to_core_metadata(self) -> dict[str, object]:
        """Convert core state to JSON-compatible metadata.

        Consumes:
        - Persistence-state fields on `self`.
        Returns:
        - Dictionary of core state fields in plain JSON-compatible types.
        """
        return {
            "leaf_shape": list(self.leaf_shape),
            "root_shape": list(self.root_shape),
            "depth": int(self.depth),
            "is_full": bool(self.is_full),
            "level_counts": [list(row) for row in self.level_counts],
            "min_level": int(self.min_level),
            "max_level": int(self.max_level),
            "coarse_axis1_step": (
                None if self.coarse_axis1_step is None else float(self.coarse_axis1_step)
            ),
            "coarse_axis2_step": (
                None if self.coarse_axis2_step is None else float(self.coarse_axis2_step)
            ),
            "block_cell_shape": None if self.block_cell_shape is None else list(self.block_cell_shape),
            "block_shape": None if self.block_shape is None else list(self.block_shape),
            "block_root_shape": None if self.block_root_shape is None else list(self.block_root_shape),
            "block_depth": None if self.block_depth is None else int(self.block_depth),
            "block_level_counts": (
                None if self.block_level_counts is None else [list(row) for row in self.block_level_counts]
            ),
            "coord_system": str(self.coord_system),
            "coarse_axis2_span": None if self.coarse_axis2_span is None else float(self.coarse_axis2_span),
            "axis_rho_tol": float(self.axis_rho_tol),
        }

    def to_file_metadata(self) -> dict[str, object]:
        """Build top-level file metadata payload.

        Consumes:
        - Core metadata from `self`.
        Returns:
        - Dictionary with file format version and nested state metadata.
        """
        return {
            "version": int(OCTREE_FILE_VERSION),
            "state": self.to_core_metadata(),
        }

    def save_npz(
        self,
        path: Path,
        *,
        arrays: OctreeArrayState,
        corners: np.ndarray | None,
        include_corners: bool,
    ) -> None:
        """Persist state and typed array payload to one compressed `.npz` file.

        Consumes:
        - Output `path`, typed `arrays`, optional `corners`, include flag.
        Returns:
        - `None`; writes one `.npz` file with versioned metadata.
        """
        has_cell_levels = arrays.cell_levels is not None
        has_axis2_center = arrays.axis2_center is not None
        has_axis2_span = arrays.axis2_span is not None
        has_expected_axis2_span = arrays.expected_axis2_span is not None
        has_corners = bool(include_corners and corners is not None)

        cell_levels = (
            np.array(arrays.cell_levels, dtype=np.int64)
            if has_cell_levels
            else np.empty((0,), dtype=np.int64)
        )
        axis2_center = (
            np.array(arrays.axis2_center, dtype=float)
            if has_axis2_center
            else np.empty((0,), dtype=float)
        )
        axis2_span = (
            np.array(arrays.axis2_span, dtype=float)
            if has_axis2_span
            else np.empty((0,), dtype=float)
        )
        expected_axis2_span = (
            np.array(arrays.expected_axis2_span, dtype=float)
            if has_expected_axis2_span
            else np.empty((0,), dtype=float)
        )
        corners_arr = (
            np.array(corners, dtype=np.int64)
            if has_corners
            else np.empty((0, 0), dtype=np.int64)
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            meta_json=np.array(json.dumps(self.to_file_metadata())),
            has_cell_levels=np.array(int(has_cell_levels), dtype=np.int8),
            has_axis2_center=np.array(int(has_axis2_center), dtype=np.int8),
            has_axis2_span=np.array(int(has_axis2_span), dtype=np.int8),
            has_expected_axis2_span=np.array(int(has_expected_axis2_span), dtype=np.int8),
            has_corners=np.array(int(has_corners), dtype=np.int8),
            cell_levels=cell_levels,
            axis2_center=axis2_center,
            axis2_span=axis2_span,
            expected_axis2_span=expected_axis2_span,
            corners=corners_arr,
        )

    @classmethod
    def load_npz(
        cls,
        path: Path,
    ) -> tuple["OctreePersistenceState", OctreeArrayState, np.ndarray | None]:
        """Load persisted octree state, array payload, and optional corners.

        Consumes:
        - Input `.npz` file `path`.
        Returns:
        - `(state, arrays, corners)` typed tuple.
        """
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            version = int(meta.get("version", 0))
            if version != OCTREE_FILE_VERSION:
                raise ValueError(
                    f"Unsupported octree file version {version}; expected {OCTREE_FILE_VERSION}."
                )

            state = cls.from_file_metadata(meta)

            has_cell_levels = bool(int(data["has_cell_levels"])) if "has_cell_levels" in data else ("cell_levels" in data)
            has_axis2_center = (
                bool(int(data["has_axis2_center"]))
                if "has_axis2_center" in data
                else ("axis2_center" in data or "center_phi" in data)
            )
            has_axis2_span = (
                bool(int(data["has_axis2_span"]))
                if "has_axis2_span" in data
                else ("axis2_span" in data or "delta_phi" in data)
            )
            has_expected_axis2_span = (
                bool(int(data["has_expected_axis2_span"]))
                if "has_expected_axis2_span" in data
                else ("expected_axis2_span" in data or "expected_delta_phi" in data)
            )

            arrays = OctreeArrayState(
                cell_levels=(np.array(data["cell_levels"], dtype=np.int64) if has_cell_levels else None),
                axis2_center=(
                    np.array(data["axis2_center"], dtype=float)
                    if ("axis2_center" in data and has_axis2_center)
                    else (np.array(data["center_phi"], dtype=float) if has_axis2_center and "center_phi" in data else None)
                ),
                axis2_span=(
                    np.array(data["axis2_span"], dtype=float)
                    if ("axis2_span" in data and has_axis2_span)
                    else (np.array(data["delta_phi"], dtype=float) if has_axis2_span and "delta_phi" in data else None)
                ),
                expected_axis2_span=(
                    np.array(data["expected_axis2_span"], dtype=float)
                    if ("expected_axis2_span" in data and has_expected_axis2_span)
                    else (
                        np.array(data["expected_delta_phi"], dtype=float)
                        if has_expected_axis2_span and "expected_delta_phi" in data
                        else None
                    )
                ),
            )

            has_corners = bool(int(data["has_corners"])) if "has_corners" in data else ("corners" in data)
            corners = np.array(data["corners"], dtype=np.int64) if has_corners else None

        return state, arrays, corners

    @classmethod
    def from_file_metadata(cls, meta: dict[str, object]) -> "OctreePersistenceState":
        """Parse file metadata into a typed core state.

        Supports both schemas:
        - current nested schema: `{\"version\": ..., \"state\": {...}}`
        - legacy flat schema where core fields live at top level.
        Consumes:
        - `meta`: decoded metadata dictionary from saved octree file.
        Returns:
        - Parsed `OctreePersistenceState`.
        """
        state_raw = meta.get("state")
        state = state_raw if isinstance(state_raw, dict) else meta

        def read(key: str, default: object | None = None) -> object | None:
            """Read one metadata value with legacy top-level override precedence."""
            if key in meta and key != "state":
                return meta[key]
            if isinstance(state, dict) and key in state:
                return state[key]
            return default

        coord_system = str(read("coord_system", DEFAULT_COORD_SYSTEM))
        if coord_system not in SUPPORTED_COORD_SYSTEMS:
            raise ValueError(
                f"Unsupported coord_system '{coord_system}' in octree file; "
                f"expected one of {SUPPORTED_COORD_SYSTEMS}."
            )

        block_level_counts_raw = read("block_level_counts")
        return cls(
            leaf_shape=tuple(int(v) for v in read("leaf_shape")),
            root_shape=tuple(int(v) for v in read("root_shape")),
            depth=int(read("depth")),
            is_full=bool(read("is_full")),
            level_counts=tuple(tuple(int(v) for v in row) for row in read("level_counts")),
            min_level=int(read("min_level")),
            max_level=int(read("max_level")),
            coarse_axis1_step=(
                float(read("coarse_axis1_step"))
                if read("coarse_axis1_step") is not None
                else (
                    float(read("coarse_dtheta"))
                    if read("coarse_dtheta") is not None
                    else None
                )
            ),
            coarse_axis2_step=(
                float(read("coarse_axis2_step"))
                if read("coarse_axis2_step") is not None
                else (
                    float(read("coarse_dphi"))
                    if read("coarse_dphi") is not None
                    else None
                )
            ),
            block_cell_shape=(
                None if read("block_cell_shape") is None else tuple(int(v) for v in read("block_cell_shape"))
            ),
            block_shape=None if read("block_shape") is None else tuple(int(v) for v in read("block_shape")),
            block_root_shape=(
                None
                if read("block_root_shape") is None
                else tuple(int(v) for v in read("block_root_shape"))
            ),
            block_depth=None if read("block_depth") is None else int(read("block_depth")),
            block_level_counts=(
                None if block_level_counts_raw is None else tuple(tuple(int(v) for v in row) for row in block_level_counts_raw)
            ),
            coord_system=coord_system,
            coarse_axis2_span=(
                None
                if read("coarse_axis2_span", read("coarse_delta_phi")) is None
                else float(read("coarse_axis2_span", read("coarse_delta_phi")))
            ),
            axis_rho_tol=float(read("axis_rho_tol", DEFAULT_AXIS_RHO_TOL)),
        )

    def instantiate_tree(
        self,
        tree_cls: type["Octree"],
        *,
        arrays: OctreeArrayState,
    ) -> "Octree":
        """Instantiate one octree object from persisted state + array payloads.

        Consumes:
        - `tree_cls`: concrete octree class to instantiate.
        - `arrays`: typed optional ndarray payloads.
        Returns:
        - Instantiated `Octree` (or subclass) object.
        """
        return tree_cls(
            leaf_shape=self.leaf_shape,
            root_shape=self.root_shape,
            depth=self.depth,
            is_full=self.is_full,
            level_counts=self.level_counts,
            min_level=self.min_level,
            max_level=self.max_level,
            coarse_axis1_step=self.coarse_axis1_step,
            coarse_axis2_step=self.coarse_axis2_step,
            block_cell_shape=self.block_cell_shape,
            block_shape=self.block_shape,
            block_root_shape=self.block_root_shape,
            block_depth=self.block_depth,
            block_level_counts=self.block_level_counts,
            coord_system=self.coord_system,
            cell_levels=arrays.cell_levels,
            axis2_center=arrays.axis2_center,
            axis2_span=arrays.axis2_span,
            expected_axis2_span=arrays.expected_axis2_span,
            coarse_axis2_span=self.coarse_axis2_span,
            axis_rho_tol=self.axis_rho_tol,
        )
