#!/usr/bin/env python3
"""Cartesian octree and Cartesian lookup implementation."""

from __future__ import annotations

from typing import ClassVar

import math
import numpy as np

from .base import GridIndex
from .base import GridPath
from .base import LookupHit
from .base import Octree
from .base import _CellLookup


class _CartesianCellLookup(_CellLookup):
    """Leaf-cell lookup accelerator for Cartesian `(x, y, z)` octrees."""

    def __init__(self, tree: Octree) -> None:
        """Build Cartesian lookup geometry and sparse 3D bin index from one bound tree."""
        if tree.ds is None or tree.corners is None:
            raise ValueError("Lookup requires a bound octree with dataset and corners.")
        ds = tree.ds
        corners = tree.corners
        if not {"X [R]", "Y [R]", "Z [R]"}.issubset(set(ds.variables)):
            raise ValueError("Lookup requires X/Y/Z variables.")

        self.tree = tree
        self._corners = np.array(corners, dtype=np.int64)
        self._points = np.column_stack(
            (
                np.array(ds.variable("X [R]"), dtype=float),
                np.array(ds.variable("Y [R]"), dtype=float),
                np.array(ds.variable("Z [R]"), dtype=float),
            )
        )
        cell_xyz = self._points[self._corners]
        self._cell_centers = np.mean(cell_xyz, axis=1)
        self._cell_x_min = np.min(cell_xyz[:, :, 0], axis=1)
        self._cell_x_max = np.max(cell_xyz[:, :, 0], axis=1)
        self._cell_y_min = np.min(cell_xyz[:, :, 1], axis=1)
        self._cell_y_max = np.max(cell_xyz[:, :, 1], axis=1)
        self._cell_z_min = np.min(cell_xyz[:, :, 2], axis=1)
        self._cell_z_max = np.max(cell_xyz[:, :, 2], axis=1)
        tiny = np.finfo(float).tiny
        self._cell_dx = np.maximum(self._cell_x_max - self._cell_x_min, tiny)
        self._cell_dy = np.maximum(self._cell_y_max - self._cell_y_min, tiny)
        self._cell_dz = np.maximum(self._cell_z_max - self._cell_z_min, tiny)
        self._xyz_min = np.array(
            [
                float(np.min(self._cell_x_min)),
                float(np.min(self._cell_y_min)),
                float(np.min(self._cell_z_min)),
            ],
            dtype=float,
        )
        self._xyz_max = np.array(
            [
                float(np.max(self._cell_x_max)),
                float(np.max(self._cell_y_max)),
                float(np.max(self._cell_z_max)),
            ],
            dtype=float,
        )
        self._xyz_span = np.maximum(self._xyz_max - self._xyz_min, tiny)

        n_cells = int(self._corners.shape[0])
        if tree.cell_levels is not None and tree.cell_levels.shape[0] == n_cells:
            self._cell_level_rel = np.array(tree.cell_levels, dtype=np.int64)
        else:
            self._cell_level_rel = np.full(n_cells, int(tree.max_level), dtype=np.int64)
        self._cell_depth = np.full(n_cells, int(tree.depth), dtype=np.int64)
        valid = self._cell_level_rel >= 0
        for level in sorted(set(int(v) for v in self._cell_level_rel[valid].tolist())):
            self._cell_depth[self._cell_level_rel == level] = int(tree.depth_for_level(level))

        # Approximate leaf-grid indices from normalized center coordinates.
        lx, ly, lz = int(tree.leaf_shape[0]), int(tree.leaf_shape[1]), int(tree.leaf_shape[2])
        fx = (self._cell_centers[:, 0] - self._xyz_min[0]) / self._xyz_span[0]
        fy = (self._cell_centers[:, 1] - self._xyz_min[1]) / self._xyz_span[1]
        fz = (self._cell_centers[:, 2] - self._xyz_min[2]) / self._xyz_span[2]
        self._i0 = np.clip(np.floor(fx * lx), 0, max(lx - 1, 0)).astype(np.int64)
        self._i1 = np.clip(np.floor(fy * ly), 0, max(ly - 1, 0)).astype(np.int64)
        self._i2 = np.clip(np.floor(fz * lz), 0, max(lz - 1, 0)).astype(np.int64)

        nbin = max(4, int(round(n_cells ** (1.0 / 3.0))))
        self._bin_shape = np.array([nbin, nbin, nbin], dtype=np.int64)
        bxyz = np.floor(((self._cell_centers - self._xyz_min[None, :]) / self._xyz_span[None, :]) * nbin).astype(
            np.int64
        )
        bxyz = np.clip(bxyz, 0, nbin - 1)
        keys = (bxyz[:, 0] * nbin + bxyz[:, 1]) * nbin + bxyz[:, 2]
        n_bins = int(nbin * nbin * nbin)
        bin_lists: list[list[int]] = [[] for _ in range(n_bins)]
        for cid in range(n_cells):
            bin_lists[int(keys[cid])].append(cid)

        bin_counts = np.zeros(n_bins, dtype=np.int64)
        for key, ids in enumerate(bin_lists):
            if not ids:
                continue
            arr = np.array(ids, dtype=np.int64)
            order = np.argsort(np.linalg.norm(self._cell_centers[arr], axis=1))
            sorted_ids = arr[order]
            bin_lists[key] = sorted_ids.tolist()
            bin_counts[key] = int(sorted_ids.size)

        bin_offsets = np.zeros(n_bins + 1, dtype=np.int64)
        if n_bins > 0:
            np.cumsum(bin_counts, out=bin_offsets[1:])
        self._bin_counts = bin_counts
        self._bin_offsets = bin_offsets
        total_refs = int(bin_offsets[-1])
        self._bin_cell_ids = np.empty(total_refs, dtype=np.int64)
        for key in range(n_bins):
            start = int(bin_offsets[key])
            end = int(bin_offsets[key + 1])
            if end > start:
                self._bin_cell_ids[start:end] = np.array(bin_lists[key], dtype=np.int64)

        self._max_radius = 2

    @staticmethod
    def _path(i0: int, i1: int, i2: int, depth: int) -> GridPath:
        """Construct root-to-leaf index path for one leaf coordinate triplet."""
        out: list[GridIndex] = []
        for level in range(depth + 1):
            shift = depth - level
            out.append((i0 >> shift, i1 >> shift, i2 >> shift))
        return tuple(out)

    def _bin_index(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Map one point to one Cartesian center-bin index triplet."""
        nbin = int(self._bin_shape[0])
        fx = (float(x) - self._xyz_min[0]) / self._xyz_span[0]
        fy = (float(y) - self._xyz_min[1]) / self._xyz_span[1]
        fz = (float(z) - self._xyz_min[2]) / self._xyz_span[2]
        bx = int(np.clip(np.floor(fx * nbin), 0, nbin - 1))
        by = int(np.clip(np.floor(fy * nbin), 0, nbin - 1))
        bz = int(np.clip(np.floor(fz * nbin), 0, nbin - 1))
        return bx, by, bz

    def _candidate_ids(self, bx: int, by: int, bz: int, radius: int) -> np.ndarray:
        """Gather candidate cell ids from a cube neighborhood around one center bin."""
        nbin = int(self._bin_shape[0])
        total = 0
        for dx in range(-radius, radius + 1):
            ix = bx + dx
            if ix < 0 or ix >= nbin:
                continue
            for dy in range(-radius, radius + 1):
                iy = by + dy
                if iy < 0 or iy >= nbin:
                    continue
                for dz in range(-radius, radius + 1):
                    iz = bz + dz
                    if iz < 0 or iz >= nbin:
                        continue
                    key = (ix * nbin + iy) * nbin + iz
                    total += int(self._bin_counts[key])
        if total <= 0:
            return np.array([], dtype=np.int64)
        out = np.empty(total, dtype=np.int64)
        cursor = 0
        for dx in range(-radius, radius + 1):
            ix = bx + dx
            if ix < 0 or ix >= nbin:
                continue
            for dy in range(-radius, radius + 1):
                iy = by + dy
                if iy < 0 or iy >= nbin:
                    continue
                for dz in range(-radius, radius + 1):
                    iz = bz + dz
                    if iz < 0 or iz >= nbin:
                        continue
                    key = (ix * nbin + iy) * nbin + iz
                    count = int(self._bin_counts[key])
                    if count <= 0:
                        continue
                    start = int(self._bin_offsets[key])
                    end = int(self._bin_offsets[key + 1])
                    out[cursor : cursor + count] = self._bin_cell_ids[start:end]
                    cursor += count
        return out

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        space: str,
        tol: float = 1e-10,
    ) -> bool:
        """Containment test of one query point in `space` against one leaf cell."""
        resolved = str(space)
        if resolved != "xyz":
            raise ValueError("Cartesian lookup supports only space='xyz'.")
        q = np.array(point, dtype=float).reshape(3)
        return self.contains_xyz_cell(
            int(cell_id),
            float(q[0]),
            float(q[1]),
            float(q[2]),
            tol=float(tol),
        )

    def lookup_cell_id(
        self,
        point: np.ndarray,
        *,
        space: str,
    ) -> int:
        """Resolve one query point to a leaf `cell_id` (or `-1`)."""
        resolved = str(space)
        if resolved != "xyz":
            raise ValueError("Cartesian lookup supports only space='xyz'.")
        q = np.array(point, dtype=float).reshape(3)
        return self.lookup_xyz_cell_id(float(q[0]), float(q[1]), float(q[2]))

    def contains_xyz_cell(self, cell_id: int, x: float, y: float, z: float, *, tol: float = 1e-10) -> bool:
        """Containment test of one Cartesian query point against one leaf cell."""
        cid = int(cell_id)
        xx = float(x)
        yy = float(y)
        zz = float(z)
        t = float(tol)
        if xx < (float(self._cell_x_min[cid]) - t) or xx > (float(self._cell_x_max[cid]) + t):
            return False
        if yy < (float(self._cell_y_min[cid]) - t) or yy > (float(self._cell_y_max[cid]) + t):
            return False
        if zz < (float(self._cell_z_min[cid]) - t) or zz > (float(self._cell_z_max[cid]) + t):
            return False
        return True

    def cell_step_hint(self, cell_id: int) -> float:
        """Return one characteristic Cartesian step size used for ray marching."""
        cid = int(cell_id)
        return float(max(float(self._cell_dx[cid]), float(self._cell_dy[cid]), float(self._cell_dz[cid]), 1e-6))

    def lookup_xyz_cell_id(self, x: float, y: float, z: float) -> int:
        """Resolve one Cartesian query to a leaf `cell_id` (or `-1`)."""
        xx = float(x)
        yy = float(y)
        zz = float(z)
        if not (math.isfinite(xx) and math.isfinite(yy) and math.isfinite(zz)):
            return -1
        inside_bbox = bool(
            (xx >= self._xyz_min[0])
            and (xx <= self._xyz_max[0])
            and (yy >= self._xyz_min[1])
            and (yy <= self._xyz_max[1])
            and (zz >= self._xyz_min[2])
            and (zz <= self._xyz_max[2])
        )
        if not inside_bbox:
            return -1

        bx, by, bz = self._bin_index(xx, yy, zz)
        fallback_best = -1
        fallback_best_d2 = np.inf
        for radius in range(self._max_radius + 1):
            cands = self._candidate_ids(bx, by, bz, radius)
            if cands.size == 0:
                continue
            inside_best = -1
            inside_best_d2 = np.inf
            for cid in cands.tolist():
                dx = float(self._cell_centers[cid, 0] - xx)
                dy = float(self._cell_centers[cid, 1] - yy)
                dz = float(self._cell_centers[cid, 2] - zz)
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < fallback_best_d2:
                    fallback_best_d2 = d2
                    fallback_best = int(cid)
                if self.contains_xyz_cell(int(cid), xx, yy, zz) and d2 < inside_best_d2:
                    inside_best_d2 = d2
                    inside_best = int(cid)
            if inside_best >= 0:
                return inside_best

        if fallback_best >= 0:
            return fallback_best

        n_cells = int(self._cell_centers.shape[0])
        all_best = -1
        all_best_d2 = np.inf
        all_inside_best = -1
        all_inside_best_d2 = np.inf
        for cid in range(n_cells):
            dx = float(self._cell_centers[cid, 0] - xx)
            dy = float(self._cell_centers[cid, 1] - yy)
            dz = float(self._cell_centers[cid, 2] - zz)
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < all_best_d2:
                all_best_d2 = d2
                all_best = cid
            if self.contains_xyz_cell(cid, xx, yy, zz) and d2 < all_inside_best_d2:
                all_inside_best_d2 = d2
                all_inside_best = cid
        if all_inside_best >= 0:
            return int(all_inside_best)
        return int(all_best)

    def _hit_from_chosen(self, chosen: int, *, allow_invalid_depth: bool = False) -> LookupHit | None:
        """Materialize lookup metadata from one chosen cell id."""
        if chosen < 0:
            return None
        center = self._cell_centers[chosen]
        depth = int(self._cell_depth[chosen])
        if depth < 0 and not allow_invalid_depth:
            return None
        cell_i0 = int(self._i0[chosen])
        cell_i1 = int(self._i1[chosen])
        cell_i2 = int(self._i2[chosen])
        return LookupHit(
            cell_id=int(chosen),
            level=depth,
            i0=cell_i0,
            i1=cell_i1,
            i2=cell_i2,
            path=self._path(cell_i0, cell_i1, cell_i2, depth),
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
        )

class CartesianOctree(Octree):
    """Octree specialization placeholder for Cartesian `(x, y, z)` datasets."""

    COORD_SYSTEM: ClassVar[str | None] = "xyz"

    def _build_lookup(
        self,
    ) -> "_CellLookup":
        """Construct Cartesian lookup state backed by xyz bins.

        Consumes:
        - Bound Cartesian tree geometry.
        Returns:
        - Cartesian lookup instance.
        """
        if self.ds is None or self.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call bind(...) before lookup.")
        return _CartesianCellLookup(
            self,
        )

