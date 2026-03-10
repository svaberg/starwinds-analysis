#!/usr/bin/env python3
"""Cartesian (`xyz`) octree level and shape inference utilities."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from starwinds_readplt.dataset import Dataset

LevelShapeStatsRow: TypeAlias = tuple[int, int, float, float, int]
"""Tuple `(n_axis1, n_axis2, d_axis1, d_axis2, n_cells_at_level)`."""

LevelShapeStatsMap: TypeAlias = dict[int, LevelShapeStatsRow]
"""Map `level -> LevelShapeStatsRow`."""


class CartesianOctreeBuilder:
    """Coordinate-specific Cartesian inference strategy used by `OctreeBuilder`."""

    @staticmethod
    def _median_positive(values: np.ndarray, *, tiny: float = 1e-12) -> float:
        """Compute the median of positive values above `tiny`."""
        pos = np.asarray(values, dtype=float)
        pos = pos[pos > float(tiny)]
        if pos.size == 0:
            raise ValueError("No positive values available to infer spacing.")
        return float(np.median(pos))

    @staticmethod
    def infer_xyz_levels_from_cell_spans(
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        *,
        rtol: float = 2e-2,
        atol: float = 1e-10,
    ) -> np.ndarray:
        """Infer dyadic xyz refinement levels from per-cell axis-aligned spans."""
        levels = np.full(dx.shape, -1, dtype=np.int64)
        tiny = max(float(atol), 1e-12)
        valid = (dx > tiny) & (dy > tiny) & (dz > tiny) & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz)
        if not np.any(valid):
            return levels

        coarse_dx = float(np.max(dx[valid]))
        coarse_dy = float(np.max(dy[valid]))
        coarse_dz = float(np.max(dz[valid]))

        raw_x = np.log2(coarse_dx / dx[valid])
        raw_y = np.log2(coarse_dy / dy[valid])
        raw_z = np.log2(coarse_dz / dz[valid])
        guess = np.maximum(np.rint((raw_x + raw_y + raw_z) / 3.0).astype(np.int64), 0)

        exp_x = coarse_dx / np.exp2(guess)
        exp_y = coarse_dy / np.exp2(guess)
        exp_z = coarse_dz / np.exp2(guess)
        ok = (
            np.isclose(dx[valid], exp_x, rtol=rtol, atol=atol)
            & np.isclose(dy[valid], exp_y, rtol=rtol, atol=atol)
            & np.isclose(dz[valid], exp_z, rtol=rtol, atol=atol)
        )
        levels[valid] = np.where(ok, guess, -1)
        return levels

    @staticmethod
    def infer_xyz_level_shapes(
        ds: Dataset,
        corners: np.ndarray,
        cell_levels: np.ndarray,
    ) -> LevelShapeStatsMap:
        """Infer per-level axis counts/spacings for Cartesian octrees."""
        x = np.asarray(ds.variable("X [R]"), dtype=float)
        y = np.asarray(ds.variable("Y [R]"), dtype=float)
        z = np.asarray(ds.variable("Z [R]"), dtype=float)
        cell_x = x[corners]
        cell_y = y[corners]
        cell_z = z[corners]
        dx = np.ptp(cell_x, axis=1)
        dy = np.ptp(cell_y, axis=1)
        dz = np.ptp(cell_z, axis=1)

        x_min = float(np.min(np.min(cell_x, axis=1)))
        x_max = float(np.max(np.max(cell_x, axis=1)))
        y_min = float(np.min(np.min(cell_y, axis=1)))
        y_max = float(np.max(np.max(cell_y, axis=1)))
        z_min = float(np.min(np.min(cell_z, axis=1)))
        z_max = float(np.max(np.max(cell_z, axis=1)))

        span_x = max(x_max - x_min, np.finfo(float).tiny)
        span_y = max(y_max - y_min, np.finfo(float).tiny)
        span_z = max(z_max - z_min, np.finfo(float).tiny)

        out: LevelShapeStatsMap = {}
        unique_levels = sorted(set(int(v) for v in cell_levels.tolist() if int(v) >= 0))
        if not unique_levels:
            raise ValueError("No valid (>=0) cell levels available for tree inference.")

        for level in unique_levels:
            mask = cell_levels == level
            med_dx = CartesianOctreeBuilder._median_positive(dx[mask])
            med_dy = CartesianOctreeBuilder._median_positive(dy[mask])
            med_dz = CartesianOctreeBuilder._median_positive(dz[mask])
            n_x = int(round(span_x / med_dx))
            n_y = int(round(span_y / med_dy))
            n_z = int(round(span_z / med_dz))
            if n_x <= 0 or n_y <= 0 or n_z <= 0:
                raise ValueError(
                    f"Invalid xyz counts inferred at level {level}: n_x={n_x}, n_y={n_y}, n_z={n_z}."
                )
            ref_dx = span_x / n_x
            ref_dy = span_y / n_y
            ref_dz = span_z / n_z
            if not np.isclose(med_dx, ref_dx, rtol=2e-2, atol=1e-9):
                raise ValueError(f"Level {level} has inconsistent dx={med_dx:.6e} vs inferred {ref_dx:.6e}.")
            if not np.isclose(med_dy, ref_dy, rtol=2e-2, atol=1e-9):
                raise ValueError(f"Level {level} has inconsistent dy={med_dy:.6e} vs inferred {ref_dy:.6e}.")
            if not np.isclose(med_dz, ref_dz, rtol=2e-2, atol=1e-9):
                raise ValueError(f"Level {level} has inconsistent dz={med_dz:.6e} vs inferred {ref_dz:.6e}.")
            out[level] = (n_y, n_z, med_dy, med_dz, int(np.count_nonzero(mask)))
        return out

    def __init__(self, *, level_rtol: float = 1e-4, level_atol: float = 1e-9) -> None:
        """Store level-inference tolerances."""
        self.level_rtol = float(level_rtol)
        self.level_atol = float(level_atol)

    def infer_level_shapes(
        self,
        ds: Dataset,
        corners: np.ndarray,
        *,
        cell_levels: np.ndarray | None = None,
    ) -> tuple[LevelShapeStatsMap, np.ndarray, int, int]:
        """Infer Cartesian level-shape map and validated levels."""
        x = np.asarray(ds.variable("X [R]"), dtype=float)
        y = np.asarray(ds.variable("Y [R]"), dtype=float)
        z = np.asarray(ds.variable("Z [R]"), dtype=float)
        cell_x = x[corners]
        cell_y = y[corners]
        cell_z = z[corners]
        dx = np.ptp(cell_x, axis=1)
        dy = np.ptp(cell_y, axis=1)
        dz = np.ptp(cell_z, axis=1)

        levels = cell_levels
        if levels is None:
            levels = self.infer_xyz_levels_from_cell_spans(
                dx,
                dy,
                dz,
                rtol=max(2e-2, float(self.level_rtol)),
                atol=max(1e-10, float(self.level_atol)),
            )
        levels = np.asarray(levels, dtype=np.int64)
        valid_levels = levels[levels >= 0]
        if valid_levels.size == 0:
            raise ValueError("No valid (>=0) levels available to infer octree.")
        level_shapes = self.infer_xyz_level_shapes(ds, corners, levels)
        min_level = int(np.min(valid_levels))
        max_level = int(np.max(valid_levels))
        return level_shapes, levels, min_level, max_level
