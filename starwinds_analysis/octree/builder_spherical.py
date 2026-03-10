#!/usr/bin/env python3
"""Spherical (`rpa`) octree level and shape inference utilities."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from starwinds_readplt.dataset import Dataset

from .base import DEFAULT_AXIS_RHO_TOL

LevelShapeStatsRow: TypeAlias = tuple[int, int, float, float, int]
"""Tuple `(n_axis1, n_axis2, d_axis1, d_axis2, n_cells_at_level)`."""

LevelShapeStatsMap: TypeAlias = dict[int, LevelShapeStatsRow]
"""Map `level -> LevelShapeStatsRow`."""


class SphericalOctreeBuilder:
    """Coordinate-specific spherical inference strategy used by `OctreeBuilder`."""

    @staticmethod
    def _median_positive(values: np.ndarray, *, tiny: float = 1e-12) -> float:
        """Compute the median of positive values above `tiny`."""
        pos = np.asarray(values, dtype=float)
        pos = pos[pos > float(tiny)]
        if pos.size == 0:
            raise ValueError("No positive values available to infer spacing.")
        return float(np.median(pos))

    @staticmethod
    def _circular_span(cell_phi: np.ndarray) -> np.ndarray:
        """Compute minimal wrapped angular span for each row of azimuth samples."""
        ordered = np.sort(np.mod(cell_phi, 2.0 * math.pi), axis=1)
        wrapped = np.concatenate((ordered, ordered[:, :1] + 2.0 * math.pi), axis=1)
        gaps = np.diff(wrapped, axis=1)
        return 2.0 * math.pi - np.max(gaps, axis=1)

    @staticmethod
    def _circular_mean(cell_phi: np.ndarray) -> np.ndarray:
        """Compute circular mean for each row of azimuth samples."""
        mean_complex = np.mean(np.exp(1j * cell_phi), axis=1)
        return np.mod(np.angle(mean_complex), 2.0 * math.pi)

    @staticmethod
    def _circular_span_and_mean(
        cell_phi: np.ndarray,
        *,
        ignore_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-cell circular span and mean with optional corner masking."""
        if ignore_mask is None:
            return (
                SphericalOctreeBuilder._circular_span(cell_phi),
                SphericalOctreeBuilder._circular_mean(cell_phi),
            )
        if ignore_mask.shape != cell_phi.shape:
            raise ValueError(
                f"ignore_mask shape {ignore_mask.shape} does not match cell_phi {cell_phi.shape}"
            )

        n_cells = cell_phi.shape[0]
        span = np.empty(n_cells, dtype=float)
        center = np.empty(n_cells, dtype=float)
        row_has_mask = np.any(ignore_mask, axis=1)
        row_no_mask = ~row_has_mask

        if np.any(row_no_mask):
            span[row_no_mask] = SphericalOctreeBuilder._circular_span(cell_phi[row_no_mask])
            center[row_no_mask] = SphericalOctreeBuilder._circular_mean(cell_phi[row_no_mask])

        for cell_id in np.flatnonzero(row_has_mask):
            vals = cell_phi[cell_id, ~ignore_mask[cell_id]]
            if vals.size < 2:
                span[cell_id] = 0.0
                center[cell_id] = np.nan
                continue
            vals = vals.reshape(1, -1)
            span[cell_id] = SphericalOctreeBuilder._circular_span(vals)[0]
            center[cell_id] = SphericalOctreeBuilder._circular_mean(vals)[0]
        return span, center

    @staticmethod
    def _axis_corner_mask(ds: Dataset, corners: np.ndarray, *, axis_rho_tol: float) -> np.ndarray:
        """Mark corners near the polar axis where azimuth is singular."""
        names = set(ds.variables)
        if not {"X [R]", "Y [R]"}.issubset(names):
            return np.zeros(corners.shape, dtype=bool)
        x = np.asarray(ds.variable("X [R]"), dtype=float)
        y = np.asarray(ds.variable("Y [R]"), dtype=float)
        rho = np.hypot(x, y)
        return rho[corners] <= float(axis_rho_tol)

    @staticmethod
    def _extract_phi(ds: Dataset) -> np.ndarray:
        """Extract wrapped azimuth values from dataset fields."""
        variable_names = set(ds.variables)
        if "Lon [deg]" in variable_names:
            lon_deg = np.asarray(ds.variable("Lon [deg]"), dtype=float)
            return np.deg2rad(np.mod(lon_deg, 360.0))
        if "Lon [rad]" in variable_names:
            lon_rad = np.asarray(ds.variable("Lon [rad]"), dtype=float)
            return np.mod(lon_rad, 2.0 * math.pi)
        if "phi [rad]" in variable_names:
            phi_rad = np.asarray(ds.variable("phi [rad]"), dtype=float)
            return np.mod(phi_rad, 2.0 * math.pi)
        if {"X [R]", "Y [R]"}.issubset(variable_names):
            x = np.asarray(ds.variable("X [R]"), dtype=float)
            y = np.asarray(ds.variable("Y [R]"), dtype=float)
            return np.mod(np.arctan2(y, x), 2.0 * math.pi)
        raise ValueError(
            "Could not determine phi. Need either (X [R], Y [R]) or Lon/phi fields. "
            f"Available variables are {list(ds.variables)}."
        )

    @staticmethod
    def infer_level_expectation(
        delta_phi: np.ndarray,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Infer dyadic levels and expected spans from observed `delta_phi`."""
        levels = np.full(delta_phi.shape, -1, dtype=np.int64)
        expected = np.full(delta_phi.shape, np.nan, dtype=float)
        positive = delta_phi > max(float(atol), 1e-12)
        if not np.any(positive):
            return levels, expected, float("nan")

        coarse = float(np.max(delta_phi[positive]))
        raw_level = np.log2(coarse / delta_phi[positive])
        guess = np.maximum(np.rint(raw_level).astype(np.int64), 0)
        expected_pos = coarse / np.exp2(guess)
        ok = np.isclose(delta_phi[positive], expected_pos, rtol=rtol, atol=atol)
        levels[positive] = np.where(ok, guess, -1)
        expected[positive] = expected_pos
        return levels, expected, coarse

    @staticmethod
    def infer_levels_from_span(
        delta_phi: np.ndarray,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-9,
    ) -> np.ndarray:
        """Infer integer dyadic levels from per-cell azimuth spans."""
        levels, _expected, _coarse = SphericalOctreeBuilder.infer_level_expectation(
            delta_phi,
            rtol=rtol,
            atol=atol,
        )
        return levels

    @staticmethod
    def compute_delta_phi_and_levels(
        ds: Dataset,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-9,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute per-cell `delta_phi` and inferred dyadic refinement levels."""
        corners = np.asarray(ds.corners, dtype=np.int64)
        if corners.ndim != 2:
            raise ValueError(f"Expected 2D corner array, got shape {corners.shape}.")
        if corners.shape[1] < 3:
            raise ValueError("Need at least 3 corners per cell to estimate delta_phi.")

        phi = SphericalOctreeBuilder._extract_phi(ds)
        cell_phi = phi[corners]
        axis_mask = SphericalOctreeBuilder._axis_corner_mask(ds, corners, axis_rho_tol=axis_rho_tol)
        delta_phi, center_phi = SphericalOctreeBuilder._circular_span_and_mean(cell_phi, ignore_mask=axis_mask)
        levels, expected, coarse = SphericalOctreeBuilder.infer_level_expectation(
            delta_phi,
            rtol=rtol,
            atol=atol,
        )
        return delta_phi, center_phi, levels, expected, coarse

    @staticmethod
    def infer_level_angular_shapes(
        ds: Dataset,
        corners: np.ndarray,
        delta_phi: np.ndarray,
        cell_levels: np.ndarray,
    ) -> LevelShapeStatsMap:
        """Infer per-level angular counts/spacings from spherical mesh geometry."""
        x = np.asarray(ds.variable("X [R]"), dtype=float)
        y = np.asarray(ds.variable("Y [R]"), dtype=float)
        z = np.asarray(ds.variable("Z [R]"), dtype=float)
        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(np.clip(z / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
        delta_theta = np.ptp(theta[corners], axis=1)

        out: LevelShapeStatsMap = {}
        unique_levels = sorted(set(int(v) for v in cell_levels.tolist() if int(v) >= 0))
        if not unique_levels:
            raise ValueError("No valid (>=0) cell levels available for tree inference.")

        for level in unique_levels:
            mask = cell_levels == level
            med_dphi = SphericalOctreeBuilder._median_positive(delta_phi[mask])
            med_dtheta = SphericalOctreeBuilder._median_positive(delta_theta[mask])
            n_phi = int(round((2.0 * math.pi) / med_dphi))
            n_theta = int(round(math.pi / med_dtheta))
            if n_phi <= 0 or n_theta <= 0:
                raise ValueError(
                    f"Invalid angular counts inferred at level {level}: n_theta={n_theta}, n_phi={n_phi}."
                )

            ref_dphi = (2.0 * math.pi) / n_phi
            ref_dtheta = math.pi / n_theta
            if not np.isclose(med_dphi, ref_dphi, rtol=2e-2, atol=1e-9):
                raise ValueError(
                    f"Level {level} has inconsistent dphi={med_dphi:.6e} vs inferred {ref_dphi:.6e}."
                )
            if not np.isclose(med_dtheta, ref_dtheta, rtol=2e-2, atol=1e-9):
                raise ValueError(
                    f"Level {level} has inconsistent dtheta={med_dtheta:.6e} vs inferred {ref_dtheta:.6e}."
                )
            out[level] = (n_theta, n_phi, med_dtheta, med_dphi, int(np.count_nonzero(mask)))
        return out

    def __init__(self, *, level_rtol: float = 1e-4, level_atol: float = 1e-9) -> None:
        """Store dyadic level-inference tolerances."""
        self.level_rtol = float(level_rtol)
        self.level_atol = float(level_atol)

    def infer_levels_from_delta_phi(self, delta_phi: np.ndarray) -> np.ndarray:
        """Infer dyadic refinement levels from per-cell `delta_phi` spans."""
        return self.infer_levels_from_span(
            delta_phi,
            rtol=self.level_rtol,
            atol=self.level_atol,
        )

    def compute_phi_levels(
        self,
        ds: Dataset,
        *,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute per-cell azimuth spans and dyadic levels from dataset geometry."""
        return self.compute_delta_phi_and_levels(
            ds,
            rtol=self.level_rtol,
            atol=self.level_atol,
            axis_rho_tol=axis_rho_tol,
        )

    def infer_level_shapes(
        self,
        ds: Dataset,
        corners: np.ndarray,
        *,
        cell_levels: np.ndarray | None = None,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> tuple[LevelShapeStatsMap, np.ndarray, int, int]:
        """Infer spherical level-shape map and validated levels."""
        delta_phi, _center_phi, auto_levels, _expected, _coarse = self.compute_phi_levels(
            ds,
            axis_rho_tol=axis_rho_tol,
        )
        levels = auto_levels if cell_levels is None else np.asarray(cell_levels, dtype=np.int64)
        levels = np.asarray(levels, dtype=np.int64)
        valid_levels = levels[levels >= 0]
        if valid_levels.size == 0:
            raise ValueError("No valid (>=0) levels available to infer octree.")
        level_shapes = self.infer_level_angular_shapes(ds, corners, delta_phi, levels)
        min_level = int(np.min(valid_levels))
        max_level = int(np.max(valid_levels))
        return level_shapes, levels, min_level, max_level
