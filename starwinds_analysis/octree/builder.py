#!/usr/bin/env python3
"""Octree level inference and tree builder utilities."""

from __future__ import annotations

from collections import Counter
import itertools
import math
import re

import numpy as np
from starwinds_readplt.dataset import Dataset

from .base import BlockAux
from .base import BlockShapeInference
from .base import CoordSystem
from .base import DEFAULT_AXIS_RHO_TOL
from .base import DEFAULT_COORD_SYSTEM
from .base import GridShape
from .base import LevelAngularShapeMap
from .base import LevelCountTable
from .base import Octree
from .base import SUPPORTED_COORD_SYSTEMS
from .base import ScoredBlockShapeCandidate
from .base import _octree_class_for_coord

def _circular_span(cell_phi: np.ndarray) -> np.ndarray:
    """Compute minimal wrapped angular span for each row of azimuth samples.

    Consumes:
    - `cell_phi`: angle array shaped `(n_cells, n_angles)` in radians.
    Returns:
    - Span array `(n_cells,)` in radians.
    """
    ordered = np.sort(np.mod(cell_phi, 2.0 * math.pi), axis=1)
    wrapped = np.concatenate((ordered, ordered[:, :1] + 2.0 * math.pi), axis=1)
    gaps = np.diff(wrapped, axis=1)
    return 2.0 * math.pi - np.max(gaps, axis=1)


def _circular_mean(cell_phi: np.ndarray) -> np.ndarray:
    """Compute circular mean for each row of azimuth samples.

    Consumes:
    - `cell_phi`: angle array shaped `(n_cells, n_angles)` in radians.
    Returns:
    - Circular-mean array `(n_cells,)` in radians wrapped to `[0, 2pi)`.
    """
    mean_complex = np.mean(np.exp(1j * cell_phi), axis=1)
    return np.mod(np.angle(mean_complex), 2.0 * math.pi)


def _circular_span_and_mean(
    cell_phi: np.ndarray,
    *,
    ignore_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-cell circular span and mean with optional corner masking.

    Consumes:
    - `cell_phi`: per-cell azimuth values.
    - Optional `ignore_mask`: same shape as `cell_phi`, `True` corners ignored.
    Returns:
    - `(span, center)` arrays, each shaped `(n_cells,)`.
    """
    if ignore_mask is None:
        return _circular_span(cell_phi), _circular_mean(cell_phi)

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
        span[row_no_mask] = _circular_span(cell_phi[row_no_mask])
        center[row_no_mask] = _circular_mean(cell_phi[row_no_mask])

    for cell_id in np.flatnonzero(row_has_mask):
        vals = cell_phi[cell_id, ~ignore_mask[cell_id]]
        if vals.size < 2:
            span[cell_id] = 0.0
            center[cell_id] = np.nan
            continue
        vals = vals.reshape(1, -1)
        span[cell_id] = _circular_span(vals)[0]
        center[cell_id] = _circular_mean(vals)[0]

    return span, center


def _axis_corner_mask(ds: Dataset, corners: np.ndarray, *, axis_rho_tol: float) -> np.ndarray:
    """Mark corners near the polar axis where azimuth is singular.

    Consumes:
    - `ds`: dataset containing `X [R]`/`Y [R]` coordinates.
    - `corners`: corner connectivity array.
    - `axis_rho_tol`: cylindrical-radius tolerance for axis detection.
    Returns:
    - Boolean mask aligned with `corners`.
    """
    names = set(ds.variables)
    if not {"X [R]", "Y [R]"}.issubset(names):
        return np.zeros(corners.shape, dtype=bool)

    x = np.array(ds.variable("X [R]"), dtype=float)
    y = np.array(ds.variable("Y [R]"), dtype=float)
    rho = np.hypot(x, y)
    return rho[corners] <= float(axis_rho_tol)


def compute_delta_phi_and_levels(
    ds: Dataset,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-9,
    axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute per-cell delta_phi and integer refinement levels.

    Levels are inferred assuming dyadic refinement in phi:
    delta_phi(level) = delta_phi_coarsest / 2**level
    Consumes:
    - `ds`: dataset with corners and azimuth-defining coordinates.
    - Level inference tolerances `rtol`, `atol`.
    - `axis_rho_tol`: axis-corner tolerance.
    Returns:
    - `(delta_phi, center_phi, levels, expected_delta_phi, coarse_delta_phi)`.
    """
    corners = np.array(ds.corners, dtype=np.int64)

    if corners.ndim != 2:
        raise ValueError(f"Expected 2D corner array, got shape {corners.shape}.")
    if corners.shape[1] < 3:
        raise ValueError("Need at least 3 corners per cell to estimate delta_phi.")

    phi = _extract_phi(ds)
    cell_phi = phi[corners]
    axis_mask = _axis_corner_mask(ds, corners, axis_rho_tol=axis_rho_tol)
    delta_phi, center_phi = _circular_span_and_mean(cell_phi, ignore_mask=axis_mask)
    levels, expected, coarse = _infer_level_expectation(
        delta_phi,
        rtol=rtol,
        atol=atol,
    )
    return delta_phi, center_phi, levels, expected, coarse


def _infer_level_expectation(
    delta_phi: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Infer dyadic levels and expected spans from observed `delta_phi`.

    Consumes:
    - `delta_phi`: per-cell azimuth spans.
    - `rtol`, `atol`: dyadic-match tolerances.
    Returns:
    - `(levels, expected, coarse)` arrays/scalar.
    """
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


def _extract_phi(ds: Dataset) -> np.ndarray:
    """Extract wrapped azimuth values from dataset fields.

    Consumes:
    - `ds`: dataset containing either lon/phi variables or `X [R]`/`Y [R]`.
    Returns:
    - Wrapped azimuth array in radians.
    """
    variable_names = set(ds.variables)
    if "Lon [deg]" in variable_names:
        lon_deg = np.array(ds.variable("Lon [deg]"), dtype=float)
        return np.deg2rad(np.mod(lon_deg, 360.0))
    if "Lon [rad]" in variable_names:
        lon_rad = np.array(ds.variable("Lon [rad]"), dtype=float)
        return np.mod(lon_rad, 2.0 * math.pi)
    if "phi [rad]" in variable_names:
        phi_rad = np.array(ds.variable("phi [rad]"), dtype=float)
        return np.mod(phi_rad, 2.0 * math.pi)
    if {"X [R]", "Y [R]"}.issubset(variable_names):
        x = np.array(ds.variable("X [R]"), dtype=float)
        y = np.array(ds.variable("Y [R]"), dtype=float)
        return np.mod(np.arctan2(y, x), 2.0 * math.pi)

    raise ValueError(
        "Could not determine phi. Need either (X [R], Y [R]) or Lon/phi fields. "
        f"Available variables are {list(ds.variables)}."
    )


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
    """Format level histogram text as `level:count` pairs.

    Consumes:
    - `levels`: integer level array.
    Returns:
    - Compact histogram string.
    """
    counts = Counter(int(v) for v in levels.tolist())
    return ", ".join(f"{lvl}:{counts[lvl]}" for lvl in sorted(counts))


def valid_cell_fraction(levels: np.ndarray) -> tuple[int, int, float]:
    """Compute valid-level fraction statistics.

    Consumes:
    - `levels`: integer level array.
    Returns:
    - `(valid_count, total_count, valid_fraction)` for `levels >= 0`.
    """
    total = int(levels.size)
    valid = int(np.count_nonzero(levels >= 0))
    frac = float(valid / total) if total > 0 else 0.0
    return valid, total, frac


def _twos_factor(n: int) -> int:
    """Compute the exponent of the largest power of two dividing `n`.

    Consumes:
    - Positive integer `n`.
    Returns:
    - Integer exponent `k` such that `2**k` divides `n`.
    """
    k = 0
    while n > 0 and (n % 2 == 0):
        n //= 2
        k += 1
    return k


def _median_positive(values: np.ndarray, *, tiny: float = 1e-12) -> float:
    """Compute median of values above a positivity threshold.

    Consumes:
    - `values`: numeric array.
    - `tiny`: lower bound for valid positive values.
    Returns:
    - Median positive value as float.
    """
    pos = np.array(values, dtype=float)
    pos = pos[pos > float(tiny)]
    if pos.size == 0:
        raise ValueError("No positive values available to infer spacing.")
    return float(np.median(pos))


def _full_tree_counts(leaf_shape: GridShape) -> tuple[LevelCountTable, GridShape, int]:
    """Compute full-tree counts, root shape, and depth from finest leaf shape.

    Consumes:
    - `leaf_shape`: finest grid shape `(nr, ntheta, nphi)`.
    Returns:
    - `(level_counts, root_shape, depth)`.
    """
    depth = min(_twos_factor(leaf_shape[0]), _twos_factor(leaf_shape[1]), _twos_factor(leaf_shape[2]))
    root_shape = (
        leaf_shape[0] >> depth,
        leaf_shape[1] >> depth,
        leaf_shape[2] >> depth,
    )
    base = int(root_shape[0] * root_shape[1] * root_shape[2])
    counts = tuple((level, base * (8**level), base * (8**level)) for level in range(depth + 1))
    return counts, root_shape, depth


def _infer_levels_from_span(
    delta_phi: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-9,
) -> np.ndarray:
    """Infer integer dyadic levels from per-cell azimuth spans.

    Consumes:
    - `delta_phi` and dyadic-match tolerances.
    Returns:
    - Integer levels array (`-1` for non-dyadic spans).
    """
    levels, _expected, _coarse = _infer_level_expectation(
        delta_phi,
        rtol=rtol,
        atol=atol,
    )
    return levels


def _infer_level_angular_shapes(
    ds: Dataset,
    corners: np.ndarray,
    delta_phi: np.ndarray,
    cell_levels: np.ndarray,
) -> LevelAngularShapeMap:
    """Infer per-level angular counts/spacings from mesh geometry.

    Consumes:
    - Dataset coordinates, corner connectivity, `delta_phi`, and cell levels.
    Returns:
    - Map `level -> (n_theta, n_phi, dtheta, dphi, n_cells)`.
    """
    x = np.array(ds.variable("X [R]"), dtype=float)
    y = np.array(ds.variable("Y [R]"), dtype=float)
    z = np.array(ds.variable("Z [R]"), dtype=float)
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.clip(z / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
    delta_theta = np.ptp(theta[corners], axis=1)

    out: LevelAngularShapeMap = {}
    unique_levels = sorted(set(int(v) for v in cell_levels.tolist() if int(v) >= 0))
    if not unique_levels:
        raise ValueError("No valid (>=0) cell levels available for tree inference.")

    for level in unique_levels:
        mask = cell_levels == level
        med_dphi = _median_positive(delta_phi[mask])
        med_dtheta = _median_positive(delta_theta[mask])
        n_phi = int(round((2.0 * math.pi) / med_dphi))
        n_theta = int(round(math.pi / med_dtheta))
        if n_phi <= 0 or n_theta <= 0:
            raise ValueError(
                f"Invalid angular counts inferred at level {level}: "
                f"n_theta={n_theta}, n_phi={n_phi}."
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


def _infer_xyz_levels_from_cell_spans(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    *,
    rtol: float = 2e-2,
    atol: float = 1e-10,
) -> np.ndarray:
    """Infer dyadic xyz refinement levels from per-cell axis-aligned spans.

    Consumes:
    - Per-cell spans `dx`, `dy`, `dz`.
    - Tolerances for dyadic consistency checks.
    Returns:
    - Integer levels array (`-1` for inconsistent/non-dyadic cells).
    """
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


def _infer_xyz_level_shapes(
    ds: Dataset,
    corners: np.ndarray,
    cell_levels: np.ndarray,
) -> LevelAngularShapeMap:
    """Infer per-level axis counts/spacings for Cartesian octrees.

    Consumes:
    - Dataset coordinates, corner connectivity, and per-cell levels.
    Returns:
    - Map `level -> (n_axis1, n_axis2, d_axis1, d_axis2, n_cells)`.
    """
    x = np.array(ds.variable("X [R]"), dtype=float)
    y = np.array(ds.variable("Y [R]"), dtype=float)
    z = np.array(ds.variable("Z [R]"), dtype=float)
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

    out: LevelAngularShapeMap = {}
    unique_levels = sorted(set(int(v) for v in cell_levels.tolist() if int(v) >= 0))
    if not unique_levels:
        raise ValueError("No valid (>=0) cell levels available for tree inference.")

    for level in unique_levels:
        mask = cell_levels == level
        med_dx = _median_positive(dx[mask])
        med_dy = _median_positive(dy[mask])
        med_dz = _median_positive(dz[mask])
        n_x = int(round(span_x / med_dx))
        n_y = int(round(span_y / med_dy))
        n_z = int(round(span_z / med_dz))
        if n_x <= 0 or n_y <= 0 or n_z <= 0:
            raise ValueError(
                f"Invalid xyz counts inferred at level {level}: "
                f"n_x={n_x}, n_y={n_y}, n_z={n_z}."
            )
        ref_dx = span_x / n_x
        ref_dy = span_y / n_y
        ref_dz = span_z / n_z
        if not np.isclose(med_dx, ref_dx, rtol=2e-2, atol=1e-9):
            raise ValueError(
                f"Level {level} has inconsistent dx={med_dx:.6e} vs inferred {ref_dx:.6e}."
            )
        if not np.isclose(med_dy, ref_dy, rtol=2e-2, atol=1e-9):
            raise ValueError(
                f"Level {level} has inconsistent dy={med_dy:.6e} vs inferred {ref_dy:.6e}."
            )
        if not np.isclose(med_dz, ref_dz, rtol=2e-2, atol=1e-9):
            raise ValueError(
                f"Level {level} has inconsistent dz={med_dz:.6e} vs inferred {ref_dz:.6e}."
            )

        # Reuse LevelAngularShapeRow slot as generic axis1/axis2 summary:
        # (n_axis1, n_axis2, d_axis1, d_axis2, n_cells).
        out[level] = (n_y, n_z, med_dy, med_dz, int(np.count_nonzero(mask)))
    return out


def _infer_leaf_shape_from_levels(
    level_shapes: LevelAngularShapeMap,
) -> tuple[GridShape, int, int]:
    """Infer finest leaf shape from per-level angular shape statistics.

    Consumes:
    - `level_shapes`: per-level angular/count data.
    Returns:
    - `(leaf_shape, weighted_fine_equivalent_cells, max_level)`.
    """
    max_level = max(level_shapes)
    n_theta_f = int(level_shapes[max_level][0])
    n_phi_f = int(level_shapes[max_level][1])
    weighted_cells = 0
    for level, (_n_theta, _n_phi, _dtheta, _dphi, count) in level_shapes.items():
        weighted_cells += int(count) * (8 ** int(max_level - level))

    denom = int(n_theta_f * n_phi_f)
    if denom <= 0:
        raise ValueError("Invalid finest angular denominator while inferring n_r.")
    n_r_float = weighted_cells / float(denom)
    n_r = int(round(n_r_float))
    if not np.isclose(n_r_float, float(n_r), rtol=0.0, atol=1e-9):
        raise ValueError(
            "Could not infer integer finest n_r from weighted cell counts: "
            f"weighted={weighted_cells}, n_theta={n_theta_f}, n_phi={n_phi_f}."
        )
    return (n_r, n_theta_f, n_phi_f), int(weighted_cells), max_level


def _parse_blocks_aux(text: str | None) -> BlockAux | None:
    """Parse BLOCKS aux string into block-count/block-shape metadata.

    Consumes:
    - Optional aux text value.
    Returns:
    - `(n_blocks, cells_per_block_xyz)` when parseable, else `None`.
    """
    if text is None:
        return None
    match = re.search(r"(\d+)\s+(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", text)
    if not match:
        return None
    n_blocks = int(match.group(1))
    block_cells = (int(match.group(2)), int(match.group(3)), int(match.group(4)))
    return n_blocks, block_cells


def _infer_block_shape(
    leaf_shape: GridShape,
    block_count: int,
    block_cells_xyz: GridShape,
) -> BlockShapeInference | None:
    """Infer block-cell shape/grid shape compatible with leaf layout.

    Consumes:
    - `leaf_shape`, block count, and aux block-cell extents.
    Returns:
    - `(cells_per_block, block_grid_shape)` if inferable, else `None`.
    """
    candidates: list[ScoredBlockShapeCandidate] = []
    for perm in set(itertools.permutations(block_cells_xyz, 3)):
        if not all(leaf_shape[i] % perm[i] == 0 for i in range(3)):
            continue
        shape = (
            leaf_shape[0] // perm[0],
            leaf_shape[1] // perm[1],
            leaf_shape[2] // perm[2],
        )
        if shape[0] * shape[1] * shape[2] != block_count:
            continue
        score = min(_twos_factor(shape[0]), _twos_factor(shape[1]), _twos_factor(shape[2]))
        candidates.append((perm, shape, score))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[2], reverse=True)
    best_perm, best_shape, _score = candidates[0]
    return best_perm, best_shape

class OctreeBuilder:
    """Build `Octree` objects from mesh geometry and inferred levels."""

    def __init__(
        self,
        *,
        level_rtol: float = 1e-4,
        level_atol: float = 1e-9,
    ) -> None:
        """Configure tolerances used for dyadic level inference.

        Consumes:
        - `level_rtol`, `level_atol` tolerance values.
        Returns:
        - `None`; stores tolerances on the builder instance.
        """
        self.level_rtol = float(level_rtol)
        self.level_atol = float(level_atol)

    def infer_levels_from_delta_phi(self, delta_phi: np.ndarray) -> np.ndarray:
        """Infer dyadic refinement levels from per-cell `delta_phi` spans.

        Consumes:
        - `delta_phi`: per-cell azimuth spans.
        Returns:
        - Integer levels array.
        """
        return _infer_levels_from_span(
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
        """Compute per-cell azimuth spans and dyadic levels from dataset geometry.

        Consumes:
        - Dataset plus axis singularity tolerance.
        Returns:
        - `(delta_phi, center_phi, levels, expected_delta_phi, coarse_delta_phi)`.
        """
        return compute_delta_phi_and_levels(
            ds,
            rtol=self.level_rtol,
            atol=self.level_atol,
            axis_rho_tol=axis_rho_tol,
        )

    def build_tree(
        self,
        ds: Dataset,
        corners: np.ndarray,
        axis2_span: np.ndarray | None = None,
        *,
        coord_system: CoordSystem = DEFAULT_COORD_SYSTEM,
        cell_levels: np.ndarray | None = None,
        axis2_center: np.ndarray | None = None,
        expected_axis2_span: np.ndarray | None = None,
        coarse_axis2_span: float | None = None,
    ) -> Octree:
        """Build an `Octree` from prepared level arrays and metadata.

        Consumes:
        - Dataset, corners, optional axis-2 span payload, coord-system tag, optional cached level data.
        Returns:
        - Constructed octree object (not yet guaranteed bound unless caller binds).
        """
        tree_cls = _octree_class_for_coord(coord_system)
        if coord_system == "rpa":
            if axis2_span is None:
                raise ValueError("axis2_span is required when coord_system='rpa'.")
            axis2_span = np.array(axis2_span, dtype=float)
            if cell_levels is None:
                cell_levels = self.infer_levels_from_delta_phi(axis2_span)
            cell_levels = np.array(cell_levels, dtype=np.int64)
            valid_levels = cell_levels[cell_levels >= 0]
            if valid_levels.size == 0:
                raise ValueError("No valid (>=0) levels available to infer octree.")
            level_shapes = _infer_level_angular_shapes(ds, corners, axis2_span, cell_levels)
            min_level = int(np.min(valid_levels))
            max_level = int(np.max(valid_levels))
            coarse_axis1_step = float(level_shapes[min_level][2])
            coarse_axis2_step = float(level_shapes[min_level][3])
            axis2_center_payload = None if axis2_center is None else np.array(axis2_center, dtype=float)
            axis2_span_payload = axis2_span
            expected_axis2_span_payload = (
                None if expected_axis2_span is None else np.array(expected_axis2_span, dtype=float)
            )
            coarse_axis2_span_payload = None if coarse_axis2_span is None else float(coarse_axis2_span)
        elif coord_system == "xyz":
            x = np.array(ds.variable("X [R]"), dtype=float)
            y = np.array(ds.variable("Y [R]"), dtype=float)
            z = np.array(ds.variable("Z [R]"), dtype=float)
            cell_x = x[corners]
            cell_y = y[corners]
            cell_z = z[corners]
            dx = np.ptp(cell_x, axis=1)
            dy = np.ptp(cell_y, axis=1)
            dz = np.ptp(cell_z, axis=1)
            if cell_levels is None:
                cell_levels = _infer_xyz_levels_from_cell_spans(
                    dx,
                    dy,
                    dz,
                    rtol=max(2e-2, float(self.level_rtol)),
                    atol=max(1e-10, float(self.level_atol)),
                )
            cell_levels = np.array(cell_levels, dtype=np.int64)
            valid_levels = cell_levels[cell_levels >= 0]
            if valid_levels.size == 0:
                raise ValueError("No valid (>=0) levels available to infer octree.")
            level_shapes = _infer_xyz_level_shapes(ds, corners, cell_levels)
            min_level = int(np.min(valid_levels))
            max_level = int(np.max(valid_levels))
            coarse_axis1_step = float(level_shapes[min_level][2])
            coarse_axis2_step = float(level_shapes[min_level][3])
            axis2_center_payload = None
            axis2_span_payload = None
            expected_axis2_span_payload = None
            coarse_axis2_span_payload = None
        else:
            raise ValueError(
                f"Unsupported coord_system '{coord_system}'; expected one of {SUPPORTED_COORD_SYSTEMS}."
            )

        # AMR-first path: this also covers uniform-level datasets.
        leaf_shape, weighted_cells, _max_level = _infer_leaf_shape_from_levels(level_shapes)
        _counts_full, root_shape, depth = _full_tree_counts(leaf_shape)
        level_counts = tuple(
            (
                int(level),
                int(level_shapes[level][4]),
                int(level_shapes[level][4] * (8 ** int(max_level - level))),
            )
            for level in sorted(level_shapes)
        )
        is_full = (
            int(valid_levels.size) == int(cell_levels.size)
            and int(sum(item[2] for item in level_counts)) == int(np.prod(leaf_shape))
            and int(weighted_cells) == int(np.prod(leaf_shape))
        )

        block_cell_shape = None
        block_shape = None
        block_root_shape = None
        block_depth = None
        block_counts = None
        parsed = _parse_blocks_aux(ds.aux.get("BLOCKS"))
        if parsed is not None:
            block_count, block_cells_xyz = parsed
            inferred = _infer_block_shape(leaf_shape, block_count, block_cells_xyz)
            if inferred is not None:
                block_cell_shape, block_shape = inferred
                block_counts, block_root_shape, block_depth = _full_tree_counts(block_shape)

        return tree_cls(
            leaf_shape=leaf_shape,
            root_shape=root_shape,
            depth=int(depth),
            is_full=bool(is_full),
            level_counts=level_counts,
            min_level=min_level,
            max_level=max_level,
            coarse_axis1_step=coarse_axis1_step,
            coarse_axis2_step=coarse_axis2_step,
            block_cell_shape=block_cell_shape,
            block_shape=block_shape,
            block_root_shape=block_root_shape,
            block_depth=block_depth,
            block_level_counts=block_counts,
            coord_system=coord_system,
            cell_levels=cell_levels,
            axis2_center=axis2_center_payload,
            axis2_span=axis2_span_payload,
            expected_axis2_span=expected_axis2_span_payload,
            coarse_axis2_span=coarse_axis2_span_payload,
        )

    def build(
        self,
        ds: Dataset,
        *,
        coord_system: CoordSystem = DEFAULT_COORD_SYSTEM,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> Octree:
        """Build and bind an `Octree` directly from dataset geometry.

        Consumes:
        - Dataset and builder options (`coord_system`, `axis_rho_tol`).
        Returns:
        - Built+bound octree instance.
        """
        if coord_system not in SUPPORTED_COORD_SYSTEMS:
            raise ValueError(
                f"Unsupported coord_system '{coord_system}'; "
                f"expected one of {SUPPORTED_COORD_SYSTEMS}."
            )
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot build octree.")
        corners = np.array(ds.corners, dtype=np.int64)
        if coord_system == "rpa":
            delta_phi, center_phi, cell_levels, expected, coarse = self.compute_phi_levels(
                ds,
                axis_rho_tol=axis_rho_tol,
            )
            tree = self.build_tree(
                ds,
                corners,
                axis2_span=delta_phi,
                coord_system=coord_system,
                cell_levels=cell_levels,
                axis2_center=center_phi,
                expected_axis2_span=expected,
                coarse_axis2_span=float(coarse),
            )
        else:
            tree = self.build_tree(
                ds,
                corners,
                axis2_span=None,
                coord_system=coord_system,
                cell_levels=None,
                axis2_center=None,
                expected_axis2_span=None,
                coarse_axis2_span=None,
            )
        tree.bind(ds, corners, axis_rho_tol=axis_rho_tol)
        return tree


def build_octree(
    ds: Dataset,
    corners: np.ndarray,
    axis2_span: np.ndarray | None = None,
    *,
    coord_system: CoordSystem = DEFAULT_COORD_SYSTEM,
    cell_levels: np.ndarray | None = None,
) -> Octree:
    """Build an octree from precomputed metadata without binding.

    Consumes:
    - Dataset, corners, optional axis-2 spans, optional `coord_system` and levels.
    Returns:
    - Constructed octree instance.
    """
    if coord_system not in SUPPORTED_COORD_SYSTEMS:
        raise ValueError(
            f"Unsupported coord_system '{coord_system}'; "
            f"expected one of {SUPPORTED_COORD_SYSTEMS}."
        )
    return OctreeBuilder().build_tree(
        ds,
        corners,
        axis2_span=axis2_span,
        coord_system=coord_system,
        cell_levels=cell_levels,
    )
