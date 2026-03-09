#!/usr/bin/env python3
"""Octree interpolator and interpolation kernels."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Literal
from typing import NamedTuple

from numba import njit
from numba import prange
import numpy as np
from starwinds_readplt.dataset import Dataset

from .base import DEFAULT_AXIS_RHO_TOL
from .base import DEFAULT_COORD_SYSTEM
from .base import Octree
from .cartesian import CartesianLookupKernelState
from .cartesian import lookup_xyz_cell_id_kernel
from .spherical import LookupKernelState
from .spherical import lookup_rpa_cell_id_kernel

logger = logging.getLogger(__name__)

_DEFAULT_SEED_CHUNK_SIZE = 1024
_TWO_PI = 2.0 * math.pi


def _clear_stale_numba_cache() -> None:
    """Remove local stale numba cache files after module-path refactors.

    Numba cache indices can reference old module paths and raise
    `ModuleNotFoundError` while unpickling. This helper clears local cache
    entries for octree kernels so warmup can recompile fresh artifacts.
    """
    cache_dir = Path(__file__).resolve().with_name("__pycache__")
    if not cache_dir.is_dir():
        return
    for ext in ("*.nbi", "*.nbc"):
        for path in cache_dir.glob(ext):
            name = path.name
            if not (name.startswith("interpolator.") or name.startswith("spherical.")):
                continue
            try:
                path.unlink()
            except OSError:
                logger.debug("Could not remove stale numba cache file %s", str(path))

class InterpKernelState(NamedTuple):
    """Numba interpolation-kernel arrays with explicit field names."""

    point_values_2d: np.ndarray
    corners: np.ndarray
    bin_to_corner: np.ndarray
    cell_r0: np.ndarray
    cell_rden: np.ndarray
    cell_t0: np.ndarray
    cell_tden: np.ndarray
    cell_p_start: np.ndarray
    cell_p_width: np.ndarray
    cell_pden: np.ndarray
    cell_phi_full: np.ndarray
    cell_phi_tiny: np.ndarray


class CartesianInterpKernelState(NamedTuple):
    """Numba Cartesian interpolation-kernel arrays with explicit field names."""

    point_values_2d: np.ndarray
    corners: np.ndarray
    bin_to_corner: np.ndarray
    cell_x0: np.ndarray
    cell_xden: np.ndarray
    cell_y0: np.ndarray
    cell_yden: np.ndarray
    cell_z0: np.ndarray
    cell_zden: np.ndarray


@njit(cache=True)
def _trilinear_from_cell(
    out_row: np.ndarray,
    cell_id: int,
    r: float,
    polar: float,
    azimuth: float,
    interp_state: InterpKernelState,
) -> None:
    """Write one trilinear interpolation result row for one resolved cell.

    Consumes:
    - `out_row`: output row for one query and all value components.
    - `cell_id`, `r`, `polar`, `azimuth`: resolved cell/query coordinates.
    - `interp_state`: packed interpolation arrays.
    Returns:
    - `None`; writes interpolated values in-place to `out_row`.
    """
    cid = int(cell_id)

    u = (r - interp_state.cell_r0[cid]) / interp_state.cell_rden[cid]
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0

    v = (polar - interp_state.cell_t0[cid]) / interp_state.cell_tden[cid]
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0

    p_rel = (azimuth - interp_state.cell_p_start[cid]) % _TWO_PI
    if interp_state.cell_phi_tiny[cid]:
        w = 0.0
    else:
        if not interp_state.cell_phi_full[cid]:
            width = interp_state.cell_p_width[cid]
            if p_rel < 0.0:
                p_rel = 0.0
            elif p_rel > width:
                p_rel = width
        w = p_rel / interp_state.cell_pden[cid]
        if w < 0.0:
            w = 0.0
        elif w > 1.0:
            w = 1.0

    w0 = (1.0 - u) * (1.0 - v) * (1.0 - w)
    w1 = u * (1.0 - v) * (1.0 - w)
    w2 = (1.0 - u) * v * (1.0 - w)
    w3 = u * v * (1.0 - w)
    w4 = (1.0 - u) * (1.0 - v) * w
    w5 = u * (1.0 - v) * w
    w6 = (1.0 - u) * v * w
    w7 = u * v * w

    local = interp_state.corners[cid]
    map_row = interp_state.bin_to_corner[cid]
    c0 = int(local[int(map_row[0])])
    c1 = int(local[int(map_row[1])])
    c2 = int(local[int(map_row[2])])
    c3 = int(local[int(map_row[3])])
    c4 = int(local[int(map_row[4])])
    c5 = int(local[int(map_row[5])])
    c6 = int(local[int(map_row[6])])
    c7 = int(local[int(map_row[7])])

    ncomp = out_row.shape[0]
    for comp in range(ncomp):
        out_row[comp] = (
            w0 * interp_state.point_values_2d[c0, comp]
            + w1 * interp_state.point_values_2d[c1, comp]
            + w2 * interp_state.point_values_2d[c2, comp]
            + w3 * interp_state.point_values_2d[c3, comp]
            + w4 * interp_state.point_values_2d[c4, comp]
            + w5 * interp_state.point_values_2d[c5, comp]
            + w6 * interp_state.point_values_2d[c6, comp]
            + w7 * interp_state.point_values_2d[c7, comp]
        )


@njit(cache=True)
def _trilinear_from_cell_xyz(
    out_row: np.ndarray,
    cell_id: int,
    x: float,
    y: float,
    z: float,
    interp_state: CartesianInterpKernelState,
) -> None:
    """Write one Cartesian trilinear interpolation result row for one cell."""
    cid = int(cell_id)
    u = (x - interp_state.cell_x0[cid]) / interp_state.cell_xden[cid]
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    v = (y - interp_state.cell_y0[cid]) / interp_state.cell_yden[cid]
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    w = (z - interp_state.cell_z0[cid]) / interp_state.cell_zden[cid]
    if w < 0.0:
        w = 0.0
    elif w > 1.0:
        w = 1.0

    w0 = (1.0 - u) * (1.0 - v) * (1.0 - w)
    w1 = u * (1.0 - v) * (1.0 - w)
    w2 = (1.0 - u) * v * (1.0 - w)
    w3 = u * v * (1.0 - w)
    w4 = (1.0 - u) * (1.0 - v) * w
    w5 = u * (1.0 - v) * w
    w6 = (1.0 - u) * v * w
    w7 = u * v * w

    local = interp_state.corners[cid]
    map_row = interp_state.bin_to_corner[cid]
    c0 = int(local[int(map_row[0])])
    c1 = int(local[int(map_row[1])])
    c2 = int(local[int(map_row[2])])
    c3 = int(local[int(map_row[3])])
    c4 = int(local[int(map_row[4])])
    c5 = int(local[int(map_row[5])])
    c6 = int(local[int(map_row[6])])
    c7 = int(local[int(map_row[7])])

    ncomp = out_row.shape[0]
    for comp in range(ncomp):
        out_row[comp] = (
            w0 * interp_state.point_values_2d[c0, comp]
            + w1 * interp_state.point_values_2d[c1, comp]
            + w2 * interp_state.point_values_2d[c2, comp]
            + w3 * interp_state.point_values_2d[c3, comp]
            + w4 * interp_state.point_values_2d[c4, comp]
            + w5 * interp_state.point_values_2d[c5, comp]
            + w6 * interp_state.point_values_2d[c6, comp]
            + w7 * interp_state.point_values_2d[c7, comp]
        )


@njit(cache=True, parallel=True)
def _interp_batch_xyz(
    queries_xyz: np.ndarray,
    fill_values: np.ndarray,
    interp_state: InterpKernelState,
    lookup_state: LookupKernelState,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a batch of Cartesian queries with local previous-cell hinting.

    Consumes:
    - `queries_xyz`: `(n_query, 3)` Cartesian query array.
    - `fill_values`: `(n_comp,)` fallback values for misses.
    - `interp_state`, `lookup_state`: packed interpolation/lookup arrays.
    Returns:
    - `(values, cell_ids)` where `values` is `(n_query, n_comp)` and `cell_ids`
      is `(n_query,)` with `-1` for misses.
    """
    n_query = queries_xyz.shape[0]
    ncomp = interp_state.point_values_2d.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values_2d.dtype)
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = int(_DEFAULT_SEED_CHUNK_SIZE)
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cid = -1
        for i in range(start, end):
            for comp in range(ncomp):
                out[i, comp] = fill_values[comp]

            x = queries_xyz[i, 0]
            y = queries_xyz[i, 1]
            z = queries_xyz[i, 2]
            r = math.sqrt(x * x + y * y + z * z)
            if r == 0.0:
                polar = 0.0
            else:
                zr = z / r
                if zr < -1.0:
                    zr = -1.0
                elif zr > 1.0:
                    zr = 1.0
                polar = math.acos(zr)
            azimuth = math.atan2(y, x) % _TWO_PI
            cid = lookup_rpa_cell_id_kernel(
                r,
                polar,
                azimuth,
                lookup_state,
                hint_cid,
            )
            if cid < 0:
                hint_cid = -1
                continue
            cell_ids[i] = cid
            hint_cid = int(cid)
            _trilinear_from_cell(
                out[i],
                cid,
                r,
                polar,
                azimuth,
                interp_state,
            )
    return out, cell_ids


@njit(cache=True, parallel=True)
def _interp_batch_rpa(
    queries_rpa: np.ndarray,
    fill_values: np.ndarray,
    interp_state: InterpKernelState,
    lookup_state: LookupKernelState,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a batch of spherical queries with local previous-cell hinting.

    Consumes:
    - `queries_rpa`: `(n_query, 3)` spherical query array.
    - `fill_values`: `(n_comp,)` fallback values for misses.
    - `interp_state`, `lookup_state`: packed interpolation/lookup arrays.
    Returns:
    - `(values, cell_ids)` where `values` is `(n_query, n_comp)` and `cell_ids`
      is `(n_query,)` with `-1` for misses.
    """
    n_query = queries_rpa.shape[0]
    ncomp = interp_state.point_values_2d.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values_2d.dtype)
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = int(_DEFAULT_SEED_CHUNK_SIZE)
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cid = -1
        for i in range(start, end):
            for comp in range(ncomp):
                out[i, comp] = fill_values[comp]

            r = queries_rpa[i, 0]
            polar = queries_rpa[i, 1]
            azimuth = queries_rpa[i, 2] % _TWO_PI
            cid = lookup_rpa_cell_id_kernel(
                r,
                polar,
                azimuth,
                lookup_state,
                hint_cid,
            )
            if cid < 0:
                hint_cid = -1
                continue
            cell_ids[i] = cid
            hint_cid = int(cid)
            _trilinear_from_cell(
                out[i],
                cid,
                r,
                polar,
                azimuth,
                interp_state,
            )
    return out, cell_ids


@njit(cache=True, parallel=True)
def _interp_batch_xyz_cartesian(
    queries_xyz: np.ndarray,
    fill_values: np.ndarray,
    interp_state: CartesianInterpKernelState,
    lookup_state: CartesianLookupKernelState,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Cartesian queries for Cartesian trees via compiled kernels."""
    n_query = queries_xyz.shape[0]
    ncomp = interp_state.point_values_2d.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values_2d.dtype)
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = int(_DEFAULT_SEED_CHUNK_SIZE)
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cid = -1
        for i in range(start, end):
            for comp in range(ncomp):
                out[i, comp] = fill_values[comp]

            x = queries_xyz[i, 0]
            y = queries_xyz[i, 1]
            z = queries_xyz[i, 2]
            cid = lookup_xyz_cell_id_kernel(x, y, z, lookup_state, hint_cid)
            if cid < 0:
                hint_cid = -1
                continue
            cell_ids[i] = cid
            hint_cid = int(cid)
            _trilinear_from_cell_xyz(
                out[i],
                cid,
                x,
                y,
                z,
                interp_state,
            )
    return out, cell_ids

class OctreeInterpolator:
    """LinearNDInterpolator-like callable built on octree leaf lookup.

    Query algorithm:
    - Find containing leaf cell with octree lookup.
    - Convert query to local `(r, polar, azimuth)` coordinates.
    - Evaluate trilinear interpolation from the 8 corner nodes of that cell.

    Ray methods additionally split cells into a fixed 6-tet decomposition and
    produce piecewise-linear functions along the ray.
    """

    def __init__(
        self,
        ds: Dataset,
        values: str | np.ndarray,
        *,
        fill_value: float | np.ndarray = np.nan,
        query_space: Literal["xyz", "rpa"] = "xyz",
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
        level_rtol: float = 1e-4,
        level_atol: float = 1e-9,
        tree: Octree | None = None,
    ) -> None:
        """Initialize lookup structures and interpolation caches from a plain dataset.

        If `tree` is provided, it is used directly.
        Consumes:
        - `ds`: point/corner dataset.
        - `values`: variable name or point-value array.
        - Optional tree/build controls (`tree`, tolerances, `query_space`).
        Returns:
        - `None`; constructs caches and compiles kernels for this interpolator.
        """
        self._ds = ds
        if ds.corners is None:
            logger.error("Dataset has no corners; cannot build interpolator.")
            raise ValueError("Dataset has no cell connectivity (corners).")
        self._corners = np.array(ds.corners, dtype=np.int64)
        self.fill_value = fill_value
        self.query_space = str(query_space)
        if self.query_space not in {"xyz", "rpa"}:
            logger.error("Invalid query_space=%s", self.query_space)
            raise ValueError("query_space must be 'xyz' or 'rpa'.")

        logger.debug(
            "Initializing OctreeInterpolator: query_space=%s, points=%d, cells=%d",
            self.query_space,
            int(ds.points.shape[0]),
            int(self._corners.shape[0]),
        )
        resolved_tree = tree
        if resolved_tree is None:
            build_coord = "rpa" if self.query_space == "rpa" else DEFAULT_COORD_SYSTEM
            resolved_tree = Octree.from_dataset(
                ds,
                coord_system=build_coord,
                axis_rho_tol=axis_rho_tol,
                level_rtol=level_rtol,
                level_atol=level_atol,
            )
        resolved_tree.bind(ds, axis_rho_tol=axis_rho_tol)
        self.tree = resolved_tree
        self.lookup = self.tree.lookup
        self._point_values = self._coerce_point_values(values)
        self._coord_system = str(self.tree.coord_system)
        if self._coord_system == "xyz" and self.query_space != "xyz":
            logger.error("query_space='rpa' is only supported for spherical (coord_system='rpa') trees.")
            raise ValueError("query_space='rpa' is only supported for coord_system='rpa'.")
        if self._coord_system == "rpa":
            self._prepare_spherical_points()
            self._prepare_trilinear_cache()
        elif self._coord_system == "xyz":
            self._prepare_trilinear_cache_xyz()
        else:
            raise NotImplementedError(
                f"Unsupported tree coord_system '{self._coord_system}' for interpolation."
            )
        self.prepare_kernel_cache()
        self.warmup_kernels()
        logger.info(
            "Interpolator ready: uniform=%s, depth=%d, leaf_shape=%s",
            self.tree.is_uniform,
            int(self.tree.depth),
            tuple(self.tree.leaf_shape),
        )

    def _coerce_point_values(self, values: str | np.ndarray) -> np.ndarray:
        """Resolve interpolation values into a point-centered array.

        Consumes:
        - `values`: variable name or value array.
        Returns:
        - `np.ndarray` whose first dimension matches dataset points.
        """
        if isinstance(values, str):
            arr = np.array(self._ds.variable(values))
        else:
            arr = np.array(values)

        n_points = int(self._ds.points.shape[0])
        if arr.shape[0] == n_points:
            logger.debug("Using point-centered values with shape=%s", tuple(arr.shape))
            return arr

        logger.error("Value size mismatch: values=%d, n_points=%d", int(arr.shape[0]), n_points)
        raise ValueError(f"values length {arr.shape[0]} does not match required n_points={n_points}.")

    def _prepare_spherical_points(self) -> None:
        """Precompute nodal spherical coordinates from dataset `X/Y/Z`.

        Consumes:
        - Bound dataset coordinate variables on `self`.
        Returns:
        - `None`; stores nodal `r/theta/phi` arrays on `self`.
        """
        x = np.array(self._ds.variable("X [R]"), dtype=float)
        y = np.array(self._ds.variable("Y [R]"), dtype=float)
        z = np.array(self._ds.variable("Z [R]"), dtype=float)
        r = np.sqrt(x * x + y * y + z * z)
        self._node_r = r
        self._node_theta = np.arccos(np.clip(z / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
        self._node_phi = np.mod(np.arctan2(y, x), 2.0 * math.pi)

    def _prepare_trilinear_cache(self) -> None:
        """Precompute per-cell trilinear mapping data.

        Each cell corner is mapped to one logical trilinear corner index using
        midpoint bit tests in `(r, polar, phi)`. Missing logical corners are
        filled by nearest bit-pattern match.
        Consumes:
        - Corner connectivity and precomputed nodal spherical coordinates.
        Returns:
        - `None`; stores per-cell interpolation mapping/cache arrays on `self`.
        """
        corners = self._corners
        vr = self._node_r[corners]
        vt = self._node_theta[corners]
        vp = self._node_phi[corners]

        self._cell_r0 = np.min(vr, axis=1)
        self._cell_r1 = np.max(vr, axis=1)
        self._cell_t0 = np.min(vt, axis=1)
        self._cell_t1 = np.max(vt, axis=1)
        self._cell_p_start = self.lookup._cell_phi_start
        self._cell_p_width = self.lookup._cell_phi_width

        tiny = np.finfo(float).tiny
        self._cell_rden = np.maximum(self._cell_r1 - self._cell_r0, tiny)
        self._cell_tden = np.maximum(self._cell_t1 - self._cell_t0, tiny)
        self._cell_pden = np.maximum(self._cell_p_width, tiny)
        self._cell_phi_full = self._cell_p_width >= (2.0 * math.pi - 1e-10)
        self._cell_phi_tiny = self._cell_p_width <= tiny

        p_rel = np.mod(vp - self._cell_p_start[:, None], 2.0 * math.pi)
        clip_mask = (~self._cell_phi_full)[:, None]
        p_rel = np.where(clip_mask, np.clip(p_rel, 0.0, self._cell_p_width[:, None]), p_rel)

        r_mid = 0.5 * (self._cell_r0 + self._cell_r1)[:, None]
        t_mid = 0.5 * (self._cell_t0 + self._cell_t1)[:, None]
        p_mid = 0.5 * self._cell_p_width[:, None]

        bit_r = (vr >= r_mid).astype(np.int8)
        bit_t = (vt >= t_mid).astype(np.int8)
        bit_p = np.zeros_like(bit_r, dtype=np.int8)
        valid_phi = ~self._cell_phi_tiny
        if np.any(valid_phi):
            bit_p[valid_phi] = (p_rel[valid_phi] >= p_mid[valid_phi]).astype(np.int8)

        bin_id = bit_r + (bit_t << 1) + (bit_p << 2)
        bit_trip = np.stack((bit_r, bit_t, bit_p), axis=2)
        target_bits = np.array(
            [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
            dtype=np.int8,
        )

        n_cells = corners.shape[0]
        bin_to_corner = np.empty((n_cells, 8), dtype=np.int8)
        for k in range(8):
            eq = bin_id == k
            has = np.any(eq, axis=1)
            pick = np.argmax(eq, axis=1).astype(np.int64)
            missing = ~has
            if np.any(missing):
                d = np.sum((bit_trip[missing] - target_bits[k]) ** 2, axis=2)
                pick[missing] = np.argmin(d, axis=1)
            bin_to_corner[:, k] = pick.astype(np.int8)
        self._bin_to_corner = bin_to_corner

    def _prepare_trilinear_cache_xyz(self) -> None:
        """Precompute per-cell trilinear mapping data for Cartesian `(x, y, z)` cells.

        Consumes:
        - Corner connectivity and nodal Cartesian coordinates.
        Returns:
        - `None`; stores per-cell Cartesian interpolation caches on `self`.
        """
        corners = self._corners
        pts = np.array(self.lookup._points, dtype=float)
        vx = pts[corners, 0]
        vy = pts[corners, 1]
        vz = pts[corners, 2]

        self._cell_x0 = np.min(vx, axis=1)
        self._cell_x1 = np.max(vx, axis=1)
        self._cell_y0 = np.min(vy, axis=1)
        self._cell_y1 = np.max(vy, axis=1)
        self._cell_z0 = np.min(vz, axis=1)
        self._cell_z1 = np.max(vz, axis=1)

        tiny = np.finfo(float).tiny
        self._cell_xden = np.maximum(self._cell_x1 - self._cell_x0, tiny)
        self._cell_yden = np.maximum(self._cell_y1 - self._cell_y0, tiny)
        self._cell_zden = np.maximum(self._cell_z1 - self._cell_z0, tiny)

        x_mid = 0.5 * (self._cell_x0 + self._cell_x1)[:, None]
        y_mid = 0.5 * (self._cell_y0 + self._cell_y1)[:, None]
        z_mid = 0.5 * (self._cell_z0 + self._cell_z1)[:, None]

        bit_x = (vx >= x_mid).astype(np.int8)
        bit_y = (vy >= y_mid).astype(np.int8)
        bit_z = (vz >= z_mid).astype(np.int8)
        bin_id = bit_x + (bit_y << 1) + (bit_z << 2)
        bit_trip = np.stack((bit_x, bit_y, bit_z), axis=2)
        target_bits = np.array(
            [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
            dtype=np.int8,
        )

        n_cells = corners.shape[0]
        bin_to_corner = np.empty((n_cells, 8), dtype=np.int8)
        for k in range(8):
            eq = bin_id == k
            has = np.any(eq, axis=1)
            pick = np.argmax(eq, axis=1).astype(np.int64)
            missing = ~has
            if np.any(missing):
                d = np.sum((bit_trip[missing] - target_bits[k]) ** 2, axis=2)
                pick[missing] = np.argmin(d, axis=1)
            bin_to_corner[:, k] = pick.astype(np.int8)
        self._bin_to_corner = bin_to_corner

    def prepare_kernel_cache(self) -> None:
        """Pack contiguous arrays used by compiled lookup/interpolation kernels."""
        flat = self._point_values.reshape(int(self._point_values.shape[0]), -1)
        self._point_values_2d = np.array(flat, dtype=np.float64, order="C")
        self._n_value_components = int(self._point_values_2d.shape[1])
        self._bin_to_corner_index = np.array(self._bin_to_corner, dtype=np.int64, order="C")
        if self._coord_system == "rpa":
            self._interp_state = InterpKernelState(
                point_values_2d=self._point_values_2d,
                corners=self._corners,
                bin_to_corner=self._bin_to_corner_index,
                cell_r0=self._cell_r0,
                cell_rden=self._cell_rden,
                cell_t0=self._cell_t0,
                cell_tden=self._cell_tden,
                cell_p_start=self._cell_p_start,
                cell_p_width=self._cell_p_width,
                cell_pden=self._cell_pden,
                cell_phi_full=self._cell_phi_full,
                cell_phi_tiny=self._cell_phi_tiny,
            )
            self._lookup_state = self.lookup._lookup_state
            return
        if self._coord_system == "xyz":
            self._interp_state_xyz = CartesianInterpKernelState(
                point_values_2d=self._point_values_2d,
                corners=self._corners,
                bin_to_corner=self._bin_to_corner_index,
                cell_x0=self._cell_x0,
                cell_xden=self._cell_xden,
                cell_y0=self._cell_y0,
                cell_yden=self._cell_yden,
                cell_z0=self._cell_z0,
                cell_zden=self._cell_zden,
            )
            self._lookup_state_xyz = self.lookup._lookup_state
            return
        raise NotImplementedError(
            f"Unsupported tree coord_system '{self._coord_system}' for kernel cache preparation."
        )

    def _fill_value_vector(self) -> np.ndarray:
        """Normalize `fill_value` into one vector matching component count.

        Consumes:
        - `self.fill_value` and cached number of value components.
        Returns:
        - `np.ndarray` with shape `(n_components,)`.
        """
        ncomp = int(self._n_value_components)
        if np.isscalar(self.fill_value):
            return np.full(ncomp, float(self.fill_value), dtype=np.float64)

        fill = np.array(self.fill_value, dtype=np.float64).reshape(-1)
        if fill.size == 1:
            return np.full(ncomp, float(fill[0]), dtype=np.float64)
        if fill.size != ncomp:
            raise ValueError(
                f"fill_value has {fill.size} entries but interpolated values require {ncomp} components."
            )
        return fill

    def warmup_kernels(self) -> None:
        """Trigger JIT compilation ahead of first real query."""
        q_xyz = np.array(self.lookup._points[:1], dtype=np.float64, order="C")
        if q_xyz.shape[0] == 0:
            q_xyz = np.zeros((1, 3), dtype=np.float64)
        fill = self._fill_value_vector()
        if self._coord_system == "xyz":
            _interp_batch_xyz_cartesian(
                q_xyz,
                fill,
                self._interp_state_xyz,
                self._lookup_state_xyz,
            )
            return
        if self._coord_system == "rpa":
            r, polar, azimuth = self.xyz_to_rpa(q_xyz[0])
            q_rpa = np.array([[r, polar, azimuth]], dtype=np.float64, order="C")
            try:
                _interp_batch_xyz(
                    q_xyz,
                    fill,
                    self._interp_state,
                    self._lookup_state,
                )
                _interp_batch_rpa(
                    q_rpa,
                    fill,
                    self._interp_state,
                    self._lookup_state,
                )
            except ModuleNotFoundError as exc:
                text = str(exc)
                stale_refs = ("starwinds_analysis.octree.lookup", "starwinds_analysis.octree.core")
                if not any(ref in text for ref in stale_refs):
                    raise
                logger.warning("Detected stale numba cache references; clearing local cache and retrying warmup.")
                _clear_stale_numba_cache()
                _interp_batch_xyz(
                    q_xyz,
                    fill,
                    self._interp_state,
                    self._lookup_state,
                )
                _interp_batch_rpa(
                    q_rpa,
                    fill,
                    self._interp_state,
                    self._lookup_state,
                )
            return
        raise NotImplementedError(
            f"Unsupported tree coord_system '{self._coord_system}' for kernel warmup."
        )

    @staticmethod
    def rpa_to_xyz(r: float, polar: float, azimuth: float) -> tuple[float, float, float]:
        """Convert one spherical point to Cartesian `(x, y, z)`.

        Consumes:
        - Spherical query scalars `r`, `polar`, `azimuth`.
        Returns:
        - Cartesian coordinate tuple `(x, y, z)`.
        """
        rr = float(r)
        pp = float(polar)
        aa = float(azimuth)
        sin_p = math.sin(pp)
        return rr * sin_p * math.cos(aa), rr * sin_p * math.sin(aa), rr * math.cos(pp)

    @staticmethod
    def prepare_queries(*args) -> tuple[np.ndarray, tuple[int, ...]]:
        """Normalize query inputs to `(N, 3)` plus broadcast output shape.

        Supports:
        - `xi` with shape `(..., 3)`
        - tuple/list of 3 broadcastable arrays
        - three separate coordinate arrays.
        Consumes:
        - Query arguments in any supported form above.
        Returns:
        - `(q, shape)` where `q` is `(N, 3)` and `shape` is the broadcasted
          leading output shape.
        """
        if len(args) == 1:
            xi = args[0]
            if isinstance(xi, tuple):
                if len(xi) != 3:
                    raise ValueError("Tuple input must have exactly 3 arrays.")
                a0, a1, a2 = np.broadcast_arrays(*[np.array(v, dtype=float) for v in xi])
                shape = a0.shape
                q = np.stack((a0, a1, a2), axis=-1).reshape(-1, 3)
                return q, shape

            arr = np.array(xi, dtype=float)
            if arr.ndim == 1:
                if arr.size != 3:
                    raise ValueError("1D xi must have length 3.")
                return arr.reshape(1, 3), ()
            if arr.shape[-1] != 3:
                raise ValueError("xi must have shape (..., 3).")
            return arr.reshape(-1, 3), arr.shape[:-1]

        if len(args) == 3:
            a0, a1, a2 = np.broadcast_arrays(*[np.array(v, dtype=float) for v in args])
            shape = a0.shape
            q = np.stack((a0, a1, a2), axis=-1).reshape(-1, 3)
            return q, shape

        raise ValueError("Call with xi or with x1, x2, x3.")

    @staticmethod
    def xyz_to_rpa(q: np.ndarray) -> tuple[float, float, float]:
        """Convert one Cartesian point to spherical `(r, polar, azimuth)`.

        Consumes:
        - `q`: one Cartesian point with 3 coordinates.
        Returns:
        - `(r, polar, azimuth)` as floats.
        """
        x = float(q[0])
        y = float(q[1])
        z = float(q[2])
        r = float(math.sqrt(x * x + y * y + z * z))
        if r == 0.0:
            polar = 0.0
        else:
            polar = float(math.acos(max(-1.0, min(1.0, z / r))))
        azimuth = float(math.atan2(y, x) % _TWO_PI)
        return r, polar, azimuth

    def __call__(
        self,
        *args,
        query_space: Literal["xyz", "rpa"] | None = None,
        return_cell_ids: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Evaluate interpolation at query points (optionally returning `cell_id`).

        For each query:
        - resolve containing cell via octree lookup,
        - convert to local spherical coordinates,
        - evaluate cached trilinear interpolation.
        Consumes:
        - Query arguments in supported scalar/array forms.
        - Optional `query_space` override and `return_cell_ids` flag.
        Returns:
        - Interpolated values reshaped to broadcasted query shape.
        - When `return_cell_ids=True`, returns `(values, cell_ids)`.
        """
        qs = self.query_space if query_space is None else str(query_space)
        if qs not in {"xyz", "rpa"}:
            logger.error("Invalid query_space=%s in call", qs)
            raise ValueError("query_space must be 'xyz' or 'rpa'.")
        if self._coord_system == "xyz" and qs == "rpa":
            logger.error("query_space='rpa' is not supported for Cartesian trees.")
            raise ValueError("query_space='rpa' is only supported for coord_system='rpa'.")

        debug_timing = logger.isEnabledFor(logging.DEBUG)
        t0_total = perf_counter() if debug_timing else 0.0

        q, shape = self.prepare_queries(*args)
        t_after_prepare = perf_counter() if debug_timing else 0.0
        q_array = np.array(q, dtype=np.float64, order="C")
        t_after_convert = perf_counter() if debug_timing else 0.0
        n = q_array.shape[0]
        trailing = self._point_values.shape[1:]
        logger.debug("Interpolating %d query points in %s space", int(n), qs)
        fill = self._fill_value_vector()
        t_after_fill = perf_counter() if debug_timing else 0.0

        if self._coord_system == "rpa":
            if qs == "xyz":
                batch_kernel = _interp_batch_xyz
            else:
                batch_kernel = _interp_batch_rpa
            if debug_timing:
                logger.debug("Interpolation kernel mode: compiled-rpa")
            out2d, cell_ids = batch_kernel(
                q_array,
                fill,
                self._interp_state,
                self._lookup_state,
            )
        else:
            q_xyz = q_array
            if debug_timing:
                logger.debug("Interpolation kernel mode: compiled-xyz")
            out2d, cell_ids = _interp_batch_xyz_cartesian(
                q_xyz,
                fill,
                self._interp_state_xyz,
                self._lookup_state_xyz,
            )
        t_after_kernel = perf_counter() if debug_timing else 0.0

        misses = int(np.count_nonzero(cell_ids < 0))
        if misses == n and n > 0:
            logger.warning("All query points were outside interpolation domain (%d/%d misses).", misses, n)
        elif misses > 0:
            logger.info("Some query points were outside interpolation domain (%d/%d misses).", misses, n)

        out = out2d.reshape((n,) + trailing).reshape(shape + trailing)
        t_after_post = perf_counter() if debug_timing else 0.0
        if debug_timing:
            prep_s = t_after_prepare - t0_total
            convert_s = t_after_convert - t_after_prepare
            fill_s = t_after_fill - t_after_convert
            kernel_s = t_after_kernel - t_after_fill
            post_s = t_after_post - t_after_kernel
            total_s = t_after_post - t0_total
            logger.debug(
                (
                    "Interpolation timings: "
                    f"n={int(n)} qs={qs} prep={prep_s:.6f}s convert={convert_s:.6f}s "
                    f"fill={fill_s:.6f}s kernel={kernel_s:.6f}s post={post_s:.6f}s "
                    f"total={total_s:.6f}s "
                    f"kernel_share={((kernel_s / total_s) if total_s > 0.0 else float('nan')):.3f}"
                )
            )
        if return_cell_ids:
            return out, cell_ids.reshape(shape)
        return out

logger.addHandler(logging.NullHandler())
