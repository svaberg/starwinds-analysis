#!/usr/bin/env python3
"""Spherical octree and spherical lookup implementation."""

from __future__ import annotations

import math
from typing import ClassVar
from typing import NamedTuple

from numba import njit
import numpy as np

from .base import GridIndex
from .base import GridPath
from .base import LookupHit
from .base import Octree
from .base import _CellLookup
from .base import _DEFAULT_LOOKUP_MAX_RADIUS
from .base import _LOOKUP_CONTAIN_TOL
from .base import _TWO_PI
from .builder import _circular_span_and_mean
from .builder import compute_delta_phi_and_levels


class LookupKernelState(NamedTuple):
    """Numba lookup-kernel arrays/scalars with explicit field names."""

    levels_desc: np.ndarray
    shape_table: np.ndarray
    dtheta_table: np.ndarray
    dphi_table: np.ndarray
    bin_level_offset: np.ndarray
    bin_offsets: np.ndarray
    bin_cell_ids: np.ndarray
    cell_r_min: np.ndarray
    cell_r_max: np.ndarray
    cell_theta_min: np.ndarray
    cell_theta_max: np.ndarray
    cell_phi_start: np.ndarray
    cell_phi_width: np.ndarray
    cell_centers: np.ndarray
    r_min: float
    r_max: float
    max_radius: int
@njit(cache=True)
def _contains_rpa_cell(
    cid: int,
    r: float,
    polar: float,
    azimuth: float,
    lookup_state: LookupKernelState,
) -> bool:
    """Check one spherical query against one cell's spherical bounds.

    Consumes:
    - `cid`: leaf-cell index.
    - `r`, `polar`, `azimuth`: query coordinates in `(r, polar, azimuth)`.
    - `lookup_state`: packed per-cell bounds arrays.
    Returns:
    - `True` when the query is inside the cell bounds (with tolerance), else `False`.
    """
    tol = _LOOKUP_CONTAIN_TOL
    if r < (lookup_state.cell_r_min[cid] - tol) or r > (lookup_state.cell_r_max[cid] + tol):
        return False
    if polar < (lookup_state.cell_theta_min[cid] - tol) or polar > (lookup_state.cell_theta_max[cid] + tol):
        return False
    width = lookup_state.cell_phi_width[cid]
    dphi = (azimuth - lookup_state.cell_phi_start[cid]) % _TWO_PI
    if width >= (_TWO_PI - tol):
        return True
    return dphi <= (width + tol)


@njit(cache=True)
def _lookup_rpa_cell_id(
    r: float,
    polar: float,
    azimuth: float,
    lookup_state: LookupKernelState,
    prev_cid: int = -1,
) -> int:
    """Resolve one spherical query to one leaf-cell id.

    Consumes:
    - Query scalars `r`, `polar`, `azimuth`.
    - `lookup_state`: packed level/bin/cell arrays.
    - `prev_cid`: previous hit hint (`-1` when unavailable).
    Returns:
    - A resolved `cell_id` on success.
    - `-1` for invalid/out-of-domain inputs.
    """
    if not (math.isfinite(r) and math.isfinite(polar) and math.isfinite(azimuth)):
        return -1
    if polar < 0.0 or polar > math.pi:
        return -1
    azimuth = azimuth % _TWO_PI
    if r < lookup_state.r_min or r > lookup_state.r_max:
        return -1
    if prev_cid >= 0 and _contains_rpa_cell(
        int(prev_cid),
        r,
        polar,
        azimuth,
        lookup_state,
    ):
        return int(prev_cid)

    sin_polar = math.sin(polar)
    qx = r * sin_polar * math.cos(azimuth)
    qy = r * sin_polar * math.sin(azimuth)
    qz = r * math.cos(polar)

    fallback_best = -1
    fallback_best_d2 = np.inf

    for radius in range(lookup_state.max_radius + 1):
        for level_index in range(lookup_state.levels_desc.shape[0]):
            level = int(lookup_state.levels_desc[level_index])
            ntheta = int(lookup_state.shape_table[level, 1])
            nphi = int(lookup_state.shape_table[level, 2])
            if ntheta <= 0 or nphi <= 0:
                continue

            dtheta = lookup_state.dtheta_table[level]
            dphi = lookup_state.dphi_table[level]
            if not (math.isfinite(dtheta) and math.isfinite(dphi) and dtheta > 0.0 and dphi > 0.0):
                continue

            ipolar = int(math.floor(polar / dtheta))
            iazimuth = int(math.floor(azimuth / dphi))
            if ipolar < 0:
                ipolar = 0
            elif ipolar >= ntheta:
                ipolar = ntheta - 1
            if iazimuth < 0:
                iazimuth = 0
            elif iazimuth >= nphi:
                iazimuth = nphi - 1

            level_offset = int(lookup_state.bin_level_offset[level])
            if level_offset < 0:
                continue

            inside_best = -1
            inside_best_d2 = np.inf

            for dt in range(-radius, radius + 1):
                tt = ipolar + dt
                if tt < 0 or tt >= ntheta:
                    continue
                for dp in range(-radius, radius + 1):
                    pp = (iazimuth + dp) % nphi
                    key = level_offset + tt * nphi + pp
                    start = int(lookup_state.bin_offsets[key])
                    end = int(lookup_state.bin_offsets[key + 1])
                    for pos in range(start, end):
                        cid = int(lookup_state.bin_cell_ids[pos])
                        dx = lookup_state.cell_centers[cid, 0] - qx
                        dy = lookup_state.cell_centers[cid, 1] - qy
                        dz = lookup_state.cell_centers[cid, 2] - qz
                        d2 = dx * dx + dy * dy + dz * dz

                        if d2 < fallback_best_d2:
                            fallback_best_d2 = d2
                            fallback_best = cid

                        if _contains_rpa_cell(
                            cid,
                            r,
                            polar,
                            azimuth,
                            lookup_state,
                        ) and d2 < inside_best_d2:
                            inside_best_d2 = d2
                            inside_best = cid

            if inside_best >= 0:
                return inside_best

    if fallback_best >= 0:
        return fallback_best

    n_cells = lookup_state.cell_centers.shape[0]
    pool_found = False
    pool_best = -1
    pool_best_d2 = np.inf
    pool_inside_best = -1
    pool_inside_best_d2 = np.inf
    for cid in range(n_cells):
        if (
            r < (lookup_state.cell_r_min[cid] - _LOOKUP_CONTAIN_TOL)
            or r > (lookup_state.cell_r_max[cid] + _LOOKUP_CONTAIN_TOL)
        ):
            continue
        pool_found = True
        dx = lookup_state.cell_centers[cid, 0] - qx
        dy = lookup_state.cell_centers[cid, 1] - qy
        dz = lookup_state.cell_centers[cid, 2] - qz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < pool_best_d2:
            pool_best_d2 = d2
            pool_best = cid
        if _contains_rpa_cell(
            cid,
            r,
            polar,
            azimuth,
            lookup_state,
        ) and d2 < pool_inside_best_d2:
            pool_inside_best_d2 = d2
            pool_inside_best = cid

    if pool_found:
        if pool_inside_best >= 0:
            return pool_inside_best
        return pool_best

    all_best = -1
    all_best_d2 = np.inf
    for cid in range(n_cells):
        dx = lookup_state.cell_centers[cid, 0] - qx
        dy = lookup_state.cell_centers[cid, 1] - qy
        dz = lookup_state.cell_centers[cid, 2] - qz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < all_best_d2:
            all_best_d2 = d2
            all_best = cid
    return all_best

class _SphericalCellLookup(_CellLookup):
    """Leaf-cell lookup accelerator for spherical/Cartesian queries.

    The index is built from leaf-cell centers and bounds in spherical space.
    Angular bins are stored in CSR-like arrays (`_bin_offsets/_bin_counts/
    _bin_cell_ids`) so candidate retrieval avoids tuple-key dict lookups in
    the hot path.
    """

    def __init__(
        self,
        tree: Octree,
    ) -> None:
        """Build lookup geometry from one bound tree.

        Consumes:
        - `tree`: an `Octree` bound to dataset+covers with valid XYZ variables.
        Returns:
        - `None`; initializes lookup arrays and compiled lookup state on `self`.
        """
        if tree.ds is None or tree.corners is None:
            raise ValueError("Lookup requires a bound octree with dataset and corners.")
        ds = tree.ds
        corners = tree.corners
        cell_levels = tree.cell_levels
        axis2_center = tree.axis2_center
        axis_rho_tol = float(tree.axis_rho_tol)
        if not {"X [R]", "Y [R]", "Z [R]"}.issubset(set(ds.variables)):
            raise ValueError("Lookup requires X/Y/Z variables.")

        self.tree = tree
        self._axis_rho_tol = axis_rho_tol
        self._corners = np.array(corners, dtype=np.int64)
        self._points = np.column_stack(
            (
                np.array(ds.variable("X [R]"), dtype=float),
                np.array(ds.variable("Y [R]"), dtype=float),
                np.array(ds.variable("Z [R]"), dtype=float),
            )
        )
        self._cell_centers = np.mean(self._points[self._corners], axis=1)
        if cell_levels is None or axis2_center is None:
            _delta_phi, axis2_center_auto, levels_auto, _expected, _coarse = compute_delta_phi_and_levels(
                ds,
                axis_rho_tol=axis_rho_tol,
            )
            if cell_levels is None:
                cell_levels = levels_auto
            if axis2_center is None:
                axis2_center = axis2_center_auto
        self._cell_level_rel = np.array(cell_levels, dtype=np.int64)
        self._axis2_center = np.array(axis2_center, dtype=float)
        self._build_index()

    def _build_index(self) -> None:
        """Build per-level lookup tables, bounds, and CSR-like angular bins.

        Steps:
        - Build dense per-level tables (shape, spacing, depth, bin offsets).
        - Assign each valid leaf cell to one `(level, itheta, iphi)` angular bin.
        - Sort cells in each bin by radial center for stable local ranking.
        - Pack bins into CSR-like arrays for candidate gathering.
        - Precompute per-cell radial/theta/phi bounds for containment tests.
        Consumes:
        - Bound dataset points/corners and per-cell levels on `self`.
        Returns:
        - `None`; writes packed lookup/index arrays to instance fields.
        """
        n_cells = self._corners.shape[0]
        valid = self._cell_level_rel >= 0
        if not np.any(valid):
            raise ValueError("Lookup requires at least one valid leaf level.")

        self._max_level = int(self.tree.max_level)
        (
            self._level_to_depth,
            self._shape_by_level,
            self._dtheta_by_level,
            self._dphi_by_level,
        ) = self.tree.level_maps(self._cell_level_rel[valid])

        levels_asc = np.array(sorted(self._shape_by_level.keys()), dtype=np.int64)
        self._levels_desc = levels_asc[::-1]
        level_cap = int(np.max(levels_asc)) + 1
        shape_table = np.full((level_cap, 3), -1, dtype=np.int64)
        dtheta_table = np.full(level_cap, np.nan, dtype=float)
        dphi_table = np.full(level_cap, np.nan, dtype=float)
        depth_table = np.full(level_cap, -1, dtype=np.int64)
        bin_level_offset = np.full(level_cap, -1, dtype=np.int64)
        running_offset = 0
        for level in levels_asc:
            lvl = int(level)
            shape = self._shape_by_level[lvl]
            shape_table[lvl, 0] = int(shape[0])
            shape_table[lvl, 1] = int(shape[1])
            shape_table[lvl, 2] = int(shape[2])
            dtheta_table[lvl] = float(self._dtheta_by_level[lvl])
            dphi_table[lvl] = float(self._dphi_by_level[lvl])
            depth_table[lvl] = int(self._level_to_depth[lvl])
            bin_level_offset[lvl] = running_offset
            running_offset += int(shape[1]) * int(shape[2])
        self._shape_table = shape_table
        self._dtheta_table = dtheta_table
        self._dphi_table = dphi_table
        self._bin_level_offset = bin_level_offset
        points_r = np.linalg.norm(self._points, axis=1)
        cell_r_center = np.linalg.norm(self._cell_centers, axis=1)
        cell_r_min = np.min(points_r[self._corners], axis=1)
        cell_r_max = np.max(points_r[self._corners], axis=1)
        theta_points = np.arccos(
            np.clip(self._points[:, 2] / np.maximum(points_r, np.finfo(float).tiny), -1.0, 1.0)
        )
        phi_points = np.mod(np.arctan2(self._points[:, 1], self._points[:, 0]), 2.0 * math.pi)
        ctheta = np.mean(theta_points[self._corners], axis=1)
        rho_points = np.hypot(self._points[:, 0], self._points[:, 1])
        axis_mask = rho_points[self._corners] <= self._axis_rho_tol
        _dphi_dummy, cphi_auto = _circular_span_and_mean(
            phi_points[self._corners],
            ignore_mask=axis_mask,
        )
        fallback_phi = np.mod(np.arctan2(self._cell_centers[:, 1], self._cell_centers[:, 0]), 2.0 * math.pi)
        cphi = np.where(np.isfinite(self._axis2_center), self._axis2_center, cphi_auto)
        cphi = np.where(np.isfinite(cphi), cphi, fallback_phi)
        cphi = np.mod(cphi, 2.0 * math.pi)

        itheta = np.full(n_cells, -1, dtype=np.int64)
        iphi = np.full(n_cells, -1, dtype=np.int64)
        ir_abs = np.full(n_cells, -1, dtype=np.int64)
        cell_depth = np.full(n_cells, -1, dtype=np.int64)

        n_bins = int(running_offset)
        bin_lists: list[list[int]] = [[] for _ in range(n_bins)]
        for cid in np.flatnonzero(valid):
            level = int(self._cell_level_rel[cid])
            ntheta = int(self._shape_table[level, 1])
            nphi = int(self._shape_table[level, 2])
            dtheta = float(self._dtheta_table[level])
            dphi = float(self._dphi_table[level])
            tt = int(np.clip(np.floor(ctheta[cid] / dtheta), 0, ntheta - 1))
            pp = int(np.clip(np.floor(cphi[cid] / dphi), 0, nphi - 1))
            itheta[cid] = tt
            iphi[cid] = pp
            cell_depth[cid] = int(depth_table[level])
            key = int(self._bin_level_offset[level] + tt * nphi + pp)
            bin_lists[key].append(int(cid))

        bin_counts = np.zeros(n_bins, dtype=np.int64)
        for key, ids in enumerate(bin_lists):
            if not ids:
                continue
            arr = np.array(ids, dtype=np.int64)
            order = np.argsort(cell_r_center[arr])
            sorted_ids = arr[order]
            bin_lists[key] = sorted_ids.tolist()
            bin_counts[key] = int(sorted_ids.size)

            level = int(self._cell_level_rel[int(sorted_ids[0])])
            nr_level = int(self._shape_table[level, 0])
            m = sorted_ids.size
            if m == 1:
                mapped = np.array([nr_level // 2], dtype=np.int64)
            else:
                mapped = np.floor(((np.arange(m, dtype=float) + 0.5) * nr_level) / float(m)).astype(np.int64)
            ir_abs[sorted_ids] = np.clip(mapped, 0, nr_level - 1)

        bin_offsets = np.zeros(n_bins + 1, dtype=np.int64)
        if n_bins > 0:
            np.cumsum(bin_counts, out=bin_offsets[1:])
        total_refs = int(bin_offsets[-1])
        bin_cell_ids = np.empty(total_refs, dtype=np.int64)
        for key in range(n_bins):
            start = int(bin_offsets[key])
            end = int(bin_offsets[key + 1])
            if end <= start:
                continue
            ids = bin_lists[key]
            bin_cell_ids[start:end] = np.array(ids, dtype=np.int64)

        self._cell_depth = cell_depth
        self._ir = ir_abs
        self._itheta = itheta
        self._iphi = iphi
        self._cell_r_min = cell_r_min
        self._cell_r_max = cell_r_max
        self._r_min = float(np.min(cell_r_min))
        self._r_max = float(np.max(cell_r_max))
        self._cell_theta_min = np.min(theta_points[self._corners], axis=1)
        self._cell_theta_max = np.max(theta_points[self._corners], axis=1)

        phi_start = np.empty(n_cells, dtype=float)
        phi_width = np.empty(n_cells, dtype=float)
        phi_corners = phi_points[self._corners]
        for cid in range(n_cells):
            vals = phi_corners[cid, ~axis_mask[cid]]
            if vals.size < 2:
                vals = phi_corners[cid]
            start, width = self._minimal_phi_interval(vals)
            phi_start[cid] = start
            phi_width[cid] = width
        self._cell_phi_start = phi_start
        self._cell_phi_width = phi_width
        self._bin_counts = bin_counts
        self._bin_offsets = bin_offsets
        self._bin_cell_ids = bin_cell_ids
        self._lookup_state = LookupKernelState(
            levels_desc=self._levels_desc,
            shape_table=self._shape_table,
            dtheta_table=self._dtheta_table,
            dphi_table=self._dphi_table,
            bin_level_offset=self._bin_level_offset,
            bin_offsets=self._bin_offsets,
            bin_cell_ids=self._bin_cell_ids,
            cell_r_min=self._cell_r_min,
            cell_r_max=self._cell_r_max,
            cell_theta_min=self._cell_theta_min,
            cell_theta_max=self._cell_theta_max,
            cell_phi_start=self._cell_phi_start,
            cell_phi_width=self._cell_phi_width,
            cell_centers=self._cell_centers,
            r_min=float(self._r_min),
            r_max=float(self._r_max),
            max_radius=int(_DEFAULT_LOOKUP_MAX_RADIUS),
        )

    @staticmethod
    def _path(ir: int, itheta: int, iphi: int, depth: int) -> GridPath:
        """Construct root-to-leaf index path for one leaf coordinate triplet.

        Consumes:
        - `ir`, `itheta`, `iphi`: leaf-grid indices.
        - `depth`: tree depth for those indices.
        Returns:
        - `GridPath` tuple from root index to leaf index (inclusive).
        """
        out: list[GridIndex] = []
        for level in range(depth + 1):
            shift = depth - level
            out.append((ir >> shift, itheta >> shift, iphi >> shift))
        return tuple(out)

    @staticmethod
    def _minimal_phi_interval(values: np.ndarray) -> tuple[float, float]:
        """Find the minimal wrapped azimuth interval covering sample azimuths.

        Consumes:
        - `values`: azimuth samples in radians (any shape accepted via flattening).
        Returns:
        - `(start, width)` in radians for the minimal wrapped interval.
        """
        vals = np.sort(np.mod(np.array(values, dtype=float), 2.0 * math.pi))
        if vals.size == 0:
            return 0.0, 2.0 * math.pi
        if vals.size == 1:
            return float(vals[0]), 0.0
        wrapped = np.concatenate((vals, vals[:1] + 2.0 * math.pi))
        gaps = np.diff(wrapped)
        k = int(np.argmax(gaps))
        start = float(wrapped[k + 1] % (2.0 * math.pi))
        width = float((2.0 * math.pi) - gaps[k])
        return start, width

    def _candidate_ids(self, level: int, itheta: int, iphi: int, radius: int) -> np.ndarray:
        """Gather candidate ids from a square angular neighborhood.

        Neighborhood bins are centered at `(itheta, iphi)` with half-width
        `radius`. Bin contents are copied from CSR-like storage into one output
        array.
        Consumes:
        - `level`, `itheta`, `iphi`: center bin coordinates.
        - `radius`: neighborhood half-width in bins.
        Returns:
        - `np.ndarray[int64]` of candidate cell ids (possibly empty).
        """
        if level < 0 or level >= self._shape_table.shape[0]:
            return np.array([], dtype=np.int64)
        ntheta = int(self._shape_table[level, 1])
        nphi = int(self._shape_table[level, 2])
        if ntheta <= 0 or nphi <= 0:
            return np.array([], dtype=np.int64)
        level_offset = int(self._bin_level_offset[level])
        if level_offset < 0:
            return np.array([], dtype=np.int64)
        total = 0
        for dt in range(-radius, radius + 1):
            tt = itheta + dt
            if tt < 0 or tt >= ntheta:
                continue
            for dp in range(-radius, radius + 1):
                pp = (iphi + dp) % nphi
                key = int(level_offset + tt * nphi + pp)
                total += int(self._bin_counts[key])
        if total <= 0:
            return np.array([], dtype=np.int64)
        out = np.empty(total, dtype=np.int64)
        cursor = 0
        for dt in range(-radius, radius + 1):
            tt = itheta + dt
            if tt < 0 or tt >= ntheta:
                continue
            for dp in range(-radius, radius + 1):
                pp = (iphi + dp) % nphi
                key = int(level_offset + tt * nphi + pp)
                count = int(self._bin_counts[key])
                if count <= 0:
                    continue
                start = int(self._bin_offsets[key])
                end = int(self._bin_offsets[key + 1])
                out[cursor : cursor + count] = self._bin_cell_ids[start:end]
                cursor += count
        return out

    def _contains_rpa(
        self,
        cids: np.ndarray,
        r: float,
        polar: float,
        azimuth: float,
    ) -> np.ndarray:
        """Vectorized containment mask for one spherical query over candidate ids.

        Consumes:
        - `cids`: candidate cell ids.
        - `r`, `polar`, `azimuth`: spherical query coordinates.
        Returns:
        - Boolean mask aligned with `cids`, `True` where the query is contained.
        """
        tol = 1e-10
        if cids.size == 0:
            return np.array([], dtype=np.bool_)
        ok_r = (r >= (self._cell_r_min[cids] - tol)) & (r <= (self._cell_r_max[cids] + tol))
        ok_t = (polar >= (self._cell_theta_min[cids] - tol)) & (
            polar <= (self._cell_theta_max[cids] + tol)
        )
        starts = self._cell_phi_start[cids]
        widths = self._cell_phi_width[cids]
        dphi = np.mod(azimuth - starts, 2.0 * math.pi)
        ok_p = (widths >= (2.0 * math.pi - tol)) | (dphi <= (widths + tol))
        return ok_r & ok_t & ok_p

    def lookup_cell_id(
        self,
        point: np.ndarray,
        *,
        space: str,
    ) -> int:
        """Resolve one query point to a leaf `cell_id` in `space`."""
        q = np.array(point, dtype=float).reshape(3)
        resolved = str(space)
        if resolved == "xyz":
            return self.lookup_xyz_cell_id(float(q[0]), float(q[1]), float(q[2]))
        if resolved == "rpa":
            return self.lookup_rpa_cell_id(float(q[0]), float(q[1]), float(q[2]))
        raise ValueError("space must be 'xyz' or 'rpa'.")

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        space: str,
        tol: float = 1e-10,
    ) -> bool:
        """Containment test of one query point in `space` against one leaf cell."""
        q = np.array(point, dtype=float).reshape(3)
        resolved = str(space)
        if resolved == "xyz":
            return self.contains_xyz_cell(
                int(cell_id),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                tol=float(tol),
            )
        if resolved == "rpa":
            return self.contains_rpa_cell(
                int(cell_id),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                tol=float(tol),
            )
        raise ValueError("space must be 'xyz' or 'rpa'.")

    def lookup_rpa_cell_id(self, r: float, polar: float, azimuth: float) -> int:
        """Resolve one spherical query to a leaf `cell_id` (or `-1`).

        Search strategy:
        - Iterate refinement levels from fine to coarse.
        - For each level, expand angular neighborhood radius (`0..2`).
        - Filter by radial bounds, then apply vectorized containment tests.
        - Break ties by nearest cell center in xyz.
        - If nothing contains, fall back to nearest-center candidates.
        Consumes:
        - `r`, `polar`, `azimuth` query coordinates.
        Returns:
        - Resolved `cell_id`, or `-1` for invalid/out-of-domain inputs.
        """
        return int(
            _lookup_rpa_cell_id(
                float(r),
                float(polar),
                float(azimuth),
                self._lookup_state,
            )
        )

    def contains_rpa_cell(self, cell_id: int, r: float, polar: float, azimuth: float, *, tol: float = 1e-10) -> bool:
        """Containment test of one spherical query point against one leaf cell."""
        cid = int(cell_id)
        rr = float(r)
        pp = float(polar)
        aa = float(azimuth)
        t = float(tol)
        if rr < (float(self._cell_r_min[cid]) - t) or rr > (float(self._cell_r_max[cid]) + t):
            return False
        if pp < (float(self._cell_theta_min[cid]) - t) or pp > (float(self._cell_theta_max[cid]) + t):
            return False
        start = float(self._cell_phi_start[cid])
        width = float(self._cell_phi_width[cid])
        dphi = float((aa - start) % (2.0 * math.pi))
        if width >= (2.0 * math.pi - t):
            return True
        return dphi <= (width + t)

    def contains_xyz_cell(self, cell_id: int, x: float, y: float, z: float, *, tol: float = 1e-10) -> bool:
        """Containment test of one Cartesian query point against one leaf cell."""
        r = float(math.sqrt(x * x + y * y + z * z))
        if r == 0.0:
            polar = 0.0
        else:
            polar = float(math.acos(max(-1.0, min(1.0, z / r))))
        azimuth = float(math.atan2(y, x) % (2.0 * math.pi))
        return self.contains_rpa_cell(int(cell_id), r, polar, azimuth, tol=float(tol))

    def cell_step_hint(self, cell_id: int) -> float:
        """Return one characteristic step size used for ray marching in this cell."""
        cid = int(cell_id)
        r_span = float(self._cell_r_max[cid] - self._cell_r_min[cid])
        theta_span = float(self._cell_theta_max[cid] - self._cell_theta_min[cid])
        phi_span = float(min(self._cell_phi_width[cid], 2.0 * math.pi))
        length_scale = max(float(self._cell_r_max[cid]), 1.0)
        return float(max(r_span, length_scale * theta_span, length_scale * phi_span, 1e-6))

    def lookup_xyz_cell_id(self, x: float, y: float, z: float) -> int:
        """Resolve one Cartesian query to a leaf `cell_id` (or `-1`).

        Converts `(x, y, z)` to `(r, polar, azimuth)` then runs the spherical
        lookup path.
        Consumes:
        - Cartesian query scalars `x`, `y`, `z`.
        Returns:
        - Resolved `cell_id`, or `-1` for invalid/out-of-domain inputs.
        """
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return -1
        r = float(math.sqrt(x * x + y * y + z * z))
        if r == 0.0:
            polar = 0.0
        else:
            polar = float(math.acos(max(-1.0, min(1.0, z / r))))
        azimuth = float(math.atan2(y, x) % (2.0 * math.pi))
        return self.lookup_rpa_cell_id(r, polar, azimuth)

    def _hit_from_chosen(self, chosen: int, *, allow_invalid_depth: bool = False) -> LookupHit | None:
        """Materialize lookup metadata from one chosen cell id.

        Consumes:
        - `chosen`: candidate cell id.
        - `allow_invalid_depth`: whether depth `< 0` may still be materialized.
        Returns:
        - `LookupHit` if `chosen` is valid and depth policy allows it, else `None`.
        """
        if chosen < 0:
            return None
        center = self._cell_centers[chosen]
        depth = int(self._cell_depth[chosen])
        if depth < 0 and not allow_invalid_depth:
            return None
        cell_ir = int(self._ir[chosen])
        cell_ipolar = int(self._itheta[chosen])
        cell_iazimuth = int(self._iphi[chosen])
        return LookupHit(
            cell_id=chosen,
            level=depth,
            i0=cell_ir,
            i1=cell_ipolar,
            i2=cell_iazimuth,
            path=self._path(cell_ir, cell_ipolar, cell_iazimuth, depth),
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
        )



class SphericalOctree(Octree):
    """Octree specialization for spherical `(r, polar, azimuth)` datasets."""

    COORD_SYSTEM: ClassVar[str | None] = "rpa"

    @property
    def center_phi(self) -> np.ndarray | None:
        """Expose spherical axis-2 center payload."""
        return self.axis2_center

    @center_phi.setter
    def center_phi(self, value: np.ndarray | None) -> None:
        """Set spherical axis-2 center payload."""
        self.axis2_center = value

    @property
    def delta_phi(self) -> np.ndarray | None:
        """Expose spherical axis-2 span payload."""
        return self.axis2_span

    @delta_phi.setter
    def delta_phi(self, value: np.ndarray | None) -> None:
        """Set spherical axis-2 span payload."""
        self.axis2_span = value

    @property
    def expected_delta_phi(self) -> np.ndarray | None:
        """Expose expected spherical axis-2 span payload."""
        return self.expected_axis2_span

    @expected_delta_phi.setter
    def expected_delta_phi(self, value: np.ndarray | None) -> None:
        """Set expected spherical axis-2 span payload."""
        self.expected_axis2_span = value

    @property
    def coarse_delta_phi(self) -> float | None:
        """Expose coarsest spherical axis-2 span scalar."""
        return self.coarse_axis2_span

    @coarse_delta_phi.setter
    def coarse_delta_phi(self, value: float | None) -> None:
        """Set coarsest spherical axis-2 span scalar."""
        self.coarse_axis2_span = value

    @staticmethod
    def xyz_to_rpa(q: np.ndarray) -> tuple[float, float, float]:
        """Convert one Cartesian point to spherical `(r, polar, azimuth)`.

        Consumes:
        - `q`: Cartesian coordinate triple.
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
        azimuth = float(math.atan2(y, x) % (2.0 * math.pi))
        return r, polar, azimuth

    def _lookup_local(self, xyz: np.ndarray, near_cid: int | None = None) -> "LookupHit | None":
        """Lookup in xyz using local spherical-bin neighborhoods around a near cell.

        Algorithm:
        - If a nearby cell is provided, test it directly.
        - Probe local angular neighborhoods at the nearby level and adjacent
          levels, mapping angular indices between levels by center alignment.
        - Run vectorized containment on merged candidates.
        - Fall back to full lookup if no local candidate contains the point.
        Consumes:
        - Cartesian query `xyz` and optional nearby `cell_id` hint.
        Returns:
        - `LookupHit` if resolved, else `None`.
        """
        q = np.array(xyz, dtype=float)
        x = float(q[0])
        y = float(q[1])
        z = float(q[2])
        lookup = self.lookup
        if near_cid is not None and int(near_cid) >= 0:
            near = int(near_cid)
            if self.contains_cell(near, q, space="xyz"):
                return self.hit_from_cell_id(near)

            near_level = int(lookup._cell_level_rel[near])
            near_ipolar = int(lookup._itheta[near])
            near_iazimuth = int(lookup._iphi[near])
            shape_table = lookup._shape_table
            near_shape: np.ndarray | None = None
            if 0 <= near_level < shape_table.shape[0] and int(shape_table[near_level, 0]) > 0:
                near_shape = shape_table[near_level]
            candidate_arrays: list[np.ndarray] = []
            for level in (near_level, near_level - 1, near_level + 1):
                if level < 0 or level >= shape_table.shape[0]:
                    continue
                shape = shape_table[level]
                if int(shape[0]) <= 0:
                    continue
                ntheta = int(shape[1])
                nphi = int(shape[2])
                if near_shape is None:
                    mapped_t = near_ipolar
                    mapped_p = near_iazimuth
                else:
                    mapped_t = int(
                        np.clip(
                            round(((near_ipolar + 0.5) * shape[1] / near_shape[1]) - 0.5),
                            0,
                            ntheta - 1,
                        )
                    )
                    mapped_p = int(
                        np.clip(
                            round(((near_iazimuth + 0.5) * shape[2] / near_shape[2]) - 0.5),
                            0,
                            nphi - 1,
                        )
                    )
                for radius in (0, 1):
                    cands = lookup._candidate_ids(int(level), mapped_t, mapped_p, radius)
                    if cands.size > 0:
                        candidate_arrays.append(cands)

            if candidate_arrays:
                candidates = np.unique(np.concatenate(candidate_arrays))
                r, polar, azimuth = self.xyz_to_rpa(q)
                inside = lookup._contains_rpa(candidates, r, polar, azimuth)
                if np.any(inside):
                    valid = candidates[inside]
                    d = np.linalg.norm(lookup._cell_centers[valid] - q, axis=1)
                    return self.hit_from_cell_id(int(valid[int(np.argmin(d))]))

        return self.lookup_point(np.array([x, y, z], dtype=float), space="xyz")

    def _build_lookup(
        self,
    ) -> "_CellLookup":
        """Construct spherical lookup state backed by angular/radial bins.

        Consumes:
        - Bound spherical tree geometry.
        Returns:
        - `_SphericalCellLookup` instance.
        """
        if self.ds is None or self.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call bind(...) before lookup.")
        return _SphericalCellLookup(
            self,
        )
